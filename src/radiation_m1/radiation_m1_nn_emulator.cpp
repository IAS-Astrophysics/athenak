//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nn_emulator.cpp
//! \brief LibTorch implementation of the NN opacity emulator.

#if ENABLE_NN_OPACITY

#include "radiation_m1/radiation_m1_nn_emulator.hpp"

#include "globals.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>    // c10::cuda::CUDAStreamGuard
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>   // c10::cuda::getStreamFromExternal
#include <c10/core/InferenceMode.h>
#include <torch/script.h>

#include <Kokkos_Core.hpp>  // Kokkos::Profiling::pushRegion/popRegion

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace radiationm1 {

struct NNOpacityEmulator::Impl {
  static constexpr int N_PROFILE_POINTS = static_cast<int>(NNProfilePoint::count);

  struct AllocatorSnapshot {
    int64_t requests = 0;
    int64_t device_allocs = 0;
    int64_t device_frees = 0;
    int64_t sync_all_streams = 0;
    int64_t alloc_retries = 0;
  };

  mutable torch::jit::Module module_;
  torch::Tensor in_mean_, in_std_;
  torch::Tensor out_mean_, out_std_;
  // Pre-allocated buffer for denormalized output (avoids cudaMalloc per call).
  // Resized on first use or when N grows.
  torch::Tensor y_phys_buf_;
  int y_phys_capacity_ = 0;  // current capacity in rows
  float h_in_mean_[N_EOS] = {};
  float h_in_std_[N_EOS] = {};
  float h_out_mean_[N_OUTPUTS] = {};
  float h_out_std_[N_OUTPUTS] = {};
  torch::Device device_{torch::kCPU};
  bool loaded_ = false;

  // ── fused-kernel weights (optional path) ──────────────────────────────────
  std::string model_path_;      // remembered so ExtractWeights() can reload cleanly
  NNWeights weights_{};         // device-resident weights for nn_forward
  bool weights_ready_ = false;

  // ── CUDA-graph state (optional InferGraph path) ───────────────────────────
  mutable at::cuda::CUDAGraph graph_;
  mutable torch::Tensor g_in_, g_out_;  // static I/O buffers the graph reads/writes
  mutable bool graph_captured_ = false;
  mutable int  graph_N_ = -1;           // batch the graph was captured for

  // ── direct-cuBLAS state (optional InferCublas path) ───────────────────────
  // cuBLAS handle (created lazily) + grow-only device scratch for the forward.
  // z/a/h are N×H activation buffers; o28 is the N×28 pre-map head output.
  cublasHandle_t cublas_ = nullptr;
  Kokkos::View<float *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      zbuf_{}, abuf_{}, hbuf_{}, o28buf_{};
  int gemm_capacity_ = 0;

  // ── opt-in sampled profiler ───────────────────────────────────────────────
  // CUDA events are recorded on the Kokkos stream.  A later call queries the
  // terminal event and only consumes completed samples, so profiling never adds
  // a CUDA fence or stream synchronize to the opacity path.
  bool profile_enabled_ = false;
  int profile_interval_ = 100;
  std::string profile_mode_ = "torch";
  mutable int64_t profile_call_count_ = 0;
  mutable int64_t profile_sample_call_ = 0;
  mutable int profile_n_cells_ = 0;
  mutable int profile_infer_cells_ = 0;
  mutable bool profile_kokkos_scratch_grew_ = false;
  mutable size_t profile_kokkos_scratch_bytes_ = 0;
  mutable bool profile_cublas_scratch_grew_ = false;
  mutable bool profile_active_ = false;
  mutable bool profile_pending_ = false;
  mutable bool profile_events_created_ = false;
  mutable std::array<cudaEvent_t, N_PROFILE_POINTS> profile_events_{};
  mutable AllocatorSnapshot profile_alloc_start_{};
  mutable AllocatorSnapshot profile_alloc_end_{};
  mutable std::chrono::steady_clock::time_point profile_host_start_{};
  mutable double profile_host_submit_ms_ = 0.0;

  ~Impl() {
    if (profile_events_created_) {
      for (auto &event : profile_events_) cudaEventDestroy(event);
    }
    if (cublas_ != nullptr) cublasDestroy(cublas_);
  }

  AllocatorSnapshot AllocatorStats() const {
    AllocatorSnapshot out{};
    if (!device_.is_cuda()) return out;
    const auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(device_.index());
    constexpr size_t aggregate = static_cast<size_t>(
        c10::CachingAllocator::StatType::AGGREGATE);
    out.requests = stats.allocation[aggregate].allocated;
    out.device_allocs = stats.num_device_alloc;
    out.device_frees = stats.num_device_free;
    out.sync_all_streams = stats.num_sync_all_streams;
    out.alloc_retries = stats.num_alloc_retries;
    return out;
  }

  static torch::Tensor LoadFloat32Bin(const std::string &path, int expected_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      throw std::runtime_error(
          std::string("NNOpacityEmulator: cannot open '") + path + "'");
    }

    std::vector<float> buf(expected_n);
    f.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(expected_n * sizeof(float)));
    if (!f) {
      throw std::runtime_error(
          std::string("NNOpacityEmulator: read error in '") + path + "'");
    }

    return torch::tensor(buf, torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torch::kCPU));
  }
};

NNOpacityEmulator::NNOpacityEmulator() : impl_(std::make_unique<Impl>()) {}
NNOpacityEmulator::~NNOpacityEmulator() = default;
NNOpacityEmulator::NNOpacityEmulator(NNOpacityEmulator &&) noexcept = default;
NNOpacityEmulator &NNOpacityEmulator::operator=(NNOpacityEmulator &&) noexcept = default;

void NNOpacityEmulator::Load(const std::string &model_path,
                             const std::string &stats_dir, bool use_cuda) {
  impl_->model_path_ = model_path;  // for the optional fused-kernel weight export
  if (use_cuda) {
    int cuda_dev = 0;
    cudaGetDevice(&cuda_dev);
    impl_->device_ = torch::Device(torch::kCUDA, cuda_dev);
    // Enable TF32 tensor cores for FP32 matmuls.  TF32 keeps the FP32 storage
    // and API but performs the matmul accumulation inputs at 10-bit mantissa on
    // the A100 tensor cores (156 TFLOPS vs 19.5 TFLOPS plain FP32).  Precision
    // is ample for opacity emulation and there is no dtype change (unlike FP16),
    // so no risk of the Half != float mismatch seen with optimize_for_inference.
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
  } else {
    impl_->device_ = torch::Device(torch::kCPU);
  }

  try {
    impl_->module_ = torch::jit::load(model_path, impl_->device_);
  } catch (const c10::Error &e) {
    throw std::runtime_error(
        std::string("NNOpacityEmulator: failed to load model from '") +
        model_path + "': " + e.what());
  }
  impl_->module_.eval();
  // Freeze the module: inlines parameters into the graph and flattens the
  // module hierarchy.  This eliminates per-op attribute-lookup overhead in the
  // TorchScript interpreter (~1-2 ms/op × ~20 ops = ~30-50 ms/call reduction).
  impl_->module_ = torch::jit::freeze(impl_->module_);
  // Fuse linear+relu and other patterns into single CUDA kernels.
  impl_->module_ = torch::jit::optimize_for_inference(impl_->module_);

  impl_->in_mean_ = Impl::LoadFloat32Bin(stats_dir + "/nn2d_in_mean.bin", N_EOS);
  impl_->in_std_ = Impl::LoadFloat32Bin(stats_dir + "/nn2d_in_std.bin", N_EOS);
  impl_->out_mean_ =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_out_mean.bin", N_OUTPUTS);
  impl_->out_std_ =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_out_std.bin", N_OUTPUTS);

  auto cpu_imean = impl_->in_mean_.contiguous();
  auto cpu_istd = impl_->in_std_.contiguous();
  auto cpu_omean = impl_->out_mean_.contiguous();
  auto cpu_ostd = impl_->out_std_.contiguous();
  for (int i = 0; i < N_EOS; ++i) {
    impl_->h_in_mean_[i] = cpu_imean.data_ptr<float>()[i];
    impl_->h_in_std_[i] = cpu_istd.data_ptr<float>()[i];
  }
  for (int i = 0; i < N_OUTPUTS; ++i) {
    impl_->h_out_mean_[i] = cpu_omean.data_ptr<float>()[i];
    impl_->h_out_std_[i] = cpu_ostd.data_ptr<float>()[i];
  }

  impl_->in_mean_ = impl_->in_mean_.to(impl_->device_);
  impl_->in_std_ = impl_->in_std_.to(impl_->device_);
  impl_->out_mean_ = impl_->out_mean_.to(impl_->device_);
  impl_->out_std_ = impl_->out_std_.to(impl_->device_);

  impl_->loaded_ = true;
  std::cout << "Loaded NN opacity emulator from " << model_path << std::endl;
}

void NNOpacityEmulator::Infer(const float *eos_data, float *out_data, int N) const {
  assert(impl_->loaded_ && "NNOpacityEmulator::Infer called before Load()");
  if (N == 0) return;

  c10::InferenceMode inference_guard;
  auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  static constexpr int CHUNK_SIZE = 65536;

  for (int base = 0; base < N; base += CHUNK_SIZE) {
    const int chunk = std::min(CHUNK_SIZE, N - base);

    torch::Tensor x_cpu =
        torch::from_blob(const_cast<float *>(eos_data + base * N_EOS),
                         {chunk, N_EOS}, opts_cpu)
            .clone();

    x_cpu.narrow(1, 0, 1).log10_();
    x_cpu.narrow(1, 1, 1).log10_();

    // Model takes (chunk, 8) and produces (chunk, 32) — no tiling/one-hot.
    torch::Tensor x = x_cpu.to(impl_->device_);
    torch::Tensor x_norm =
        (x - impl_->in_mean_.unsqueeze(0)) / impl_->in_std_.unsqueeze(0);

    torch::Tensor y_norm = impl_->module_.forward({x_norm}).toTensor();
    torch::Tensor y_phys =
        torch::pow(10.0, y_norm * impl_->out_std_.unsqueeze(0) +
                             impl_->out_mean_.unsqueeze(0));

    // out_data layout: [N × N_OUTPUTS] = [N × 32]
    torch::Tensor y_cpu = y_phys.to(torch::kCPU).contiguous();
    const float *src = y_cpu.data_ptr<float>();
    std::copy(src, src + chunk * N_OUTPUTS,
              out_data + static_cast<long long>(base) * N_OUTPUTS);
  }
}

void NNOpacityEmulator::InferPrebuilt(const float *x_full_ptr, float *nn_out_ptr,
                                      int N) const {
  assert(impl_->loaded_ && IsGPU() && "InferPrebuilt requires GPU (nn_use_cuda=true)");
  if (N == 0) return;

  c10::InferenceMode inference_guard;
  auto cuda_opts = torch::TensorOptions().dtype(torch::kFloat32).device(impl_->device_);

  // Model takes (N, 8) EOS features and produces (N, 32) outputs directly for
  // all 4 species — no tiling or one-hot needed.  Process all cells in one
  // pass: each rank has ≤25 MBs × 48³ ≈ 2.76M cells; A100 has 40 GB so the
  // full (2.76M,8) input + (2.76M,32) output fits easily (~440 MB).  Large N
  // also gives better tensor-core GEMM efficiency than small chunks.
  static constexpr int CHUNK_SIZE = 4194304;  // 4M — covers any realistic mesh

  // SCALING FIX: run LibTorch on the *Kokkos* CUDA stream, so this forward is
  // ordered in-line with gather_pack (before) and the readout kernel (after) on
  // a single stream.  That removes the need for the cross-stream fence + event +
  // cudaStreamSynchronize handoff, which serialised this part of the step and is
  // a likely contributor to the NN-only multi-node scaling loss.
  cudaStream_t kokkos_stream = Kokkos::Cuda().cuda_stream();
  c10::cuda::CUDAStream torch_on_kokkos =
      c10::cuda::getStreamFromExternal(kokkos_stream, impl_->device_.index());
  c10::cuda::CUDAStreamGuard stream_guard(torch_on_kokkos);

  for (int base = 0; base < N; base += CHUNK_SIZE) {
    const int chunk = std::min(CHUNK_SIZE, N - base);

    // Zero-copy wrap of the pre-normalized (N, N_INPUTS=8) input buffer on GPU
    torch::Tensor x_in = torch::from_blob(
        const_cast<float *>(x_full_ptr +
                            static_cast<long long>(base) * N_INPUTS),
        {chunk, N_INPUTS}, cuda_opts);

    // Forward pass: (chunk, 8) → (chunk, 32)
    Kokkos::Profiling::pushRegion("NN::forward");
    torch::Tensor y_norm = impl_->module_.forward({x_in}).toTensor();
    Kokkos::Profiling::popRegion();

    // Copy raw normalized output; denorm is fused into the readout kernel.
    Kokkos::Profiling::pushRegion("NN::copy_out");
    torch::Tensor out_wrap = torch::from_blob(
        nn_out_ptr + static_cast<long long>(base) * N_OUTPUTS,
        {static_cast<long long>(chunk) * N_OUTPUTS}, cuda_opts);
    out_wrap.copy_(y_norm.reshape({-1}));
    Kokkos::Profiling::popRegion();
  }
  // No cudaStreamSynchronize here: the shared Kokkos stream already guarantees the
  // downstream readout kernel runs after this forward.  (Was:
  // cudaStreamSynchronize(lt_stream);)  Keeping a sync here would re-serialise the
  // step and re-break comm/compute overlap.
}

//----------------------------------------------------------------------------------------
//! \fn NNOpacityEmulator::InferGraph
//! \brief Capture a fixed-size forward into a CUDA graph and replay it in chunks.
//!        This removes eager forward dispatch overhead, at the cost of a fixed
//!        minimum-work quantum. Same zero-copy device I/O contract as InferPrebuilt.
void NNOpacityEmulator::InferGraph(const float *x_full_ptr, float *nn_out_ptr,
                                   int N) const {
  assert(impl_->loaded_ && IsGPU() && "InferGraph requires GPU (nn_use_cuda=true)");
  if (N == 0) return;
  c10::InferenceMode inference_guard;
  Impl &im = *impl_;
  const auto dev = im.device_.index();
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(im.device_);
  cudaStream_t kokkos_stream = Kokkos::Cuda().cuda_stream();

  // ── Fixed-shape chunked capture/replay ──────────────────────────────────────
  // Capture ONE graph for a bounded chunk (GRAPH_CHUNK rows) and replay it
  // ceil(N / GRAPH_CHUNK) times over the real cells.  This (a) bounds the graph's
  // private-pool activation memory to ~GRAPH_CHUNK*HIDDEN floats regardless of N or
  // node count — so it fits even at low node counts where each GPU holds many
  // MeshBlocks and free VRAM is scarce — and (b) avoids padding the whole batch:
  // only the final partial chunk over-computes up to GRAPH_CHUNK rows, and those
  // padded outputs are simply not copied back.  The network is strictly per-row
  // (GEMM + per-sample LayerNorm never mix rows), so stale padded rows in g_in_
  // cannot corrupt any valid output.  N is no longer part of the capture key, so a
  // varying N (AMR) never triggers a re-capture.
  static constexpr int GRAPH_CHUNK = 262144;  // 256k rows (~1 GiB peak activations)

  if (!im.graph_captured_) {
    im.g_in_  = torch::zeros({GRAPH_CHUNK, N_INPUTS},  opts);   // static graph input
    im.g_out_ = torch::zeros({GRAPH_CHUNK, N_OUTPUTS}, opts);   // static graph output
    // Warm up (triggers cuBLAS/cuDNN algo selection + lazy init) and capture on a
    // side stream so Kokkos work is not swept into the graph.
    c10::cuda::CUDAStream side = c10::cuda::getStreamFromPool(false, dev);
    {
      c10::cuda::CUDAStreamGuard sg(side);
      for (int i = 0; i < 3; ++i) {
        torch::Tensor y = im.module_.forward({im.g_in_}).toTensor();
        im.g_out_.copy_(y);
      }
      side.synchronize();
      im.graph_.capture_begin();
      torch::Tensor y = im.module_.forward({im.g_in_}).toTensor();
      im.g_out_.copy_(y);
      im.graph_.capture_end();
    }
    im.graph_captured_ = true;
    im.graph_N_ = GRAPH_CHUNK;
  }

  // Loop the real cells through the captured graph in GRAPH_CHUNK-row bites.  Copy
  // in, replay, copy out — all on the Kokkos stream so the whole sequence is
  // ordered with the surrounding gather/readout kernels (no fence/event needed).
  c10::cuda::CUDAStream torch_on_kokkos =
      c10::cuda::getStreamFromExternal(kokkos_stream, dev);
  c10::cuda::CUDAStreamGuard sguard(torch_on_kokkos);
  float *gin  = im.g_in_.data_ptr<float>();
  float *gout = im.g_out_.data_ptr<float>();
  for (int base = 0; base < N; base += GRAPH_CHUNK) {
    const int chunk = std::min(GRAPH_CHUNK, N - base);
    cudaMemcpyAsync(gin, x_full_ptr + static_cast<size_t>(base) * N_INPUTS,
                    static_cast<size_t>(chunk) * N_INPUTS * sizeof(float),
                    cudaMemcpyDeviceToDevice, kokkos_stream);
    im.graph_.replay();  // launches on the current (Kokkos) stream — no cudaMalloc/sync
    cudaMemcpyAsync(nn_out_ptr + static_cast<size_t>(base) * N_OUTPUTS, gout,
                    static_cast<size_t>(chunk) * N_OUTPUTS * sizeof(float),
                    cudaMemcpyDeviceToDevice, kokkos_stream);
  }
}

//----------------------------------------------------------------------------------------
//! \fn NNOpacityEmulator::InferCublas
//! \brief Evaluate the SymMLP with direct cuBLAS GEMMs (TF32) + small Kokkos
//!        elementwise kernels, all on the Kokkos stream.  This removes LibTorch
//!        from the steady-state forward and mirrors nn_forward()/the torch forward.
void NNOpacityEmulator::InferCublas(const float *x_full_ptr, float *nn_out_ptr,
                                    int N) const {
  assert(impl_->loaded_ && IsGPU() && "InferCublas requires GPU (nn_use_cuda=true)");
  if (N == 0) return;
  Impl &im = *impl_;
  if (!im.weights_ready_) {
    throw std::runtime_error(
        "NNOpacityEmulator::InferCublas: ExtractWeights() must be called first");
  }
  const NNWeights &w = im.weights_;
  constexpr int H  = NN_FWD_HIDDEN;   // 256
  constexpr int IN = NN_FWD_IN;       // 8
  constexpr int O28 = NN_FWD_OUT28;   // 28
  constexpr int O32 = NN_FWD_OUT32;   // 32
  constexpr float eps = 1e-5f;
  cudaStream_t stream = Kokkos::Cuda().cuda_stream();
  auto check_cublas = [](cublasStatus_t status, const char *where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("NNOpacityEmulator::InferCublas: ") +
                               where + " failed (status=" +
                               std::to_string(static_cast<int>(status)) + ")");
    }
  };

  // Lazy handle; TF32 tensor-op math to match the torch/cuBLAS GEMM throughput.
  if (im.cublas_ == nullptr) {
    check_cublas(cublasCreate(&im.cublas_), "cublasCreate");
    check_cublas(cublasSetMathMode(im.cublas_, CUBLAS_TF32_TENSOR_OP_MATH),
                 "cublasSetMathMode");
    std::cout << "NN opacity: direct-cuBLAS forward enabled (TF32)" << std::endl;
  }
  check_cublas(cublasSetStream(im.cublas_, stream), "cublasSetStream");

  // Grow-only scratch (N×H activations, N×28 head output).
  if (N > im.gemm_capacity_) {
    if (im.profile_active_) im.profile_cublas_scratch_grew_ = true;
    Kokkos::realloc(im.zbuf_,   static_cast<size_t>(N) * H);
    Kokkos::realloc(im.abuf_,   static_cast<size_t>(N) * H);
    Kokkos::realloc(im.hbuf_,   static_cast<size_t>(N) * H);
    Kokkos::realloc(im.o28buf_, static_cast<size_t>(N) * O28);
    im.gemm_capacity_ = N;
  }
  float *Z  = im.zbuf_.data();
  float *A  = im.abuf_.data();
  float *Hb = im.hbuf_.data();
  float *Ob = im.o28buf_.data();
  const float one = 1.0f, zero = 0.0f;
  using Range = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>;

  // Each Linear Y(N,Mout) row-major = X(N,Kin) · W(Mout,Kin)ᵀ (PyTorch layout, no
  // bias) maps to one cublasSgemm: row-major (N,Mout) ≡ col-major (Mout,N), so we
  // compute C(Mout,N)=Wᵀ·X with A=W (col-major Kin×Mout, ld=Kin, op=T), B=X
  // (col-major Kin×N, ld=Kin, op=N), ldc=Mout.  Bias/activation follow in a kernel.

  // ── embed: Z = GELU(X · W_embedᵀ + b_embed) ────────────────────────────────
  check_cublas(cublasSgemm(im.cublas_, CUBLAS_OP_T, CUBLAS_OP_N, H, N, IN,
                           &one, w.W_embed.data(), IN, x_full_ptr, IN, &zero, Z, H),
               "embed GEMM");
  {
    auto b = w.b_embed;
    Kokkos::parallel_for("nn_cublas_embed_act",
        Range(0, static_cast<size_t>(N) * H), KOKKOS_LAMBDA(const size_t idx) {
          Z[idx] = nn_gelu(Z[idx] + b(static_cast<int>(idx % H)));
        });
  }

  // ── residual blocks ────────────────────────────────────────────────────────
  for (int blk = 0; blk < w.n_blocks; ++blk) {
    // A = LayerNorm(Z)  (per-row over H, biased variance, eps)
    {
      auto lng = w.ln_g; auto lnb = w.ln_b;
      Kokkos::parallel_for("nn_cublas_layernorm", Range(0, N),
          KOKKOS_LAMBDA(const int n) {
            const size_t base = static_cast<size_t>(n) * H;
            float mean = 0.0f;
            for (int o = 0; o < H; ++o) mean += Z[base + o];
            mean /= static_cast<float>(H);
            float var = 0.0f;
            for (int o = 0; o < H; ++o) { float d = Z[base + o] - mean; var += d * d; }
            var /= static_cast<float>(H);
            const float inv = 1.0f / Kokkos::sqrt(var + eps);
            for (int o = 0; o < H; ++o)
              A[base + o] = (Z[base + o] - mean) * inv * lng(blk, o) + lnb(blk, o);
          });
    }
    // Hb = GELU(A · W1ᵀ + b1)
    check_cublas(cublasSgemm(
        im.cublas_, CUBLAS_OP_T, CUBLAS_OP_N, H, N, H, &one,
        w.W1.data() + static_cast<size_t>(blk) * H * H, H, A, H, &zero, Hb, H),
        "residual block linear1 GEMM");
    {
      auto b1 = w.b1;
      Kokkos::parallel_for("nn_cublas_lin1_act",
          Range(0, static_cast<size_t>(N) * H), KOKKOS_LAMBDA(const size_t idx) {
            Hb[idx] = nn_gelu(Hb[idx] + b1(blk, static_cast<int>(idx % H)));
          });
    }
    // A = Hb · W2ᵀ ; Z += A + b2   (residual)
    check_cublas(cublasSgemm(
        im.cublas_, CUBLAS_OP_T, CUBLAS_OP_N, H, N, H, &one,
        w.W2.data() + static_cast<size_t>(blk) * H * H, H, Hb, H, &zero, A, H),
        "residual block linear2 GEMM");
    {
      auto b2 = w.b2;
      Kokkos::parallel_for("nn_cublas_lin2_resid",
          Range(0, static_cast<size_t>(N) * H), KOKKOS_LAMBDA(const size_t idx) {
            Z[idx] += A[idx] + b2(blk, static_cast<int>(idx % H));
          });
    }
  }

  // ── heads → 28 outputs (no bias yet), packed into rows [0,8),[8,16),[16,28)
  // of the N×28 row-major buffer via ldc=28 ──────────────────────────────────
  check_cublas(cublasSgemm(im.cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 8, N, H,
                           &one, w.W_nue.data(), H, Z, H, &zero, Ob, O28),
               "nue head GEMM");
  check_cublas(cublasSgemm(im.cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 8, N, H,
                           &one, w.W_anue.data(), H, Z, H, &zero, Ob + 8, O28),
               "anue head GEMM");
  check_cublas(cublasSgemm(im.cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 12, N, H,
                           &one, w.W_nux.data(), H, Z, H, &zero, Ob + 16, O28),
               "nux head GEMM");

  // head bias + 28→32 map (OUT_MAP_28_TO_32) into the output buffer
  {
    auto bnue = w.b_nue; auto banue = w.b_anue; auto bnux = w.b_nux;
    Kokkos::parallel_for("nn_cublas_head_map", Range(0, N),
        KOKKOS_LAMBDA(const int n) {
          const size_t ib = static_cast<size_t>(n) * O28;
          float o[O28];
          for (int c = 0; c < 8;  ++c) o[c]      = Ob[ib + c]      + bnue(c);
          for (int c = 0; c < 8;  ++c) o[8 + c]  = Ob[ib + 8 + c]  + banue(c);
          for (int c = 0; c < 12; ++c) o[16 + c] = Ob[ib + 16 + c] + bnux(c);
          float *out = nn_out_ptr + static_cast<size_t>(n) * O32;
          for (int i = 0; i < 24; ++i) out[i] = o[i];  // nue anue nux (identity)
          out[24] = o[16]; out[25] = o[17];            // anux ch0,1 = nux ch0,1
          out[26] = o[24]; out[27] = o[25];            // anux ch2,3 = anux NEPS
          out[28] = o[20]; out[29] = o[21];            // anux ch4,5 = nux ch4,5
          out[30] = o[26]; out[31] = o[27];            // anux ch6,7 = anux NEPS
        });
  }
}

void NNOpacityEmulator::InferDevice(const float *eos_dev_ptr, float *nn_out_ptr,
                                    int N) const {
  assert(impl_->loaded_ && IsGPU() && "InferDevice requires GPU (nn_use_cuda=true)");
  if (N == 0) return;

  c10::InferenceMode inference_guard;
  auto cuda_opts = torch::TensorOptions().dtype(torch::kFloat32).device(impl_->device_);
  static constexpr int CHUNK_SIZE = 65536;

  for (int base = 0; base < N; base += CHUNK_SIZE) {
    const int chunk = std::min(CHUNK_SIZE, N - base);

    torch::Tensor x = torch::from_blob(
        const_cast<float *>(eos_dev_ptr + base * N_EOS), {chunk, N_EOS},
        cuda_opts);
    torch::Tensor x_proc = x.clone();
    x_proc.narrow(1, 0, 1).log10_();
    x_proc.narrow(1, 1, 1).log10_();

    // Model takes (chunk, 8), produces (chunk, 32) — no tiling/one-hot.
    torch::Tensor x_norm =
        (x_proc - impl_->in_mean_.unsqueeze(0)) / impl_->in_std_.unsqueeze(0);

    torch::Tensor y_norm = impl_->module_.forward({x_norm}).toTensor();
    torch::Tensor y_phys =
        torch::pow(10.0, y_norm * impl_->out_std_.unsqueeze(0) +
                             impl_->out_mean_.unsqueeze(0));

    // Output layout: (chunk, 32) → stored flat as [N × N_OUTPUTS]
    torch::Tensor out_wrap = torch::from_blob(
        nn_out_ptr + static_cast<long long>(base) * N_OUTPUTS,
        {static_cast<long long>(chunk) * N_OUTPUTS}, cuda_opts);
    out_wrap.copy_(y_phys.reshape({-1}));
  }
}

bool NNOpacityEmulator::IsLoaded() const { return impl_->loaded_; }
bool NNOpacityEmulator::IsGPU() const { return impl_->device_.is_cuda(); }
int NNOpacityEmulator::DeviceIndex() const { return impl_->device_.index(); }

const float *NNOpacityEmulator::HostInMean() const { return impl_->h_in_mean_; }
const float *NNOpacityEmulator::HostInStd() const { return impl_->h_in_std_; }
const float *NNOpacityEmulator::HostOutMean() const { return impl_->h_out_mean_; }
const float *NNOpacityEmulator::HostOutStd() const { return impl_->h_out_std_; }

void NNOpacityEmulator::ConfigureProfiling(bool enabled, int interval,
                                           const std::string &mode) {
  Impl &im = *impl_;
  im.profile_enabled_ = enabled && im.device_.is_cuda();
  im.profile_interval_ = std::max(interval, 1);
  im.profile_mode_ = mode;
  if (!im.profile_enabled_) return;

  for (auto &event : im.profile_events_) {
    const cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDefault);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("NN profiler: cudaEventCreate failed: ") +
          cudaGetErrorString(err));
    }
  }
  im.profile_events_created_ = true;
  if (global_variable::my_rank == 0) {
    std::cout << "NN profiler enabled: sample every " << im.profile_interval_
              << " opacity calls; CUDA events are queried asynchronously"
              << std::endl;
  }
}

void NNOpacityEmulator::ProfilePollAndReport() const {
  Impl &im = *impl_;
  if (!im.profile_enabled_ || !im.profile_pending_) return;

  const cudaError_t query = cudaEventQuery(
      im.profile_events_[static_cast<int>(NNProfilePoint::kirchhoff)]);
  int local_ready = (query == cudaSuccess) ? 1 : 0;
  if (query != cudaSuccess && query != cudaErrorNotReady) {
    throw std::runtime_error(
        std::string("NN profiler: cudaEventQuery failed: ") +
        cudaGetErrorString(query));
  }

  // A small readiness collective is performed only for a pending sampled call.
  // It prevents rank-dependent event completion from making the reporting
  // reductions diverge, while never waiting for GPU work on the host.
  int all_ready = local_ready;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
#endif
  if (all_ready == 0) return;

  constexpr int ngpu = 6;
  constexpr int ntime = ngpu + 1;
  std::array<double, ntime> timing_ms{};
  for (int p = 0; p < ngpu - 1; ++p) {
    float elapsed = 0.0f;
    const cudaError_t err = cudaEventElapsedTime(
        &elapsed, im.profile_events_[p], im.profile_events_[p + 1]);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("NN profiler: cudaEventElapsedTime failed: ") +
          cudaGetErrorString(err));
    }
    timing_ms[p] = elapsed;
  }
  float total = 0.0f;
  const cudaError_t total_err = cudaEventElapsedTime(
      &total, im.profile_events_[static_cast<int>(NNProfilePoint::start)],
      im.profile_events_[static_cast<int>(NNProfilePoint::kirchhoff)]);
  if (total_err != cudaSuccess) {
    throw std::runtime_error(
        std::string("NN profiler: total cudaEventElapsedTime failed: ") +
        cudaGetErrorString(total_err));
  }
  timing_ms[ngpu - 1] = total;
  timing_ms[ngpu] = im.profile_host_submit_ms_;

  const auto &a0 = im.profile_alloc_start_;
  const auto &a1 = im.profile_alloc_end_;
  constexpr int nwork = 10;
  std::array<double, nwork> work{{
      static_cast<double>(im.profile_n_cells_),
      static_cast<double>(im.profile_infer_cells_),
      im.profile_kokkos_scratch_grew_ ? 1.0 : 0.0,
      static_cast<double>(im.profile_kokkos_scratch_bytes_),
      im.profile_cublas_scratch_grew_ ? 1.0 : 0.0,
      static_cast<double>(a1.requests - a0.requests),
      static_cast<double>(a1.device_allocs - a0.device_allocs),
      static_cast<double>(a1.device_frees - a0.device_frees),
      static_cast<double>(a1.sync_all_streams - a0.sync_all_streams),
      static_cast<double>(a1.alloc_retries - a0.alloc_retries)}};

  constexpr int nmetrics = ntime + nwork;
  std::array<double, nmetrics> local{}, minv{}, maxv{}, sumv{};
  for (int i = 0; i < ntime; ++i) local[i] = timing_ms[i];
  for (int i = 0; i < nwork; ++i) local[ntime + i] = work[i];

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Reduce(local.data(), minv.data(), nmetrics, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(local.data(), maxv.data(), nmetrics, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(local.data(), sumv.data(), nmetrics, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else
#endif
  {
    minv = local;
    maxv = local;
    sumv = local;
  }

  if (global_variable::my_rank == 0) {
    const double nranks = static_cast<double>(global_variable::nranks);
    static constexpr const char *timing_name[ntime] = {
        "gather", "forward", "readout", "exact1d", "kirchhoff", "gpu_total",
        "host_submit"};
    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(3)
              << "NN_PROFILE call=" << im.profile_sample_call_
              << " mode=" << im.profile_mode_
              << " ranks=" << global_variable::nranks << std::endl;
    std::cout << "NN_PROFILE time_ms min/mean/max";
    for (int i = 0; i < ntime; ++i) {
      std::cout << " " << timing_name[i] << "=" << minv[i] << "/"
                << sumv[i] / nranks << "/" << maxv[i];
    }
    std::cout << std::endl;

    static constexpr const char *work_name[nwork] = {
        "cells", "infer_cells", "kokkos_grow", "kokkos_required_bytes",
        "cublas_grow", "torch_requests", "torch_device_alloc",
        "torch_device_free", "torch_sync_all", "torch_alloc_retries"};
    std::cout << "NN_PROFILE counters min/mean/max";
    for (int i = 0; i < nwork; ++i) {
      const int m = ntime + i;
      std::cout << " " << work_name[i] << "=" << minv[m] << "/"
                << sumv[m] / nranks << "/" << maxv[m];
    }
    std::cout << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
  }

  im.profile_pending_ = false;
}

void NNOpacityEmulator::ProfileBegin(int n_cells, int infer_cells,
                                     bool kokkos_scratch_grew,
                                     size_t kokkos_scratch_bytes) const {
  Impl &im = *impl_;
  if (!im.profile_enabled_) return;
  ++im.profile_call_count_;
  if (im.profile_pending_ ||
      (im.profile_call_count_ % im.profile_interval_) != 0) {
    return;
  }

  im.profile_sample_call_ = im.profile_call_count_;
  im.profile_n_cells_ = n_cells;
  im.profile_infer_cells_ = infer_cells;
  im.profile_kokkos_scratch_grew_ = kokkos_scratch_grew;
  im.profile_kokkos_scratch_bytes_ = kokkos_scratch_bytes;
  im.profile_cublas_scratch_grew_ = false;
  im.profile_host_start_ = std::chrono::steady_clock::now();
  im.profile_alloc_start_ = im.AllocatorStats();
  im.profile_active_ = true;
  const cudaError_t err = cudaEventRecord(
      im.profile_events_[static_cast<int>(NNProfilePoint::start)],
      Kokkos::Cuda().cuda_stream());
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("NN profiler: start cudaEventRecord failed: ") +
        cudaGetErrorString(err));
  }
}

void NNOpacityEmulator::ProfileMark(NNProfilePoint point) const {
  Impl &im = *impl_;
  if (!im.profile_active_ || point == NNProfilePoint::start ||
      point == NNProfilePoint::count) {
    return;
  }
  const cudaError_t err = cudaEventRecord(
      im.profile_events_[static_cast<int>(point)], Kokkos::Cuda().cuda_stream());
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("NN profiler: cudaEventRecord failed: ") +
        cudaGetErrorString(err));
  }
  if (point == NNProfilePoint::kirchhoff) {
    const auto host_end = std::chrono::steady_clock::now();
    im.profile_host_submit_ms_ =
        std::chrono::duration<double, std::milli>(host_end -
                                                  im.profile_host_start_)
            .count();
    im.profile_alloc_end_ = im.AllocatorStats();
    im.profile_active_ = false;
    im.profile_pending_ = true;
  }
}

bool NNOpacityEmulator::WeightsReady() const { return impl_->weights_ready_; }
const NNWeights &NNOpacityEmulator::GetWeights() const { return impl_->weights_; }

//----------------------------------------------------------------------------------------
//! \fn NNOpacityEmulator::ExtractWeights
//! \brief Copy the trained parameters out of the .pt into device Views (NNWeights)
//!        for the fused Kokkos path.  Loads a fresh, UNfrozen module so that
//!        named_parameters() carry clean dotted names (freeze/optimize_for_inference
//!        inline them into constants).  Matching is by name SUFFIX, so any module
//!        prefix (e.g. "core.") is tolerated.  On a missing tensor it throws and
//!        lists the names that were found, so mismatches are self-diagnosing.
void NNOpacityEmulator::ExtractWeights() {
  if (impl_->model_path_.empty()) {
    throw std::runtime_error(
        "NNOpacityEmulator::ExtractWeights: Load() must be called first");
  }

  torch::jit::Module m;
  try {
    m = torch::jit::load(impl_->model_path_, torch::kCPU);
  } catch (const c10::Error &e) {
    throw std::runtime_error(
        std::string("NNOpacityEmulator::ExtractWeights: failed to reload '") +
        impl_->model_path_ + "': " + e.what());
  }
  m.eval();

  std::map<std::string, torch::Tensor> P;
  int n_blocks = 0;
  for (const auto &p : m.named_parameters()) {
    P[p.name] = p.value.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto pos = p.name.find("blocks.");
    if (pos != std::string::npos) {
      const int i = std::atoi(p.name.c_str() + pos + 7);
      n_blocks = std::max(n_blocks, i + 1);
    }
  }

  auto dump_names = [&P]() {
    std::string s;
    for (const auto &kv : P) s += "    " + kv.first + "\n";
    return s;
  };
  if (n_blocks == 0) {
    throw std::runtime_error(
        "NNOpacityEmulator::ExtractWeights: no 'blocks.*' parameters found; "
        "available:\n" + dump_names());
  }

  // Fetch a parameter by name suffix (prefix-agnostic).
  auto get = [&](const std::string &suffix) -> const torch::Tensor & {
    for (const auto &kv : P) {
      const std::string &k = kv.first;
      if (k.size() >= suffix.size() &&
          k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return kv.second;
      }
    }
    throw std::runtime_error(
        "NNOpacityEmulator::ExtractWeights: no parameter ending in '" + suffix +
        "'; available:\n" + dump_names());
  };

  auto cp_mat = [](const torch::Tensor &t, DvceArray2D<float> &d) {
    const int R = static_cast<int>(t.size(0)), C = static_cast<int>(t.size(1));
    Kokkos::realloc(d, R, C);
    auto h = Kokkos::create_mirror_view(d);
    const float *s = t.data_ptr<float>();
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) h(r, c) = s[r * C + c];
    Kokkos::deep_copy(d, h);
  };
  auto cp_vec = [](const torch::Tensor &t, DvceArray1D<float> &d) {
    const int R = static_cast<int>(t.size(0));
    Kokkos::realloc(d, R);
    auto h = Kokkos::create_mirror_view(d);
    const float *s = t.data_ptr<float>();
    for (int r = 0; r < R; ++r) h(r) = s[r];
    Kokkos::deep_copy(d, h);
  };

  NNWeights &w = impl_->weights_;
  w.n_blocks = n_blocks;
  const int H = HIDDEN;

  cp_mat(get("embed.0.weight"), w.W_embed);
  cp_vec(get("embed.0.bias"), w.b_embed);

  Kokkos::realloc(w.ln_g, n_blocks, H);
  Kokkos::realloc(w.ln_b, n_blocks, H);
  Kokkos::realloc(w.W1, n_blocks, H, H);
  Kokkos::realloc(w.b1, n_blocks, H);
  Kokkos::realloc(w.W2, n_blocks, H, H);
  Kokkos::realloc(w.b2, n_blocks, H);
  auto h_ln_g = Kokkos::create_mirror_view(w.ln_g);
  auto h_ln_b = Kokkos::create_mirror_view(w.ln_b);
  auto h_W1 = Kokkos::create_mirror_view(w.W1);
  auto h_b1 = Kokkos::create_mirror_view(w.b1);
  auto h_W2 = Kokkos::create_mirror_view(w.W2);
  auto h_b2 = Kokkos::create_mirror_view(w.b2);
  for (int i = 0; i < n_blocks; ++i) {
    const std::string pre = "blocks." + std::to_string(i) + ".net.";
    const float *slng = get(pre + "0.weight").data_ptr<float>();  // LayerNorm gamma
    const float *slnb = get(pre + "0.bias").data_ptr<float>();    // LayerNorm beta
    const float *sw1 = get(pre + "1.weight").data_ptr<float>();   // Linear1 (H,H)
    const float *sb1 = get(pre + "1.bias").data_ptr<float>();
    const float *sw2 = get(pre + "3.weight").data_ptr<float>();   // Linear2 (H,H)
    const float *sb2 = get(pre + "3.bias").data_ptr<float>();
    for (int o = 0; o < H; ++o) {
      h_ln_g(i, o) = slng[o];
      h_ln_b(i, o) = slnb[o];
      h_b1(i, o) = sb1[o];
      h_b2(i, o) = sb2[o];
      for (int k = 0; k < H; ++k) {
        h_W1(i, o, k) = sw1[o * H + k];
        h_W2(i, o, k) = sw2[o * H + k];
      }
    }
  }
  Kokkos::deep_copy(w.ln_g, h_ln_g);
  Kokkos::deep_copy(w.ln_b, h_ln_b);
  Kokkos::deep_copy(w.W1, h_W1);
  Kokkos::deep_copy(w.b1, h_b1);
  Kokkos::deep_copy(w.W2, h_W2);
  Kokkos::deep_copy(w.b2, h_b2);

  cp_mat(get("head_nue.weight"), w.W_nue);
  cp_vec(get("head_nue.bias"), w.b_nue);
  cp_mat(get("head_anue.weight"), w.W_anue);
  cp_vec(get("head_anue.bias"), w.b_anue);
  cp_mat(get("head_nux_anux.weight"), w.W_nux);
  cp_vec(get("head_nux_anux.bias"), w.b_nux);

  impl_->weights_ready_ = true;
  std::cout << "NNOpacityEmulator: extracted fused-kernel weights ("
            << n_blocks << " blocks, hidden=" << H << ")" << std::endl;
}

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
