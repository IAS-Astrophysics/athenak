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
#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>    // c10::cuda::CUDAStreamGuard
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>   // c10::cuda::getStreamFromExternal
#include <c10/core/InferenceMode.h>
#include <torch/script.h>

// NOTE: intentionally no <Kokkos_Core.hpp> here.  This TU already pulls in the
// heavy LibTorch headers; adding Kokkos (also heavy, and CUDA-compiled) made it
// the slowest file in the build.  The CUDA stream is injected via SetStream() and
// this TU has no device code, so it can be built by the host compiler.

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
  float h_in_mean_[N_EOS] = {};
  float h_in_std_[N_EOS] = {};
  float h_out_mean_[N_OUTPUTS] = {};
  float h_out_std_[N_OUTPUTS] = {};
  torch::Device device_{torch::kCPU};
  bool loaded_ = false;

  // Kokkos CUDA stream, injected by SetStream() (the forward and profiler CUDA
  // events run on it).  Default 0 = the default stream until the caller sets it.
  mutable cudaStream_t stream_ = nullptr;

  // ── opt-in sampled profiler ───────────────────────────────────────────────
  // CUDA events are recorded on the Kokkos stream.  A later call queries the
  // terminal event and only consumes completed samples, so profiling never adds
  // a CUDA fence or stream synchronize to the opacity path.
  bool profile_enabled_ = false;
  int profile_interval_ = 100;
  mutable int64_t profile_call_count_ = 0;
  mutable int64_t profile_sample_call_ = 0;
  mutable int profile_n_cells_ = 0;
  mutable bool profile_kokkos_scratch_grew_ = false;
  mutable size_t profile_kokkos_scratch_bytes_ = 0;
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
  if (!use_cuda) {
    throw std::runtime_error(
        "NNOpacityEmulator::Load requires CUDA; CPU inference is not supported");
  }
  int cuda_dev = 0;
  const cudaError_t device_err = cudaGetDevice(&cuda_dev);
  if (device_err != cudaSuccess) {
    throw std::runtime_error(
        std::string("NNOpacityEmulator: cudaGetDevice failed: ") +
        cudaGetErrorString(device_err));
  }
  impl_->device_ = torch::Device(torch::kCUDA, cuda_dev);
  // Enable TF32 tensor cores for FP32 matmuls without changing tensor storage.
  at::globalContext().setAllowTF32CuBLAS(true);
  at::globalContext().setAllowTF32CuDNN(true);

  try {
    impl_->module_ = torch::jit::load(model_path, impl_->device_);
  } catch (const c10::Error &e) {
    throw std::runtime_error(
        std::string("NNOpacityEmulator: failed to load model from '") +
        model_path + "': " + e.what());
  }
  impl_->module_.eval();
  // Inline parameters and apply TorchScript's inference optimizations.
  impl_->module_ = torch::jit::freeze(impl_->module_);
  impl_->module_ = torch::jit::optimize_for_inference(impl_->module_);

  const torch::Tensor in_mean =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_in_mean.bin", N_EOS);
  const torch::Tensor in_std =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_in_std.bin", N_EOS);
  const torch::Tensor out_mean =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_out_mean.bin", N_OUTPUTS);
  const torch::Tensor out_std =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_out_std.bin", N_OUTPUTS);

  const auto cpu_imean = in_mean.contiguous();
  const auto cpu_istd = in_std.contiguous();
  const auto cpu_omean = out_mean.contiguous();
  const auto cpu_ostd = out_std.contiguous();
  for (int i = 0; i < N_EOS; ++i) {
    impl_->h_in_mean_[i] = cpu_imean.data_ptr<float>()[i];
    impl_->h_in_std_[i] = cpu_istd.data_ptr<float>()[i];
  }
  for (int i = 0; i < N_OUTPUTS; ++i) {
    impl_->h_out_mean_[i] = cpu_omean.data_ptr<float>()[i];
    impl_->h_out_std_[i] = cpu_ostd.data_ptr<float>()[i];
  }

  impl_->loaded_ = true;
  std::cout << "Loaded NN opacity emulator from " << model_path << std::endl;
}

void NNOpacityEmulator::SetStream(void *stream) const {
  impl_->stream_ = static_cast<cudaStream_t>(stream);
}

void NNOpacityEmulator::InferPrebuilt(const float *x_full_ptr, float *nn_out_ptr,
                                      int N) const {
  if (!impl_->loaded_ || !impl_->device_.is_cuda()) {
    throw std::runtime_error(
        "NNOpacityEmulator::InferPrebuilt requires a loaded CUDA model");
  }
  if (N == 0) return;

  c10::InferenceMode inference_guard;
  auto cuda_opts = torch::TensorOptions().dtype(torch::kFloat32).device(impl_->device_);

  // Model takes (N, 8) EOS features and produces (N, 32) outputs directly for
  // all 4 species — no tiling or one-hot needed.  Process all cells in one
  // pass: each rank has ≤25 MBs × 48³ ≈ 2.76M cells; A100 has 40 GB so the
  // full (2.76M,8) input + (2.76M,32) output fits easily (~440 MB).  Large N
  // also gives better tensor-core GEMM efficiency than small chunks.
  static constexpr int CHUNK_SIZE = 4194304;  // 4M — covers any realistic mesh

  // SCALING FIX: run LibTorch on the *Kokkos* CUDA stream (injected via
  // SetStream), so this forward is ordered in-line with gather_pack (before) and
  // the readout kernel (after) on a single stream.  That removes the need for the
  // cross-stream fence + event + cudaStreamSynchronize handoff, which serialised
  // this part of the step.  (kp_reader still resolves this forward under the
  // caller's "NN::InferPrebuilt" region, so no internal Kokkos marker is needed.)
  c10::cuda::CUDAStream torch_on_kokkos =
      c10::cuda::getStreamFromExternal(impl_->stream_, impl_->device_.index());
  c10::cuda::CUDAStreamGuard stream_guard(torch_on_kokkos);

  for (int base = 0; base < N; base += CHUNK_SIZE) {
    const int chunk = std::min(CHUNK_SIZE, N - base);

    // Zero-copy wrap of the pre-normalized (N, N_INPUTS=8) input buffer on GPU
    torch::Tensor x_in = torch::from_blob(
        const_cast<float *>(x_full_ptr +
                            static_cast<long long>(base) * N_INPUTS),
        {chunk, N_INPUTS}, cuda_opts);

    // Forward pass: (chunk, 8) → (chunk, 32)
    torch::Tensor y_norm = impl_->module_.forward({x_in}).toTensor();

    // Copy raw normalized output; denorm is fused into the readout kernel.
    torch::Tensor out_wrap = torch::from_blob(
        nn_out_ptr + static_cast<long long>(base) * N_OUTPUTS,
        {static_cast<long long>(chunk) * N_OUTPUTS}, cuda_opts);
    out_wrap.copy_(y_norm.reshape({-1}));
  }
  // No cudaStreamSynchronize here: the shared Kokkos stream already guarantees the
  // downstream readout kernel runs after this forward.  (Was:
  // cudaStreamSynchronize(lt_stream);)  Keeping a sync here would re-serialise the
  // step and re-break comm/compute overlap.
}

const float *NNOpacityEmulator::HostInMean() const { return impl_->h_in_mean_; }
const float *NNOpacityEmulator::HostInStd() const { return impl_->h_in_std_; }
const float *NNOpacityEmulator::HostOutMean() const { return impl_->h_out_mean_; }
const float *NNOpacityEmulator::HostOutStd() const { return impl_->h_out_std_; }

void NNOpacityEmulator::ConfigureProfiling(bool enabled, int interval) {
  Impl &im = *impl_;
  im.profile_enabled_ = enabled && im.device_.is_cuda();
  im.profile_interval_ = std::max(interval, 1);
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
  constexpr int nwork = 8;
  std::array<double, nwork> work{{
      static_cast<double>(im.profile_n_cells_),
      im.profile_kokkos_scratch_grew_ ? 1.0 : 0.0,
      static_cast<double>(im.profile_kokkos_scratch_bytes_),
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
              << " mode=torch"
              << " ranks=" << global_variable::nranks << std::endl;
    std::cout << "NN_PROFILE time_ms min/mean/max";
    for (int i = 0; i < ntime; ++i) {
      std::cout << " " << timing_name[i] << "=" << minv[i] << "/"
                << sumv[i] / nranks << "/" << maxv[i];
    }
    std::cout << std::endl;

    static constexpr const char *work_name[nwork] = {
        "cells", "kokkos_grow", "kokkos_required_bytes",
        "torch_requests", "torch_device_alloc",
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

void NNOpacityEmulator::ProfileBegin(int n_cells, bool kokkos_scratch_grew,
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
  im.profile_kokkos_scratch_grew_ = kokkos_scratch_grew;
  im.profile_kokkos_scratch_bytes_ = kokkos_scratch_bytes;
  im.profile_host_start_ = std::chrono::steady_clock::now();
  im.profile_alloc_start_ = im.AllocatorStats();
  im.profile_active_ = true;
  const cudaError_t err = cudaEventRecord(
      im.profile_events_[static_cast<int>(NNProfilePoint::start)], im.stream_);
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
      im.profile_events_[static_cast<int>(point)], im.stream_);
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

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
