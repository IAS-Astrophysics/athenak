//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nn_emulator.cpp
//! \brief LibTorch implementation of the NN opacity emulator.

#if ENABLE_NN_OPACITY

#include "radiation_m1/radiation_m1_nn_emulator.hpp"

#include <algorithm>
#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/InferenceMode.h>
#include <torch/script.h>

#include <Kokkos_Core.hpp>  // Kokkos::Profiling::pushRegion/popRegion

namespace radiationm1 {

struct NNOpacityEmulator::Impl {
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

  cudaStream_t lt_stream =
      at::cuda::getCurrentCUDAStream(impl_->device_.index()).stream();

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
  cudaStreamSynchronize(lt_stream);
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

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
