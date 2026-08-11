//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nn_emulator.cpp
//! \brief LibTorch implementation of the NN opacity emulator (CUDA and Intel XPU).
//!
//! Backend-conditional code is kept to a minimum and is resolved entirely at compile time
//! (macros Kokkos/Torch define: KOKKOS_ENABLE_CUDA / KOKKOS_ENABLE_SYCL,
//! NN_TORCH_BACKEND_CUDA / NN_TORCH_BACKEND_XPU); there is no runtime backend dispatch.
//! Most of it lives in the "backend shim" anonymous namespace below, but NOT all of it --
//! two sites branch outside the shim because the two backends expose different APIs there
//! and cannot be unified without a wrapper:
//!   - InferPrebuilt() below: CUDA has c10::cuda::CUDAStreamGuard, XPU has no
//!     XPUStreamGuard and must go through the generic c10::StreamGuard.
//!   - radiation_m1_calc_opacities_nn.cpp: Kokkos hands back cuda_stream() by value but
//!     sycl_queue() by reference, so the two SetStream() calls differ.
//! If either site grows further, fold it into the shim rather than adding a third.
//!
//! HARDWARE COVERAGE (read this before trusting a result): the CUDA path is the original,
//! production-exercised one.  The XPU path was written against Aurora's LibTorch 2.10
//! headers, whose relevant declarations were verified directly
//! (c10/xpu/XPUStream.h:182 getStreamFromExternal(sycl::queue*, DeviceIndex),
//! c10/xpu/XPUEvent.h record/query/elapsed_time, c10/xpu/XPUCachingAllocator.h:39
//! getDeviceStats), and against the vendored Kokkos submodule's actual SYCL accessor
//! (kokkos/core/src/SYCL/Kokkos_SYCL.hpp:71 `sycl::queue& sycl_queue() const noexcept`).
//! It has NOT been compiled or executed on a PVC device.  Until it has, treat it as
//! "intended to compile" -- header signatures were checked by eye, not by a build.

#if ENABLE_NN_OPACITY

#include "radiation_m1/radiation_m1_nn_emulator.hpp"

#include "globals.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Kokkos_Macros.hpp is the ONLY Kokkos header this TU includes, and it is included
// solely for the KOKKOS_ENABLE_* config macros the shim below switches on.  It is a
// macro/config header (no templates, no device code, no execution spaces), so it does
// not reintroduce the compile-time cost the NOTE further down is about.  Taking the
// backend from Kokkos's own config rather than a parallel CMake-defined macro means the
// two can never silently disagree about which backend this build actually uses.
#include <Kokkos_Macros.hpp>

#include <c10/core/InferenceMode.h>
#include <torch/script.h>
// torch/script.h does NOT pull this in (checked: it includes types.h, the autograd and
// jit headers, and ATen/ATen.h, none of which reach it).  torch::set_num_threads /
// torch::set_num_interop_threads are declared here, as using-declarations for the at::
// versions -- ConfigureTorchThreadingOnce() below needs both.
#include <torch/utils.h>

#if defined(KOKKOS_ENABLE_CUDA)
#define NN_TORCH_BACKEND_CUDA 1
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>   // c10::cuda::CUDAStreamGuard
#include <c10/cuda/CUDAStream.h>  // c10::cuda::getStreamFromExternal
#elif defined(KOKKOS_ENABLE_SYCL)
#define NN_TORCH_BACKEND_XPU 1
#include <sycl/sycl.hpp>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUEvent.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#elif defined(KOKKOS_ENABLE_HIP)
// Deliberately not implemented rather than guessed at.  ROCm LibTorch hipifies its
// public API, so the *Torch* side of a HIP port is identical to the CUDA branch above
// (torch::kCUDA, c10::cuda::CUDAStreamGuard, c10::cuda::getStreamFromExternal all take
// HIP handles) -- but the raw device-runtime calls in the profiler shim are not:
// cuda_runtime_api.h -> hip/hip_runtime_api.h and every cudaEvent*/cudaGetErrorString
// call needs its hip* spelling.  That is a mechanical substitution, but it is one no
// one here can compile or run, and an untested silent-wrong path is worse than a build
// error.  Fill in the shim below and delete this #error if you have the hardware.
#error "NN opacity: the HIP backend is not wired up; see the shim in this file."
#else
#error "NN opacity requires a GPU-backed Kokkos build (Kokkos_ENABLE_CUDA or _SYCL)."
#endif

// NOTE: intentionally no <Kokkos_Core.hpp> here.  This TU already pulls in the
// heavy LibTorch headers; adding Kokkos (also heavy, and device-compiled) made it
// the slowest file in the build.  The device stream/queue is injected via SetStream()
// and this TU has no device code, so it can be built by the host compiler.

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace radiationm1 {

// The shim below is a NAMED namespace, not an anonymous one, for a specific reason:
// ProfileEvents is stored by value in NNOpacityEmulator::Impl, and Impl is forward
// declared in the header, so it has external linkage.  A member of internal-linkage
// type inside an external-linkage class is what -Wsubobject-linkage exists to complain
// about.  A named namespace + using-directive keeps the call sites clean without that.
namespace nn_backend {

//----------------------------------------------------------------------------------------
// ─────────────────────────────── backend shim ───────────────────────────────
// Everything below this block is backend-free.  Exactly one of the two
// implementations is ever compiled.
//
// The three things that genuinely differ between CUDA and XPU:
//   1. the torch::DeviceType and the opaque handle SetStream() carries,
//   2. how a Torch stream guard is built over a foreign (Kokkos-owned) stream/queue,
//   3. the device-event API the sampled profiler records into.
//----------------------------------------------------------------------------------------

constexpr int kNumProfilePoints = static_cast<int>(NNProfilePoint::count);

// What stream/queue Kokkos actually hands us, and why the guard is load-bearing:
// Kokkos does NOT run on the runtime's default stream/queue.  Under CUDA,
// Cuda::impl_initialize() explicitly cudaStreamCreate()s a singleton stream
// (kokkos/core/src/Cuda/Kokkos_Cuda_Instance.cpp) and cuda_stream() returns it; under
// SYCL, Kokkos constructs its own in-order sycl::queue
// (kokkos/core/src/SYCL/Kokkos_SYCL_Instance.cpp:95, with property::queue::in_order()).
// Torch's own current stream is a different one in both cases.  Without the guards
// below, the LibTorch forward would therefore NOT be ordered against the Kokkos
// gather/readout kernels that produce and consume its buffers -- this is a correctness
// requirement, not a performance nicety.

#if defined(NN_TORCH_BACKEND_CUDA)

constexpr torch::DeviceType kTorchDeviceType = torch::kCUDA;
constexpr const char *kBackendName = "cuda";
using StreamHandle = cudaStream_t;

inline StreamHandle StreamFromOpaque(void *p) { return static_cast<cudaStream_t>(p); }

//! \brief Apply backend-specific one-shot inference tuning at load time.
inline void ConfigureDeviceMathMode() {
  // Enable TF32 tensor cores for FP32 matmuls without changing tensor storage.
  at::globalContext().setAllowTF32CuBLAS(true);
  at::globalContext().setAllowTF32CuDNN(true);
}

//! \brief Torch caching-allocator counters, for the profiler's alloc-churn deltas.
inline auto AllocatorDeviceStats(c10::DeviceIndex index) {
  return c10::cuda::CUDACachingAllocator::getDeviceStats(index);
}

//! \brief Can this device time the profiler's events?  Always, under CUDA.
inline bool ProfilingSupported(c10::DeviceIndex) { return true; }

//----------------------------------------------------------------------------------------
//! \class ProfileEvents
//! \brief Fixed set of device events, one per NNProfilePoint, recorded on the Kokkos
//! stream/queue.  Uniform interface across backends so the profiler logic that uses it
//! stays backend-free.
class ProfileEvents {
 public:
  ~ProfileEvents() { Destroy(); }

  void Create(c10::DeviceIndex) {
    for (auto &event : events_) {
      const cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDefault);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("NN profiler: cudaEventCreate failed: ") +
                                 cudaGetErrorString(err));
      }
    }
    created_ = true;
  }

  void Destroy() noexcept {
    if (!created_) return;
    for (auto &event : events_) cudaEventDestroy(event);
    created_ = false;
  }

  void Record(int point, StreamHandle stream) {
    const cudaError_t err = cudaEventRecord(events_[point], stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("NN profiler: cudaEventRecord failed: ") +
                               cudaGetErrorString(err));
    }
  }

  //! \brief Non-blocking completion test.  NEVER synchronizes.
  bool Ready(int point) const {
    const cudaError_t query = cudaEventQuery(events_[point]);
    if (query != cudaSuccess && query != cudaErrorNotReady) {
      throw std::runtime_error(std::string("NN profiler: cudaEventQuery failed: ") +
                               cudaGetErrorString(query));
    }
    return query == cudaSuccess;
  }

  double ElapsedMs(int from, int to) const {
    float elapsed = 0.0f;
    const cudaError_t err = cudaEventElapsedTime(&elapsed, events_[from], events_[to]);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("NN profiler: cudaEventElapsedTime failed: ") +
          cudaGetErrorString(err));
    }
    return static_cast<double>(elapsed);
  }

 private:
  std::array<cudaEvent_t, kNumProfilePoints> events_{};
  bool created_ = false;
};

#elif defined(NN_TORCH_BACKEND_XPU)

constexpr torch::DeviceType kTorchDeviceType = torch::kXPU;
constexpr const char *kBackendName = "xpu";
// Unlike cudaStream_t (itself a pointer, passed by value), the SYCL handle is a POINTER
// TO the Kokkos-owned sycl::queue -- c10::xpu::getStreamFromExternal takes sycl::queue*.
using StreamHandle = sycl::queue *;

inline StreamHandle StreamFromOpaque(void *p) { return static_cast<sycl::queue *>(p); }

//! \brief No XPU analogue of TF32: PVC's FP32 matmul path has no equivalent knob, and
//! at::globalContext()'s TF32 setters are CUDA/cuDNN-specific.  Intentionally empty.
inline void ConfigureDeviceMathMode() {}

//! \brief Same shared c10::CachingDeviceAllocator::DeviceStats struct CUDA returns, so
//! every counter the profiler reads below (allocation[], num_device_alloc, ...) exists
//! identically on both backends -- verified in c10/core/CachingDeviceAllocator.h.
inline auto AllocatorDeviceStats(c10::DeviceIndex index) {
  return c10::xpu::XPUCachingAllocator::getDeviceStats(index);
}

//! \brief Whether XPUEvent::elapsed_time() can actually work on this device.
//!
//! XPUEvent(enable_timing=true) records via
//! sycl::ext::oneapi::experimental::submit_profiling_tag(), which (profiling_tag.hpp:24-38)
//! needs EITHER aspect::ext_oneapi_queue_profiling_tag on the device OR
//! property::queue::enable_profiling on the queue -- and it throws if it has neither.
//! Kokkos creates its queue with only property::queue::in_order()
//! (Kokkos_SYCL_Instance.cpp:95), never enable_profiling, and we deliberately do not
//! reach in to change that: we borrow Kokkos's queue, we do not own it.  So the device
//! aspect is the real requirement, and it is checked here (at ConfigureProfiling time)
//! rather than being discovered as a thrown exception mid-run.
inline bool ProfilingSupported(c10::DeviceIndex index) {
  return c10::xpu::get_raw_device(index).has(
      sycl::aspect::ext_oneapi_queue_profiling_tag);
}

//----------------------------------------------------------------------------------------
//! \class ProfileEvents
//! \brief XPU counterpart of the CUDA class above; same contract.
class ProfileEvents {
 public:
  void Create(c10::DeviceIndex index) {
    device_index_ = index;
    // enable_timing=true is what makes elapsed_time() legal on these events
    // (XPUEvent.h:134-136 rejects untimed ones).  Reserve first so the vector never
    // reallocates: XPUEvent is move-only by design (C10_DISABLE_COPY_AND_ASSIGN).
    events_.reserve(kNumProfilePoints);
    for (int i = 0; i < kNumProfilePoints; ++i) {
      events_.emplace_back(/*enable_timing=*/true);
    }
  }

  //! \brief No-op: ~XPUEvent releases the underlying sycl::event.  Kept so the
  //! backend-free Impl destructor below has one spelling on both backends.
  void Destroy() noexcept {}

  void Record(int point, StreamHandle queue) {
    // record() re-assigns the event on every call (XPUEvent.h reassignEvent), so
    // re-recording at each sampling interval behaves exactly like cudaEventRecord
    // overwriting a previously recorded event.
    events_[point].record(c10::xpu::getStreamFromExternal(queue, device_index_));
  }

  //! \brief Non-blocking completion test: reads command_execution_status, never waits.
  bool Ready(int point) const { return events_[point].query(); }

  double ElapsedMs(int from, int to) const {
    // XPUEvent::elapsed_time(other) returns (other.command_end - this.command_end) in ms
    // -- the same sign convention as cudaEventElapsedTime(&ms, from, to).
    return events_[from].elapsed_time(events_[to]);
  }

 private:
  std::vector<c10::xpu::XPUEvent> events_;
  c10::DeviceIndex device_index_ = 0;
};

#endif  // backend shim

//----------------------------------------------------------------------------------------
//! \fn void ConfigureTorchThreadingOnce()
//! \brief Cap Torch's CPU thread pools to 1, once per process.
//!
//! On a device backend Torch's intra-op CPU width is irrelevant to the forward pass, but
//! it is not free: Aurora packs many ranks per node, and a Torch pool sized to the full
//! socket in every rank oversubscribes the node against itself.  set_num_interop_threads
//! is a hard one-shot-per-process call (a second call aborts), so the guard is
//! load-bearing, not just an optimization.
void ConfigureTorchThreadingOnce() {
  static bool configured = false;
  if (configured) return;
  torch::set_num_interop_threads(1);
  torch::set_num_threads(1);
  configured = true;
}

}  // namespace nn_backend

using namespace nn_backend;  // NOLINT(build/namespaces) — see the note above the shim

struct NNOpacityEmulator::Impl {
  static constexpr int N_PROFILE_POINTS = kNumProfilePoints;

  struct AllocatorSnapshot {
    int64_t requests = 0;
    int64_t device_allocs = 0;
    int64_t device_frees = 0;
    int64_t sync_all_streams = 0;
    int64_t alloc_retries = 0;
  };

  mutable torch::jit::Module module_;
  float h_in_mean_[N_INPUTS] = {};
  float h_in_std_[N_INPUTS] = {};
  float h_out_mean_[N_OUTPUTS] = {};
  float h_out_std_[N_OUTPUTS] = {};
  torch::Device device_{torch::kCPU};
  bool loaded_ = false;

  // Kokkos device stream/queue, injected by SetStream() (the forward and the profiler
  // events run on it).  Null until the caller sets it; see the header for what each
  // backend expects to be passed.
  mutable StreamHandle stream_ = nullptr;

  // ── opt-in sampled profiler ───────────────────────────────────────────────
  // Device events are recorded on the Kokkos stream/queue.  A later call queries the
  // terminal event and only consumes completed samples, so profiling never adds
  // a fence or stream/queue synchronize to the opacity path.
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
  mutable ProfileEvents profile_events_{};
  mutable AllocatorSnapshot profile_alloc_start_{};
  mutable AllocatorSnapshot profile_alloc_end_{};
  mutable std::chrono::steady_clock::time_point profile_host_start_{};
  mutable double profile_host_submit_ms_ = 0.0;

  ~Impl() {
    if (profile_events_created_) profile_events_.Destroy();
  }

  bool OnDevice() const { return device_.type() == kTorchDeviceType; }

  AllocatorSnapshot AllocatorStats() const {
    AllocatorSnapshot out{};
    if (!OnDevice()) return out;
    const auto stats = AllocatorDeviceStats(device_.index());
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
                             const std::string &stats_dir, int device_index) {
  if (device_index < 0) {
    throw std::runtime_error(
        "NNOpacityEmulator::Load: invalid device index; pass Kokkos::device_id()");
  }
  impl_->device_ =
      torch::Device(kTorchDeviceType, static_cast<c10::DeviceIndex>(device_index));
  ConfigureDeviceMathMode();
  ConfigureTorchThreadingOnce();

  try {
    // Pass the device to load() rather than relying on a later .to(device): a
    // TorchScript archive records the device its tensors were serialized from and
    // torch::jit::load restores them there unless overridden, so a CUDA-saved
    // checkpoint otherwise fails inside deserialization -- before anything gets a
    // chance to move it.
    impl_->module_ = torch::jit::load(model_path, impl_->device_);
  } catch (const c10::Error &e) {
    throw std::runtime_error(
        std::string("NNOpacityEmulator: failed to load model from '") +
        model_path + "': " + e.what());
  }
  impl_->module_.eval();
  // Inline parameters and apply TorchScript's inference optimizations.  Both wrappers
  // hard-require a `forward` method to exist; this model exports one (it is invoked as
  // module_.forward(...) below), so unlike Rhea's predict_all-only contract they are
  // usable directly here.
  impl_->module_ = torch::jit::freeze(impl_->module_);
  impl_->module_ = torch::jit::optimize_for_inference(impl_->module_);

  // Input stats are N_INPUTS-dim (3 for the reduced build, 8 otherwise).
  const torch::Tensor in_mean =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_in_mean.bin", N_INPUTS);
  const torch::Tensor in_std =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_in_std.bin", N_INPUTS);
  const torch::Tensor out_mean =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_out_mean.bin", N_OUTPUTS);
  const torch::Tensor out_std =
      Impl::LoadFloat32Bin(stats_dir + "/nn2d_out_std.bin", N_OUTPUTS);

  const auto cpu_imean = in_mean.contiguous();
  const auto cpu_istd = in_std.contiguous();
  const auto cpu_omean = out_mean.contiguous();
  const auto cpu_ostd = out_std.contiguous();
  for (int i = 0; i < N_INPUTS; ++i) {
    impl_->h_in_mean_[i] = cpu_imean.data_ptr<float>()[i];
    impl_->h_in_std_[i] = cpu_istd.data_ptr<float>()[i];
  }
  for (int i = 0; i < N_OUTPUTS; ++i) {
    impl_->h_out_mean_[i] = cpu_omean.data_ptr<float>()[i];
    impl_->h_out_std_[i] = cpu_ostd.data_ptr<float>()[i];
  }

  impl_->loaded_ = true;
  if (global_variable::my_rank == 0) {
    std::cout << "Loaded NN opacity emulator from " << model_path << " on "
              << kBackendName << ":" << device_index << std::endl;
  }
}

void NNOpacityEmulator::SetStream(void *stream) const {
  impl_->stream_ = StreamFromOpaque(stream);
}

int NNOpacityEmulator::DeviceIndex() const {
  return impl_->loaded_ ? static_cast<int>(impl_->device_.index()) : -1;
}

void NNOpacityEmulator::InferPrebuilt(const float *x_full_ptr, float *nn_out_ptr,
                                      int N) const {
  if (!impl_->loaded_ || !impl_->OnDevice()) {
    throw std::runtime_error(
        std::string("NNOpacityEmulator::InferPrebuilt requires a loaded ") +
        kBackendName + " model");
  }
  if (impl_->stream_ == nullptr) {
    throw std::runtime_error(
        "NNOpacityEmulator::InferPrebuilt: SetStream() was never called; the forward "
        "would run off the Kokkos stream/queue and race the gather/readout kernels");
  }
  if (N == 0) return;

  c10::InferenceMode inference_guard;
  auto dev_opts = torch::TensorOptions().dtype(torch::kFloat32).device(impl_->device_);

  // Model takes (N, 8) EOS features and produces (N, 32) outputs directly for
  // all 4 species — no tiling or one-hot needed.  Process all cells in one
  // pass: each rank has ≤25 MBs × 48³ ≈ 2.76M cells; A100 has 40 GB (a PVC tile
  // 64 GB) so the full (2.76M,8) input + (2.76M,32) output fits easily (~440 MB).
  // Large N also gives better GEMM efficiency than small chunks.
  static constexpr int CHUNK_SIZE = 4194304;  // 4M — covers any realistic mesh

  // SCALING FIX: run LibTorch on the *Kokkos* stream/queue (injected via SetStream),
  // so this forward is ordered in-line with gather_pack (before) and the readout
  // kernel (after) on a single stream.  That removes the need for the cross-stream
  // fence + event + synchronize handoff, which serialised this part of the step.
  // (kp_reader still resolves this forward under the caller's "NN::InferPrebuilt"
  // region, so no internal Kokkos marker is needed.)
#if defined(NN_TORCH_BACKEND_CUDA)
  c10::cuda::CUDAStreamGuard stream_guard(
      c10::cuda::getStreamFromExternal(impl_->stream_, impl_->device_.index()));
#elif defined(NN_TORCH_BACKEND_XPU)
  // There is no XPUStreamGuard: XPU registers a c10::impl::DeviceGuardImplInterface
  // (c10/xpu/impl/XPUGuardImpl.h) and relies on the generic guard machinery.  XPUStream
  // has an implicit operator Stream(), so it converts cleanly to what c10::StreamGuard
  // wants.
  c10::StreamGuard stream_guard(
      c10::xpu::getStreamFromExternal(impl_->stream_, impl_->device_.index()));
#endif

  for (int base = 0; base < N; base += CHUNK_SIZE) {
    const int chunk = std::min(CHUNK_SIZE, N - base);

    // Zero-copy wrap of the pre-normalized (N, N_INPUTS=8) input buffer on device
    torch::Tensor x_in = torch::from_blob(
        const_cast<float *>(x_full_ptr +
                            static_cast<long long>(base) * N_INPUTS),
        {chunk, N_INPUTS}, dev_opts);

    // Forward pass: (chunk, 8) → (chunk, 32)
    torch::Tensor y_norm = impl_->module_.forward({x_in}).toTensor();

    // Copy raw normalized output; denorm is fused into the readout kernel.
    torch::Tensor out_wrap = torch::from_blob(
        nn_out_ptr + static_cast<long long>(base) * N_OUTPUTS,
        {static_cast<long long>(chunk) * N_OUTPUTS}, dev_opts);
    out_wrap.copy_(y_norm.reshape({-1}));
  }
  // No synchronize here: the shared Kokkos stream/queue already guarantees the
  // downstream readout kernel runs after this forward.  Keeping a sync here would
  // re-serialise the step and re-break comm/compute overlap.
}

const float *NNOpacityEmulator::HostInMean() const { return impl_->h_in_mean_; }
const float *NNOpacityEmulator::HostInStd() const { return impl_->h_in_std_; }
const float *NNOpacityEmulator::HostOutMean() const { return impl_->h_out_mean_; }
const float *NNOpacityEmulator::HostOutStd() const { return impl_->h_out_std_; }

void NNOpacityEmulator::ConfigureProfiling(bool enabled, int interval) {
  Impl &im = *impl_;
  im.profile_interval_ = std::max(interval, 1);
  im.profile_enabled_ = false;
  if (!enabled || !im.OnDevice()) return;

  // Decline rather than throw when the device cannot time events: the profiler is a
  // diagnostic, and losing it must never take the run down.
  if (!ProfilingSupported(im.device_.index())) {
    if (global_variable::my_rank == 0) {
      std::cout << "NN profiler requested but unavailable on this " << kBackendName
                << " device (no event-timing support); continuing without it"
                << std::endl;
    }
    return;
  }

  im.profile_events_.Create(im.device_.index());
  im.profile_events_created_ = true;
  im.profile_enabled_ = true;
  if (global_variable::my_rank == 0) {
    std::cout << "NN profiler enabled: sample every " << im.profile_interval_
              << " opacity calls; " << kBackendName
              << " events are queried asynchronously" << std::endl;
  }
}

void NNOpacityEmulator::ProfilePollAndReport() const {
  Impl &im = *impl_;
  if (!im.profile_enabled_ || !im.profile_pending_) return;

  // Querying only the terminal event is sufficient: both backends' streams are
  // in-order (Kokkos creates its SYCL queue with property::queue::in_order()), so a
  // completed last event implies every earlier one has completed too -- which is
  // exactly the precondition XPUEvent::elapsed_time() checks for.
  const int local_ready =
      im.profile_events_.Ready(static_cast<int>(NNProfilePoint::kirchhoff)) ? 1 : 0;

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
    timing_ms[p] = im.profile_events_.ElapsedMs(p, p + 1);
  }
  timing_ms[ngpu - 1] = im.profile_events_.ElapsedMs(
      static_cast<int>(NNProfilePoint::start),
      static_cast<int>(NNProfilePoint::kirchhoff));
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
              << " backend=" << kBackendName
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
  im.profile_events_.Record(static_cast<int>(NNProfilePoint::start), im.stream_);
}

void NNOpacityEmulator::ProfileMark(NNProfilePoint point) const {
  Impl &im = *impl_;
  if (!im.profile_active_ || point == NNProfilePoint::start ||
      point == NNProfilePoint::count) {
    return;
  }
  im.profile_events_.Record(static_cast<int>(point), im.stream_);
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
