//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_rhea.cpp
//! \brief RheaModel implementation: Kokkos <-> LibTorch interop, no M1 physics.
//!
//! This file (together with radiation_m1_rhea.hpp) is the only place in radiation_m1
//! allowed to branch on which Kokkos device backend is enabled. Every backend-conditional
//! block below is guarded by the same macros Kokkos itself defines
//! (`KOKKOS_ENABLE_CUDA`/`KOKKOS_ENABLE_HIP`/`KOKKOS_ENABLE_SYCL`), so exactly one of the
//! CUDA/HIP/SYCL/CPU code paths is ever compiled for a given build -- there is no runtime
//! backend dispatch here, only compile-time.
//!
//! NOTE on hardware coverage: this was implemented and structurally reviewed against
//! local LibTorch 2.12.1 headers (CPU-only build; the pinned version is 2.6.0) and
//! against the vendored Kokkos >=4.7 submodule's actual CUDA/HIP/SYCL accessor
//! implementations, but the CUDA/HIP/SYCL code paths below were never compiled or
//! executed on real hardware in this environment -- only the `#else` (CPU/Serial/OpenMP)
//! path was. Do not read "structurally validated" as "confirmed working."

#include "radiation_m1/radiation_m1_rhea.hpp"

#if ENABLE_TORCH

#include <cassert>
#include <filesystem>  // NOLINT std::filesystem::canonical (RheaModuleCache key)
#include <map>
#include <mutex>  // NOLINT cpplint's default deny-list predates broad C++11 header
                  // approval; this codebase already requires C++17 throughout (Kokkos),
                  // and std::mutex is exactly what RheaModuleCache needs (it is
                  // explicitly mutex-guarded below).
#include <set>
#include <string>
#include <utility>
#include <vector>

// torch::jit::freeze_module, the pass-level function behind torch::jit::freeze. Used
// directly (not through the torch::jit::freeze/torch::jit::optimize_for_inference
// convenience wrappers in torch/csrc/jit/api/module.h) -- see the RheaModel constructor
// below for why: those wrappers hard-require a `forward` method to exist REGARDLESS of
// what is passed as preserved_attrs/other_methods (verified directly against local
// LibTorch 2.12.1: both `torch::jit::freeze(model_, {"predict_all"})` and
// `torch::jit::optimize_for_inference(model_, {"predict_all"})` throw "Method 'forward' is
// not defined", because their C++ implementations call module.get_method("forward")
// unconditionally before ever looking at the method list). Rhea's TorchScript contract
// exports `predict_all`, not `forward`, as its entry point, so neither convenience
// wrapper is usable here. torch::jit::freeze_module (this header) is the lower-level pass
// both wrappers are ultimately built on; it takes an explicit list of method names with no
// hardcoded "forward" requirement, confirmed to inline submodule calls and constant-fold
// parameters into predict_all's graph correctly (spot-checked bit-for-bit against the
// unfrozen module's output).
#include <torch/csrc/jit/passes/freeze_module.h>

#if defined(KOKKOS_ENABLE_CUDA)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#elif defined(KOKKOS_ENABLE_HIP)
// HIP masquerades as CUDA in LibTorch's public API (official ROCm PyTorch builds
// reuse torch::kCUDA/c10::cuda::* for HIP devices -- there is no separate torch::kHIP
// code path to write here). We therefore pull in the same c10::cuda headers CUDA uses;
// only the Kokkos-side accessor differs (DevExeSpace().hip_stream()/hip_device() below).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#elif defined(KOKKOS_ENABLE_SYCL)
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#endif

namespace radiationm1 {

namespace {

//----------------------------------------------------------------------------------------
//! \fn torch::Device ResolveDevice()
//! \brief Resolve the torch::Device Rhea must run on. NEVER independently compute a
//! device index (e.g. from MPI rank or the launcher's environment) -- AthenaK has no
//! explicit MPI-rank-to-GPU binding code anywhere in src/; always query the exact index
//! Kokkos itself resolved to, so Torch and Kokkos agree on physical device regardless of
//! how the launcher or Kokkos's own rank-to-GPU heuristics chose it.
torch::Device ResolveDevice() {
#if defined(KOKKOS_ENABLE_CUDA)
  // Kokkos_Cuda.hpp:248: `static int device_id(const Cuda& exec) { return
  // exec.cuda_device(); }` -- confirmed present in the vendored Kokkos >=4.7 submodule.
  int const dev = Kokkos::Cuda::device_id(DevExeSpace());
  return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(dev));
#elif defined(KOKKOS_ENABLE_HIP)
  // Kokkos_HIP.hpp:156: `static int device_id(const HIP& exec) { return
  // exec.hip_device(); }`. Device type is still torch::kCUDA (masquerade, see the HIP
  // #include block above).
  int const dev = Kokkos::HIP::device_id(DevExeSpace());
  return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(dev));
#elif defined(KOKKOS_ENABLE_SYCL)
  // Kokkos_SYCL.hpp:139: `static int device_id(const Kokkos::SYCL& exec)` -- a static
  // function (SYCL has no instance-method form the way Cuda/HIP do).
  int const dev = Kokkos::SYCL::device_id(DevExeSpace());
  return torch::Device(torch::kXPU, static_cast<c10::DeviceIndex>(dev));
#else
  return torch::Device(torch::kCPU);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void CapAllocatorMemoryFraction(const torch::Device &device, double mem_fraction)
//! \brief Cap Torch's caching allocator as a safety margin (not because growth is
//! expected -- batch extents are deterministic and preallocated). No-op on CPU: the
//! host allocator has no analogous "fraction of device memory" concept to cap.
void CapAllocatorMemoryFraction(const torch::Device &device, double mem_fraction) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // c10/cuda/CUDACachingAllocator.h:130-131 (interface), :313-314 (free-function
  // forwarder) -- confirmed present in local LibTorch 2.12.1 headers.
  c10::cuda::CUDACachingAllocator::setMemoryFraction(
      mem_fraction, static_cast<c10::DeviceIndex>(device.index()));
#elif defined(KOKKOS_ENABLE_SYCL)
  // c10/xpu/XPUCachingAllocator.h:73 -- `C10_XPU_API void setMemoryFraction(double
  // fraction, DeviceIndex device);`. Found directly in the pinned-adjacent (2.12.1)
  // headers -- XPU has a symmetric cap to CUDA's.
  c10::xpu::XPUCachingAllocator::setMemoryFraction(
      mem_fraction, static_cast<c10::DeviceIndex>(device.index()));
#else
  (void)device;
  (void)mem_fraction;
#endif
}

//----------------------------------------------------------------------------------------
//! \class RheaModuleCache
//! \brief Process-global cache of loaded/frozen Rhea TorchScript modules, keyed by
//! (canonicalized model path, device index). A Meyers singleton, private to this
//! translation unit -- RheaModel (radiation_m1_rhea.hpp) never sees this type, only the
//! torch::jit::Module handle Get() returns, which it copies into its own model_ member.
//!
//! WHY a cache at all: every RheaModel construction used to unconditionally perform
//! three expensive steps that are pure functions of (model path, device) --
//! torch::jit::load (disk read + deserialization), model.to(device) (host->device weight
//! upload), and torch::jit::freeze_module (a full JIT optimization pass) -- even though
//! none of them depend on n_capacity, the only argument that ever varies between
//! RheaModel instances. The capacity-sized scratch buffer (rhea_f4_in_scratch, sized at
//! rank capacity so it survives AMR regrids without reallocation) already makes AMR
//! regrids need no RheaModel reconstruction at all; this cache makes any construction
//! that still happens anyway (repeated test construction, or a future workflow that does
//! rebuild the pack) cheap regardless -- a cache hit is a mutex lock, a map lookup, and a
//! torch::jit::Module copy (an intrusive_ptr bump, not a deep copy: Module is internally
//! a shared object handle, so the copy returned here and the one RheaModel stores both
//! point at the SAME frozen graph and GraphExecutor shape-specialization cache -- a
//! second RheaModel for the same key inherits the first one's warmed-up JIT state
//! instead of starting cold).
//!
//! Sharing safety: the module is immutable after freeze_module (weights are
//! constant-folded into the frozen graph), and only one RheaModel runs Predict() at a
//! time per rank (one MeshBlockPack, host-driven task loop) -- the mutex below protects
//! the maps themselves, not concurrent inference (there is none to protect against). MPI
//! ranks are separate processes, so this cache is per-rank by construction: one disk read
//! and one device upload per rank at startup, same as before this cache existed.
//!
//! No invalidation: the model file is assumed immutable for the lifetime of the process,
//! so canonicalizing the path (std::filesystem::canonical) is enough to avoid
//! same-file-different-spelling duplicate cache entries; content checksumming is not
//! worth it.
class RheaModuleCache {
 public:
  static RheaModuleCache &Instance() {
    static RheaModuleCache instance;
    return instance;
  }

  // Returns a cheap-to-copy handle to the loaded+frozen module for (model_path, device),
  // doing the one-shot load/freeze work only the first time this exact (canonicalized
  // path, device index) key is requested, and the one-shot threading/allocator-cap work
  // only the first time ever / first time this device index is seen, respectively.
  torch::jit::Module Get(const std::string &model_path, const torch::Device &device,
                         double mem_fraction) {
    // std::filesystem::canonical requires the path to exist -- true here regardless,
    // since torch::jit::load below would fail on a missing file anyway; this merely
    // fails slightly earlier, with a clearer error.
    const std::string canonical_path = std::filesystem::canonical(model_path).string();
    const int device_index = static_cast<int>(device.index());
    const Key key{canonical_path, device_index};

    std::lock_guard<std::mutex> lock(mutex_);

    // Process-wide one-shot threading configuration, moved here from RheaModel's
    // constructor (this is the natural home for it now: it is a process-global concern,
    // not a per-instance one). set_num_interop_threads is a hard one-shot-per-process
    // call (must happen before any inter-op/JIT work starts) -- the guard below is
    // load-bearing (a second call would abort). set_num_threads is safe to call
    // repeatedly; it is folded into this same one-shot block only because there is
    // nothing to gain from repeating it. Reasoning for capping both to 1,
    // unconditionally, on every backend: on device backends Torch's CPU intra-op
    // thread count is irrelevant to the device-side forward pass; on CPU our
    // parallelism source is Kokkos::OpenMP batching over zones (one batched Torch
    // call, not per-zone CPU threading), so giving Torch's CPU thread pool any width
    // only risks oversubscription against Kokkos::OpenMP's own threads.
    //
    // CAVEAT (flagged, not resolved here): pytorch/pytorch#19213 reports set_num_threads
    // does not always reliably cap real CPU usage on all systems/MKL configurations.
    // torch::get_num_threads() == 1 (the API's own reported value) is what is checkable
    // standalone; it does NOT independently confirm the OS thread count actually
    // launched matches, which would need an OS-level thread audit.
    if (!threading_configured_) {
      torch::set_num_interop_threads(1);
      torch::set_num_threads(1);
      threading_configured_ = true;
    }

    auto it = modules_.find(key);
    if (it == modules_.end()) {
      torch::jit::Module model = torch::jit::load(model_path);
      model.to(device);
      model.eval();

      // Freeze once here, at first-load time for this key -- low risk (one-time cost,
      // amortized across every RheaModel that shares this key), always-on default. Uses
      // the pass-level torch::jit::freeze_module directly rather than the higher-level
      // torch::jit::freeze/torch::jit::optimize_for_inference convenience wrappers: see
      // the #include block above this file's namespace for why (both wrappers
      // hard-require a `forward` method, which Rhea's `predict_all`-only contract does
      // not have -- verified directly against local LibTorch 2.12.1). Explicitly
      // preserving "predict_all" is what makes freeze_module inline submodule calls and
      // constant-fold parameters into ITS graph specifically. freeze_module returns a
      // new (frozen) Module; the loaded model is intentionally replaced, not mutated in
      // place.
      model = torch::jit::freeze_module(model, std::vector<std::string>{"predict_all"});
      it = modules_.emplace(key, std::move(model)).first;
    }

    // Allocator memory-fraction cap: first-call-wins PER DEVICE -- keyed on device_index
    // alone, not the full (path, device) key, since the cap is an allocator-level
    // concept, not a per-model one. A second model loaded onto an already-capped device
    // does not recap it, even if called with a different mem_fraction argument.
    if (capped_devices_.insert(device_index).second) {
      CapAllocatorMemoryFraction(device, mem_fraction);
    }

    return it->second;
  }

 private:
  RheaModuleCache() = default;

  // (canonicalized model path, device index) -- the cache key exactly.
  using Key = std::pair<std::string, int>;

  std::mutex mutex_;
  std::map<Key, torch::jit::Module> modules_;
  std::set<int> capped_devices_;
  bool threading_configured_ = false;
};

//----------------------------------------------------------------------------------------
// Per-backend stream-guard construction. Each guard is bound to
// DevExeSpace()'s own stream/queue for the scope of the predict_all call below, rather
// than fencing before and after -- same-stream/queue ordering is guaranteed by the
// device runtime at zero host-side-wait cost. Exactly one of these is ever compiled.
//
// What stream a default-constructed Kokkos::Cuda() actually holds: read directly from
// the vendored Kokkos submodule,
// kokkos/core/src/Cuda/Kokkos_Cuda_Instance.cpp:631-632,640: Cuda::impl_initialize()
// calls `cudaStreamCreate(&singleton_stream)` and then
// `Impl::CudaInternal::singleton().initialize(singleton_stream)`; cuda_stream() (:738)
// returns `m_space_instance->m_stream`, i.e. that same explicitly-created stream. This
// is a Kokkos-owned NON-default, NON-legacy CUDA stream, not the CUDA legacy default
// stream (stream 0 / nullptr). Consequence: without this guard, Torch's own
// default/current CUDA stream would NOT be the same stream Kokkos's pack/unpack kernels
// run on, so the guard is load-bearing for correctness, not merely best-practice.

#if defined(KOKKOS_ENABLE_CUDA)
c10::cuda::CUDAStreamGuard MakeCudaStreamGuard() {
  // c10/cuda/CUDAStream.h:216-217 `getStreamFromExternal(cudaStream_t, DeviceIndex)`;
  // c10/cuda/CUDAGuard.h:145-152 `CUDAStreamGuard(Stream)` -- both confirmed present in
  // local LibTorch 2.12.1 headers.
  return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(
      DevExeSpace().cuda_stream(),
      static_cast<c10::DeviceIndex>(DevExeSpace().cuda_device())));
}
#elif defined(KOKKOS_ENABLE_HIP)
c10::cuda::CUDAStreamGuard MakeHipStreamGuard() {
  // Identical call to CUDA's, since HIP masquerades as CUDA in LibTorch.
  // Confirmed: Kokkos_HIP.hpp:92 `hipStream_t hip_stream() const;`, :103
  // `int hip_device() const;`.
  return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(
      DevExeSpace().hip_stream(),
      static_cast<c10::DeviceIndex>(DevExeSpace().hip_device())));
}
#elif defined(KOKKOS_ENABLE_SYCL)
c10::StreamGuard MakeXpuStreamGuard(c10::DeviceIndex device_index) {
  // Grepped the local (2.12.1) LibTorch install's full include tree (`c10/xpu/`,
  // `c10/core/`, `torch/csrc/`) for "XPUStreamGuard" -- the only hits are in
  // torch/csrc/inductor/aoti_{runtime,torch} (AOT-compile shims, unrelated). There is NO
  // dedicated XPUStreamGuard class, confirmed by directly reading
  // c10/xpu/impl/XPUGuardImpl.h: XPU registers a c10::impl::DeviceGuardImplInterface
  // implementation (XPUGuardImpl) for kXPU and relies entirely on the generic
  // c10::StreamGuard/c10::DeviceGuard machinery -- there is no XPU-specific guard type
  // the way CUDAStreamGuard exists for CUDA. c10::xpu::getStreamFromExternal(sycl::
  // queue*, DeviceIndex) (c10/xpu/XPUStream.h:187) returns an XPUStream, which has an
  // implicit `operator Stream() const` (XPUStream.h:68-70), so it converts cleanly to
  // the c10::Stream the generic c10::StreamGuard(Stream) constructor
  // (c10/core/StreamGuard.h:35) wants. This was checked against 2.12.1, newer than the
  // pinned 2.6.0 -- genuinely NOT verified against 2.6.0 itself in this environment (no
  // 2.6.0 install available here); flagged, not assumed identical.
  return c10::StreamGuard(
      c10::xpu::getStreamFromExternal(&DevExeSpace().sycl_queue(), device_index));
}
#endif

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn RheaModel::RheaModel
//! \brief Resolve the Kokkos-agreeing device, then hand off to the process-global
//! RheaModuleCache (above) for everything else. Collapses to a cache lookup plus a
//! torch::jit::Module handle copy -- no disk read, no device
//! upload, and no JIT freeze pass happen here unless this is the first RheaModel ever
//! constructed for this (model_path, device) pair in this process. n_capacity is stored
//! only for Predict()'s extent(0) <= n_capacity_ bounds check; it plays no
//! role in what the cache loads, since none of the cached one-shot work depends on it.
RheaModel::RheaModel(const std::string &model_path, int n_capacity, double mem_fraction)
    : device_(ResolveDevice()), n_capacity_(n_capacity) {
  model_ = RheaModuleCache::Instance().Get(model_path, device_, mem_fraction);
}

//----------------------------------------------------------------------------------------
//! \fn RheaModel::~RheaModel
RheaModel::~RheaModel() = default;

//----------------------------------------------------------------------------------------
//! \fn RheaModel::Prediction RheaModel::Predict
RheaModel::Prediction RheaModel::Predict(
    Kokkos::View<const float****, LayoutWrapper, DevMemSpace> f4_in) {
  // extent(0) is the ACTIVE batch size for this call and may be less than n_capacity_ --
  // e.g. after an AMR regrid shrinks the live nmb_thispack below the capacity
  // rhea_f4_in_scratch/this RheaModel were sized at. Only extents 1-3 (2, NF, 4) are
  // still asserted exactly; the batch axis is the only dynamic one.
  assert(static_cast<int>(f4_in.extent(0)) <= n_capacity_);
  assert(static_cast<int>(f4_in.extent(1)) == 2);
  assert(static_cast<int>(f4_in.extent(2)) == kNumFlavors);
  assert(static_cast<int>(f4_in.extent(3)) == 4);

  // torch::from_blob never takes ownership by default -- with no custom deleter, the
  // context is {nullptr, noopDelete} (ATen/ops/from_blob.h), so destroying the resulting
  // Tensor is a true no-op w.r.t. f4_in's storage. f4_in's shape is read directly off its
  // own extents (not off n_capacity_), so a smaller active extent(0) than capacity is
  // handled automatically here -- no separate code path. f4_in is always a LayoutRight
  // (row-major) view sized [<=n_capacity_,2,NF,4] -- either rhea_f4_in_scratch itself, or
  // a leading-index-range subview/reconstruction of it that stays LayoutRight-contiguous,
  // matching Torch's default contiguous strides for this shape -- so no explicit strides
  // argument is ever needed here. The const_cast is safe:
  // Predict() only ever reads through f4_in_t (feeds it to the model as input); it never
  // writes through it, and f4_in itself is caller-owned memory this function borrows,
  // not allocates.
  torch::Tensor f4_in_t = torch::from_blob(
      const_cast<float *>(f4_in.data()),
      {static_cast<int64_t>(f4_in.extent(0)), static_cast<int64_t>(f4_in.extent(1)),
       static_cast<int64_t>(f4_in.extent(2)), static_cast<int64_t>(f4_in.extent(3))},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  // torch::InferenceMode guard around the actual predict_all invocation -- essentially
  // free, always-on default (no autograd bookkeeping needed; Predict() never needs
  // gradients). Scoped as the OUTER guard here, enclosing each backend's own stream/queue
  // guard below (MakeCudaStreamGuard() etc., constructed inside the per-backend blocks):
  // both guards must be active for the actual predict_all call, and InferenceMode's
  // thread-local dispatch-key state is independent of the per-backend stream guard's
  // thread-local current-stream state, so the nesting order between the two has no
  // functional consequence -- InferenceMode is placed outermost here only because it
  // applies uniformly across all four backend branches (including the CPU fallback,
  // which has no stream guard of its own), while the stream guard is backend-specific
  // and lives inside each branch.
  torch::InferenceMode inference_mode_guard;

  torch::IValue out_ivalue;
#if defined(KOKKOS_ENABLE_CUDA)
  {
    auto stream_guard = MakeCudaStreamGuard();
    out_ivalue = model_.get_method("predict_all")({f4_in_t});
  }
#elif defined(KOKKOS_ENABLE_HIP)
  {
    auto stream_guard = MakeHipStreamGuard();
    out_ivalue = model_.get_method("predict_all")({f4_in_t});
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  {
    auto stream_guard =
        MakeXpuStreamGuard(static_cast<c10::DeviceIndex>(device_.index()));
    out_ivalue = model_.get_method("predict_all")({f4_in_t});
  }
#else
  // CPU (Serial/OpenMP): no stream/queue concept. Host par_for calls block the
  // calling thread until completion (unlike async device dispatch), so the pack kernel,
  // this call, and the unpack kernel are trivially ordered by sequential host
  // execution -- no guard, no fence needed.
  out_ivalue = model_.get_method("predict_all")({f4_in_t});
#endif

  // predict_all returns (F4_out, growthrate, stability), in that order
  // (ml_neuralnet.py:324).
  auto elements = out_ivalue.toTuple()->elements();

  Prediction pred;
  pred.f4_out_t = elements[0].toTensor();
  pred.growthrate_t = elements[1].toTensor();
  pred.stability_t = elements[2].toTensor();

  // Construct unmanaged Kokkos Views directly over the Torch-owned output buffers -- no
  // separate output-side copy/unpack buffer. pred keeps f4_out_t/growthrate_t/stability_t
  // alive for as long as the Views below are valid; the caller (ApplyRheaMixing) must
  // hold the whole Prediction as a local for the duration of the par_for that reads
  // these Views.
  pred.F4_out = Kokkos::View<const float****, LayoutWrapper, DevMemSpace>(
      pred.f4_out_t.data_ptr<float>(), pred.f4_out_t.size(0), pred.f4_out_t.size(1),
      pred.f4_out_t.size(2), pred.f4_out_t.size(3));
  pred.growthrate = Kokkos::View<const float*, LayoutWrapper, DevMemSpace>(
      pred.growthrate_t.data_ptr<float>(), pred.growthrate_t.size(0));
  pred.stability = Kokkos::View<const float*, LayoutWrapper, DevMemSpace>(
      pred.stability_t.data_ptr<float>(), pred.stability_t.size(0));

  return pred;
}

}  // namespace radiationm1

#endif  // ENABLE_TORCH
