#ifndef RADIATION_M1_RHEA_HPP
#define RADIATION_M1_RHEA_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_rhea.hpp
//! \brief RheaModel: the Kokkos <-> LibTorch interop boundary for the Rhea ML
//! fast-flavor-conversion mixing model.
//!
//! This header (together with radiation_m1_rhea.cpp) is the only translation unit in
//! radiation_m1 permitted to contain backend-conditional compilation
//! (`#if defined(KOKKOS_ENABLE_CUDA)` / `KOKKOS_ENABLE_HIP` / `KOKKOS_ENABLE_SYCL`).
//! RheaModel's public interface below is backend-agnostic -- callers never see
//! torch::kCUDA vs torch::kXPU distinctions; the per-backend stream guard and
//! device-binding logic live entirely inside radiation_m1_rhea.cpp.
//!
//! RheaModel owns nothing about M1 physics: it moves a float32 `[n <= n_capacity, 2, NF,
//! 4]` device tensor into a loaded Rhea TorchScript module's `predict_all` method and
//! hands back read-only Kokkos Views over the (Torch-owned) outputs.
//!
//! The loaded/frozen torch::jit::Module itself is owned by a process-global cache
//! (`RheaModuleCache`, private to radiation_m1_rhea.cpp) keyed by (canonicalized model
//! path, device index) -- RheaModel only holds a cheap shared-handle copy of it, so
//! constructing more than one RheaModel for the same (path, device) does not re-read the
//! model from disk or re-upload weights.

#include <string>

#include "config.hpp"

#if ENABLE_TORCH

#include "athena.hpp"

#include <torch/script.h>  // NOLINT torch::jit::Module, torch::jit::load
#include <torch/torch.h>   // NOLINT

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \class RheaModel
//! \brief Owns the loaded Rhea TorchScript module and all backend-specific device/stream
//! state. Constructed once per RadiationM1 instance, at startup, iff
//! params.flavor_mix_type == FlavMixRhea.
class RheaModel {
 public:
  //! Number of flavors Rhea's contract fixes: F4_in/F4_out axis 2 has this extent.
  //! nspecies==4 (e, ebar, x, xbar) maps onto NF=3 (e, mu, tau) via i_flv_map/flv_fac; do
  //! not confuse this with RadiationM1::nspecies.
  static constexpr int kNumFlavors = 3;

  //--------------------------------------------------------------------------------------
  //! \struct Prediction
  //! \brief Owning handles + unmanaged device Views over the SAME memory, returned by
  //! Predict(). The torch::Tensor members exist ONLY to keep the underlying buffers alive
  //! -- never read through them, read through the Views. Views are `const` because
  //! ApplyRheaMixing only ever reads them.
  //!
  //! Callers (ApplyRheaMixing) must hold the whole Prediction struct as a local for the
  //! duration of any par_for that reads these Views, not just the Views themselves --
  //! destroying f4_out_t/growthrate_t/stability_t early can free the memory the Views
  //! point at.
  struct Prediction {
    Kokkos::View<const float****, LayoutWrapper, DevMemSpace> F4_out;      // [n,2,NF,4]
    Kokkos::View<const float*, LayoutWrapper, DevMemSpace> growthrate;     // [n]
    Kokkos::View<const float*, LayoutWrapper, DevMemSpace> stability;      // [n]
    torch::Tensor f4_out_t, growthrate_t, stability_t;  // ownership only; do not read
  };

  //--------------------------------------------------------------------------------------
  //! model_path: required, no default (rhea_model_path has no default and startup fails
  //! without it; enforced by the caller, not here).
  //!
  //! n_capacity: batch CAPACITY, i.e. the largest extent(0) any call to Predict() on this
  //! instance will ever be given -- std::max(nmb_thispack, nmb_maxperrank) * nx1*nx2*nx3
  //! for this rank (radiation_m1.cpp), the same capacity-not-live-count sizing u0/u1/etc.
  //! already use so they survive AMR regrids without reallocation. SUPERSEDES the old
  //! fixed-n_batch contract: Predict() now accepts any active extent(0) <= n_capacity,
  //! not only exactly n_capacity, because a regrid can shrink the live nmb_thispack below
  //! the capacity this instance/its scratch buffer were sized at without requiring
  //! RheaModel to be reconstructed.
  //!
  //! mem_fraction: Torch caching-allocator cap, fraction of device memory, CUDA/HIP/XPU
  //! only (no-op on CPU). Applied first-call-wins PER DEVICE by the process-global module
  //! cache (RheaModuleCache, radiation_m1_rhea.cpp): the cap is an allocator-level, not a
  //! per-instance, concept, so a second RheaModel construction targeting a device that
  //! has already been capped does not recap it, even if given a different mem_fraction
  //! here. Default is a conservatively small value for the rhea_mem_fraction input
  //! parameter.
  RheaModel(const std::string &model_path, int n_capacity, double mem_fraction = 0.05);
  ~RheaModel();

  // Backend/stream/device state below makes this non-copyable; moving is not needed
  // (constructed once, owned by a std::unique_ptr in RadiationM1).
  RheaModel(const RheaModel &) = delete;
  RheaModel &operator=(const RheaModel &) = delete;
  RheaModel(RheaModel &&) = delete;
  RheaModel &operator=(RheaModel &&) = delete;

  //--------------------------------------------------------------------------------------
  //! f4_in: [extent(0) <= n_capacity_, 2, NF, 4], float32, device-resident,
  //! LayoutRight-contiguous. extent(0) (the ACTIVE batch size for this call) may be
  //! smaller than the capacity this RheaModel/its caller's scratch buffer were
  //! constructed with -- a live nmb_thispack that shrinks below capacity (e.g. after an
  //! AMR regrid) needs no RheaModel reconstruction, it simply calls Predict() with a
  //! smaller active extent over the SAME preallocated buffer. Only extent(0) is dynamic;
  //! extents 1-3 (2, NF, 4) stay exactly fixed. The caller
  //! (radiation_m1_flavor_mix.cpp's FlavMixRhea branch) is responsible for slicing
  //! rhea_f4_in_scratch down to the live active extent before calling Predict() -- see
  //! that call site for the LayoutRight-contiguity argument this relies on (a
  //! leading-index-range subview of a LayoutRight view stays LayoutRight-contiguous, so
  //! from_blob below still needs no explicit strides). Returns once Torch's forward pass
  //! has been ENQUEUED on DevExeSpace()'s stream/queue -- not necessarily complete. Safe
  //! to immediately enqueue further DevExeSpace() kernels that consume the returned
  //! Prediction with no explicit Kokkos::fence() in between, PROVIDED they too run on
  //! DevExeSpace() (same-stream ordering, not a real completion guarantee).
  Prediction Predict(Kokkos::View<const float****, LayoutWrapper, DevMemSpace> f4_in);

  //! The torch::Device Predict() runs on -- resolved once at construction from Kokkos's
  //! own device query, never independently computed. Exposed for tests (the
  //! Kokkos/Torch device-index agreement check) and diagnostics.
  const torch::Device &device() const { return device_; }

 private:
  torch::jit::Module model_;
  torch::Device device_;
  // Batch CAPACITY (not the live per-call active count) -- see the constructor comment
  // above and Predict()'s extent(0) <= n_capacity_ assert (radiation_m1_rhea.cpp).
  int n_capacity_;
};

}  // namespace radiationm1

#endif  // ENABLE_TORCH
#endif  // RADIATION_M1_RHEA_HPP
