#ifndef RADIATION_M1_RADIATION_M1_RHEA_HPP_
#define RADIATION_M1_RADIATION_M1_RHEA_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_rhea.hpp
//! \brief RheaModel: the model-evaluation boundary for the Rhea ML fast-flavor-conversion
//! mixing model.
//!
//! The interface below is backend-agnostic. Exactly one implementation is compiled
//! (src/CMakeLists.txt):
//!
//!   radiation_m1_rhea_kokkos.cpp  (default)  reads a flat `.rhea` file and evaluates the
//!       whole network for one cell inside a single Kokkos kernel. No external library.
//!   radiation_m1_rhea_torch.cpp   (ENABLE_TORCH)  reads a TorchScript `.pt` checkpoint
//!       and calls its `predict_all` method through LibTorch. Kept as the independent
//!       cross-check the Kokkos backend is validated against.
//!
//! Both read the same trained model (Rhea's export_rhea.py writes the `.rhea` from the
//! `.pt`) and return the same three per-cell outputs in the same units, so a
//! `rhea_model_path` naming a `.pt` or a `.rhea` is what distinguishes the two at run
//! time.
//!
//! RheaModel owns nothing about M1 physics: it takes a float32 `[n <= n_capacity, 2, NF,
//! 4]` device tensor of number 4-currents and hands back read-only Kokkos Views over the
//! predicted 4-currents, growth rate and stability flag.

#include <string>

#include "config.hpp"

#if ENABLE_RHEA

#include <memory>

#include "athena.hpp"

#if ENABLE_TORCH
#include <torch/script.h>  // NOLINT torch::jit::Module, torch::jit::load
#include <torch/torch.h>   // NOLINT
#endif

namespace radiationm1 {

#if !ENABLE_TORCH
// Opaque handle to the backend's loaded model, so this header stays free of Rhea's
// evaluator types (which live in Athena_RHEA_DIR and are only on the include path of
// radiation_m1_rhea_kokkos.cpp).
struct RheaKokkosImpl;
#endif

//----------------------------------------------------------------------------------------
//! \class RheaModel
//! \brief Owns the loaded Rhea model and all backend-specific state. Constructed once per
//! RadiationM1 instance, at startup, iff params.flavor_mix_type == FlavMixRhea.
class RheaModel {
 public:
  //! Number of flavors Rhea's contract fixes: F4_in/F4_out axis 2 has this extent.
  //! nspecies==4 (e, ebar, x, xbar) maps onto NF=3 (e, mu, tau) via i_flv_map/flv_fac; do
  //! not confuse this with RadiationM1::nspecies.
  static constexpr int kNumFlavors = 3;

  //--------------------------------------------------------------------------------------
  //! \struct Prediction
  //! \brief Read-only device Views over the three outputs, valid until the next Predict()
  //! call on the same RheaModel.
  //!
  //! With the LibTorch backend the torch::Tensor members below are what keep the
  //! underlying buffers alive -- never read through them, read through the Views -- and
  //! callers (ApplyRheaMixing) must hold the whole Prediction struct as a local for the
  //! duration of any par_for that reads these Views. The Kokkos backend's buffers are
  //! owned by RheaModel itself and live for the whole run.
  struct Prediction {
    Kokkos::View<const float****, LayoutWrapper, DevMemSpace> F4_out;      // [n,2,NF,4]
    Kokkos::View<const float*, LayoutWrapper, DevMemSpace> growthrate;     // [n]
    Kokkos::View<const float*, LayoutWrapper, DevMemSpace> stability;      // [n]
#if ENABLE_TORCH
    torch::Tensor f4_out_t, growthrate_t, stability_t;  // ownership only; do not read
#endif
  };

  //--------------------------------------------------------------------------------------
  //! model_path: required, no default (rhea_model_path has no default and startup fails
  //! without it; enforced by the caller, not here). A `.rhea` file for the Kokkos
  //! backend, a TorchScript `.pt` for the LibTorch one.
  //!
  //! n_capacity: batch CAPACITY, i.e. the largest extent(0) any call to Predict() on this
  //! instance will ever be given -- std::max(nmb_thispack, nmb_maxperrank) * nx1*nx2*nx3
  //! for this rank (radiation_m1.cpp), the same capacity-not-live-count sizing u0/u1/etc.
  //! already use so they survive AMR regrids without reallocation. Predict() accepts any
  //! active extent(0) <= n_capacity, so a regrid that shrinks the live nmb_thispack below
  //! the capacity this instance was sized at does not require RheaModel to be
  //! reconstructed.
  //!
  //! team_size: Kokkos backend only, the per-team thread count of the evaluation kernel
  //! (0 = the evaluator's own default), clamped to what the backend accepts. Ignored by
  //! the LibTorch backend.
  RheaModel(const std::string &model_path, int n_capacity, int team_size = 0);
  ~RheaModel();

  // Backend/device state below makes this non-copyable; moving is not needed
  // (constructed once, owned by a std::unique_ptr in RadiationM1).
  RheaModel(const RheaModel &) = delete;
  RheaModel &operator=(const RheaModel &) = delete;
  RheaModel(RheaModel &&) = delete;
  RheaModel &operator=(RheaModel &&) = delete;

  //--------------------------------------------------------------------------------------
  //! f4_in: [extent(0) <= n_capacity, 2, NF, 4], float32, device-resident,
  //! LayoutRight-contiguous. extent(0) (the ACTIVE batch size for this call) may be
  //! smaller than the capacity this RheaModel/its caller's scratch buffer were
  //! constructed with; only extent(0) is dynamic, extents 1-3 (2, NF, 4) stay exactly
  //! fixed. Both backends derive the flat buffer's strides from its shape alone, so the
  //! caller (radiation_m1_flavor_mix.cpp's FlavMixRhea branch) is responsible for the
  //! contiguity of the subview it slices out -- see the static_assert at that call site.
  //!
  //! Returns once the forward pass has been ENQUEUED on DevExeSpace()'s stream/queue --
  //! not necessarily complete. Safe to immediately enqueue further DevExeSpace() kernels
  //! that consume the returned Prediction with no explicit Kokkos::fence() in between,
  //! PROVIDED they too run on DevExeSpace() (same-stream ordering, not a real completion
  //! guarantee).
  Prediction Predict(Kokkos::View<const float****, LayoutWrapper, DevMemSpace> f4_in);

#if ENABLE_TORCH
  //! The torch::Device Predict() runs on -- resolved once at construction from Kokkos's
  //! own device query, never independently computed. Exposed for tests (the
  //! Kokkos/Torch device-index agreement check) and diagnostics.
  const torch::Device &device() const { return device_; }
#endif

 private:
  // Batch CAPACITY (not the live per-call active count) -- see the constructor comment
  // above and Predict()'s extent(0) <= n_capacity_ assert.
  int n_capacity_;

#if ENABLE_TORCH
  torch::jit::Module model_;
  torch::Device device_;
#else
  // Rhea's loaded tables plus the persistent output buffers, hidden behind a pimpl so
  // that RheaKokkos.hpp is included by radiation_m1_rhea_kokkos.cpp alone.
  std::unique_ptr<RheaKokkosImpl> impl_;
#endif
};

}  // namespace radiationm1

#endif  // ENABLE_RHEA
#endif  // RADIATION_M1_RADIATION_M1_RHEA_HPP_
