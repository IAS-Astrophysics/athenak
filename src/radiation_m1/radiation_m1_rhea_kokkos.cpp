//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_rhea_kokkos.cpp
//! \brief RheaModel's Kokkos backend: Rhea's standalone evaluator, no M1 physics.
//! Compiled instead of radiation_m1_rhea_torch.cpp unless ENABLE_TORCH.
//!
//! The evaluator itself lives in Rhea (Athena_RHEA_DIR/cpp_interface/kokkos, GPLv3): it
//! reads a flat `.rhea` file written by Rhea's export_rhea.py and runs Box3D, every
//! network block and the conservation projections for one cell inside a single
//! KOKKOS_INLINE_FUNCTION, so one Predict() is one kernel launch. Nothing in it depends
//! on LibTorch or e3nn, and this translation unit is the only one that includes it.
//!
//! There is no backend-conditional compilation here and none is needed: the evaluator is
//! ordinary Kokkos and runs wherever DevExeSpace() does.

#include "radiation_m1/radiation_m1_rhea.hpp"

#if ENABLE_RHEA && !ENABLE_TORCH

#include <cassert>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

// athena.hpp (via the header above) has already pulled in Kokkos_Core.hpp, which is what
// makes rhea_model.hpp resolve RHEA_FN to KOKKOS_INLINE_FUNCTION rather than to a plain
// host `inline` (rhea_model.hpp's RHEA_FN block). RheaKokkos.hpp includes Kokkos_Core.hpp
// itself, so the order is not fragile, but do not move this include above athena.hpp.
#include <RheaKokkos.hpp>  // NOLINT rhea_predict_cell, RheaModelKokkos

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \struct RheaKokkosImpl
//! \brief Rhea's loaded model plus the output buffers Predict() writes into. Both are
//! allocated once, at construction: the model's tables are mirrored to the device by
//! RheaModelKokkos's constructor, and the outputs are sized at batch CAPACITY the same
//! way rhea_f4_in_scratch and every other radiation_m1 device array are, so an AMR regrid
//! that changes the live block count never reallocates and the run's device footprint is
//! fixed after startup.
struct RheaKokkosImpl {
  RheaKokkosImpl(const std::string &model_path, int n_capacity, int team_size)
      : model(model_path, team_size),
        f4_out("rhea_f4_out", static_cast<std::size_t>(n_capacity) * 2 *
                                  RheaModel::kNumFlavors * 4),
        growthrate("rhea_growthrate", static_cast<std::size_t>(n_capacity)),
        stability("rhea_stability", static_cast<std::size_t>(n_capacity)) {}

  RheaModelKokkos<DevExeSpace> model;
  // Flat, because that is the shape RheaModelKokkos::predict_all takes; Predict() hands
  // ApplyRheaMixing unmanaged [n,2,NF,4] / [n] Views over the same memory.
  Kokkos::View<float*, DevMemSpace> f4_out, growthrate, stability;
};

//----------------------------------------------------------------------------------------
//! \fn RheaModel::RheaModel
//! \brief Load the .rhea model and allocate the output buffers.
//!
//! Rhea's loader throws std::runtime_error for a file that is not a .rhea, an NF other
//! than 3, an irrep wider than RHEA_MAX_IRREP_DIM, or a model needing more per-cell
//! scratch than RHEA_MAX_SCRATCH -- the last names the value to rebuild with, which
//! reaches the header through the Athena_RHEA_MAX_SCRATCH cache variable. Report these as
//! startup failures rather than letting an exception escape into the task list.
RheaModel::RheaModel(const std::string &model_path, int n_capacity, int team_size)
    : n_capacity_(n_capacity) {
  try {
    impl_ = std::make_unique<RheaKokkosImpl>(model_path, n_capacity, team_size);
  } catch (const std::exception &e) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Could not load the Rhea model '" << model_path << "': " << e.what()
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
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

  const std::size_t n = f4_in.extent(0);
  const std::size_t nper = 2 * kNumFlavors * 4;

  // Unmanaged Views built from a raw pointer and an extent, rather than assigned from the
  // managed Views above: rank-1 Views are contiguous under any layout, but constructing
  // explicitly keeps the argument types exactly what RheaModelKokkos::predict_all
  // declares, with no layout conversion to reason about.
  Kokkos::View<const float*, DevMemSpace> in(f4_in.data(), n * nper);
  Kokkos::View<float*, DevMemSpace> out(impl_->f4_out.data(), n * nper);
  Kokkos::View<float*, DevMemSpace> gr(impl_->growthrate.data(), n);
  Kokkos::View<float*, DevMemSpace> st(impl_->stability.data(), n);

  impl_->model.predict_all(in, out, gr, st);

  Prediction pred;
  pred.F4_out = Kokkos::View<const float****, LayoutWrapper, DevMemSpace>(
      impl_->f4_out.data(), n, 2, kNumFlavors, 4);
  pred.growthrate =
      Kokkos::View<const float*, LayoutWrapper, DevMemSpace>(impl_->growthrate.data(), n);
  pred.stability =
      Kokkos::View<const float*, LayoutWrapper, DevMemSpace>(impl_->stability.data(), n);
  return pred;
}

}  // namespace radiationm1

#endif  // ENABLE_RHEA && !ENABLE_TORCH
