//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_newdt.cpp
//! \brief function to compute radiation timestep across all MeshBlock(s) in a
// MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_m1/radiation_m1.hpp"


namespace radiationm1 {

//----------------------------------------------------------------------------------------
// \!fn void RadiationM1::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for radiation problems

TaskStatus RadiationM1::NewTimeStep(Driver *pdrive, int stage) {
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &mbsize = pmy_pack->pmb->mb_size;
  int nmb1 = pmy_pack->nmb_thispack;

  Kokkos::parallel_reduce("radiation_m1_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb1),
                          KOKKOS_LAMBDA(const int &m, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
                              min_dt1 = fmin(mbsize.d_view(m).dx1, min_dt1);
                              min_dt2 = fmin(mbsize.d_view(m).dx2, min_dt2);
                              min_dt3 = fmin(mbsize.d_view(m).dx3, min_dt3);
                          }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace radiationm1
