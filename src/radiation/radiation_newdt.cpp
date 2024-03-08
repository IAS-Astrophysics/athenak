//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_newdt.cpp
//! \brief function to compute rad timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>
#include <float.h>

#include <limits>
#include <iostream>
#include <iomanip>    // std::setprecision()
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "driver/driver.hpp"
#include "radiation.hpp"
#include "radiation_tetrad.hpp"

namespace radiation {

//----------------------------------------------------------------------------------------
// \!fn void Radiation::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for radiation problems.
//        Only computed once at beginning of calculation.

TaskStatus Radiation::NewTimeStep(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &nx1 = indcs.nx1;
  int &js = indcs.js, &nx2 = indcs.nx2;
  int &ks = indcs.ks, &nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();
  Real dta = std::numeric_limits<float>::max();

  // setup indicies for Kokkos parallel reduce
  auto &size = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nang1 = prgeo->nangles - 1;

  // data needed to compute angular dt
  bool &angular_fluxes_ = angular_fluxes;
  auto &nh_c_ = nh_c;
  auto &na_ = na;
  auto &tet_c_ = tet_c;
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  auto &numn = prgeo->num_neighbors;
  auto &indn = prgeo->ind_neighbors;

  // find smallest (dx/c) and (dangle/na) in each direction for radiation problems
  Kokkos::parallel_reduce("RadiationNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx,Real &min_dt1,Real &min_dt2,Real &min_dt3,Real &min_dta) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real tmp_min_dta = (FLT_MAX);
    if (angular_fluxes_) {
      for (int n=0; n<=nang1; ++n) {
        // find position at angle center
        Real x = nh_c_.d_view(n,1);
        Real y = nh_c_.d_view(n,2);
        Real z = nh_c_.d_view(n,3);
        for (int nb=0; nb<numn.d_view(n); ++nb) {
          // find position at neighbor's angle center
          Real xn = nh_c_.d_view(indn.d_view(n,nb),1);
          Real yn = nh_c_.d_view(indn.d_view(n,nb),2);
          Real zn = nh_c_.d_view(indn.d_view(n,nb),3);
          // compute timestep limitation
          Real n0 = tet_c_(m,0,0,k,j,i);
          Real adt = fmin(tmp_min_dta,(acos(x*xn+y*yn+z*zn)/fabs(na_(m,n,k,j,i,nb)/n0)));
          // set timestep limitation if not excising this cell
          if (excise) {
            if (!(rad_mask_(m,k,j,i))) { tmp_min_dta = adt; }
          } else {
            tmp_min_dta = adt;
          }
        }
      }
    }
    min_dt1 = fmin((size.d_view(m).dx1), min_dt1);
    min_dt2 = fmin((size.d_view(m).dx2), min_dt2);
    min_dt3 = fmin((size.d_view(m).dx3), min_dt3);
    min_dta = fmin((tmp_min_dta),        min_dta);
  }, Kokkos::Min<Real>(dt1),  Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3),
     Kokkos::Min<Real>(dta));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }
  if (angular_fluxes_) { dtnew = std::min(dtnew, dta); }

  return TaskStatus::complete;
}
} // namespace radiation
