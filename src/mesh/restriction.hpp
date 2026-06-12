#ifndef MESH_RESTRICTION_HPP_
#define MESH_RESTRICTION_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restriction.hpp
//! \brief restriction operators for cell-centered variables,
//! implemented as templated inline functions so they can be used for z4c
//! with different order of spatial differencing order.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real RestrictInterpolation(const int m, const int v, const int fk, const int fj,
                          const int fi, const int nx1, const int nx2, const int nx3,
                          const DvceArray5D<Real> &a,
                          const DualArray1D<Real> &restrict_2nd,
                          const DualArray1D<Real> &restrict_4th,
                          const DualArray1D<Real> &restrict_4th_edge) {
  // interpolated value at new grid point
  Real ivals = 0;

  bool offseti = (fi<nx1/2+NGHOST);
  bool offsetj = (fj<nx2/2+NGHOST);
  bool offsetk = (fk<nx3/2+NGHOST);

  if (NGHOST ==2) {
    int refi = (offseti) ? fi : fi-1;
    int refj = (offsetj) ? fj : fj-1;
    int refk = (offsetk) ? fk : fk-1;
    for (int ii=0; ii<NGHOST+1; ii++) {
      for (int jj=0; jj<NGHOST+1; jj++) {
        for (int kk=0; kk<NGHOST+1; kk++) {
          int wghti = (offseti) ? ii : NGHOST-ii;
          int wghtj = (offsetj) ? jj : NGHOST-jj;
          int wghtk = (offsetk) ? kk : NGHOST-kk;
          Real iwght = restrict_2nd.d_view(wghti)
                       *restrict_2nd.d_view(wghtj)
                       *restrict_2nd.d_view(wghtk);
          ivals += iwght*a(m,v,refk+kk,refj+jj,refi+ii);
        }
      }
    }
  }

  if (NGHOST ==4) {
    int refi = (offseti) ? fi-1 : fi-2;
    int refj = (offsetj) ? fj-1 : fj-2;
    int refk = (offsetk) ? fk-1 : fk-2;

    // edge cases
    refi = (fi==NGHOST) ? refi+1 : refi;
    refj = (fj==NGHOST) ? refj+1 : refj;
    refk = (fk==NGHOST) ? refk+1 : refk;

    refi = (fi==NGHOST+nx1-2) ? refi-1 : refi;
    refj = (fj==NGHOST+nx2-2) ? refj-1 : refj;
    refk = (fk==NGHOST+nx3-2) ? refk-1 : refk;

    for (int ii=0; ii<NGHOST+1; ii++) {
      for (int jj=0; jj<NGHOST+1; jj++) {
        for (int kk=0; kk<NGHOST+1; kk++) {
          int wghti = (offseti) ? ii : NGHOST-ii;
          int wghtj = (offsetj) ? jj : NGHOST-jj;
          int wghtk = (offsetk) ? kk : NGHOST-kk;
          Real iwght = 1;
          if (fi==NGHOST || fi==NGHOST+nx1-2) {
            iwght *= restrict_4th_edge.d_view(wghti);
          } else {
            iwght *= restrict_4th.d_view(wghti);
          }
          if (fj==NGHOST || fj==NGHOST+nx2-2) {
            iwght *= restrict_4th_edge.d_view(wghtj);
          } else {
            iwght *= restrict_4th.d_view(wghtj);
          }
          if (fk==NGHOST || fk==NGHOST+nx3-2) {
            iwght *= restrict_4th_edge.d_view(wghtk);
          } else {
            iwght *= restrict_4th.d_view(wghtk);
          }
          ivals += iwght*a(m,v,refk+kk,refj+jj,refi+ii);
        }
      }
    }
  }
  return ivals;
}
#endif // MESH_RESTRICTION_HPP_
