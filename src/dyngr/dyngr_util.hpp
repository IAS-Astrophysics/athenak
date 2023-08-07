#ifndef DYNGR_UTIL_HPP_
#define DYNGR_UTIL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_util.hpp
//  \brief Utility functions for use with dynamic hydro

#include "athena.hpp"
#include "adm/adm.hpp"
#include "dyngr.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "reconstruct/plm.hpp"

namespace dyngr {

template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void ExtractPrimitives(const DvceArray5D<Real>& prim, Real prim_pt[NPRIM],
                       const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
                       const int& nhyd, const int& nscal,
                       const int m, const int k, const int j, const int i) {
  Real mb = eos.ps.GetEOS().GetBaryonMass();
  prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
  prim_pt[PVX] = prim(m, IVX, k, j, i);
  prim_pt[PVY] = prim(m, IVY, k, j, i);
  prim_pt[PVZ] = prim(m, IVZ, k, j, i);
  for (int s = 0; s < nscal; s++) {
    prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
  }
  prim_pt[PPR] = prim(m, IPR, k, j, i);
  prim_pt[PTM] = eos.ps.GetEOS().GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
}

template<int dir, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void ExtractPrimitivesWithMinmod(const DvceArray5D<Real>& prim, Real prim_l[NPRIM], Real prim_r[NPRIM],
                                 const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
                                 const int& nhyd, const int& nscal,
                                 const int m, const int k, const int j, const int i) {
  int im1, jm1, km1;
  int ip1, jp1, kp1;
  // Modify the extraction indices based on direction.
  switch(dir) {
    case IVX:
      im1 = i-1;
      ip1 = i+1;
      jm1 = jp1 = j;
      km1 = kp1 = k;
      break;
    case IVY:
      im1 = ip1 = i;
      jm1 = j-1;
      jp1 = j+1;
      km1 = kp1 = k;
      break;
    case IVZ:
      im1 = ip1 = i;
      jm1 = jp1 = j;
      km1 = k-1;
      kp1 = k+1;
      break;
  }
  Real mb = eos.ps.GetEOS().GetBaryonMass();
  // Reconstruct all the interfaces for each primitive variable. A loop would be more elegant,
  // but the primitive variables aren't necessarily indexed the same between PrimitiveSolver
  // and AthenaK.
  Minmod(prim(m, IDN, km1, jm1, im1), prim(m, IDN, k, j, i), prim(m, IDN, kp1, jp1, ip1),
         prim_l[PRH], prim_r[PRH]);
  Minmod(prim(m, IVX, km1, jm1, im1), prim(m, IVX, k, j, i), prim(m, IVX, kp1, jp1, ip1),
         prim_l[PVX], prim_r[PVX]);
  Minmod(prim(m, IVY, km1, jm1, im1), prim(m, IVY, k, j, i), prim(m, IVY, kp1, jp1, ip1),
         prim_l[PVY], prim_r[PVY]);
  Minmod(prim(m, IVZ, km1, jm1, im1), prim(m, IVZ, k, j, i), prim(m, IVZ, kp1, jp1, ip1),
         prim_l[PVZ], prim_r[PVZ]);
  Minmod(prim(m, IPR, km1, jm1, im1), prim(m, IPR, k, j, i), prim(m, IPR, kp1, jp1, ip1),
         prim_l[PPR], prim_r[PPR]);
  for (int s = 0; s < nscal; s++) {
    Minmod(prim(m, nhyd + s, km1, jm1, im1),
           prim(m, nhyd + s, k, j, i),
           prim(m, nhyd + s, kp1, jp1, ip1),
           prim_l[PYF + s], prim_r[PYF + s]);
  }
  // Convert the density to number density and calculate the reconstructed temperature
  prim_l[PRH] = prim_l[PRH]/mb;
  prim_r[PRH] = prim_r[PRH]/mb;
  prim_l[PTM] = eos.ps.GetEOS().GetTemperatureFromP(prim_l[PRH], prim_l[PPR], &prim_l[PYF]);
  prim_r[PTM] = eos.ps.GetEOS().GetTemperatureFromP(prim_r[PRH], prim_r[PPR], &prim_r[PYF]);
  if (prim_r[PTM] < 0 || prim_l[PTM] < 0) {
    printf("There's a problem with the temperature!\n");
  }
  return;
}

KOKKOS_INLINE_FUNCTION
void InsertFluxes(const Real flux_pt[NCONS], const DvceArray5D<Real>& flx,
                  const int m, const int k, const int j, const int i) {
  flx(m, IDN, k, j, i) = flux_pt[CDN];
  flx(m, IM1, k, j, i) = flux_pt[CSX];
  flx(m, IM2, k, j, i) = flux_pt[CSY];
  flx(m, IM3, k, j, i) = flux_pt[CSZ];
  flx(m, IEN, k, j, i) = flux_pt[CTA];
}

} // namespace dyngr

#endif
