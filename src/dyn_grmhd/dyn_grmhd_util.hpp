#ifndef DYN_GRMHD_DYN_GRMHD_UTIL_HPP_
#define DYN_GRMHD_DYN_GRMHD_UTIL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_util.hpp
//  \brief Utility functions for use with dynamic hydro

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "reconstruct/plm.hpp"

namespace dyngr {

template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void ExtractPrimitives(Real prim_pt[NPRIM], const DvceArray5D<Real>& prim,
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
  prim_pt[PTM] = eos.ps.GetEOS().GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR],
                                                     &prim_pt[PYF]);
}

KOKKOS_INLINE_FUNCTION
void ExtractBField(Real bu_pt[NMAG], const DvceArray5D<Real> bcc,
                   int ibx, int iby, int ibz,
                   const int m, const int k, const int j, const int i) {
  bu_pt[iby] = bcc(m, iby, k, j, i);
  bu_pt[ibz] = bcc(m, ibz, k, j, i);
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

#endif  // DYN_GRMHD_DYN_GRMHD_UTIL_HPP_
