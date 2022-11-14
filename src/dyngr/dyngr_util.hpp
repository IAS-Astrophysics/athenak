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
  // FIXME: Change to pressure later on!
  Real e = prim(m, IEN, k, j, i) + prim(m, IDN, k, j, i);
  prim_pt[PTM] = eos.ps.GetEOS().GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
  prim_pt[PPR] = eos.ps.GetEOS().GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
}

} // namespace dyngr

#endif
