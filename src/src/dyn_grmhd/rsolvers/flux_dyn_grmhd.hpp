#ifndef DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file flux_dyngrmhd.hpp
//! \brief Calculate left and right fluxes for a central scheme in GRMHD
#include <stdio.h>
#include <math.h>

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateFlux
//! \brief inline function for calculating GRMHD fluxes

template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateFlux(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    Real prim_l[NPRIM], Real prim_r[NPRIM], Real Bu_l[NPRIM], Real Bu_r[NPRIM],
    const int nmhd, const int nscal,
    Real g3d[NSPMETRIC], Real beta_u[3], Real alpha,
    Real cons_l[NCONS], Real cons_r[NCONS],
    Real flux_l[NCONS], Real flux_r[NCONS], Real bflux_l[NMAG], Real bflux_r[NMAG],
    Real& bsql, Real& bsqr) {
  constexpr int pvx = PVX + (ivx - IVX);
  constexpr int pvy = PVX + ((ivx - IVX) + 1)%3;
  constexpr int pvz = PVX + ((ivx - IVX) + 2)%3;

  constexpr int csx = CSX + (ivx - IVX);

  constexpr int ibx = ivx - IVX;
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;

  const Real ialpha = 1.0/alpha;

  // Calculate conserved variables
  eos.ps.PrimToCon(prim_l, cons_l, Bu_l, g3d);
  eos.ps.PrimToCon(prim_r, cons_r, Bu_r, g3d);

  // Calculate W for the left state.
  Real uul[3] = {prim_l[IVX], prim_l[IVY], prim_l[IVZ]};
  Real udl[3];
  Primitive::LowerVector(udl, uul, g3d);
  Real iWsql = 1.0/(1.0 + Primitive::Contract(uul, udl));
  Real iWl = sqrt(iWsql);
  Real vcl = prim_l[pvx]*iWl - beta_u[ivx-IVX]*ialpha;

  // Calculate 4-magnetic field for the left state.
  Real bul0 = Primitive::Contract(Bu_l, udl)*ialpha;
  Real bdl[3], Bd_l[3];
  Primitive::LowerVector(Bd_l, Bu_l, g3d);
  for (int a = 0; a < 3; a++) {
    bdl[a] = (alpha*bul0*udl[a] + Bd_l[a])*iWl;
  }
  bsql = (Primitive::SquareVector(Bu_l, g3d) + SQR(alpha*bul0))*iWsql;

  // Calculate fluxes for the left state.
  flux_l[CDN] = cons_l[CDN]*vcl;
  flux_l[CSX] = (cons_l[CSX]*vcl - bdl[0]*Bu_l[ibx]*iWl);
  flux_l[CSY] = (cons_l[CSY]*vcl - bdl[1]*Bu_l[ibx]*iWl);
  flux_l[CSZ] = (cons_l[CSZ]*vcl - bdl[2]*Bu_l[ibx]*iWl);
  flux_l[csx] += (prim_l[PPR] + 0.5*bsql);
  flux_l[CTA] = (cons_l[CTA]*vcl - alpha*bul0*Bu_l[ibx]*iWl
          + (prim_l[PPR] + 0.5*bsql)*prim_l[ivx]*iWl);

  bflux_l[ibx] = 0.0;
  bflux_l[iby] = (Bu_l[iby]*vcl -
                    Bu_l[ibx]*(prim_l[pvy]*iWl - beta_u[pvy - PVX]*ialpha));
  bflux_l[ibz] = (Bu_l[ibz]*vcl -
                    Bu_l[ibx]*(prim_l[pvz]*iWl - beta_u[pvz - PVX]*ialpha));

  // Calculate W for the right state.
  Real uur[3] = {prim_r[IVX], prim_r[IVY], prim_r[IVZ]};
  Real udr[3];
  Primitive::LowerVector(udr, uur, g3d);
  Real iWsqr = 1.0/(1.0 + Primitive::Contract(uur, udr));
  Real iWr = sqrt(iWsqr);
  Real vcr = prim_r[pvx]*iWr - beta_u[ivx-IVX]*ialpha;

  // Calculate 4-magnetic field for the right state.
  Real bur0 = Primitive::Contract(Bu_r, udr)*ialpha;
  Real bdr[3], Bd_r[3];
  Primitive::LowerVector(Bd_r, Bu_r, g3d);
  for (int a = 0; a < 3; a++) {
    bdr[a] = (alpha*bur0*udr[a] + Bd_r[a])*iWr;
  }
  bsqr = (Primitive::SquareVector(Bu_r, g3d) + SQR(alpha*bur0))*iWsqr;

  // Calculate fluxes for the right state.
  flux_r[CDN] = cons_r[CDN]*vcr;
  flux_r[CSX] = (cons_r[CSX]*vcr - bdr[0]*Bu_r[ibx]*iWr);
  flux_r[CSY] = (cons_r[CSY]*vcr - bdr[1]*Bu_r[ibx]*iWr);
  flux_r[CSZ] = (cons_r[CSZ]*vcr - bdr[2]*Bu_r[ibx]*iWr);
  flux_r[csx] += (prim_r[PPR] + 0.5*bsqr);
  flux_r[CTA] = (cons_r[CTA]*vcr - alpha*bur0*Bu_r[ibx]*iWr
          + (prim_r[PPR] + 0.5*bsqr)*prim_r[ivx]*iWr);

  bflux_r[ibx] = 0.0;
  bflux_r[iby] = (Bu_r[iby]*vcr -
                    Bu_r[ibx]*(prim_r[pvy]*iWr - beta_u[pvy - PVX]*ialpha));
  bflux_r[ibz] = (Bu_r[ibz]*vcr -
                    Bu_r[ibx]*(prim_r[pvz]*iWr - beta_u[pvz - PVX]*ialpha));
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
