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

//----------------------------------------------------------------------------------------
//! \fn void SingleStateTetradFlux
//! \brief inline function for calculating GRMHD fluxes in a tetrad frame
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateTetradFlux(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    Real prim[NPRIM], Real Bt[NPRIM],
    Real cons[NCONS],
    Real flux[NCONS], Real bflux[NMAG], Real& bsq) {
  constexpr int pvx = PVX + (ivx - IVX);
  constexpr int pvy = PVX + ((ivx - IVX) + 1)%3;
  constexpr int pvz = PVX + ((ivx - IVX) + 2)%3;

  constexpr int csx = CSX + (ivx - IVX);

  constexpr int ibx = ivx - IVX;
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;

  // Auxiliary quantities
  Real W = Kokkos::sqrt(prim[PVX]*prim[PVX] + prim[PVY]*prim[PVY] + prim[PVZ]*prim[PVZ]
                         + 1.0);
  Real iW = 1.0/W;
  Real b0 = Bt[0]*prim[PVX] + Bt[1]*prim[PVY] + Bt[2]*prim[PVZ];
  Real Bv = b0*iW;
  Real Bsq = Bt[0]*Bt[0] + Bt[1]*Bt[1] + Bt[2]*Bt[2];
  bsq = Bsq*iW*iW + Bv*Bv;
  Real rhohB = (eos.ps.GetEOS().GetEnergy(prim[PRH], prim[PTM], &prim[PYF]) +
                prim[PPR])*W*W + Bsq;

  // Compute conserved variables
  cons[CDN] = eos.ps.GetEOS().GetBaryonMass()*prim[PRH]*W;
  cons[CSX] = rhohB*prim[PVX]*iW - Bv*Bt[0];
  cons[CSY] = rhohB*prim[PVY]*iW - Bv*Bt[1];
  cons[CSZ] = rhohB*prim[PVZ]*iW - Bv*Bt[2];
  cons[CTA] = rhohB - prim[PPR] - 0.5*bsq - cons[CDN];

  // Calculate the fluxes
  Real vx = prim[pvx]*iW;
  Real bu[3] = {(Bt[0] + b0*prim[PVX])*iW,
                (Bt[1] + b0*prim[PVY])*iW,
                (Bt[2] + b0*prim[PVZ])*iW};
  flux[CDN] = cons[CDN]*vx;
  flux[CSX] = cons[CSX]*vx - bu[0]*Bt[ibx]*iW;
  flux[CSY] = cons[CSY]*vx - bu[1]*Bt[ibx]*iW;
  flux[CSZ] = cons[CSZ]*vx - bu[2]*Bt[ibx]*iW;
  flux[CTA] = (cons[CTA] + prim[PPR] + 0.5*bsq)*vx - b0*Bt[ibx]*iW;
  flux[csx] += prim[PPR] + 0.5*bsq;

  bflux[ibx] = 0.0;
  bflux[iby] = (Bt[iby]*vx - Bt[ibx]*prim[pvy]*iW);
  bflux[ibz] = (Bt[ibz]*vx - Bt[ibx]*prim[pvz]*iW);
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeOrthonormalTetrad
//! \brief Calculate orthonormal tetrad to local Minkowski spacetime. Based on
//         White et al. (1511.00943) but uses notation from Kiuchi et al. (2205.04487).
template<int ivx, class arr2D>
KOKKOS_INLINE_FUNCTION
void ComputeOrthonormalTetrad(Real g3d[NSPMETRIC], Real beta_u[3], Real alpha, Real idetg,
                              arr2D& e, arr2D& ie) {
  // Index permutations
  // 0, 1, 2 for spatial quantities
  constexpr int ix = (ivx - IVX);
  constexpr int iy = ((ivx - IVX) + 1) % 3;
  constexpr int iz = ((ivx - IVX) + 2) % 3;

  // 1, 2, 3 for space components of spacetime quantities
  constexpr int sx = ix + 1;
  constexpr int sy = iy + 1;
  constexpr int sz = iz + 1;

  // Indices for accessing metric
  constexpr int diag[3] = {S11, S22, S33};
  constexpr int nearoff[3] = {S12, S23, S13};
  constexpr int faroff[3] = {S23, S13, S12};
  constexpr int corner[3] = {S13, S12, S23};
  constexpr int s22 = diag[iy];
  constexpr int s33 = diag[iz];
  constexpr int s12 = nearoff[ix];
  constexpr int s13 = corner[ix];
  constexpr int s23 = faroff[ix];

  // Compute some necessary components of the spatial metric. "gab" is \gamma^{ab}, and
  // "g_ab" is \gamma_{ab}.
  Real g11 = (-g3d[s23]*g3d[s23] + g3d[s22]*g3d[s33])*idetg;
  Real g12 = (g3d[s13]*g3d[s23] - g3d[s12]*g3d[s33])*idetg;
  Real g13 = (g3d[s12]*g3d[s23] - g3d[s13]*g3d[s22])*idetg;

  Real g_12 = g3d[s12];
  Real g_13 = g3d[s13];
  Real g_22 = g3d[s22];
  Real g_23 = g3d[s23];
  Real g_33 = g3d[s33];

  Real beta_d[3];
  Primitive::LowerVector(beta_d, beta_u, g3d);

  // Compute utility quantities
  Real A = -1.0/alpha;
  Real B = 1.0/Kokkos::sqrt(g11);
  Real C = 1.0/Kokkos::sqrt(g_33);
  Real D = 1.0/Kokkos::sqrt(g_33*(g_22*g_33 - g_23*g_23));

  // Fill in transformation for covariant vectors. This corresponds to e^\mu_{(\nu)}.
  e[0][0] = -A;
  e[sx][0] = A*beta_u[ix];
  e[sy][0] = A*beta_u[iy];
  e[sz][0] = A*beta_u[iz];

  e[0 ][sx] = 0.0;
  e[sx][sx] = B*g11;
  e[sy][sx] = B*g12;
  e[sz][sx] = B*g13;

  e[0 ][sy] = 0.0;
  e[sx][sy] = 0.0;
  e[sy][sy] = D*g_33;
  e[sz][sy] = -D*g_23;

  e[0 ][sz] = 0.0;
  e[sx][sz] = 0.0;
  e[sy][sz] = 0.0;
  e[sz][sz] = C;

  // Fill in transformation for contravariant vectors. This corresponds to e_{(\mu)}_\nu
  ie[0][0] = -alpha;
  ie[0][1] = 0.0;
  ie[0][2] = 0.0;
  ie[0][3] = 0.0;

  ie[sx][0 ] = B*beta_u[ix];
  ie[sx][sx] = B;
  ie[sx][sy] = 0.0;
  ie[sx][sz] = 0.0;

  ie[sy][0 ] = D*(beta_d[iy]*g_33 - beta_d[iz]*g_23);
  ie[sy][sx] = D*(g_12*g_33 - g_13*g_23);
  ie[sy][sy] = D*(g_22*g_33 - g_23*g_23);
  ie[sy][sz] = 0.0;

  ie[sz][0 ] = C*beta_d[iz];
  ie[sz][sx] = C*g_13;
  ie[sz][sy] = C*g_23;
  ie[sz][sz] = C*g_33;
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_FLUX_DYN_GRMHD_HPP_
