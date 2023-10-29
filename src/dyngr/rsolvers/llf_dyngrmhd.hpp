#ifndef DYNGR_RSOLVERS_LLF_DYNGRMHD_HPP_
#define DYNGR_RSOLVERS_LLF_DYNGRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_dyngrmhd.hpp
//! \brief LLF Riemann solver for general relativistic magnetohydrodynamics

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "adm/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyngrmhd.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_DYNGR
//! \brief inline function for calculating GRMHD fluxes via Lax-Friedrichs
//! TODO: This could potentially be sped up by calculating the conserved variables without
//  the help of PrimitiveSolver; there are redundant calculations with B^i v_i and W that
//  may not be needed.
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateLLF_DYNGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    Real prim_l[NPRIM], Real prim_r[NPRIM], Real Bu_l[NPRIM], Real Bu_r[NPRIM],
    const int nmhd, const int nscal,
    Real g3d[NSPMETRIC], Real beta_u[3], Real alpha,
    Real flux[NCONS], Real bflux[NMAG]) {
  constexpr int ibx = ivx - IVX;
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;

  constexpr int diag[3] = {S11, S22, S33};
  constexpr int idx = diag[ivx - IVX];

  constexpr int pvx = PVX + (ivx - IVX);

  Real sdetg = sqrt(Primitive::GetDeterminant(g3d));
  Real isdetg = 1.0/sdetg;

  // Undensitize the magnetic field before calculating the conserved variables
  Real Bu_lund[NMAG], Bu_rund[NMAG];
  for (int n = 0; n < NMAG; n++) {
    Bu_lund[n] = Bu_l[n]*isdetg;
    Bu_rund[n] = Bu_r[n]*isdetg;
  }

  // Calculate the left and right fluxes
  Real cons_l[NCONS], cons_r[NCONS];
  Real fl[NCONS], fr[NCONS], bfl[NMAG], bfr[NMAG];
  Real bsql, bsqr;
  SingleStateFlux<ivx>(eos, prim_l, prim_r, Bu_lund, Bu_rund, nmhd, nscal, g3d, beta_u,
                       alpha, cons_l, cons_r, fl, fr, bfl, bfr, bsql, bsqr);
  

  // Calculate the magnetosonic speeds for both states
  Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
  eos.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                  g3d, beta_u, alpha, g3d[idx], pvx);
  eos.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                  g3d, beta_u, alpha, g3d[idx], pvx);

  // Get the extremal wavespeeds
  Real lambda_l = fmin(lambda_ml, lambda_mr);
  Real lambda_r = fmax(lambda_pl, lambda_pr);
  Real lambda = fmax(lambda_r, -lambda_l);

  Real vol = sdetg*alpha;

  // Calculate the fluxes
  flux[CDN] = 0.5*(fl[CDN] + fr[CDN] - lambda*(cons_r[CDN] - cons_l[CDN]));
  flux[CSX] = 0.5*(fl[CSX] + fr[CSX] - lambda*(cons_r[CSX] - cons_l[CSX]));
  flux[CSY] = 0.5*(fl[CSY] + fr[CSY] - lambda*(cons_r[CSY] - cons_l[CSY]));
  flux[CSZ] = 0.5*(fl[CSZ] + fr[CSZ] - lambda*(cons_r[CSZ] - cons_l[CSZ]));
  flux[CTA] = 0.5*(fl[CTA] + fr[CTA] - lambda*(cons_r[CTA] - cons_l[CTA]));

  bflux[IBY] = - 0.5 * vol *
               (bfl[iby] + bfr[iby] - lambda * (Bu_rund[iby] - Bu_lund[iby]));
  bflux[IBZ] = 0.5 * vol *
               (bfl[ibz] + bfr[ibz] - lambda * (Bu_rund[ibz] - Bu_lund[ibz]));
}

//----------------------------------------------------------------------------------------
//! \fn void LLF_DYNGR
//! \brief inline function for calculating GRMHD fluxes via Lax-Friedrichs
//! TODO: This could potentially be sped up by calculating the conserved variables without
//  the help of PrimitiveSolver; there are redundant calculations with B^i v_i and W that
//  may not be needed.
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void LLF_DYNGR(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const int& nhyd, const int& nscal,
     const adm::ADM::ADM_vars& adm,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  par_for_inner(member, il, iu, [&](const int i) {
    constexpr int ibx = ivx - IVX;
    constexpr int iby = ((ivx - IVX) + 1)%3;
    constexpr int ibz = ((ivx - IVX) + 2)%3;

    constexpr int diag[3] = {S11, S22, S33};
    constexpr int idx = diag[ivx - IVX];

    constexpr int pvx = PVX + (ivx - IVX);

    Real g3d[NSPMETRIC];
    Real beta_u[3];
    Real alpha;
    if constexpr (ivx == IVX) {
      adm::Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVY) {
      adm::Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVZ) {
      adm::Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    }

    Real sdetg = sqrt(Primitive::GetDeterminant(g3d));
    Real isdetg = 1.0/sdetg;

    // Extract left and right primitives
    Real prim_l[NPRIM], prim_r[NPRIM];
    Real Bu_l[NMAG], Bu_r[NMAG];
    Real mb = eos.ps.GetEOS().GetBaryonMass();

    prim_l[PRH] = wl(IDN, i)/mb;
    prim_l[PVX] = wl(IVX, i);
    prim_l[PVY] = wl(IVY, i);
    prim_l[PVZ] = wl(IVZ, i);
    for (int n = 0; n < nscal; n++) {
      prim_l[PYF + n] = wl(nhyd + n, i);
    }
    prim_l[PPR] = wl(IPR, i);
    prim_l[PTM] = eos.ps.GetEOS().GetTemperatureFromP(
                  prim_l[PRH], prim_l[PPR], &prim_l[PYF]);
    Bu_l[ibx] = bx(m, k, j, i)*isdetg;
    Bu_l[iby] = bl(iby, i)*isdetg;
    Bu_l[ibz] = bl(ibz, i)*isdetg;

    prim_r[PRH] = wr(IDN, i)/mb;
    prim_r[PVX] = wr(IVX, i);
    prim_r[PVY] = wr(IVY, i);
    prim_r[PVZ] = wr(IVZ, i);
    for (int n = 0; n < nscal; n++) {
      prim_r[PYF + n] = wr(nhyd + n, i);
    }
    prim_r[PPR] = wr(IPR, i);
    prim_r[PTM] = eos.ps.GetEOS().GetTemperatureFromP(
                  prim_r[PRH], prim_r[PPR], &prim_r[PYF]);
    Bu_r[ibx] = bx(m, k, j, i)*isdetg;
    Bu_r[iby] = br(iby, i)*isdetg;
    Bu_r[ibz] = br(ibz, i)*isdetg;

    // Apply floors to make sure these values are physical.
    eos.ps.GetEOS().ApplyPrimitiveFloor(prim_l[PRH], &prim_l[PVX], prim_l[PPR],
                                    prim_l[PTM], &prim_l[PYF]);
    eos.ps.GetEOS().ApplyPrimitiveFloor(prim_r[PRH], &prim_r[PVX], prim_r[PPR],
                                    prim_r[PTM], &prim_r[PYF]);
    
    // Calculate the left and right fluxes
    Real cons_l[NCONS], cons_r[NCONS];
    Real fl[NCONS], fr[NCONS], bfl[NMAG], bfr[NMAG];
    Real bsql, bsqr;
    SingleStateFlux<ivx>(eos, prim_l, prim_r, Bu_l, Bu_r, nhyd, nscal, g3d, beta_u, alpha,
                         cons_l, cons_r, fl, fr, bfl, bfr, bsql, bsqr);

    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    eos.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                    g3d, beta_u, alpha, g3d[idx], pvx);
    eos.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                    g3d, beta_u, alpha, g3d[idx], pvx);

    // Get the extremal wavespeeds
    Real lambda_l = fmin(lambda_ml, lambda_mr);
    Real lambda_r = fmax(lambda_pl, lambda_pr);
    Real lambda = fmax(lambda_r, -lambda_l);

    Real vol = sdetg*alpha;

    // Calculate the fluxes
    flx(m, IDN, k, j, i) = 0.5 * vol * (fl[CDN] + fr[CDN] -
                                  lambda * (cons_r[CDN] - cons_l[CDN]));
    flx(m, IEN, k, j, i) = 0.5 * vol * (fl[CTA] + fr[CTA] -
                                  lambda * (cons_r[CTA] - cons_l[CTA]));
    flx(m, IVX, k, j, i) = 0.5 * vol * (fl[CSX] + fr[CSX] -
                                  lambda * (cons_r[CSX] - cons_l[CSX]));
    flx(m, IVY, k, j, i) = 0.5 * vol * (fl[CSY] + fr[CSY] -
                                  lambda * (cons_r[CSY] - cons_l[CSY]));
    flx(m, IVZ, k, j, i) = 0.5 * vol * (fl[CSZ] + fr[CSZ] -
                                  lambda * (cons_r[CSZ] - cons_l[CSZ]));
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = - 0.5 * vol * (bfl[iby] + bfr[iby] - lambda * (Bu_r[iby] - Bu_l[iby]));
    ez(m, k, j, i) = 0.5 * vol * (bfl[ibz] + bfr[ibz] - lambda * (Bu_r[ibz] - Bu_l[ibz]));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void LLF_DYNGR
//! \brief inline function for calculating GRMHD fluxes via Lax-Friedrichs
//! TODO: This could potentially be sped up by calculating the conserved variables without
//  the help of PrimitiveSolver; there are redundant calculations with B^i v_i and W that
//  may not be needed.
/*template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void LLF_DYNGR(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const int& nhyd, const int& nscal,
     const adm::ADM::ADM_vars& adm,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  par_for_inner(member, il, iu, [&](const int i) {
    // Cyclic permutation of array indices
    constexpr int diag[3] = {S11, S22, S33};

    constexpr int pvx = PVX + (ivx - IVX);
    constexpr int pvy = PVX + ((ivx - IVX) + 1)%3;
    constexpr int pvz = PVX + ((ivx - IVX) + 2)%3;

    constexpr int csx = CSX + (ivx - IVX);

    constexpr int idx = diag[ivx - IVX];

    constexpr int ibx = ivx - IVX;
    constexpr int iby = ((ivx - IVX) + 1)%3;
    constexpr int ibz = ((ivx - IVX) + 2)%3;

    constexpr int imap[3][3] = {{S11, S12, S13}, {S12, S22, S23}, {S13, S23, S33}};
    // Extract metric components
    Real g3d[NSPMETRIC];
    Real beta_u[3];
    Real alpha;
    if constexpr (ivx == IVX) {
      adm::Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVY) {
      adm::Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVZ) {
      adm::Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } 
    Real ialpha = 1.0/alpha;
    Real detg = Primitive::GetDeterminant(g3d);
    Real sdetg = sqrt(detg);
    Real isdetg = 1.0/sdetg;

    // Extract left primitives and calculate left conserved variables.
    Real prim_l[NPRIM], cons_l[NCONS], Bu_l[NMAG];
    eos.PrimToConsPt(wl, bl, bx, prim_l, cons_l, Bu_l,
                      g3d, sdetg, m, k, j, i, nhyd, nscal, ibx, iby, ibz);

    // Extract right primitives and calculate right conserved variables.
    Real prim_r[NPRIM], cons_r[NCONS], Bu_r[NMAG];
    eos.PrimToConsPt(wr, br, bx, prim_r, cons_r, Bu_r,
                      g3d, sdetg, m, k, j, i, nhyd, nscal, ibx, iby, ibz);

    // Calculate W for the left state.
    Real uul[3] = {prim_l[IVX], prim_l[IVY], prim_l[IVZ]};
    Real udl[3];
    Primitive::LowerVector(udl, uul, g3d);
    Real iWsql = 1./(1.0 + Primitive::Contract(uul, udl));
    Real iWl = sqrt(iWsql);
    Real vcl = prim_l[pvx]*iWl - beta_u[ivx-IVX]*ialpha;

    // Calculate 4-magnetic field (undensitized) for the left state.
    Real bul0 = Primitive::Contract(Bu_l, udl)*(ialpha*isdetg);
    Real bdl[3];
    Real Bd_l[3];
    Primitive::LowerVector(Bd_l, Bu_l, g3d);
    for (int a = 0; a < 3; a++) {
      bdl[a] = (Bd_l[a]*isdetg + alpha*bul0*udl[a])*iWl;
    }
    Real bsql = (Primitive::SquareVector(Bu_l, g3d)*(isdetg*isdetg) + 
                  SQR(alpha*bul0))*iWsql;

    // Calculate W for the right state.
    Real uur[3] = {prim_r[IVX], prim_r[IVY], prim_r[IVZ]};
    Real udr[3];
    Primitive::LowerVector(udr, uur, g3d);
    Real iWsqr = 1.0/(1.0 + Primitive::Contract(uur, udr));
    Real iWr = sqrt(iWsqr);
    Real vcr = prim_r[pvx]*iWr - beta_u[ivx-IVX]*ialpha;

    // Calculate 4-magnetic field (densitized) for the right state.
    Real bur0 = Primitive::Contract(Bu_r, udr)*(ialpha*isdetg);
    Real bdr[3];
    Real Bd_r[3];
    Primitive::LowerVector(Bd_r, Bu_r, g3d);
    for (int a = 0; a < 3; a++) {
      bdr[a] = (Bd_r[a]*isdetg + alpha*bur0*udr[a])*iWr;
    }
    Real bsqr = (Primitive::SquareVector(Bu_r, g3d)*(isdetg*isdetg) + 
                  SQR(alpha*bur0))*iWsqr;

    // Calculate fluxes for the left state.
    Real fl[NCONS], bfl[NMAG];
    fl[CDN] = alpha*cons_l[CDN]*vcl;
    fl[CSX] = alpha*(cons_l[CSX]*vcl - bdl[0]*Bu_l[ibx]*iWl);
    fl[CSY] = alpha*(cons_l[CSY]*vcl - bdl[1]*Bu_l[ibx]*iWl);
    fl[CSZ] = alpha*(cons_l[CSZ]*vcl - bdl[2]*Bu_l[ibx]*iWl);
    fl[csx] += alpha*sdetg*(prim_l[PPR] + 0.5*bsql);
    fl[CTA] = alpha*(cons_l[CTA]*vcl - alpha*bul0*Bu_l[ibx]*iWl
            + sdetg*(prim_l[PPR] + 0.5*bsql)*prim_l[ivx]*iWl);

    bfl[ibx] = 0.0;
    bfl[iby] = alpha*(Bu_l[iby]*vcl -
                      Bu_l[ibx]*(prim_l[pvy]*iWl - beta_u[pvy - PVX]*ialpha));
    bfl[ibz] = alpha*(Bu_l[ibz]*vcl -
                      Bu_l[ibx]*(prim_l[pvz]*iWl - beta_u[pvz - PVX]*ialpha));

    // Calculate fluxes for the right state.
    Real fr[NCONS], bfr[NMAG];
    fr[CDN] = alpha*cons_r[CDN]*vcr;
    fr[CSX] = alpha*(cons_r[CSX]*vcr - bdr[0]*Bu_r[ibx]*iWr);
    fr[CSY] = alpha*(cons_r[CSY]*vcr - bdr[1]*Bu_r[ibx]*iWr);
    fr[CSZ] = alpha*(cons_r[CSZ]*vcr - bdr[2]*Bu_r[ibx]*iWr);
    fr[csx] += alpha*sdetg*(prim_r[PPR] + 0.5*bsqr);
    fr[CTA] = alpha*(cons_r[CTA]*vcr - alpha*bur0*Bu_r[ibx]*iWr
            + sdetg*(prim_r[PPR] + 0.5*bsqr)*prim_r[ivx]*iWr);

    bfr[ibx] = 0.0;
    bfr[iby] = alpha*(Bu_r[iby]*vcr -
                      Bu_r[ibx]*(prim_r[pvy]*iWr - beta_u[pvy - PVX]*ialpha));
    bfr[ibz] = alpha*(Bu_r[ibz]*vcr -
                      Bu_r[ibx]*(prim_r[pvz]*iWr - beta_u[pvz - PVX]*ialpha));


    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    eos.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                    g3d, beta_u, alpha, g3d[idx], pvx);
    eos.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                    g3d, beta_u, alpha, g3d[idx], pvx);

    // Get the extremal wavespeeds
    Real lambda_l = fmin(lambda_ml, lambda_mr);
    Real lambda_r = fmax(lambda_pl, lambda_pr);
    Real lambda = fmax(lambda_r, -lambda_l);
    //Real lambda = 1.0;

    // Calculate the fluxes
    flx(m, IDN, k, j, i) = 0.5 * (fl[CDN] + fr[CDN] -
                                  lambda * (cons_r[CDN] - cons_l[CDN]));
    flx(m, IEN, k, j, i) = 0.5 * (fl[CTA] + fr[CTA] -
                                  lambda * (cons_r[CTA] - cons_l[CTA]));
    flx(m, IVX, k, j, i) = 0.5 * (fl[CSX] + fr[CSX] -
                                  lambda * (cons_r[CSX] - cons_l[CSX]));
    flx(m, IVY, k, j, i) = 0.5 * (fl[CSY] + fr[CSY] -
                                  lambda * (cons_r[CSY] - cons_l[CSY]));
    flx(m, IVZ, k, j, i) = 0.5 * (fl[CSZ] + fr[CSZ] -
                                  lambda * (cons_r[CSZ] - cons_l[CSZ]));
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = - 0.5 * (bfl[iby] + bfr[iby] - lambda * (Bu_r[iby] - Bu_l[iby]));
    ez(m, k, j, i) = 0.5 * (bfl[ibz] + bfr[ibz] - lambda * (Bu_r[ibz] - Bu_l[ibz]));
  });
}*/

} // namespace dyngr

#endif  // DYNGR_RSOLVERS_LLF_DYNGRMHD_HPP_
