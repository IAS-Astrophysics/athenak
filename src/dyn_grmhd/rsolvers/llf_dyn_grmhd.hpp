#ifndef DYN_GRMHD_RSOLVERS_LLF_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_LLF_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_dyngrmhd.hpp
//! \brief LLF Riemann solver for general relativistic magnetohydrodynamics

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyn_grmhd.hpp"

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
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;

  constexpr int diag[3] = {S11, S22, S33};
  constexpr int offdiag[3] = {S23, S13, S12};
  constexpr int offidx = offdiag[ivx - IVX];
  constexpr int idxy = diag[(ivx - IVX + 1) % 3];
  constexpr int idxz = diag[(ivx - IVX + 2) % 3];

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
  Real gii = (g3d[idxy]*g3d[idxz] - g3d[offidx]*g3d[offidx])*(isdetg*isdetg);
  eos.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                  g3d, beta_u, alpha, gii, pvx);
  eos.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                  g3d, beta_u, alpha, gii, pvx);

  // Get the extremal wavespeeds
  Real lambda_l = fmin(lambda_ml, lambda_mr);
  Real lambda_r = fmax(lambda_pl, lambda_pr);
  Real lambda = fmax(lambda_r, -lambda_l);

  //Real vol = sdetg*alpha;

  // Calculate the fluxes
  flux[CDN] = 0.5*sdetg*(alpha*(fl[CDN] + fr[CDN]) - lambda*(cons_r[CDN] - cons_l[CDN]));
  flux[CSX] = 0.5*sdetg*(alpha*(fl[CSX] + fr[CSX]) - lambda*(cons_r[CSX] - cons_l[CSX]));
  flux[CSY] = 0.5*sdetg*(alpha*(fl[CSY] + fr[CSY]) - lambda*(cons_r[CSY] - cons_l[CSY]));
  flux[CSZ] = 0.5*sdetg*(alpha*(fl[CSZ] + fr[CSZ]) - lambda*(cons_r[CSZ] - cons_l[CSZ]));
  flux[CTA] = 0.5*sdetg*(alpha*(fl[CTA] + fr[CTA]) - lambda*(cons_r[CTA] - cons_l[CTA]));

  bflux[IBY] = - 0.5 * sdetg *
               (alpha*(bfl[iby] + bfr[iby]) - lambda * (Bu_rund[iby] - Bu_lund[iby]));
  bflux[IBZ] = 0.5 * sdetg *
               (alpha*(bfl[ibz] + bfr[ibz]) - lambda * (Bu_rund[ibz] - Bu_lund[ibz]));
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
    constexpr int offdiag[3] = {S23, S13, S12};
    constexpr int offidx = offdiag[ivx - IVX];
    constexpr int idxy = diag[(ivx - IVX + 1) % 3];
    constexpr int idxz = diag[(ivx - IVX + 2) % 3];

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
    eos.ps.GetEOS().ApplyDensityLimits(prim_l[PRH]);
    eos.ps.GetEOS().ApplySpeciesLimits(&prim_l[PYF]);
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
    eos.ps.GetEOS().ApplyDensityLimits(prim_r[PRH]);
    eos.ps.GetEOS().ApplySpeciesLimits(&prim_r[PYF]);
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
    Real gii = (g3d[idxy]*g3d[idxz] - g3d[offidx]*g3d[offidx])*(isdetg*isdetg);
    eos.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                    g3d, beta_u, alpha, gii, pvx);
    eos.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                    g3d, beta_u, alpha, gii, pvx);

    // Get the extremal wavespeeds
    Real lambda_l = fmin(lambda_ml, lambda_mr);
    Real lambda_r = fmax(lambda_pl, lambda_pr);
    Real lambda = fmax(lambda_r, -lambda_l);

    //Real vol = sdetg*alpha;

    // Calculate the fluxes
    flx(m, IDN, k, j, i) = 0.5 * sdetg * (alpha*(fl[CDN] + fr[CDN]) -
                                  lambda * (cons_r[CDN] - cons_l[CDN]));
    flx(m, IEN, k, j, i) = 0.5 * sdetg * (alpha*(fl[CTA] + fr[CTA]) -
                                  lambda * (cons_r[CTA] - cons_l[CTA]));
    flx(m, IVX, k, j, i) = 0.5 * sdetg * (alpha*(fl[CSX] + fr[CSX]) -
                                  lambda * (cons_r[CSX] - cons_l[CSX]));
    flx(m, IVY, k, j, i) = 0.5 * sdetg * (alpha*(fl[CSY] + fr[CSY]) -
                                  lambda * (cons_r[CSY] - cons_l[CSY]));
    flx(m, IVZ, k, j, i) = 0.5 * sdetg * (alpha*(fl[CSZ] + fr[CSZ]) -
                                  lambda * (cons_r[CSZ] - cons_l[CSZ]));
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = -0.5*sdetg*(alpha*(bfl[iby] + bfr[iby]) -
                          lambda * (Bu_r[iby] - Bu_l[iby]));
    ez(m, k, j, i) = 0.5*sdetg*(alpha*(bfl[ibz] + bfr[ibz]) -
                          lambda * (Bu_r[ibz] - Bu_l[ibz]));
  });
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_LLF_DYN_GRMHD_HPP_
