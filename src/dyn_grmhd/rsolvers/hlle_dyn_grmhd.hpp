#ifndef DYN_GRMHD_RSOLVERS_HLLE_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_HLLE_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_dyngrmhd.hpp
//! \brief HLLE Riemann solver for general relativistic magnetohydrodynamics

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyn_grmhd.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateHLLE_DYNGR
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateHLLE_DYNGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    Real prim_l[NPRIM], Real prim_r[NPRIM], Real Bu_l[NPRIM], Real Bu_r[NPRIM],
    const int nmhd, const int nscal,
    Real g3d[NSPMETRIC], Real beta_u[3], Real alpha,
    Real flux[NCONS], Real bflux[NMAG]) {
  constexpr int ibx = ivx - IVX;
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

  // Calculate fluxes in HLL region
  Real qa = lambda_r*lambda_l/alpha;
  Real qb = 1.0/(lambda_r - lambda_l);
  Real f_hll[NCONS], bf_hll[NMAG];
  f_hll[CDN] = (lambda_r*fl[CDN] - lambda_l*fr[CDN] +
                qa*(cons_r[CDN] - cons_l[CDN])) * qb;
  f_hll[CSX] = (lambda_r*fl[CSX] - lambda_l*fr[CSX] +
                qa*(cons_r[CSX] - cons_l[CSX])) * qb;
  f_hll[CSY] = (lambda_r*fl[CSY] - lambda_l*fr[CSY] +
                qa*(cons_r[CSY] - cons_l[CSY])) * qb;
  f_hll[CSZ] = (lambda_r*fl[CSZ] - lambda_l*fr[CSZ] +
                qa*(cons_r[CSZ] - cons_l[CSZ])) * qb;
  f_hll[CTA] = (lambda_r*fl[CTA] - lambda_l*fr[CTA] +
                qa*(cons_r[CTA] - cons_l[CTA])) * qb;
  bf_hll[ibx] = 0.0;
  bf_hll[iby] = (lambda_r*bfl[iby] - lambda_l*bfr[iby] +
                 qa*(Bu_r[iby] - Bu_l[iby])) * qb;
  bf_hll[ibz] = (lambda_r*bfl[ibz] - lambda_l*bfr[ibz] +
                 qa*(Bu_r[ibz] - Bu_l[ibz])) * qb;

  Real *f_interface, *bf_interface;
  if (lambda_l >= 0.) {
    f_interface = &fl[0];
    bf_interface = &bfl[0];
  } else if (lambda_r <= 0.) {
    f_interface = &fr[0];
    bf_interface = &bfr[0];
  } else {
    f_interface = &f_hll[0];
    bf_interface = &bf_hll[0];
  }

  Real vol = sdetg*alpha;

  // Calculate the fluxes
  flux[CDN] = vol * f_interface[CDN];
  flux[CSX] = vol * f_interface[CSX];
  flux[CSY] = vol * f_interface[CSY];
  flux[CSZ] = vol * f_interface[CSZ];
  flux[CTA] = vol * f_interface[CTA];

  bflux[IBY] = - vol * bf_interface[iby];
  bflux[IBZ] = vol * bf_interface[ibz];
}

//----------------------------------------------------------------------------------------
//! \fn void HLLE_DYNGR
//! \brief inline function for calculating GRMHD fluxes via HLLE
//----------------------------------------------------------------------------------------
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void HLLE_DYNGR(TeamMember_t const &member,
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

    // Calculate fluxes in HLL region
    Real qa = lambda_r*lambda_l/alpha;
    Real qb = 1.0/(lambda_r - lambda_l);
    Real f_hll[NCONS], bf_hll[NMAG];
    f_hll[CDN] = ((lambda_r*fl[CDN] - lambda_l*fr[CDN]) +
                  qa*(cons_r[CDN] - cons_l[CDN])) * qb;
    f_hll[CSX] = ((lambda_r*fl[CSX] - lambda_l*fr[CSX]) +
                  qa*(cons_r[CSX] - cons_l[CSX])) * qb;
    f_hll[CSY] = ((lambda_r*fl[CSY] - lambda_l*fr[CSY]) +
                  qa*(cons_r[CSY] - cons_l[CSY])) * qb;
    f_hll[CSZ] = ((lambda_r*fl[CSZ] - lambda_l*fr[CSZ]) +
                  qa*(cons_r[CSZ] - cons_l[CSZ])) * qb;
    f_hll[CTA] = ((lambda_r*fl[CTA] - lambda_l*fr[CTA]) +
                  qa*(cons_r[CTA] - cons_l[CTA])) * qb;
    bf_hll[ibx] = 0.0;
    bf_hll[iby] = ((lambda_r*bfl[iby] - lambda_l*bfr[iby]) +
                   qa*(Bu_r[iby] - Bu_l[iby])) * qb;
    bf_hll[ibz] = ((lambda_r*bfl[ibz] - lambda_l*bfr[ibz]) +
                   qa*(Bu_r[ibz] - Bu_l[ibz])) * qb;

    Real *f_interface, *bf_interface;
    if (lambda_l >= 0.) {
      f_interface = &fl[0];
      bf_interface = &bfl[0];
    } else if (lambda_r <= 0.) {
      f_interface = &fr[0];
      bf_interface = &bfr[0];
    } else {
      f_interface = &f_hll[0];
      bf_interface = &bf_hll[0];
    }

    Real vol = sdetg*alpha;

    // Calculate the fluxes
    flx(m, IDN, k, j, i) = vol * f_interface[CDN];
    flx(m, IEN, k, j, i) = vol * f_interface[CTA];
    flx(m, IVX, k, j, i) = vol * f_interface[CSX];
    flx(m, IVY, k, j, i) = vol * f_interface[CSY];
    flx(m, IVZ, k, j, i) = vol * f_interface[CSZ];
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = -vol * bf_interface[iby];
    ez(m, k, j, i) = vol * bf_interface[ibz];
  });
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_HLLE_DYN_GRMHD_HPP_
