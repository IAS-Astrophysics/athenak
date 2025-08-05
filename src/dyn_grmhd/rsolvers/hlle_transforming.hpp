#ifndef DYN_GRMHD_RSOLVERS_HLLE_TRANSFORMING_HPP_
#define DYN_GRMHD_RSOLVERS_HLLE_TRANSFORMING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_dyngrmhd.hpp
//! \brief HLLE Riemann solver for GRMHD that first transforms to local flat space

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyn_grmhd.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void HLLE_TRANSFORMING
//! \brief inline function for calculating GRMHD fluxes via HLLE with frame transform
//----------------------------------------------------------------------------------------
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void HLLE_TRANSFORMING(TeamMember_t const &member,
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
    constexpr int ivy = IVX + ((ivx - IVX) + 1)%3;
    constexpr int ivz = IVX + ((ivx - IVX) + 2)%3;

    constexpr int ibx = ivx - IVX;
    constexpr int iby = ((ivx - IVX) + 1)%3;
    constexpr int ibz = ((ivx - IVX) + 2)%3;

    constexpr int diag[3] = {S11, S22, S33};
    constexpr int offdiag[3] = {S23, S13, S12};
    constexpr int offidx = offdiag[ivx - IVX];
    constexpr int idxy = diag[(ivx - IVX + 1) % 3];
    constexpr int idxz = diag[(ivx - IVX + 2) % 3];

    constexpr int pvx = PVX + (ivx - IVX);
    constexpr int csx = CSX + (ivx - IVX);
    constexpr int csy = CSX + ((ivx - IVX) + 1) % 3;
    constexpr int csz = CSX + ((ivx - IVX) + 2) % 3;

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

    // Compute tetrad transformation
    Real e_cov[4][4], e_cont[4][4];
    ComputeOrthonormalTetrad<ivx>(g3d, beta_u, alpha, isdetg*isdetg, e_cov, e_cont);
    Real ialpha = e_cov[0][0]; // Save a division this way

    // Transform the velocity to the tetrad frame -- this is a purely spatial vector with
    // a null time component, so the raised and lowered forms are identical in the tetrad
    // frame. One could do this from the true four-velocity, but it's more expensive and
    // could lead to increased round-off error or catastrophic cancellation.
    Real ult[3] = {0.0};
    Real urt[3] = {0.0};
    for (int b = 0; b < 3; b++) {
      for (int a = 0; a < 3; a++) {
        ult[b] += e_cont[b+1][a+1]*prim_l[PVX+a];
        urt[b] += e_cont[b+1][a+1]*prim_r[PVX+a];
      }
    }

    // Compute the magnetic field in the tetrad frame -- this is a purely spatial vector
    // with a null time component, so the raised and lowered forms are identical in the
    // tetrad frame.
    Real Blt[3] = {0.0};
    Real Brt[3] = {0.0};
    for (int b = 0; b < 3; b++) {
      for (int a = 0; a < 3; a++) {
        Blt[b] += e_cont[b+1][a+1]*Bu_l[a];
        Brt[b] += e_cont[b+1][a+1]*Bu_r[a];
      }
    }

    //------------------------------------------------------------------------------------
    // Calculate the left and right fluxes. We have a function for this, but we'll do a
    // lot of extra work raising and lowering things if we use it directly.
    // FIXME(JF): If we add more transforming solvers, we should add a utility function
    // for the SRMHD fluxes and refactor.
    //------------------------------------------------------------------------------------

    // LEFT STATE

    // Start by calculating the conserved variables.
    Real cons_l[NCONS];
    Real Wl = Kokkos::sqrt(ult[0]*ult[0] + ult[1]*ult[1] + ult[2]*ult[2] + 1.0);
    Real iWl = 1.0/Wl;
    Real bl0 = (Blt[0]*ult[0] + Blt[1]*ult[1] + Blt[2]*ult[2]);
    Real Bvl = bl0*iWl;
    Real Bsq_l = Blt[0]*Blt[0] + Blt[1]*Blt[1] + Blt[2]*Blt[2];
    Real bsq_l = Bsq_l*iWl*iWl + Bvl*Bvl;
    Real rhohB_l = (eos.ps.GetEOS().GetEnergy(prim_l[PRH], prim_l[PTM], &prim_l[PYF]) +
                  prim_l[PPR])*Wl*Wl + Bsq_l;
    cons_l[CDN] = mb*prim_l[PRH]*Wl;
    cons_l[CSX] = rhohB_l*ult[0]*iWl - Bvl*Blt[0];
    cons_l[CSY] = rhohB_l*ult[1]*iWl - Bvl*Blt[1];
    cons_l[CSZ] = rhohB_l*ult[2]*iWl - Bvl*Blt[2];
    cons_l[CTA] = rhohB_l - prim_l[PPR] - 0.5*bsq_l - cons_l[CDN];
    /*for (int s = 0; s < nscal; s++) {
      cons_l[CYD + s] = cons_l[CDN]*prim_l[PYF + s];
    }*/

    // Now calculate the fluxes.
    Real flux_l[NCONS];
    Real vx = ult[ibx]*iWl;
    Real blu[3] = {(Blt[0] + bl0*ult[0])*iWl,
                   (Blt[1] + bl0*ult[1])*iWl,
                   (Blt[2] + bl0*ult[2])*iWl};
    flux_l[CDN] = cons_l[CDN]*vx;
    flux_l[CSX] = cons_l[CSX]*vx - blu[0]*Blt[ibx]*iWl;
    flux_l[CSY] = cons_l[CSY]*vx - blu[1]*Blt[ibx]*iWl;
    flux_l[CSZ] = cons_l[CSZ]*vx - blu[2]*Blt[ibx]*iWl;
    flux_l[CTA] = (cons_l[CTA] + prim_l[PPR] + 0.5*bsq_l)*vx - bl0*Blt[ibx]*iWl;
    flux_l[csx] += prim_l[PPR] + 0.5*bsq_l;

    Real bflux_l[NMAG];
    bflux_l[ibx] = 0.0;
    bflux_l[iby] = (Blt[iby]*vx - Blt[ibx]*ult[iby]*iWl);
    bflux_l[ibz] = (Blt[ibz]*vx - Blt[ibx]*ult[ibz]*iWl);


    // RIGHT STATE

    // Start by calculating the conserved variables.
    Real cons_r[NCONS];
    Real Wr = Kokkos::sqrt(urt[0]*urt[0] + urt[1]*urt[1] + urt[2]*urt[2] + 1.0);
    Real iWr = 1.0/Wr;
    Real br0 = (Brt[0]*urt[0] + Brt[1]*urt[1] + Brt[2]*urt[2]);
    Real Bvr = br0*iWr;
    Real Bsq_r = Brt[0]*Brt[0] + Brt[1]*Brt[1] + Brt[2]*Brt[2];
    Real bsq_r = Bsq_r*iWr*iWr + Bvr*Bvr;
    Real rhohB_r = (eos.ps.GetEOS().GetEnergy(prim_r[PRH], prim_r[PTM], &prim_r[PYF]) +
                  prim_r[PPR])*Wr*Wr + Bsq_r;
    cons_r[CDN] = mb*prim_r[PRH]*Wr;
    cons_r[CSX] = rhohB_r*urt[0]*iWr - Bvr*Brt[0];
    cons_r[CSY] = rhohB_r*urt[1]*iWr - Bvr*Brt[1];
    cons_r[CSZ] = rhohB_r*urt[2]*iWr - Bvr*Brt[2];
    cons_r[CTA] = rhohB_r - prim_r[PPR] - 0.5*bsq_r - cons_r[CDN];
    /*for (int s = 0; s < nscal; s++) {
      cons_r[CYD + s] = cons_r[CDN]*prim_r[PYF + s];
    }*/

    // Now calculate the fluxes.
    Real flux_r[NCONS];
    vx = urt[ibx]*iWr;
    Real bru[3] = {(Brt[0] + br0*urt[0])*iWr,
                   (Brt[1] + br0*urt[1])*iWr,
                   (Brt[2] + br0*urt[2])*iWr};
    flux_r[CDN] = cons_r[CDN]*vx;
    flux_r[CSX] = cons_r[CSX]*vx - bru[0]*Brt[ibx]*iWr;
    flux_r[CSY] = cons_r[CSY]*vx - bru[1]*Brt[ibx]*iWr;
    flux_r[CSZ] = cons_r[CSZ]*vx - bru[2]*Brt[ibx]*iWr;
    flux_r[CTA] = (cons_r[CTA] + prim_r[PPR] + 0.5*bsq_r)*vx - br0*Brt[ibx]*iWr;
    flux_r[csx] += prim_r[PPR] + 0.5*bsq_r;

    Real bflux_r[NMAG];
    bflux_r[ibx] = 0.0;
    bflux_r[iby] = (Brt[iby]*vx - Brt[ibx]*urt[iby]*iWr);
    bflux_r[ibz] = (Brt[ibz]*vx - Brt[ibx]*urt[ibz]*iWr);


    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    // Replace the velocity in the primitive variables
    prim_l[PVX] = ult[0];
    prim_l[PVY] = ult[1];
    prim_l[PVZ] = ult[2];
    prim_r[PVX] = urt[0];
    prim_r[PVY] = urt[1];
    prim_r[PVZ] = urt[2];
    eos.GetSRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsq_l, pvx);
    eos.GetSRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsq_r, pvx);
    
    // Get the extremal wavespeeds
    Real lambda_l = Kokkos::fmin(lambda_ml, lambda_mr);
    Real lambda_r = Kokkos::fmax(lambda_pl, lambda_pr);

    // Calculate fluxes in HLL region -- Note that even though we're in the tetrad frame,
    // a factor of 1/\alpha is still needed to get the correct volume form because the
    // flux is actually \alpha F, not F, and \alpha is a scalar that does not get
    // transformed away in the tetrad frame.
    Real qa = lambda_r*lambda_l;
    Real qb = 1.0/(lambda_r - lambda_l);
    Real f_hll[NCONS], bf_hll[NMAG];
    f_hll[CDN] = ((lambda_r*flux_l[CDN] - lambda_l*flux_r[CDN]) +
                  qa*(cons_r[CDN] - cons_l[CDN])) * qb;
    f_hll[CSX] = ((lambda_r*flux_l[CSX] - lambda_l*flux_r[CSX]) +
                  qa*(cons_r[CSX] - cons_l[CSX])) * qb;
    f_hll[CSY] = ((lambda_r*flux_l[CSY] - lambda_l*flux_r[CSY]) +
                  qa*(cons_r[CSY] - cons_l[CSY])) * qb;
    f_hll[CSZ] = ((lambda_r*flux_l[CSZ] - lambda_l*flux_r[CSZ]) +
                  qa*(cons_r[CSZ] - cons_l[CSZ])) * qb;
    f_hll[CTA] = ((lambda_r*flux_l[CTA] - lambda_l*flux_r[CTA]) +
                  qa*(cons_r[CTA] - cons_l[CTA])) * qb;
    bf_hll[ibx] = 0.0;
    bf_hll[iby] = ((lambda_r*bflux_l[iby] - lambda_l*bflux_r[iby]) +
                   qa*(Brt[iby] - Blt[iby])) * qb;
    bf_hll[ibz] = ((lambda_r*bflux_l[ibz] - lambda_l*bflux_r[ibz]) +
                   qa*(Brt[ibz] - Blt[ibz])) * qb;

    // Conserved state in the HLL region
    Real cons_hll[NCONS], b_hll[NMAG];
    cons_hll[CDN] = ((lambda_r*cons_r[CDN] - lambda_l*cons_l[CDN]) +
                     (flux_l[CDN] - flux_r[CDN])) * qb;
    cons_hll[CSX] = ((lambda_r*cons_r[CSX] - lambda_l*cons_l[CSX]) +
                     (flux_l[CSX] - flux_r[CSX])) * qb;
    cons_hll[CSY] = ((lambda_r*cons_r[CSY] - lambda_l*cons_l[CSY]) +
                     (flux_l[CSY] - flux_r[CSY])) * qb;
    cons_hll[CSZ] = ((lambda_r*cons_r[CSZ] - lambda_l*cons_l[CSZ]) +
                     (flux_l[CSZ] - flux_r[CSZ])) * qb;
    cons_hll[CTA] = ((lambda_r*cons_r[CTA] - lambda_l*cons_l[CTA]) +
                     (flux_l[CTA] - flux_r[CTA])) * qb;
    b_hll[ibx] = Blt[ibx];
    b_hll[iby] = ((lambda_r*Brt[iby] - lambda_l*Blt[iby]) +
                  (bflux_l[iby] - bflux_r[iby])) * qb;
    b_hll[ibz] = ((lambda_r*Brt[ibz] - lambda_l*Blt[ibz]) +
                  (bflux_r[ibz] - bflux_r[ibz])) * qb;

    // Note that the interface is moving! e_cont[ivx][ivx] = 1/sqrt(g^{xx}), so we use it
    // to save an expensive computation.
    Real vint = beta_u[ibx]*ialpha*e_cont[ivx][ivx];
    Real *f_interface, *bf_interface, *cons_interface, *b_interface;
    if (lambda_l >= vint) {
      f_interface = &flux_l[0];
      bf_interface = &bflux_l[0];
      cons_interface = &cons_l[0];
      b_interface = &Blt[0];
    } else if (lambda_r <= vint) {
      f_interface = &flux_r[0];
      bf_interface = &bflux_r[0];
      cons_interface = &cons_r[0];
      b_interface = &Brt[0];
    } else {
      f_interface = &f_hll[0];
      bf_interface = &bf_hll[0];
      cons_interface = &cons_hll[0];
      b_interface = &b_hll[0];
    }

    Real vol = alpha*sdetg;

    // Calculate the fluxes, transforming back into the lab frame
    flx(m, IDN, k, j, i) = vol * (e_cov[ivx][0]*cons_interface[CDN] +
                                  e_cov[ivx][ivx]*f_interface[CDN]);
    flx(m, IEN, k, j, i) = vol * (e_cov[ivx][0]*cons_interface[CTA] +
                                  e_cov[ivx][ivx]*f_interface[CTA]);
    flx(m, ivx, k, j, i) = vol * (e_cov[ivx][0]*(e_cont[ivx][ivx]*cons_interface[csx] +
                                                 e_cont[ivy][ivx]*cons_interface[csy] +
                                                 e_cont[ivz][ivx]*cons_interface[csz]) +
                                  e_cov[ivx][ivx]*(e_cont[ivx][ivx]*f_interface[csx] +
                                                   e_cont[ivy][ivx]*f_interface[csy] +
                                                   e_cont[ivz][ivx]*f_interface[csz]));
    flx(m, ivy, k, j, i) = vol * (e_cov[ivx][0]*(e_cont[ivx][ivy]*cons_interface[csx] +
                                                 e_cont[ivy][ivy]*cons_interface[csy] +
                                                 e_cont[ivz][ivy]*cons_interface[csz]) +
                                  e_cov[ivx][ivx]*(e_cont[ivx][ivy]*f_interface[csx] +
                                                   e_cont[ivy][ivy]*f_interface[csy] +
                                                   e_cont[ivz][ivy]*f_interface[csz]));
    flx(m, ivz, k, j, i) = vol * (e_cov[ivx][0]*e_cont[ivz][ivz]*cons_interface[csz] +
                                  e_cov[ivx][ivx]*e_cont[ivz][ivz]*f_interface[csz]);
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = -vol * (e_cov[ivx][0]*(e_cov[ivy][ivx]*b_interface[ibx] +
                                            e_cov[ivy][ivy]*b_interface[iby] +
                                            e_cov[ivy][ivz]*b_interface[ibz]) -
                             e_cov[ivx][ivx]*e_cov[ivy][0]*b_interface[ibx] +
                             e_cov[ivx][ivx]*e_cov[ivy][ivy]*bf_interface[iby]);
    ez(m, k, j, i) = vol * (e_cov[ivx][0]*(e_cov[ivz][ivx]*b_interface[ibx] +
                                           e_cov[ivz][ivy]*b_interface[iby] +
                                           e_cov[ivz][ivz]*b_interface[ibz]) -
                             e_cov[ivx][ivx]*e_cov[ivz][0]*b_interface[ibx] +
                             e_cov[ivx][ivx]*(e_cov[ivz][ivy]*bf_interface[iby] +
                                              e_cov[ivz][ivz]*bf_interface[ibz]));
  });
}

} // namespace dyngr


#endif // DYN_GRMHD_RSOLVERS_HLLE_TRANSFORMING_HPP_
