#ifndef DYN_GRMHD_RSOLVERS_HLLE_TRANSFORMING_HPP_
#define DYN_GRMHD_RSOLVERS_HLLE_TRANSFORMING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_transforming.hpp
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
    ExtractPrimitives(eos, wl, bl, bx, isdetg, prim_l, Bu_l,
                      nhyd, nscal, m, k, j, i, ibx, iby, ibz);
    ExtractPrimitives(eos, wr, br, bx, isdetg, prim_r, Bu_r,
                      nhyd, nscal, m, k, j, i, ibx, iby, ibz);

    // Compute tetrad transformation
    Real e_cov[4][4], e_cont[4][4];
    ComputeOrthonormalTetrad<ivx>(g3d, beta_u, alpha, isdetg*isdetg, e_cov, e_cont);
    Real ialpha = e_cov[0][0]; // Save a division this way
    TransformPrimitivesToTetrad(prim_l, Bu_l, e_cont);
    TransformPrimitivesToTetrad(prim_r, Bu_r, e_cont);

    // LEFT STATE
    Real cons_l[NCONS], flux_l[NCONS], bflux_l[NMAG], bsq_l;
    SingleStateTetradFlux<ivx>(eos, prim_l, Bu_l, cons_l, flux_l, bflux_l, bsq_l);


    // RIGHT STATE
    Real cons_r[NCONS], flux_r[NCONS], bflux_r[NMAG], bsq_r;
    SingleStateTetradFlux<ivx>(eos, prim_r, Bu_r, cons_r, flux_r, bflux_r, bsq_r);


    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    eos.GetSRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsq_l, pvx);
    eos.GetSRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsq_r, pvx);
    
    // Get the extremal wavespeeds
    Real lambda_l = Kokkos::fmin(lambda_ml, lambda_mr);
    Real lambda_r = Kokkos::fmax(lambda_pl, lambda_pr);

    // Calculate fluxes in HLL region
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
                   qa*(Bu_r[iby] - Bu_l[iby])) * qb;
    bf_hll[ibz] = ((lambda_r*bflux_l[ibz] - lambda_l*bflux_r[ibz]) +
                   qa*(Bu_r[ibz] - Bu_l[ibz])) * qb;

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
    b_hll[ibx] = Bu_l[ibx];
    b_hll[iby] = ((lambda_r*Bu_r[iby] - lambda_l*Bu_l[iby]) +
                  (bflux_l[iby] - bflux_r[iby])) * qb;
    b_hll[ibz] = ((lambda_r*Bu_r[ibz] - lambda_l*Bu_l[ibz]) +
                  (bflux_r[ibz] - bflux_r[ibz])) * qb;

    // Note that the interface is moving! e_cont[ivx][ivx] = 1/sqrt(g^{xx}), so we use it
    // to save an expensive computation.
    Real vint = beta_u[ibx]*ialpha*e_cont[ivx][ivx];
    Real *f_interface, *bf_interface, *cons_interface, *b_interface;
    if (lambda_l >= vint) {
      f_interface = &flux_l[0];
      bf_interface = &bflux_l[0];
      cons_interface = &cons_l[0];
      b_interface = &Bu_l[0];
    } else if (lambda_r <= vint) {
      f_interface = &flux_r[0];
      bf_interface = &bflux_r[0];
      cons_interface = &cons_r[0];
      b_interface = &Bu_r[0];
    } else {
      f_interface = &f_hll[0];
      bf_interface = &bf_hll[0];
      cons_interface = &cons_hll[0];
      b_interface = &b_hll[0];
    }

    Real vol = alpha*sdetg;

    // Transform the fluxes and store them in the global flux arrays.
    TransformFluxesToGlobal(cons_interface, f_interface, b_interface, bf_interface,
                    e_cont, e_cov, flx, ey, ez, vol, m, k, j, i, ivx, ivy, ivz,
                    ibx, iby, ibz, csx, csy, csz);
  });
}

} // namespace dyngr


#endif // DYN_GRMHD_RSOLVERS_HLLE_TRANSFORMING_HPP_
