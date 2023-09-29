#ifndef DYNGR_RSOLVERS_HLLE_DYNGRMHD_HPP_
#define DYNGR_RSOLVERS_HLLE_DYNGRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_dyngrmhd.hpp
//! \brief HLLE Riemann solver for general relativistic magnetohydrodynamics

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "adm/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void HLLE_DYNGR
//! \brief inline function for calculating GRMHD fluxes via Lax-Friedrichs
//! TODO: This could potentially be sped up by calculating the conserved variables without
//  the help of PrimitiveSolver; there are redundant calculations with B^i v_i and W that
//  may not be needed.
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void HLLE_DYNGR(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const int& nhyd, const int& nscal,
     const AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> &gamma_dd,
     const AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> &b_u,
     const AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> &alp, DvceArray5D<Real> flx,
     DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  // Cyclic permutation of array indices
  int pvx, pvy, pvz;
  //int idx, idy, idz;
  int idx;
  //int csx, csy, csz;
  int csx;
  int ibx, iby, ibz;

  int diag[3] = {S11, S22, S33};

  pvx = PVX + (ivx - IVX);
  pvy = PVX + ((ivx - IVX) + 1)%3;
  pvz = PVX + ((ivx - IVX) + 2)%3;

  csx = CSX + (ivx - IVX);
  //csy = CSX + ((ivx - IVX) + 1)%3;
  //csz = CSX + ((ivx - IVX) + 2)%3;

  idx = diag[ivx - IVX];
  //idy = diag[((ivx - IVX) + 1)%3];
  //idz = diag[((ivx - IVX) + 2)%3];

  ibx = ivx - IVX;
  iby = ((ivx - IVX) + 1)%3;
  ibz = ((ivx - IVX) + 2)%3;

  //int is = indcs.is;
  //int js = indcs.js;
  //int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
    // Extract metric components
    Real g3d[NSPMETRIC];
    g3d[S11] = gamma_dd(0, 0, i);
    g3d[S12] = gamma_dd(0, 1, i);
    g3d[S13] = gamma_dd(0, 2, i);
    g3d[S22] = gamma_dd(1, 1, i);
    g3d[S23] = gamma_dd(1, 2, i);
    g3d[S33] = gamma_dd(2, 2, i);
    Real detg = Primitive::GetDeterminant(g3d);
    Real sdetg = sqrt(detg);
    Real g3u[NSPMETRIC];
    Primitive::InvertMatrix(g3u, g3d, detg);

    // Shift vector
    Real beta_u[3];
    beta_u[0] = b_u(0, i);
    beta_u[1] = b_u(1, i);
    beta_u[2] = b_u(2, i);

    // Lapse
    Real alpha = alp(i);

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
    Real Wsql = 1.0 + Primitive::Contract(uul, udl);
    Real Wl = sqrt(Wsql);
    Real vcl = prim_l[pvx]/Wl - beta_u[ivx-IVX]/alpha;

    // Calculate 4-magnetic field (undensitized) for the left state.
    Real bul0 = Primitive::Contract(Bu_l, udl)/(alpha*sdetg);
    /*Real bul[3];
    for (int a = 0; a < 3; a++) {
      bul[a] = (Bu_l[a] + alpha*bul0*(uul[a] - Wl*beta_u[a]/alpha))/Wl;
    }*/
    Real bdl[3];
    for (int a = 0; a < 3; a++) {
      bdl[a] = alpha*bul0*udl[a];
      for (int b = 0; b < 3; b++) {
        bdl[a] += gamma_dd(a, b, i)*Bu_l[b]/sdetg;
      }
      bdl[a] = bdl[a]/Wl;
    }
    Real bsql = (Primitive::SquareVector(Bu_l, g3d)/detg + SQR(alpha*bul0))/(Wsql);

    // Calculate W for the right state.
    Real uur[3] = {prim_r[IVX], prim_r[IVY], prim_r[IVZ]};
    Real udr[3];
    Primitive::LowerVector(udr, uur, g3d);
    Real Wsqr = 1.0 + Primitive::Contract(uur, udr);
    Real Wr = sqrt(Wsqr);
    Real vcr = prim_r[pvx]/Wr - beta_u[ivx-IVX]/alpha;

    // Calculate 4-magnetic field (densitized) for the right state.
    Real bur0 = Primitive::Contract(Bu_r, udr)/(alpha*sdetg);
    /*Real bur[3];
    for (int a = 0; a < 3; a++) {
      bur[a] = (Bu_r[a] + alpha*bur0*(uur[a] - Wr*beta_u[a]/alpha))/Wr;
    }*/
    Real bdr[3];
    for (int a = 0; a < 3; a++) {
      bdr[a] = alpha*bur0*udr[a];
      for (int b = 0; b < 3; b++) {
        bdr[a] += gamma_dd(a, b, i)*Bu_r[b]/sdetg;
      }
      bdr[a] = bdr[a]/Wr;
    }
    Real bsqr = (Primitive::SquareVector(Bu_r, g3d)/detg + SQR(alpha*bur0))/(Wsqr);


    // Calculate fluxes for the left state.
    Real fl[NCONS], bfl[NMAG];
    fl[CDN] = alpha*cons_l[CDN]*vcl;
    fl[CSX] = alpha*(cons_l[CSX]*vcl - bdl[0]*Bu_l[ibx]/Wl);
    fl[CSY] = alpha*(cons_l[CSY]*vcl - bdl[1]*Bu_l[ibx]/Wl);
    fl[CSZ] = alpha*(cons_l[CSZ]*vcl - bdl[2]*Bu_l[ibx]/Wl);
    fl[csx] += alpha*sdetg*(prim_l[PPR] + 0.5*bsql);
    fl[CTA] = alpha*(cons_l[CTA]*vcl - alpha*bul0*Bu_l[ibx]/Wl
            + sdetg*(prim_l[PPR] + 0.5*bsql)*prim_l[ivx]/Wl);

    bfl[ibx] = 0.0;
    bfl[iby] = alpha*(Bu_l[iby]*vcl -
                      Bu_l[ibx]*(prim_l[pvy]/Wl - beta_u[pvy - PVX]/alpha));
    bfl[ibz] = alpha*(Bu_l[ibz]*vcl -
                      Bu_l[ibx]*(prim_l[pvz]/Wl - beta_u[pvz - PVX]/alpha));

    // Calculate fluxes for the right state.
    Real fr[NCONS], bfr[NMAG];
    fr[CDN] = alpha*cons_r[CDN]*vcr;
    fr[CSX] = alpha*(cons_r[CSX]*vcr - bdr[0]*Bu_r[ibx]/Wr);
    fr[CSY] = alpha*(cons_r[CSY]*vcr - bdr[1]*Bu_r[ibx]/Wr);
    fr[CSZ] = alpha*(cons_r[CSZ]*vcr - bdr[2]*Bu_r[ibx]/Wr);
    fr[csx] += alpha*sdetg*(prim_r[PPR] + 0.5*bsqr);
    fr[CTA] = alpha*(cons_r[CTA]*vcr - alpha*bur0*Bu_r[ibx]/Wr
            + sdetg*(prim_r[PPR] + 0.5*bsqr)*prim_r[ivx]/Wr);

    bfr[ibx] = 0.0;
    bfr[iby] = alpha*(Bu_r[iby]*vcr -
                      Bu_r[ibx]*(prim_r[pvy]/Wr - beta_u[pvy - PVX]/alpha));
    bfr[ibz] = alpha*(Bu_r[ibz]*vcr -
                      Bu_r[ibx]*(prim_r[pvz]/Wr - beta_u[pvz - PVX]/alpha));


    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    eos.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                    g3d, beta_u, alpha, g3d[idx], pvx);
    eos.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                    g3d, beta_u, alpha, g3d[idx], pvx);

    // Get the extremal wavespeeds
    Real lambda_l = fmin(lambda_ml, lambda_mr);
    Real lambda_r = fmax(lambda_pl, lambda_pr);

    // Calculate fluxes in HLL region
    Real qa = lambda_r*lambda_l;
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


    // Flux at the interface
    Real *f_interface, *bf_interface;
    if (lambda_l >= 0) {        // L region
      f_interface = &fl[0];
      bf_interface = &bfl[0];
    } else if (lambda_r <= 0) {   // R region
      f_interface = &fr[0];
      bf_interface = &bfr[0];
    } else {                      // HLL region
      f_interface = &f_hll[0];
      bf_interface = &bf_hll[0];
    }

    flx(m, IDN, k, j, i) = f_interface[CDN];
    flx(m, IEN, k, j, i) = f_interface[CTA];
    flx(m, IVX, k, j, i) = f_interface[CSX];
    flx(m, IVY, k, j, i) = f_interface[CSY];
    flx(m, IVZ, k, j, i) = f_interface[CSZ];
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = -bf_interface[iby];
    ez(m, k, j, i) = bf_interface[ibz];
  });
}

} // namespace dyngr

#endif  // DYNGR_RSOLVERS_HLLE_DYNGRMHD_HPP_
