#ifndef HYDRO_RSOLVERS_HLLE_GRHYD_HPP_
#define HYDRO_RSOLVERS_HLLE_GRHYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grhyd.hpp
//! \brief HLLE Riemann solver for general relativistic hydrodynamics.
//!
//! Notes:
//!  - cf. HLLE solver in hlle_rel_no_transform.cpp in Athena++

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void HLLE_GR
//! \brief HLLE for GR hydrodynamics

KOKKOS_INLINE_FUNCTION
void HLLE_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);
  auto &flat = coord.is_minkowski;
  auto &spin = coord.bh_spin;

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
    // References to left primitives
    Real &wl_idn=wl(IDN,i);
    Real &wl_ivx=wl(ivx,i);
    Real &wl_ivy=wl(ivy,i);
    Real &wl_ivz=wl(ivz,i);

    // References to right primitives
    Real &wr_idn=wr(IDN,i);
    Real &wr_ivx=wr(ivx,i);
    Real &wr_ivy=wr(ivy,i);
    Real &wr_ivz=wr(ivz,i);

    Real wl_ipr, wr_ipr;
    wl_ipr = eos.IdealGasPressure(wl(IEN,i));
    wr_ipr = eos.IdealGasPressure(wr(IEN,i));

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v,x2v,x3v;
    if (ivx == IVX) {
      x1v = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else if (ivx == IVY) {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);
    }
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Calculate 4-velocity in left state (contravariant compt)
    Real q = glower[ivx][ivx] * SQR(wl_ivx) + glower[ivy][ivy] * SQR(wl_ivy) +
             glower[ivz][ivz] * SQR(wl_ivz) + 2.0*glower[ivx][ivy] * wl_ivx * wl_ivy +
         2.0*glower[ivx][ivz] * wl_ivx * wl_ivz + 2.0*glower[ivy][ivz] * wl_ivy * wl_ivz;

    Real alpha = std::sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + q);
    Real uul[4];
    uul[0] = gamma / alpha;
    uul[ivx] = wl_ivx - alpha * gamma * gupper[0][ivx];
    uul[ivy] = wl_ivy - alpha * gamma * gupper[0][ivy];
    uul[ivz] = wl_ivz - alpha * gamma * gupper[0][ivz];

    // lower vector indices (covariant compt)
    Real ull[4];
    ull[0]   = glower[0][0]  *uul[0]   + glower[0][ivx]*uul[ivx] +
               glower[0][ivy]*uul[ivy] + glower[0][ivz]*uul[ivz];

    ull[ivx] = glower[ivx][0]  *uul[0]   + glower[ivx][ivx]*uul[ivx] +
               glower[ivx][ivy]*uul[ivy] + glower[ivx][ivz]*uul[ivz];

    ull[ivy] = glower[ivy][0]  *uul[0]   + glower[ivy][ivx]*uul[ivx] +
               glower[ivy][ivy]*uul[ivy] + glower[ivy][ivz]*uul[ivz];

    ull[ivz] = glower[ivz][0]  *uul[0]   + glower[ivz][ivx]*uul[ivx] +
               glower[ivz][ivy]*uul[ivy] + glower[ivz][ivz]*uul[ivz];

    // Calculate 4-velocity in right state (contravariant compt)
    q = glower[ivx][ivx] * SQR(wr_ivx) + glower[ivy][ivy] * SQR(wr_ivy) +
        glower[ivz][ivz] * SQR(wr_ivz) + 2.0*glower[ivx][ivy] * wr_ivx * wr_ivy +
        2.0*glower[ivx][ivz] * wr_ivx * wr_ivz + 2.0*glower[ivy][ivz] * wr_ivy * wr_ivz;

    gamma = sqrt(1.0 + q);
    Real uur[4];
    uur[0] = gamma / alpha;
    uur[ivx] = wr_ivx - alpha * gamma * gupper[0][ivx];
    uur[ivy] = wr_ivy - alpha * gamma * gupper[0][ivy];
    uur[ivz] = wr_ivz - alpha * gamma * gupper[0][ivz];

    // lower vector indices (covariant compt)
    Real ulr[4];
    ulr[0]   = glower[0][0]  *uur[0]   + glower[0][ivx]*uur[ivx] +
               glower[0][ivy]*uur[ivy] + glower[0][ivz]*uur[ivz];

    ulr[ivx] = glower[ivx][0]  *uur[0]   + glower[ivx][ivx]*uur[ivx] +
               glower[ivx][ivy]*uur[ivy] + glower[ivx][ivz]*uur[ivz];

    ulr[ivy] = glower[ivy][0]  *uur[0]   + glower[ivy][ivx]*uur[ivx] +
               glower[ivy][ivy]*uur[ivy] + glower[ivy][ivz]*uur[ivz];

    ulr[ivz] = glower[ivz][0]  *uur[0]   + glower[ivz][ivx]*uur[ivx] +
               glower[ivz][ivy]*uur[ivy] + glower[ivz][ivz]*uur[ivz];

    // Calculate wavespeeds in left state
    Real lp_l, lm_l;
    eos.IdealGRHydroSoundSpeeds(wl_idn, wl_ipr, uul[0], uul[ivx], gupper[0][0],
                                gupper[0][ivx], gupper[ivx][ivx], lp_l, lm_l);

    // Calculate wavespeeds in right state
    Real lp_r, lm_r;
    eos.IdealGRHydroSoundSpeeds(wr_idn, wr_ipr, uur[0], uur[ivx], gupper[0][0],
                                gupper[0][ivx], gupper[ivx][ivx], lp_r, lm_r);

    // Calculate extremal wavespeeds
    Real lambda_l = fmin(lm_l, lm_r);
    Real lambda_r = fmax(lp_l, lp_r);

    // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
    HydCons1D du;
    Real wgas_l = wl_idn + gamma_prime * wl_ipr;
    Real wgas_r = wr_idn + gamma_prime * wr_ipr;
    Real qa = wgas_r * uur[0];
    Real qb = wgas_l * uul[0];
    du.d  = wr_idn * uur[0] - wl_idn * uul[0];
    du.mx = qa * ulr[ivx] - qb * ull[ivx];
    du.my = qa * ulr[ivy] - qb * ull[ivy];
    du.mz = qa * ulr[ivz] - qb * ull[ivz];
    du.e  = qa * ulr[0] - qb * ull[0] + wr_ipr - wl_ipr;

    // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
    HydCons1D fl;
    qa = wgas_l * uul[ivx];
    fl.d  = wl_idn * uul[ivx];
    fl.mx = qa * ull[ivx] + wl_ipr;
    fl.my = qa * ull[ivy];
    fl.mz = qa * ull[ivz];
    fl.e  = qa * ull[0];

    // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
    HydCons1D fr;
    qa = wgas_r * uur[ivx];
    fr.d  = wr_idn * uur[ivx];
    fr.mx = qa * ulr[ivx] + wr_ipr;
    fr.my = qa * ulr[ivy];
    fr.mz = qa * ulr[ivz];
    fr.e  = qa * ulr[0];

    // Calculate fluxes in HLL region
    HydCons1D flux_hll;
    qa = lambda_r*lambda_l;
    qb = 1.0/(lambda_r - lambda_l);
    flux_hll.d  = (lambda_r*fl.d  - lambda_l*fr.d  + qa*du.d ) * qb;
    flux_hll.mx = (lambda_r*fl.mx - lambda_l*fr.mx + qa*du.mx) * qb;
    flux_hll.my = (lambda_r*fl.my - lambda_l*fr.my + qa*du.my) * qb;
    flux_hll.mz = (lambda_r*fl.mz - lambda_l*fr.mz + qa*du.mz) * qb;
    flux_hll.e  = (lambda_r*fl.e  - lambda_l*fr.e  + qa*du.e ) * qb;

    // Determine region of wavefan
    HydCons1D *flux_interface;
    if (lambda_l >= 0.0) {  // L region
      flux_interface = &fl;
    } else if (lambda_r <= 0.0) { // R region
      flux_interface = &fr;
    } else {  // HLL region
      flux_interface = &flux_hll;
    }

    // Set fluxes
    flx(m,IDN,k,j,i) = flux_interface->d;
    flx(m,ivx,k,j,i) = flux_interface->mx;
    flx(m,ivy,k,j,i) = flux_interface->my;
    flx(m,ivz,k,j,i) = flux_interface->mz;
    flx(m,IEN,k,j,i) = flux_interface->e;

    // We evolve tau = T^t_t + D
    flx(m,IEN,k,j,i) += flx(m,IDN,k,j,i);
  });

  return;
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLE_GRHYD_HPP_
