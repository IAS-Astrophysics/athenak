//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grhyd.cpp
//! \brief HLLE Riemann solver for general relativistic hydrodynamics.
//
//! Notes:
//!  - cf. HLLE solver in hlle_rel_no_transform.cpp in Athena++

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void HLLE_GR
//! \brief
//

KOKKOS_INLINE_FUNCTION
void HLLE_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  const Real gm1 = (eos.gamma - 1.0);
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
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

    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                            coord.bh_spin, g_, gi_);

    const Real
      &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
      &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
      &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
      &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];
    const Real
      &g00 = gi_[I00], &g01 = gi_[I01], &g02 = gi_[I02], &g03 = gi_[I03],
                       &g11 = gi_[I11],
                                        &g22 = gi_[I22],
                                                         &g33 = gi_[I33];
    Real alpha = std::sqrt(-1.0/g00);
    Real gii, g0i;
    if (ivx == IVX) {
      gii = g11;
      g0i = g01;
    } else if (ivx == IVY) {
      gii = g22;
      g0i = g02;
    } else {
      gii = g33;
      g0i = g03;
    }

    // Extract left primitives
    const Real &rho_l  = wl(IDN,i);
    const Real &uu1_l  = wl(IVX,i);
    const Real &uu2_l  = wl(IVY,i);
    const Real &uu3_l  = wl(IVZ,i);

    // Extract right primitives
    const Real &rho_r  = wr(IDN,i);
    const Real &uu1_r  = wr(IVX,i);
    const Real &uu2_r  = wr(IVY,i);
    const Real &uu3_r  = wr(IVZ,i);

    Real pgas_l, pgas_r;
    pgas_l = eos.IdealGasPressure(wl(IDN,i), wl(IEN,i));
    pgas_r = eos.IdealGasPressure(wr(IDN,i), wr(IEN,i));

    // Calculate 4-velocity in left state
    Real ucon_l[4], ucov_l[4];
    Real tmp = g_11*SQR(uu1_l) + 2.0*g_12*uu1_l*uu2_l + 2.0*g_13*uu1_l*uu3_l
             + g_22*SQR(uu2_l) + 2.0*g_23*uu2_l*uu3_l
             + g_33*SQR(uu3_l);
    Real gamma_l = sqrt(1.0 + tmp);
    ucon_l[0] = gamma_l / alpha;
    ucon_l[1] = uu1_l - alpha * gamma_l * g01;
    ucon_l[2] = uu2_l - alpha * gamma_l * g02;
    ucon_l[3] = uu3_l - alpha * gamma_l * g03;
    ucov_l[0] = g_00*ucon_l[0] + g_01*ucon_l[1] + g_02*ucon_l[2] + g_03*ucon_l[3];
    ucov_l[1] = g_10*ucon_l[0] + g_11*ucon_l[1] + g_12*ucon_l[2] + g_13*ucon_l[3];
    ucov_l[2] = g_20*ucon_l[0] + g_21*ucon_l[1] + g_22*ucon_l[2] + g_23*ucon_l[3];
    ucov_l[3] = g_30*ucon_l[0] + g_31*ucon_l[1] + g_32*ucon_l[2] + g_33*ucon_l[3];

    // Calculate 4-velocity in right state
    Real ucon_r[4], ucov_r[4];
    tmp = g_11*SQR(uu1_r) + 2.0*g_12*uu1_r*uu2_r + 2.0*g_13*uu1_r*uu3_r
        + g_22*SQR(uu2_r) + 2.0*g_23*uu2_r*uu3_r
        + g_33*SQR(uu3_r);
    Real gamma_r = sqrt(1.0 + tmp);
    ucon_r[0] = gamma_r / alpha;
    ucon_r[1] = uu1_r - alpha * gamma_r * g01;
    ucon_r[2] = uu2_r - alpha * gamma_r * g02;
    ucon_r[3] = uu3_r - alpha * gamma_r * g03;
    ucov_r[0] = g_00*ucon_r[0] + g_01*ucon_r[1] + g_02*ucon_r[2] + g_03*ucon_r[3];
    ucov_r[1] = g_10*ucon_r[0] + g_11*ucon_r[1] + g_12*ucon_r[2] + g_13*ucon_r[3];
    ucov_r[2] = g_20*ucon_r[0] + g_21*ucon_r[1] + g_22*ucon_r[2] + g_23*ucon_r[3];
    ucov_r[3] = g_30*ucon_r[0] + g_31*ucon_r[1] + g_32*ucon_r[2] + g_33*ucon_r[3];

    // Calculate wavespeeds in left state
    Real lp_l, lm_l;
    eos.IdealGRHydroSoundSpeeds(rho_l,pgas_l,ucon_l[0],ucon_l[ivx],g00,g0i,gii,lp_l,lm_l);

    // Calculate wavespeeds in right state
    Real lp_r, lm_r;
    eos.IdealGRHydroSoundSpeeds(rho_r,pgas_r,ucon_r[0],ucon_r[ivx],g00,g0i,gii,lp_r,lm_r);

    // Calculate extremal wavespeeds
    Real lambda_l = fmin(lm_l, lm_r);
    Real lambda_r = fmax(lp_l, lp_r);

    // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
    Real wgas_l = rho_l + gamma_prime * pgas_l;
    Real wgas_r = rho_r + gamma_prime * pgas_r;
    Real du[5];
    Real qa = wgas_r * ucon_r[0];
    Real qb = wgas_l * ucon_l[0];
    du[IDN] = rho_r * ucon_r[0] - rho_l * ucon_l[0];
    du[IVX] = qa * ucov_r[1] - qb * ucov_l[1];
    du[IVY] = qa * ucov_r[2] - qb * ucov_l[2];
    du[IVZ] = qa * ucov_r[3] - qb * ucov_l[3];
    du[IEN] = qa * ucov_r[0] - qb * ucov_l[0] + pgas_r - pgas_l;

    // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
    Real flux_l[5];
    qa = wgas_l * ucon_l[ivx];
    flux_l[IDN] = rho_l * ucon_l[ivx];
    flux_l[IEN] = qa * ucov_l[0];
    flux_l[IVX] = qa * ucov_l[1];
    flux_l[IVY] = qa * ucov_l[2];
    flux_l[IVZ] = qa * ucov_l[3];
    flux_l[ivx] += pgas_l;

    // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
    Real flux_r[5];
    qa = wgas_r * ucon_r[ivx];
    flux_r[IDN] = rho_r * ucon_r[ivx];
    flux_r[IEN] = qa * ucov_r[0];
    flux_r[IVX] = qa * ucov_r[1];
    flux_r[IVY] = qa * ucov_r[2];
    flux_r[IVZ] = qa * ucov_r[3];
    flux_r[ivx] += pgas_r;

    // Calculate fluxes in HLL region
    Real flux_hll[5];
    qa = lambda_r*lambda_l;
    qb = lambda_r - lambda_l;
    flux_hll[IDN] = (lambda_r*flux_l[IDN] - lambda_l*flux_r[IDN] + qa*du[IDN]) / qb;
    flux_hll[IVX] = (lambda_r*flux_l[IVX] - lambda_l*flux_r[IVX] + qa*du[IVX]) / qb;
    flux_hll[IVY] = (lambda_r*flux_l[IVY] - lambda_l*flux_r[IVY] + qa*du[IVY]) / qb;
    flux_hll[IVZ] = (lambda_r*flux_l[IVZ] - lambda_l*flux_r[IVZ] + qa*du[IVZ]) / qb;
    flux_hll[IEN] = (lambda_r*flux_l[IEN] - lambda_l*flux_r[IEN] + qa*du[IEN]) / qb;

    // Determine region of wavefan
    Real *flux_interface;
    if (lambda_l >= 0.0) {  // L region
      flux_interface = flux_l;
    } else if (lambda_r <= 0.0) { // R region
      flux_interface = flux_r;
    } else {  // HLL region
      flux_interface = flux_hll;
    }

    // Set fluxes
    flx(m,IDN,k,j,i) = flux_interface[IDN];
    flx(m,IVX,k,j,i) = flux_interface[IVX];
    flx(m,IVY,k,j,i) = flux_interface[IVY];
    flx(m,IVZ,k,j,i) = flux_interface[IVZ];
    flx(m,IEN,k,j,i) = flux_interface[IEN];

    // We evolve tau = E - D
    flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
  });

  return;
}

} // namespace hydro
