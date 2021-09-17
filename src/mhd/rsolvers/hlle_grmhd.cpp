//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grmhd.cpp
//! \brief HLLE Riemann solver for general relativistic MHD.
//
//! Notes:
//!  - implements HLLE algorithm similar to that of fluxcalc() in step_ch.c in Harm
//!  - cf. HLLENonTransforming() in hlle_mhd_rel.cpp in Athena++

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "coordinates/cartesian_ks.hpp"

//----------------------------------------------------------------------------------------
//! \fn void HLLE_GR
//! \brief
//

KOKKOS_INLINE_FUNCTION
void HLLE_GR(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray1D<Real> &x1, Real x2, Real x3, Real spin,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez)
{
  // Cyclic permutation of array indices corresponding to velocity/b_field components
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  par_for_inner(member, il, iu, [&](const int i)
  {
    // Extract components of metric
    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1(i), x2, x3, spin, g_, gi_);
    const Real
      &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
      &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
      &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
      &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];
    const Real
      &g00 = gi_[I00], &g01 = gi_[I01], &g02 = gi_[I02], &g03 = gi_[I03],
      &g10 = gi_[I01], &g11 = gi_[I11], &g12 = gi_[I12], &g13 = gi_[I13],
      &g20 = gi_[I02], &g21 = gi_[I12], &g22 = gi_[I22], &g23 = gi_[I23],
      &g30 = gi_[I03], &g31 = gi_[I13], &g32 = gi_[I23], &g33 = gi_[I33];
    Real alpha = sqrt(-1.0/g00);
    Real gii, g0i;
    switch (ivx) {
      case IVX:
        gii = g11;
        g0i = g01;
        break;
      case IVY:
        gii = g22;
        g0i = g02;
        break;
      case IVZ:
        gii = g33;
        g0i = g03;
        break;
    }

    // Extract left primitives
    const Real &rho_l  = wl(IDN,i);
    const Real &pgas_l = wl(IPR,i);
    const Real &uu1_l  = wl(ivx,i);
    const Real &uu2_l  = wl(ivy,i);
    const Real &uu3_l  = wl(ivz,i);
    const Real &bb2_l  = bx(m,k,j,i);
    const Real &bb3_l  = bl(iby,i);
    const Real &bb1_l  = bl(ibz,i);

    // Extract right primitives
    const Real &rho_r  = wr(IDN,i);
    const Real &pgas_r = wr(IPR,i);
    const Real &uu1_r  = wr(ivx,i);
    const Real &uu2_r  = wr(ivy,i);
    const Real &uu3_r  = wr(ivz,i);
    const Real &bb2_r  = bx(m,k,j,i);
    const Real &bb3_r  = br(iby,i);
    const Real &bb1_r  = br(ibz,i);

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

    // Calculate 4-magnetic field in left state
    Real bcon_l[4], bcov_l[4];
    bcon_l[0] = ucon_l[0] * (g_01*bb1_l + g_02*bb2_l + g_03*bb3_l)
              + ucon_l[1] * (g_11*bb1_l + g_12*bb2_l + g_13*bb3_l)
              + ucon_l[2] * (g_21*bb1_l + g_22*bb2_l + g_23*bb3_l)
              + ucon_l[3] * (g_31*bb1_l + g_32*bb2_l + g_33*bb3_l);
    bcon_l[1] = (bb1_l + bcon_l[0] * ucon_l[1]) / ucon_l[0];
    bcon_l[2] = (bb2_l + bcon_l[0] * ucon_l[2]) / ucon_l[0];
    bcon_l[3] = (bb3_l + bcon_l[0] * ucon_l[3]) / ucon_l[0];
    bcov_l[0] = g_00*bcon_l[0] + g_01*bcon_l[1] + g_02*bcon_l[2] + g_03*bcon_l[3];
    bcov_l[1] = g_10*bcon_l[0] + g_11*bcon_l[1] + g_12*bcon_l[2] + g_13*bcon_l[3];
    bcov_l[2] = g_20*bcon_l[0] + g_21*bcon_l[1] + g_22*bcon_l[2] + g_23*bcon_l[3];
    bcov_l[3] = g_30*bcon_l[0] + g_31*bcon_l[1] + g_32*bcon_l[2] + g_33*bcon_l[3];
    Real b_sq_l = bcon_l[0]*bcov_l[0] + bcon_l[1]*bcov_l[1] + bcon_l[2]*bcov_l[2]
                + bcon_l[3]*bcov_l[3];

    // Calculate 4-magnetic field in right state
    Real bcon_r[4], bcov_r[4];
    bcon_r[0] = ucon_r[0] * (g_01*bb1_r + g_02*bb2_r + g_03*bb3_r)
              + ucon_r[1] * (g_11*bb1_r + g_12*bb2_r + g_13*bb3_r)
              + ucon_r[2] * (g_21*bb1_r + g_22*bb2_r + g_23*bb3_r)
              + ucon_r[3] * (g_31*bb1_r + g_32*bb2_r + g_33*bb3_r);
    bcon_r[1] = (bb1_r + bcon_r[0] * ucon_r[1]) / ucon_r[0];
    bcon_r[2] = (bb2_r + bcon_r[0] * ucon_r[2]) / ucon_r[0];
    bcon_r[3] = (bb3_r + bcon_r[0] * ucon_r[3]) / ucon_r[0];
    bcov_r[0] = g_00*bcon_r[0] + g_01*bcon_r[1] + g_02*bcon_r[2] + g_03*bcon_r[3];
    bcov_r[1] = g_10*bcon_r[0] + g_11*bcon_r[1] + g_12*bcon_r[2] + g_13*bcon_r[3];
    bcov_r[2] = g_20*bcon_r[0] + g_21*bcon_r[1] + g_22*bcon_r[2] + g_23*bcon_r[3];
    bcov_r[3] = g_30*bcon_r[0] + g_31*bcon_r[1] + g_32*bcon_r[2] + g_33*bcon_r[3];
    Real b_sq_r = bcon_r[0]*bcov_r[0] + bcon_r[1]*bcov_r[1] + bcon_r[2]*bcov_r[2]
                + bcon_r[3]*bcov_r[3];

    // Calculate wavespeeds in left state
    Real lp_l, lm_l;
    Real wgas_l = rho_l + gamma_prime * pgas_l;
    eos.FastSpeedsGR(wgas_l, pgas_l, ucon_l[0], ucon_l[IVX], b_sq_l,
                     g00, g0i, gii, lp_l, lm_l);

    // Calculate wavespeeds in right state
    Real lp_r, lm_r;
    Real wgas_r = rho_r + gamma_prime * pgas_r;
    eos.FastSpeedsGR(wgas_r, pgas_r, ucon_r[0], ucon_r[IVX], b_sq_r,
                     g00, g0i, gii, lp_r, lm_r);

    // Calculate extremal wavespeeds
    Real lambda_l = min(lm_l, lm_r);
    Real lambda_r = max(lp_l, lp_r);

    // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
    MHDCons1D du;
    Real wtot_r = wgas_r + b_sq_r;
    Real ptot_r = pgas_r + 0.5*b_sq_r;
    Real qa = wtot_r * ucon_r[0];
    du.d  = rho_r * ucon_r[0];
    du.e  = qa * ucov_r[0] - bcon_r[0] * bcov_r[0] + ptot_r;
    du.mx = qa * ucov_r[1] - bcon_r[0] * bcov_r[1];
    du.my = qa * ucov_r[2] - bcon_r[0] * bcov_r[2];
    du.mz = qa * ucov_r[3] - bcon_r[0] * bcov_r[3];
    du.by = bcon_r[IVY] * ucon_r[0] - bcon_r[0] * ucon_r[IVY];
    du.bz = bcon_r[IVZ] * ucon_r[0] - bcon_r[0] * ucon_r[IVZ];


    Real wtot_l = wgas_l + b_sq_l;
    Real ptot_l = pgas_l + 0.5*b_sq_l;
    Real qb = wtot_l * ucon_l[0];
    du.d  -= (rho_l * ucon_l[0]);
    du.e  -= (qb * ucov_l[0] - bcon_l[0] * bcov_l[0] + ptot_l);
    du.mx -= (qb * ucov_l[1] - bcon_l[0] * bcov_l[1]);
    du.my -= (qb * ucov_l[2] - bcon_l[0] * bcov_l[2]);
    du.mz -= (qb * ucov_l[3] - bcon_l[0] * bcov_l[3]);
    du.by -= bcon_l[IVY] * ucon_l[0] - bcon_l[0] * ucon_l[IVY];
    du.bz -= bcon_l[IVZ] * ucon_l[0] - bcon_l[0] * ucon_l[IVZ];

    // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
    MHDCons1D fl;
    qa = wtot_l * ucon_l[IVX];
    fl.d  = rho_l * ucon_l[IVX];
    fl.e  = qa * ucov_l[0] - bcon_l[IVX] * bcov_l[0];
    fl.mx = qa * ucov_l[1] - bcon_l[IVX] * bcov_l[1] + ptot_l;
    fl.my = qa * ucov_l[2] - bcon_l[IVX] * bcov_l[2];
    fl.mz = qa * ucov_l[3] - bcon_l[IVX] * bcov_l[3];
    fl.by = bcon_l[IVY] * ucon_l[IVX] - bcon_l[IVX] * ucon_l[IVY];
    fl.bz = bcon_l[IVZ] * ucon_l[IVX] - bcon_l[IVX] * ucon_l[IVZ];

    // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
    MHDCons1D fr;
    qa = wtot_r * ucon_r[IVX];
    fr.d  = rho_r * ucon_r[IVX];
    fr.e  = qa * ucov_r[0] - bcon_r[IVX] * bcov_r[0];
    fr.mx = qa * ucov_r[1] - bcon_r[IVX] * bcov_r[1] + ptot_r;
    fr.my = qa * ucov_r[2] - bcon_r[IVX] * bcov_r[2];
    fr.mz = qa * ucov_r[3] - bcon_r[IVX] * bcov_r[3];
    fr.by = bcon_r[IVY] * ucon_r[IVX] - bcon_r[IVX] * ucon_r[IVY];
    fr.bz = bcon_r[IVZ] * ucon_r[IVX] - bcon_r[IVX] * ucon_r[IVZ];

    // Calculate fluxes in HLL region
    MHDCons1D flux_hll;
    qa = lambda_r*lambda_l;
    qb = lambda_r - lambda_l;
    flux_hll.d  = (lambda_r*fl.d  - lambda_l*fr.d  + qa*du.d ) / qb;
    flux_hll.e  = (lambda_r*fl.mx - lambda_l*fr.mx + qa*du.mx) / qb;
    flux_hll.mx = (lambda_r*fl.my - lambda_l*fr.my + qa*du.my) / qb;
    flux_hll.my = (lambda_r*fl.mz - lambda_l*fr.mz + qa*du.mz) / qb;
    flux_hll.mz = (lambda_r*fl.e  - lambda_l*fr.e  + qa*du.e ) / qb;
    flux_hll.by = (lambda_r*fl.by - lambda_l*fr.by + qa*du.by) / qb;
    flux_hll.bz = (lambda_r*fl.bz - lambda_l*fr.bz + qa*du.bz) / qb;

    // Determine region of wavefan
    MHDCons1D *flux_interface;
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

    ey(m,k,j,i) = -flux_interface->by;
    ez(m,k,j,i) = flux_interface->bz;
  });

  return;
}
