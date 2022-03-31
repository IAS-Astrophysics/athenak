//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_grmhd.cpp
//! \brief LLF Riemann solver for general relativistic MHD.

#include <cmath>      // sqrt()

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF_GR
//! \brief
//

KOKKOS_INLINE_FUNCTION
void LLF_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  // Cyclic permutation of array indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  const Real gm1 = (eos.gamma - 1.0);
  const Real gamma_prime = eos.gamma/(gm1);

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
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);
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
    Real alpha = sqrt(-1.0/g00);

    // Extract left primitives.  Note 1/2/3 always refers to x1/2/3 dirs
    const Real &rho_l = wl(IDN,i);
    const Real &uu1_l = wl(IVX,i);
    const Real &uu2_l = wl(IVY,i);
    const Real &uu3_l = wl(IVZ,i);

    // Extract right primitives.  Note 1/2/3 always refers to x1/2/3 dirs
    const Real &rho_r  = wr(IDN,i);
    const Real &uu1_r  = wr(IVX,i);
    const Real &uu2_r  = wr(IVY,i);
    const Real &uu3_r  = wr(IVZ,i);

    Real pgas_l, pgas_r;
    pgas_l = eos.IdealGasPressure(wl(IEN,i));
    pgas_r = eos.IdealGasPressure(wr(IEN,i));

    // on input;
    //   bx = face-centered field in direction of slice
    //   bl/r contain bcc1, bcc2, bcc3 in IBX/IBY/IBZ components
    // extract magnetic field and metric components according to direction of slice
    Real gii, g0i;
    Real bb1_l, bb2_l, bb3_l, bb1_r, bb2_r, bb3_r;
    if (ivx == IVX) {
      gii = g11;
      g0i = g01;
      bb1_r = bx(m,k,j,i);
      bb2_r = br(IBY,i);
      bb3_r = br(IBZ,i);
      bb1_l = bx(m,k,j,i);
      bb2_l = bl(IBY,i);
      bb3_l = bl(IBZ,i);
    } else if (ivx == IVY) {
      gii = g22;
      g0i = g02;
      bb1_l = bl(IBX,i);
      bb2_l = bx(m,k,j,i);
      bb3_l = bl(IBZ,i);
      bb1_r = br(IBX,i);
      bb2_r = bx(m,k,j,i);
      bb3_r = br(IBZ,i);
    } else {
      gii = g33;
      g0i = g03;
      bb1_l = bl(IBX,i);
      bb2_l = bl(IBY,i);
      bb3_l = bx(m,k,j,i);
      bb1_r = br(IBX,i);
      bb2_r = br(IBY,i);
      bb3_r = bx(m,k,j,i);
    }

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
    eos.IdealGRMHDFastSpeeds(rho_l, pgas_l, ucon_l[0], ucon_l[ivx], b_sq_l, g00, g0i, gii,
                             lp_l, lm_l);

    // Calculate wavespeeds in right state
    Real lp_r, lm_r;
    eos.IdealGRMHDFastSpeeds(rho_r, pgas_r, ucon_r[0], ucon_r[ivx], b_sq_r, g00, g0i, gii,
                             lp_r, lm_r);

    // Calculate extremal wavespeeds
    Real lambda_l = fmin(lm_l, lm_r);
    Real lambda_r = fmax(lp_l, lp_r);
    Real lambda = fmax(lambda_r, -lambda_l);

    // Calculate conserved quantities in left state (rho u^0 and T^0_\mu)
    MHDCons1D consl;
    Real wgas_l = rho_l + gamma_prime * pgas_l;
    Real wtot_l = wgas_l + b_sq_l;
    Real ptot_l = pgas_l + 0.5*b_sq_l;
    Real qa = wtot_l * ucon_l[0];
    consl.d  = rho_l * ucon_l[0];
    consl.e  = qa * ucov_l[0] - bcon_l[0] * bcov_l[0] + ptot_l;
    consl.mx = qa * ucov_l[1] - bcon_l[0] * bcov_l[1];
    consl.my = qa * ucov_l[2] - bcon_l[0] * bcov_l[2];
    consl.mz = qa * ucov_l[3] - bcon_l[0] * bcov_l[3];
    consl.by = bcon_l[ivy] * ucon_l[0] - bcon_l[0] * ucon_l[ivy];
    consl.bz = bcon_l[ivz] * ucon_l[0] - bcon_l[0] * ucon_l[ivz];

    // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
    MHDCons1D fl;
    qa = wtot_l * ucon_l[ivx];
    fl.d  = rho_l * ucon_l[ivx];
    fl.e  = qa * ucov_l[0] - bcon_l[ivx] * bcov_l[0];
    fl.mx = qa * ucov_l[1] - bcon_l[ivx] * bcov_l[1];
    fl.my = qa * ucov_l[2] - bcon_l[ivx] * bcov_l[2];
    fl.mz = qa * ucov_l[3] - bcon_l[ivx] * bcov_l[3];
    fl.by = bcon_l[ivy] * ucon_l[ivx] - bcon_l[ivx] * ucon_l[ivy];
    fl.bz = bcon_l[ivz] * ucon_l[ivx] - bcon_l[ivx] * ucon_l[ivz];

    // Calculate conserved quantities in right state (rho u^0 and T^0_\mu)
    MHDCons1D consr;
    Real wgas_r = rho_r + gamma_prime * pgas_r;
    Real wtot_r = wgas_r + b_sq_r;
    Real ptot_r = pgas_r + 0.5*b_sq_r;
    qa = wtot_r * ucon_r[0];
    consr.d  = rho_r * ucon_r[0];
    consr.e  = qa * ucov_r[0] - bcon_r[0] * bcov_r[0] + ptot_r;
    consr.mx = qa * ucov_r[1] - bcon_r[0] * bcov_r[1];
    consr.my = qa * ucov_r[2] - bcon_r[0] * bcov_r[2];
    consr.mz = qa * ucov_r[3] - bcon_r[0] * bcov_r[3];
    consr.by = bcon_r[ivy] * ucon_r[0] - bcon_r[0] * ucon_r[ivy];
    consr.bz = bcon_r[ivz] * ucon_r[0] - bcon_r[0] * ucon_r[ivz];

    // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
    MHDCons1D fr;
    qa = wtot_r * ucon_r[ivx];
    fr.d  = rho_r * ucon_r[ivx];
    fr.e  = qa * ucov_r[0] - bcon_r[ivx] * bcov_r[0];
    fr.mx = qa * ucov_r[1] - bcon_r[ivx] * bcov_r[1];
    fr.my = qa * ucov_r[2] - bcon_r[ivx] * bcov_r[2];
    fr.mz = qa * ucov_r[3] - bcon_r[ivx] * bcov_r[3];
    fr.by = bcon_r[ivy] * ucon_r[ivx] - bcon_r[ivx] * ucon_r[ivy];
    fr.bz = bcon_r[ivz] * ucon_r[ivx] - bcon_r[ivx] * ucon_r[ivz];

    if (ivx == IVX) {
      fl.mx += ptot_l;
      fr.mx += ptot_r;
    } else if (ivx == IVY) {
      fl.my += ptot_l;
      fr.my += ptot_r;
    } else {
      fl.mz += ptot_l;
      fr.mz += ptot_r;
    }

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = 0.5 * (fl.d  + fr.d  - lambda * (consr.d  - consl.d ));
    flx(m,IEN,k,j,i) = 0.5 * (fl.e  + fr.e  - lambda * (consr.e  - consl.e ));
    flx(m,IVX,k,j,i) = 0.5 * (fl.mx + fr.mx - lambda * (consr.mx - consl.mx));
    flx(m,IVY,k,j,i) = 0.5 * (fl.my + fr.my - lambda * (consr.my - consl.my));
    flx(m,IVZ,k,j,i) = 0.5 * (fl.mz + fr.mz - lambda * (consr.mz - consl.mz));

    ey(m,k,j,i) = -0.5 * (fl.by + fr.by - lambda * (consr.by - consl.by));
    ez(m,k,j,i) =  0.5 * (fl.bz + fr.bz - lambda * (consr.bz - consl.bz));

    // We evolve tau = E - D
    flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_GR
//  \brief The LLF Riemann solver for GR MHD for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_GR(const MHDPrim1D wl, const MHDPrim1D wr, const Real bx,
                       const Real x1v, const Real x2v, const Real x3v, const int ivx,
                       const CoordData &coord, const EOS_Data &eos, MHDCons1D &flux) {
  // Cyclic permutation of array indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  Real g_[NMETRIC], gi_[NMETRIC];
  ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);
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
  Real alpha = sqrt(-1.0/g00);

  // Extract left primitives.  Note 1/2/3 always refers to x1/2/3 dirs
  const Real &rho_l = wl.d;
  const Real &uu1_l = wl.vx;
  const Real &uu2_l = wl.vy;
  const Real &uu3_l = wl.vz;
  const Real &pgas_l = wl.p;

  // Extract right primitives.  Note 1/2/3 always refers to x1/2/3 dirs
  const Real &rho_r  = wr.d;
  const Real &uu1_r  = wr.vx;
  const Real &uu2_r  = wr.vy;
  const Real &uu3_r  = wr.vz;
  const Real &pgas_r = wr.p;

  // on input;
  //   bx = face-centered field in direction of slice
  //   bl/r contain bcc1, bcc2, bcc3 in IBX/IBY/IBZ components
  // extract magnetic field and metric components according to direction of slice
  Real gii, g0i;
  Real bb1_l, bb2_l, bb3_l, bb1_r, bb2_r, bb3_r;
  if (ivx == IVX) {
    gii = g11;
    g0i = g01;
    bb1_r = bx;
    bb2_r = wr.by;
    bb3_r = wr.bz;
    bb1_l = bx;
    bb2_l = wl.by;
    bb3_l = wl.bz;
  } else if (ivx == IVY) {
    gii = g22;
    g0i = g02;
    bb1_l = wl.bz;
    bb2_l = bx;
    bb3_l = wl.by;
    bb1_r = wr.bz;
    bb2_r = bx;
    bb3_r = wr.by;
  } else {
    gii = g33;
    g0i = g03;
    bb1_l = wl.by;
    bb2_l = wl.bz;
    bb3_l = bx;
    bb1_r = wr.by;
    bb2_r = wr.bz;
    bb3_r = bx;
  }

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
  eos.IdealGRMHDFastSpeeds(rho_l, pgas_l, ucon_l[0], ucon_l[ivx], b_sq_l, g00, g0i, gii,
                           lp_l, lm_l);

  // Calculate wavespeeds in right state
  Real lp_r, lm_r;
  eos.IdealGRMHDFastSpeeds(rho_r, pgas_r, ucon_r[0], ucon_r[ivx], b_sq_r, g00, g0i, gii,
                           lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate conserved quantities in left state (rho u^0 and T^0_\mu)
  MHDCons1D consl;
  Real wgas_l = rho_l + gamma_prime * pgas_l;
  Real wtot_l = wgas_l + b_sq_l;
  Real ptot_l = pgas_l + 0.5*b_sq_l;
  Real qa = wtot_l * ucon_l[0];
  consl.d  = rho_l * ucon_l[0];
  consl.e  = qa * ucov_l[0] - bcon_l[0] * bcov_l[0] + ptot_l;
  consl.mx = qa * ucov_l[1] - bcon_l[0] * bcov_l[1];
  consl.my = qa * ucov_l[2] - bcon_l[0] * bcov_l[2];
  consl.mz = qa * ucov_l[3] - bcon_l[0] * bcov_l[3];
  consl.by = bcon_l[ivy] * ucon_l[0] - bcon_l[0] * ucon_l[ivy];
  consl.bz = bcon_l[ivz] * ucon_l[0] - bcon_l[0] * ucon_l[ivz];

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fl;
  qa = wtot_l * ucon_l[ivx];
  fl.d  = rho_l * ucon_l[ivx];
  fl.e  = qa * ucov_l[0] - bcon_l[ivx] * bcov_l[0];
  fl.mx = qa * ucov_l[1] - bcon_l[ivx] * bcov_l[1];
  fl.my = qa * ucov_l[2] - bcon_l[ivx] * bcov_l[2];
  fl.mz = qa * ucov_l[3] - bcon_l[ivx] * bcov_l[3];
  fl.by = bcon_l[ivy] * ucon_l[ivx] - bcon_l[ivx] * ucon_l[ivy];
  fl.bz = bcon_l[ivz] * ucon_l[ivx] - bcon_l[ivx] * ucon_l[ivz];

  // Calculate conserved quantities in right state (rho u^0 and T^0_\mu)
  MHDCons1D consr;
  Real wgas_r = rho_r + gamma_prime * pgas_r;
  Real wtot_r = wgas_r + b_sq_r;
  Real ptot_r = pgas_r + 0.5*b_sq_r;
  qa = wtot_r * ucon_r[0];
  consr.d  = rho_r * ucon_r[0];
  consr.e  = qa * ucov_r[0] - bcon_r[0] * bcov_r[0] + ptot_r;
  consr.mx = qa * ucov_r[1] - bcon_r[0] * bcov_r[1];
  consr.my = qa * ucov_r[2] - bcon_r[0] * bcov_r[2];
  consr.mz = qa * ucov_r[3] - bcon_r[0] * bcov_r[3];
  consr.by = bcon_r[ivy] * ucon_r[0] - bcon_r[0] * ucon_r[ivy];
  consr.bz = bcon_r[ivz] * ucon_r[0] - bcon_r[0] * ucon_r[ivz];

  // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fr;
  qa = wtot_r * ucon_r[ivx];
  fr.d  = rho_r * ucon_r[ivx];
  fr.e  = qa * ucov_r[0] - bcon_r[ivx] * bcov_r[0];
  fr.mx = qa * ucov_r[1] - bcon_r[ivx] * bcov_r[1];
  fr.my = qa * ucov_r[2] - bcon_r[ivx] * bcov_r[2];
  fr.mz = qa * ucov_r[3] - bcon_r[ivx] * bcov_r[3];
  fr.by = bcon_r[ivy] * ucon_r[ivx] - bcon_r[ivx] * ucon_r[ivy];
  fr.bz = bcon_r[ivz] * ucon_r[ivx] - bcon_r[ivx] * ucon_r[ivz];

  if (ivx == IVX) {
    fl.mx += ptot_l;
    fr.mx += ptot_r;
  } else if (ivx == IVY) {
    fl.my += ptot_l;
    fr.my += ptot_r;
  } else {
    fl.mz += ptot_l;
    fr.mz += ptot_r;
  }

  // Store results in 3D array of fluxes
  flux.d  = 0.5 * (fl.d  + fr.d  - lambda * (consr.d  - consl.d ));
  flux.e  = 0.5 * (fl.e  + fr.e  - lambda * (consr.e  - consl.e ));
  flux.mx = 0.5 * (fl.mx + fr.mx - lambda * (consr.mx - consl.mx));
  flux.my = 0.5 * (fl.my + fr.my - lambda * (consr.my - consl.my));
  flux.mz = 0.5 * (fl.mz + fr.mz - lambda * (consr.mz - consl.mz));
  flux.by = -0.5 * (fl.by + fr.by - lambda * (consr.by - consl.by));
  flux.bz =  0.5 * (fl.bz + fr.bz - lambda * (consr.bz - consl.bz));

  // We evolve tau = E - D
  flux.e  -= flux.d;
  return;
}

} // namespace mhd
