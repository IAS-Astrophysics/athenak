#ifndef MHD_RSOLVERS_LLF_MHD_SINGLESTATE_HPP_
#define MHD_RSOLVERS_LLF_MHD_SINGLESTATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_mhd_singlestate.hpp
//! \brief various Local Lax Friedrichs (LLF) Riemann solvers, also known as Rusanov's
//! method, for NR/SR/GR MHD. This flux is very diffusive, even more diffusive than HLLE,
//! and so it is not recommended for use in applications.  However, it is useful for
//! testing, or for problems where other Riemann solvers fail.
//!
//! Each solver in this file works on a single L/R state
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.

#include "coordinates/cartesian_ks.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_MHD
//! \brief The LLF Riemann solver for MHD for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_MHD(const MHDPrim1D &wl, const MHDPrim1D &wr, const Real &bxi,
                        const EOS_Data &eos, MHDCons1D &flux) {
  const Real igm1 = 1.0/(eos.gamma - 1.0);

  // Compute sum of L/R fluxes
  Real qa = wl.d*wl.vx;
  Real qb = wr.d*wr.vx;
  Real qc = 0.5*(SQR(wl.by) + SQR(wl.bz) - SQR(bxi));
  Real qd = 0.5*(SQR(wr.by) + SQR(wr.bz) - SQR(bxi));

  MHDCons1D fsum;
  fsum.d  = qa       + qb;
  fsum.mx = qa*wl.vx + qb*wr.vx + qc + qd;
  fsum.my = qa*wl.vy + qb*wr.vy - bxi*(wl.by + wr.by);
  fsum.mz = qa*wl.vz + qb*wr.vz - bxi*(wl.bz + wr.bz);
  fsum.by = wl.by*wl.vx + wr.by*wr.vx - bxi*(wl.vy + wr.vy);
  fsum.bz = wl.bz*wl.vx + wr.bz*wr.vx - bxi*(wl.vz + wr.vz);

  Real el,er;
  if (eos.is_ideal) {
    el = wl.p*igm1 + 0.5*wl.d*(SQR(wl.vx)+SQR(wl.vy)+SQR(wl.vz)) + qc + SQR(bxi);
    er = wr.p*igm1 + 0.5*wr.d*(SQR(wr.vx)+SQR(wr.vy)+SQR(wr.vz)) + qd + SQR(bxi);
    fsum.mx += (wl.p + wr.p);
    fsum.e  = (el + wl.p + qc)*wl.vx + (er + wr.p + qd)*wr.vx;
    fsum.e  -= bxi*(wl.by*wl.vy + wl.bz*wl.vz);
    fsum.e  -= bxi*(wr.by*wr.vy + wr.bz*wr.vz);
  } else {
    fsum.mx += SQR(eos.iso_cs)*(wl.d + wr.d);
  }

  // Compute max wave speed in L,R states (see Toro eq. 10.43)
  if (eos.is_ideal) {
    qa = eos.IdealMHDFastSpeed(wl.d, wl.p, bxi, wl.by, wl.bz);
    qb = eos.IdealMHDFastSpeed(wr.d, wr.p, bxi, wr.by, wr.bz);
  } else {
    qa = eos.IdealMHDFastSpeed(wl.d, bxi, wl.by, wl.bz);
    qb = eos.IdealMHDFastSpeed(wr.d, bxi, wr.by, wr.bz);
  }
  Real a = fmax( (fabs(wl.vx) + qa), (fabs(wr.vx) + qb) );

  // Compute difference in L/R states dU, multiplied by max wave speed
  MHDCons1D du;
  du.d  = a*(wr.d       - wl.d);
  du.mx = a*(wr.d*wr.vx - wl.d*wl.vx);
  du.my = a*(wr.d*wr.vy - wl.d*wl.vy);
  du.mz = a*(wr.d*wr.vz - wl.d*wl.vz);
  if (eos.is_ideal) du.e = a*(er - el);
  du.by = a*(wr.by - wl.by);
  du.bz = a*(wr.bz - wl.bz);

  // Compute the LLF flux at interface (see Toro eq. 10.42).
  flux.d  = 0.5*(fsum.d  - du.d);
  flux.mx = 0.5*(fsum.mx - du.mx);
  flux.my = 0.5*(fsum.my - du.my);
  flux.mz = 0.5*(fsum.mz - du.mz);
  if (eos.is_ideal) {flux.e = 0.5*(fsum.e  - du.e);}
  flux.by = -0.5*(fsum.by - du.by);
  flux.bz =  0.5*(fsum.bz - du.bz);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_SRMHD
//! \brief The LLF Riemann solver for SR MHD for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_SRMHD(const MHDPrim1D &wl, const MHDPrim1D &wr, const Real &bxi,
                          const EOS_Data &eos, MHDCons1D &flux) {
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  // Calculate 4-magnetic field in left state
  Real gam_l = sqrt(1.0 + SQR(wl.vx) + SQR(wl.vy) + SQR(wl.vz));
  Real b_l[4];
  b_l[0] = bxi*wl.vx + wl.by*wl.vy + wl.bz*wl.vz;
  b_l[1] = (bxi   + b_l[0] * wl.vx) / gam_l;
  b_l[2] = (wl.by + b_l[0] * wl.vy) / gam_l;
  b_l[3] = (wl.bz + b_l[0] * wl.vz) / gam_l;
  Real b_sq_l = -SQR(b_l[0]) + SQR(b_l[1]) + SQR(b_l[2]) + SQR(b_l[3]);

  // Calculate 4-magnetic field in right state
  Real gam_r = sqrt(1.0 + SQR(wr.vx) + SQR(wr.vy) + SQR(wr.vz));
  Real b_r[4];
  b_r[0] = bxi*wr.vx + wr.by*wr.vy + wr.bz*wr.vz;
  b_r[1] = (bxi   + b_r[0] * wr.vx) / gam_r;
  b_r[2] = (wr.by + b_r[0] * wr.vy) / gam_r;
  b_r[3] = (wr.bz + b_r[0] * wr.vz) / gam_r;
  Real b_sq_r = -SQR(b_r[0]) + SQR(b_r[1]) + SQR(b_r[2]) + SQR(b_r[3]);

  // Calculate left wavespeeds
  Real lm_l, lp_l;
  eos.IdealSRMHDFastSpeeds(wl.d, wl.p, wl.vx, gam_l, b_sq_l, lp_l, lm_l);

  // Calculate right wavespeeds
  Real lm_r, lp_r;
  eos.IdealSRMHDFastSpeeds(wr.d, wr.p, wr.vx, gam_r, b_sq_r, lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);  // (MB 55)
  Real lambda_r = fmax(lp_l, lp_r);  // (MB 55)
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate conserved quantities in L region (MUB 8)
  MHDCons1D consl;
  Real wgas_l = wl.d + gamma_prime * wl.p;
  Real wtot_l = wgas_l + b_sq_l;
  Real ptot_l = wl.p + 0.5*b_sq_l;
  consl.d  = wl.d * gam_l;
  consl.e  = wtot_l * gam_l * gam_l - b_l[0] * b_l[0] - ptot_l;
  consl.mx = wtot_l * wl.vx * gam_l - b_l[1] * b_l[0];
  consl.my = wtot_l * wl.vy * gam_l - b_l[2] * b_l[0];
  consl.mz = wtot_l * wl.vz * gam_l - b_l[3] * b_l[0];
  consl.by = b_l[2] * gam_l - b_l[0] * wl.vy;
  consl.bz = b_l[3] * gam_l - b_l[0] * wl.vz;

  // Calculate fluxes in L region (MUB 15)
  MHDCons1D fl;
  fl.d  = wl.d * wl.vx;
  fl.e  = wtot_l * gam_l * wl.vx - b_l[0] * b_l[1];
  fl.mx = wtot_l * wl.vx * wl.vx - b_l[1] * b_l[1] + ptot_l;
  fl.my = wtot_l * wl.vy * wl.vx - b_l[2] * b_l[1];
  fl.mz = wtot_l * wl.vz * wl.vx - b_l[3] * b_l[1];
  fl.by = b_l[2] * wl.vx - b_l[1] * wl.vy;
  fl.bz = b_l[3] * wl.vx - b_l[1] * wl.vz;

  // Calculate conserved quantities in R region (MUB 8)
  MHDCons1D consr;
  Real wgas_r = wr.d + gamma_prime * wr.p;
  Real wtot_r = wgas_r + b_sq_r;
  Real ptot_r = wr.p + 0.5*b_sq_r;
  consr.d  = wr.d * gam_r;
  consr.e  = wtot_r * gam_r * gam_r - b_r[0] * b_r[0] - ptot_r;
  consr.mx = wtot_r * wr.vx * gam_r - b_r[1] * b_r[0];
  consr.my = wtot_r * wr.vy * gam_r - b_r[2] * b_r[0];
  consr.mz = wtot_r * wr.vz * gam_r - b_r[3] * b_r[0];
  consr.by = b_r[2] * gam_r - b_r[0] * wr.vy;
  consr.bz = b_r[3] * gam_r - b_r[0] * wr.vz;

  // Calculate fluxes in R region (MUB 15)
  MHDCons1D fr;
  fr.d  = wr.d * wr.vx;
  fr.e  = wtot_r * gam_r * wr.vx - b_r[0] * b_r[1];
  fr.mx = wtot_r * wr.vx * wr.vx - b_r[1] * b_r[1] + ptot_r;
  fr.my = wtot_r * wr.vy * wr.vx - b_r[2] * b_r[1];
  fr.mz = wtot_r * wr.vz * wr.vx - b_r[3] * b_r[1];
  fr.by = b_r[2] * wr.vx - b_r[1] * wr.vy;
  fr.bz = b_r[3] * wr.vx - b_r[1] * wr.vz;

  // Compute the LLF flux at the interface
  flux.d  = 0.5 * (fl.d  + fr.d  - lambda * (consr.d  - consl.d ));
  flux.e  = 0.5 * (fl.e  + fr.e  - lambda * (consr.e  - consl.e ));
  flux.mx = 0.5 * (fl.mx + fr.mx - lambda * (consr.mx - consl.mx));
  flux.my = 0.5 * (fl.my + fr.my - lambda * (consr.my - consl.my));
  flux.mz = 0.5 * (fl.mz + fr.mz - lambda * (consr.mz - consl.mz));
  flux.by = -0.5 * (fl.by + fr.by - lambda * (consr.by - consl.by));
  flux.bz =  0.5 * (fl.bz + fr.bz - lambda * (consr.bz - consl.bz));

  // We evolve tau = E - D
  flux.e -= flux.d;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_GRMHD
//! \brief The LLF Riemann solver for GR MHD for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_GRMHD(const MHDPrim1D wl, const MHDPrim1D wr, const Real bx,
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

  // Compute the LLF flux at the interface
  flux.d  = 0.5 * (fl.d  + fr.d  - lambda * (consr.d  - consl.d ));
  flux.e  = 0.5 * (fl.e  + fr.e  - lambda * (consr.e  - consl.e ));
  flux.mx = 0.5 * (fl.mx + fr.mx - lambda * (consr.mx - consl.mx));
  flux.my = 0.5 * (fl.my + fr.my - lambda * (consr.my - consl.my));
  flux.mz = 0.5 * (fl.mz + fr.mz - lambda * (consr.mz - consl.mz));
  flux.by = -0.5 * (fl.by + fr.by - lambda * (consr.by - consl.by));
  flux.bz =  0.5 * (fl.bz + fr.bz - lambda * (consr.bz - consl.bz));

  // We evolve tau = T^t_t + D
  flux.e  += flux.d;
  return;
}

} // namespace mhd
#endif // MHD_RSOLVERS_LLF_MHD_SINGLESTATE_HPP_
