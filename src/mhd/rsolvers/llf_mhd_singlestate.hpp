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

  Real el,er,pl,pr;
  if (eos.is_ideal) {
    pl = eos.IdealGasPressure(wl.e);
    pr = eos.IdealGasPressure(wr.e);
    el = wl.e + 0.5*wl.d*(SQR(wl.vx)+SQR(wl.vy)+SQR(wl.vz)) + qc + SQR(bxi);
    er = wr.e + 0.5*wr.d*(SQR(wr.vx)+SQR(wr.vy)+SQR(wr.vz)) + qd + SQR(bxi);
    fsum.mx += (pl + pr);
    fsum.e  = (el + pl + qc)*wl.vx + (er + pr + qd)*wr.vx;
    fsum.e  -= bxi*(wl.by*wl.vy + wl.bz*wl.vz);
    fsum.e  -= bxi*(wr.by*wr.vy + wr.bz*wr.vz);
  } else {
    fsum.mx += SQR(eos.iso_cs)*(wl.d + wr.d);
  }

  // Compute max wave speed in L,R states (see Toro eq. 10.43)
  if (eos.is_ideal) {
    qa = eos.IdealMHDFastSpeed(wl.d, pl, bxi, wl.by, wl.bz);
    qb = eos.IdealMHDFastSpeed(wr.d, pr, bxi, wr.by, wr.bz);
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
  Real pl = eos.IdealGasPressure(wl.e);
  Real lm_l, lp_l;
  eos.IdealSRMHDFastSpeeds(wl.d, pl, wl.vx, gam_l, b_sq_l, lp_l, lm_l);

  // Calculate right wavespeeds
  Real pr = eos.IdealGasPressure(wr.e);
  Real lm_r, lp_r;
  eos.IdealSRMHDFastSpeeds(wr.d, pr, wr.vx, gam_r, b_sq_r, lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);  // (MB 55)
  Real lambda_r = fmax(lp_l, lp_r);  // (MB 55)
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate conserved quantities in L region (MUB 8)
  MHDCons1D consl;
  Real wgas_l = wl.d + eos.gamma * wl.e;
  Real wtot_l = wgas_l + b_sq_l;
  Real ptot_l = pl + 0.5*b_sq_l;
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
  Real wgas_r = wr.d + eos.gamma * wr.e;
  Real wtot_r = wgas_r + b_sq_r;
  Real ptot_r = pr + 0.5*b_sq_r;
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

  // References to left primitives
  const Real &wl_idn=wl.d;
  const Real &wl_ivx=wl.vx;
  const Real &wl_ivy=wl.vy;
  const Real &wl_ivz=wl.vz;
  const Real &wl_iby=wl.by;
  const Real &wl_ibz=wl.bz;

  // References to right primitives
  const Real &wr_idn=wr.d;
  const Real &wr_ivx=wr.vx;
  const Real &wr_ivy=wr.vy;
  const Real &wr_ivz=wr.vz;
  const Real &wr_iby=wr.by;
  const Real &wr_ibz=wr.bz;

  Real wl_ipr, wr_ipr;
  wl_ipr = eos.IdealGasPressure(wl.e);
  wr_ipr = eos.IdealGasPressure(wr.e);

  // reference to longitudinal field
  const Real &bxi = bx;

  // Extract components of metric
  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski, coord.bh_spin, glower, gupper);

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

  // lower vector indices on left state (covariant compt)
  Real ull[4];
  ull[0]   = glower[0][0]  *uul[0]   + glower[0][ivx]*uul[ivx] +
             glower[0][ivy]*uul[ivy] + glower[0][ivz]*uul[ivz];

  ull[ivx] = glower[ivx][0]  *uul[0]   + glower[ivx][ivx]*uul[ivx] +
             glower[ivx][ivy]*uul[ivy] + glower[ivx][ivz]*uul[ivz];

  ull[ivy] = glower[ivy][0]  *uul[0]   + glower[ivy][ivx]*uul[ivx] +
             glower[ivy][ivy]*uul[ivy] + glower[ivy][ivz]*uul[ivz];

  ull[ivz] = glower[ivz][0]  *uul[0]   + glower[ivz][ivx]*uul[ivx] +
             glower[ivz][ivy]*uul[ivy] + glower[ivz][ivz]*uul[ivz];

  // calculate 4-magnetic field in left state (contravariant compt)
  Real bul[4];
  bul[0]   = ull[ivx]*bxi + ull[ivy]*wl_iby + ull[ivz]*wl_ibz;
  bul[ivx] = (bxi    + bul[0] * uul[ivx]) / uul[0];
  bul[ivy] = (wl_iby + bul[0] * uul[ivy]) / uul[0];
  bul[ivz] = (wl_ibz + bul[0] * uul[ivz]) / uul[0];

  // lower vector indices (covariant compt)
  Real bll[4];
  bll[0]   = glower[0][0]  *bul[0]   + glower[0][ivx]*bul[ivx] +
             glower[0][ivy]*bul[ivy] + glower[0][ivz]*bul[ivz];

  bll[ivx] = glower[ivx][0]  *bul[0]   + glower[ivx][ivx]*bul[ivx] +
             glower[ivx][ivy]*bul[ivy] + glower[ivx][ivz]*bul[ivz];

  bll[ivy] = glower[ivy][0]  *bul[0]   + glower[ivy][ivx]*bul[ivx] +
             glower[ivy][ivy]*bul[ivy] + glower[ivy][ivz]*bul[ivz];

  bll[ivz] = glower[ivz][0]  *bul[0]   + glower[ivz][ivx]*bul[ivx] +
             glower[ivz][ivy]*bul[ivy] + glower[ivz][ivz]*bul[ivz];

  Real bsq_l = bll[0]*bul[0] + bll[ivx]*bul[ivx] + bll[ivy]*bul[ivy] +bll[ivz]*bul[ivz];

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

  // lower vector indices on right state (covariant compt)
  Real ulr[4];
  ulr[0]   = glower[0][0]  *uur[0]   + glower[0][ivx]*uur[ivx] +
             glower[0][ivy]*uur[ivy] + glower[0][ivz]*uur[ivz];

  ulr[ivx] = glower[ivx][0]  *uur[0]   + glower[ivx][ivx]*uur[ivx] +
             glower[ivx][ivy]*uur[ivy] + glower[ivx][ivz]*uur[ivz];

  ulr[ivy] = glower[ivy][0]  *uur[0]   + glower[ivy][ivx]*uur[ivx] +
             glower[ivy][ivy]*uur[ivy] + glower[ivy][ivz]*uur[ivz];

  ulr[ivz] = glower[ivz][0]  *uur[0]   + glower[ivz][ivx]*uur[ivx] +
             glower[ivz][ivy]*uur[ivy] + glower[ivz][ivz]*uur[ivz];

  // Calculate 4-magnetic field in right state
  Real bur[4];
  bur[0]   = ulr[ivx]*bxi + ulr[ivy]*wr_iby + ulr[ivz]*wr_ibz;
  bur[ivx] = (bxi    + bur[0] * uur[ivx]) / uur[0];
  bur[ivy] = (wr_iby + bur[0] * uur[ivy]) / uur[0];
  bur[ivz] = (wr_ibz + bur[0] * uur[ivz]) / uur[0];

  // lower vector indices (covariant compt)
  Real blr[4];
  blr[0]   = glower[0][0]  *bur[0]   + glower[0][ivx]*bur[ivx] +
             glower[0][ivy]*bur[ivy] + glower[0][ivz]*bur[ivz];

  blr[ivx] = glower[ivx][0]  *bur[0]   + glower[ivx][ivx]*bur[ivx] +
             glower[ivx][ivy]*bur[ivy] + glower[ivx][ivz]*bur[ivz];

  blr[ivy] = glower[ivy][0]  *bur[0]   + glower[ivy][ivx]*bur[ivx] +
             glower[ivy][ivy]*bur[ivy] + glower[ivy][ivz]*bur[ivz];

  blr[ivz] = glower[ivz][0]  *bur[0]   + glower[ivz][ivx]*bur[ivx] +
             glower[ivz][ivy]*bur[ivy] + glower[ivz][ivz]*bur[ivz];

  Real bsq_r = blr[0]*bur[0] + blr[ivx]*bur[ivx] + blr[ivy]*bur[ivy] +blr[ivz]*bur[ivz];

  // Calculate wavespeeds in left state
  Real lp_l, lm_l;
  eos.IdealGRMHDFastSpeeds(wl_idn, wl_ipr, uul[0], uul[ivx], bsq_l, gupper[0][0],
                           gupper[0][ivx], gupper[ivx][ivx], lp_l, lm_l);

  // Calculate wavespeeds in right state
  Real lp_r, lm_r;
  eos.IdealGRMHDFastSpeeds(wr_idn, wr_ipr, uur[0], uur[ivx], bsq_r, gupper[0][0],
                           gupper[0][ivx], gupper[ivx][ivx], lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
  MHDCons1D du;
  Real wtot_r = wr_idn + gamma_prime * wr_ipr + bsq_r;
  Real ptot_r = wr_ipr + 0.5*bsq_r;
  Real qa = wtot_r * uur[0];
  Real wtot_l = wl_idn + gamma_prime * wl_ipr + bsq_l;
  Real ptot_l = wl_ipr + 0.5*bsq_l;
  Real qb = wtot_l * uul[0];
  du.d  = (wr_idn*uur[0]) - (wl_idn*uul[0]);
  du.mx = (qa*ulr[ivx] - bur[0]*blr[ivx]) - (qb*ull[ivx] - bul[0]*bll[ivx]);
  du.my = (qa*ulr[ivy] - bur[0]*blr[ivy]) - (qb*ull[ivy] - bul[0]*bll[ivy]);
  du.mz = (qa*ulr[ivz] - bur[0]*blr[ivz]) - (qb*ull[ivz] - bul[0]*bll[ivz]);
  du.e  = (qa*ulr[0] - bur[0]*blr[0] + ptot_r) - (qb*ull[0] - bul[0]*bll[0] + ptot_l);
  du.by = (bur[ivy]*uur[0] - bur[0]*uur[ivy]) - (bul[ivy]*uul[0] - bul[0]*uul[ivy]);
  du.bz = (bur[ivz]*uur[0] - bur[0]*uur[ivz]) - (bul[ivz]*uul[0] - bul[0]*uul[ivz]);

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fl;
  qa = wtot_l * uul[ivx];
  fl.d  = wl_idn * uul[ivx];
  fl.mx = qa * ull[ivx] - bul[ivx] * bll[ivx] + ptot_l;
  fl.my = qa * ull[ivy] - bul[ivx] * bll[ivy];
  fl.mz = qa * ull[ivz] - bul[ivx] * bll[ivz];
  fl.e  = qa * ull[0]   - bul[ivx] * bll[0];
  fl.by = bul[ivy] * uul[ivx] - bul[ivx] * uul[ivy];
  fl.bz = bul[ivz] * uul[ivx] - bul[ivx] * uul[ivz];

  // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fr;
  qa = wtot_r * uur[ivx];
  fr.d  = wr_idn * uur[ivx];
  fr.mx = qa * ulr[ivx] - bur[ivx] * blr[ivx] + ptot_r;
  fr.my = qa * ulr[ivy] - bur[ivx] * blr[ivy];
  fr.mz = qa * ulr[ivz] - bur[ivx] * blr[ivz];
  fr.e  = qa * ulr[0]   - bur[ivx] * blr[0];
  fr.by = bur[ivy] * uur[ivx] - bur[ivx] * uur[ivy];
  fr.bz = bur[ivz] * uur[ivx] - bur[ivx] * uur[ivz];

  // Compute the LLF flux at the interface
  flux.d  =  0.5 * (fl.d  + fr.d  - lambda * du.d);
  flux.mx =  0.5 * (fl.mx + fr.mx - lambda * du.mx);
  flux.my =  0.5 * (fl.my + fr.my - lambda * du.my);
  flux.mz =  0.5 * (fl.mz + fr.mz - lambda * du.mz);
  flux.e  =  0.5 * (fl.e  + fr.e  - lambda * du.e);
  flux.by = -0.5 * (fl.by + fr.by - lambda * du.by);
  flux.bz =  0.5 * (fl.bz + fr.bz - lambda * du.bz);

  // We evolve tau = T^t_t + D
  flux.e  += flux.d;
  return;
}

} // namespace mhd
#endif // MHD_RSOLVERS_LLF_MHD_SINGLESTATE_HPP_
