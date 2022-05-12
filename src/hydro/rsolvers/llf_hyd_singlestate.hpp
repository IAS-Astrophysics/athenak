#ifndef HYDRO_RSOLVERS_LLF_HYD_SINGLESTATE_HPP_
#define HYDRO_RSOLVERS_LLF_HYD_SINGLESTATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_hyd_singlestate.hpp
//! \brief various Local Lax Friedrichs (LLF) Riemann solvers, also known as Rusanov's
//! method, for NR/SR/GR hydrodynamics.  This flux is very diffusive, even more diffusive
//! than HLLE, and so it is not recommended for use in applications.  However, it is
//! useful for testing, or for problems where other Riemann solvers fail.
//!
//! Each solver in this file works on a single L/R state
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.

#include "coordinates/cartesian_ks.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_HYD
//  \brief The LLF Riemann solver for hydrodynamics for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_Hyd(const HydPrim1D &wl, const HydPrim1D &wr, const EOS_Data &eos,
                        HydCons1D &flux) {
  const Real igm1 = 1.0/(eos.gamma - 1.0);

  // Compute sum of L/R fluxes
  Real qa = wl.d*wl.vx;
  Real qb = wr.d*wr.vx;

  HydCons1D fsum;
  fsum.d  = qa        + qb;
  fsum.mx = qa*wl.vx + qb*wr.vx;
  fsum.my = qa*wl.vy + qb*wr.vy;
  fsum.mz = qa*wl.vz + qb*wr.vz;

  Real el,er;
  if (eos.is_ideal) {
    el = wl.p*igm1 + 0.5*wl.d*(SQR(wl.vx) + SQR(wl.vy) + SQR(wl.vz));
    er = wr.p*igm1 + 0.5*wr.d*(SQR(wr.vx) + SQR(wr.vy) + SQR(wr.vz));
    fsum.mx += (wl.p + wr.p);
    fsum.e  = (el + wl.p)*wl.vx + (er + wr.p)*wr.vx;
  } else {
    fsum.mx += SQR(eos.iso_cs)*(wl.d + wr.d);
  }

  // Compute max wave speed in L,R states (see Toro eq. 10.43)
  if (eos.is_ideal) {
    qa = eos.IdealHydroSoundSpeed(wl.d, wl.p);
    qb = eos.IdealHydroSoundSpeed(wr.d, wr.p);
  } else {
    qa = eos.iso_cs;
    qb = eos.iso_cs;
  }
  Real a = fmax( (fabs(wl.vx) + qa), (fabs(wr.vx) + qb) );

  // Compute difference in L/R states dU, multiplied by max wave speed
  HydCons1D du;
  du.d  = a*(wr.d       - wl.d);
  du.mx = a*(wr.d*wr.vx - wl.d*wl.vx);
  du.my = a*(wr.d*wr.vy - wl.d*wl.vy);
  du.mz = a*(wr.d*wr.vz - wl.d*wl.vz);
  if (eos.is_ideal) du.e = a*(er - el);

  // Compute the LLF flux at interface (see Toro eq. 10.42).
  flux.d  = 0.5*(fsum.d  - du.d );
  flux.mx = 0.5*(fsum.mx - du.mx);
  flux.my = 0.5*(fsum.my - du.my);
  flux.mz = 0.5*(fsum.mz - du.mz);
  if (eos.is_ideal) {flux.e = 0.5*(fsum.e - du.e);}

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_SRHyd
//  \brief The LLF Riemann solver for SR hydrodynamics for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_SRHyd(const HydPrim1D &wl, const HydPrim1D &wr, const EOS_Data &eos,
                        HydCons1D &flux) {
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  // Recall in SR the primitive variables are (\rho, u^i, P_g), where
  //  \rho is the mass density in the comoving/fluid frame,
  //  u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is the 3-vel),
  //  P_g is the pressure.

  Real u0l  = sqrt(1.0 + SQR(wl.vz) + SQR(wl.vy) + SQR(wl.vx)); // Lorentz fact in L
  Real u0r  = sqrt(1.0 + SQR(wr.vz) + SQR(wr.vy) + SQR(wr.vx)); // Lorentz fact in R
  // FIXME ERM: Ideal fluid for now
  Real wgas_l = wl.d + gamma_prime * wl.p;  // total enthalpy in L-state
  Real wgas_r = wr.d + gamma_prime * wr.p;  // total enthalpy in R-state

  // Compute wave speeds in L,R states (see Toro eq. 10.43)
  Real lp_l, lm_l;
  eos.IdealSRHydroSoundSpeeds(wl.d, wl.p, wl.vx, u0l, lp_l, lm_l);

  Real lp_r, lm_r;
  eos.IdealSRHydroSoundSpeeds(wr.d, wr.p, wr.vx, u0r, lp_r, lm_r);

  Real qa = fmax(-fmin(lm_l,lm_r), 0.0);
  Real a = fmax(fmax(lp_l,lp_r), qa);

  // Compute sum of L/R fluxes
  qa = wgas_l * wl.vx;
  Real qb = wgas_r * wr.vx;

  HydCons1D fsum;
  fsum.d  = wl.d * wl.vx + wr.d * wr.vx;
  fsum.mx = qa*wl.vx + qb*wr.vx + (wl.p + wr.p);
  fsum.my = qa*wl.vy + qb*wr.vy;
  fsum.mz = qa*wl.vz + qb*wr.vz;
  fsum.e  = qa*u0l + qb*u0r;

  // Compute difference dU = U_R - U_L multiplied by max wave speed
  HydCons1D du;
  qa = wgas_r*u0r;
  qb = wgas_l*u0l;
  Real er = qa*u0r - wr.p;
  Real el = qb*u0l - wl.p;
  du.d  = a*(u0r*wr.d  - u0l*wl.d);
  du.mx = a*( qa*wr.vx -  qb*wl.vx);
  du.my = a*( qa*wr.vy -  qb*wl.vy);
  du.mz = a*( qa*wr.vz -  qb*wl.vz);
  du.e  = a*(er - el);

  // Compute the LLF flux at the interface
  flux.d  = 0.5*(fsum.d  - du.d );
  flux.mx = 0.5*(fsum.mx - du.mx);
  flux.my = 0.5*(fsum.my - du.my);
  flux.mz = 0.5*(fsum.mz - du.mz);
  flux.e  = 0.5*(fsum.e  - du.e );

  // We evolve tau = E - D
  flux.e -= flux.d;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_GRHyd
//! \brief The LLF Riemann solver for GR hydrodynamics for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_GRHyd(const HydPrim1D wl, const HydPrim1D wr,
                       const Real x1v, const Real x2v, const Real x3v, const int ivx,
                       const CoordData &coord, const EOS_Data &eos, HydCons1D &flux) {
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
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate conserved quantities in left state (rho u^0 and T^0_\mu)
  HydCons1D consl;
  Real wgas_l = rho_l + gamma_prime * pgas_l;
  Real qa = wgas_l * ucon_l[0];
  consl.d  = rho_l * ucon_l[0];
  consl.e  = qa * ucov_l[0] + pgas_l;
  consl.mx = qa * ucov_l[1];
  consl.my = qa * ucov_l[2];
  consl.mz = qa * ucov_l[3];

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  HydCons1D fl;
  qa = wgas_l * ucon_l[ivx];
  fl.d  = rho_l * ucon_l[ivx];
  fl.e  = qa * ucov_l[0];
  fl.mx = qa * ucov_l[1];
  fl.my = qa * ucov_l[2];
  fl.mz = qa * ucov_l[3];

  // Calculate conserved quantities in right state (rho u^0 and T^0_\mu)
  HydCons1D consr;
  Real wgas_r = rho_r + gamma_prime * pgas_r;
  qa = wgas_r * ucon_r[0];
  consr.d  = rho_r * ucon_r[0];
  consr.e  = qa * ucov_r[0] + pgas_r;
  consr.mx = qa * ucov_r[1];
  consr.my = qa * ucov_r[2];
  consr.mz = qa * ucov_r[3];

  // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
  HydCons1D fr;
  qa = wgas_r * ucon_r[ivx];
  fr.d  = rho_r * ucon_r[ivx];
  fr.e  = qa * ucov_r[0];
  fr.mx = qa * ucov_r[1];
  fr.my = qa * ucov_r[2];
  fr.mz = qa * ucov_r[3];

  if (ivx == IVX) {
    fl.mx += pgas_l;
    fr.mx += pgas_r;
  } else if (ivx == IVY) {
    fl.my += pgas_l;
    fr.my += pgas_r;
  } else {
    fl.mz += pgas_l;
    fr.mz += pgas_r;
  }

  // Compute the LLF flux at the interface
  flux.d  = 0.5 * (fl.d  + fr.d  - lambda * (consr.d  - consl.d ));
  flux.e  = 0.5 * (fl.e  + fr.e  - lambda * (consr.e  - consl.e ));
  flux.mx = 0.5 * (fl.mx + fr.mx - lambda * (consr.mx - consl.mx));
  flux.my = 0.5 * (fl.my + fr.my - lambda * (consr.my - consl.my));
  flux.mz = 0.5 * (fl.mz + fr.mz - lambda * (consr.mz - consl.mz));

  // We evolve tau = T^t_t + D
  flux.e  += flux.d;
  return;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_HYD_SINGLESTATE_HPP_
