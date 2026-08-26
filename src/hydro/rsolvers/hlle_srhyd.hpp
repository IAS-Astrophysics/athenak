#ifndef HYDRO_RSOLVERS_HLLE_SRHYD_HPP_
#define HYDRO_RSOLVERS_HLLE_SRHYD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_srhyd.hpp
//! \brief HLLE Riemann solver for special relativistic hydrodynamics.  Per-face
//! implementation for the split-kernel path.
//!
//! REFERENCES
//! - implements HLLE algorithm from Mignone & Bodo 2005, MNRAS 364 126 (MB)

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn HLLE_SR<ivx>()
//! \brief HLLE flux at face (m,k,j,i) for SR. Based on HLLETransforming() in Athena++
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLE_SR(const EOS_Data &eos,
             const int m, const int k, const int j, const int i,
             const int is, const int js, const int ks,
             const DvceArray5D<Real> &wl,
             const DvceArray5D<Real> &wr,
             const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gm1 = (eos.gamma - 1.0);
  const Real gamma_prime = eos.gamma/gm1;


  // Left/right primitives.
  // Recall in SR the primitive variables are (\rho, u^i, P_g), where
  //   \rho is the mass density in the comoving/fluid frame,
  //   u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is the 3-vel),
  //   P_g is the pressure.
  Real wl_idn = wl(m,IDN,k,j,i);
  Real wl_ivx = wl(m,ivx,k,j,i);
  Real wl_ivy = wl(m,ivy,k,j,i);
  Real wl_ivz = wl(m,ivz,k,j,i);

  Real wr_idn = wr(m,IDN,k,j,i);
  Real wr_ivx = wr(m,ivx,k,j,i);
  Real wr_ivy = wr(m,ivy,k,j,i);
  Real wr_ivz = wr(m,ivz,k,j,i);

  Real wl_ipr = eos.IdealGasPressure(wl(m,IEN,k,j,i));
  Real wr_ipr = eos.IdealGasPressure(wr(m,IEN,k,j,i));

  Real u2l = SQR(wl_ivz) + SQR(wl_ivy) + SQR(wl_ivx);
  Real u2r = SQR(wr_ivz) + SQR(wr_ivy) + SQR(wr_ivx);

  Real u0l  = sqrt(1.0 + u2l);  // Lorentz factor in L-state
  Real u0r  = sqrt(1.0 + u2r);  // Lorentz factor in R-state

  // FIXME ERM: Ideal fluid for now
  Real wgas_l = wl_idn + gamma_prime * wl_ipr;  // total enthalpy in L-state
  Real wgas_r = wr_idn + gamma_prime * wr_ipr;  // total enthalpy in R-state

  // Calculate wavespeeds in left state (MB 23)
  Real lp_l, lm_l;
  eos.IdealSRHydroSoundSpeeds(wl_idn, wl_ipr, wl_ivx, u0l, lp_l, lm_l);

  // Calculate wavespeeds in right state (MB 23)
  Real lp_r, lm_r;
  eos.IdealSRHydroSoundSpeeds(wr_idn, wr_ipr, wr_ivx, u0r, lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);

  // Calculate difference dU = U_R - U_L (MB 3)
  HydCons1D du;
  Real qa = wgas_r*u0r;
  Real qb = wgas_l*u0l;
  Real er = qa*u0r - wr_ipr;
  Real el = qb*u0l - wl_ipr;

  du.d  = wr_idn*u0r - wl_idn*u0l;
  du.mx = wr_ivx*qa  - wl_ivx*qb;
  du.my = wr_ivy*qa  - wl_ivy*qb;
  du.mz = wr_ivz*qa  - wl_ivz*qb;
  du.e  = er - el;

  // Calculate fluxes in L region (MB 2,3)
  HydCons1D fl, fr;
  qa = wgas_l * wl_ivx;
  fl.d  = wl_idn * wl_ivx;
  fl.mx = qa * wl_ivx + wl_ipr;
  fl.my = qa * wl_ivy;
  fl.mz = qa * wl_ivz;
  fl.e  = qa * u0l;

  // Calculate fluxes in R region (MB 2,3)
  qa = wgas_r * wr_ivx;
  fr.d  = wr_idn * wr_ivx;
  fr.mx = qa * wr_ivx + wr_ipr;
  fr.my = qa * wr_ivy;
  fr.mz = qa * wr_ivz;
  fr.e  = qa * u0r;

  // Calculate fluxes in HLL region (MB 11)
  HydCons1D flux_hll;
  qa = lambda_r * lambda_l;
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

  // We evolve tau = E - D
  flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLE_SRHYD_HPP_
