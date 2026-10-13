#ifndef HYDRO_RSOLVERS_HLLC_SRHYD_HPP_
#define HYDRO_RSOLVERS_HLLC_SRHYD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hllc_srhyd.hpp
//! \brief Implements HLLC Riemann solver for special relativistic hydrodynamics.
//!
//! REFERENCES:
//!  - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!    Springer-Verlag, Berlin, (1999) chpt. 10.
//!  - Mignone & Bodo 2005, MNRAS 364 126 (MB2005)
//!  - Mignone & Bodo 2006, MNRAS 368 1040 (MB2006)

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void HLLC
//! \brief The HLLC Riemann solver for SR hydrodynamics.  Based on HLLCTransforming()
//! function in Athena++ (C++ version)

KOKKOS_INLINE_FUNCTION
void HLLC_SR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  par_for_inner(member, il, iu, [&](const int i) {
    // Create local references for L/R states (helps compiler vectorize)
    // Recall in SR the primitive variables are (\rho, u^i, P_gas), where \rho is the
    // mass density in the comoving/fluid frame, u^i = \gamma v^i are the spatial
    // components of the 4-velocity (v^i is the 3-velocity), and P_gas is the pressure.
    Real &rho_l  = wl(IDN,i);
    Real &ux_l   = wl(ivx,i);
    Real &uy_l   = wl(ivy,i);
    Real &uz_l   = wl(ivz,i);
    Real u_l[4];
    u_l[0] = sqrt(1.0 + SQR(ux_l) + SQR(uy_l) + SQR(uz_l));  // Lorentz factor in L-state
    u_l[1] = ux_l;
    u_l[2] = uy_l;
    u_l[3] = uz_l;

    // Extract right primitives
    Real &rho_r  = wr(IDN,i);
    Real &ux_r   = wr(ivx,i);
    Real &uy_r   = wr(ivy,i);
    Real &uz_r   = wr(ivz,i);
    Real u_r[4];
    u_r[0] = sqrt(1.0 + SQR(ux_r) + SQR(uy_r) + SQR(uz_r));  // Lorentz factor in R-state
    u_r[1] = ux_r;
    u_r[2] = uy_r;
    u_r[3] = uz_r;

    Real pgas_l, pgas_r;
    pgas_l = eos.IdealGasPressure(wl(IEN,i));
    pgas_r = eos.IdealGasPressure(wr(IEN,i));

    Real wgas_l = rho_l + gamma_prime * pgas_l;  // total enthalpy in L-state
    Real wgas_r = rho_r + gamma_prime * pgas_r;  // total enthalpy in R-state

    // Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real lm,lp,qa,qb;
    eos.IdealSRHydroSoundSpeeds(rho_l, pgas_l, u_l[1], u_l[0], lp, lm);
    eos.IdealSRHydroSoundSpeeds(rho_r, pgas_r, u_r[1], u_r[0], qb, qa);

    // Calculate extremal wavespeeds
    Real lambda_l = fmin(lm, qa);
    Real lambda_r = fmax(lp, qb);

    // Calculate conserved quantities in L region (MB2005 3)
    Real cons_l[5];
    cons_l[IDN] = rho_l  * u_l[0];
    cons_l[IEN] = wgas_l * u_l[0] * u_l[0] - pgas_l;
    cons_l[ivx] = wgas_l * u_l[1] * u_l[0];
    cons_l[ivy] = wgas_l * u_l[2] * u_l[0];
    cons_l[ivz] = wgas_l * u_l[3] * u_l[0];

    // Calculate fluxes in L region (MB2005 2,3)
    Real flux_l[5];
    flux_l[IDN] = rho_l  * u_l[1];
    flux_l[IEN] = wgas_l * u_l[0] * u_l[1];
    flux_l[ivx] = wgas_l * u_l[1] * u_l[1] + pgas_l;
    flux_l[ivy] = wgas_l * u_l[2] * u_l[1];
    flux_l[ivz] = wgas_l * u_l[3] * u_l[1];

    // Calculate conserved quantities in R region (MB2005 3)
    Real cons_r[5];
    cons_r[IDN] = rho_r * u_r[0];
    cons_r[IEN] = wgas_r * u_r[0] * u_r[0] - pgas_r;
    cons_r[ivx] = wgas_r * u_r[1] * u_r[0];
    cons_r[ivy] = wgas_r * u_r[2] * u_r[0];
    cons_r[ivz] = wgas_r * u_r[3] * u_r[0];

    // Calculate fluxes in R region (MB2005 2,3)
    Real flux_r[5];
    flux_r[IDN] = rho_r * u_r[1];
    flux_r[IEN] = wgas_r * u_r[0] * u_r[1];
    flux_r[ivx] = wgas_r * u_r[1] * u_r[1] + pgas_r;
    flux_r[ivy] = wgas_r * u_r[2] * u_r[1];
    flux_r[ivz] = wgas_r * u_r[3] * u_r[1];

    Real lambda_diff_inv = 1.0 / (lambda_r-lambda_l);
    // Calculate conserved quantities in HLL region in GR (MB2005 9)
    Real cons_hll[5];
    for (int n = 0; n < 5; ++n) {
      cons_hll[n] = (lambda_r*cons_r[n] - lambda_l*cons_l[n] + flux_l[n] - flux_r[n])
                    * lambda_diff_inv;
    }

    // Calculate fluxes in HLL region (MB2005 11)
    Real flux_hll[5];
    for (int n = 0; n < 5; ++n) {
      flux_hll[n] = (lambda_r*flux_l[n] - lambda_l*flux_r[n]
                     + lambda_l*lambda_r * (cons_r[n] - cons_l[n])) * lambda_diff_inv;
    }

    // Calculate contact wavespeed (MB2005 18)
    Real lambda_star;
    Real b = -(cons_hll[IEN] + flux_hll[ivx]);
    if (std::abs(flux_hll[IEN]-flux_hll[IDN]) > 1.e-12) {  // use quadratic formula
      // Follows algorithm in Numerical Recipes (section 5.6) for avoiding cancellations
      lambda_star = - 2.0 * cons_hll[ivx]
                    / (b - std::sqrt(SQR(b) - 4.0*flux_hll[IEN]*cons_hll[ivx]));
    } else { // no quadratic term
      lambda_star = - cons_hll[ivx] / b;
    }

    // Calculate contact pressure (MB2006 48)
    // Note: Could also use (MB2005 17), but note the first minus sign there is wrong.
    Real pgas_star = -flux_hll[IEN] * lambda_star + flux_hll[ivx];

    // Calculate conserved quantities in L* region (MB2005 16)
    Real cons_lstar[5];
    Real vx_l_ratio = u_l[1] / u_l[0];
    lambda_diff_inv = 1.0 / (lambda_l - lambda_star);
    for (int n = 0; n < 5; ++n) {
      cons_lstar[n] = cons_l[n] * (lambda_l-vx_l_ratio);
    }
    cons_lstar[IEN] += pgas_star*lambda_star - pgas_l*vx_l_ratio;
    cons_lstar[ivx] += pgas_star - pgas_l;
    for (int n = 0; n < 5; ++n) {
      cons_lstar[n] *= lambda_diff_inv;
    }

    // Calculate fluxes in L* region (MB2005 14)
    Real flux_lstar[5];
    for (int n = 0; n < 5; ++n) {
      flux_lstar[n] = flux_l[n] + lambda_l * (cons_lstar[n] - cons_l[n]);
    }

    // Calculate conserved quantities in R* region (MB2005 16)
    Real cons_rstar[5];
    Real vx_r_ratio = u_r[1] / u_r[0];
    lambda_diff_inv = 1.0 / (lambda_r - lambda_star);
    for (int n = 0; n < 5; ++n) {
      cons_rstar[n] = cons_r[n] * (lambda_r-vx_r_ratio);
    }
    cons_rstar[IEN] += pgas_star*lambda_star - pgas_r*vx_r_ratio;
    cons_rstar[ivx] += pgas_star - pgas_r;
    for (int n = 0; n < 5; ++n) {
      cons_rstar[n] *= lambda_diff_inv;
    }

    // Calculate fluxes in R* region (MB2005 14)
    Real flux_rstar[5];
    for (int n = 0; n < 5; ++n) {
      flux_rstar[n] = flux_r[n] + lambda_r * (cons_rstar[n] - cons_r[n]);
    }

    // Calculate interface velocity
    Real const v_interface = 0.0;

    // Determine region of wavefan
    Real *flux_interface;
    if (lambda_l >= v_interface) {  // L region
      flux_interface = flux_l;
    } else if (lambda_r <= v_interface) { // R region
      flux_interface = flux_r;
    } else if (lambda_star >= v_interface) {  // aL region
      flux_interface = flux_lstar;
    } else {  // c region
      flux_interface = flux_rstar;
    }

    // Set fluxes
    for (int n = 0; n < 5; ++n) {
      flx(m,n,k,j,i) = flux_interface[n];
    }

    // We evolve tau = E - D
    flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
  });
  return;
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLC_SRHYD_HPP_
