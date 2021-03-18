//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hllc_rel.cpp
//  \brief Implements HLLC Riemann solver for relativistic hydrodynamics.
//
//  Computes 1D fluxes using the HLLC Riemann solver.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for hydrodynamics (both adiabatic and isothermal)

KOKKOS_INLINE_FUNCTION
void HLLC_rel(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j, const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     DvceArray5D<Real> flx)
{


  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5];
  Real gm1 = eos.gamma - 1.0;
//  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Load L/R states into local variables

    Real &rho_l  = wl(IDN,i);
    Real &pgas_l = wl(IPR,i);
    Real &ux_l   = wl(ivx,i);
    Real &uy_l   = wl(ivy,i);
    Real &uz_l   = wl(ivz,i);
    Real u_l[4];
    u_l[0] = sqrt(1.0 + SQR(ux_l) + SQR(uy_l) + SQR(uz_l));
    u_l[1] = ux_l;
    u_l[2] = uy_l;
    u_l[3] = uz_l;

    // Extract right primitives
    Real &rho_r  = wr(IDN,i);
    Real &pgas_r = wr(IPR,i);
    Real &ux_r   = wr(ivx,i);
    Real &uy_r   = wr(ivy,i);
    Real &uz_r   = wr(ivz,i);
    Real u_r[4];
    u_r[0] = sqrt(1.0 + SQR(ux_r) + SQR(uy_r) + SQR(uz_r));
    u_r[1] = ux_r;
    u_r[2] = uy_r;
    u_r[3] = uz_r;

    Real wgas_l = rho_l + (gm1 +1.)/gm1 * pgas_l;
    Real wgas_r = rho_r + (gm1 +1.)/gm1 * pgas_r;

    //--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)

    Real lm,lp,qa,qb;
    eos.SoundSpeed_SR(wgas_l, pgas_l, u_l[1]/u_l[0], u_l[0]*u_l[0], lp, lm);
    eos.SoundSpeed_SR(wgas_r, pgas_r, u_r[1]/u_r[0], u_r[0]*u_r[0], qb,qa);

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
    Real *cons_interface, *flux_interface;
    if (lambda_l >= v_interface) {  // L region
      cons_interface = cons_l;
      flux_interface = flux_l;
    } else if (lambda_r <= v_interface) { // R region
      cons_interface = cons_r;
      flux_interface = flux_r;
    } else if (lambda_star >= v_interface) {  // aL region
      cons_interface = cons_lstar;
      flux_interface = flux_lstar;
    } else {  // c region
      cons_interface = cons_rstar;
      flux_interface = flux_rstar;
    }

    // Set fluxes
    for (int n = 0; n < 5; ++n) {
      flx(m,n,k,j,i) = flux_interface[n];
    }

    // We evolve tau = U - D

    flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);

  });
  return;
}

} // namespace hydro
