#ifndef SHEARING_BOX_REMAP_FLUXES_HPP_
#define SHEARING_BOX_REMAP_FLUXES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_fluxes.hpp
//! \brief Inline functions to compute "fluxes" for conservative remap in both orbital
//! advection and shearing box BCs. Based on RemapFlux functions in athena4.2.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn DC_RemapFlx()

KOKKOS_INLINE_FUNCTION
void DC_RemapFlx(TeamMember_t const &tmember, const int jl, const int ju, const Real eps,
const ScrArray1D<Real> &u, ScrArray1D<Real> &ust) {
  if (eps > 0.0) {
    par_for_inner(tmember, jl, ju, [&](const int j) {
      ust(j) = eps*u(j-1);
    });
    tmember.team_barrier();
  } else {
    par_for_inner(tmember, jl, ju, [&](const int j) {
      ust(j) = eps*u(j);
    });
    tmember.team_barrier();
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PLM_RemapFlx()

KOKKOS_INLINE_FUNCTION
void PLM_RemapFlx(TeamMember_t const &tmember, const int jl, const int ju, const Real eps,
const ScrArray1D<Real> &u, ScrArray1D<Real> &ust) {
  // compute upwind state (U_star)
  if (eps > 0.0) {
    par_for_inner(tmember, jl, ju, [&](const int j) {
      Real dql = u(j-1) - u(j-2);
      Real dqr = u(j  ) - u(j-1);
      // Apply limiter
      Real dq2 = dql*dqr;
      Real dqm = 2.0*dq2/(dql + dqr);
      if (dq2 <= 0.0) dqm = 0.0;
      ust(j) = eps*(u(j-1) + 0.5*(1.0 - eps)*dqm);
    });
    tmember.team_barrier();
  } else {
    par_for_inner(tmember, jl, ju, [&](const int j) {
      Real dql = u(j  ) - u(j-1);
      Real dqr = u(j+1) - u(j  );
      // Apply limiter
      Real dq2 = dql*dqr;
      Real dqm = 2.0*dq2/(dql + dqr);
      if (dq2 <= 0.0) dqm = 0.0;
      ust(j) = eps*(u(j) - 0.5*(1.0 + eps)*dqm);
    });
    tmember.team_barrier();
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn WENOZ_RemapFlx()

KOKKOS_INLINE_FUNCTION
void WENOZ_RemapFlx(TeamMember_t const &tmember, const int jl, const int ju,
const Real eps, const ScrArray1D<Real> &u, ScrArray1D<Real> &ust) {
  const Real beta_coeff[2]{13. / 12., 0.25};
  const Real epsL = 1.0e-42;
  par_for_inner(tmember, jl-1, ju, [&](const int j) {
    // Smooth WENO weights: Note that these are from Del Zanna et al. 2007 (A.18)
    Real beta[3];
    beta[0] = beta_coeff[0] * SQR(u(j-2) +     u(j) - 2.0*u(j-1)) +
              beta_coeff[1] * SQR(u(j-2) + 3.0*u(j) - 4.0*u(j-1));

    beta[1] = beta_coeff[0] * SQR(u(j-1) + u(j+1) - 2.0*u(j)) +
              beta_coeff[1] * SQR(u(j-1) - u(j+1));

    beta[2] = beta_coeff[0] * SQR(u(j+2) +     u(j) - 2.0*u(j+1)) +
              beta_coeff[1] * SQR(u(j+2) + 3.0*u(j) - 4.0*u(j+1));

    // WENO-Z+: Acker et al. 2016
    const Real tau_5 = fabs(beta[0] - beta[2]);

    Real indicator[3];
    indicator[0] = tau_5 / (beta[0] + epsL);
    indicator[1] = tau_5 / (beta[1] + epsL);
    indicator[2] = tau_5 / (beta[2] + epsL);

    // compute F-
    // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
    Real f[3];
    if (eps > 0.0) {
      f[0] = ( 2.0*u(j-2) - 7.0*u(j-1) + 11.0*u(j)  );
      f[1] = (-1.0*u(j-1) + 5.0*u(j)   + 2.0 *u(j+1));
      f[2] = ( 2.0*u(j)   + 5.0*u(j+1) -      u(j+2));

      Real alpha[3];
      alpha[0] = 0.1*(1.0 + SQR(indicator[0]));
      alpha[1] = 0.6*(1.0 + SQR(indicator[1]));
      alpha[2] = 0.3*(1.0 + SQR(indicator[2]));
      Real alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

      ust(j+1) = eps*(f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;
    } else {
      // F+ is mirror symmetric
      f[0] = ( 2.0*u(j+2) - 7.0*u(j+1) + 11.0*u(j)  );
      f[1] = (-1.0*u(j+1) + 5.0*u(j)   + 2.0 *u(j-1));
      f[2] = ( 2.0*u(j)   + 5.0*u(j-1) -      u(j-2));

      Real alpha[3];
      alpha[0] = 0.1*(1.0 + SQR(indicator[2]));
      alpha[1] = 0.6*(1.0 + SQR(indicator[1]));
      alpha[2] = 0.3*(1.0 + SQR(indicator[0]));
      Real alpha_sum = 6.0*(alpha[0] + alpha[1] + alpha[2]);

      ust(j) = eps*(f[0]*alpha[0] + f[1]*alpha[1] + f[2]*alpha[2])/alpha_sum;
    }
  });
  return;
}

#endif // SHEARING_BOX_REMAP_FLUXES_HPP_
