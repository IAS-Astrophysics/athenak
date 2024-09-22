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
  } else {
    par_for_inner(tmember, jl, ju, [&](const int j) {
      ust(j) = eps*u(j);
    });
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
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PPM_RemapFlx()
//! \brief third order reconstruction for conservative remap
//!  using Colella & Sekora extremum preserving algorithm (PPMX)

KOKKOS_INLINE_FUNCTION
void PPMX_RemapFlx(TeamMember_t const &tmember, const int jl, const int ju,
const Real eps, const ScrArray1D<Real> &u, ScrArray1D<Real> &ust) {
  par_for_inner(tmember, jl-1, ju, [&](const int j) {
    Real ulv=(7.0*(u(j-1)+u(j)) - (u(j-2)+u(j+1)))/12.0;
    Real d2uc = 3.0*(u(j-1) - 2.0*ulv + u(j));
    Real d2ul = (u(j-2) - 2.0*u(j-1) + u(j  ));
    Real d2ur = (u(j-1) - 2.0*u(j  ) + u(j+1));
    Real d2ulim = 0.0;
    Real lim_slope = fmin(fabs(d2ul),fabs(d2ur));
    if (d2uc > 0.0 && d2ul > 0.0 && d2ur > 0.0) {
      d2ulim = SIGN(d2uc)*fmin(1.25*lim_slope,fabs(d2uc));
    }
    if (d2uc < 0.0 && d2ul < 0.0 && d2ur < 0.0) {
      d2ulim = SIGN(d2uc)*fmin(1.25*lim_slope,fabs(d2uc));
    }
    ulv = 0.5*((u(j-1)+u(j)) - d2ulim/3.0);

    Real urv=(7.0*(u(j)+u(j+1)) - (u(j-1)+u(j+2)))/12.0;
    d2uc = 3.0*(u(j) - 2.0*urv + u(j+1));
    d2ul = (u(j-1) - 2.0*u(j  ) + u(j+1));
    d2ur = (u(j  ) - 2.0*u(j+1) + u(j+2));
    d2ulim = 0.0;
    lim_slope = fmin(fabs(d2ul),fabs(d2ur));
    if (d2uc > 0.0 && d2ul > 0.0 && d2ur > 0.0) {
      d2ulim = SIGN(d2uc)*fmin(1.25*lim_slope,fabs(d2uc));
    }
    if (d2uc < 0.0 && d2ul < 0.0 && d2ur < 0.0) {
      d2ulim = SIGN(d2uc)*fmin(1.25*lim_slope,fabs(d2uc));
    }
    urv = 0.5*((u(j)+u(j+1)) - d2ulim/3.0);

    Real qa = (urv-u(j))*(u(j)-ulv);
    Real qb = (u(j-1)-u(j))*(u(j)-u(j+1));
    if (qa <= 0.0 && qb <= 0.0) {
      Real d2u = -12.0*(u(j) - 0.5*(ulv+urv));
      d2uc = (u(j-1) - 2.0*u(j  ) + u(j+1));
      d2ul = (u(j-2) - 2.0*u(j-1) + u(j  ));
      d2ur = (u(j  ) - 2.0*u(j+1) + u(j+2));
      d2ulim = 0.0;
      lim_slope = fmin(fabs(d2ul),fabs(d2ur));
      lim_slope = fmin(fabs(d2uc),lim_slope);
      if (d2uc > 0.0 && d2ul > 0.0 && d2ur > 0.0 && d2u > 0.0) {
        d2ulim = SIGN(d2u)*fmin(1.25*lim_slope,fabs(d2u));
      }
      if (d2uc < 0.0 && d2ul < 0.0 && d2ur < 0.0 && d2u < 0.0) {
        d2ulim = SIGN(d2u)*fmin(1.25*lim_slope,fabs(d2u));
      }
      if (d2u == 0.0) {
        ulv = u(j);
        urv = u(j);
      } else {
        ulv = u(j) + (ulv - u(j))*d2ulim/d2u;
        urv = u(j) + (urv - u(j))*d2ulim/d2u;
      }
    }

    qa = (urv-u(j))*(u(j)-ulv);
    qb = urv-ulv;
    Real qc = 6.0*(u(j) - 0.5*(ulv+urv));
    if (qa <= 0.0) {
      ulv = u(j);
      urv = u(j);
    } else if ((qb*qc) > (qb*qb)) {
      ulv = 3.0*u(j) - 2.0*urv;
    } else if ((qb*qc) < -(qb*qb)) {
      urv = 3.0*u(j) - 2.0*ulv;
    }

    Real du = urv - ulv;
    Real u6 = 6.0*(u(j) - 0.5*(ulv + urv));

    if (eps > 0.0) {
      Real qx = TWO_3RDS*eps;
      ust(j+1) = eps*(urv - 0.75*qx*(du - (1.0 - qx)*u6));

    } else {         /* eps always < 0 for outer i boundary */
      Real qx = -TWO_3RDS*eps;
      ust(j  ) = eps*(ulv + 0.75*qx*(du + (1.0 - qx)*u6));
    }
  });
  return;
}

#endif // SHEARING_BOX_REMAP_FLUXES_HPP_
