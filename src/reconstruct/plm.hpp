#ifndef RECONSTRUCT_PLM_HPP_
#define RECONSTRUCT_PLM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm.hpp
//! \brief  piecewise linear reconstruction implemented as inline functions
//! This version only works with uniform mesh spacing

#include <math.h>
#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn PLM()
//! \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1,
         Real &ql_ip1, Real &qr_i) {
  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
  Real dq2 = dql*dqr;
  Real dqm = dq2/(dql + dqr);
  //Real dqm = 0.5*fmin(fabs(dql), fabs(dqr));
  if (dq2 <= 0.0) dqm = 0.0;

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm;
  qr_i   = q_i - dqm;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX1()
//! \brief Wrapper function for PLM reconstruction in x1-direction.
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX1(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      PLM(q(m,n,k,j,i-1), q(m,n,k,j,i), q(m,n,k,j,i+1), ql(n,i+1), qr(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearWBX1()
//! \brief Wrapper function for well-balanced PLM reconstruction in x1-direction.
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearWBX1(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql, ScrArray2D<Real> &qr, ScrArray1D<Real> &pl, ScrArray1D<Real>& pr) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    if (n == IPR) {
      par_for_inner(member, il, iu, [&](const int i) {
        const Real &peqp = pl(i+1);
        const Real &peqm = pr(i);
        const Real &peq  = q(m,n,k,j,i);
        Real pep1 = Kokkos::fmax(3.*(peqp - peq) + peqm, 0.0);
        Real pem1 = Kokkos::fmax(3.*(peqm - peq) + peqp, 0.0);
        Real p0 = 0.0;
        Real pp = q(m,n,k,j,i+1) - pep1;
        Real pm = q(m,n,k,j,i-1) - pem1;
        Real left, right;
        PLM(pm, p0, pp, left, right);
        qr(n,i) = right + peqm;
        ql(n,i+1) = left + peqp;
      });
    } else {
      par_for_inner(member, il, iu, [&](const int i) {
        PLM(q(m,n,k,j,i-1), q(m,n,k,j,i), q(m,n,k,j,i+1), ql(n,i+1), qr(n,i));
      });
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX2()
//! \brief Wrapper function for PLM reconstruction in x2-direction.
//! This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX2(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      PLM(q(m,n,k,j-1,i), q(m,n,k,j,i), q(m,n,k,j+1,i), ql_jp1(n,i), qr_j(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearWBX2()
//! \brief Wrapper function for well-balanced PLM reconstruction in x2-direction.
//! This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearWBX2(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j, ScrArray1D<Real>& pl_jp1, ScrArray1D<Real>& pr_j) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    if (n == IPR) {
      par_for_inner(member, il, iu, [&](const int i) {
        const Real &peqp = pl_jp1(i);
        const Real &peqm = pr_j(i);
        const Real &peq = q(m,n,k,j,i);
        Real pep1 = Kokkos::fmax(3.*(peqp - peq) + peqm, 0.0);
        Real pem1 = Kokkos::fmax(3.*(peqm - peq) + peqp, 0.0);
        Real p0 = 0.0;
        Real pp = q(m,n,k,j+1,i) - pep1;
        Real pm = q(m,n,k,j-1,i) - pem1;
        PLM(pm, p0, pp, ql_jp1(n,i), qr_j(n,i));
        qr_j(n,i) += peqm;
        ql_jp1(n,i) += peqp;
      });
    } else {
      par_for_inner(member, il, iu, [&](const int i) {
        PLM(q(m,n,k,j-1,i), q(m,n,k,j,i), q(m,n,k,j+1,i), ql_jp1(n,i), qr_j(n,i));
      });
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX3()
//! \brief Wrapper function for PLM reconstruction in x3-direction.
//! This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX3(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i) {
      PLM(q(m,n,k-1,j,i), q(m,n,k,j,i), q(m,n,k+1,j,i), ql_kp1(n,i), qr_k(n,i));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearWBX3()
//! \brief Wrapper function for well-balanced PLM reconstruction in x3-direction.
//! This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

KOKKOS_INLINE_FUNCTION
void PiecewiseLinearWBX3(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceArray5D<Real> &q,
     ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k, ScrArray1D<Real>& pl_kp1, ScrArray1D<Real>& pr_k) {
  int nvar = q.extent_int(1);
  for (int n=0; n<nvar; ++n) {
    if (n == IPR) {
        par_for_inner(member, il, iu, [&](const int i) {
        const Real &peqp = pl_kp1(i);
        const Real &peqm = pr_k(i);
        const Real &peq = q(m,n,k,j,i);
        Real pep1 = Kokkos::fmax(3.*(peqp - peq) + peqm, 0.0);
        Real pem1 = Kokkos::fmax(3.*(peqm - peq) + peqp, 0.0);
        Real p0 = 0.0;
        Real pp = q(m,n,k+1,j,i) - pep1;
        Real pm = q(m,n,k-1,j,i) - pem1;
        PLM(pm, p0, pp, ql_kp1(n,i), qr_k(n,i));
        qr_k(n,i) += peqm;
        ql_kp1(n,i) += peqp;
      });
    } else {
      par_for_inner(member, il, iu, [&](const int i) {
        PLM(q(m,n,k-1,j,i), q(m,n,k,j,i), q(m,n,k+1,j,i), ql_kp1(n,i), qr_k(n,i));
      });
    }
  }
  return;
}

#endif // RECONSTRUCT_PLM_HPP_
