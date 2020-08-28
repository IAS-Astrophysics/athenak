//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file plm.cpp
//  \brief  piecewise linear reconstruction implemented in a derived class
//  This version only works with uniform mesh spacing

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
// PiecewiseLinear constructor

PiecewiseLinear::PiecewiseLinear(ParameterInput *pin, int nvar, int ncells1) :
  Reconstruction(pin, nvar, ncells1)
{
  // allocate space for scratch arrays
  dql_.SetSize(ncells1_);
  dqr_.SetSize(ncells1_);
  dqm_.SetSize(ncells1_);
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i) over [il,iu]
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

void PiecewiseLinear::ReconstructX1(const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  // compute L/R slopes for each variable
  int nvar = q.GetDim(4);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      dql_(i) = (q(n,k,j,i  ) - q(n,k,j,i-1));
      dqr_(i) = (q(n,k,j,i+1) - q(n,k,j,i  ));
    }

    // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
    for (int i=il; i<=iu; ++i) {
      Real dq2 = dql_(i)*dqr_(i);
      dqm_(i) = dq2/(dql_(i) + dqr_(i));
      if (dq2 <= 0.0) dqm_(i) = 0.0;
    }

    // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
    for (int i=il; i<=iu; ++i) {
      ql(n,i+1) = q(n,k,j,i) + dqm_(i);
      qr(n,i  ) = q(n,k,j,i) - dqm_(i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief Reconstructs linear slope in cell j to cmpute ql(j+1) and qr(j) over [il,iu]
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

void PiecewiseLinear::ReconstructX2(const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q, AthenaArray<Real> &ql_jp1, AthenaArray<Real> &qr_j)
{
  // compute L/R slopes for each variable
  int nvar = q.GetDim(4);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      dql_(i) = (q(n,k,j  ,i) - q(n,k,j-1,i));
      dqr_(i) = (q(n,k,j+1,i) - q(n,k,j  ,i));
    }

    // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
    for (int i=il; i<=iu; ++i) {
      Real dq2 = dql_(i)*dqr_(i);
      dqm_(i) = dq2/(dql_(i) + dqr_(i));
      if (dq2 <= 0.0) dqm_(i) = 0.0;
    }

    // compute ql_(j+1/2) and qr_(j-1/2) using limited slopes
    for (int i=il; i<=iu; ++i) {
      ql_jp1(n,i) = q(n,k,j,i) + dqm_(i);
      qr_j(n,i)   = q(n,k,j,i) - dqm_(i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief Reconstructs linear slope in cell k to cmpute ql(k+1) and qr(k) over [il,iu]
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

void PiecewiseLinear::ReconstructX3(const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q, AthenaArray<Real> &ql_kp1, AthenaArray<Real> &qr_k)
{
  // compute L/R slopes for each variable
  int nvar = q.GetDim(4);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      dql_(i) = (q(n,k  ,j,i) - q(n,k-1,j,i));
      dqr_(i) = (q(n,k+1,j,i) - q(n,k  ,j,i));
    }

    // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
    for (int i=il; i<=iu; ++i) {
      Real dq2 = dql_(i)*dqr_(i);
      dqm_(i) = dq2/(dql_(i) + dqr_(i));
      if (dq2 <= 0.0) dqm_(i) = 0.0;
    }

    // compute ql_(k+1/2) and qr_(k-1/2) using limited slopes
    for (int i=il; i<=iu; ++i) {
      ql_kp1(n,i) = q(n,k,j,i) + dqm_(i);
      qr_k(n,i)   = q(n,k,j,i) - dqm_(i);
    }
  }
  return;
}
