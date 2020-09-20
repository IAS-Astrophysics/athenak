//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dc.cpp
//  \brief piecewise constant (donor cell) reconstruction

#include "athena.hpp"
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief For each cell-centered value q(i), returns ql(i+1) and qr(i) over il to iu.
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

void Reconstruction::DonorCellX1(const int k, const int j, const int il, const int iu,
     const AthenaArray4D<Real> &q, AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      ql(n,i+1) = q(n,k,j,i);
      qr(n,i  ) = q(n,k,j,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief For each cell-centered value q(j), returns ql(j+1) and qr(j) over il to iu.
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

void Reconstruction::DonorCellX2(const int k, const int j, const int il, const int iu,
     const AthenaArray4D<Real> &q, AthenaArray2D<Real> &ql_jp1, AthenaArray2D<Real> &qr_j)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      ql_jp1(n,i) = q(n,k,j,i);
      qr_j  (n,i) = q(n,k,j,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief For each cell-centered value q(k), returns ql(k+1) and qr(k) over il to iu.
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

void Reconstruction::DonorCellX3(const int k, const int j, const int il, const int iu,
     const AthenaArray4D<Real> &q, AthenaArray2D<Real> &ql_kp1, AthenaArray2D<Real> &qr_k)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      ql_kp1(n,i) = q(n,k,j,i);
      qr_k  (n,i) = q(n,k,j,i);
    }
  }
  return;
}
