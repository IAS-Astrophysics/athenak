//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dc.cpp
//  \brief piecewise constant (donor cell) reconstruction implemented in a derived class

#include <iostream>
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
// DonorCell constructor

DonorCell::DonorCell(ParameterInput *pin) : Reconstruction(pin)
{
}

//----------------------------------------------------------------------------------------
//! \fn DonorCell::ReconstructX1()
//  \brief reconstruct L/R surfaces of the i-th cells

void DonorCell::ReconstructX1(const int k, const int j, const int il, const int iu,
     const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  int nvar = q.GetDim(4);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      ql(n,i) = q(n,k,j,i-1);
      qr(n,i) = q(n,k,j,i  );
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn DonorCell::ReconstructX2()
//  \brief


void DonorCell::ReconstructX2(const int k, const int j, const int il, const int iu,
     const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  int nvar = q.GetDim(4);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      ql(n,i) = q(n,k,j-1,i);
      qr(n,i) = q(n,k,j  ,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn DonorCell::ReconstructX3()
//  \brief

void DonorCell::ReconstructX3(const int k, const int j, const int il, const int iu,
     const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  int nvar = q.GetDim(4);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      ql(n,i) = q(n,k-1,j,i);
      qr(n,i) = q(n,k  ,j,i);
    }
  }
  return;
}
