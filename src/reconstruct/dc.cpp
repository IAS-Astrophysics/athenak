//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dc.cpp
//  \brief piecewise constant (donor cell) reconstruction

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief reconstruct L/R surfaces of the i-th cells

void DonorCell::ReconstructX1(const int il, const int iu, const AthenaArray<Real> &w,
                              AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  int nvar = w.GetDim(2);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      wl(n,i+1) = w(n,i);
      wr(n,i  ) = w(n,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief


void DonorCell::ReconstructX2(const int il, const int iu, const AthenaArray<Real> &w,
                              AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  int nvar = w.GetDim(2);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      wl(n,i) = w(n,i);
      wr(n,i) = w(n,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief

void DonorCell::ReconstructX3(const int il, const int iu, const AthenaArray<Real> &w,
                                 AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  int nvar = w.GetDim(2);
  for (int n=0; n<nvar; ++n) {
    for (int i=il; i<=iu; ++i) {
      wl(n,i) = w(n,i);
      wr(n,i) = w(n,i);
    }
  }
  return;
}
