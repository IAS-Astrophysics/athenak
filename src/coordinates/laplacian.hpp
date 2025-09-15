#ifndef COORDINATES_LAPLACIAN_HPP_
#define COORDINATES_LAPLACIAN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file laplacian.hpp
//! \brief  Laplacian operator implemented as inline functions
//! This version only works with uniform mesh spacing

#include <math.h>
#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn Laplacian()
//! \brief Computes Laplacian of a scalar field in 3D with uniform mesh spacing

KOKKOS_INLINE_FUNCTION
Real Laplacian(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const int dim) {
    Real q_p = q(m,n,k+(dim==3),j+(dim==2),i+(dim==1));
    Real q_m = q(m,n,k-(dim==3),j-(dim==2),i-(dim==1));
    return (q_m - 2.0*q(m,n,k,j,i) + q_p);
  }

//----------------------------------------------------------------------------------------
//! \fn LaplacianX1()
//! \brief Wrapper function for Laplacian in x1-direction.
//! This function should be called over [is-1,ie+1] to get Laplacian over [is,ie]
KOKKOS_INLINE_FUNCTION
Real LaplacianX1(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q) {
    return Laplacian(m,n,k,j,i,q,1);
  }

//----------------------------------------------------------------------------------------
//! \fn LaplacianX2()
//! \brief Wrapper function for Laplacian in x2-direction.
//! This function should be called over [js-1,je+1] to get Laplacian over [js,je]
KOKKOS_INLINE_FUNCTION
Real LaplacianX2(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q) {
    return Laplacian(m,n,k,j,i,q,2);
  }

//----------------------------------------------------------------------------------------
//! \fn LaplacianX3()
//! \brief Wrapper function for Laplacian in x3-direction.
//! This function should be called over [ks-1,ke+1] to get Laplacian over [ks,ke]
KOKKOS_INLINE_FUNCTION
Real LaplacianX3(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q) {
    return Laplacian(m,n,k,j,i,q,3);
  }

KOKKOS_INLINE_FUNCTION
Real Laplacian3D(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const int ndim) {
      Real laplacian;
      laplacian = Laplacian(m,n,k,j,i,q,1);
      laplacian += (ndim>1) ? Laplacian(m,n,k,j,i,q,2) : 0.0;
      laplacian += (ndim>2) ? Laplacian(m,n,k,j,i,q,3) : 0.0;
      return laplacian;
  }


KOKKOS_INLINE_FUNCTION
Real Laplacian2D(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const int dim) {
      Real delta_x1, delta_x2;
      if(dim == 1){
        delta_x1 = Laplacian(m,n,k,j,i,q,2);
        delta_x2 = Laplacian(m,n,k,j,i,q,3);
      }
      if(dim == 2){
        delta_x1 = Laplacian(m,n,k,j,i,q,3);
        delta_x2 = Laplacian(m,n,k,j,i,q,1);
      }
      if(dim == 3){
        delta_x1 = Laplacian(m,n,k,j,i,q,1);
        delta_x2 = Laplacian(m,n,k,j,i,q,2);
      }
      return delta_x1 + delta_x2;
  }

KOKKOS_INLINE_FUNCTION
Real Laplacian1D(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const int dim) {
      if(dim == 1) return Laplacian(m,n,k,j,i,q,2);
      if(dim == 2) return Laplacian(m,n,k,j,i,q,1);
      return 0.0;
  }
#endif // COORDINATES_LAPLACIAN_HPP_