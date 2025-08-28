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
    const DvceArray5D<Real> &q, const Real &dx, const int dim) {
    Real q_p = q(m,n,k+(dim==3),j+(dim==2),i+(dim==1));
    Real q_m = q(m,n,k-(dim==3),j-(dim==2),i-(dim==1));
    return (q_p - 2.0*q(m,n,k,j,i) + q_m)/(dx*dx);
  }

//----------------------------------------------------------------------------------------
//! \fn LaplacianX1()
//! \brief Wrapper function for Laplacian in x1-direction.
//! This function should be called over [is-1,ie+1] to get Laplacian over [is,ie]
KOKKOS_INLINE_FUNCTION
Real LaplacianX1(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const Real &dx) {
    return Laplacian(m,n,k,j,i,q,dx,1);
  }

//----------------------------------------------------------------------------------------
//! \fn LaplacianX2()
//! \brief Wrapper function for Laplacian in x2-direction.
//! This function should be called over [js-1,je+1] to get Laplacian over [js,je]
KOKKOS_INLINE_FUNCTION
Real LaplacianX2(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const Real &dx) {
    return Laplacian(m,n,k,j,i,q,dx,2);
  }

//----------------------------------------------------------------------------------------
//! \fn LaplacianX3()
//! \brief Wrapper function for Laplacian in x3-direction.
//! This function should be called over [ks-1,ke+1] to get Laplacian over [ks,ke]
KOKKOS_INLINE_FUNCTION
Real LaplacianX3(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const Real &dx) {
    return Laplacian(m,n,k,j,i,q,dx,3);
  }

KOKKOS_INLINE_FUNCTION
Real Laplacian2D(const int m, const int n, const int k, const int j, const int i,
    const DvceArray5D<Real> &q, const Real &dx1, const Real &dx2, const Real &dx3,
    const int dim) {
      Real delta_x1, delta_x2;
      if(dim == 1){
        delta_x1 = Laplacian(m,n,k,j,i,q,dx2,2);
        delta_x2 = Laplacian(m,n,k,j,i,q,dx3,3);
      }
      if(dim == 2){
        delta_x1 = Laplacian(m,n,k,j,i,q,dx3,3);
        delta_x2 = Laplacian(m,n,k,j,i,q,dx1,1);
      }
      if(dim == 3){
        delta_x1 = Laplacian(m,n,k,j,i,q,dx1,1);
        delta_x2 = Laplacian(m,n,k,j,i,q,dx2,2);
      }
      return delta_x1 + delta_x2;
  }
#endif // COORDINATES_LAPLACIAN_HPP_