#ifndef EOS_PRIMITIVE_SOLVER_GEOM_MATH_HPP_
#define EOS_PRIMITIVE_SOLVER_GEOM_MATH_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geom_math.hpp
//  \brief Provides some simple tensor algebra functions.
//
//  The provided functions work for three dimensional
//  vectors, one-forms, and metrics.

#include <math.h>
#include "ps_types.hpp"

namespace Primitive {

//! \brief Calculate the determinant from a 3-metric
//
//  \param[in] g3d The spatial metric
//  \return The determinant of g3d
KOKKOS_INLINE_FUNCTION Real GetDeterminant(const Real g3d[NSPMETRIC]) {
  return g3d[S11]*g3d[S22]*g3d[S33] + 2.0*g3d[S12]*g3d[S13]*g3d[S23] -
        (g3d[S11]*g3d[S23]*g3d[S23] + g3d[S12]*g3d[S12]*g3d[S33] +
         g3d[S13]*g3d[S13]*g3d[S22]);
}

//! \brief Convert a vector to a one-form
//
//  \param[out] vd  The output one-form
//  \param[in]  vu  The input vector
//  \param[in]  g3d The 3-metric
KOKKOS_INLINE_FUNCTION void LowerVector(Real vd[3], const Real vu[3],
                                        const Real g3d[NSPMETRIC]) {
  vd[0] = g3d[S11]*vu[0] + g3d[S12]*vu[1] + g3d[S13]*vu[2];
  vd[1] = g3d[S12]*vu[0] + g3d[S22]*vu[1] + g3d[S23]*vu[2];
  vd[2] = g3d[S13]*vu[0] + g3d[S23]*vu[1] + g3d[S33]*vu[2];
}

//! \brief Convert a one-form to a vector
//
//  \param[out] vu  The output vector
//  \param[in]  vd  The input one-form
//  \param[in]  g3u The inverse 3-metric
KOKKOS_INLINE_FUNCTION void RaiseForm(Real vu[3], const Real vd[3],
                                      const Real g3u[NSPMETRIC]) {
  vu[0] = g3u[S11]*vd[0] + g3u[S12]*vd[1] + g3u[S13]*vd[2];
  vu[1] = g3u[S12]*vd[0] + g3u[S22]*vd[1] + g3u[S23]*vd[2];
  vu[2] = g3u[S13]*vd[0] + g3u[S23]*vd[1] + g3u[S33]*vd[2];
}

//! \brief Contract a one-form with a vector
//
//  \param[in] au  The input vector
//  \param[in] bd  The input one-form
//  \return The contraction \f$a^u b_d\f$.
KOKKOS_INLINE_FUNCTION Real Contract(const Real au[3], const Real bd[3]) {
  return au[0]*bd[0] + au[1]*bd[1] + au[2]*bd[2];
}

//! \brief Square a vector
//
//  \param[in] vu  The input vector
//  \param[in] g3d The input metric
//  \return The square of v
KOKKOS_INLINE_FUNCTION Real SquareVector(const Real vu[3], const Real g3d[NSPMETRIC]) {
  return g3d[S11]*vu[0]*vu[0] + g3d[S22]*vu[1]*vu[1] + g3d[S33]*vu[2]*vu[2] +
        2.0*((g3d[S12]*vu[1] + g3d[S13]*vu[2])*vu[0] + g3d[S23]*vu[1]*vu[2]);
}

//! \brief Invert a 3x3 matrix
//
//  \param[out] m_out The output matrix
//  \param[in]  m_in The input matrix
//  \param[in]  det  The determinant of m_in
KOKKOS_INLINE_FUNCTION
void InvertMatrix(Real m_out[NSPMETRIC], const Real m_in[NSPMETRIC], const Real det) {
  Real idet = 1.0/det;
  m_out[S11] = (m_in[S22]*m_in[S33] - m_in[S23]*m_in[S23])*idet;
  m_out[S12] = (m_in[S13]*m_in[S23] - m_in[S12]*m_in[S33])*idet;
  m_out[S13] = (m_in[S12]*m_in[S23] - m_in[S13]*m_in[S22])*idet;
  m_out[S22] = (m_in[S11]*m_in[S33] - m_in[S13]*m_in[S13])*idet;
  m_out[S23] = (m_in[S12]*m_in[S13] - m_in[S11]*m_in[S23])*idet;
  m_out[S33] = (m_in[S11]*m_in[S22] - m_in[S12]*m_in[S12])*idet;
}

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_GEOM_MATH_HPP_
