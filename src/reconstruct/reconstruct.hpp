#ifndef RECONSTRUCT_EOS_HPP_
#define RECONSTRUCT_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.hpp
//  \brief Function prototypes for all reconstruction methods

#include "athena.hpp"

// first-order donor cell reconstruction
KOKKOS_INLINE_FUNCTION
void DonorCellX1(TeamMember_t const &member, const int k, const int j,
                 const int il, const int iu, const AthenaArray4D<Real> &q,
                 AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
KOKKOS_INLINE_FUNCTION
void DonorCellX2(TeamMember_t const &member, const int k, const int j,
                 const int il, const int iu, const AthenaArray4D<Real> &q,
                 AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
KOKKOS_INLINE_FUNCTION
void DonorCellX3(TeamMember_t const &member, const int k, const int j,
                 const int il, const int iu, const AthenaArray4D<Real> &q,
                 AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

// second-order piecewise linear reconstruction in the primitive variables
KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX1(TeamMember_t const &member, const int k, const int j,
           const int il, const int iu, const AthenaArray4D<Real> &q,
           AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX2(TeamMember_t const &member, const int k, const int j,
           const int il, const int iu, const AthenaArray4D<Real> &q,
           AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
KOKKOS_INLINE_FUNCTION
void PiecewiseLinearX3(TeamMember_t const &member, const int k, const int j,
           const int il, const int iu, const AthenaArray4D<Real> &q,
           AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

// third-order piecewise parabolic reconstruction in the primitive variables
KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX1(TeamMember_t const &member, const int k, const int j,
           const int il, const int iu, const AthenaArray4D<Real> &q,
           AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX2(TeamMember_t const &member, const int k, const int j,
           const int il, const int iu, const AthenaArray4D<Real> &q,
           AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
KOKKOS_INLINE_FUNCTION
void PiecewiseParabolicX3(TeamMember_t const &member, const int k, const int j,
           const int il, const int iu, const AthenaArray4D<Real> &q,
           AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

#endif // RECONSTRUCT_HPP_
