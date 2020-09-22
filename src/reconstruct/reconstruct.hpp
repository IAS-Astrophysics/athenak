#ifndef RECONSTRUCT_EOS_HPP_
#define RECONSTRUCT_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.hpp
//  \brief Contains data and functions that implement different spatial reconstruction
//  algorithms, e.g. first-order donor cell, second-order piecewise linear, and
//  third-order piecewise parabolic.

#include "athena.hpp"
#include "parameter_input.hpp"

enum class ReconstructionMethod {donor_cell, piecewise_linear, piecewise_parabolic};

//----------------------------------------------------------------------------------------
//! \class Reconstruction
//  \brief functions for reconstruction methods

class Reconstruction
{
 public:
  Reconstruction(ParameterInput *pin, int nghost);
  ~Reconstruction() = default;

  // wrapper functions that call different methods
  KOKKOS_FUNCTION
  void ReconstructX1(TeamMember_t const &member, const int k, const int j,
                     const int il, const int iu, const AthenaArray4D<Real> &q,
                     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void ReconstructX2(TeamMember_t const &member, const int k, const int j,
                     const int il, const int iu, const AthenaArray4D<Real> &q,
                     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void ReconstructX3(TeamMember_t const &member, const int k, const int j,
                     const int il, const int iu, const AthenaArray4D<Real> &q,
                     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

  // first-order donor cell reconstruction
  KOKKOS_FUNCTION
  void DonorCellX1(TeamMember_t const &member, const int k, const int j,
                   const int il, const int iu, const AthenaArray4D<Real> &q,
                   AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void DonorCellX2(TeamMember_t const &member, const int k, const int j,
                   const int il, const int iu, const AthenaArray4D<Real> &q,
                   AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void DonorCellX3(TeamMember_t const &member, const int k, const int j,
                   const int il, const int iu, const AthenaArray4D<Real> &q,
                   AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

  // second-order piecewise linear reconstruction in the primitive variables
  KOKKOS_FUNCTION
  void PLMX1(TeamMember_t const &member, const int k, const int j,
             const int il, const int iu, const AthenaArray4D<Real> &q,
             AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void PLMX2(TeamMember_t const &member, const int k, const int j,
             const int il, const int iu, const AthenaArray4D<Real> &q,
             AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void PLMX3(TeamMember_t const &member, const int k, const int j,
             const int il, const int iu, const AthenaArray4D<Real> &q,
             AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

  // third-order piecewise parabolic reconstruction in the primitive variables
  KOKKOS_FUNCTION
  void PPMX1(TeamMember_t const &member, const int k, const int j,
             const int il, const int iu, const AthenaArray4D<Real> &q,
             AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void PPMX2(TeamMember_t const &member, const int k, const int j,
             const int il, const int iu, const AthenaArray4D<Real> &q,
             AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);
  KOKKOS_FUNCTION
  void PPMX3(TeamMember_t const &member, const int k, const int j,
             const int il, const int iu, const AthenaArray4D<Real> &q,
             AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr);

 private:
  ReconstructionMethod recon_method_;  // enum that selects which method to use
};

#endif // RECONSTRUCT_HPP_
