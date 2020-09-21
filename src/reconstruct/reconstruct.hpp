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
  void ReconstructX1(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void ReconstructX2(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void ReconstructX3(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);

  // first-order donor cell reconstruction
  void DonorCellX1(const int k, const int j, const int il, const int iu,
                   const AthenaArray4D<Real> &q,
                   AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void DonorCellX2(const int k, const int j, const int il, const int iu,
                   const AthenaArray4D<Real> &q,
                   AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void DonorCellX3(const int k, const int j, const int il, const int iu,
                   const AthenaArray4D<Real> &q,
                   AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);

  // second-order piecewise linear reconstruction in the primitive variables
  void PLMX1(const int k, const int j, const int il, const int iu,
             const AthenaArray4D<Real> &q,
             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void PLMX2(const int k, const int j, const int il, const int iu,
             const AthenaArray4D<Real> &q,
             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void PLMX3(const int k, const int j, const int il, const int iu,
             const AthenaArray4D<Real> &q,
             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);

  // third-order piecewise parabolic reconstruction in the primitive variables
  void PPMX1(const int k, const int j, const int il, const int iu,
             const AthenaArray4D<Real> &q,
             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void PPMX2(const int k, const int j, const int il, const int iu,
             const AthenaArray4D<Real> &q,
             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);
  void PPMX3(const int k, const int j, const int il, const int iu,
             const AthenaArray4D<Real> &q,
             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr);

 private:
  ReconstructionMethod recon_method_;  // enum that selects which method to use
};

#endif // RECONSTRUCT_HPP_
