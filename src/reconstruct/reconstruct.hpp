#ifndef RECONSTRUCT_EOS_HPP_
#define RECONSTRUCT_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.hpp
//  \brief defines abstract base class Reconstruction, and various derived classes
//  Each derived class Contains data and functions that implement different spatial
// reconstruction algorithms, e.g. first-order donor cell, second-order piecewise linear,
// and third-order piecewise parabolic.

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

// constants that enumerate reconstruction options
//enum class ReconstructionMethod {donor_cell, piecewise_linear, piecewise_parabolic};

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief abstract base class for all Reconstruction classes

class Reconstruction {
 public:
  Reconstruction(ParameterInput *pin);
  virtual ~Reconstruction() = default;

  virtual void ReconstructX1(const int il, const int iu, const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) = 0;
  virtual void ReconstructX2(const int il, const int iu, const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) = 0;
  virtual void ReconstructX3(const int il, const int iu, const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) = 0;

 protected:
};

//----------------------------------------------------------------------------------------
//! \class DonorCell
//  \brief derived Reconstruction class for first-order donor cell method

class DonorCell : public Reconstruction {
 public:
  DonorCell(ParameterInput *pin);

  void ReconstructX1(const int il, const int iu, const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) override;
  void ReconstructX2(const int il, const int iu, const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) override;
  void ReconstructX3(const int il, const int iu, const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) override;

 private:
};

#endif // RECONSTRUCT_HPP_
