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
//  reconstruction algorithms, e.g. first-order donor cell, second-order piecewise linear,
//  and third-order piecewise parabolic.

#include "athena.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class Reconstruction
//  \brief abstract base class for all Reconstruction classes

class Reconstruction
{
 public:
  Reconstruction(ParameterInput *pin, int nvar, int ncells1);
  virtual ~Reconstruction() = default;

  // pure virtual reconstruction functions overwritten in each derived class
  virtual void ReconstructX1(const int k, const int j, const int il, const int iu,
                             const AthenaArray4D<Real> &q,
                             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) = 0;
  virtual void ReconstructX2(const int k, const int j, const int il, const int iu,
                             const AthenaArray4D<Real> &q,
                             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) = 0;
  virtual void ReconstructX3(const int k, const int j, const int il, const int iu,
                             const AthenaArray4D<Real> &q,
                             AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) = 0;

 protected:
  int nvar_, ncells1_;
};

//----------------------------------------------------------------------------------------
//! \class DonorCell
//  \brief derived Reconstruction class for first-order donor cell method

class DonorCell : public Reconstruction
{
 public:
  DonorCell(ParameterInput *pin, int nvar, int ncells1);

  void ReconstructX1(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
  void ReconstructX2(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
  void ReconstructX3(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
};

//----------------------------------------------------------------------------------------
//! \class PiecewiseLinear
//  \brief derived Reconstruction class for second-order PLM method

class PiecewiseLinear : public Reconstruction
{
 public:
  PiecewiseLinear(ParameterInput *pin, int nvar, int ncells1);

  void ReconstructX1(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
  void ReconstructX2(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
  void ReconstructX3(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
};

//----------------------------------------------------------------------------------------
//! \class PiecewiseParabolic
//  \brief derived Reconstruction class for third-order PPM method

class PiecewiseParabolic : public Reconstruction
{
 public:
  PiecewiseParabolic(ParameterInput *pin, int nvar, int ncells1);

  void ReconstructX1(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
  void ReconstructX2(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
  void ReconstructX3(const int k, const int j, const int il, const int iu,
                     const AthenaArray4D<Real> &q,
                     AthenaArray2D<Real> &ql, AthenaArray2D<Real> &qr) override;
};

#endif // RECONSTRUCT_HPP_
