#ifndef RECONSTRUCT_EOS_HPP_
#define RECONSTRUCT_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.hpp
//  \brief defines class Reconstruction

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief abstract base class for all EOS classes

class Reconstruction {
 public:
  Reconstruction(std::unique_ptr<ParameterInput> &pin);
  virtual ~Reconstruction() = default;

  virtual void ReconstructX1(const int il, const int iu, const AthenaCenterArray<Real> &w,
    AthenaCenterArray<Real> &wl, AthenaCenterArray<Real> &wr) = 0;
  virtual void ReconstructX2(const int il, const int iu, const AthenaCenterArray<Real> &w,
    AthenaCenterArray<Real> &wl, AthenaCenterArray<Real> &wr) = 0;
  virtual void ReconstructX3(const int il, const int iu, const AthenaCenterArray<Real> &w,
    AthenaCenterArray<Real> &wl, AthenaCenterArray<Real> &wr) = 0;

 protected:
};

//----------------------------------------------------------------------------------------
//! \class DonorCell
//  \brief derived class

class DonorCell : public Reconstruction {
 public:
  DonorCell(std::unique_ptr<ParameterInput> &pin);

  void ReconstructX1(const int il, const int iu, const AthenaCenterArray<Real> &w,
    AthenaCenterArray<Real> &wl, AthenaCenterArray<Real> &wr) override;
  void ReconstructX2(const int il, const int iu, const AthenaCenterArray<Real> &w,
    AthenaCenterArray<Real> &wl, AthenaCenterArray<Real> &wr) override;
  void ReconstructX3(const int il, const int iu, const AthenaCenterArray<Real> &w,
    AthenaCenterArray<Real> &wl, AthenaCenterArray<Real> &wr) override;

 private:
};

#endif // RECONSTRUCT_HPP_
