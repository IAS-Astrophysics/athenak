#ifndef SRCTERMS_TURB_DRIVER_HPP_
#define SRCTERMS_TURB_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriver

class TurbulenceDriver {
 public:
  TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriver();

  // data
  DvceArray5D<Real> force;      // force for driving hydro/mhd variables
  DvceArray5D<Real> force_new;  // second force register for OU evolution

  DvceArray3D<Real> x1sin;   // array for pre-computed sin(k x)
  DvceArray3D<Real> x1cos;   // array for pre-computed cos(k x)
  DvceArray3D<Real> x2sin;   // array for pre-computed sin(k y)
  DvceArray3D<Real> x2cos;   // array for pre-computed cos(k y)
  DvceArray3D<Real> x3sin;   // array for pre-computed sin(k z)
  DvceArray3D<Real> x3cos;   // array for pre-computed cos(k z)

  DualArray2D<Real> amp1, amp2, amp3;

  // parameters of driving
  int nlow,nhigh,ntot,nwave;
  Real tcorr,dedt;
  Real expo;
  Real last_dt;
  int64_t seed; // for generating amp1,amp2,amp3 arrays

  // functions
  void IncludeInitializeModesTask(TaskList &tl, TaskID start);
  void IncludeAddForcingTask(TaskList &tl, TaskID start);
  TaskStatus InitializeModes(Driver *pdrive, int stage);
  TaskStatus AddForcing(Driver *pdrive, int stage);

 private:
  bool first_time=true;     // flag to enable initialization on first call
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver
};

#endif // SRCTERMS_TURB_DRIVER_HPP_
