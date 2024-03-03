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

#include <memory>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriver

class TurbulenceDriver {
 public:
  TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriver();

  DvceArray5D<Real> force, force_tmp;  // arrays used for turb forcing
  RNG_State rstate;                    // random state

  DualArray1D<Real> xccc, xccs, xcsc, xcss, xscc, xscs, xssc, xsss;
  DualArray1D<Real> yccc, yccs, ycsc, ycss, yscc, yscs, yssc, ysss;
  DualArray1D<Real> zccc, zccs, zcsc, zcss, zscc, zscs, zssc, zsss;
  DualArray1D<Real> kx_mode, ky_mode, kz_mode;
  DvceArray3D<Real> xcos, xsin, ycos, ysin, zcos, zsin;

  // parameters of driving
  int nlow, nhigh;
  int mode_count;
  Real tcorr, dedt;
  Real expo, exp_prl, exp_prp;
  int driving_type;

  // functions
  void IncludeInitializeModesTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus InitializeModes(Driver *pdrive, int stage);
  TaskStatus AddForcing(Driver *pdrive, int stage);
  void Initialize();

 private:
  bool first_time = true;   // flag to enable initialization on first call
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver
};

#endif  // SRCTERMS_TURB_DRIVER_HPP_
