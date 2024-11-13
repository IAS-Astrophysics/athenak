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

  DvceArray5D<Real> force, force_tmp1, force_tmp2;  // arrays used for turb forcing
  RNG_State rstate;                    // random state

  DualArray2D<Real> aka, akb; //to store amplitude coefficients
  DualArray1D<Real> kx_mode, ky_mode, kz_mode;
  DvceArray3D<Real> xcos, xsin, ycos, ysin, zcos, zsin;

  // parameters of driving
  int nlow, nhigh, spect_form;
  int mode_count;
  Real kpeak;
  Real tcorr, dedt, tdriv_duration, tdriv_start;
  Real expo, exp_prl, exp_prp;
  int driving_type, turb_flag;
  int min_kz, max_kz;
  Real sol_fraction; // To store fraction of energy in solenoidal modes
  Real dt_turb_update,dt_turb_thresh;
  // Real t_last_update;
  int n_turb_updates_yet;

  // spatially varying driving
  Real x_turb_scale_height, y_turb_scale_height, z_turb_scale_height;
  Real x_turb_center, y_turb_center, z_turb_center;


  // functions
  void IncludeInitializeModesTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus InitializeModes(Driver *pdrive, int stage);
  TaskStatus UpdateForcing(Driver *pdrive, int stage);
  TaskStatus AddForcing(Driver *pdrive, int stage);
  void Initialize();

 private:
  bool first_time = true;   // flag to enable initialization on first call
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver
};

#endif  // SRCTERMS_TURB_DRIVER_HPP_
