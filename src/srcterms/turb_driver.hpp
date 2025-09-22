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
#include <cmath>

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

  DualArray2D<Real> aka, akb; //to store amplitude coefficients (same pattern repeated on all tiles)
  DualArray1D<Real> kx_mode, ky_mode, kz_mode;
  DvceArray3D<Real> xcos, xsin, ycos, ysin, zcos, zsin;


  // AMR tracking variables
  int current_nmb_;            // current number of mesh blocks
  int last_nmb_created_;        // last tracked created blocks count  
  int last_nmb_deleted_;        // last tracked deleted blocks count

  // parameters of driving
  int nlow, nhigh, spect_form;
  int mode_count;
  int rseed; // random seed for turbulence driving
  Real kpeak;
  Real tcorr, dedt, tdriv_duration, tdriv_start;
  Real expo, exp_prl, exp_prp;
  int driving_type, turb_flag;
  int min_kz, max_kz, min_kx, max_kx, min_ky, max_ky;
  Real sol_fraction; // To store fraction of energy in solenoidal modes
  Real dt_turb_update;
  // Real t_last_update;
  int n_turb_updates_yet;

  // drive with constant edot or constant acceleration
  bool constant_edot;

  // spatially varying driving
  Real x_turb_scale_height, y_turb_scale_height, z_turb_scale_height;
  Real x_turb_center, y_turb_center, z_turb_center;

  // tiled driving configuration
  bool tile_driving;
  int tile_nx, tile_ny, tile_nz;
  int num_tiles;
  Real tile_lx, tile_ly, tile_lz;
  Real inv_tile_lx, inv_tile_ly, inv_tile_lz;
  Real domain_x1min, domain_x2min, domain_x3min;


  // functions
  void IncludeInitializeModesTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus InitializeModes(Driver *pdrive, int stage);
  TaskStatus EnsureBasisSize(Driver *pdrive, int stage);
  TaskStatus UpdateForcing(Driver *pdrive, int stage);
  TaskStatus AddForcing(Driver *pdrive, int stage);
  void Initialize();

 private:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver
};


#endif  // SRCTERMS_TURB_DRIVER_HPP_
