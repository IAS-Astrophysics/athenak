//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceDriver

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <cmath>
#include <vector>
#include <utility>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "driver/driver.hpp"
#include "utils/random.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "turb_driver.hpp"
#include "globals.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriver::TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  force("force",1,1,1,1,1),
  force_tmp1("force_tmp1",1,1,1,1,1),
  force_tmp2("force_tmp2",1,1,1,1,1),
  aka("aka",1,1),akb("akb",1,1),
  kx_mode("kx_mode",1),ky_mode("ky_mode",1),kz_mode("kz_mode",1),
  xcos("xcos",1,1,1),xsin("xsin",1,1,1),ycos("ycos",1,1,1),
  ysin("ysin",1,1,1),zcos("zcos",1,1,1),zsin("zsin",1,1,1) {
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  // Initialize AMR tracking variables
  current_nmb_ = nmb;
  Mesh *pm = pmy_pack->pmesh;
  if (pm->adaptive && pm->pmr != nullptr) {
    last_nmb_created_ = pm->pmr->nmb_created;
    last_nmb_deleted_ = pm->pmr->nmb_deleted;
  } else {
    last_nmb_created_ = 0;
    last_nmb_deleted_ = 0;
  }

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp1, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp2, nmb, 3, ncells3, ncells2, ncells1);

  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("turb_driving", "nlow", 1);
  nhigh = pin->GetOrAddInteger("turb_driving", "nhigh", 3);
  // Peak of power when spectral form is parabolic, in units of 2*(PI/L)
  // Support both npeak (wavenumber index) and kpeak (actual k value)
  if (pin->DoesParameterExist("turb_driving", "npeak")) {
    Real npeak = pin->GetReal("turb_driving", "npeak");
    // Convert npeak to kpeak using fundamental wavenumber
    Real dkfund = 2.0*M_PI; // Assuming box size = 1
    kpeak = npeak * dkfund;
  } else {
    kpeak = pin->GetOrAddReal("turb_driving", "kpeak", 4.0*M_PI);
  }
  // spect form - 1 for parabola, 2 for power-law
  spect_form = pin->GetOrAddInteger("turb_driving", "spect_form", 1);
  // driving type - 0 for 3D isotropic, 1 for xy plane
  driving_type = pin->GetOrAddInteger("turb_driving", "driving_type", 0);
  // min kz zero should be 0 for including kz modes and 1 for not including
  min_kz = pin->GetOrAddInteger("turb_driving", "min_kz", 0);
  max_kz = pin->GetOrAddInteger("turb_driving", "max_kz", nhigh);
  min_kx = pin->GetOrAddInteger("turb_driving", "min_kx", 0);
  max_kx = pin->GetOrAddInteger("turb_driving", "max_kx", nhigh);
  min_ky = pin->GetOrAddInteger("turb_driving", "min_ky", 0);
  max_ky = pin->GetOrAddInteger("turb_driving", "max_ky", nhigh);
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("turb_driving", "expo", 5.0/3.0);
  exp_prp = pin->GetOrAddReal("turb_driving", "exp_prp", 5.0/3.0);
  exp_prl = pin->GetOrAddReal("turb_driving", "exp_prl", 0.0);
  // energy injection rate
  dedt = pin->GetOrAddReal("turb_driving", "dedt", 0.0);
  // correlation time
  tcorr = pin->GetOrAddReal("turb_driving", "tcorr", 0.0);
  // update time for the turbulence driver
  dt_turb_update=pin->GetOrAddReal("turb_driving","dt_turb_update",0.01);
  // To store fraction of energy in solenoidal modes
  sol_fraction=pin->GetOrAddReal("turb_driving","sol_fraction",1.0);

  // random seed for turbulence driving (-1 = use time-based seed)
  rseed = pin->GetOrAddInteger("turb_driving", "rseed", -1);

  // drive with constant edot or constant acceleration
  constant_edot = pin->GetOrAddBoolean("turb_driving", "constant_edot", true);

  // spatially varying driving
  x_turb_scale_height = pin->GetOrAddReal("turb_driving", "x_turb_scale_height", -1.0);
  y_turb_scale_height = pin->GetOrAddReal("turb_driving", "y_turb_scale_height", -1.0);
  z_turb_scale_height = pin->GetOrAddReal("turb_driving", "z_turb_scale_height", -1.0);
  x_turb_center = pin->GetOrAddReal("turb_driving", "x_turb_center", 0.0);
  y_turb_center = pin->GetOrAddReal("turb_driving", "y_turb_center", 0.0);
  z_turb_center = pin->GetOrAddReal("turb_driving", "z_turb_center", 0.0);

  // tiled driving configuration
  tile_driving = pin->GetOrAddBoolean("turb_driving", "tile_driving", false);
  int tile_factor = pin->GetOrAddInteger("turb_driving", "tile_factor", 1);
  tile_nx = pin->GetOrAddInteger("turb_driving", "tile_nx", tile_factor);
  tile_ny = pin->GetOrAddInteger("turb_driving", "tile_ny", tile_factor);
  tile_nz = pin->GetOrAddInteger("turb_driving", "tile_nz", tile_factor);
  if (!tile_driving) {
    tile_nx = 1;
    tile_ny = 1;
    tile_nz = 1;
  }

  domain_x1min = pm->mesh_size.x1min;
  domain_x2min = pm->mesh_size.x2min;
  domain_x3min = pm->mesh_size.x3min;

  auto &mesh_indcs_root = pm->mesh_indcs;
  if (tile_nx < 1 || tile_ny < 1 || tile_nz < 1) {
    std::cout << "### FATAL ERROR in turbulence driver: tile counts must be >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (mesh_indcs_root.nx1 % tile_nx != 0) {
    std::cout << "### FATAL ERROR in turbulence driver: tile_nx must evenly divide nx1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs_root.nx2 <= 1) {
    tile_ny = 1;
  } else if (mesh_indcs_root.nx2 % tile_ny != 0) {
    std::cout << "### FATAL ERROR in turbulence driver: tile_ny must evenly divide nx2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs_root.nx3 <= 1) {
    tile_nz = 1;
  } else if (mesh_indcs_root.nx3 % tile_nz != 0) {
    std::cout << "### FATAL ERROR in turbulence driver: tile_nz must evenly divide nx3" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real lx_global = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly_global = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz_global = pm->mesh_size.x3max - pm->mesh_size.x3min;

  tile_lx = lx_global / static_cast<Real>(tile_nx);
  tile_ly = (tile_ny > 0) ? ly_global / static_cast<Real>(tile_ny) : ly_global;
  tile_lz = (tile_nz > 0) ? lz_global / static_cast<Real>(tile_nz) : lz_global;

  inv_tile_lx = (tile_lx > 0.0) ? 1.0/tile_lx : 0.0;
  inv_tile_ly = (tile_ly > 0.0) ? 1.0/tile_ly : 0.0;
  inv_tile_lz = (tile_lz > 0.0) ? 1.0/tile_lz : 0.0;

  num_tiles = tile_nx * tile_ny * tile_nz;

  // decaying/constant energy injection - 1 for decaying, 2 continuously driven
  turb_flag = pin->GetOrAddInteger("turb_driving", "turb_flag", 2);
  if(turb_flag ==1) {
    tdriv_duration = pin->GetOrAddReal("turb_driving", "tdriv_duration", tcorr); // If not specified, drive for one correlation time
  }
  else {
    tdriv_duration = static_cast<Real>(std::numeric_limits<float>::max()); // For constantly stirred turbulence, set this to float max
  }
  tdriv_start = pin->GetOrAddReal("turb_driving", "tdriv_start", 0.); // If not specified, start driving at t=0
  if (global_variable::my_rank == 0){
    std::cout << "Initialising turbulence driving module" << std::endl <<
    " dedt = " << dedt << " tcorr = " << tcorr << " dt_turb_update = " << dt_turb_update << std::endl;
  }

  // Initialize n_turb_updates_yet based on current simulation time
  Real current_time = pmy_pack->pmesh->time;
  if (current_time < tdriv_start) {
    n_turb_updates_yet = 0;
  } else {
    Real t_since_start = current_time - tdriv_start;
    n_turb_updates_yet = (int) (t_since_start/dt_turb_update);
  }

  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;

  mode_count = 0;

  // Count Cartesian modes
  int nkx, nky, nkz;
  Real nsqr;
  for (nkx = min_kx; nkx <= max_kx; nkx++) {
    for (nky = min_ky; nky <= max_ky; nky++) {
      for (nkz = min_kz; nkz <= max_kz; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_type == 0) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_type == 1) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          mode_count++;
        }
      }
    }
  }

  if (mode_count == 0) {
    std::cout << "ERROR: mode_count is 0! Check turbulence driving parameters." << std::endl;
    std::cout << "  nlow=" << nlow << ", nhigh=" << nhigh << std::endl;
    std::cout << "  driving_type=" << driving_type << std::endl;
    exit(EXIT_FAILURE);
  }

  Kokkos::realloc(aka, 3, mode_count); // Amplitude of real component (repeated on all tiles)
  Kokkos::realloc(akb, 3, mode_count); // Amplitude of imaginary component (repeated on all tiles)

  // Allocate Cartesian mode arrays
  Kokkos::realloc(kx_mode, mode_count);
  Kokkos::realloc(ky_mode, mode_count);
  Kokkos::realloc(kz_mode, mode_count);

  Kokkos::realloc(xcos, nmb, mode_count, ncells1);
  Kokkos::realloc(xsin, nmb, mode_count, ncells1);
  Kokkos::realloc(ycos, nmb, mode_count, ncells2);
  Kokkos::realloc(ysin, nmb, mode_count, ncells2);
  Kokkos::realloc(zcos, nmb, mode_count, ncells3);
  Kokkos::realloc(zsin, nmb, mode_count, ncells3);

  Initialize();
}

//----------------------------------------------------------------------------------------
// destructor

TurbulenceDriver::~TurbulenceDriver() {
}

//----------------------------------------------------------------------------------------
//! \fn  noid Initialize
//  \brief Function to initialize the driver

void TurbulenceDriver::Initialize() {
  Mesh *pm = pmy_pack->pmesh;
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;

  auto force_tmp1_ = force_tmp1;
  auto force_tmp2_ = force_tmp2;
  par_for("force_init_pgen",DevExeSpace(),
          0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_tmp1_(m,n,k,j,i) = 0.0;
    force_tmp2_(m,n,k,j,i) = 0.0;
  });

  rstate.idum = rseed;

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  const bool tile_enabled = tile_driving;
  const int tile_nx_local = tile_nx;
  const int tile_ny_local = tile_ny;
  const int tile_nz_local = tile_nz;
  const Real tile_lx_local = tile_lx;
  const Real tile_ly_local = tile_ly;
  const Real tile_lz_local = tile_lz;
  const Real inv_tile_lx_local = inv_tile_lx;
  const Real inv_tile_ly_local = inv_tile_ly;
  const Real inv_tile_lz_local = inv_tile_lz;
  const Real domain_x1min_local = domain_x1min;
  const Real domain_x2min_local = domain_x2min;
  const Real domain_x3min_local = domain_x3min;

  // Cartesian plane-wave precomputations
  Real dkx, dky, dkz, kx, ky, kz;
    Real lx = tile_lx_local;
    Real ly = tile_ly_local;
    Real lz = tile_lz_local;
    dkx = (lx > 0.0) ? 2.0*M_PI/lx : 0.0;
    dky = (ly > 0.0) ? 2.0*M_PI/ly : 0.0;
    dkz = (lz > 0.0) ? 2.0*M_PI/lz : 0.0;

    int nmode = 0;
    int nkx, nky, nkz;
    Real nsqr;
    Real nlow_sqr = nlow*nlow;
    Real nhigh_sqr = nhigh*nhigh;
    for (nkx = min_kx; nkx <= max_kx; nkx++) {
      for (nky = min_ky; nky <= max_ky; nky++) {
        for (nkz = min_kz; nkz <= max_kz; nkz++) {
          if (nkx == 0 && nky == 0 && nkz == 0) continue;
          nsqr = 0.0;
          bool flag_prl = true;
          if (driving_type == 0) {
            nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
          } else if (driving_type == 1) {
            nsqr = SQR(nkx) + SQR(nky);
            Real nprlsqr = SQR(nkz);
            if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
              flag_prl = true;
            } else {
              flag_prl = false;
            }
          }
          if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
            kx = dkx*nkx;
            ky = dky*nky;
            kz = dkz*nkz;
            kx_mode_.h_view(nmode) = kx;
            ky_mode_.h_view(nmode) = ky;
            kz_mode_.h_view(nmode) = kz;
            nmode++;
          }
        }
      }
    }

  kx_mode_.template modify<HostMemSpace>();
  kx_mode_.template sync<DevExeSpace>();
  ky_mode_.template modify<HostMemSpace>();
  ky_mode_.template sync<DevExeSpace>();
  kz_mode_.template modify<HostMemSpace>();
  kz_mode_.template sync<DevExeSpace>();

  // Ensure MeshBlock geometry (mb_size) is current on device before kernels use size_view.d_view(...)
  auto size_view = pmy_pack->pmb->mb_size;   // copy the DualView handle
  size_view.template modify<HostMemSpace>();
  size_view.template sync<DevExeSpace>();
  const int drivingtype = driving_type;      // value, not reference

  par_for("xsin/xcos", DevExeSpace(),0,nmb-1,0,mode_count-1,is,ie,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size_view.d_view(m).x1min;
    Real &x1max = size_view.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    Real arg = x1v;
    if (tile_enabled && tile_nx_local > 1) {
      Real rel = x1v - domain_x1min_local;
      int tile_i = static_cast<int>(floor(rel * inv_tile_lx_local));
      tile_i = (tile_i < 0) ? 0 : ((tile_i >= tile_nx_local) ? tile_nx_local - 1 : tile_i);
      Real tile_origin = domain_x1min_local + tile_i * tile_lx_local;
      arg = x1v - tile_origin;
    }
    xsin_(m,n,i) = sin(k1v*arg);
    xcos_(m,n,i) = cos(k1v*arg);
  });

  par_for("ysin/ycos", DevExeSpace(),0,nmb-1,0,mode_count-1,js,je,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size_view.d_view(m).x2min;
    Real &x2max = size_view.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    Real arg = x2v;
    if (tile_enabled && tile_ny_local > 1) {
      Real rel = x2v - domain_x2min_local;
      int tile_j = static_cast<int>(floor(rel * inv_tile_ly_local));
      tile_j = (tile_j < 0) ? 0 : ((tile_j >= tile_ny_local) ? tile_ny_local - 1 : tile_j);
      Real tile_origin = domain_x2min_local + tile_j * tile_ly_local;
      arg = x2v - tile_origin;
    }
    ysin_(m,n,j) = sin(k2v*arg);
    ycos_(m,n,j) = cos(k2v*arg);
    if (ncells2-1 == 0) {
      ysin_(m,n,j) = 0.0;
      ycos_(m,n,j) = 1.0;
    }
  });

  par_for("zsin/zcos", DevExeSpace(),0,nmb-1,0,mode_count-1,ks,ke,
  KOKKOS_LAMBDA(int m, int n, int k) {
    Real &x3min = size_view.d_view(m).x3min;
    Real &x3max = size_view.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real k3v = kz_mode_.d_view(n);
    Real arg = x3v;
    if (tile_enabled && tile_nz_local > 1) {
      Real rel = x3v - domain_x3min_local;
      int tile_k = static_cast<int>(floor(rel * inv_tile_lz_local));
      tile_k = (tile_k < 0) ? 0 : ((tile_k >= tile_nz_local) ? tile_nz_local - 1 : tile_k);
      Real tile_origin = domain_x3min_local + tile_k * tile_lz_local;
      arg = x3v - tile_origin;
    }
    zsin_(m,n,k) = sin(k3v*arg);
    zcos_(m,n,k) = cos(k3v*arg);
    if (ncells3-1 == 0 || (drivingtype == 1)) {
      zsin_(m,n,k) = 0.0;
      zcos_(m,n,k) = 1.0;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeModeEvolutionTasks
//  \brief Includes task in the operator split task list that constructs new modes with
//  random amplitudes and phases that can be used to evolve the force via an O-U process
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeInitializeModesTask(std::shared_ptr<TaskList> tl,
                                                  TaskID start) {
  //  We check for mesh changes, then initialize modes and update the forcing
  auto id_resize = tl->AddTask(&TurbulenceDriver::EnsureBasisSize, this, start);
  auto id_init   = tl->AddTask(&TurbulenceDriver::InitializeModes, this, id_resize);
  auto id_add    = tl->AddTask(&TurbulenceDriver::UpdateForcing, this, id_init);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeForcingTasks
//  \brief includes task in the stage_run task list for adding random forcing to fluid
//  as an explicit source terms in each stage of integrator
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start) {
  // These must be inserted after update task, but before the source terms
  // We apply the forcing in each step of the time integration,
  // note that we do not update the forcing in each RK stage
  if (pmy_pack->pionn == nullptr) {
    if (pmy_pack->phydro != nullptr) {
      auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                              pmy_pack->phydro->id.rkupdt, pmy_pack->phydro->id.srctrms);
    }
    if (pmy_pack->pmhd != nullptr) {
      auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                              pmy_pack->pmhd->id.rkupdt, pmy_pack->pmhd->id.srctrms);
    }
  } else {
    auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                            pmy_pack->pionn->id.n_rkupdt, pmy_pack->pionn->id.n_flux);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn InitializeModes()
// \brief Initializes the turbulence driving modes, and so is only executed once at start of calculation.
// Cannot be included in constructor since (it seems) Kokkos::par_for not allowed in cons.

TaskStatus TurbulenceDriver::InitializeModes(Driver *pdrive, int stage) {

  if (pmy_pack == nullptr) {
    return TaskStatus::complete;
  }

  Mesh *pm = pmy_pack->pmesh;
  if (pm == nullptr) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real current_time=pm->time;
  if (current_time < tdriv_start) return TaskStatus::complete;
  Real t_since_start = current_time - tdriv_start;
  int n_turb_updates_reqd = (int) (t_since_start/dt_turb_update) + 1;

  int nlow_sqr = SQR(nlow);
  int nhigh_sqr = SQR(nhigh);
  auto mode_count_ = mode_count;

  auto aka_ = aka;
  auto akb_ = akb;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = tile_driving ? tile_lx : (pm->mesh_size.x1max - pm->mesh_size.x1min);
  Real ly = tile_driving ? tile_ly : (pm->mesh_size.x2max - pm->mesh_size.x2min);
  Real lz = tile_driving ? tile_lz : (pm->mesh_size.x3max - pm->mesh_size.x3min);
  dkx = (lx > 0.0) ? 2.0*M_PI/lx : 0.0;
  dky = (ly > 0.0) ? 2.0*M_PI/ly : 0.0;
  dkz = (lz > 0.0) ? 2.0*M_PI/lz : 0.0;

  Real &ex = expo;
  Real &ex_prp = exp_prp;
  Real &ex_prl = exp_prl;
  Real norm, kprl, kprp, kiso;
  Real khigh = nhigh*fmax(fmax(dkx,dky),dkz);
  Real klow  = nlow *fmin(fmin(dkx,dky),dkz);
  Real parab_prefact = -4.0 / pow(khigh-klow,2.0);
  Real &k_peak = kpeak;

  // Now compute new force using new random amplitudes and phases
  // no need to evolve force_new if dt_turb_update hasn't passed since the last update

  if ((t_since_start < tdriv_duration) || turb_flag != 1){ // Update forcing if continuous or t<tdriv_duration

    for(int i_turb_update = n_turb_updates_yet; i_turb_update < n_turb_updates_reqd; i_turb_update++){
      if (global_variable::my_rank == 0) std::cout << "i_turb_update = " << i_turb_update << std::endl;

      auto force_tmp2_ = force_tmp2;
      const int nmb = pmy_pack->nmb_thispack;

      // Zero out new force array
      par_for("force_init", DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        force_tmp2_(m,n,k,j,i) = 0.0;
      });

      // if (global_variable::my_rank == 0) std::cout << "force_tmp2_ zeroed." << std::endl;

      int no_dir=3;
      int nmode = 0;

      // Cartesian mode generation
      int nkx, nky, nkz, nsqr;

        for (nkx = min_kx; nkx <= max_kx; nkx++) {
          for (nky = min_ky; nky <= max_ky; nky++) {
            for (nkz = min_kz; nkz <= max_kz; nkz++) {
              if (nkx == 0 && nky == 0 && nkz == 0) continue;
              norm = 0.0;
              nsqr = 0.0;
              bool flag_prl = true;
              if (driving_type == 0) {
                nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
              } else if (driving_type == 1) {
                nsqr = SQR(nkx) + SQR(nky);
                Real nprlsqr = SQR(nkz);
                if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
                  flag_prl = true;
                } else {
                  flag_prl = false;
                }
              }
              if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
                kx = dkx*nkx;
                ky = dky*nky;
                kz = dkz*nkz;

                Real k[3] = {kx, ky, kz};
                // Always define kiso; used below for the solenoidal/compressive split
                kiso = sqrt(SQR(kx) + SQR(ky) + SQR(kz));


                // Generate Fourier amplitudes

                if (driving_type == 0) {
                if (kiso > 1e-16) {
                  if(spect_form==2) norm = 1.0/pow(kiso,(ex+2.0)/2.0); // power-law driving
                  else if (spect_form==1)
                  {
                    norm = fabs(parab_prefact*pow(kiso-k_peak,2.0)+1.0);// parabola in k-space
                    norm = pow(norm,0.5) * pow(k_peak/kiso, ((int)no_dir-1)/2.);
                  }
                  else {
                  norm = 0.0;
                  }
                } else {
                  norm = 0.0;
                }
                } else if (driving_type == 1) {
                  no_dir = 2;
                  kprl = sqrt(SQR(kx));
                  kprp = sqrt(SQR(ky) + SQR(kz));
                  if (kprl > 1e-16 && kprp > 1e-16) {
                    if(spect_form==2) norm = 1.0/pow(kprp,(ex_prp+1.0)/2.0)/pow(kprl,ex_prl/2.0);

                    else if (spect_form==1)
                    {
                      norm = fabs(parab_prefact*pow(kprp-k_peak,2.0)+1.0);// parabola in kperp-space
                      norm = pow(norm,0.5) * pow(k_peak/kprp, ((int)no_dir-1)/2.);
                    }
                  } else {
                    norm = 0.0;
                  }
                }
                // Generate coefficients once (same pattern repeated on all tiles)
                Real ka = 0.0;
                Real kb = 0.0;

                for (int dir = 0; dir < no_dir; dir ++) {
                  Real aval = norm*RanGaussianSt(&(rstate));
                  Real bval = norm*RanGaussianSt(&(rstate));
                  aka_.h_view(dir,nmode) = aval;
                  akb_.h_view(dir,nmode) = bval;

                  ka += k[dir]*bval;
                  kb += k[dir]*aval;
                }

                // Now decompose into solenoidal/compressive modes
                if (norm > 0.) {
                  for (int dir = 0; dir < no_dir; dir ++) {
                    Real diva = k[dir]*ka/SQR(kiso);
                    Real divb = k[dir]*kb/SQR(kiso);

                    Real curla = aka_.h_view(dir,nmode) - divb;
                    Real curlb = akb_.h_view(dir,nmode) - diva;
                    aka_.h_view(dir,nmode) = sol_fraction*curla+(1.0-sol_fraction)*divb;
                    akb_.h_view(dir,nmode) = sol_fraction*curlb+(1.0-sol_fraction)*diva;
                  }
                }

                nmode++;
              }
            }
          }
        }

      aka_.template modify<HostMemSpace>();
      aka_.template sync<DevExeSpace>();
      akb_.template modify<HostMemSpace>();
      akb_.template sync<DevExeSpace>();

      // if (global_variable::my_rank == 0) std::cout << "Sines and cosines updated on device" << std::endl;
      auto xcos_ = xcos;
      auto xsin_ = xsin;
      auto ycos_ = ycos;
      auto ysin_ = ysin;
      auto zcos_ = zcos;
      auto zsin_ = zsin;

      int mode_count_ = mode_count;
      par_for("force_compute", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // No tile index needed - same coefficients used everywhere
        // The basis functions (xcos, xsin, etc.) already use tile-local coords
        // so the pattern automatically repeats with period = tile_size
        for (int n=0; n<mode_count_; ++n) {
          Real forc_real = ( xcos_(m,n,i)*ycos_(m,n,j) - xsin_(m,n,i)*ysin_(m,n,j) ) * zcos_(m,n,k) -
                          ( xsin_(m,n,i)*ycos_(m,n,j) + xcos_(m,n,i)*ysin_(m,n,j) ) * zsin_(m,n,k);
          Real forc_imag = ( ycos_(m,n,j)*zsin_(m,n,k) + ysin_(m,n,j)*zcos_(m,n,k) ) * xcos_(m,n,i) +
                          ( ycos_(m,n,j)*zcos_(m,n,k) - ysin_(m,n,j)*zsin_(m,n,k) ) * xsin_(m,n,i);
          for (int dir = 0; dir < 3; dir ++){
            force_tmp2_(m,dir,k,j,i) += aka_.d_view(dir,n)*forc_real -
                                        akb_.d_view(dir,n)*forc_imag;
          }
        }
      });
      // if (global_variable::my_rank == 0) std::cout << "force_tmp2_ computed." << std::endl;
      // Let's skip the momentum subtraction during restarts -- or alternatively move this to add force
      Real fcorr, gcorr;
      if ((tcorr <= 1e-6) || (i_turb_update==0)) {  // use whitenoise
        fcorr = 0.0;
        gcorr = 1.0;
      } else {
        fcorr = std::exp(-dt_turb_update/tcorr);
        gcorr = std::sqrt(1.0 - fcorr*fcorr);
      }
      // update force if number of required steps is greater than 1
      auto force_tmp1_ = force_tmp1;
      par_for("OU_process", DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        force_tmp1_(m,n,k,j,i) = fcorr*force_tmp1_(m,n,k,j,i) + gcorr*force_tmp2_(m,n,k,j,i);
      });
    } // end of for loop over i_turb_update
  }
  n_turb_updates_yet = n_turb_updates_reqd;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Update forcing
//
// @brief Updates the forcing applied in the turbulence driver.
//
// This function updates the forcing applied in the turbulence driver,
// It initializes various parameters and arrays, scales the forcing, and
// handles momentum and energy updates. Additionally, it supports two-fluid
// and magnetohydrodynamic (MHD) scenarios.
// It is called before the time integrator. The acceleration field is fixed
// for the sub-steps of the RK integrator.
//
// @param pdrive Pointer to the driver object.
// @param stage The current stage of the driver.
// @return TaskStatus indicating the completion status of the task.
//
// The function performs the following main steps:
// 1. Copies values from temporary force array to the main force array.
// 2. Applies Gaussian weighting to the forcing in x, y, and z directions if requested.
// 3. Computes net momentum and applies corrections to ensure momentum conservation.
// 4. Scales the forcing to input dedt.
//

TaskStatus TurbulenceDriver::UpdateForcing(Driver *pdrive, int stage) {

  if (pmy_pack == nullptr) {
    return TaskStatus::complete;
  }

  Mesh *pm = pmy_pack->pmesh;
  if (pm == nullptr) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;


  Real dt = pm->dt;
  Real current_time=pm->time;
  Real t_since_start = current_time - tdriv_start;

  bool scale_forcing = true;

  DvceArray5D<Real> u0, u0_;
  DvceArray5D<Real> w0, w0_;
  if (pmy_pack->phydro != nullptr) u0 = (pmy_pack->phydro->u0);
  if (pmy_pack->phydro != nullptr) w0 = (pmy_pack->phydro->w0);
  if (pmy_pack->pmhd != nullptr) u0 = (pmy_pack->pmhd->u0);
  if (pmy_pack->pmhd != nullptr) w0 = (pmy_pack->pmhd->w0);
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    w0 = (pmy_pack->phydro->w0);
    w0_ = (pmy_pack->pmhd->w0);
    flag_twofl = true;
  }

  auto force_ = force;
  auto force_tmp1_ = force_tmp1;

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  // Copy the DualView handle by value, sync device, and use this in all kernels
  auto mb_size = pmy_pack->pmb->mb_size;
  mb_size.template modify<HostMemSpace>();
  mb_size.template sync<DevExeSpace>();

  auto x_turb_scale_height_ = x_turb_scale_height;
  auto y_turb_scale_height_ = y_turb_scale_height;
  auto z_turb_scale_height_ = z_turb_scale_height;
  auto x_turb_center_ = x_turb_center;
  auto y_turb_center_ = y_turb_center;
  auto z_turb_center_ = z_turb_center;

  // Copy values of force_tmp1 into force array
  // perform operations such as normalisation,
  // momentum subtraction directly on the force array

  // if (global_variable::my_rank == 0) std::cout << " norm_factor = " << norm_factor << std::endl;

  if ((pm->ncycle >=1) && (current_time >= tdriv_start) &&
      ((t_since_start < tdriv_duration) || turb_flag != 1))
  {
    // Update the forcing and add momentum and energy only if the driving is continuous or t < tdriv_duration
    par_for("force_OU_process",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_(m,0,k,j,i) = force_tmp1_(m,0,k,j,i);
      force_(m,1,k,j,i) = force_tmp1_(m,1,k,j,i);
      force_(m,2,k,j,i) = force_tmp1_(m,2,k,j,i);
    });

    // Weight the forcing by a Gaussian in each direction, if requested
    // First for the x direction
    if (x_turb_scale_height_ > 0) {
      par_for("force_OU_process",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x1min = mb_size.d_view(m).x1min;
        const Real x1max = mb_size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        force_(m,0,k,j,i) *= std::exp(-SQR(x1v-x_turb_center_)/(2*SQR(x_turb_scale_height_)));
        force_(m,1,k,j,i) *= std::exp(-SQR(x1v-x_turb_center_)/(2*SQR(x_turb_scale_height_)));
        force_(m,2,k,j,i) *= std::exp(-SQR(x1v-x_turb_center_)/(2*SQR(x_turb_scale_height_)));
      });
    }
    // Now for the y direction
    if (y_turb_scale_height_ > 0) {
      par_for("force_OU_process",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x2min = mb_size.d_view(m).x2min;
        const Real x2max = mb_size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
        force_(m,0,k,j,i) *= std::exp(-SQR(x2v-y_turb_center_)/(2*SQR(y_turb_scale_height_)));
        force_(m,1,k,j,i) *= std::exp(-SQR(x2v-y_turb_center_)/(2*SQR(y_turb_scale_height_)));
        force_(m,2,k,j,i) *= std::exp(-SQR(x2v-y_turb_center_)/(2*SQR(y_turb_scale_height_)));
      });
    }
    // Now for the z direction
    if (z_turb_scale_height_ > 0) {
      par_for("force_OU_process",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x3min = mb_size.d_view(m).x3min;
        const Real x3max = mb_size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
        force_(m,0,k,j,i) *= std::exp(-SQR(x3v-z_turb_center_)/(2*SQR(z_turb_scale_height_)));
        force_(m,1,k,j,i) *= std::exp(-SQR(x3v-z_turb_center_)/(2*SQR(z_turb_scale_height_)));
        force_(m,2,k,j,i) *= std::exp(-SQR(x3v-z_turb_center_)/(2*SQR(z_turb_scale_height_)));
      });
    }

    Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1,
                                  Real &sum_t2, Real &sum_t3) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real vol = mb_size.d_view(m).dx1 * mb_size.d_view(m).dx2 * mb_size.d_view(m).dx3;
      Real den = u0(m,IDN,k,j,i);
      if (flag_twofl) {
        den += u0_(m,IDN,k,j,i);
      }
      sum_t0 += den*vol;
      sum_t1 += den*force_(m,0,k,j,i)*vol;
      sum_t2 += den*force_(m,1,k,j,i)*vol;
      sum_t3 += den*force_(m,2,k,j,i)*vol;
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
      Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));


  #if MPI_PARALLEL_ENABLED
    Real m[4], gm[4];
    m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
    MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
  #endif

    par_for("force_remove_net_mom", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_(m,0,k,j,i) -= t1/t0;
      force_(m,1,k,j,i) -= t2/t0;
      force_(m,2,k,j,i) -= t3/t0;
    });

    t0 = 0.0;
    t1 = 0.0;
    Real totvol=0.0;
    bool flag_constant_edot = constant_edot;
    Kokkos::parallel_reduce("net_mom_2", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &totvol_) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real vol = mb_size.d_view(m).dx1 * mb_size.d_view(m).dx2 * mb_size.d_view(m).dx3;

      Real den  = u0(m,IDN,k,j,i);
      Real mom1 = u0(m,IM1,k,j,i);
      Real mom2 = u0(m,IM2,k,j,i);
      Real mom3 = u0(m,IM3,k,j,i);
      if (flag_twofl) {
        den  += u0_(m,IDN,k,j,i);
        mom1 += u0_(m,IM1,k,j,i);
        mom2 += u0_(m,IM2,k,j,i);
        mom3 += u0_(m,IM3,k,j,i);
      }
      Real a1 = force_(m,0,k,j,i);
      Real a2 = force_(m,1,k,j,i);
      Real a3 = force_(m,2,k,j,i);

      if (flag_constant_edot){
        sum_t0 += den*0.5*(a1*a1 + a2*a2 + a3*a3)*dt*vol;
        sum_t1 += (mom1*a1 + mom2*a2 + mom3*a3)*vol;
      } else {
        sum_t0 += 0.5*(a1*a1 + a2*a2 + a3*a3)*dt;
        sum_t1 += 0.0;
      }
      totvol_ += vol;
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1), Kokkos::Sum<Real>(totvol));

  #if MPI_PARALLEL_ENABLED
    m[0] = t0; m[1] = t1; m[2] = totvol;
    MPI_Allreduce(m, gm, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = gm[0]; t1 = gm[1]; totvol = gm[2];
  #endif

    t0 = std::max(t0, 1.0e-20);

    Real m0 = t0;
    Real m1 = t1;

    Real s;
    if (constant_edot) {
      // 1/2 rho (s vdot dt)^2 / dt + rho (s vdot dt).v / dt = dedt
      if (m1 >= 0) {
        s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
      } else {
        s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
      }
    } else {
      // 1/2 rho (s vdot dt)^2 / dt = dedt
      // s = sqrt(dedt * dt / (1/2 rho (vdot dt)^2))
      // s = sqrt(dedt / (1/2 rho vdot^2 dt))
      s = sqrt(dedt/m0);
    }
    if (scale_forcing){
      par_for("force_norm", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        force_(m,0,k,j,i) *= s;
        force_(m,1,k,j,i) *= s;
        force_(m,2,k,j,i) *= s;
      });
    }
  }
  else { // set force to zero
    par_for("force_zero",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_(m,0,k,j,i) = 0.0;
      force_(m,1,k,j,i) = 0.0;
      force_(m,2,k,j,i) = 0.0;
    });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn apply forcing

//
// @brief Adds forcing in the turbulence driver.
//
// This function applies forcing in the turbulence driver based on the provided
// driver and stage. It updates the conserved variables with the applied forces
// and handles both relativistic and non-relativistic cases. Additionally, it
// supports two-fluid and magnetohydrodynamic (MHD) scenarios.
//
// @param pdrive Pointer to the driver object.
// @param stage The current stage of the driver.
// @return TaskStatus indicating the completion status of the task.
//
// The function performs the following main steps:
// 1. Applies forcing to the conserved variables using a parallel loop.
// 2. Handles relativistic transformations if required.
//

TaskStatus TurbulenceDriver::AddForcing(Driver *pdrive, int stage) {

  if (pmy_pack == nullptr) {
    return TaskStatus::complete;
  }

  Mesh *pm = pmy_pack->pmesh;
  if (pm == nullptr) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;


  Real dt = pm->dt;
  Real bdt = (pdrive->beta[stage-1])*dt;
  Real current_time=pm->time;
  Real t_since_start = current_time - tdriv_start;

  EquationOfState *peos;

  DvceArray5D<Real> u0, u0_;
  DvceArray5D<Real> w0, w0_;
  DvceFaceFld4D<Real> *bcc0;
  if (pmy_pack->phydro != nullptr) u0 = (pmy_pack->phydro->u0);
  if (pmy_pack->phydro != nullptr) w0 = (pmy_pack->phydro->w0);
  if (pmy_pack->phydro != nullptr) peos = (pmy_pack->phydro->peos);
  if (pmy_pack->pmhd != nullptr) u0 = (pmy_pack->pmhd->u0);
  if (pmy_pack->pmhd != nullptr) w0 = (pmy_pack->pmhd->w0);
  if (pmy_pack->pmhd != nullptr) bcc0 = &(pmy_pack->pmhd->b0);
  if (pmy_pack->pmhd != nullptr) peos = pmy_pack->pmhd->peos;
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    w0 = (pmy_pack->phydro->w0);
    w0_ = (pmy_pack->pmhd->w0);
    flag_twofl = true;
  }

  bool flag_relativistic = pmy_pack->pcoord->is_special_relativistic;

  auto force_ = force;
  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto eos = peos->eos_data;      // copy-by-value (POD expected)

  if ((current_time >= tdriv_start) &&
      ((t_since_start < tdriv_duration) || turb_flag != 1))
  {
    par_for("push",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real a1 = force_(m,0,k,j,i);
      Real a2 = force_(m,1,k,j,i);
      Real a3 = force_(m,2,k,j,i);

      Real den = w0(m,IDN,k,j,i);
      auto &ux = w0(m,IVX,k,j,i);
      auto &uy = w0(m,IVY,k,j,i);
      auto &uz = w0(m,IVZ,k,j,i);

      Real Fv = (a1*ux + a2*uy + a3*uz);
      if (flag_relativistic) {
        // Compute Lorentz factor
        Real ut = 1. + ux*ux + uy*uy + uz*uz;
        ut = sqrt(ut);
        den /= ut;
        Fv = (a1*ux + a2*uy + a3*uz)/ut;
      }
      u0(m,IM1,k,j,i) += den*a1*bdt;
      u0(m,IM2,k,j,i) += den*a2*bdt;
      u0(m,IM3,k,j,i) += den*a3*bdt;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) += (Fv+0.5*(a1*a1+a2*a2+a3*a3)*bdt)*den*bdt;
        // u0(m,IEN,k,j,i) += Fv*den*bdt;
      }

      if (flag_twofl) {
        den = u0_(m,IDN,k,j,i);
        u0_(m,IM1,k,j,i) += den*a1*bdt;
        u0_(m,IM2,k,j,i) += den*a2*bdt;
        u0_(m,IM3,k,j,i) += den*a3*bdt;
        u0_(m,IEN,k,j,i) += (Fv+0.5*(a1*a1+a2*a2+a3*a3)*bdt)*den*bdt;
        // u0_(m,IEN,k,j,i) += Fv*den*bdt;
      }
    });

    // Relativistic case will require a Lorentz transformation
    if (flag_relativistic) {
      if (pmy_pack->pmhd != nullptr) {
        auto &b = *bcc0;

        par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          // load single state conserved variables
          MHDCons1D u;
          u.d = u0(m,IDN,k,j,i);
          u.mx = u0(m,IM1,k,j,i);
          u.my = u0(m,IM2,k,j,i);
          u.mz = u0(m,IM3,k,j,i);
          u.e = u0(m,IEN,k,j,i);

          u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
          u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
          u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

          // Compute (S^i S_i) (eqn C2)
          Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
          Real b2 = SQR(u.bx) + SQR(u.by) + SQR(u.bz);
          Real rpar = (u.bx*u.mx + u.by*u.my + u.bz*u.mz)/u.d;

          // call c2p function
          // (inline function in ideal_c2p_mhd.hpp file)
          HydPrim1D w;
          bool dfloor_used = false, efloor_used = false;
          //bool vceiling_used = false;
          bool c2p_failure = false;
          int iter_used = 0;
          SingleC2P_IdealSRMHD(u, eos, s2, b2, rpar, w, dfloor_used,
                              efloor_used, c2p_failure, iter_used);
          // apply velocity ceiling if necessary
          Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
          if (lor > eos.gamma_max) {
            //vceiling_used = true;
            Real factor = sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lor) - 1.0));
            w.vx *= factor;
            w.vy *= factor;
            w.vz *= factor;
          }

          // Temporarily store primitives in conserved state
          u0(m,IDN,k,j,i) = w.d;
          u0(m,IM1,k,j,i) = w.vx;
          u0(m,IM2,k,j,i) = w.vy;
          u0(m,IM3,k,j,i) = w.vz;
          u0(m,IEN,k,j,i) = w.e;
        });
      } else {
        auto eos = peos->eos_data;      // copy-by-value (POD expected)

        par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

          // load single state conserved variables
          HydCons1D u;
          u.d = u0(m,IDN,k,j,i);
          u.mx = u0(m,IM1,k,j,i);
          u.my = u0(m,IM2,k,j,i);
          u.mz = u0(m,IM3,k,j,i);
          u.e = u0(m,IEN,k,j,i);

          // Compute (S^i S_i) (eqn C2)
          Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);

          // call c2p function
          // (inline function in ideal_c2p_mhd.hpp file)
          HydPrim1D w;
          bool dfloor_used = false, efloor_used = false;
          //bool vceiling_used = false;
          bool c2p_failure = false;
          int iter_used = 0;
          SingleC2P_IdealSRHyd(u, eos, s2, w, dfloor_used, efloor_used,
                              c2p_failure, iter_used);
          // apply velocity ceiling if necessary
          Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
          if (lor > eos.gamma_max) {
            //vceiling_used = true;
            Real factor = sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lor) - 1.0));
            w.vx *= factor;
            w.vy *= factor;
            w.vz *= factor;
          }

          u0(m,IDN,k,j,i) = w.d;
          u0(m,IM1,k,j,i) = w.vx;
          u0(m,IM2,k,j,i) = w.vy;
          u0(m,IM3,k,j,i) = w.vz;
          u0(m,IEN,k,j,i) = w.e;
        });
      }

      // remove net momentum
      Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
      Kokkos::parallel_reduce("net_mom_3", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &sum_t2,
                    Real &sum_t3) {
        // compute n,k,j,i indices of thread
        int m = (idx)/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks;
        j += js;

        Real u_t = sqrt(1. + u0(m,IVX,k,j,i)*u0(m,IVX,k,j,i) +
                            u0(m,IVY,k,j,i)*u0(m,IVY,k,j,i) +
                            u0(m,IVZ,k,j,i)*u0(m,IVZ,k,j,i));

        Real den = u0(m,IDN,k,j,i)*u_t;
        Real mom1 = den*u0(m,IVX,k,j,i);
        Real mom2 = den*u0(m,IVY,k,j,i);
        Real mom3 = den*u0(m,IVZ,k,j,i);

        sum_t0 += den;
        sum_t1 += mom1;
        sum_t2 += mom2;
        sum_t3 += mom3;
      }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
        Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));

    #if MPI_PARALLEL_ENABLED
      Real m[4], gm[4];
      m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
      MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
    #endif

      // Compute average velocity
      Real uA_x = t1/t0;
      Real uA_y = t2/t0;
      Real uA_z = t3/t0;

      Real uA_0 = sqrt(1. + uA_x*uA_x + uA_y*uA_y + uA_z*uA_z);
      Real betaA = sqrt(uA_x*uA_x + uA_y*uA_y + uA_z*uA_z)/uA_0;

      Real vx = uA_x/uA_0;
      Real vy = uA_y/uA_0;
      Real vz = uA_z/uA_0;

      // LIMIT temp

      if (pmy_pack->pmhd != nullptr) {
        auto b = *bcc0;                        // copy handle by value
        auto eos = peos->eos_data;      // copy-by-value (POD expected)

        par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

          // load single state conserved variables
          MHDPrim1D u;
          u.d = u0(m,IDN,k,j,i);
          u.vx = u0(m,IM1,k,j,i);
          u.vy = u0(m,IM2,k,j,i);
          u.vz = u0(m,IM3,k,j,i);
          u.e = u0(m,IEN,k,j,i);

          u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
          u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
          u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

          HydCons1D u_out;
          SingleP2C_IdealSRMHD(u, eos.gamma, u_out);

          Real en = u_out.d + u_out.e;
          Real sx = u_out.mx;
          Real sy = u_out.my;
          Real sz = u_out.mz;

          Real dens = u_out.d;

          auto &w = u;

          Real lorentz = sqrt(1. + w.vx*w.vx + w.vy*w.vy + w.vz*w.vz);
          Real beta = sqrt(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)/lorentz;

          u0(m,IDN,k,j,i) = dens;  // *uA_0*(1.-beta*betaA);

          // Does not require knowledge of v
          u0(m,IEN,k,j,i) = uA_0*en - uA_0*(sx*vx + sy*vy + sz*vz);
          u0(m,IEN,k,j,i) -= u0(m,IDN,k,j,i);

          u0(m,IM1,k,j,i) = sx + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vx;
          u0(m,IM2,k,j,i) = sy + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vy;
          u0(m,IM3,k,j,i) = sz + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vz;

          u0(m,IM1,k,j,i) -= uA_0*en*vx;
          u0(m,IM2,k,j,i) -= uA_0*en*vy;
          u0(m,IM3,k,j,i) -= uA_0*en*vz;
        });
      } else {
        auto eos = peos->eos_data;      // copy-by-value (POD expected)

        par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

          // load single state conserved variables
          HydPrim1D u;
          u.d = u0(m,IDN,k,j,i);
          u.vx = u0(m,IM1,k,j,i);
          u.vy = u0(m,IM2,k,j,i);
          u.vz = u0(m,IM3,k,j,i);
          u.e = u0(m,IEN,k,j,i);

          HydCons1D u_out;
          SingleP2C_IdealSRHyd(u, eos.gamma, u_out);

          Real en = u_out.d + u_out.e;
          Real sx = u_out.mx;
          Real sy = u_out.my;
          Real sz = u_out.mz;

          Real dens = u_out.d;

          auto &w = u;

          Real lorentz = sqrt(1. + w.vx*w.vx + w.vy*w.vy + w.vz*w.vz);
          Real beta = sqrt(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)/lorentz;

          u0(m,IDN,k,j,i) = dens;  //*uA_0*(1.-beta*betaA);

          // Does not require knowledge of v
          u0(m,IEN,k,j,i) = uA_0*en - uA_0*(sx*vx + sy*vy + sz*vz);
          u0(m,IEN,k,j,i) -= u0(m,IDN,k,j,i);
          u0(m,IM1,k,j,i) = sx + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vx;
          u0(m,IM2,k,j,i) = sy + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vy;
          u0(m,IM3,k,j,i) = sz + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vz;
          u0(m,IM1,k,j,i) -= uA_0*en*vx;
          u0(m,IM2,k,j,i) -= uA_0*en*vy;
          u0(m,IM3,k,j,i) -= uA_0*en*vz;
        });
      }

    } // end relativistic case

  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn EnsureBasisSize()
// \brief Detect mesh/AMR changes and ensure forcing basis and arrays match current mesh.
//        Recomputes basis for all blocks when change is detected.

TaskStatus TurbulenceDriver::EnsureBasisSize(Driver *pdrive, int stage) {

  if (pmy_pack == nullptr) {
    return TaskStatus::complete;
  }

  Mesh *pm = pmy_pack->pmesh;
  if (pm == nullptr) return TaskStatus::complete;

  // Update cached domain offsets in case AMR has modified the root-grid geometry.
  domain_x1min = pm->mesh_size.x1min;
  domain_x2min = pm->mesh_size.x2min;
  domain_x3min = pm->mesh_size.x3min;

  // --- change detection (idempotent) ---
  int nmb = pmy_pack->nmb_thispack;
  bool needs_resize = false;
  if (nmb != current_nmb_) needs_resize = true;
  if (pm->adaptive && pm->pmr != nullptr) {
    if (pm->pmr->nmb_created != last_nmb_created_ ||
        pm->pmr->nmb_deleted != last_nmb_deleted_) {
      needs_resize = true;
    }
  }
  if (!needs_resize) return TaskStatus::complete;

  // --- resize/rebuild path ---

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;

  int old_nmb = current_nmb_;

  // Always recompute for all blocks after any detected geometry change or AMR.
  const int m_start = 0;

  // Reallocate arrays only if MeshBlock count changed
  if (force.extent(0) != nmb) {

    auto force_old = force;
    auto force_tmp1_old = force_tmp1;
    auto force_tmp2_old = force_tmp2;
    auto xcos_old = xcos;
    auto xsin_old = xsin;
    auto ycos_old = ycos;
    auto ysin_old = ysin;
    auto zcos_old = zcos;
    auto zsin_old = zsin;

    Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(force_tmp1, nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(force_tmp2, nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(xcos, nmb, mode_count, ncells1);
    Kokkos::realloc(xsin, nmb, mode_count, ncells1);
    Kokkos::realloc(ycos, nmb, mode_count, ncells2);
    Kokkos::realloc(ysin, nmb, mode_count, ncells2);
    Kokkos::realloc(zcos, nmb, mode_count, ncells3);
    Kokkos::realloc(zsin, nmb, mode_count, ncells3);

    int copy_n = std::min(old_nmb, nmb);

    if (copy_n > 0) {
      Kokkos::deep_copy(
          Kokkos::subview(force, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(force_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL));

      Kokkos::deep_copy(
          Kokkos::subview(force_tmp1, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(force_tmp1_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL));

      Kokkos::deep_copy(
          Kokkos::subview(force_tmp2, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(force_tmp2_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL));
      Kokkos::deep_copy(
          Kokkos::subview(xcos, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(xcos_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL));
      Kokkos::deep_copy(
          Kokkos::subview(xsin, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(xsin_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL));
      Kokkos::deep_copy(
          Kokkos::subview(ycos, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(ycos_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL));
      Kokkos::deep_copy(
          Kokkos::subview(ysin, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(ysin_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL));
      Kokkos::deep_copy(
          Kokkos::subview(zcos, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(zcos_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL));
      Kokkos::deep_copy(
          Kokkos::subview(zsin, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL),
          Kokkos::subview(zsin_old, std::make_pair(0, copy_n), Kokkos::ALL, Kokkos::ALL));
    }
  }

  // CRITICAL: Do NOT call Initialize() which would reset the turbulence state
  // Instead, recompute basis functions while preserving mode amplitudes (aka_, akb_)


  // Zero out force arrays for all blocks; we will rebuild everywhere
  auto force_tmp1_ = force_tmp1;
  auto force_tmp2_ = force_tmp2;
  par_for("force_resize_zero", DevExeSpace(),
          m_start, nmb-1, 0, 2, 0, ncells3-1, 0, ncells2-1, 0, ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_tmp1_(m,n,k,j,i) = 0.0;
    force_tmp2_(m,n,k,j,i) = 0.0;
  });

  // Recompute basis functions for new mesh blocks (preserving wavenumbers)

  if (pmy_pack->pmb == nullptr) {
    return TaskStatus::complete;
  }

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;
  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  // MeshBlock sizes are updated on host during AMR. Make sure device view is fresh.
  auto size_view = pmy_pack->pmb->mb_size;   // copy the DualView handle
  size_view.template modify<HostMemSpace>();
  size_view.template sync<DevExeSpace>();
  const int drivingtype = driving_type;      // value, not reference
  const bool tile_enabled = tile_driving;
  const int tile_nx_local = tile_nx;
  const int tile_ny_local = tile_ny;
  const int tile_nz_local = tile_nz;
  const Real tile_lx_local = tile_lx;
  const Real tile_ly_local = tile_ly;
  const Real tile_lz_local = tile_lz;
  const Real inv_tile_lx_local = inv_tile_lx;
  const Real inv_tile_ly_local = inv_tile_ly;
  const Real inv_tile_lz_local = inv_tile_lz;
  const Real domain_x1min_local = domain_x1min;
  const Real domain_x2min_local = domain_x2min;
  const Real domain_x3min_local = domain_x3min;

  // Recompute x-direction basis functions
  par_for("xsin/xcos_resize", DevExeSpace(), m_start, nmb-1, 0, mode_count-1, is, ie,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size_view.d_view(m).x1min;
    Real &x1max = size_view.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    Real arg = x1v;
    if (tile_enabled && tile_nx_local > 1) {
      Real rel = x1v - domain_x1min_local;
      int tile_i = static_cast<int>(floor(rel * inv_tile_lx_local));
      tile_i = (tile_i < 0) ? 0 : ((tile_i >= tile_nx_local) ? tile_nx_local - 1 : tile_i);
      Real tile_origin = domain_x1min_local + tile_i * tile_lx_local;
      arg = x1v - tile_origin;
    }
    xsin_(m,n,i) = sin(k1v*arg);
    xcos_(m,n,i) = cos(k1v*arg);
  });

  // Recompute y-direction basis functions
  par_for("ysin/ycos_resize", DevExeSpace(), m_start, nmb-1, 0, mode_count-1, js, je,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size_view.d_view(m).x2min;
    Real &x2max = size_view.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    Real arg = x2v;
    if (tile_enabled && tile_ny_local > 1) {
      Real rel = x2v - domain_x2min_local;
      int tile_j = static_cast<int>(floor(rel * inv_tile_ly_local));
      tile_j = (tile_j < 0) ? 0 : ((tile_j >= tile_ny_local) ? tile_ny_local - 1 : tile_j);
      Real tile_origin = domain_x2min_local + tile_j * tile_ly_local;
      arg = x2v - tile_origin;
    }
    ysin_(m,n,j) = sin(k2v*arg);
    ycos_(m,n,j) = cos(k2v*arg);
    if (ncells2-1 == 0) {
      ysin_(m,n,j) = 0.0;
      ycos_(m,n,j) = 1.0;
    }
  });

  // Recompute z-direction basis functions
  par_for("zsin/zcos_resize", DevExeSpace(), m_start, nmb-1, 0, mode_count-1, ks, ke,
  KOKKOS_LAMBDA(int m, int n, int k) {
    Real &x3min = size_view.d_view(m).x3min;
    Real &x3max = size_view.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real k3v = kz_mode_.d_view(n);
    Real arg = x3v;
    if (tile_enabled && tile_nz_local > 1) {
      Real rel = x3v - domain_x3min_local;
      int tile_k = static_cast<int>(floor(rel * inv_tile_lz_local));
      tile_k = (tile_k < 0) ? 0 : ((tile_k >= tile_nz_local) ? tile_nz_local - 1 : tile_k);
      Real tile_origin = domain_x3min_local + tile_k * tile_lz_local;
      arg = x3v - tile_origin;
    }
    zsin_(m,n,k) = sin(k3v*arg);
    zcos_(m,n,k) = cos(k3v*arg);
    if (ncells3-1 == 0 || (drivingtype == 1)) {
      zsin_(m,n,k) = 0.0;
      zcos_(m,n,k) = 1.0;
    }
  });

  // Recompute forcing field using PRESERVED mode amplitudes (aka_, akb_)
  auto aka_ = aka;
  auto akb_ = akb;
  aka_.template sync<DevExeSpace>();
  akb_.template sync<DevExeSpace>();
  int mode_count_ = mode_count;
  par_for("force_recalc_resize", DevExeSpace(), m_start, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // No tile index needed - same coefficients used everywhere
    // The basis functions already use tile-local coords for periodicity
    for (int n = 0; n < mode_count_; n++) {
      Real forc_real = (xcos_(m,n,i)*ycos_(m,n,j) - xsin_(m,n,i)*ysin_(m,n,j)) * zcos_(m,n,k) -
                      (xsin_(m,n,i)*ycos_(m,n,j) + xcos_(m,n,i)*ysin_(m,n,j)) * zsin_(m,n,k);
      Real forc_imag = (ycos_(m,n,j)*zsin_(m,n,k) + ysin_(m,n,j)*zcos_(m,n,k)) * xcos_(m,n,i) +
                      (ycos_(m,n,j)*zcos_(m,n,k) - ysin_(m,n,j)*zsin_(m,n,k)) * xsin_(m,n,i);
      for (int dir = 0; dir < 3; dir++) {
        force_tmp2_(m,dir,k,j,i) += aka_.d_view(dir,n)*forc_real -
                                    akb_.d_view(dir,n)*forc_imag;
      }
    }
  });

  // Copy reconstructed field back into working arrays
  Kokkos::deep_copy(force_tmp1, force_tmp2);
  Kokkos::deep_copy(force, force_tmp2);

  // Update tracking variables
  current_nmb_ = nmb;

  if (pm->adaptive && pm->pmr != nullptr) {
    last_nmb_created_ = pm->pmr->nmb_created;
    last_nmb_deleted_ = pm->pmr->nmb_deleted;
  }

  return TaskStatus::complete;
}
