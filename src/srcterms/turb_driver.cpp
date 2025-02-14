//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceDriver

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

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
  aka("aka",1,1),akb("zssc",1,1),
  kx_mode("kx_mode",1),ky_mode("ky_mode",1),kz_mode("kz_mode",1),
  xcos("xcos",1,1,1),xsin("xsin",1,1,1),ycos("ycos",1,1,1),
  ysin("ysin",1,1,1),zcos("zcos",1,1,1),zsin("zsin",1,1,1) {
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp1, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp2, nmb, 3, ncells3, ncells2, ncells1);

  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("turb_driving", "nlow", 1);
  nhigh = pin->GetOrAddInteger("turb_driving", "nhigh", 3);
  // Peak of power when spectral form is parabolic, in units of 2*(PI/L)
  kpeak = pin->GetOrAddReal("turb_driving", "kpeak", M_PI*(nlow+nhigh));
  // spect form - 1 for parabola, 2 for power-law
  spect_form = pin->GetOrAddInteger("turb_driving", "spect_form", 1);
  // driving type - 0 for 3D isotropic, 1 for xy plane
  driving_type = pin->GetOrAddInteger("turb_driving", "driving_type", 0);
  // min kz zero should be 0 for including kz modes and 1 for not including
  min_kz = pin->GetOrAddInteger("turb_driving", "min_kz", 0);
  max_kz = pin->GetOrAddInteger("turb_driving", "max_kz", nhigh);
  // Seed for random number generator
  random_seed = pin->GetOrAddInteger("turb_driving", "random_seed", 1);
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("turb_driving", "expo", 5.0/3.0);
  exp_prp = pin->GetOrAddReal("turb_driving", "exp_prp", 5.0/3.0);
  exp_prl = pin->GetOrAddReal("turb_driving", "exp_prl", 0.0);
  // energy injection rate
  dedt = pin->GetOrAddReal("turb_driving", "dedt", 1.0);
  // correlation time - time over which the OU process decorrelates
  tcorr = pin->GetOrAddReal("turb_driving", "tcorr", 1.0);
  // update time for the turbulence driver - we update the force field every dt_turb_update
  dt_turb_update=pin->GetOrAddReal("turb_driving","dt_turb_update",tcorr/100.0);
  // We make sure that the code time is within dt_turb_thresh of an integer multiple of dt_turb_update
  dt_turb_thresh=pin->GetOrAddReal("turb_driving","dt_turb_thresh",dt_turb_update/100.); //We'll match the code time-step within this value of dt_turb_update steps
  sol_fraction=pin->GetOrAddReal("turb_driving","sol_fraction",1.0); // To store fraction of energy in solenoidal modes

  // spatially varying driving
  x_turb_scale_height = pin->GetOrAddReal("turb_driving", "x_turb_scale_height", -1.0);
  y_turb_scale_height = pin->GetOrAddReal("turb_driving", "y_turb_scale_height", -1.0);
  z_turb_scale_height = pin->GetOrAddReal("turb_driving", "z_turb_scale_height", -1.0);
  x_turb_center = pin->GetOrAddReal("turb_driving", "x_turb_center", 0.0);
  y_turb_center = pin->GetOrAddReal("turb_driving", "y_turb_center", 0.0);
  z_turb_center = pin->GetOrAddReal("turb_driving", "z_turb_center", 0.0);

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
    std::cout << "Initializing turbulence driving module" << std::endl <<
    " dedt = " << dedt << " tcorr = " << tcorr << " dt_turb_update = " << dt_turb_update << std::endl;
  }
  n_turb_updates_yet = 0;

  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;

  mode_count = 0;

  int nkx, nky, nkz;
  Real nsqr;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
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

  Kokkos::realloc(aka, 3, mode_count); // Amplitude of real component
  Kokkos::realloc(akb, 3, mode_count); // Amplitude of imaginary component

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
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  auto force_tmp1_ = force_tmp1;
  auto force_tmp2_ = force_tmp2;
  par_for("force_init_pgen",DevExeSpace(),
          0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_tmp1_(m,n,k,j,i) = 0.0;
  });

  rstate.idum = -random_seed;

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  int nmode = 0;
  int nkx, nky, nkz;
  Real nsqr;
  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
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

  auto &size = pmy_pack->pmb->mb_size;
  auto &drivingtype = driving_type;

  par_for("xsin/xcos", DevExeSpace(),0,nmb-1,0,mode_count-1,is,ie,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    xsin_(m,n,i) = sin(k1v*x1v);
    xcos_(m,n,i) = cos(k1v*x1v);
  });

  par_for("ysin/ycos", DevExeSpace(),0,nmb-1,0,mode_count-1,js,je,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    ysin_(m,n,j) = sin(k2v*x2v);
    ycos_(m,n,j) = cos(k2v*x2v);
    if (ncells2-1 == 0) {
      ysin_(m,n,j) = 0.0;
      ycos_(m,n,j) = 1.0;
    }
  });

  par_for("zsin/zcos", DevExeSpace(),0,nmb-1,0,mode_count-1,ks,ke,
  KOKKOS_LAMBDA(int m, int n, int k) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real k3v = kz_mode_.d_view(n);
    zsin_(m,n,k) = sin(k3v*x3v);
    zcos_(m,n,k) = cos(k3v*x3v);
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
  //  We initialize the modes and update the forcing before the time integration loop
  auto id_init = tl->AddTask(&TurbulenceDriver::InitializeModes, this, start);
  auto id_add  = tl->AddTask(&TurbulenceDriver::UpdateForcing, this, id_init);
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
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real current_time=pm->time;
  int n_turb_updates_reqd = (int) (current_time/dt_turb_update) + 1;

  int nlow_sqr = SQR(nlow);
  int nhigh_sqr = SQR(nhigh);
  auto mode_count_ = mode_count;

  auto aka_ = aka;
  auto akb_ = akb;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

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

  if ((current_time < tdriv_duration) || turb_flag != 1){ // Update the forcing only if the driving is continuous or t<tdriv_duration

    for(int i_turb_update = n_turb_updates_yet; i_turb_update < n_turb_updates_reqd; i_turb_update++){
      if (global_variable::my_rank == 0) std::cout << "i_turb_update = " << i_turb_update << std::endl;

      auto force_tmp2_ = force_tmp2;
      int &nmb = pmy_pack->nmb_thispack;

      // Zero out new force array
      par_for("force_init", DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        force_tmp2_(m,n,k,j,i) = 0.0;
      });


      int no_dir=3;
      int nmode = 0;
      int nkx, nky, nkz, nsqr;

      for (nkx = 0; nkx <= nhigh; nkx++) {
        for (nky = 0; nky <= nhigh; nky++) {
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


              // Generate Fourier amplitudes

              if (driving_type == 0) {

              kiso = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
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
              Real ka = 0.0;
              Real kb = 0.0;

              for (int dir = 0; dir < no_dir; dir ++){
                aka_.h_view(dir,nmode) = norm*RanGaussianSt(&(rstate));
                akb_.h_view(dir,nmode) = norm*RanGaussianSt(&(rstate));

                // ka = ka + k[dir]*aka_.h_view(dir,nmode);
                // kb = kb + k[dir]*akb_.h_view(dir,nmode);
                ka = ka + k[dir]*akb_.h_view(dir,nmode);
                kb = kb + k[dir]*aka_.h_view(dir,nmode);
              }

              // Now decompose into solenoidal/compressive modes
              if(norm > 0.){
                for (int dir = 0; dir < no_dir; dir ++){
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

      auto xcos_ = xcos;
      auto xsin_ = xsin;
      auto ycos_ = ycos;
      auto ysin_ = ysin;
      auto zcos_ = zcos;
      auto zsin_ = zsin;

      for (int n=0; n<mode_count_; n++) {
        par_for("force_compute", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          Real forc_real = ( xcos_(m,n,i)*ycos_(m,n,j) - xsin_(m,n,i)*ysin_(m,n,j) ) * zcos_(m,n,k) -
                      ( xsin_(m,n,i)*ycos_(m,n,j) + xcos_(m,n,i)*ysin_(m,n,j) ) * zsin_(m,n,k);
          Real forc_imag = ( ycos_(m,n,j)*zsin_(m,n,k) + ysin_(m,n,j)*zcos_(m,n,k) ) * xcos_(m,n,i) +
                      ( ycos_(m,n,j)*zcos_(m,n,k) - ysin_(m,n,j)*zsin_(m,n,k) ) * xsin_(m,n,i);
          for (int dir = 0; dir < no_dir; dir ++){
            force_tmp2_(m,dir,k,j,i) += aka_.d_view(dir,n)*forc_real - akb_.d_view(dir,n)*forc_imag;
          }
        });
      }
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
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  Real dt = pm->dt;
  Real current_time=pm->time;

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
  auto &size = pmy_pack->pmb->mb_size;

  auto x_turb_scale_height_ = x_turb_scale_height;
  auto y_turb_scale_height_ = y_turb_scale_height;
  auto z_turb_scale_height_ = z_turb_scale_height;
  auto x_turb_center_ = x_turb_center;
  auto y_turb_center_ = y_turb_center;
  auto z_turb_center_ = z_turb_center;

  // Copy values of force_tmp1 into force array
  // perform operations such as normalisation,
  // momentum subtraction directly on the force array

  if ((pm->ncycle >=1) && ((current_time < tdriv_duration) || turb_flag != 1))
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
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
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
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
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
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
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
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
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
    Kokkos::parallel_reduce("net_mom_2", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &totvol_) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

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

      sum_t0 += den*0.5*(a1*a1+a2*a2+a3*a3)*dt*vol;
      sum_t1 += (mom1*a1+mom2*a2+mom3*a3)*vol;
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
    if (m1 >= 0) {
      s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
    } else {
      s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
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
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  Real dt = pm->dt;
  Real bdt = (pdrive->beta[stage-1])*dt;
  Real current_time=pm->time;

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

  auto &eos = peos->eos_data;

  if ((current_time < tdriv_duration) || turb_flag != 1)
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
        u0_(m,IEN,k,j,i) += (Fv+0.5*(a1*a1+a2*a2+a3*a3)*dt)*den*bdt;
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
        auto &eos = peos->eos_data;

        par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
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
        auto &b = *bcc0;
        auto &eos = peos->eos_data;

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
        auto &eos = peos->eos_data;

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
