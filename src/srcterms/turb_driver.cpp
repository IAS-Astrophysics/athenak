//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceDriver

#include <limits>
#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion_neutral.hpp"
#include "driver/driver.hpp"
#include "utils/random.hpp"
#include "turb_driver.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriver::TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  force("force",1,1,1,1,1),
  force_new("force_new",1,1,1,1,1),
  x1sin("x1sin",1,1,1),
  x1cos("x1cos",1,1,1),
  x2sin("x2sin",1,1,1),
  x2cos("x2cos",1,1,1),
  x3sin("x3sin",1,1,1),
  x3cos("x3cos",1,1,1),
  amp1("amp1",1,1),
  amp2("amp2",1,1),
  amp3("amp3",1,1)
{
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_new, nmb, 3, ncells3, ncells2, ncells1);

  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("turb_driving","nlow",1);
  nhigh = pin->GetOrAddInteger("turb_driving","nhigh",2);
  if (ncells3>1) { // 3D
    ntot = (nhigh+1)*(nhigh+1)*(nhigh+1);
    nwave = 8;
  } else if (ncells2>1) { // 2D
    ntot = (nhigh+1)*(nhigh+1);
    nwave = 4;
  } else { // 1D
    ntot = (nhigh+1);
    nwave = 2;
  }
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("turb_driving","expo",5.0/3.0);
  // energy injection rate
  dedt = pin->GetOrAddReal("turb_driving","dedt",0.0);
  // correlation time
  tcorr = pin->GetOrAddReal("turb_driving","tcorr",0.0); 

  Kokkos::realloc(x1sin, nmb, ntot, ncells1);
  Kokkos::realloc(x1cos, nmb, ntot, ncells1);
  Kokkos::realloc(x2sin, nmb, ntot, ncells2);
  Kokkos::realloc(x2cos, nmb, ntot, ncells2);
  Kokkos::realloc(x3sin, nmb, ntot, ncells3);
  Kokkos::realloc(x3cos, nmb, ntot, ncells3);

  Kokkos::realloc(amp1, ntot, nwave);
  Kokkos::realloc(amp2, ntot, nwave);
  Kokkos::realloc(amp3, ntot, nwave);
}

//----------------------------------------------------------------------------------------
// destructor
  
TurbulenceDriver::~TurbulenceDriver()
{
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeModeEvolutionTasks
//  \brief Includes task in the operator split task list that constructs new modes with
//  random amplitudes and phases that can be used to evolve the force via an O-U process
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeInitializeModesTask(TaskList &tl, TaskID start)
{
  auto id = tl.AddTask(&TurbulenceDriver::InitializeModes, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeAddForcingTask
//  \brief includes task in the stage_run task list for adding random forcing to fluid
//  as an explicit source terms in each stage of integrator
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeAddForcingTask(TaskList &tl, TaskID start)
{   
  // These must be inserted after update task, but before send_u
  if (pmy_pack->pionn == nullptr) {
    if (pmy_pack->phydro != nullptr) {
      auto id = tl.InsertTask(&TurbulenceDriver::AddForcing, this, 
                         pmy_pack->phydro->id.flux, pmy_pack->phydro->id.expl);
    }
    if (pmy_pack->pmhd != nullptr) {
      auto id = tl.InsertTask(&TurbulenceDriver::AddForcing, this, 
                         pmy_pack->pmhd->id.flux, pmy_pack->pmhd->id.expl);
    }
  } else {
    auto id = tl.InsertTask(&TurbulenceDriver::AddForcing, this,
                       pmy_pack->pionn->id.n_flux, pmy_pack->pionn->id.n_expl);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn InitializeModes()
// \brief Initializes driving, and so is only executed once at start of calc.
// Cannot be included in constructor since (it seems) Kokkos::par_for not allowed in cons.

TaskStatus TurbulenceDriver::InitializeModes(Driver *pdrive, int stage)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Real lx = pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min;
  Real ly = pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min;
  Real lz = pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min;
  Real dkx = 2.0*M_PI/lx;
  Real dky = 2.0*M_PI/ly;
  Real dkz = 2.0*M_PI/lz;

  int nw2 = 1; int nw3 = 1;
  if (ncells2>1) {
    nw2 = nhigh+1;
  }
  if (ncells3>1) {
    nw3 = nhigh+1;
  }
  int nw23 = nw3*nw2;

  // On first call to this function, initialize seeds, sin/cos arrays
  if (first_time) {

    // initialize force to zero
    int &nmb = pmy_pack->nmb_thispack;
    auto force_ = force;
    par_for("force_init", DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
      {
        force_(m,n,k,j,i) = 0.0;
      }
    );

    // initalize seeds
    int &nt = ntot;
    seed = -1;

    // Initialize sin and cos arrays
    // bad design: requires saving sin/cos during restarts
    auto &size = pmy_pack->pmb->mb_size;
    auto x1sin_ = x1sin;
    auto x1cos_ = x1cos;
    par_for("kx_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells1-1,
      KOKKOS_LAMBDA(int m, int n, int i)
      { 
        int nk1 = n/nw23;
        Real kx = nk1*dkx;
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      
        x1sin_(m,n,i) = sin(kx*x1v);
        x1cos_(m,n,i) = cos(kx*x1v);
      }
    );

    auto x2sin_ = x2sin;
    auto x2cos_ = x2cos;
    par_for("ky_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells2-1,
      KOKKOS_LAMBDA(int m, int n, int j)
      { 
        int nk1 = n/nw23;
        int nk2 = (n - nk1*nw23)/nw2;
        Real ky = nk2*dky;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        x2sin_(m,n,j) = sin(ky*x2v);
        x2cos_(m,n,j) = cos(ky*x2v);
      }
    );

    auto x3sin_ = x3sin;
    auto x3cos_ = x3cos;
    par_for("kz_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells3-1,
      KOKKOS_LAMBDA(int m, int n, int k)
      { 
        int nk1 = n/nw23;
        int nk2 = (n - nk1*nw23)/nw2;
        int nk3 = n - nk1*nw23 - nk2*nw2;
        Real kz = nk3*dkz;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

        x3sin_(m,n,k) = sin(kz*x3v);
        x3cos_(m,n,k) = cos(kz*x3v);
      }
    );
    first_time = false;

  // if this is NOT the first call, evolve force according to O-U process, using "new"
  // force computed last time step and still stored in "force_new" array
  } else {
    // TODO(leva): if not first call, there should also be initializtion of x#sin and x#cos,
    //             unless they are saved in the restart.
    Real fcorr=0.0;
    Real gcorr=1.0;
    if ((pmy_pack->pmesh->time > 0.0) and (tcorr > 0.0)) { 
      fcorr=exp(-(last_dt/tcorr));
      gcorr=sqrt(1.0-fcorr*fcorr);
    } 
  
    auto force_ = force;
    auto force_new_ = force_new;
    int &nmb = pmy_pack->nmb_thispack;
    par_for("OU_process", DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
      {
        force_(m,n,k,j,i) = fcorr*force_(m,n,k,j,i) + gcorr*force_new_(m,n,k,j,i);
      }
    );
    last_dt = pmy_pack->pmesh->dt;  // store this dt for call to this fn next timestep
  }

  // Now compute new force using new random amplitudes and phases

  // Zero out new force array
  auto force_new_ = force_new;
  int &nmb = pmy_pack->nmb_thispack;
  par_for("forcing_init", DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      force_new_(m,0,k,j,i) = 0.0;
      force_new_(m,1,k,j,i) = 0.0;
      force_new_(m,2,k,j,i) = 0.0;
    }
  );

  int nlow_sq  = SQR(nlow);
  int nhigh_sq = SQR(nhigh);

  int &nt = ntot;
  Real &ex = expo;
  auto amp1_ = amp1;
  auto amp2_ = amp2;
  auto amp3_ = amp3;

  // TODO(leva): move this for loop to the host

  for (int n=0; n<=nt-1; n++) {
    int nk1 = n/nw23;
    int nk2 = (n - nk1*nw23)/nw2;
    int nk3 = n - nk1*nw23 - nk2*nw2;
    Real kx = nk1*dkx;
    Real ky = nk2*dky;
    Real kz = nk3*dkz;

    int nsq = nk1*nk1 + nk2*nk2 + nk3*nk3;

    Real kmag = sqrt(kx*kx + ky*ky + kz*kz);
    Real norm = 1.0/pow(kmag,(ex+2.0)/2.0); 

    // TODO(leva): check whether those coefficients are needed
    //if(nk1 > 0) norm *= 0.5;
    //if(nk2 > 0) norm *= 0.5;
    //if(nk3 > 0) norm *= 0.5;

    if (nsq >= nlow_sq && nsq <= nhigh_sq) {
      //Generate Fourier amplitudes
      if(nk3 != 0){
        Real ikz = 1.0/(dkz*((Real) nk3));

        amp1_.h_view(n,0) = RanGaussian(&(seed));
        amp1_.h_view(n,1) = RanGaussian(&(seed));
        amp1_.h_view(n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,3) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,5) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,7) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));

        amp2_.h_view(n,0) = RanGaussian(&(seed));
        amp2_.h_view(n,1) = RanGaussian(&(seed));
        amp2_.h_view(n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp2_.h_view(n,3) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp2_.h_view(n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp2_.h_view(n,5) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
        amp2_.h_view(n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));
        amp2_.h_view(n,7) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));

        // incompressibility
        amp3_.h_view(n,0) =  ikz*( kx*amp1_.h_view(n,5) + ky*amp2_.h_view(n,3));
        amp3_.h_view(n,1) = -ikz*( kx*amp1_.h_view(n,4) + ky*amp2_.h_view(n,2));
        amp3_.h_view(n,2) =  ikz*( kx*amp1_.h_view(n,7) - ky*amp2_.h_view(n,1));
        amp3_.h_view(n,3) =  ikz*(-kx*amp1_.h_view(n,6) + ky*amp2_.h_view(n,0));
        amp3_.h_view(n,4) =  ikz*(-kx*amp1_.h_view(n,1) + ky*amp2_.h_view(n,7));
        amp3_.h_view(n,5) =  ikz*( kx*amp1_.h_view(n,0) - ky*amp2_.h_view(n,6));
        amp3_.h_view(n,6) = -ikz*( kx*amp1_.h_view(n,3) + ky*amp2_.h_view(n,5));
        amp3_.h_view(n,7) =  ikz*( kx*amp1_.h_view(n,2) + ky*amp2_.h_view(n,4));

      } else if(nk2 != 0){ // kz == 0
        Real iky = 1.0/(dky*((Real) nk2));

        amp1_.h_view(n,0) = RanGaussian(&(seed));
        amp1_.h_view(n,2) = RanGaussian(&(seed));
        amp1_.h_view(n,4) = (nk1 == 0) ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,6) = (nk1 == 0) ? 0.0 :RanGaussian(&(seed));
        amp1_.h_view(n,1) = 0.0;
        amp1_.h_view(n,3) = 0.0;
        amp1_.h_view(n,5) = 0.0;
        amp1_.h_view(n,7) = 0.0;

        amp3_.h_view(n,0) = RanGaussian(&(seed));
        amp3_.h_view(n,2) = RanGaussian(&(seed));
        amp3_.h_view(n,4) = (nk1 == 0) ? 0.0 : RanGaussian(&(seed));
        amp3_.h_view(n,6) = (nk1 == 0) ? 0.0 : RanGaussian(&(seed));
        amp3_.h_view(n,1) = 0.0;
        amp3_.h_view(n,3) = 0.0;
        amp3_.h_view(n,5) = 0.0;
        amp3_.h_view(n,7) = 0.0;

        // incompressibility
        amp2_.h_view(n,0) =  iky*kx*amp1_.h_view(n,6);
        amp2_.h_view(n,2) = -iky*kx*amp1_.h_view(n,4);
        amp2_.h_view(n,4) = -iky*kx*amp1_.h_view(n,2);
        amp2_.h_view(n,6) =  iky*kx*amp1_.h_view(n,0);
        amp2_.h_view(n,1) = 0.0;
        amp2_.h_view(n,3) = 0.0;
        amp2_.h_view(n,5) = 0.0;
        amp2_.h_view(n,7) = 0.0;

      } else {// kz == ky == 0, kx != 0 by initial if statement
        amp3_.h_view(n,0) = RanGaussian(&(seed));
        amp3_.h_view(n,4) = RanGaussian(&(seed));
        amp3_.h_view(n,1) = 0.0;
        amp3_.h_view(n,2) = 0.0;
        amp3_.h_view(n,3) = 0.0;
        amp3_.h_view(n,5) = 0.0;
        amp3_.h_view(n,6) = 0.0;
        amp3_.h_view(n,7) = 0.0;

        amp2_.h_view(n,0) = RanGaussian(&(seed));
        amp2_.h_view(n,4) = RanGaussian(&(seed));
        amp2_.h_view(n,1) = 0.0;
        amp2_.h_view(n,2) = 0.0;
        amp2_.h_view(n,3) = 0.0;
        amp2_.h_view(n,5) = 0.0;
        amp2_.h_view(n,6) = 0.0;
        amp2_.h_view(n,7) = 0.0;

        // incompressibility
        amp1_.h_view(n,0) = 0.0;
        amp1_.h_view(n,4) = 0.0;
        amp1_.h_view(n,1) = 0.0;
        amp1_.h_view(n,2) = 0.0;
        amp1_.h_view(n,3) = 0.0;
        amp1_.h_view(n,5) = 0.0;
        amp1_.h_view(n,6) = 0.0;
        amp1_.h_view(n,7) = 0.0;
      }

      amp1_.h_view(n,0) *= norm;
      amp1_.h_view(n,4) *= norm;
      amp1_.h_view(n,1) *= norm;
      amp1_.h_view(n,2) *= norm;
      amp1_.h_view(n,3) *= norm;
      amp1_.h_view(n,5) *= norm;
      amp1_.h_view(n,6) *= norm;
      amp1_.h_view(n,7) *= norm;

      amp2_.h_view(n,0) *= norm;
      amp2_.h_view(n,4) *= norm;
      amp2_.h_view(n,1) *= norm;
      amp2_.h_view(n,2) *= norm;
      amp2_.h_view(n,3) *= norm;
      amp2_.h_view(n,5) *= norm;
      amp2_.h_view(n,6) *= norm;
      amp2_.h_view(n,7) *= norm;

      amp3_.h_view(n,0) *= norm;
      amp3_.h_view(n,4) *= norm;
      amp3_.h_view(n,1) *= norm;
      amp3_.h_view(n,2) *= norm;
      amp3_.h_view(n,3) *= norm;
      amp3_.h_view(n,5) *= norm;
      amp3_.h_view(n,6) *= norm;
      amp3_.h_view(n,7) *= norm;
    }
  }

  // for index DualArray, mark host views as modified, and then sync to device array
  amp1_.template modify<HostMemSpace>();
  amp2_.template modify<HostMemSpace>();
  amp3_.template modify<HostMemSpace>();

  amp1_.template sync<DevExeSpace>();
  amp2_.template sync<DevExeSpace>();
  amp3_.template sync<DevExeSpace>();

  // Compute new force array (force_new)
  auto x1cos_ = x1cos;
  auto x1sin_ = x1sin;
  auto x2cos_ = x2cos;
  auto x2sin_ = x2sin;
  auto x3cos_ = x3cos;
  auto x3sin_ = x3sin;
  par_for("force_array",DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      for (int n=0; n<nt; n++) {
        int n1 = n/nw23;
        int n2 = (n - n1*nw23)/nw2;
        int n3 = n - n1*nw23 - n2*nw2;
        int nsqr = n1*n1 + n2*n2 + n3*n3;

        if (nsqr >= nlow_sq && nsqr <= nhigh_sq) {
          force_new_(m,0,k,j,i) += amp1_.d_view(n,0)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_.d_view(n,1)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp1_.d_view(n,2)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_.d_view(n,3)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                   amp1_.d_view(n,4)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_.d_view(n,5)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp1_.d_view(n,6)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp1_.d_view(n,7)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k);
          force_new_(m,1,k,j,i) += amp2_.d_view(n,0)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_.d_view(n,1)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp2_.d_view(n,2)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_.d_view(n,3)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                   amp2_.d_view(n,4)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_.d_view(n,5)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp2_.d_view(n,6)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp2_.d_view(n,7)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k);
          force_new_(m,2,k,j,i) += amp3_.d_view(n,0)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_.d_view(n,1)*x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp3_.d_view(n,2)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_.d_view(n,3)*x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                   amp3_.d_view(n,4)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_.d_view(n,5)*x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                   amp3_.d_view(n,6)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                   amp3_.d_view(n,7)*x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k);
        }
      }
    }
  );

  // Subtract any global mean from new force array (force_new)

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Real m0 = static_cast<Real>(nmkji);
  Real m1 = 0.0, m2 = 0.0, m3 = 0.0;
  Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_m1, Real &sum_m2, Real &sum_m3)
    {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      sum_m1 += force_new_(m,0,k,j,i);
      sum_m2 += force_new_(m,1,k,j,i);
      sum_m3 += force_new_(m,2,k,j,i);
    }, Kokkos::Sum<Real>(m1), Kokkos::Sum<Real>(m2), Kokkos::Sum<Real>(m3)
  );

#if MPI_PARALLEL_ENABLED
    Real m_sum4[4] = {m0,m1,m2,m3};
    Real gm_sum4[4];
    MPI_Allreduce(m_sum4, gm_sum4, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    m0 = gm_sum4[0];
    m1 = gm_sum4[1];
    m2 = gm_sum4[2];
    m3 = gm_sum4[3];
#endif

  par_for("net_mom_2", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      force_new_(m,0,k,j,i) -= m1/m0;
      force_new_(m,1,k,j,i) -= m2/m0;
      force_new_(m,2,k,j,i) -= m3/m0;
    }
  );

  // Calculate normalization of new force array so that energy input rate ~ dedt

  DvceArray5D<Real> u;
  if (pmy_pack->phydro != nullptr) u = (pmy_pack->phydro->u0);
  if (pmy_pack->pmhd != nullptr) u = (pmy_pack->pmhd->u0);
  if (pmy_pack->pionn != nullptr) u = (pmy_pack->phydro->u0); // assume neutral density
                                                              //     >> ionized density

  m0 = 0.0, m1 = 0.0;
  Kokkos::parallel_reduce("forcing_norm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1)
    { 
       // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real v1 = force_new_(m,0,k,j,i);
      Real v2 = force_new_(m,1,k,j,i);
      Real v3 = force_new_(m,2,k,j,i);

      /* two options here
      Real u1 = u(m,IM1,k,j,i)/u(m,IDN,k,j,i);
      Real u2 = u(m,IM2,k,j,i)/u(m,IDN,k,j,i);
      Real u3 = u(m,IM3,k,j,i)/u(m,IDN,k,j,i);      


      force_sum::GlobalSum fsum;
      fsum.the_array[IDN] = (v1*v1+v2*v2+v3*v3);
      fsum.the_array[IM1] = u1*v1 + u2*v2 + u3*v3;
      */

      Real u1 = u(m,IM1,k,j,i);
      Real u2 = u(m,IM2,k,j,i);
      Real u3 = u(m,IM3,k,j,i);      

      array_sum::GlobalSum fsum;
      sum_m0 += u(m,IDN,k,j,i)*(v1*v1+v2*v2+v3*v3);
      sum_m1 += u1*v1 + u2*v2 + u3*v3;
    }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1)
  );
  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<float>::min()) );

#if MPI_PARALLEL_ENABLED
    Real m_sum2[2] = {m0,m1};
    Real gm_sum2[2];
    MPI_Allreduce(m_sum2, gm_sum2, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    m0 = gm_sum2[0];
    m1 = gm_sum2[1];
#endif

/* old normalization
  aa = 0.5*m0;
  aa = max(aa,static_cast<Real>(1.0e-20));
  if (tcorr<=1e-20) {
    s = sqrt(dedt/dt/dvol/aa);
  } else {
    s = sqrt(dedt/tcorr/dvol/aa);
  }
*/

  // new normalization: assume constant energy injection per unit mass
  // explicit solution of <sF . (v + sF dt)> = dedt
  
  Real dvol = 1.0/(nx1*nx2*nx3); // old: Lx*Ly*Lz/nx1/nx2/nx3;
  m0 = m0*dvol*(pmy_pack->pmesh->dt);
  m1 = m1*dvol;

  Real s;
  if (m1 >= 0) {
    s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  } else {
    s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  }

  // Now normalize new force array
  par_for("OU_process", DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {
      force_new_(m,n,k,j,i) *= s;
    }
  );

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  apply forcing

TaskStatus TurbulenceDriver::AddForcing(Driver *pdrive, int stage)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  Real fcorr=0.0;
  Real gcorr=1.0;
  if ((pmy_pack->pmesh->time > 0.0) and (tcorr > 0.0)) {
    fcorr=exp(-((beta_dt)/tcorr));
    gcorr=sqrt(1.0-fcorr*fcorr);
  }

  if (pmy_pack->pionn == nullptr) {

    // modify conserved variables
    DvceArray5D<Real> u,w;
    if (pmy_pack->phydro != nullptr) {
      u = (pmy_pack->phydro->u0);
      w = (pmy_pack->phydro->w0);
    }
    if (pmy_pack->pmhd != nullptr) {
      u = (pmy_pack->pmhd->u0);
      w = (pmy_pack->pmhd->w0);
    }

    auto force_ = force;
    auto force_new_ = force_new;
    par_for("push", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real den = w(m,IDN,k,j,i);
        Real v1 = (fcorr*force_(m,0,k,j,i) + gcorr*force_new_(m,0,k,j,i))*beta_dt;
        Real v2 = (fcorr*force_(m,1,k,j,i) + gcorr*force_new_(m,1,k,j,i))*beta_dt;
        Real v3 = (fcorr*force_(m,2,k,j,i) + gcorr*force_new_(m,2,k,j,i))*beta_dt;
        Real m1 = den*w(m,IVX,k,j,i);
        Real m2 = den*w(m,IVY,k,j,i);
        Real m3 = den*w(m,IVZ,k,j,i);

  //      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
        u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3;
        u(m,IM1,k,j,i) += den*v1;
        u(m,IM2,k,j,i) += den*v2;
        u(m,IM3,k,j,i) += den*v3;
      }
    );
  } else {

    // modify conserved variables
    DvceArray5D<Real> u,w,u_,w_;
    u = (pmy_pack->pmhd->u0);
    w = (pmy_pack->pmhd->w0);
    u_ = (pmy_pack->phydro->u0);
    w_ = (pmy_pack->phydro->w0);

    auto force_ = force;
    auto force_new_ = force_new;
    par_for("push", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        // TODO:need to rescale forcing depending on ionization fraction

        Real v1 = (fcorr*force_(m,0,k,j,i) + gcorr*force_new_(m,0,k,j,i))*beta_dt;
        Real v2 = (fcorr*force_(m,1,k,j,i) + gcorr*force_new_(m,1,k,j,i))*beta_dt;
        Real v3 = (fcorr*force_(m,2,k,j,i) + gcorr*force_new_(m,2,k,j,i))*beta_dt;

        Real den = w(m,IDN,k,j,i);
        Real m1 = den*w(m,IVX,k,j,i);
        Real m2 = den*w(m,IVY,k,j,i);
        Real m3 = den*w(m,IVZ,k,j,i);

  //      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
        u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3;
        u(m,IM1,k,j,i) += den*v1;
        u(m,IM2,k,j,i) += den*v2;
        u(m,IM3,k,j,i) += den*v3;


        Real den_ = w_(m,IDN,k,j,i);
        Real m1_ = den_*w_(m,IVX,k,j,i);
        Real m2_ = den_*w_(m,IVY,k,j,i);
        Real m3_ = den_*w_(m,IVZ,k,j,i);

  //      u_(m,IEN,k,j,i) += m1_*v1 + m2_*v2 + m3_*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
        u_(m,IEN,k,j,i) += m1_*v1 + m2_*v2 + m3_*v3;
        u_(m,IM1,k,j,i) += den_*v1;
        u_(m,IM2,k,j,i) += den_*v2;
        u_(m,IM3,k,j,i) += den_*v3;
      }
    );
  }

  return TaskStatus::complete;
}
