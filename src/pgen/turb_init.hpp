#ifndef PGEN_TURB_INIT_HPP_
#define PGEN_TURB_INIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_init.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process

#include <limits>
#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion_neutral.hpp"
#include "driver/driver.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceInit

class TurbulenceInit {
 public:
  TurbulenceInit(std::string bk, MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceInit();

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
  int n0;
  int turb_flag;
  int turb_count;
  Real tcorr,dedt;
  Real expo;
  Real last_dt;
  int64_t seed; // for generating amp1,amp2,amp3 arrays

  // functions
  TaskStatus InitializeModes(int stage);
  TaskStatus AddForcing(int stage);

 private:
  bool first_time=true;     // flag to enable initialization on first call
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceInit
};

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceInit



//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceInit::TurbulenceInit(std::string bk, MeshBlockPack *pp, ParameterInput *pin) :
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
  amp3("amp3",1,1) {
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_new, nmb, 3, ncells3, ncells2, ncells1);

  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger(bk,"nlow",1);
  nhigh = pin->GetOrAddInteger(bk,"nhigh",2);
  n0 = pin->GetOrAddInteger(bk,"n0",1);
  turb_flag = pin->GetOrAddInteger(bk,"turb_flag",0);
  turb_count = pin->GetOrAddInteger(bk,"turb_count",0);
  seed = pin->GetOrAddInteger(bk,"seed",-1);
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
  expo = pin->GetOrAddReal(bk,"expo",5.0/3.0);
  // energy injection rate
  dedt = pin->GetOrAddReal(bk,"dedt",0.0);
  // correlation time
  tcorr = pin->GetOrAddReal(bk,"tcorr",0.0);

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

TurbulenceInit::~TurbulenceInit() {
}

//----------------------------------------------------------------------------------------
//! \fn InitializeModes()
// \brief Initializes driving, and so is only executed once at start of calc.
// Cannot be included in constructor since (it seems) Kokkos::par_for not allowed in cons.

TaskStatus TurbulenceInit::InitializeModes(int stage) {
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
  Real dkx = 2.0*M_PI/lx*n0;
  Real dky = 2.0*M_PI/ly*n0;
  Real dkz = 2.0*M_PI/lz*n0;

  int nw2 = 1; int nw3 = 1;
  if (ncells2>1) {
    nw2 = nhigh+1;
  }
  if (ncells3>1) {
    nw3 = nhigh+1;
  }
  int nw23 = nw3*nw2;

  // turb_flag == 1 : decaying turbulence
  if (turb_flag == 1 && turb_count == 0) {
    return TaskStatus::complete;
  }
  // On first call to this function, initialize seeds, sin/cos arrays
  if (first_time) {
    // initialize force to zero
    int &nmb = pmy_pack->nmb_thispack;
    auto force_ = force;
    par_for("force_init", DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      force_(m,n,k,j,i) = 0.0;
    });

    // initalize seeds
    int &nt = ntot;
    //seed = -1;

    // Initialize sin and cos arrays
    // bad design: requires saving sin/cos during restarts
    auto &size = pmy_pack->pmb->mb_size;
    auto x1sin_ = x1sin;
    auto x1cos_ = x1cos;
    par_for("kx_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells1-1,
    KOKKOS_LAMBDA(int m, int n, int i) {
      int nk1 = n/nw23;
      Real kx = nk1*dkx;
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      x1sin_(m,n,i) = sin(kx*x1v);
      x1cos_(m,n,i) = cos(kx*x1v);
    });

    auto x2sin_ = x2sin;
    auto x2cos_ = x2cos;
    par_for("ky_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells2-1,
    KOKKOS_LAMBDA(int m, int n, int j) {
      int nk1 = n/nw23;
      int nk2 = (n - nk1*nw23)/nw3;
      Real ky = nk2*dky;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      x2sin_(m,n,j) = sin(ky*x2v);
      x2cos_(m,n,j) = cos(ky*x2v);
    });

    auto x3sin_ = x3sin;
    auto x3cos_ = x3cos;
    par_for("kz_loop", DevExeSpace(), 0, nmb-1, 0, nt-1, 0, ncells3-1,
    KOKKOS_LAMBDA(int m, int n, int k) {
      int nk1 = n/nw23;
      int nk2 = (n - nk1*nw23)/nw3;
      int nk3 = n - nk1*nw23 - nk2*nw3;
      Real kz = nk3*dkz;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      x3sin_(m,n,k) = sin(kz*x3v);
      x3cos_(m,n,k) = cos(kz*x3v);
    });
    first_time = false;

  // if this is NOT the first call, evolve force according to O-U process, using "new"
  // force computed last time step and still stored in "force_new" array
  } else {
    // TODO(@leva): if not first call, there should also be initializtion of
    // x#sin and x#cos, unless they are saved in the restart.
    Real fcorr=0.0;
    Real gcorr=1.0;
    if ((pmy_pack->pmesh->time > 0.0) && (tcorr > 0.0)) {
      fcorr=exp(-(last_dt/tcorr));
      gcorr=sqrt(1.0-fcorr*fcorr);
    }

    auto force_ = force;
    auto force_new_ = force_new;
    int &nmb = pmy_pack->nmb_thispack;
    par_for("OU_process", DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      force_(m,n,k,j,i) = fcorr*force_(m,n,k,j,i) + gcorr*force_new_(m,n,k,j,i);
    });
    last_dt = 1.0;  // store this dt for call to this fn next timestep
  }

  // Now compute new force using new random amplitudes and phases

  // Zero out new force array
  auto force_new_ = force_new;
  int &nmb = pmy_pack->nmb_thispack;
  par_for("forcing_init", DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_new_(m,0,k,j,i) = 0.0;
    force_new_(m,1,k,j,i) = 0.0;
    force_new_(m,2,k,j,i) = 0.0;
  });

  int nlow_sq  = SQR(nlow);
  int nhigh_sq = SQR(nhigh);

  int &nt = ntot;
  Real &ex = expo;
  auto amp1_ = amp1;
  auto amp2_ = amp2;
  auto amp3_ = amp3;

  HostArray1D<Real> amps;
  Kokkos::realloc(amps, nwave);
  HostArray2D<Real> ampt;
  Kokkos::realloc(ampt, 3, nwave);

  // TODO(@leva): move this for loop to the host
  for (int n=0; n<=nt-1; n++) {
    int nk1 = n/nw23;
    int nk2 = (n - nk1*nw23)/nw3;
    int nk3 = n - nk1*nw23 - nk2*nw3;
    Real kx = nk1*dkx;
    Real ky = nk2*dky;
    Real kz = nk3*dkz;

    int nsq = nk1*nk1 + nk2*nk2 + nk3*nk3;

    Real kmag = sqrt(kx*kx + ky*ky + kz*kz);
    Real norm = 1.0/pow(kmag,(ex+2.0)/2.0);

    // TODO(@leva): check whether those coefficients are needed
    // if (nk1 > 0) norm *= 0.5;
    // if (nk2 > 0) norm *= 0.5;
    // if (nk3 > 0) norm *= 0.5;

    if (nsq >= nlow_sq && nsq <= nhigh_sq) {
      // Generate Fourier amplitudes
      // Symmetric method
      amp1_.h_view(n,0) = RanGaussian(&(seed));
      amp1_.h_view(n,1) = (nk3 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp1_.h_view(n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp1_.h_view(n,3) = (nk2 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));
      amp1_.h_view(n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp1_.h_view(n,5) = (nk1 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));
      amp1_.h_view(n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));
      amp1_.h_view(n,7) = (nk1 == 0 || nk2 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));

      amp2_.h_view(n,0) = RanGaussian(&(seed));
      amp2_.h_view(n,1) = (nk3 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp2_.h_view(n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp2_.h_view(n,3) = (nk2 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));
      amp2_.h_view(n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp2_.h_view(n,5) = (nk1 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));
      amp2_.h_view(n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));
      amp2_.h_view(n,7) = (nk1 == 0 || nk2 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));

      amp3_.h_view(n,0) = RanGaussian(&(seed));
      amp3_.h_view(n,1) = (nk3 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp3_.h_view(n,2) = (nk2 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp3_.h_view(n,3) = (nk2 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));
      amp3_.h_view(n,4) = (nk1 == 0)             ? 0.0 :RanGaussian(&(seed));
      amp3_.h_view(n,5) = (nk1 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));
      amp3_.h_view(n,6) = (nk1 == 0 || nk2 == 0) ? 0.0 :RanGaussian(&(seed));
      amp3_.h_view(n,7) = (nk1 == 0 || nk2 == 0 || nk3 == 0) ? 0.0 :RanGaussian(&(seed));

      // incompressibility
      // by compute curl(amp)
      ampt(0,0) =  ky*amp3_.h_view(n,2) - kz*amp2_.h_view(n,1);
      ampt(0,1) =  ky*amp3_.h_view(n,3) + kz*amp2_.h_view(n,0);
      ampt(0,2) = -ky*amp3_.h_view(n,0) - kz*amp2_.h_view(n,3);
      ampt(0,3) = -ky*amp3_.h_view(n,1) + kz*amp2_.h_view(n,2);
      ampt(0,4) =  ky*amp3_.h_view(n,6) - kz*amp2_.h_view(n,5);
      ampt(0,5) =  ky*amp3_.h_view(n,7) + kz*amp2_.h_view(n,4);
      ampt(0,6) = -ky*amp3_.h_view(n,4) - kz*amp2_.h_view(n,7);
      ampt(0,7) = -ky*amp3_.h_view(n,5) + kz*amp2_.h_view(n,6);

      ampt(1,0) =  kz*amp1_.h_view(n,1) - kx*amp3_.h_view(n,4);
      ampt(1,1) = -kz*amp1_.h_view(n,0) - kx*amp3_.h_view(n,5);
      ampt(1,2) =  kz*amp1_.h_view(n,3) - kx*amp3_.h_view(n,6);
      ampt(1,3) = -kz*amp1_.h_view(n,2) - kx*amp3_.h_view(n,7);
      ampt(1,4) =  kz*amp1_.h_view(n,5) + kx*amp3_.h_view(n,0);
      ampt(1,5) = -kz*amp1_.h_view(n,4) + kx*amp3_.h_view(n,1);
      ampt(1,6) =  kz*amp1_.h_view(n,7) + kx*amp3_.h_view(n,2);
      ampt(1,7) = -kz*amp1_.h_view(n,6) + kx*amp3_.h_view(n,3);

      ampt(2,0) =  kx*amp2_.h_view(n,4) - ky*amp1_.h_view(n,2);
      ampt(2,1) =  kx*amp2_.h_view(n,5) - ky*amp1_.h_view(n,3);
      ampt(2,2) =  kx*amp2_.h_view(n,6) + ky*amp1_.h_view(n,0);
      ampt(2,3) =  kx*amp2_.h_view(n,7) + ky*amp1_.h_view(n,1);
      ampt(2,4) = -kx*amp2_.h_view(n,0) - ky*amp1_.h_view(n,6);
      ampt(2,5) = -kx*amp2_.h_view(n,1) - ky*amp1_.h_view(n,7);
      ampt(2,6) = -kx*amp2_.h_view(n,2) + ky*amp1_.h_view(n,4);
      ampt(2,7) = -kx*amp2_.h_view(n,3) + ky*amp1_.h_view(n,5);
      
      for (int i=0; i<8; ++i) {
        amp1_.h_view(n,i) = (nk2*nk2+nk3*nk3)>0? ampt(0,i)/sqrt(ky*ky+kz*kz) : 0.0;
        amp2_.h_view(n,i) = (nk3*nk3+nk1*nk1)>0? ampt(1,i)/sqrt(kz*kz+kx*kx) : 0.0;
        amp3_.h_view(n,i) = (nk1*nk1+nk2*nk2)>0? ampt(2,i)/sqrt(kx*kx+ky*ky) : 0.0;
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
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<nt; n++) {
      int n1 = n/nw23;
      int n2 = (n - n1*nw23)/nw3;
      int n3 = n - n1*nw23 - n2*nw3;
      int nsqr = n1*n1 + n2*n2 + n3*n3;

      if (nsqr >= nlow_sq && nsqr <= nhigh_sq) {
        force_new_(m,0,k,j,i) += (amp1_.d_view(n,0)*
                                   x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                  amp1_.d_view(n,1)*
                                   x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                  amp1_.d_view(n,2)*
                                   x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                  amp1_.d_view(n,3)*
                                   x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                  amp1_.d_view(n,4)*
                                   x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                  amp1_.d_view(n,5)*
                                   x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                  amp1_.d_view(n,6)*
                                   x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                  amp1_.d_view(n,7)*
                                   x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k));
        force_new_(m,1,k,j,i) += (amp2_.d_view(n,0)*
                                   x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                  amp2_.d_view(n,1)*
                                   x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                  amp2_.d_view(n,2)*
                                   x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                  amp2_.d_view(n,3)*
                                   x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                  amp2_.d_view(n,4)*
                                   x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                  amp2_.d_view(n,5)*
                                   x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                  amp2_.d_view(n,6)*
                                   x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                  amp2_.d_view(n,7)*
                                   x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k));
        force_new_(m,2,k,j,i) += (amp3_.d_view(n,0)*
                                   x1cos_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                  amp3_.d_view(n,1)*
                                   x1cos_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                  amp3_.d_view(n,2)*
                                   x1cos_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                  amp3_.d_view(n,3)*
                                   x1cos_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k)+
                                  amp3_.d_view(n,4)*
                                   x1sin_(m,n,i)*x2cos_(m,n,j)*x3cos_(m,n,k)+
                                  amp3_.d_view(n,5)*
                                   x1sin_(m,n,i)*x2cos_(m,n,j)*x3sin_(m,n,k)+
                                  amp3_.d_view(n,6)*
                                   x1sin_(m,n,i)*x2sin_(m,n,j)*x3cos_(m,n,k)+
                                  amp3_.d_view(n,7)*
                                   x1sin_(m,n,i)*x2sin_(m,n,j)*x3sin_(m,n,k));
      }
    }
  });

  // Subtract any global mean from new force array (force_new)

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Real m0 = static_cast<Real>(nmkji);
  Real m1 = 0.0, m2 = 0.0, m3 = 0.0;
  if (!(pmy_pack->pmesh->multilevel)) {
  Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_m1, Real &sum_m2, Real &sum_m3) {
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
  }, Kokkos::Sum<Real>(m1), Kokkos::Sum<Real>(m2), Kokkos::Sum<Real>(m3));
  } else {
    m0 = 0.0;
    if (pmy_pack->pionn == nullptr) {
      // modify conserved variables
      DvceArray5D<Real> u;
      if (pmy_pack->phydro != nullptr) {
        u = (pmy_pack->phydro->u0);
      }
      if (pmy_pack->pmhd != nullptr) {
        u = (pmy_pack->pmhd->u0);
      }
      auto &size = pmy_pack->pmb->mb_size;
      Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1, Real &sum_m2,
      Real &sum_m3) {
        // compute n,k,j,i indices of thread
        int m = (idx)/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks;
        j += js;

        Real dens = u(m,IDN,k,j,i);
        Real dm = dens*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        sum_m0 += dm;
        sum_m1 += dm*force_new_(m,0,k,j,i);
        sum_m2 += dm*force_new_(m,1,k,j,i);
        sum_m3 += dm*force_new_(m,2,k,j,i);
      }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1), Kokkos::Sum<Real>(m2),
      Kokkos::Sum<Real>(m3));
    }
  }

#if MPI_PARALLEL_ENABLED
    Real m_sum4[4] = {m0,m1,m2,m3};
    Real gm_sum4[4];
    MPI_Allreduce(m_sum4, gm_sum4, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    m0 = gm_sum4[0];
    m1 = gm_sum4[1];
    m2 = gm_sum4[2];
    m3 = gm_sum4[3];
#endif

  // TODO(@mhguo): rm this!
  //std::cout<<"m0="<<m0<<"  m1="<<m1<<"  m2="<<m2<<"  m3="<<m3<<std::endl;
  Real m00 = m0;
  par_for("net_mom_2", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_new_(m,0,k,j,i) -= m1/m0;
    force_new_(m,1,k,j,i) -= m2/m0;
    force_new_(m,2,k,j,i) -= m3/m0;
  });

  // Calculate normalization of new force array so that energy input rate ~ dedt
  DvceArray5D<Real> u;
  if (pmy_pack->phydro != nullptr) u = (pmy_pack->phydro->u0);
  if (pmy_pack->pmhd != nullptr) u = (pmy_pack->pmhd->u0);
  if (pmy_pack->pionn != nullptr) u = (pmy_pack->phydro->u0); // assume neutral density
                                                              //     >> ionized density

  m0 = 0.0, m1 = 0.0;
  if (!(pmy_pack->pmesh->multilevel)) {
  Kokkos::parallel_reduce("forcing_norm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1) {
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

    // two options here
    // Real u1 = u(m,IM1,k,j,i)/u(m,IDN,k,j,i);
    // Real u2 = u(m,IM2,k,j,i)/u(m,IDN,k,j,i);
    // Real u3 = u(m,IM3,k,j,i)/u(m,IDN,k,j,i);

    // force_sum::GlobalSum fsum;
    // fsum.the_array[IDN] = (v1*v1+v2*v2+v3*v3);
    // fsum.the_array[IM1] = u1*v1 + u2*v2 + u3*v3;

    Real u1 = u(m,IM1,k,j,i);
    Real u2 = u(m,IM2,k,j,i);
    Real u3 = u(m,IM3,k,j,i);

    array_sum::GlobalSum fsum;
    sum_m0 += u(m,IDN,k,j,i)*(v1*v1+v2*v2+v3*v3);
    sum_m1 += u1*v1 + u2*v2 + u3*v3;
  }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1));
  m0 = std::max(m0, static_cast<Real>(std::numeric_limits<float>::min()) );
  } else {
    auto &size = pmy_pack->pmb->mb_size;
    Kokkos::parallel_reduce("forcing_norm2", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      Real v1 = force_new_(m,0,k,j,i);
      Real v2 = force_new_(m,1,k,j,i);
      Real v3 = force_new_(m,2,k,j,i);

      // two options here
      // Real u1 = u(m,IM1,k,j,i)/u(m,IDN,k,j,i);
      // Real u2 = u(m,IM2,k,j,i)/u(m,IDN,k,j,i);
      // Real u3 = u(m,IM3,k,j,i)/u(m,IDN,k,j,i);

      // force_sum::GlobalSum fsum;
      // fsum.the_array[IDN] = (v1*v1+v2*v2+v3*v3);
      // fsum.the_array[IM1] = u1*v1 + u2*v2 + u3*v3;

      Real u1 = u(m,IM1,k,j,i);
      Real u2 = u(m,IM2,k,j,i);
      Real u3 = u(m,IM3,k,j,i);

      //if (m==0&&k==6&&j==6&&i==6) {
      //  printf("normden: %.6e\n",u(m,IDN,k,j,i));
      //printf("normmom: %.6e %.6e %.6e\n",u(m,IM1,k,j,i),u(m,IM2,k,j,i),u(m,IM3,k,j,i));
      //  printf("normv: %.6e %.6e %.6e\n",v1,v2,v3);
      //}

      array_sum::GlobalSum fsum;
      sum_m0 += vol*u(m,IDN,k,j,i)*(v1*v1+v2*v2+v3*v3);
      sum_m1 += vol*(u1*v1 + u2*v2 + u3*v3);
    }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1));
    m0 = std::max(m0, static_cast<Real>(std::numeric_limits<float>::min()) );
  }

#if MPI_PARALLEL_ENABLED
    Real m_sum2[2] = {m0,m1};
    Real gm_sum2[2];
    MPI_Allreduce(m_sum2, gm_sum2, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    m0 = gm_sum2[0];
    m1 = gm_sum2[1];
#endif

  // old normalization
  // aa = 0.5*m0;
  // aa = max(aa,static_cast<Real>(1.0e-20));
  // if (tcorr<=1e-20) {
  //   s = sqrt(dedt/dt/dvol/aa);
  // } else {
  //   s = sqrt(dedt/tcorr/dvol/aa);
  // }

  // new normalization: assume constant energy injection per unit mass
  // explicit solution of <sF . (v + sF dt)> = dedt

  Real dvol = 1.0/(nx1*nx2*nx3); // old: Lx*Ly*Lz/nx1/nx2/nx3;
  // TODO(@mhguo): Note It should be nx of mesh, not meshblocks!
  auto &mesh_indcs = pmy_pack->pmesh->mesh_indcs;
  dvol = 1.0/(mesh_indcs.nx1*mesh_indcs.nx2*mesh_indcs.nx3);
  if (!(pmy_pack->pmesh->multilevel)) {
  m0 = m0*dvol*(1.0);
  m1 = m1*dvol;
  } else {
    m0 = m0*(1.0)/m00;
    m1 = m1/m00;
  }

  Real s;
  if (m1 >= 0) {
    s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  } else {
    s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
  }

  // TODO(@mhguo): rm this cout!
  //std::cout<<"m00="<<m00<<"  m0="<<m0<<"  m1="<<m1<<"  s="<<s<<std::endl;

  // Now normalize new force array
  par_for("OU_process", DevExeSpace(),0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_new_(m,n,k,j,i) *= s;
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  apply forcing

TaskStatus TurbulenceInit::AddForcing(int stage) {
  // turb_flag == 1 : decaying turbulence
  // TODO(@mhguo): rm this!
  //std::cout << "turb_count=" << turb_count << std::endl;
  if (turb_flag == 1) {
    if (turb_count == 0) {
      return TaskStatus::complete;
    } else if (turb_count>0) {
      turb_count -= 1;
      //if (turb_count == 1) {
      //  return TaskStatus::complete;
      //}
    }
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real beta_dt = 1.0;
  // turb_flag == 1 : decaying turbulence
  if (turb_flag == 1) {
    beta_dt = 1.0;
  }
  Real fcorr=0.0;
  Real gcorr=1.0;
  if ((pmy_pack->pmesh->time > 0.0) && (tcorr > 0.0)) {
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
    std::cout<<"fcorr="<<fcorr<<"  gcorr="<<gcorr<<std::endl;
    par_for("push", DevExeSpace(),0,(pmy_pack->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real den = u(m,IDN,k,j,i);
      Real v1 = (fcorr*force_(m,0,k,j,i) + gcorr*force_new_(m,0,k,j,i))*beta_dt;
      Real v2 = (fcorr*force_(m,1,k,j,i) + gcorr*force_new_(m,1,k,j,i))*beta_dt;
      Real v3 = (fcorr*force_(m,2,k,j,i) + gcorr*force_new_(m,2,k,j,i))*beta_dt;
      Real m1 = u(m,IM1,k,j,i);
      Real m2 = u(m,IM2,k,j,i);
      Real m3 = u(m,IM3,k,j,i);

      // TODO(@mhguo): rm this cout!
      //if (m==0&&k==6&&j==6&&i==6) {
      //  printf("den: %.6e\n",u(m,IDN,k,j,i));
      //  printf("mom: %.6e %.6e %.6e\n",u(m,IM1,k,j,i),u(m,IM2,k,j,i),u(m,IM3,k,j,i));
      //  printf("v: %.6e %.6e %.6e\n",v1,v2,v3);
      //}

      // u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
      // u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.25*den*(v1*v1+v2*v2+v3*v3);
      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3;
      u(m,IM1,k,j,i) += den*v1;
      u(m,IM2,k,j,i) += den*v2;
      u(m,IM3,k,j,i) += den*v3;
      if (m==0&&k==6&&j==6&&i==6) {
        printf("den_new: %.6e\n",u(m,IDN,k,j,i));
        printf("mom_new: %.6e %.6e %.6e\n",u(m,IM1,k,j,i),u(m,IM2,k,j,i),u(m,IM3,k,j,i));
      }
    });
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
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // TODO(@user): need to rescale forcing depending on ionization fraction

      Real v1 = (fcorr*force_(m,0,k,j,i) + gcorr*force_new_(m,0,k,j,i))*beta_dt;
      Real v2 = (fcorr*force_(m,1,k,j,i) + gcorr*force_new_(m,1,k,j,i))*beta_dt;
      Real v3 = (fcorr*force_(m,2,k,j,i) + gcorr*force_new_(m,2,k,j,i))*beta_dt;

      Real den = w(m,IDN,k,j,i);
      Real m1 = den*w(m,IVX,k,j,i);
      Real m2 = den*w(m,IVY,k,j,i);
      Real m3 = den*w(m,IVZ,k,j,i);

      // u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
      u(m,IEN,k,j,i) += m1*v1 + m2*v2 + m3*v3;
      u(m,IM1,k,j,i) += den*v1;
      u(m,IM2,k,j,i) += den*v2;
      u(m,IM3,k,j,i) += den*v3;


      Real den_ = w_(m,IDN,k,j,i);
      Real m1_ = den_*w_(m,IVX,k,j,i);
      Real m2_ = den_*w_(m,IVY,k,j,i);
      Real m3_ = den_*w_(m,IVZ,k,j,i);

      // u_(m,IEN,k,j,i) += m1_*v1 + m2_*v2 + m3_*v3 + 0.5*den*(v1*v1+v2*v2+v3*v3);
      u_(m,IEN,k,j,i) += m1_*v1 + m2_*v2 + m3_*v3;
      u_(m,IM1,k,j,i) += den_*v1;
      u_(m,IM2,k,j,i) += den_*v2;
      u_(m,IM3,k,j,i) += den_*v3;
    });
  }

  return TaskStatus::complete;
}

#endif // PGEN_TURB_INIT_HPP_
