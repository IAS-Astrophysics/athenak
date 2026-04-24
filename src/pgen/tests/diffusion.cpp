//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file diffusion.cpp
//! \brief problem generator for simultaneous tests of diffusion modules (viscosity,
//! resistivity, thermal conduction).  Sets up Gaussian profile of transverse velocity,
//! magnetic field and temperature in x-direction. Should be run in kinematic mode
//! Errors in final solution are computed from analytic profile at final time in
//! DiffusionErrors() function.

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"

// Prototype for function to compute errors in solution at end of run
void DiffusionErrors(ParameterInput *pin, Mesh *pm);
// Prototype for user-defined BCs
void GaussianProfileBCs(Mesh *pm);

// Anonymous namespace used to prevent name collisions outside of this file
namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;
// input parameters passed to user-defined BC function
struct DiffusionVariables {
  int prob_dir;
  Real amp, t0, x10;
};

DiffusionVariables dvars;

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Diffusion()
//! \brief Sets initial conditions for diffusion tests

void ProblemGenerator::Diffusion(ParameterInput *pin, const bool restart) {
  // set diffusion errors function
  pgen_final_func = DiffusionErrors;
  // user-define BC
  user_bcs_func = GaussianProfileBCs;
  if (restart) return;

  // Read problem parameters
  dvars.prob_dir = pin->GetOrAddInteger("problem","direction",1);
  dvars.amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  dvars.t0 = pin->GetOrAddReal("problem", "t0", 0.5);
  dvars.x10 = pin->GetOrAddReal("problem", "x10", 0.0);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &time = pmbp->pmesh->time;

  // capture variables for the kernel
  auto amp_ = dvars.amp, x10_ = dvars.x10;
  // add stopping time when called at end of run
  Real t1 = dvars.t0;
  if (!(set_initial_conditions)) {t1 += time;}

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    if (pmbp->phydro->pvisc == nullptr || pmbp->phydro->pcond == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Diffusion test requires viscosity and conduction to be defined in"
              << " Hydro input block" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!(eos.is_ideal)) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Diffusion test requires ideal EOS in Hydro block" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    auto &nu_iso = pmbp->phydro->pvisc->nu_iso;
    auto &kappa_iso = pmbp->phydro->pcond->kappa_iso;

    // compute solution in u1 register. For initial conditions, set u1 -> u0.
    auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;

    // Initialize Gaussian profile of transverse (x2 and x3) velocity in x1-direction
    // Initialize Gaussian profile of temperature in x1-direction
    par_for("pgen_diff1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      u1(m,IDN,k,j,i) = 1,0;
      u1(m,IM1,k,j,i) = 0.0;
      u1(m,IM2,k,j,i) = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))/sqrt(4.*M_PI*nu_iso*t1);
      u1(m,IM3,k,j,i) = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))/sqrt(4.*M_PI*nu_iso*t1);
      Real press = amp_*exp(SQR(x1v-x10_)/(-4.0*kappa_iso*t1))/sqrt(4.*M_PI*kappa_iso*t1);
      u1(m,IEN,k,j,i) = press/gm1 + 0.5*(SQR(u1(m,IM2,k,j,i)) + SQR(u1(m,IM3,k,j,i)));
    });
  } // End initialization of Hydro variables
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DiffusionErrors_()
//! \brief Computes errors in diffusion solution by calling initialization function
//! again to compute initial condictions, and subtracting current solution from ICs, and
//! outputs errors to file.

void DiffusionErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->Diffusion(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn GaussianProfileBCs
//! \brief Sets boundary conditions to time-dependent Gaussian profile on edges of
//! computational domain

void GaussianProfileBCs(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  //int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;

  EOS_Data &eos = pm->pmb_pack->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real p0 = 1.0/eos.gamma;
  auto &nu_iso = pm->pmb_pack->phydro->pvisc->nu_iso;
  auto &kappa_iso = pm->pmb_pack->phydro->pcond->kappa_iso;
  auto &u0 = pm->pmb_pack->phydro->u0;

  // capture variables for the kernel
  //auto dv_=dv;
  auto amp_ = dvars.amp, x10_ = dvars.x10;
  Real t1 = dvars.t0 + pm->time;

  if (dvars.prob_dir == 1) {
    par_for("diffusion_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      for (int i=0; i<ng; ++i) {
        // Inner X1-boundary
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(-1-i, nx1, x1min, x1max);

        u0(m,IDN,k,j,is-i-1) = 1.0;
        u0(m,IM1,k,j,is-i-1) = 0.0;
        u0(m,IM2,k,j,is-i-1) = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                               /sqrt(4.*M_PI*nu_iso*t1);
        u0(m,IM3,k,j,is-i-1) = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                               /sqrt(4.*M_PI*nu_iso*t1);
        Real press = amp_*exp(SQR(x1v-x10_)/(-4.0*kappa_iso*t1))
                     /sqrt(4.*M_PI*kappa_iso*t1);
        u0(m,IEN,k,j,is-i-1) = press/gm1 +
                               0.5*(SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)));

        // Outer X1-boundary
        x1v = CellCenterX(ie-is+1+i, nx1, x1min, x1max);

        u0(m,IDN,k,j,ie+i+1) = 1,0;
        u0(m,IM1,k,j,ie+i+1) = 0.0;
        u0(m,IM2,k,j,ie+i+1) = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                               /sqrt(4.*M_PI*nu_iso*t1);
        u0(m,IM3,k,j,ie+i+1) = amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                               /sqrt(4.*M_PI*nu_iso*t1);
        press = amp_*exp(SQR(x1v-x10_)/(-4.0*kappa_iso*t1))/sqrt(4.*M_PI*kappa_iso*t1);
        u0(m,IEN,k,j,ie+i+1) = press/gm1 +
                               0.5*(SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)));
      }
    });
  }
  return;
}
