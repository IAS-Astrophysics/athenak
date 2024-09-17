//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file diffusion.cpp
//! \brief problem generator for tests of diffusion modules (viscosity, resistivity,
//! thermal conduction).  Sets up Gaussian profile in x-direction.
//! CURRENTLY ONLY VISCOSITY IMPLEMENTED
//! This file also contains a function to compute L1 errors in solution, called in
//! Driver::Finalize().

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

// Prototype for function to compute errors in solution at end of run
void DiffusionErrors(ParameterInput *pin, Mesh *pm);
// Prototype for user-defined BCs
void GaussianProfile(Mesh *pm);

// Anonymous namespace used to prevent name collisions outside of this file
namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;
// input parameters passed to user-defined BC function
struct DiffusionVariables {
  Real d0, amp, t0, x10;
};

DiffusionVariables dv;

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Diffusion()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::Diffusion(ParameterInput *pin, const bool restart) {
  // set diffusion errors function
  pgen_final_func = DiffusionErrors;
  // user-define BC
  user_bcs_func = GaussianProfile;
  if (restart) return;

  // Read problem parameters
  dv.d0 = 1.0;
  dv.amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  dv.t0 = pin->GetOrAddReal("problem", "t0", 0.5);
  dv.x10 = pin->GetOrAddReal("problem", "x10", 0.0);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &time = pmbp->pmesh->time;

  // capture variables for the kernel
  //auto dv_=dv;
  auto d0_=dv.d0, amp_=dv.amp, x10_=dv.x10;
  // add stopping time when called at end of run
  Real t1 = dv.t0;
  if (!(set_initial_conditions)) {t1 += time;}

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    if (pmbp->phydro->pvisc == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Hydro diffusion problem requires viscosity to be defined" << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    auto &nu_iso = pmbp->phydro->pvisc->nu_iso;

    // compute solution in u1 register. For initial conditions, set u1 -> u0.
    auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;

    // Initialize Gaussian profile of transverse (x2 and x3) velocity in x1-direction
    par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      u1(m,IDN,k,j,i) = d0_;
      u1(m,IM1,k,j,i) = 0.0;
      u1(m,IM2,k,j,i) = d0_*amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                        /sqrt(4.*M_PI*nu_iso*t1);
      u1(m,IM3,k,j,i) = d0_*amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                        /sqrt(4.*M_PI*nu_iso*t1);
      if (eos.is_ideal) {
        u1(m,IEN,k,j,i) = p0/gm1 + 0.5*(SQR(u1(m,IM2,k,j,i)) + SQR(u1(m,IM3,k,j,i)))/d0_;
      }
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

  Real l1_err[8];
  Real linfty_err=0.0;
  int nvars=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("LW-err",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nvars, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] = l1_err[i]/vol;
  linfty_err /= vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1    L-infty       ");
      std::fprintf(pfile,"d_L1         M1_L1         M2_L1         M3_L1         E_L1");
      if (pmbp->pmhd != nullptr) {
        std::fprintf(pfile,"          B1_L1         B2_L1         B3_L1");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn GaussianProfile
//  \brief Sets boundary condition on surfaces of computational domain
// FIXME: Boundaries need to be adjusted for DynGRMHD

void GaussianProfile(Mesh *pm) {
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
  auto &u0 = pm->pmb_pack->phydro->u0;

  // capture variables for the kernel
  //auto dv_=dv;
  auto d0_=dv.d0, amp_=dv.amp, x10_=dv.x10;
  Real t1 = dv.t0 + pm->time;

  par_for("diffusion_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(-1-i, nx1, x1min, x1max);

        u0(m,IDN,k,j,is-i-1) = d0_;
        u0(m,IM1,k,j,is-i-1) = 0.0;
        u0(m,IM2,k,j,is-i-1) = d0_*amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                          /sqrt(4.*M_PI*nu_iso*t1);
        u0(m,IM3,k,j,is-i-1) = d0_*amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                          /sqrt(4.*M_PI*nu_iso*t1);
        if (eos.is_ideal) {
          u0(m,IEN,k,j,is-i-1) = p0/gm1 +
                                 0.5*(SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)))/d0_;
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(ie-is+1+i, nx1, x1min, x1max);

        u0(m,IDN,k,j,ie+i+1) = d0_;
        u0(m,IM1,k,j,ie+i+1) = 0.0;
        u0(m,IM2,k,j,ie+i+1) = d0_*amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                          /sqrt(4.*M_PI*nu_iso*t1);
        u0(m,IM3,k,j,ie+i+1) = d0_*amp_*exp(SQR(x1v-x10_)/(-4.0*nu_iso*t1))
                          /sqrt(4.*M_PI*nu_iso*t1);
        if (eos.is_ideal) {
          u0(m,IEN,k,j,ie+i+1) = p0/gm1 +
                                 0.5*(SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)))/d0_;
        }
      }
    }
  });
  return;
}
