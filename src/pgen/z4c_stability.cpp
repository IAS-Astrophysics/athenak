//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_stability.cpp
//! \brief z4c stability test


// C/C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <string>    // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

#include <Kokkos_Random.hpp>


// history function to output error as a function of time
void Z4cStabilityErrors(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for robust stability test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &Z4cStabilityErrors;
  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Z4c stability test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Prepare Initial Data

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  int ncell1 = 2*indcs.ng + indcs.nx1;
  int ncell2 = 2*indcs.ng + indcs.nx2;
  int ncell12 = ncell1 * ncell2;

  auto &adm = pmbp->padm->adm;
  auto &z4c = pmbp->pz4c->z4c;

  // perturbation amplitude 1e-10 dx^2
  Real rho = pin->GetOrAddReal("problem", "rho", 1);
  Real amp = 1e-10/rho/rho;

  // random number
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  size_t scr_size = ScrArray1D<Real>::shmem_size(12);
  int scr_level = 0;
  par_for_outer("pgen_stability",DevExeSpace(), scr_size, scr_level, 0,nmb-1,ksg,keg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
    // get random number state this thread
    auto rand_gen = rand_pool64.get_state();

    // scratch array holding random numbers for 12 variables.
    // only perturbaion in the z direction is added
    // in the x and y directions the values of perturbation
    // are always the same
    ScrArray1D<Real> rnum(member.team_scratch(scr_level), 12);
    for(int a = 0; a < 12; ++a) {
      rnum(a) = amp*2*(rand_gen.drand() - 0.5);
    }

    par_for_inner(member, 0, ncell12-1, [&](const int ij) {
      const int j = ij/ncell1;
      const int i = ij%ncell1;

      // Minkowski spacetime
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b ? 1. : 0.);
      }
      adm.alpha(m,k,j,i) = 1.0;

      // g_ab perturbation
      int c = 0;
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        adm.g_dd(m,a,b,k,j,i) += rnum(c);
        c++;
      }

      // K_ab perturbation
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        adm.vK_dd(m,a,b,k,j,i) += rnum(c);
        c++;
      }

      rand_pool64.free_state(rand_gen);  // free state for use by other threads

      z4c.alpha(m,k,j,i) = 1;
      z4c.chi(m,k,j,i) = 1;
      z4c.vKhat(m,k,j,i) = 0;
      z4c.vTheta(m,k,j,i) = 0;
    });
  });

  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  return;
}

void Z4cStabilityErrors(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 12;
  pdata->label[0] = "LINF-Err";
  pdata->label[1] = "RMS-Err";

  Real l1_err[6];
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
  if (pmbp->pz4c != nullptr) {
    nvars = 6; // 6 metric components
    auto &pz4c = pmbp->pz4c;
    auto &u0_ = pmbp->pz4c->u0;

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

      // g_ij's:
      array_sum::GlobalSum evars;
      evars.the_array[0] = vol*fabs(1.
                          - u0_(m,pz4c->I_Z4C_GXX,k,j,i));
      max_err = fmax(max_err, evars.the_array[0]);
      evars.the_array[1] = vol*fabs(0.
                          - u0_(m,pz4c->I_Z4C_GXY,k,j,i));
      max_err = fmax(max_err, evars.the_array[1]);
      evars.the_array[2] = vol*fabs(0.
                          - u0_(m,pz4c->I_Z4C_GXZ,k,j,i));
      max_err = fmax(max_err, evars.the_array[2]);
      evars.the_array[3] = vol*fabs(1.
                          - u0_(m,pz4c->I_Z4C_GYY,k,j,i));
      max_err = fmax(max_err, evars.the_array[3]);
      evars.the_array[4] = vol*fabs(0.
                          - u0_(m,pz4c->I_Z4C_GYZ,k,j,i));
      max_err = fmax(max_err, evars.the_array[4]);
      evars.the_array[5] = vol*fabs(1.
                          - u0_(m,pz4c->I_Z4C_GZZ,k,j,i));
      max_err = fmax(max_err, evars.the_array[5]);

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

  pdata->hdata[0] = linfty_err;
  pdata->hdata[1] = rms_err;
  return;
}
