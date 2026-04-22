//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_linear_wave.cpp
//! \brief z4c linear (gravitational) wave test


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
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void Z4cLinearWaveErrors(ParameterInput *pin, Mesh *pm);
void LWRefinementCondition(MeshBlockPack* pmbp);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

template <typename Z4cState>
void FillLinearWaveZ4c(MeshBlockPack *pmbp, Z4cState &state, ParameterInput *pin) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &pz4c = pmbp->pz4c;

  Real x1size = pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min;
  Real x2size = pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min;
  Real x3size = pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min;

  Real amp = pin->GetOrAddReal("problem", "amp", 0.001);
  Real kx1 = pin->GetOrAddReal("problem", "kx1", 1. / x1size);
  Real kx2 = pin->GetOrAddReal("problem", "kx2", 1. / x2size);
  Real kx3 = pin->GetOrAddReal("problem", "kx3", 1. / x3size);

  Real knorm = sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));
  Real theta = std::atan2(sqrt(kx2 * kx2 + kx1 * kx1), kx3);
  Real phi = std::atan2(kx1, kx2);

  Real axx = -SQR(cos(theta))*cos(2*phi)-SQR(cos(phi))*SQR(sin(theta));
  Real axy = -0.25*(3+cos(2*theta))*sin(2*phi);
  Real axz = -cos(theta)*sin(theta)*sin(phi);
  Real ayy = SQR(cos(theta))*cos(2*phi)-SQR(sin(theta))*SQR(sin(phi));
  Real ayz = cos(theta)*sin(theta)*cos(phi);
  Real azz = SQR(sin(theta));

  par_for("pgen_linwave1", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
      ks, ke, js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;

        int nx1 = indcs.nx1;
        int nx2 = indcs.nx2;
        int nx3 = indcs.nx3;

        Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
        Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
        Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);
        Real phase = 2 * M_PI * (kx1 * x1v + kx2 * x2v + kx3 * x3v);
        Real sinkx = sin(phase);
        Real coskx = knorm * M_PI * cos(phase);

        state(m,pz4c->I_Z4C_GXX,k,j,i) = 1 + axx * amp * sinkx;
        state(m,pz4c->I_Z4C_GXY,k,j,i) = axy * amp * sinkx;
        state(m,pz4c->I_Z4C_GXZ,k,j,i) = axz * amp * sinkx;
        state(m,pz4c->I_Z4C_GYY,k,j,i) = 1 + ayy * amp * sinkx;
        state(m,pz4c->I_Z4C_GYZ,k,j,i) = ayz * amp * sinkx;
        state(m,pz4c->I_Z4C_GZZ,k,j,i) = 1 + azz * amp * sinkx;

        state(m,pz4c->I_Z4C_AXX,k,j,i) = axx * amp * coskx;
        state(m,pz4c->I_Z4C_AXY,k,j,i) = axy * amp * coskx;
        state(m,pz4c->I_Z4C_AXZ,k,j,i) = axz * amp * coskx;
        state(m,pz4c->I_Z4C_AYY,k,j,i) = ayy * amp * coskx;
        state(m,pz4c->I_Z4C_AYZ,k,j,i) = ayz * amp * coskx;
        state(m,pz4c->I_Z4C_AZZ,k,j,i) = azz * amp * coskx;

        state(m,pz4c->I_Z4C_ALPHA,k,j,i) = 1.0;
        state(m,pz4c->I_Z4C_CHI,k,j,i) = 1.0;
        state(m,pz4c->I_Z4C_KHAT,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_THETA,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_GAMX,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_GAMY,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_GAMZ,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_BETAX,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_BETAY,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_BETAZ,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_BX,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_BY,k,j,i) = 0.0;
        state(m,pz4c->I_Z4C_BZ,k,j,i) = 0.0;
      });
}

template <typename ADMState>
void FillMinkowskiADM(MeshBlockPack *pmbp, ADMState &adm_state) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  par_for("pgen_flat_background", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm_state.g_dd(m,0,0,k,j,i) = 1.0;
    adm_state.g_dd(m,0,1,k,j,i) = 0.0;
    adm_state.g_dd(m,0,2,k,j,i) = 0.0;
    adm_state.g_dd(m,1,1,k,j,i) = 1.0;
    adm_state.g_dd(m,1,2,k,j,i) = 0.0;
    adm_state.g_dd(m,2,2,k,j,i) = 1.0;

    adm_state.vK_dd(m,0,0,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,2,2,k,j,i) = 0.0;

    adm_state.psi4(m,k,j,i) = 1.0;
    adm_state.alpha(m,k,j,i) = 1.0;
    adm_state.beta_u(m,0,k,j,i) = 0.0;
    adm_state.beta_u(m,1,k,j,i) = 0.0;
    adm_state.beta_u(m,2,k,j,i) = 0.0;
  });
}

void SetADMBackgroundFlat(MeshBlockPack *pmbp, Real /*time*/) {
  FillMinkowskiADM(pmbp, pmbp->pz4c->adm_bg);
}
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Z4cLinearWave()
//! \brief Sets initial conditions for gw linear wave tests

void ProblemGenerator::Z4cLinearWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = Z4cLinearWaveErrors;
  user_ref_func  = LWRefinementCondition;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c != nullptr && pmbp->pz4c->use_analytic_background) {
    pmbp->pz4c->SetADMBackground = &SetADMBackgroundFlat;
  }

  if (restart)
    return;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Z4c Wave test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &pz4c = pmbp->pz4c;

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // Initialize wavevector
  Real kx1 = pin->GetOrAddReal("problem", "kx1", 1. / x1size);
  Real kx2 = pin->GetOrAddReal("problem", "kx2", 1. / x2size);
  Real kx3 = pin->GetOrAddReal("problem", "kx3", 1. / x3size);

  // Wavevector length
  Real knorm = sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));

  // Calculate angular offset of the wavevector from zhat
  Real theta = std::atan2(sqrt(kx2 * kx2 + kx1 * kx1), kx3);
  Real phi = std::atan2(kx1, kx2);

  // Find wavelength
  Real lambda = 1/knorm;

  // set new time limit in ParameterInput (to be read by Driver constructor) based on
  // wave speed of selected mode.
  // input tlim is interpreted as number of wave periods for evolution
  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*lambda);
  }

  if (set_initial_conditions) {
    if (pz4c->use_analytic_background && pz4c->SetADMBackground != nullptr) {
      FillLinearWaveZ4c(pmbp, pz4c->u_full, pin);
      pz4c->UpdateBackgroundState(pmbp->pmesh->time);
      pz4c->RecastResidualState();
      pz4c->PrescribeGaugeResidual();
    } else {
      FillLinearWaveZ4c(pmbp, pz4c->u0, pin);
    }
  } else {
    FillLinearWaveZ4c(pmbp, pz4c->u1, pin);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4cLinearWaveErrors()
//! Computes errors in selected variables

void Z4cLinearWaveErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.
  set_initial_conditions = false;
  pm->pgen->Z4cLinearWave(pin, false);

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

  // compute errors
  if (pmbp->pz4c != nullptr) {
    nvars = 6; // 6 metric components
    auto &pz4c = pmbp->pz4c;
    auto &u0_ = (pz4c->use_analytic_background && pz4c->SetADMBackground != nullptr) ?
                pz4c->u_full : pz4c->u0;
    auto &u1_ = pmbp->pz4c->u1;

    if (pz4c->use_analytic_background && pz4c->SetADMBackground != nullptr) {
      pz4c->UpdateBackgroundState(pm->time);
      pz4c->ReconstructFullState();
    }

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
      evars.the_array[0] = vol*fabs(u0_(m,pz4c->I_Z4C_GXX,k,j,i)
                                  - u1_(m,pz4c->I_Z4C_GXX,k,j,i));
      max_err = fmax(max_err, evars.the_array[0]);
      evars.the_array[1] = vol*fabs(u0_(m,pz4c->I_Z4C_GXY,k,j,i)
                                  - u1_(m,pz4c->I_Z4C_GXY,k,j,i));
      max_err = fmax(max_err, evars.the_array[1]);
      evars.the_array[2] = vol*fabs(u0_(m,pz4c->I_Z4C_GXZ,k,j,i)
                                  - u1_(m,pz4c->I_Z4C_GXZ,k,j,i));
      max_err = fmax(max_err, evars.the_array[2]);
      evars.the_array[3] = vol*fabs(u0_(m,pz4c->I_Z4C_GYY,k,j,i)
                                  - u1_(m,pz4c->I_Z4C_GYY,k,j,i));
      max_err = fmax(max_err, evars.the_array[3]);
      evars.the_array[4] = vol*fabs(u0_(m,pz4c->I_Z4C_GYZ,k,j,i)
                                  - u1_(m,pz4c->I_Z4C_GYZ,k,j,i));
      max_err = fmax(max_err, evars.the_array[4]);
      evars.the_array[5] = vol*fabs(u0_(m,pz4c->I_Z4C_GZZ,k,j,i)
                                  - u1_(m,pz4c->I_Z4C_GZZ,k,j,i));
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
      std::fprintf(pfile, "# Nx1   Nx2   Nx3   Ncycle    RMS-L1    L-infty   ");
      std::fprintf(pfile, "gxx_L1   gxy_L1    gxz_L1    gyy_L1    ");
      std::fprintf(pfile, "gyz_L1   gzz_L1    \n");
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
//! \fn void LWRefinementCondition()
//! Implements custom AMR refinement condition for linear gravitational waves

void LWRefinementCondition(MeshBlockPack* pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int I_Z4C_GXY  = pmbp->pz4c->I_Z4C_GXY;
  int nmb           = pmbp->nmb_thispack;
  auto &indcs       = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  int mbs           = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &u0       = pmbp->pz4c->u0;

  par_for_outer("Z4c_AMR::GXYMAX", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_dmax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &dmax) {
        int k = (idx) / nji;
        int j = (idx - k * nji) / nx1;
        int i = (idx - k * nji - j * nx1) + is;
        j += js;
        k += ks;
        dmax = fmax(u0(m, I_Z4C_GXY, k, j, i), dmax);
      },
      Kokkos::Max<Real>(team_dmax));

    if (team_dmax > 0) {
      refine_flag.d_view(m + mbs) = 1;
    } else {
      refine_flag.d_view(m + mbs) = -1;
    }
  });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
