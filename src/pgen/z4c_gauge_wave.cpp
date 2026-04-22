//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_gauge_wave.cpp
//! \brief z4c gauge wave test


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

// function to compute errors in solution at end of run
void Z4cGaugeWaveErrors(ParameterInput *pin, Mesh *pm);

namespace {

Real gauge_wave_amp = 0.0;

template <typename ADMState>
void FillGaugeWaveADM(MeshBlockPack *pmbp, ADMState &adm_state, Real time) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = js - indcs.ng;
  int jeg = je + indcs.ng;
  int ksg = ks - indcs.ng;
  int keg = ke + indcs.ng;

  Real x1size = pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min;

  par_for("pgen_gauge_wave", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
    Real phase = 2.0 * M_PI * (x1v - time) / x1size;
    Real H = gauge_wave_amp * sin(phase);
    Real one_minus_H = 1.0 - H;
    Real dH_dt = -gauge_wave_amp * 2.0 * M_PI / x1size * cos(phase);

    adm_state.g_dd(m,0,0,k,j,i) = one_minus_H;
    adm_state.g_dd(m,0,1,k,j,i) = 0.0;
    adm_state.g_dd(m,0,2,k,j,i) = 0.0;
    adm_state.g_dd(m,1,1,k,j,i) = 1.0;
    adm_state.g_dd(m,1,2,k,j,i) = 0.0;
    adm_state.g_dd(m,2,2,k,j,i) = 1.0;

    adm_state.vK_dd(m,0,0,k,j,i) = 0.5 * dH_dt / std::sqrt(one_minus_H);
    adm_state.vK_dd(m,0,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,2,2,k,j,i) = 0.0;

    adm_state.psi4(m,k,j,i) = std::pow(one_minus_H, 1.0/3.0);
    adm_state.alpha(m,k,j,i) = std::sqrt(one_minus_H);
    adm_state.beta_u(m,0,k,j,i) = 0.0;
    adm_state.beta_u(m,1,k,j,i) = 0.0;
    adm_state.beta_u(m,2,k,j,i) = 0.0;
  });
}

void SetADMBackgroundGaugeWave(MeshBlockPack *pmbp, Real time) {
  FillGaugeWaveADM(pmbp, pmbp->pz4c->adm_bg, time);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for gw linear wave tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c != nullptr) {
    pmbp->pz4c->SetADMBackground = &SetADMBackgroundGaugeWave;
    if (pmbp->pz4c->use_analytic_background) {
      pgen_final_func = Z4cGaugeWaveErrors;
    }
  }
  if (restart)
    return;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Z4c Wave test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Wave amplitude
  gauge_wave_amp = pin->GetOrAddReal("problem", "amp", 0.001);

  auto &pz4c = pmbp->pz4c;
  FillGaugeWaveADM(pmbp, pmbp->padm->adm, pmbp->pmesh->time);

  auto &indcs = pmbp->pmesh->mb_indcs;
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  if (pz4c->use_analytic_background && pz4c->SetADMBackground != nullptr) {
    pz4c->UpdateBackgroundState(pmbp->pmesh->time);
    Kokkos::deep_copy(DevExeSpace(), pz4c->u_full, pz4c->u0);
    pz4c->RecastResidualState();
    pz4c->PrescribeGaugeResidual();
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

void Z4cGaugeWaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pz4c = pmbp->pz4c;
  if (pz4c == nullptr || pz4c->SetADMBackground == nullptr) {
    return;
  }

  pz4c->UpdateBackgroundState(pm->time);
  pz4c->ReconstructFullState();
  pz4c->RecastResidualState();
  pz4c->PrescribeGaugeResidual();

  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;

  auto &u_full = pz4c->u_full;
  auto &u_bg = pz4c->u_bg;
  auto &u0 = pz4c->u0;

  const int nmb = pmbp->nmb_thispack;
  const int ncell = nmb * nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const int nkji = nx3 * nji;
  const int nz4c = pz4c->nz4c;

  Real full_bg_sq = 0.0;
  Real full_bg_linf = 0.0;
  Kokkos::parallel_reduce(
      "GaugeWaveFullBgError",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell * nz4c),
      KOKKOS_LAMBDA(const int idx, Real &sum_sq, Real &max_err) {
        const int n = idx % nz4c;
        const int cell = idx / nz4c;
        const int m = cell / nkji;
        const int rem = cell - m * nkji;
        const int k = rem / nji + ks;
        const int rem2 = rem - (k - ks) * nji;
        const int j = rem2 / nx1 + js;
        const int i = rem2 - (j - js) * nx1 + is;

        Real diff = u_full(m,n,k,j,i) - u_bg(m,n,k,j,i);
        Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
        sum_sq += vol * diff * diff;
        max_err = fmax(max_err, fabs(diff));
      },
      Kokkos::Sum<Real>(full_bg_sq), Kokkos::Max<Real>(full_bg_linf));

  constexpr int ngauge = 7;
  Real gauge_sq = 0.0;
  Real gauge_linf = 0.0;
  Kokkos::parallel_reduce(
      "GaugeWaveResidualGaugeError",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell * ngauge),
      KOKKOS_LAMBDA(const int idx, Real &sum_sq, Real &max_err) {
        const int n = idx % ngauge;
        const int cell = idx / ngauge;
        const int m = cell / nkji;
        const int rem = cell - m * nkji;
        const int k = rem / nji + ks;
        const int rem2 = rem - (k - ks) * nji;
        const int j = rem2 / nx1 + js;
        const int i = rem2 - (j - js) * nx1 + is;

        int var = pz4c->I_Z4C_ALPHA;
        switch (n) {
          case 0: var = pz4c->I_Z4C_ALPHA; break;
          case 1: var = pz4c->I_Z4C_BETAX; break;
          case 2: var = pz4c->I_Z4C_BETAY; break;
          case 3: var = pz4c->I_Z4C_BETAZ; break;
          case 4: var = pz4c->I_Z4C_BX; break;
          case 5: var = pz4c->I_Z4C_BY; break;
          case 6: var = pz4c->I_Z4C_BZ; break;
        }

        Real diff = u0(m,var,k,j,i);
        Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
        sum_sq += vol * diff * diff;
        max_err = fmax(max_err, fabs(diff));
      },
      Kokkos::Sum<Real>(gauge_sq), Kokkos::Max<Real>(gauge_linf));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &full_bg_sq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &full_bg_linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gauge_sq, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gauge_linf, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  Real domain_vol = (pm->mesh_size.x1max - pm->mesh_size.x1min) *
                    (pm->mesh_size.x2max - pm->mesh_size.x2min) *
                    (pm->mesh_size.x3max - pm->mesh_size.x3min);
  Real full_bg_rms = std::sqrt(full_bg_sq / (domain_vol * nz4c));
  Real gauge_rms = std::sqrt(gauge_sq / (domain_vol * ngauge));

  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job", "basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile,
                   "# Nx1   Nx2   Nx3   Ncycle   Time   FullBg_RMS   FullBg_Linf"
                   "   GaugeRes_RMS   GaugeRes_Linf\n");
    }

    std::fprintf(pfile, "%04d", pm->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pm->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pm->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e  %e  %e  %e  %e",
                 pm->ncycle, pm->time, full_bg_rms, full_bg_linf, gauge_rms, gauge_linf);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}
