//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file jeans_wave.cpp
//  \brief Problem generator for Jeans wave test with self-gravity
//
//  Tests the linear dispersion relation for sound+gravity waves in a uniform medium.
//  For wave vector k and sound speed cs, the dispersion relation is:
//    omega^2 = k^2*cs^2 - 4*pi*G*rho0
//  The Jeans wavenumber is k_J = sqrt(4*pi*G*rho0)/cs
//  For k < k_J: exponential growth (gravitational instability)
//  For k > k_J: oscillatory waves (modified sound waves)
//
//  References:
//  - Jeans (1902) Phil. Trans. R. Soc. Lond. A 199, 1-53
//  - Stone et al. (2008) ApJS 178, 137-177 (Athena test suite)

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
  Real jw_iso_cs = 1.0;
  Real jw_four_pi_G = 1.0;
  Real jw_njeans_threshold = 16.0;
  Real jw_njeans_derefine = 2.5;
  Real jw_rho0 = 1.0;
  Real jw_amp = 1.0e-6;
  Real jw_k_wave = 0.0;
  Real jw_omega = 0.0;
  Real jw_n_jeans = -1.0;
  Real jw_cos_a2 = 1.0;
  Real jw_cos_a3 = 1.0;
  Real jw_sin_a2 = 0.0;
  Real jw_sin_a3 = 0.0;
  Real jw_v0 = 0.0;
}  // namespace

void JeansWaveRefinement(MeshBlockPack *pmbp);
void JeansWaveErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SelfGravity()
//  \brief Problem Generator for Jeans instability/wave test
//
//  Sets up a sinusoidal density perturbation in a uniform medium:
//    rho = rho0 * (1 + amp * sin(k*x))
//    v = v0 * sin(k*x)  [optionally for traveling wave]
//    P = P0 * (1 + amp * sin(k*x)) [for isentropic perturbation]
//
//  For periodic domain with box size L, wavelength lambda = L/n_waves:
//    k = 2*pi*n_waves/L
//  Compare k to k_J = sqrt(4*pi*G*rho0)/cs to predict behavior

void ProblemGenerator::SelfGravity(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool use_mhd = (pmbp->pmhd != nullptr);
  std::string soe = use_mhd ? "mhd" : "hydro";

  // --- AMR Jeans criterion (must be set on both fresh start and restart) ---
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real n_jeans = pin->GetOrAddReal("problem", "n_jeans", -1.0);
  std::string eos_type = pin->GetString(soe, "eos");
  Real cs;
  Real gamma = 0.0, gm1 = 0.0, p0 = 0.0;
  bool is_isothermal = (eos_type == "isothermal");
  if (is_isothermal) {
    cs = pin->GetReal(soe, "iso_sound_speed");
  } else {
    gamma = pin->GetOrAddReal(soe, "gamma", 5.0/3.0);
    gm1 = gamma - 1.0;
    p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    cs = std::sqrt(gamma * p0 / rho0);
  }

  jw_four_pi_G = four_pi_G;
  jw_iso_cs = cs;
  jw_njeans_threshold = pin->GetOrAddReal("problem", "njeans_amr", 16.0);
  jw_njeans_derefine = pin->GetOrAddReal("problem", "njeans_derefine", 2.5);
  user_ref_func = JeansWaveRefinement;
  pgen_final_func = JeansWaveErrors;

  if (restart) return;

  Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  Real v0 = pin->GetOrAddReal("problem", "v0", 0.0);

  // Get domain size to compute actual wavenumber
  Real Lx1 = pin->GetReal("mesh", "x1max") - pin->GetReal("mesh", "x1min");
  Real Lx2 = pin->GetReal("mesh", "x2max") - pin->GetReal("mesh", "x2min");
  Real Lx3 = pin->GetReal("mesh", "x3max") - pin->GetReal("mesh", "x3min");

  Real ang_3 = std::atan(Lx1/Lx2);
  Real sin_a3 = std::sin(ang_3);
  Real cos_a3 = std::cos(ang_3);
  Real ang_2 = std::atan(0.5*(Lx1*cos_a3 + Lx2*sin_a3)/Lx3);
  Real sin_a2 = std::sin(ang_2);
  Real cos_a2 = std::cos(ang_2);

  Real x1 = Lx1*cos_a2*cos_a3;
  Real x2 = Lx2*cos_a2*sin_a3;
  Real x3 = Lx3*sin_a2;

  Real lambda = std::min(x1, std::min(x2, x3));
  Real lambda_jeans = lambda/n_jeans;
  // If n_jeans is specified, set effective four_pi_G to enforce k_J = n_jeans * k
  if (n_jeans > 0.0) {
    Real G = M_PI * (cs*cs)/(rho0*lambda_jeans*lambda_jeans);
    four_pi_G = 4 * M_PI * G;
    pin->SetReal("gravity", "four_pi_G", four_pi_G);
    if (pmy_mesh_->pmb_pack->pgrav != nullptr) {
      pmy_mesh_->pmb_pack->pgrav->four_pi_G = four_pi_G;
      if (pmy_mesh_->pmb_pack->pgrav->pmgd != nullptr) {
        pmy_mesh_->pmb_pack->pgrav->pmgd->SetFourPiG(four_pi_G);
      }
    }
    jw_four_pi_G = four_pi_G;
  }
  Real k_wave = 2.0*M_PI/lambda;
  Real k_jeans = 2.0*M_PI/lambda_jeans;
  Real omega2 = SQR(k_wave)*cs*cs*(1.0 - SQR(n_jeans));
  Real omega = std::sqrt(std::abs(omega2));

  if (global_variable::my_rank == 0) {
    std::cout << "Jeans wave test parameters:" << std::endl;
    std::cout << "  rho0 = " << rho0 << ", cs = " << cs;
    if (!is_isothermal) std::cout << ", p0 = " << p0;
    std::cout << std::endl;
    std::cout << "  k_Jeans = " << k_jeans << ", k_wave = " << k_wave << std::endl;
    std::cout << "  k/k_J = " << k_wave/k_jeans << std::endl;
    if (n_jeans > 0.0) {
      std::cout << "  n_Jeans = " << n_jeans << " (lambda/lambda_Jeans)" << std::endl;
    }
    std::cout << "  four_pi_G = " << four_pi_G << std::endl;
    if (v0 != 0.0) {
      std::cout << "  Background velocity v0 = " << v0 << std::endl;
    }
    if (k_wave < k_jeans) {
      std::cout << "  Gravitationally unstable! Growth rate = " << omega << std::endl;
    } else {
      std::cout << "  Stable oscillation. Frequency = " << omega << std::endl;
    }
  }

  // Store parameters for JeansWaveErrors
  jw_rho0 = rho0;
  jw_amp = amp;
  jw_k_wave = k_wave;
  jw_omega = omega;
  jw_n_jeans = n_jeans;
  jw_cos_a2 = cos_a2;
  jw_cos_a3 = cos_a3;
  jw_sin_a2 = sin_a2;
  jw_sin_a3 = sin_a3;
  jw_v0 = v0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;

    par_for("jeans_wave_hydro_init", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
      Real sinkx = std::sin(x*k_wave);
      Real coskx = std::cos(x*k_wave);
      Real M = (n_jeans > 1.0) ? rho0*(omega/k_wave)*amp*coskx : 0.0;

      Real dens = rho0*(1.0+amp*sinkx+amp*amp*std::sin(x1v*k_wave));
      u0(m, IDN, k, j, i) = dens;
      u0(m, IM1, k, j, i) = M*cos_a3*cos_a2 + dens*v0*cos_a3*cos_a2;
      u0(m, IM2, k, j, i) = M*sin_a3*cos_a2 + dens*v0*sin_a3*cos_a2;
      u0(m, IM3, k, j, i) = M*sin_a2         + dens*v0*sin_a2;
      if (!is_isothermal) {
        u0(m, IEN, k, j, i) = p0/gm1*(1.0 + gamma*amp*sinkx);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM1, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM2, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM3, k, j, i))/u0(m, IDN, k, j, i);
      }
    });

  // Initialize MHD variables ---------------------------------
  } else if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    Real b0_val = pin->GetOrAddReal("problem", "b0", 0.0);

    par_for("jeans_wave_mhd_init", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
      Real sinkx = std::sin(x*k_wave);
      Real coskx = std::cos(x*k_wave);
      Real M = (n_jeans > 1.0) ? rho0*(omega/k_wave)*amp*coskx : 0.0;

      Real dens = rho0*(1.0+amp*sinkx+amp*amp*std::sin(x1v*k_wave));
      u0(m, IDN, k, j, i) = dens;
      u0(m, IM1, k, j, i) = M*cos_a3*cos_a2 + dens*v0*cos_a3*cos_a2;
      u0(m, IM2, k, j, i) = M*sin_a3*cos_a2 + dens*v0*sin_a3*cos_a2;
      u0(m, IM3, k, j, i) = M*sin_a2         + dens*v0*sin_a2;
      if (!is_isothermal) {
        u0(m, IEN, k, j, i) = p0/gm1*(1.0 + gamma*amp*sinkx);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM1, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM2, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM3, k, j, i))/u0(m, IDN, k, j, i);
        u0(m, IEN, k, j, i) += 0.5*SQR(b0_val);
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("jeans_wave_bfield", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = b0_val;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = b0_val;
      if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
      if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = b0_val;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveRefinement()
//! \brief Jeans-length AMR criterion for the Jeans wave test.
//!
//! Computes the minimum number of cells per Jeans length in each MeshBlock:
//!   nJ = 2*pi*cs / (dx * sqrt(four_pi_G * rho_max))
//! Refines if nJ < threshold, derefines if nJ > 2.5*threshold.

void JeansWaveRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  DvceArray5D<Real> u0;
  if (pmbp->phydro != nullptr) {
    u0 = pmbp->phydro->u0;
  } else {
    u0 = pmbp->pmhd->u0;
  }
  auto &size = pmbp->pmb->mb_size;
  Real cs = jw_iso_cs;
  Real fpG = jw_four_pi_G;
  Real njeans = jw_njeans_threshold;
  Real derefine = jw_njeans_derefine;

  par_for_outer("JeansWaveAMR", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_rhomax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &rhomax) {
        int k = (idx) / nji;
        int j = (idx - k * nji) / nx1;
        int i = (idx - k * nji - j * nx1) + is;
        j += js;
        k += ks;
        rhomax = Kokkos::fmax(u0(m, IDN, k, j, i), rhomax);
      },
      Kokkos::Max<Real>(team_rhomax));

    Real dx = size.d_view(m).dx1;
    Real nj_min = 2.0 * M_PI * cs / (dx * Kokkos::sqrt(fpG * team_rhomax));

    if (nj_min < njeans) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (nj_min > njeans * derefine) {
      refine_flag.d_view(m + mbs) = -1;
    } else {
      refine_flag.d_view(m + mbs) = 0;
    }
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveErrors()
//! \brief Measures the Jeans wave growth rate (unstable) or oscillation
//!        frequency (stable)
//!        by Fourier-projecting the density perturbation onto the initial sin(k*x) mode.
//!
//! Computes A_sin = (2/V) * integral[ (rho/rho0 - 1) * sin(k*x) dV ], then:
//!   unstable (n_jeans > 1): omega_measured = ln(|A_sin| / amp) / t
//!   stable   (n_jeans < 1): omega_measured = arccos(A_sin / amp) / t
//! and compares against the analytical dispersion relation omega.

void JeansWaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  DvceArray5D<Real> u0;
  if (pmbp->phydro != nullptr) {
    u0 = pmbp->phydro->u0;
  } else if (pmbp->pmhd != nullptr) {
    u0 = pmbp->pmhd->u0;
  } else {
    return;
  }

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nk = ke - ks + 1, nj = je - js + 1, ni = ie - is + 1;
  int nmb = pmbp->nmb_thispack;
  int nmkji = nmb * nk * nj * ni;

  auto &size = pmbp->pmb->mb_size;
  Real t = pm->time;

  Real rho0 = jw_rho0;
  Real amp = jw_amp;
  Real k_wave = jw_k_wave;
  Real omega = jw_omega;
  Real n_jeans = jw_n_jeans;
  Real cos_a2 = jw_cos_a2;
  Real cos_a3 = jw_cos_a3;
  Real sin_a2 = jw_sin_a2;
  Real sin_a3 = jw_sin_a3;
  Real v0 = jw_v0;
  bool is_unstable = (n_jeans > 1.0);

  Real sin_proj = 0.0;

  Kokkos::parallel_reduce(
      "jeans_wave_sin_proj",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &sp) {
        int ii = idx % ni;
        int jj = (idx / ni) % nj;
        int kk = (idx / (ni * nj)) % nk;
        int mm = idx / (ni * nj * nk);
        int i = is + ii, j = js + jj, k = ks + kk;

        Real x1min = size.d_view(mm).x1min, x1max = size.d_view(mm).x1max;
        Real x2min = size.d_view(mm).x2min, x2max = size.d_view(mm).x2max;
        Real x3min = size.d_view(mm).x3min, x3max = size.d_view(mm).x3max;
        Real vol = size.d_view(mm).dx1 * size.d_view(mm).dx2 * size.d_view(mm).dx3;

        Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
        Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);
        Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

        Real x = cos_a2*(x1v*cos_a3 + x2v*sin_a3) + x3v*sin_a2;
        sp += (u0(mm, IDN, k, j, i) / rho0 - 1.0)
              * std::sin(k_wave * (x - v0 * t)) * vol;
      },
      Kokkos::Sum<Real>(sin_proj));

#if MPI_PARALLEL_ENABLED
  Real global_sp = 0.0;
  MPI_Allreduce(&sin_proj, &global_sp, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sin_proj = global_sp;
#endif

  Real x1size = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real x2size = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real x3size = pm->mesh_size.x3max - pm->mesh_size.x3min;
  Real tvol = x1size * x2size * x3size;

  Real A_sin = 2.0 * sin_proj / tvol;
  Real omega_measured;
  if (is_unstable) {
    omega_measured = std::log(std::fabs(A_sin) / amp) / t;
  } else {
    Real clamped = fmax(-1.0, fmin(1.0, A_sin / amp));
    omega_measured = std::acos(clamped) / t;
  }

  if (global_variable::my_rank == 0) {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1);
    std::cout << "=====================================================" << std::endl;
    std::cout << "Jeans wave mode amplitude  : " << A_sin << std::endl;
    std::cout << "Jeans wave growth (A/amp)  : " << A_sin / amp << std::endl;
    std::cout << "Jeans wave omega measured  : " << omega_measured << std::endl;
    std::cout << "Jeans wave omega analytical: " << omega << std::endl;
    std::cout << "=====================================================" << std::endl;
  }
}
