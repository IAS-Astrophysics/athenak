//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file diffusion.cpp
//! \brief Problem generator for exact diffusion tests used by the regression suite.
//! Supports:
//!   - Hydro viscous diffusion of Gaussian transverse momentum
//!   - Hydro thermal diffusion of Gaussian internal energy
//!   - MHD Ohmic diffusion of Gaussian transverse magnetic field
//! This file also contains the final-error routine called from Driver::Finalize().

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // string

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/resistivity.hpp"

// Prototype for function to compute errors in solution at end of run
void DiffusionErrors(ParameterInput *pin, Mesh *pm);
// Prototype for user-defined BCs
void GaussianProfile(Mesh *pm);

namespace {

constexpr Real kPi = 3.14159265358979323846;

enum class DiffusionTestKind {
  hydro_viscosity,
  hydro_conduction,
  mhd_resistivity
};

bool set_initial_conditions = true;

struct DiffusionVariables {
  Real d0 = 1.0;
  Real amp = 1.0e-6;
  Real t0 = 0.5;
  Real x10 = 0.0;
  DiffusionTestKind test_kind = DiffusionTestKind::hydro_viscosity;
};

DiffusionVariables dv;

[[noreturn]] void DiffusionFatal(const char *file, int line, const std::string &message) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

DiffusionTestKind ParseDiffusionTestKind(ParameterInput *pin) {
  std::string diffusion_test =
      pin->GetOrAddString("problem", "diffusion_test", "hydro_viscosity");
  if (diffusion_test == "hydro_viscosity") {
    return DiffusionTestKind::hydro_viscosity;
  } else if (diffusion_test == "hydro_conduction") {
    return DiffusionTestKind::hydro_conduction;
  } else if (diffusion_test == "mhd_resistivity") {
    return DiffusionTestKind::mhd_resistivity;
  }

  DiffusionFatal(__FILE__, __LINE__,
                 "problem/diffusion_test = '" + diffusion_test +
                     "' must be one of 'hydro_viscosity', 'hydro_conduction', "
                     "or 'mhd_resistivity'");
}

KOKKOS_INLINE_FUNCTION
Real GaussianKernel1D(const Real x1, const Real x10, const Real diffusivity,
                      const Real time) {
  return exp(SQR(x1 - x10) / (-4.0 * diffusivity * time)) /
         sqrt(4.0 * kPi * diffusivity * time);
}

void ValidateHydroDiffusionConfig(hydro::Hydro *phydro, const DiffusionTestKind test_kind) {
  if (phydro == nullptr) {
    DiffusionFatal(__FILE__, __LINE__,
                   "Requested Hydro diffusion test, but no Hydro module is active");
  }
  if (!(phydro->peos->eos_data.is_ideal)) {
    DiffusionFatal(__FILE__, __LINE__,
                   "Hydro diffusion tests require the ideal-gas Hydro EOS");
  }

  if (test_kind == DiffusionTestKind::hydro_viscosity) {
    if (phydro->pvisc == nullptr) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_viscosity requires <hydro>/viscosity");
    }
    if (phydro->pvisc->nu_iso <= 0.0) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_viscosity requires a positive "
                     "<hydro>/viscosity coefficient");
    }
    if (phydro->pcond != nullptr) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_viscosity expects viscosity only; "
                     "disable thermal conduction for this exact test");
    }
  } else if (test_kind == DiffusionTestKind::hydro_conduction) {
    if (phydro->pcond == nullptr) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_conduction requires "
                     "<hydro>/conductivity or <hydro>/tdep_conductivity");
    }
    if (phydro->pvisc != nullptr) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_conduction expects thermal "
                     "conduction only; disable viscosity for this exact test");
    }
    if (phydro->pcond->tdep_kappa) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_conduction only supports constant "
                     "conductivity, not temperature-dependent conductivity");
    }
    if (phydro->pcond->sat_hflux) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_conduction does not support "
                     "saturated heat flux");
    }
    if (phydro->pcond->kappa <= 0.0) {
      DiffusionFatal(__FILE__, __LINE__,
                     "problem/diffusion_test = hydro_conduction requires a positive "
                     "<hydro>/conductivity coefficient");
    }
  }
}

void ValidateMHDDiffusionConfig(mhd::MHD *pmhd) {
  if (pmhd == nullptr) {
    DiffusionFatal(__FILE__, __LINE__,
                   "Requested MHD diffusion test, but no MHD module is active");
  }
  if (pmhd->presist == nullptr) {
    DiffusionFatal(__FILE__, __LINE__,
                   "problem/diffusion_test = mhd_resistivity requires "
                   "<mhd>/ohmic_resistivity");
  }
  if (pmhd->pvisc != nullptr || pmhd->pcond != nullptr) {
    DiffusionFatal(__FILE__, __LINE__,
                   "problem/diffusion_test = mhd_resistivity expects Ohmic diffusion "
                   "only; disable MHD viscosity and thermal conduction for this exact test");
  }
  if (pmhd->peos->eos_data.is_ideal) {
    DiffusionFatal(__FILE__, __LINE__,
                   "problem/diffusion_test = mhd_resistivity requires isothermal MHD "
                   "so resistive heating does not spoil the exact reference");
  }
  if (pmhd->presist->eta_ohm <= 0.0) {
    DiffusionFatal(__FILE__, __LINE__,
                   "problem/diffusion_test = mhd_resistivity requires a positive "
                   "<mhd>/ohmic_resistivity coefficient");
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Diffusion()
//! \brief Problem generator for the regression-suite diffusion problems.

void ProblemGenerator::Diffusion(ParameterInput *pin, const bool restart) {
  pgen_final_func = DiffusionErrors;
  user_bcs_func = GaussianProfile;
  if (restart) return;

  dv.test_kind = ParseDiffusionTestKind(pin);
  dv.d0 = 1.0;
  dv.amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  dv.t0 = pin->GetOrAddReal("problem", "t0", 0.5);
  dv.x10 = pin->GetOrAddReal("problem", "x10", 0.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  Real solution_time = dv.t0;
  if (!set_initial_conditions) {
    solution_time += pmbp->pmesh->time;
  }

  switch (dv.test_kind) {
    case DiffusionTestKind::hydro_viscosity: {
      if (pmbp->pmhd != nullptr) {
        DiffusionFatal(__FILE__, __LINE__,
                       "problem/diffusion_test = hydro_viscosity requires Hydro only; "
                       "do not use the MHD module for this exact test");
      }
      ValidateHydroDiffusionConfig(pmbp->phydro, dv.test_kind);

      hydro::Hydro *phydro = pmbp->phydro;
      EOS_Data &eos = phydro->peos->eos_data;
      Real gm1 = eos.gamma - 1.0;
      Real p0 = 1.0 / eos.gamma;
      Real nu_iso = phydro->pvisc->nu_iso;
      int nhydro = phydro->nhydro;
      int nscalars = phydro->nscalars;
      auto &u = (set_initial_conditions) ? phydro->u0 : phydro->u1;
      auto d0_ = dv.d0;
      auto amp_ = dv.amp;
      auto x10_ = dv.x10;

      par_for("pgen_diffusion_viscosity", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
              js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
                Real profile = d0_ * amp_ * GaussianKernel1D(x1v, x10_, nu_iso, solution_time);

                u(m, IDN, k, j, i) = d0_;
                u(m, IM1, k, j, i) = 0.0;
                u(m, IM2, k, j, i) = profile;
                u(m, IM3, k, j, i) = profile;
                if (eos.is_ideal) {
                  u(m, IEN, k, j, i) = p0 / gm1 + SQR(profile) / d0_;
                }
                for (int n = nhydro; n < (nhydro + nscalars); ++n) {
                  u(m, n, k, j, i) = 0.0;
                }
              });
      break;
    }

    case DiffusionTestKind::hydro_conduction: {
      if (pmbp->pmhd != nullptr) {
        DiffusionFatal(__FILE__, __LINE__,
                       "problem/diffusion_test = hydro_conduction requires Hydro only; "
                       "do not use the MHD module for this exact test");
      }
      ValidateHydroDiffusionConfig(pmbp->phydro, dv.test_kind);

      hydro::Hydro *phydro = pmbp->phydro;
      EOS_Data &eos = phydro->peos->eos_data;
      Real gm1 = eos.gamma - 1.0;
      Real p0 = 1.0 / eos.gamma;
      Real chi = phydro->pcond->kappa * gm1 / dv.d0;
      int nhydro = phydro->nhydro;
      int nscalars = phydro->nscalars;
      auto &u = (set_initial_conditions) ? phydro->u0 : phydro->u1;
      auto d0_ = dv.d0;
      auto amp_ = dv.amp;
      auto x10_ = dv.x10;

      par_for("pgen_diffusion_conduction", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
              js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
                Real profile = amp_ * GaussianKernel1D(x1v, x10_, chi, solution_time);

                u(m, IDN, k, j, i) = d0_;
                u(m, IM1, k, j, i) = 0.0;
                u(m, IM2, k, j, i) = 0.0;
                u(m, IM3, k, j, i) = 0.0;
                u(m, IEN, k, j, i) = p0 / gm1 + profile;
                for (int n = nhydro; n < (nhydro + nscalars); ++n) {
                  u(m, n, k, j, i) = 0.0;
                }
              });
      break;
    }

    case DiffusionTestKind::mhd_resistivity: {
      if (pmbp->phydro != nullptr) {
        DiffusionFatal(__FILE__, __LINE__,
                       "problem/diffusion_test = mhd_resistivity requires MHD only; "
                       "do not enable the Hydro module for this exact test");
      }
      ValidateMHDDiffusionConfig(pmbp->pmhd);

      mhd::MHD *pmhd = pmbp->pmhd;
      Real eta_ohm = pmhd->presist->eta_ohm;
      int nmhd = pmhd->nmhd;
      int nscalars = pmhd->nscalars;
      auto &u = (set_initial_conditions) ? pmhd->u0 : pmhd->u1;
      auto &b = (set_initial_conditions) ? pmhd->b0 : pmhd->b1;
      auto d0_ = dv.d0;
      auto amp_ = dv.amp;
      auto x10_ = dv.x10;

      par_for("pgen_diffusion_mhd_cons", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
              js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                u(m, IDN, k, j, i) = d0_;
                u(m, IM1, k, j, i) = 0.0;
                u(m, IM2, k, j, i) = 0.0;
                u(m, IM3, k, j, i) = 0.0;
                for (int n = nmhd; n < (nmhd + nscalars); ++n) {
                  u(m, n, k, j, i) = 0.0;
                }
              });

      par_for("pgen_diffusion_mhd_bx", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
              js, je, is, ie + 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                b.x1f(m, k, j, i) = 0.0;
              });

      par_for("pgen_diffusion_mhd_by", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
              js, je + 1, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
                b.x2f(m, k, j, i) = amp_ * GaussianKernel1D(x1v, x10_, eta_ohm, solution_time);
              });

      par_for("pgen_diffusion_mhd_bz", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke + 1,
              js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
                b.x3f(m, k, j, i) = amp_ * GaussianKernel1D(x1v, x10_, eta_ohm, solution_time);
              });
      break;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn DiffusionErrors()
//! \brief Compute the L1 and Linfty errors against the analytic diffusion solution.

void DiffusionErrors(ParameterInput *pin, Mesh *pm) {
  set_initial_conditions = false;
  pm->pgen->Diffusion(pin, false);
  set_initial_conditions = true;

  Real l1_err[NREDUCTION_VARIABLES];
  for (int n = 0; n < NREDUCTION_VARIABLES; ++n) {
    l1_err[n] = 0.0;
  }
  Real linfty_err = 0.0;
  int nvars = 0;

  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  if (dv.test_kind == DiffusionTestKind::hydro_viscosity ||
      dv.test_kind == DiffusionTestKind::hydro_conduction) {
    auto *phydro = pmbp->phydro;
    nvars = phydro->nhydro;

    auto &u0_ = phydro->u0;
    auto &u1_ = phydro->u1;
    const int nmkji = (pmbp->nmb_thispack) * nx3 * nx2 * nx1;
    const int nkji = nx3 * nx2 * nx1;
    const int nji = nx2 * nx1;
    array_sum::GlobalSum sum_this_mb;

    Kokkos::parallel_reduce(
        "diffusion_hydro_err", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
          int m = (idx) / nkji;
          int k = (idx - m * nkji) / nji;
          int j = (idx - m * nkji - k * nji) / nx1;
          int i = (idx - m * nkji - k * nji - j * nx1) + is;
          k += ks;
          j += js;

          Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
          array_sum::GlobalSum evars;
          for (int n = 0; n < nvars; ++n) {
            evars.the_array[n] = vol * fabs(u0_(m, n, k, j, i) - u1_(m, n, k, j, i));
            max_err = fmax(max_err, evars.the_array[n]);
          }
          for (int n = nvars; n < NREDUCTION_VARIABLES; ++n) {
            evars.the_array[n] = 0.0;
          }
          mb_sum += evars;
        },
        Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    for (int n = 0; n < nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  } else if (dv.test_kind == DiffusionTestKind::mhd_resistivity) {
    auto *pmhd = pmbp->pmhd;
    nvars = pmhd->nmhd + NMAG;

    auto &u0_ = pmhd->u0;
    auto &u1_ = pmhd->u1;
    auto &b0_ = pmhd->b0;
    auto &b1_ = pmhd->b1;
    const int nmkji = (pmbp->nmb_thispack) * nx3 * nx2 * nx1;
    const int nkji = nx3 * nx2 * nx1;
    const int nji = nx2 * nx1;
    array_sum::GlobalSum sum_this_mb;

    Kokkos::parallel_reduce(
        "diffusion_mhd_err", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
          int m = (idx) / nkji;
          int k = (idx - m * nkji) / nji;
          int j = (idx - m * nkji - k * nji) / nx1;
          int i = (idx - m * nkji - k * nji - j * nx1) + is;
          k += ks;
          j += js;

          Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
          array_sum::GlobalSum evars;
          for (int n = 0; n < pmhd->nmhd; ++n) {
            evars.the_array[n] = vol * fabs(u0_(m, n, k, j, i) - u1_(m, n, k, j, i));
            max_err = fmax(max_err, evars.the_array[n]);
          }

          Real bx0 = 0.5 * (b0_.x1f(m, k, j, i) + b0_.x1f(m, k, j, i + 1));
          Real bx1 = 0.5 * (b1_.x1f(m, k, j, i) + b1_.x1f(m, k, j, i + 1));
          evars.the_array[pmhd->nmhd + IBX] = vol * fabs(bx0 - bx1);
          evars.the_array[pmhd->nmhd + IBY] =
              vol * fabs(b0_.x2f(m, k, j, i) - b1_.x2f(m, k, j, i));
          evars.the_array[pmhd->nmhd + IBZ] =
              vol * fabs(b0_.x3f(m, k, j, i) - b1_.x3f(m, k, j, i));
          max_err = fmax(max_err, evars.the_array[pmhd->nmhd + IBX]);
          max_err = fmax(max_err, evars.the_array[pmhd->nmhd + IBY]);
          max_err = fmax(max_err, evars.the_array[pmhd->nmhd + IBZ]);

          for (int n = nvars; n < NREDUCTION_VARIABLES; ++n) {
            evars.the_array[n] = 0.0;
          }
          mb_sum += evars;
        },
        Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    for (int n = 0; n < nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nvars, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  Real vol = (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min) *
             (pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min) *
             (pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int n = 0; n < nvars; ++n) {
    l1_err[n] /= vol;
  }
  linfty_err /= vol;

  Real rms_err = 0.0;
  for (int n = 0; n < nvars; ++n) {
    rms_err += SQR(l1_err[n]);
  }
  rms_err = std::sqrt(rms_err);

  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job", "basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        DiffusionFatal(__FILE__, __LINE__, "Error output file could not be opened");
      }
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        DiffusionFatal(__FILE__, __LINE__, "Error output file could not be opened");
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1    L-infty");
      if (dv.test_kind == DiffusionTestKind::mhd_resistivity) {
        std::fprintf(pfile, "       d_L1         M1_L1         M2_L1         M3_L1");
        std::fprintf(pfile, "         B1_L1         B2_L1         B3_L1");
      } else {
        std::fprintf(pfile, "       d_L1         M1_L1         M2_L1         M3_L1");
        std::fprintf(pfile, "         E_L1");
      }
      std::fprintf(pfile, "\n");
    }

    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int n = 0; n < nvars; ++n) {
      std::fprintf(pfile, "  %e", l1_err[n]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}

//----------------------------------------------------------------------------------------
//! \fn GaussianProfile()
//! \brief User boundary conditions that enforce the analytic diffusion profile at x1.

void GaussianProfile(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;
  Real solution_time = dv.t0 + pm->time;
  auto d0_ = dv.d0;
  auto amp_ = dv.amp;
  auto x10_ = dv.x10;

  if (dv.test_kind == DiffusionTestKind::hydro_viscosity ||
      dv.test_kind == DiffusionTestKind::hydro_conduction) {
    hydro::Hydro *phydro = pm->pmb_pack->phydro;
    EOS_Data &eos = phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0 / eos.gamma;
    auto &u0 = phydro->u0;
    int nhydro = phydro->nhydro;
    int nscalars = phydro->nscalars;
    Real diffusivity = (dv.test_kind == DiffusionTestKind::hydro_viscosity)
                           ? phydro->pvisc->nu_iso
                           : phydro->pcond->kappa * gm1 / dv.d0;
    bool is_viscosity_test = (dv.test_kind == DiffusionTestKind::hydro_viscosity);

    par_for("diffusion_hydro_x1", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0, (n2 - 1),
            KOKKOS_LAMBDA(int m, int k, int j) {
              if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
                for (int i = 0; i < ng; ++i) {
                  Real &x1min = size.d_view(m).x1min;
                  Real &x1max = size.d_view(m).x1max;
                  int nx1 = indcs.nx1;
                  Real x1v = CellCenterX(-1 - i, nx1, x1min, x1max);
                  Real profile =
                      amp_ * GaussianKernel1D(x1v, x10_, diffusivity, solution_time);

                  u0(m, IDN, k, j, is - i - 1) = d0_;
                  u0(m, IM1, k, j, is - i - 1) = 0.0;
                  if (is_viscosity_test) {
                    Real mom = d0_ * profile;
                    u0(m, IM2, k, j, is - i - 1) = mom;
                    u0(m, IM3, k, j, is - i - 1) = mom;
                    u0(m, IEN, k, j, is - i - 1) = p0 / gm1 + SQR(mom) / d0_;
                  } else {
                    u0(m, IM2, k, j, is - i - 1) = 0.0;
                    u0(m, IM3, k, j, is - i - 1) = 0.0;
                    u0(m, IEN, k, j, is - i - 1) = p0 / gm1 + profile;
                  }
                  for (int n = nhydro; n < (nhydro + nscalars); ++n) {
                    u0(m, n, k, j, is - i - 1) = 0.0;
                  }
                }
              }

              if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
                for (int i = 0; i < ng; ++i) {
                  Real &x1min = size.d_view(m).x1min;
                  Real &x1max = size.d_view(m).x1max;
                  int nx1 = indcs.nx1;
                  Real x1v = CellCenterX(ie - is + 1 + i, nx1, x1min, x1max);
                  Real profile =
                      amp_ * GaussianKernel1D(x1v, x10_, diffusivity, solution_time);

                  u0(m, IDN, k, j, ie + i + 1) = d0_;
                  u0(m, IM1, k, j, ie + i + 1) = 0.0;
                  if (is_viscosity_test) {
                    Real mom = d0_ * profile;
                    u0(m, IM2, k, j, ie + i + 1) = mom;
                    u0(m, IM3, k, j, ie + i + 1) = mom;
                    u0(m, IEN, k, j, ie + i + 1) = p0 / gm1 + SQR(mom) / d0_;
                  } else {
                    u0(m, IM2, k, j, ie + i + 1) = 0.0;
                    u0(m, IM3, k, j, ie + i + 1) = 0.0;
                    u0(m, IEN, k, j, ie + i + 1) = p0 / gm1 + profile;
                  }
                  for (int n = nhydro; n < (nhydro + nscalars); ++n) {
                    u0(m, n, k, j, ie + i + 1) = 0.0;
                  }
                }
              }
            });
    return;
  }

  mhd::MHD *pmhd = pm->pmb_pack->pmhd;
  auto &u0 = pmhd->u0;
  auto &b0 = pmhd->b0;
  int nmhd = pmhd->nmhd;
  int nscalars = pmhd->nscalars;
  Real eta_ohm = pmhd->presist->eta_ohm;

  par_for("diffusion_mhd_x1", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0, (n2 - 1),
          KOKKOS_LAMBDA(int m, int k, int j) {
            if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
              for (int i = 0; i < ng; ++i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1v = CellCenterX(-1 - i, nx1, x1min, x1max);
                Real bprofile = amp_ * GaussianKernel1D(x1v, x10_, eta_ohm, solution_time);

                u0(m, IDN, k, j, is - i - 1) = d0_;
                u0(m, IM1, k, j, is - i - 1) = 0.0;
                u0(m, IM2, k, j, is - i - 1) = 0.0;
                u0(m, IM3, k, j, is - i - 1) = 0.0;
                for (int n = nmhd; n < (nmhd + nscalars); ++n) {
                  u0(m, n, k, j, is - i - 1) = 0.0;
                }

                b0.x1f(m, k, j, is - i - 1) = 0.0;
                b0.x2f(m, k, j, is - i - 1) = bprofile;
                if (j == n2 - 1) {
                  b0.x2f(m, k, j + 1, is - i - 1) = bprofile;
                }
                b0.x3f(m, k, j, is - i - 1) = bprofile;
                if (k == n3 - 1) {
                  b0.x3f(m, k + 1, j, is - i - 1) = bprofile;
                }
              }
            }

            if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
              for (int i = 0; i < ng; ++i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1v = CellCenterX(ie - is + 1 + i, nx1, x1min, x1max);
                Real bprofile = amp_ * GaussianKernel1D(x1v, x10_, eta_ohm, solution_time);

                u0(m, IDN, k, j, ie + i + 1) = d0_;
                u0(m, IM1, k, j, ie + i + 1) = 0.0;
                u0(m, IM2, k, j, ie + i + 1) = 0.0;
                u0(m, IM3, k, j, ie + i + 1) = 0.0;
                for (int n = nmhd; n < (nmhd + nscalars); ++n) {
                  u0(m, n, k, j, ie + i + 1) = 0.0;
                }

                b0.x1f(m, k, j, ie + i + 2) = 0.0;
                b0.x2f(m, k, j, ie + i + 1) = bprofile;
                if (j == n2 - 1) {
                  b0.x2f(m, k, j + 1, ie + i + 1) = bprofile;
                }
                b0.x3f(m, k, j, ie + i + 1) = bprofile;
                if (k == n3 - 1) {
                  b0.x3f(m, k + 1, j, ie + i + 1) = bprofile;
                }
              }
            }
          });
}
