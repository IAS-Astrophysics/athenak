//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_diffusiontest_erf.cpp
//! \brief initializes the 1d diffusion test in a moving medium with erf initial data

// C++ headers
#include <iostream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "adm/adm.hpp"

void ProblemGenerator::RadiationFEMNDiffusiontestErf(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion problem generator can only be run with radiation-femn, but no "
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!pmbp->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion problem generator can only be run with one dimension, but parfile"
              << "grid setup is not in 1d" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradfemn->num_energy_bins != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 1d diffusion problem generator can only be run with one energy bin!" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int npts1 = pmbp->pradfemn->num_points_total - 1;

  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;
  int nmb = pmbp->nmb_thispack;
  auto &u_mu_ = pmbp->pradfemn->u_mu;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  auto &f0_ = pmbp->pradfemn->f0;
  auto &energy_grid_ = pmbp->pradfemn->energy_grid;
  auto &kappa_s_ = pmbp->pradfemn->kappa_s;
  auto vx = pin->GetOrAddReal("radiation-femn", "fluid_velocity_x", 0.87);
  auto shock = pin->GetOrAddBoolean("problem", "shock", false);
  auto steepness_par = pin->GetOrAddReal("problem", "tanh_par", 5.);
  auto lorentz_w = 1. / sqrt(1 - vx * vx);

  if(shock) {
    if (!pmbp->pradfemn->fpn) {
    par_for("pgen_diffusiontest_radiation_femn_shock", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, npts1, ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int A, int k, int j, int i) {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              f0_(m, A, k, j, i) = (1. / (4. * M_PI)) * (x1 < -0.5);
            });
  } else {
    par_for("pgen_diffusiontest_radiation_fpn_shock", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              f0_(m, 0, k, j, i) = (1. / (4. * M_PI)) * 2. * sqrt(M_PI) * (x1 < -0.5);
            });
  }

    par_for("pgen_diffusiontest_metric_velocity_shock_initialize", DevExeSpace(),
            0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            for (int a = 0; a < 3; ++a) {
              for (int b = a; b < 3; ++b) {
                adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
              }
            }

            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            int nx1 = indcs.nx1;
            Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

            adm.psi4(m, k, j, i) = 1.; // adm.psi4

            adm.alpha(m, k, j, i) = 1.;

            Real velocity = -Kokkos::abs(vx) * Kokkos::tanh(steepness_par * x1);
            Real lorentz_factor = 1. / sqrt(1 - velocity * velocity);

            u_mu_(m, 0, k, j, i) = lorentz_factor;
            u_mu_(m, 1, k, j, i) = velocity * lorentz_factor;
            u_mu_(m, 2, k, j, i) = 0.;
            u_mu_(m, 3, k, j, i) = 0.;

    });
  } else {
    if (!pmbp->pradfemn->fpn) {
    par_for("pgen_diffusiontest_radiation_femn", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, npts1, ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int A, int k, int j, int i) {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              f0_(m, A, k, j, i) = (1. / (4. * M_PI)) * (x1 < 0);
            });
  } else {
    par_for("pgen_diffusiontest_radiation_fpn", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              f0_(m, 0, k, j, i) = (1. / (4. * M_PI)) * 2. * sqrt(M_PI) * (x1 < 0);
            });
  }

    // set metric to minkowski, initialize velocity and opacity
    par_for("pgen_diffusiontest_metric_velocity_initialize",
            DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
              for (int a = 0; a < 3; ++a) {
                for (int b = a; b < 3; ++b) {
                  adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
                }
              }

              adm.psi4(m, k, j, i) = 1.; // adm.psi4

              adm.alpha(m, k, j, i) = 1.;

              u_mu_(m, 0, k, j, i) = lorentz_w;
              u_mu_(m, 1, k, j, i) = vx * lorentz_w;
              u_mu_(m, 2, k, j, i) = 0.;
              u_mu_(m, 3, k, j, i) = 0.;

              kappa_s_(m, k, j, i) = 1e3;
            });
  }
}