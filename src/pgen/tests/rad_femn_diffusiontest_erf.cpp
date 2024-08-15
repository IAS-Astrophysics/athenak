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
  int nmb1 = nmb - 1;
  auto &u_mu_ = pmbp->pradfemn->u_mu;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  auto &f0_ = pmbp->pradfemn->f0;
  auto &tetr_mu_muhat0_ = pmbp->pradfemn->L_mu_muhat0;
  auto &beam_source_1_vals_ = pmbp->pradfemn->beam_source_1_vals;
  auto &num_points_ = pmbp->pradfemn->num_points;
  auto &rad_E_floor_ = pmbp->pradfemn->rad_E_floor;
  auto &rad_eps_ = pmbp->pradfemn->rad_eps;
  auto &energy_grid_ = pmbp->pradfemn->energy_grid;
  auto &kappa_s_ = pmbp->pradfemn->kappa_s;
  auto vx = pin->GetOrAddReal("radiation-femn", "fluid_velocity_x", 0.87);
  auto shock = pin->GetOrAddBoolean("problem", "shock", false);
  auto steepness_par = pin->GetOrAddReal("problem", "tanh_par", 50.);
  auto lorentz_w = 1. / sqrt(1 - vx * vx);

  if(shock) {
    // start of advection test through velocity jump
    /*
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
  */

    user_bcs = true;
    user_bcs_func = radiationfemn::ApplyBeamSourcesFEMN1D;

    HostArray1D<Real> beam_source_1_vals_h;
    Kokkos::realloc(beam_source_1_vals_h, num_points_);

    Real fnorm = 1.;
    Real en_dens = fnorm;
    Real fx = fnorm;
    Real fy = 0;
    Real fz = 0;
    Real f2 = fx * fx + fy * fy + fz * fz;

    en_dens = Kokkos::fmax(en_dens, rad_E_floor_);
    Real lim = en_dens * en_dens * (1. - rad_eps_);
    if (f2 > lim) {
      Real fac = lim / f2;
      fx = fac * fx;
      fy = fac * fy;
      fz = fac * fz;
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

            //Real velocity = -Kokkos::abs(vx) * Kokkos::tanh(steepness_par * x1);
            Real velocity = vx;
            Real lorentz_factor = 1. / sqrt(1 - velocity * velocity);

            u_mu_(m, 0, k, j, i) = lorentz_factor;
            u_mu_(m, 1, k, j, i) = velocity * lorentz_factor;
            u_mu_(m, 2, k, j, i) = 0.;
            u_mu_(m, 3, k, j, i) = 0.;
    });

    // construct tetrad
    // L^mu_0 = u^mu
  par_for("radiation_femn_tetrad_compute_L_mu_0", DevExeSpace(),
          0, nmb1, 0, 3, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            tetr_mu_muhat0_(m, mu, 0, k, j, i) = u_mu_(m, mu, k, j, i);
          });

  // L^mu_1 = d_x + (d_x.L^mu_0)L^mu_0
  // d_x = (0,1,0,0), d_x.L^mu_0 = g_mu_nu d_x^mu L^nu_0
  par_for("radiation_femn_tetrad_compute_L_mu_1", DevExeSpace(),
          0, nmb1, 0, 3, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            Real g_dd[16];
            adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                 adm.beta_u(m, 0, k, j, i),
                                 adm.beta_u(m, 1, k, j, i),
                                 adm.beta_u(m, 2, k, j, i),
                                 adm.g_dd(m, 0, 0, k, j, i),
                                 adm.g_dd(m, 0, 1, k, j, i),
                                 adm.g_dd(m, 0, 2, k, j, i),
                                 adm.g_dd(m, 1, 1, k, j, i),
                                 adm.g_dd(m, 1, 2, k, j, i),
                                 adm.g_dd(m, 2, 2, k, j, i),
                                 g_dd);

            Real tetr_val = 0;
            for (int nu = 0; nu < 4; nu++) {
              tetr_val += g_dd[4 + nu] * tetr_mu_muhat0_(m, nu, 0, k, j, i);
            }
            tetr_val *= tetr_mu_muhat0_(m, mu, 0, k, j, i);
            tetr_val += static_cast<int>(mu == 1);
            tetr_mu_muhat0_(m, mu, 1, k, j, i) = tetr_val;
          });

  // L^mu_1 = L^mu_1/||L^mu_1||
  // ||L^mu_1|| = sqrt(g_mu_mu L^mu_1 L^nu_1)
  par_for("radiation_femn_tetrad_normalize_L_mu_1", DevExeSpace(),
          0, nmb1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
            Real g_dd[16];
            adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                 adm.beta_u(m, 0, k, j, i),
                                 adm.beta_u(m, 1, k, j, i),
                                 adm.beta_u(m, 2, k, j, i),
                                 adm.g_dd(m, 0, 0, k, j, i),
                                 adm.g_dd(m, 0, 1, k, j, i),
                                 adm.g_dd(m, 0, 2, k, j, i),
                                 adm.g_dd(m, 1, 1, k, j, i),
                                 adm.g_dd(m, 1, 2, k, j, i),
                                 adm.g_dd(m, 2, 2, k, j, i),
                                 g_dd);

            Real tetr_mu_1_norm = 0.;
            for (int munu = 0; munu < 16; munu++) {
              const int mu = static_cast<int>(munu / 4);
              const int nu = munu - 4 * mu;
              tetr_mu_1_norm += g_dd[munu] * tetr_mu_muhat0_(m, mu, 1, k, j, i)
                  * tetr_mu_muhat0_(m, nu, 1, k, j, i);
            }
            tetr_mu_1_norm = Kokkos::sqrt(tetr_mu_1_norm);

            for (int nu = 0; nu < 4; nu++) {
              tetr_mu_muhat0_(m, nu, 1, k, j, i) =
                  tetr_mu_muhat0_(m, nu, 1, k, j, i) / tetr_mu_1_norm;
            }
          });

    // compute fluxes in the tetrad frame
    Real fx_tetrad = tetr_mu_muhat0_(0, 1, 1, 0, 0, 2) * fx;
    Real fy_tetrad = 0;
    Real fz_tetrad = 0;

    beam_source_1_vals_h(0) = (1. / Kokkos::sqrt(4. * M_PI)) * en_dens;
    beam_source_1_vals_h(1) = -Kokkos::sqrt(3. / (4. * M_PI)) * fy_tetrad;
    beam_source_1_vals_h(2) = Kokkos::sqrt(3. / (4. * M_PI)) * fz_tetrad;
    beam_source_1_vals_h(3) = -Kokkos::sqrt(3. / (4. * M_PI)) * fx_tetrad;

    Kokkos::deep_copy(beam_source_1_vals_, beam_source_1_vals_h);
    // end of advection test through velocity jump
  } else {
    // start of diffusion test with erf initial data
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
    // end of diffusion test with erf initial data
  }
}