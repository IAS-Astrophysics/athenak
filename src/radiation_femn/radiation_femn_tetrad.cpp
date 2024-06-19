//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_tetrad.cpp
//  \brief construct an orthogonal tetrad in the fluid frame

#include <math.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "adm/adm.hpp"

namespace radiationfemn {
TaskStatus RadiationFEMN::TetradOrthogonalize(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto tetr_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  auto u_mu_ = pmy_pack->pradfemn->u_mu;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

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

  // L^mu_2 = p_y - (d_y.L^mu_1)L^mu_1 + (d_x.L^mu_0)L^mu_0
  // d_x = (0,1,0,0), d_y = (0,0,1,0)
  par_for("radiation_femn_tetrad_compute_L_mu_2", DevExeSpace(),
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
              tetr_val -= g_dd[8 + nu] * tetr_mu_muhat0_(m, nu, 1, k, j, i);
            }
            tetr_val *= tetr_mu_muhat0_(m, mu, 1, k, j, i);
            Real tetr_val_2 = 0;
            for (int nu = 0; nu < 4; nu++) {
              tetr_val_2 += g_dd[8 + nu] * tetr_mu_muhat0_(m, nu, 0, k, j, i);
            }
            tetr_val_2 *= tetr_mu_muhat0_(m, mu, 0, k, j, i);
            tetr_mu_muhat0_(m, mu, 2, k, j, i) =
                static_cast<int>(mu == 2) + tetr_val + tetr_val_2;
          });

  // L^mu_2 = L^mu_2/||L^mu_2||
  // ||L^mu_2|| = sqrt(g_mu_mu L^mu_2 L^nu_2)
  par_for("radiation_femn_tetrad_normalize_L_mu_2", DevExeSpace(),
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

            Real tetr_mu_2_norm = 0.;
            for (int munu = 0; munu < 16; munu++) {
              int mu = static_cast<int>(munu / 4);
              int nu = munu - 4 * mu;
              tetr_mu_2_norm += g_dd[munu] * tetr_mu_muhat0_(m, mu, 2, k, j, i)
                  * tetr_mu_muhat0_(m, nu, 2, k, j, i);
            }
            tetr_mu_2_norm = Kokkos::sqrt(tetr_mu_2_norm);

            for (int nu = 0; nu < 4; nu++) {
              tetr_mu_muhat0_(m, nu, 2, k, j, i) =
                  tetr_mu_muhat0_(m, nu, 2, k, j, i) / tetr_mu_2_norm;
            }
          });

  // L^mu_3 = d_y - (d_z.L^mu_1) L^mu_1 - (d_z.L^mu_2) L^mu_2 + (d_x.L^mu_0) L^mu_0
  // d_x = (0,1,0,0), d_y = (0,0,1,0), d_z = (0,0,0,1)
  par_for("radiation_femn_tetrad_compute_L_mu_3", DevExeSpace(),
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
              tetr_val -= g_dd[12 + nu] * tetr_mu_muhat0_(m, nu, 1, k, j, i);
            }
            tetr_val *= tetr_mu_muhat0_(m, mu, 1, k, j, i);
            Real tetr_val_2 = 0;
            for (int nu = 0; nu < 4; nu++) {
              tetr_val_2 -= g_dd[12 + nu] * tetr_mu_muhat0_(m, nu, 2, k, j, i);
            }
            tetr_val_2 *= tetr_mu_muhat0_(m, mu, 2, k, j, i);
            Real tetr_val_3 = 0;
            for (int nu = 0; nu < 4; nu++) {
              tetr_val_3 += g_dd[12 + nu] * tetr_mu_muhat0_(m, nu, 0, k, j, i);
            }
            tetr_val_3 *= tetr_mu_muhat0_(m, mu, 0, k, j, i);
            tetr_mu_muhat0_(m, mu, 3, k, j, i) =
                static_cast<int>(mu == 3) + tetr_val + tetr_val_2 + tetr_val_3;
          });

  // L^mu_3 = L^mu_3/||L^mu_3||
  // ||L^mu_3|| = sqrt(g_mu_mu L^mu_3 L^nu_3)
  par_for("radiation_femn_tetrad_normalize_L_mu_3", DevExeSpace(),
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

            Real tetr_mu_3_norm = 0.;
            for (int munu = 0; munu < 16; munu++) {
              int mu = static_cast<int>(munu / 4);
              int nu = munu - 4 * mu;
              tetr_mu_3_norm += g_dd[munu] * tetr_mu_muhat0_(m, mu, 3, k, j, i)
                  * tetr_mu_muhat0_(m, nu, 3, k, j, i);
            }
            tetr_mu_3_norm = Kokkos::sqrt(tetr_mu_3_norm);

            for (int nu = 0; nu < 4; nu++) {
              tetr_mu_muhat0_(m, nu, 3, k, j, i) =
                  tetr_mu_muhat0_(m, nu, 3, k, j, i) / tetr_mu_3_norm;
            }
          });

  return TaskStatus::complete;
}
} // namespace radiationfemn
