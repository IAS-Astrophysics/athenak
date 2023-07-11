//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_tetrad.cpp
//  \brief construct tetrad for radiation FEM_N

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {
void RadiationFEMN::TetradInitialize() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  // L^mu_0 = u^mu
  par_for("radiation_femn_tetrad_initialize_L_mu_0", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            L_mu_muhat0(m, mu, 0, k, j, i) = u_mu(m, mu, k, j, i);
          });

  // L^mu_1 = p_x + (d_x.L^mu_0)L^mu_0, p_x = (0,1,0,0), d_x.L^mu_0 = g_mu_nu d_x^mu L^nu_0
  par_for("radiation_femn_tetrad_initialize_L_mu_1", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            L_mu_muhat0(m, mu, 1, k, j, i) = (mu == 1) + (g_dd(m, 1, 0, k, j, i) * L_mu_muhat0(m, 0, 0, k, j, i) + g_dd(m, 1, 1, k, j, i) * L_mu_muhat0(m, 1, 0, k, j, i)
                + g_dd(m, 1, 2, k, j, i) * L_mu_muhat0(m, 2, 0, k, j, i) + g_dd(m, 1, 3, k, j, i) * L_mu_muhat0(m, 3, 0, k, j, i)) * L_mu_muhat0(m, mu, 0, k, j, i);
          });

  // L^mu_1 = L^mu_1/||L^mu_1||, ||L^mu_1|| = sqrt(g_mu_mu L^mu_1 L^nu_1)
  par_for("radiation_femn_tetrad_normalize_L_mu_1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, 3,
          KOKKOS_LAMBDA(int m, int k, int j, int i, int mu) {
            Real L_mu_1_norm = sqrt(g_dd(m, 0, 0, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i)
                                        + 2. * g_dd(m, 0, 1, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i)
                                        + 2. * g_dd(m, 0, 2, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i)
                                        + 2. * g_dd(m, 0, 3, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i)
                                        + g_dd(m, 1, 1, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i)
                                        + 2. * g_dd(m, 1, 2, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i)
                                        + 2. * g_dd(m, 1, 3, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i)
                                        + g_dd(m, 2, 2, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i)
                                        + 2. * g_dd(m, 2, 3, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i)
                                        + g_dd(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i));

            L_mu_muhat0(m, mu, 1, k, j, i) = L_mu_muhat0(m, mu, 1, k, j, i) / L_mu_1_norm;
          });

  // L^mu_2 = p_y - (d_y.L^mu_1)L^mu_1 + (d_x.L^mu_0)L^mu_0, d_x = (0,1,0,0), d_y = (0,0,1,0)
  par_for("radiation_femn_tetrad_initialize_L_mu_2", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            L_mu_muhat0(m, mu, 2, k, j, i) = (mu == 2) - (g_dd(m, 2, 0, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i) + g_dd(m, 2, 1, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i)
                + g_dd(m, 2, 2, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i) + g_dd(m, 2, 3, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i)) * L_mu_muhat0(m, mu, 1, k, j, i)
                + (g_dd(m, 2, 0, k, j, i) * L_mu_muhat0(m, 0, 0, k, j, i) + g_dd(m, 2, 1, k, j, i) * L_mu_muhat0(m, 1, 0, k, j, i)
                    + g_dd(m, 2, 2, k, j, i) * L_mu_muhat0(m, 2, 0, k, j, i) + g_dd(m, 2, 3, k, j, i) * L_mu_muhat0(m, 3, 0, k, j, i)) * L_mu_muhat0(m, mu, 0, k, j, i);
          });

  // L^mu_2 = L^mu_2/||L^mu_2||, ||L^mu_2|| = sqrt(g_mu_mu L^mu_2 L^nu_2)
  par_for("radiation_femn_tetrad_normalize_L_mu_2", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, 3,
          KOKKOS_LAMBDA(int m, int k, int j, int i, int mu) {
            Real L_mu_2_norm = sqrt(g_dd(m, 0, 0, k, j, i) * L_mu_muhat0(m, 0, 2, k, j, i) * L_mu_muhat0(m, 0, 2, k, j, i)
                                        + 2. * g_dd(m, 0, 1, k, j, i) * L_mu_muhat0(m, 0, 2, k, j, i) * L_mu_muhat0(m, 1, 2, k, j, i)
                                        + 2. * g_dd(m, 0, 2, k, j, i) * L_mu_muhat0(m, 0, 2, k, j, i) * L_mu_muhat0(m, 2, 2, k, j, i)
                                        + 2. * g_dd(m, 0, 3, k, j, i) * L_mu_muhat0(m, 0, 2, k, j, i) * L_mu_muhat0(m, 3, 2, k, j, i)
                                        + g_dd(m, 1, 1, k, j, i) * L_mu_muhat0(m, 1, 2, k, j, i) * L_mu_muhat0(m, 1, 2, k, j, i)
                                        + 2. * g_dd(m, 1, 2, k, j, i) * L_mu_muhat0(m, 1, 2, k, j, i) * L_mu_muhat0(m, 2, 2, k, j, i)
                                        + 2. * g_dd(m, 1, 3, k, j, i) * L_mu_muhat0(m, 1, 2, k, j, i) * L_mu_muhat0(m, 3, 2, k, j, i)
                                        + g_dd(m, 2, 2, k, j, i) * L_mu_muhat0(m, 2, 2, k, j, i) * L_mu_muhat0(m, 2, 2, k, j, i)
                                        + 2. * g_dd(m, 2, 3, k, j, i) * L_mu_muhat0(m, 2, 2, k, j, i) * L_mu_muhat0(m, 3, 2, k, j, i)
                                        + g_dd(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 2, k, j, i) * L_mu_muhat0(m, 3, 2, k, j, i));

            L_mu_muhat0(m, mu, 2, k, j, i) = L_mu_muhat0(m, mu, 2, k, j, i) / L_mu_2_norm;
          });

  // L^mu_3 = p_y - (d_z.L^mu_1) L^mu_1 - (d_z.L^mu_2) L^mu_2 + (d_x.L^mu_0) L^mu_0, d_x = (0,1,0,0), d_y = (0,0,1,0), d_z = (0,0,0,1)
  par_for("radiation_femn_tetrad_initialize_L_mu_3", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            L_mu_muhat0(m, mu, 3, k, j, i) = (mu == 3) - (g_dd(m, 3, 0, k, j, i) * L_mu_muhat0(m, 0, 1, k, j, i) + g_dd(m, 3, 1, k, j, i) * L_mu_muhat0(m, 1, 1, k, j, i)
                + g_dd(m, 3, 2, k, j, i) * L_mu_muhat0(m, 2, 1, k, j, i) + g_dd(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 1, k, j, i)) * L_mu_muhat0(m, mu, 1, k, j, i)
                - (g_dd(m, 3, 0, k, j, i) * L_mu_muhat0(m, 0, 2, k, j, i) + g_dd(m, 3, 1, k, j, i) * L_mu_muhat0(m, 1, 2, k, j, i)
                    + g_dd(m, 3, 2, k, j, i) * L_mu_muhat0(m, 2, 2, k, j, i) + g_dd(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 2, k, j, i)) * L_mu_muhat0(m, mu, 2, k, j, i)
                + (g_dd(m, 3, 0, k, j, i) * L_mu_muhat0(m, 0, 0, k, j, i) + g_dd(m, 3, 1, k, j, i) * L_mu_muhat0(m, 1, 0, k, j, i)
                    + g_dd(m, 3, 2, k, j, i) * L_mu_muhat0(m, 2, 0, k, j, i) + g_dd(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 0, k, j, i)) * L_mu_muhat0(m, mu, 0, k, j, i);
          });

  // L^mu_3 = L^mu_3/||L^mu_3||, ||L^mu_3|| = sqrt(g_mu_mu L^mu_3 L^nu_3)
  par_for("radiation_femn_tetrad_normalize_L_mu_3", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, 3,
          KOKKOS_LAMBDA(int m, int k, int j, int i, int mu) {
            Real L_mu_3_norm = sqrt(g_dd(m, 0, 0, k, j, i) * L_mu_muhat0(m, 0, 3, k, j, i) * L_mu_muhat0(m, 0, 3, k, j, i)
                                        + 2. * g_dd(m, 0, 1, k, j, i) * L_mu_muhat0(m, 0, 3, k, j, i) * L_mu_muhat0(m, 1, 3, k, j, i)
                                        + 2. * g_dd(m, 0, 2, k, j, i) * L_mu_muhat0(m, 0, 3, k, j, i) * L_mu_muhat0(m, 2, 3, k, j, i)
                                        + 2. * g_dd(m, 0, 3, k, j, i) * L_mu_muhat0(m, 0, 3, k, j, i) * L_mu_muhat0(m, 3, 3, k, j, i)
                                        + g_dd(m, 1, 1, k, j, i) * L_mu_muhat0(m, 1, 3, k, j, i) * L_mu_muhat0(m, 1, 3, k, j, i)
                                        + 2. * g_dd(m, 1, 2, k, j, i) * L_mu_muhat0(m, 1, 3, k, j, i) * L_mu_muhat0(m, 2, 3, k, j, i)
                                        + 2. * g_dd(m, 1, 3, k, j, i) * L_mu_muhat0(m, 1, 3, k, j, i) * L_mu_muhat0(m, 3, 3, k, j, i)
                                        + g_dd(m, 2, 2, k, j, i) * L_mu_muhat0(m, 2, 3, k, j, i) * L_mu_muhat0(m, 2, 3, k, j, i)
                                        + 2. * g_dd(m, 2, 3, k, j, i) * L_mu_muhat0(m, 2, 3, k, j, i) * L_mu_muhat0(m, 3, 3, k, j, i)
                                        + g_dd(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 3, k, j, i) * L_mu_muhat0(m, 3, 3, k, j, i));

            L_mu_muhat0(m, mu, 3, k, j, i) = L_mu_muhat0(m, mu, 3, k, j, i) / L_mu_3_norm;
          });

}
}  // namespace radiationfemn