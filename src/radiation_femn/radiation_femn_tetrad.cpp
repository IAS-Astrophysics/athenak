//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_tetrad.cpp
//  \brief compute tetrad quantities

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {
    TaskStatus RadiationFEMN::ComputeTetrad(Driver *d, int stage) {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &mbsize = pmy_pack->pmb->mb_size;

        par_for("radiation_femn_compute_tetrad", DevExeSpace(), 0, nmb1, 0, 4, 0, 4, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int mu, int muhat, int k, int j, int i) {
                    double v1 = u_mu(m, 1, k, j, i) / Lambda(m, k, j, i) - n_mu(m, 1, k, j, i);
                    double v2 = u_mu(m, 2, k, j, i) / Lambda(m, k, j, i) - n_mu(m, 2, k, j, i);
                    double v3 = u_mu(m, 3, k, j, i) / Lambda(m, k, j, i) - n_mu(m, 3, k, j, i);

                    double v_d1 = g_dd(m, 1, 1, k, j, i) * v1 + g_dd(m, 1, 2, k, j, i) * v2 + g_dd(m, 1, 3, k, j, i) * v3;
                    double v_d2 = g_dd(m, 2, 1, k, j, i) * v1 + g_dd(m, 2, 2, k, j, i) * v2 + g_dd(m, 2, 3, k, j, i) * v3;;
                    double v_d3 = g_dd(m, 3, 1, k, j, i) * v1 + g_dd(m, 3, 2, k, j, i) * v2 + g_dd(m, 3, 3, k, j, i) * v3;;

                    double c1 = 0.;
                    double c2 = 0.;
                    double c3 = 1. / sqrt(g_dd(m, 3, 3, k, j, i) - v_d3 * v_d3);

                    double b1 = 0.;
                    double b2 = -(sqrt(g_dd(m, 3, 3, k, j, i) - v_d3 * v_d3)) /
                                sqrt(-g_dd(m, 2, 3, k, j, i) * g_dd(m, 2, 3, k, j, i) - g_dd(m, 3, 3, k, j, i) * v_d2 * v_d2 + 2 * g_dd(m, 2, 3, k, j, i) * v_d2 * v_d3 +
                                     g_dd(m, 2, 2, k, j, i) * (g_dd(m, 3, 3, k, j, i) - v_d3 * v_d3));
                    double b3 = (g_dd(m, 2, 3, k, j, i) - v_d2 * v_d3) / (sqrt(g_dd(m, 3, 3, k, j, i) - v_d3 * v_d3) *
                                                                          sqrt(-g_dd(m, 2, 3, k, j, i) * g_dd(m, 2, 3, k, j, i) - g_dd(m, 3, 3, k, j, i) * v_d2 * v_d2 +
                                                                               2 * g_dd(m, 2, 3, k, j, i) * v_d2 * v_d3 +
                                                                               g_dd(m, 2, 2, k, j, i) * (g_dd(m, 3, 3, k, j, i) - v_d3 * v_d3)));

                    double a1 = sqrt(pow(g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) - g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) +
                                         g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) - (g_dd(m, 2, 3, k, j, i) + g_dd(m, 3, 2, k, j, i)) * v_d2 * v_d3 +
                                         g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2), 2) /
                                     (-(pow(g_dd(m, 1, 2, k, j, i), 2) * pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i)) +
                                      2 * pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) -
                                      pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) -
                                      pow(g_dd(m, 2, 3, k, j, i), 2) * pow(g_dd(m, 3, 2, k, j, i), 2) * pow(v_d1, 2) +
                                      2 * g_dd(m, 2, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) * pow(v_d1, 2) -
                                      pow(g_dd(m, 2, 2, k, j, i), 2) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d1, 2) +
                                      2 * g_dd(m, 1, 2, k, j, i) * pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 -
                                      4 * g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 +
                                      2 * g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * v_d1 * v_d2 +
                                      pow(g_dd(m, 1, 2, k, j, i), 2) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d2, 2) -
                                      pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i) * pow(v_d1, 2) * pow(v_d2, 2) +
                                      g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d1, 2) * pow(v_d2, 2) -
                                      2 * g_dd(m, 1, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * v_d1 * pow(v_d2, 3) + 2 * (-(pow(g_dd(m, 1, 2, k, j, i), 2) *
                                                                                                                                 g_dd(m, 3, 2, k, j, i) *
                                                                                                                                 g_dd(m, 3, 3, k, j, i) * v_d2) +
                                                                                                                               g_dd(m, 3, 2, k, j, i) *
                                                                                                                               (pow(g_dd(m, 2, 3, k, j, i), 2) -
                                                                                                                                g_dd(m, 2, 2, k, j, i) *
                                                                                                                                g_dd(m, 3, 3, k, j, i)) * pow(v_d1, 2) *
                                                                                                                               v_d2 +
                                                                                                                               g_dd(m, 1, 2, k, j, i) * v_d1 *
                                                                                                                               (g_dd(m, 2, 3, k, j, i) *
                                                                                                                                (pow(g_dd(m, 3, 2, k, j, i), 2) -
                                                                                                                                 g_dd(m, 2, 2, k, j, i) *
                                                                                                                                 g_dd(m, 3, 3, k, j, i)) +
                                                                                                                                (g_dd(m, 2, 3, k, j, i) +
                                                                                                                                 2 * g_dd(m, 3, 2, k, j, i)) *
                                                                                                                                g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2))) *
                                                                                                                          v_d3 +
                                      (g_dd(m, 2, 2, k, j, i) * (-2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) + pow(g_dd(m, 3, 2, k, j, i), 2) +
                                                                 g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) * pow(v_d1, 2) - 2 * g_dd(m, 1, 2, k, j, i) *
                                                                                                                                   (pow(g_dd(m, 2, 3, k, j, i), 2) +
                                                                                                                                    pow(g_dd(m, 3, 2, k, j, i), 2) +
                                                                                                                                    g_dd(m, 2, 2, k, j, i) *
                                                                                                                                    g_dd(m, 3, 3, k, j, i)) * v_d1 *
                                                                                                                                   v_d2 +
                                       pow(g_dd(m, 1, 2, k, j, i), 2) * (pow(g_dd(m, 2, 3, k, j, i), 2) - 2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) +
                                                                         2 * g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) - g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2))) *
                                      pow(v_d3, 2) +
                                      2 * g_dd(m, 1, 2, k, j, i) *
                                      (g_dd(m, 2, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * v_d1 + g_dd(m, 1, 2, k, j, i) * g_dd(m, 3, 2, k, j, i) * v_d2) * pow(v_d3, 3) -
                                      pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 2, k, j, i) * pow(v_d3, 4) +
                                      pow(g_dd(m, 1, 3, k, j, i), 2) * (g_dd(m, 2, 2, k, j, i) - pow(v_d2, 2)) *
                                      (-2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) + pow(g_dd(m, 3, 2, k, j, i), 2) +
                                       g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) - g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) +
                                       2 * g_dd(m, 2, 3, k, j, i) * v_d2 * v_d3 -
                                       g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2)) + g_dd(m, 1, 1, k, j, i) * pow(g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) -
                                                                                                             g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) +
                                                                                                             g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) -
                                                                                                             (g_dd(m, 2, 3, k, j, i) + g_dd(m, 3, 2, k, j, i)) * v_d2 *
                                                                                                             v_d3 + g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2), 2) +
                                      2 * g_dd(m, 1, 3, k, j, i) * (g_dd(m, 2, 3, k, j, i) - g_dd(m, 3, 2, k, j, i)) * (g_dd(m, 3, 2, k, j, i) - v_d2 * v_d3) *
                                      (-(g_dd(m, 2, 3, k, j, i) * v_d1 * v_d2) + g_dd(m, 2, 2, k, j, i) * v_d1 * v_d3 +
                                       g_dd(m, 1, 2, k, j, i) * (g_dd(m, 2, 3, k, j, i) - v_d2 * v_d3))));
                    double a2 = sqrt(
                            pow(g_dd(m, 1, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) - g_dd(m, 1, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) + g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 -
                                (g_dd(m, 3, 2, k, j, i) * v_d1 + g_dd(m, 1, 3, k, j, i) * v_d2) * v_d3 + g_dd(m, 1, 2, k, j, i) * pow(v_d3, 2), 2) /
                            (-(pow(g_dd(m, 1, 2, k, j, i), 2) * pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i)) +
                             2 * pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) -
                             pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) -
                             pow(g_dd(m, 2, 3, k, j, i), 2) * pow(g_dd(m, 3, 2, k, j, i), 2) * pow(v_d1, 2) +
                             2 * g_dd(m, 2, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) * pow(v_d1, 2) -
                             pow(g_dd(m, 2, 2, k, j, i), 2) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d1, 2) +
                             2 * g_dd(m, 1, 2, k, j, i) * pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 -
                             4 * g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 +
                             2 * g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * v_d1 * v_d2 +
                             pow(g_dd(m, 1, 2, k, j, i), 2) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d2, 2) -
                             pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i) * pow(v_d1, 2) * pow(v_d2, 2) +
                             g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d1, 2) * pow(v_d2, 2) -
                             2 * g_dd(m, 1, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * v_d1 * pow(v_d2, 3) + 2 * (-(pow(g_dd(m, 1, 2, k, j, i), 2) *
                                                                                                                        g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) *
                                                                                                                        v_d2) + g_dd(m, 3, 2, k, j, i) *
                                                                                                                                (pow(g_dd(m, 2, 3, k, j, i), 2) -
                                                                                                                                 g_dd(m, 2, 2, k, j, i) *
                                                                                                                                 g_dd(m, 3, 3, k, j, i)) * pow(v_d1, 2) *
                                                                                                                                v_d2 +
                                                                                                                      g_dd(m, 1, 2, k, j, i) * v_d1 *
                                                                                                                      (g_dd(m, 2, 3, k, j, i) *
                                                                                                                       (pow(g_dd(m, 3, 2, k, j, i), 2) -
                                                                                                                        g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) +
                                                                                                                       (g_dd(m, 2, 3, k, j, i) +
                                                                                                                        2 * g_dd(m, 3, 2, k, j, i)) *
                                                                                                                       g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2))) * v_d3 +
                             (g_dd(m, 2, 2, k, j, i) *
                              (-2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) + pow(g_dd(m, 3, 2, k, j, i), 2) + g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) *
                              pow(v_d1, 2) - 2 * g_dd(m, 1, 2, k, j, i) *
                                             (pow(g_dd(m, 2, 3, k, j, i), 2) + pow(g_dd(m, 3, 2, k, j, i), 2) + g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) * v_d1 *
                                             v_d2 +
                              pow(g_dd(m, 1, 2, k, j, i), 2) * (pow(g_dd(m, 2, 3, k, j, i), 2) - 2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) +
                                                                2 * g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) - g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2))) *
                             pow(v_d3, 2) +
                             2 * g_dd(m, 1, 2, k, j, i) *
                             (g_dd(m, 2, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * v_d1 + g_dd(m, 1, 2, k, j, i) * g_dd(m, 3, 2, k, j, i) * v_d2) * pow(v_d3, 3) -
                             pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 2, k, j, i) * pow(v_d3, 4) +
                             pow(g_dd(m, 1, 3, k, j, i), 2) * (g_dd(m, 2, 2, k, j, i) - pow(v_d2, 2)) *
                             (-2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) + pow(g_dd(m, 3, 2, k, j, i), 2) + g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) -
                              g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) + 2 * g_dd(m, 2, 3, k, j, i) * v_d2 * v_d3 -
                              g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2)) + g_dd(m, 1, 1, k, j, i) * pow(g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) -
                                                                                                    g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) +
                                                                                                    g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) -
                                                                                                    (g_dd(m, 2, 3, k, j, i) + g_dd(m, 3, 2, k, j, i)) * v_d2 * v_d3 +
                                                                                                    g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2), 2) +
                             2 * g_dd(m, 1, 3, k, j, i) * (g_dd(m, 2, 3, k, j, i) - g_dd(m, 3, 2, k, j, i)) * (g_dd(m, 3, 2, k, j, i) - v_d2 * v_d3) *
                             (-(g_dd(m, 2, 3, k, j, i) * v_d1 * v_d2) + g_dd(m, 2, 2, k, j, i) * v_d1 * v_d3 +
                              g_dd(m, 1, 2, k, j, i) * (g_dd(m, 2, 3, k, j, i) - v_d2 * v_d3))));
                    double a3 = sqrt(
                            pow(g_dd(m, 1, 3, k, j, i) * g_dd(m, 2, 2, k, j, i) - g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) + g_dd(m, 2, 3, k, j, i) * v_d1 * v_d2 -
                                g_dd(m, 1, 3, k, j, i) * pow(v_d2, 2) - g_dd(m, 2, 2, k, j, i) * v_d1 * v_d3 + g_dd(m, 1, 2, k, j, i) * v_d2 * v_d3, 2) /
                            (-(pow(g_dd(m, 1, 2, k, j, i), 2) * pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i)) +
                             2 * pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) -
                             pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) -
                             pow(g_dd(m, 2, 3, k, j, i), 2) * pow(g_dd(m, 3, 2, k, j, i), 2) * pow(v_d1, 2) +
                             2 * g_dd(m, 2, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) * pow(v_d1, 2) -
                             pow(g_dd(m, 2, 2, k, j, i), 2) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d1, 2) +
                             2 * g_dd(m, 1, 2, k, j, i) * pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 -
                             4 * g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) * v_d1 * v_d2 +
                             2 * g_dd(m, 1, 2, k, j, i) * g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * v_d1 * v_d2 +
                             pow(g_dd(m, 1, 2, k, j, i), 2) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d2, 2) -
                             pow(g_dd(m, 2, 3, k, j, i), 2) * g_dd(m, 3, 3, k, j, i) * pow(v_d1, 2) * pow(v_d2, 2) +
                             g_dd(m, 2, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * pow(v_d1, 2) * pow(v_d2, 2) -
                             2 * g_dd(m, 1, 2, k, j, i) * pow(g_dd(m, 3, 3, k, j, i), 2) * v_d1 * pow(v_d2, 3) + 2 * (-(pow(g_dd(m, 1, 2, k, j, i), 2) *
                                                                                                                        g_dd(m, 3, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) *
                                                                                                                        v_d2) + g_dd(m, 3, 2, k, j, i) *
                                                                                                                                (pow(g_dd(m, 2, 3, k, j, i), 2) -
                                                                                                                                 g_dd(m, 2, 2, k, j, i) *
                                                                                                                                 g_dd(m, 3, 3, k, j, i)) * pow(v_d1, 2) *
                                                                                                                                v_d2 +
                                                                                                                      g_dd(m, 1, 2, k, j, i) * v_d1 *
                                                                                                                      (g_dd(m, 2, 3, k, j, i) *
                                                                                                                       (pow(g_dd(m, 3, 2, k, j, i), 2) -
                                                                                                                        g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) +
                                                                                                                       (g_dd(m, 2, 3, k, j, i) +
                                                                                                                        2 * g_dd(m, 3, 2, k, j, i)) *
                                                                                                                       g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2))) * v_d3 +
                             (g_dd(m, 2, 2, k, j, i) *
                              (-2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) + pow(g_dd(m, 3, 2, k, j, i), 2) + g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) *
                              pow(v_d1, 2) - 2 * g_dd(m, 1, 2, k, j, i) *
                                             (pow(g_dd(m, 2, 3, k, j, i), 2) + pow(g_dd(m, 3, 2, k, j, i), 2) + g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i)) * v_d1 *
                                             v_d2 +
                              pow(g_dd(m, 1, 2, k, j, i), 2) * (pow(g_dd(m, 2, 3, k, j, i), 2) - 2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) +
                                                                2 * g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) - g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2))) *
                             pow(v_d3, 2) +
                             2 * g_dd(m, 1, 2, k, j, i) *
                             (g_dd(m, 2, 2, k, j, i) * g_dd(m, 2, 3, k, j, i) * v_d1 + g_dd(m, 1, 2, k, j, i) * g_dd(m, 3, 2, k, j, i) * v_d2) * pow(v_d3, 3) -
                             pow(g_dd(m, 1, 2, k, j, i), 2) * g_dd(m, 2, 2, k, j, i) * pow(v_d3, 4) +
                             pow(g_dd(m, 1, 3, k, j, i), 2) * (g_dd(m, 2, 2, k, j, i) - pow(v_d2, 2)) *
                             (-2 * g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) + pow(g_dd(m, 3, 2, k, j, i), 2) + g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) -
                              g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) + 2 * g_dd(m, 2, 3, k, j, i) * v_d2 * v_d3 -
                              g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2)) + g_dd(m, 1, 1, k, j, i) * pow(g_dd(m, 2, 3, k, j, i) * g_dd(m, 3, 2, k, j, i) -
                                                                                                    g_dd(m, 2, 2, k, j, i) * g_dd(m, 3, 3, k, j, i) +
                                                                                                    g_dd(m, 3, 3, k, j, i) * pow(v_d2, 2) -
                                                                                                    (g_dd(m, 2, 3, k, j, i) + g_dd(m, 3, 2, k, j, i)) * v_d2 * v_d3 +
                                                                                                    g_dd(m, 2, 2, k, j, i) * pow(v_d3, 2), 2) +
                             2 * g_dd(m, 1, 3, k, j, i) * (g_dd(m, 2, 3, k, j, i) - g_dd(m, 3, 2, k, j, i)) * (g_dd(m, 3, 2, k, j, i) - v_d2 * v_d3) *
                             (-(g_dd(m, 2, 3, k, j, i) * v_d1 * v_d2) + g_dd(m, 2, 2, k, j, i) * v_d1 * v_d3 +
                              g_dd(m, 1, 2, k, j, i) * (g_dd(m, 2, 3, k, j, i) - v_d2 * v_d3))));

                    double A = v_d1 * a1 + v_d2 * a2 + v_d3 * a3;
                    double B = v_d1 * b1 + v_d2 * b2 + v_d3 * b3;
                    double C = v_d1 * c1 + v_d2 * c2 + v_d3 * c3;

                    L_mu_muhat(m, 0, 0, k, j, i) = u_mu(m, 0, k, j, i);
                    L_mu_muhat(m, 1, 0, k, j, i) = u_mu(m, 1, k, j, i);
                    L_mu_muhat(m, 2, 0, k, j, i) = u_mu(m, 2, k, j, i);
                    L_mu_muhat(m, 3, 0, k, j, i) = u_mu(m, 3, k, j, i);

                    L_mu_muhat(m, 0, 1, k, j, i) = A * n_mu(m, 0, k, j, i) + 0.;
                    L_mu_muhat(m, 1, 1, k, j, i) = A * n_mu(m, 1, k, j, i) + a1;
                    L_mu_muhat(m, 2, 1, k, j, i) = A * n_mu(m, 2, k, j, i) + a2;
                    L_mu_muhat(m, 3, 1, k, j, i) = A * n_mu(m, 3, k, j, i) + a3;

                    L_mu_muhat(m, 0, 2, k, j, i) = B * n_mu(m, 0, k, j, i) + 0.;
                    L_mu_muhat(m, 1, 2, k, j, i) = B * n_mu(m, 1, k, j, i) + b1;
                    L_mu_muhat(m, 2, 2, k, j, i) = B * n_mu(m, 2, k, j, i) + b2;
                    L_mu_muhat(m, 3, 2, k, j, i) = B * n_mu(m, 3, k, j, i) + b3;

                    L_mu_muhat(m, 0, 3, k, j, i) = C * n_mu(m, 0, k, j, i) + 0.;
                    L_mu_muhat(m, 1, 3, k, j, i) = C * n_mu(m, 1, k, j, i) + c1;
                    L_mu_muhat(m, 2, 3, k, j, i) = C * n_mu(m, 2, k, j, i) + c2;
                    L_mu_muhat(m, 3, 3, k, j, i) = C * n_mu(m, 3, k, j, i) + c3;

                });

        return TaskStatus::complete;
    }
}  // namespace radiationfemn