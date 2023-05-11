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
    void RadiationFEMN::TetradInitialize() {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &mbsize = pmy_pack->pmb->mb_size;

        par_for("radiation_femn_tetrad_initialize_Lmu_0_L_mu_1", DevExeSpace(), 0, nmb1, 0, 4, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
                    L_mu_muhat0(m, mu, 0, k, j, i) = u_mu(m, mu, k, j, i);
                    L_mu_muhat0(m, mu, 1, k, j, i) = (m == 1) + u_mu(m, 1, k, j, i) * L_mu_muhat0(m, mu, 0, k, j, i);
                });

        par_for("radiation_femn_tetrad_normalize_L_mu_1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, 4,
                KOKKOS_LAMBDA(int m, int k, int j, int i, int mu) {
                    Real L_mu_1_norm = sqrt(pow(L_mu_muhat0(m, 0, 1, k, j, i), 2) + pow(L_mu_muhat0(m, 1, 1, k, j, i), 2) +
                                        pow(L_mu_muhat0(m, 2, 1, k, j, i), 2) + pow(L_mu_muhat0(m, 3, 1, k, j, i), 2));
                    L_mu_muhat0(m, mu, 1, k, j, i) = L_mu_muhat0(m, mu, 1, k, j, i)/L_mu_1_norm;
            });

        par_for("radiation_femn_tetrad_initialize_L_mu_2", DevExeSpace(), 0, nmb1, 0, 4, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
                    L_mu_muhat0(m, mu, 2, k, j, i) = (m == 2) + u_mu(m, 1, k, j, i) * L_mu_muhat0(m, mu, 0, k, j, i) - u_mu(m, 2, k, j, i) * L_mu_muhat0(m, mu, 1, k, j, i);
                });

        par_for("radiation_femn_tetrad_normalize_L_mu_2", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, 4,
                KOKKOS_LAMBDA(int m, int k, int j, int i, int mu) {
                    Real L_mu_2_norm = sqrt(pow(L_mu_muhat0(m, 0, 2, k, j, i), 2) + pow(L_mu_muhat0(m, 1, 2, k, j, i), 2) +
                                        pow(L_mu_muhat0(m, 2, 2, k, j, i), 2) + pow(L_mu_muhat0(m, 3, 2, k, j, i), 2));
                    L_mu_muhat0(m, mu, 2, k, j, i) = L_mu_muhat0(m, mu, 2, k, j, i)/L_mu_2_norm;
            });

        par_for("radiation_femn_tetrad_initialize_L_mu_3", DevExeSpace(), 0, nmb1, 0, 4, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
                    L_mu_muhat0(m, mu, 3, k, j, i) = (m == 3) + u_mu(m, 1, k, j, i) * L_mu_muhat0(m, mu, 0, k, j, i) - u_mu(m, 2, k, j, i) * L_mu_muhat0(m, mu, 1, k, j, i) - - u_mu(m, 3, k, j, i) * L_mu_muhat0(m, mu, 2, k, j, i);
                });

        par_for("radiation_femn_tetrad_normalize_L_mu_3", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, 4,
                KOKKOS_LAMBDA(int m, int k, int j, int i, int mu) {
                    Real L_mu_3_norm = sqrt(pow(L_mu_muhat0(m, 0, 3, k, j, i), 2) + pow(L_mu_muhat0(m, 1, 3, k, j, i), 2) +
                                        pow(L_mu_muhat0(m, 3, 1, k, j, i), 2) + pow(L_mu_muhat0(m, 3, 3, k, j, i), 2));
                    L_mu_muhat0(m, mu, 3, k, j, i) = L_mu_muhat0(m, mu, 3, k, j, i)/L_mu_3_norm;
            });
    }
}  // namespace radiationfemn