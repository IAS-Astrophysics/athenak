//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

    TaskStatus RadiationFEMN::CalculateFluxes(Driver *pdriver, int stage) {
        RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nang1 = num_points - 1;
        int nmb1 = pmy_pack->nmb_thispack - 1;

        bool &multi_d = pmy_pack->pmesh->multi_d;
        bool &three_d = pmy_pack->pmesh->three_d;

        /*
        auto &mm_ = mass_matrix;
        auto &stildex_ = stilde_matrix_x;
        auto &stildey_ = stilde_matrix_y;
        auto &stildez_ = stilde_matrix_z;
        auto &stildemodx_ = stildemod_matrix_x;
        auto &stildemody_ = stildemod_matrix_y;
        auto &stildemodz_ = stildemod_matrix_z;

        auto &i0_ = i0;

        //--------------------------------------------------------------------------------------
        // i-direction

        auto &flx1 = iflx.x1f;
        Kokkos::deep_copy(flx1, 0.);
        par_for("radiation_femn_flux_x", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, int(ie / 2) + 1, 0, nang1, 0,
                nang1,
                KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int A, const int B) {

                    auto kk = k;
                    auto jj = j;
                    auto ii = 2 * i - 2;

                    auto Favg = (0.5) *
                                (stildex_(A, B) * i0_(m, B, kk, jj, ii) + stildex_(A, B) * i0_(m, B, kk, jj, ii + 1));
                    auto Fminus = (0.5) * (stildex_(A, B) *
                                           ((1.5) * i0_(m, B, kk, jj, ii) - (0.5) * i0_(m, B, kk, jj, ii + 1) +
                                            (1.5) * i0_(m, B, kk, jj, ii - 1) - (0.5) * i0_(m, B, kk, jj, ii - 2)) \
 + stildemodx_(A, B) *
   ((1.5) * i0_(m, B, kk, jj, ii - 1) - (0.5) * i0_(m, B, kk, jj, ii - 2) - (1.5) * i0_(m, B, kk, jj, ii) +
    (0.5) * i0_(m, B, kk, jj, ii + 1)));
                    auto Fplus = (0.5) * (stildex_(A, B) *
                                          ((1.5) * i0_(m, B, kk, jj, ii + 2) - (0.5) * i0_(m, B, kk, jj, ii + 3) +
                                           (1.5) * i0_(m, B, kk, jj, ii + 1) - (0.5) * i0_(m, B, kk, jj, ii)) \
 + stildemodx_(A, B) *
   ((1.5) * i0_(m, B, kk, jj, ii + 1) - (0.5) * i0_(m, B, kk, jj, ii) - (1.5) * i0_(m, B, kk, jj, ii + 2) +
    (0.5) * i0_(m, B, kk, jj, ii + 3)));

                    flx1(m, A, kk, jj, ii) += ((1.5) * Fminus - Favg - (0.5) * Fplus) / (2.0);
                    flx1(m, A, kk, jj, ii + 1) += ((0.5) * Fminus + Favg - (1.5) * Fplus) / (2.0);

                });

        //--------------------------------------------------------------------------------------
        // j-direction

        auto &flx2 = iflx.x2f;
        Kokkos::deep_copy(flx2, 0.);
        if (multi_d) {
            par_for("radiation_femn_flux_y", DevExeSpace(), 0, nmb1, ks, ke, js, int(je / 2) + 1, is, ie, 0, nang1, 0,
                    nang1,
                    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int A, const int B) {

                        auto kk = k;
                        auto jj = 2 * j - 2;
                        auto ii = i;

                        auto Favg = (0.5) * (stildey_(A, B) * i0_(m, B, kk, jj, ii) +
                                             stildey_(A, B) * i0_(m, B, kk, jj + 1, ii));
                        auto Fminus = (0.5) * (stildey_(A, B) *
                                               ((1.5) * i0_(m, B, kk, jj, ii) - (0.5) * i0_(m, B, kk, jj + 1, ii) +
                                                (1.5) * i0_(m, B, kk, jj - 1, ii) - (0.5) * i0_(m, B, kk, jj - 2, ii)) \
 + stildemody_(A, B) *
   ((1.5) * i0_(m, B, kk, jj - 1, ii) - (0.5) * i0_(m, B, kk, jj - 2, ii) - (1.5) * i0_(m, B, kk, jj, ii) +
    (0.5) * i0_(m, B, kk, jj + 1, ii)));
                        auto Fplus = (0.5) * (stildey_(A, B) *
                                              ((1.5) * i0_(m, B, kk, jj + 2, ii) - (0.5) * i0_(m, B, kk, jj + 3, ii) +
                                               (1.5) * i0_(m, B, kk, jj + 1, ii) - (0.5) * i0_(m, B, kk, jj, ii)) \
 + stildemody_(A, B) *
   ((1.5) * i0_(m, B, kk, jj + 1, ii) - (0.5) * i0_(m, B, kk, jj, ii) - (1.5) * i0_(m, B, kk, jj + 2, ii) +
    (0.5) * i0_(m, B, kk, jj + 3, ii)));

                        flx2(m, A, kk, jj, ii) += ((1.5) * Fminus - Favg - (0.5) * Fplus) / (2.0);
                        flx2(m, A, kk, jj + 1, ii) += ((0.5) * Fminus + Favg - (1.5) * Fplus) / (2.0);

                    });
        }

        //--------------------------------------------------------------------------------------
        // k-direction

        auto &flx3 = iflx.x3f;
        Kokkos::deep_copy(flx3, 0.);
        if (three_d) {
            par_for("radiation_femn_flux_z", DevExeSpace(), 0, nmb1, ks, int(ke / 2) + 1, js, je, is, ie, 0, nang1, 0,
                    nang1,
                    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int A, const int B) {

                        auto kk = 2 * k - 1;
                        auto jj = j;
                        auto ii = i;

                        auto Favg = (0.5) * (stildez_(A, B) * i0_(m, B, kk, jj, ii) +
                                             stildez_(A, B) * i0_(m, B, kk + 1, jj, ii));
                        auto Fminus = (0.5) * (stildez_(A, B) *
                                               ((1.5) * i0_(m, B, kk, jj, ii) - (0.5) * i0_(m, B, kk + 1, jj, ii) +
                                                (1.5) * i0_(m, B, kk - 1, jj, ii) - (0.5) * i0_(m, B, kk - 2, jj, ii)) \
 + stildemodz_(A, B) *
   ((1.5) * i0_(m, B, kk - 1, jj, ii) - (0.5) * i0_(m, B, kk - 2, jj, ii) - (1.5) * i0_(m, B, kk, jj, ii) +
    (0.5) * i0_(m, B, kk + 1, jj, ii)));
                        auto Fplus = (0.5) * (stildez_(A, B) *
                                              ((1.5) * i0_(m, B, kk + 2, jj, ii) - (0.5) * i0_(m, B, kk + 3, jj, ii) +
                                               (1.5) * i0_(m, B, kk + 1, jj, ii) - (0.5) * i0_(m, B, kk, jj, ii)) \
 + stildemodz_(A, B) *
   ((1.5) * i0_(m, B, kk + 1, jj, ii) - (0.5) * i0_(m, B, kk, jj, ii) - (1.5) * i0_(m, B, kk + 2, jj, ii) +
    (0.5) * i0_(m, B, kk + 3, jj, ii)));

                        flx3(m, A, kk, jj, ii) += ((1.5) * Fminus - Favg - (0.5) * Fplus) / (2.0);
                        flx3(m, A, kk + 1, jj, ii) += ((0.5) * Fminus + Favg - (1.5) * Fplus) / (2.0);

                    });
        }
    */
        return TaskStatus::complete;
    }

} // namespace radiationfemn