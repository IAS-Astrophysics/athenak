//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_update.cpp
//  \brief Performs update of Radiation conserved variables (i0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.
//  Explicit (not implicit) radiation source terms are included in this update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

    TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nang1 = nangles - 1;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &mbsize = pmy_pack->pmb->mb_size;

        bool &multi_d = pmy_pack->pmesh->multi_d;
        bool &three_d = pmy_pack->pmesh->three_d;

        Real &gam0 = pdriver->gam0[stage - 1];
        Real &gam1 = pdriver->gam1[stage - 1];
        Real beta_dt = (pdriver->beta[stage - 1]) * (pmy_pack->pmesh->dt);

        auto i0_ = i0;
        auto i1_ = i1;
        auto &flx1 = iflx.x1f;
        auto &flx2 = iflx.x2f;
        auto &flx3 = iflx.x3f;


        par_for("radiation_femn_update", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
                    Real divf_s = (flx1(m, n, k, j, i + 1) - flx1(m, n, k, j, i)) / mbsize.d_view(m).dx1;
                    if (multi_d) {
                        divf_s += (flx2(m, n, k, j + 1, i) - flx2(m, n, k, j, i)) / mbsize.d_view(m).dx2;
                    }
                    if (three_d) {
                        divf_s += (flx3(m, n, k + 1, j, i) - flx3(m, n, k, j, i)) / mbsize.d_view(m).dx3;
                    }
                    i0_(m, n, k, j, i) = gam0 * i0_(m, n, k, j, i) + gam1 * i1_(m, n, k, j, i) - beta_dt * divf_s;
                });

        // TODO: add explicit source terms
        //if (psrc->source_terms_enabled) {
        //    if (psrc->beam) psrc->AddBeamSource(i0_, beta_dt);
        //}

        return TaskStatus::complete;
    }
} // namespace radiationfemn