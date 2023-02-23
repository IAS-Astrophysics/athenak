//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_source.cpp

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {
    TaskStatus RadiationFEMN::AddRadiationSourceTerm(Driver *pdriver, int stage) {
        // Return if radiation source term disabled
        if (!(rad_source)) {
            return TaskStatus::complete;
        }
        return TaskStatus::complete;
    }

    void RadiationFEMN::AddBeamSource(DvceArray5D<Real> &i0) {
        /*!
         * \brief Populates the relevant ghost points with a value of 1.
         *
         * A beam_mask (5D array) is required to populate for sources. This is defined in pgen.
         *
         */

        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nmb1 = (pmy_pack->nmb_thispack - 1);
        int nang1 = (pmy_pack->pradfemn->num_points - 1);

        auto &beam_mask_ = pmy_pack->pradfemn->beam_mask;

        par_for("radiation_femn_beam_source", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
                    if (beam_mask_(m, n, k, j, i)) {

                        i0(m, n, k, j, i) = 1.0;

                    }
                });
        return;
    }
}