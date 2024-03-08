//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_initialize.cpp
//  \brief this is a temporary file -- it hardcodes fluid velocities

#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn
{
    void RadiationFEMN::InitializeFluidVelocity()
    {
        auto& indices = pmy_pack->pmesh->mb_indcs;
        int &is = indices.is, &ie = indices.ie;
        int &js = indices.js, &je = indices.je;
        int &ks = indices.ks, &ke = indices.ke;

        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto& indcs = pmy_pack->pmesh->mb_indcs;

        auto &u_mu_ = pmy_pack->pradfemn->u_mu;
        /*
        par_for("radiation_femn_dummy_initialize_1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(int m, int k, int j, int i)
                {
                    u_mu_(m, 0, k, j, i) = 1;
                }); */
    }
}
