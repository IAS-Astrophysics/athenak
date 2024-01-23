//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_newdt.cpp
//! \brief function to compute rad timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>
#include <float.h>

#include <limits>
#include <iostream>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

//----------------------------------------------------------------------------------------
// \!fn void RadiationFEMN::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for radiation problems.
//        Only computed once at beginning of calculation.

    TaskStatus RadiationFEMN::NewTimeStep(Driver *pdriver, int stage) {
        //auto &indcs = pmy_pack->pmesh->mb_indcs;
        //int &is = indcs.is, &nx1 = indcs.nx1;
        //int &js = indcs.js, &nx2 = indcs.nx2;
        //int &ks = indcs.ks, &nx3 = indcs.nx3;

        Real dt1 = std::numeric_limits<float>::max();
        Real dt2 = std::numeric_limits<float>::max();
        Real dt3 = std::numeric_limits<float>::max();

        // capture class variables for kernel
        auto &mbsize = pmy_pack->pmb->mb_size;
        int nmb1 = pmy_pack->nmb_thispack;

        Kokkos::parallel_reduce("RadFEMNDt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb1),
                                KOKKOS_LAMBDA(const int &m, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
                                    min_dt1 = fmin(mbsize.d_view(m).dx1, min_dt1);
                                    min_dt2 = fmin(mbsize.d_view(m).dx2, min_dt2);
                                    min_dt3 = fmin(mbsize.d_view(m).dx3, min_dt3);
                                }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));

        // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
        dtnew = dt1;
        if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
        if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

        return TaskStatus::complete;
    }
} // namespace radiationfemn
