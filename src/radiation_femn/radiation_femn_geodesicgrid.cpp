//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief implementation of the radiation FEM_N class constructor and other functions

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

    void RadiationFEMN::CartesianToSpherical() {
        if (!fpn) {
            for (size_t i = 0; i < num_points; i++) {
                r(i) = sqrt(x(i) * x(i) + y(i) * y(i) + z(i) * z(i));
                theta(i) = acos(z(i) / r(i));
                phi(i) = atan2(y(i), x(i));
            }
        }
    }
}