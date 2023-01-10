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

        return TaskStatus::complete;
    }
} // namespace radiationfemn
