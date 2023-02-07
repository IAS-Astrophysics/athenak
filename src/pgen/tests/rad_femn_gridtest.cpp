//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_gridtest.cpp
//! \brief tests the geodesic grid and associated matrices for radiation FEM_N

// C++ headers
#include <iostream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"

void ProblemGenerator::RadiationFEMNGridtest(ParameterInput *pin, const bool restart) {
    if (restart) return;

    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

    if (pmbp->pradfemn == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "The radiation FEM_N grid test can only be run with radiation-femn, but no "
                  << "<radiation-femn> block in input file" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Coordinates of angles in cartesian and polar:" << std::endl;
    std::cout << "x y z r theta phi" << std::endl;
    for (size_t i = 0; i < pmbp->pradfemn->num_points; i++) {
        std::cout << pmbp->pradfemn->x(i) << " " << pmbp->pradfemn->y(i) << " " << pmbp->pradfemn->z(i) << " "
                  << pmbp->pradfemn->r(i) << " " << pmbp->pradfemn->theta(i) << " " << pmbp->pradfemn->phi(i)
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Triangle information:" << std::endl;
    for (size_t i = 0; i < pmbp->pradfemn->num_triangles; i++) {
        std::cout << "[ " << pmbp->pradfemn->triangles(i, 0) << ", " << pmbp->pradfemn->triangles(i, 1) << ", "
                  << pmbp->pradfemn->triangles(i, 2) << " ]" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Edge information:" << std::endl;
    for (size_t i = 0; i < pmbp->pradfemn->num_edges; i++) {
        std::cout << "[ " << pmbp->pradfemn->edges(i, 0) << ", " << pmbp->pradfemn->edges(i, 1) << " ]" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "End of information " << std::endl;

    return;
}