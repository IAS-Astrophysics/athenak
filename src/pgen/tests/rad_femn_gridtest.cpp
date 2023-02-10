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
#include <fstream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
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

    std::string pathdir = pin->GetString("radiation-femn", "geogrid_savedir");
    std::string filenamepart = "/geogrid_basis_" + std::to_string(pmbp->pradfemn->basis) + "_ref_" +
                               std::to_string(pmbp->pradfemn->num_ref);

    std::ofstream fout(pathdir + filenamepart + "_metadata" + ".txt");
    fout << "Geodesic grid metadata: " << std::endl;
    fout << "basis = " << pmbp->pradfemn->basis << std::endl;
    fout << "num_ref = " << pmbp->pradfemn->num_ref << std::endl;
    fout << "nangles = " << pmbp->pradfemn->nangles << std::endl;
    fout << "num_points = " << pmbp->pradfemn->num_points << std::endl;
    fout << "num_edges = " << pmbp->pradfemn->num_edges << std::endl;
    fout << "num_triangles = " << pmbp->pradfemn->num_triangles << std::endl;

    std::ofstream fout2(pathdir + filenamepart + "_coords_xyzrthph" + ".txt");
    for (size_t i = 0; i < pmbp->pradfemn->num_points; i++) {
        fout2 << pmbp->pradfemn->x(i) << " " << pmbp->pradfemn->y(i) << " " << pmbp->pradfemn->z(i) << " "
                  << pmbp->pradfemn->r(i) << " " << pmbp->pradfemn->theta(i) << " " << pmbp->pradfemn->phi(i)
                  << std::endl;
    }

    std::ofstream fout3(pathdir + filenamepart + "_triangles" + ".txt");
    for (size_t i = 0; i < pmbp->pradfemn->num_triangles; i++) {
        fout3 << pmbp->pradfemn->triangles(i, 0) << " " << pmbp->pradfemn->triangles(i, 1) << " "
                  << pmbp->pradfemn->triangles(i, 2) << std::endl;
    }

    std::ofstream fout4(pathdir + filenamepart + "_edges" + ".txt");
    for (size_t i = 0; i < pmbp->pradfemn->num_edges; i++) {
        fout4 << pmbp->pradfemn->edges(i, 0) << " " << pmbp->pradfemn->edges(i, 1) << std::endl;
    }

    std::ofstream fout5(pathdir + filenamepart + "_massmatrix" + ".txt");
    auto mass_matrix = pmbp->pradfemn->mass_matrix;
    double mass_matrix_sum{0.};
    for (size_t i = 0; i < pmbp->pradfemn->num_points; i++) {
        for (size_t j = 0; j < pmbp->pradfemn->num_points; j++) {
            fout5 << mass_matrix(i, j) << " ";
            mass_matrix_sum += mass_matrix(i, j);
        }
        fout5 << std::endl;
    }
    std::cout << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "Geodesic grid information" << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << "Choice of basis: " << pmbp->pradfemn->basis << std::endl;
    std::cout << "Refinement level: " << pmbp->pradfemn->num_ref << std::endl;
    std::cout << "The sum of all the elements of the mass matrix is: " << mass_matrix_sum << std::endl;
    std::cout << std::endl;

    std::cout << "End of information " << std::endl;

    return;
}