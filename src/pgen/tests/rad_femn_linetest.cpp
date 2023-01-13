//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_linetest.cpp
//! \brief the 2d linetest problem with FEM_N

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

void ProblemGenerator::RadiationFEMNLinetest(ParameterInput *pin, const bool restart) {
    if (restart) return;

    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

    if (pmbp->pradfemn == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "The 2d line source problem generator can only be run with radiation-femn, but no "
                  << "<radiation-femn> block in input file" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (pmbp->pmesh->two_d == false) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "The 2d line source problem generator can only be run with two dimensions, but parfile"
                  << "grid setup is not in 2d" << std::endl;
        exit(EXIT_FAILURE);
    }

    // capture var pmy_mesh_->mb_indcs;
    auto &indcs = pmy_mesh_->mb_indcs;
    auto &size = pmbp->pmb->mb_size;
    int &is = indcs.is;
    int &ie = indcs.ie;
    int &js = indcs.js;
    int &je = indcs.je;
    int &ks = indcs.ks;
    int &ke = indcs.ke;
    int nang1 = pmbp->pradfemn->nangles - 1;

    Real omega = 0.03; // Garrett and Hauck's recommendation for the choice of omega

    auto &i0_ = pmbp->pradfemn->i0;
    par_for("pgen_linetest_radiation_femn", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, nang1, ks, ke, js, je, is,
            ie,
            KOKKOS_LAMBDA(int m, int A, int k, int j, int i) {
                Real &x1min = size.d_view(m).x1min;
                Real &x1max = size.d_view(m).x1max;
                int nx1 = indcs.nx1;
                Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

                Real &x2min = size.d_view(m).x2min;
                Real &x2max = size.d_view(m).x2max;
                int nx2 = indcs.nx2;
                Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

                // Implement Eqn. (58) of Garrett & Hauck 2013 (DOI: 10.1080/00411450.2014.910226)
                i0_(m, A, k, j, i) = std::max(
                        exp(-(x1 * x1 + x2 * x2) / (2.0 * omega * omega)) / (8.0 * M_PI * omega * omega), 1e-4);

            }
    );

    return;
}