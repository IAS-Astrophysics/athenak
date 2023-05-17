//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_loadmatrices.cpp
//  \brief generate and load matrices for the angular grid

#include <iostream>

#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

    void RadiationFEMN::LoadFEMNMatrices() {
        std::cout << "Geodesic grid: Generating ..." << std::endl;

        HostArray1D<Real> x;
        HostArray1D<Real> y;
        HostArray1D<Real> z;
        HostArray1D<Real> r;
        HostArray1D<Real> theta;
        HostArray1D<Real> phi;
        HostArray2D<int> edges;
        HostArray2D<int> triangles;

        GeodesicGridBaseGenerate(num_ref, num_points, num_edges, num_triangles, x, y, z, r, theta, phi, edges, triangles);

        std::cout << "Geodesic grid: Base grid generated! num_ref = " << num_ref << ", num_points = " << num_points << ", num_edges = " << num_edges << ", num_triangles = "
                  << num_triangles << std::endl;

        if (refinement_level > 0) {
            for (size_t i = num_ref; i < refinement_level; i++) {
                GeodesicGridRefine(num_ref, num_points, num_edges, num_triangles, x, y, z, r, theta, phi, edges, triangles);
                std::cout << "Geodesic grid: Refined grid generated! num_ref = " << num_ref << ", num_points = " << num_points << ", num_edges = " << num_edges
                          << ", num_triangles = " << num_triangles << std::endl;
            }
        }

        for (size_t i = 0; i < num_points; i++) {
            angular_grid(i, 0) = phi(i);
            angular_grid(i, 1) = theta(i);
        }

        
    }
}  // namespace radiationfemn