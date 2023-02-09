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
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

    // ------------------------------------------
    // Convert cartesian to spherical coordinates
    KOKKOS_INLINE_FUNCTION
    void RadiationFEMN::CartesianToSpherical(double xvar, double yvar, double zvar, double &rvar, double &thetavar,
                                             double &phivar) {
        rvar = sqrt(xvar * xvar + yvar * yvar + zvar * zvar);
        thetavar = acos(zvar / rvar);
        phivar = atan2(yvar, xvar);
    }

    // -------------------------------------------------------------------------------
    // Given two points of an edge, find the index of the edge array of their location
    KOKKOS_INLINE_FUNCTION
    double RadiationFEMN::FindEdgesIndex(int e1, int e2) {
        int index{-42};
        for (size_t i = 0; i < num_edges; i++) {
            if ((edges(i, 0) == e1 && edges(i, 1) == e2) || (edges(i, 0) == e2 && edges(i, 1) == e1)) {
                index = i;
                break;
            }
        }
        return index;
    }

    //KOKKOS_INLINE_FUNCTION
    void RadiationFEMN::GeodesicGridRefine() {
        int new_num_ref = num_ref + 1;
        int new_num_points = 12 * pow(4, new_num_ref);
        if (new_num_ref != 0) {
            for (size_t i = 0; i < new_num_ref; i++) {
                new_num_points -= 6 * pow(4, i);
            }
        }
        int new_num_edges = 3 * (new_num_points - 2);
        int new_num_triangles = 2 * (new_num_points - 2);

        DvceArray1D<Real> xnew("xnew", new_num_points);
        DvceArray1D<Real> ynew("ynew", new_num_points);
        DvceArray1D<Real> znew("znew", new_num_points);
        DvceArray1D<Real> rnew("rnew", new_num_points);
        DvceArray1D<Real> thetanew("thetanew", new_num_points);
        DvceArray1D<Real> phinew("phinew", new_num_points);

        DvceArray2D<int> edgesnewtemp("edgesnewtemp", 9 * num_triangles, 2);
        DvceArray2D<int> edgesnew("edgesnew", new_num_edges, 2);
        DvceArray2D<int> trianglesnew("trianglesnew", new_num_triangles, 3);

        for (size_t i = 0; i < num_points; i++) {
            xnew(i) = x(i);
            ynew(i) = y(i);
            znew(i) = z(i);
        }

        for (size_t i = 0; i < num_edges; i++) {
            int e1 = edges(i, 0);
            int e2 = edges(i, 1);

            xnew(num_points + i) = (x(e1) + x(e2)) / 2.0;
            ynew(num_points + i) = (y(e1) + y(e2)) / 2.0;
            znew(num_points + i) = (z(e1) + z(e2)) / 2.0;

            double mod_point = sqrt(
                    xnew(num_points + i) * xnew(num_points + i) + ynew(num_points + i) * ynew(num_points + i) +
                    znew(num_points + i) * znew(num_points + i));
            double scaling_factor = sqrt(x(0) * x(0) + y(0) * y(0) + z(0) * z(0)) / mod_point;

            xnew(num_points + i) = scaling_factor * xnew(num_points + i);
            ynew(num_points + i) = scaling_factor * ynew(num_points + i);
            znew(num_points + i) = scaling_factor * znew(num_points + i);

        }

        for (size_t i = 0; i < new_num_points; i++) {
            CartesianToSpherical(xnew(i), ynew(i), znew(i), rnew(i), thetanew(i), phinew(i));
        }

        for (size_t i = 0; i < num_triangles; i++) {
            int t1 = triangles(i, 0);
            int t2 = triangles(i, 1);
            int t3 = triangles(i, 2);

            int midpoint_index_0 = num_points + FindEdgesIndex(t1, t2);
            int midpoint_index_1 = num_points + FindEdgesIndex(t2, t3);
            int midpoint_index_2 = num_points + FindEdgesIndex(t1, t3);

            trianglesnew(4 * i, 0) = t1;
            trianglesnew(4 * i, 1) = midpoint_index_0;
            trianglesnew(4 * i, 2) = midpoint_index_2;
            trianglesnew(4 * i + 1, 0) = midpoint_index_0;
            trianglesnew(4 * i + 1, 1) = midpoint_index_1;
            trianglesnew(4 * i + 1, 2) = t2;
            trianglesnew(4 * i + 2, 0) = midpoint_index_1;
            trianglesnew(4 * i + 2, 1) = midpoint_index_2;
            trianglesnew(4 * i + 2, 2) = t3;
            trianglesnew(4 * i + 3, 0) = midpoint_index_0;
            trianglesnew(4 * i + 3, 1) = midpoint_index_1;
            trianglesnew(4 * i + 3, 2) = midpoint_index_2;

            edgesnewtemp(9 * i, 0) = t1;
            edgesnewtemp(9 * i, 1) = midpoint_index_0;
            edgesnewtemp(9 * i + 1, 0) = t1;
            edgesnewtemp(9 * i + 1, 1) = midpoint_index_2;
            edgesnewtemp(9 * i + 2, 0) = t2;
            edgesnewtemp(9 * i + 2, 1) = midpoint_index_0;
            edgesnewtemp(9 * i + 3, 0) = t2;
            edgesnewtemp(9 * i + 3, 1) = midpoint_index_1;
            edgesnewtemp(9 * i + 4, 0) = t3;
            edgesnewtemp(9 * i + 4, 1) = midpoint_index_1;
            edgesnewtemp(9 * i + 5, 0) = t3;
            edgesnewtemp(9 * i + 5, 1) = midpoint_index_2;
            edgesnewtemp(9 * i + 6, 0) = midpoint_index_0;
            edgesnewtemp(9 * i + 6, 1) = midpoint_index_1;
            edgesnewtemp(9 * i + 7, 0) = midpoint_index_0;
            edgesnewtemp(9 * i + 7, 1) = midpoint_index_2;
            edgesnewtemp(9 * i + 8, 0) = midpoint_index_1;
            edgesnewtemp(9 * i + 8, 1) = midpoint_index_2;
        }


        size_t index{0};
        for (size_t i = 0; i < 9 * num_triangles; i++) {
            bool is_present{false};
            for (size_t j = 0; j < new_num_edges; j++) {
                if ((edgesnew(j, 0) == edgesnewtemp(i, 0) && edgesnew(j, 1) == edgesnewtemp(i, 1)) ||
                    (edgesnew(j, 0) == edgesnewtemp(i, 1) && edgesnew(j, 1) == edgesnewtemp(i, 0))) {
                    is_present = true;
                }

            }

            if (!is_present) {
                edgesnew(index, 0) = (edgesnewtemp(i, 0) < edgesnewtemp(i, 1)) * edgesnewtemp(i, 0) +
                                     (edgesnewtemp(i, 0) > edgesnewtemp(i, 1)) * edgesnewtemp(i, 1);
                edgesnew(index, 1) = (edgesnewtemp(i, 0) < edgesnewtemp(i, 1)) * edgesnewtemp(i, 1) +
                                     (edgesnewtemp(i, 0) > edgesnewtemp(i, 1)) * edgesnewtemp(i, 0);
                index++;
            }
        }

        Kokkos::realloc(x, new_num_points);
        Kokkos::realloc(y, new_num_points);
        Kokkos::realloc(z, new_num_points);
        Kokkos::realloc(r, new_num_points);
        Kokkos::realloc(theta, new_num_points);
        Kokkos::realloc(phi, new_num_points);
        Kokkos::realloc(edges, new_num_edges, 2);
        Kokkos::realloc(triangles, new_num_triangles, 3);

        Kokkos::deep_copy(x, xnew);
        Kokkos::deep_copy(y, ynew);
        Kokkos::deep_copy(z, znew);
        Kokkos::deep_copy(r, rnew);
        Kokkos::deep_copy(theta, thetanew);
        Kokkos::deep_copy(phi, phinew);
        Kokkos::deep_copy(edges, edgesnew);
        Kokkos::deep_copy(triangles, trianglesnew);

        num_ref = new_num_ref;
        num_points = new_num_points;
        nangles = new_num_points;
        num_edges = new_num_edges;
        num_triangles = new_num_triangles;

    }

}