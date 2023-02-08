//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_matrices.cpp
//  \brief implementation of the radiation FEM_N mass and stiffness matrices and the integration routines

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

    // --------------------------------------------
    // Calculate determinant of Jacobian term
    KOKKOS_INLINE_FUNCTION
    double CalculateDeterminantJacobian(double x1, double y1, double z1, double x2, double y2, double z2,
                                        double x3, double y3, double z3, double xi1, double xi2,
                                        double xi3) {
        double result = 0.5 *
                        pow((x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3),
                            2) /
                        (x1 * x1 * xi1 * xi1 + 2 * x1 * x2 * xi1 * xi2 + x2 * x2 * xi2 * xi2 +
                         x3 * x3 * (-1 + xi1 + xi2) * (-1 + xi1 + xi2) -
                         2 * x3 * (-1 + xi1 + xi2) * (x1 * xi1 + x2 * xi2) +
                         xi1 * xi1 * y1 * y1 + 2 * xi1 * xi2 * y1 * y2 +
                         xi2 * xi2 * y2 * y2 + 2 * xi1 * y1 * y3 - 2 * xi1 * xi1 * y1 * y3 - 2 * xi1 * xi2 * y1 * y3 +
                         2 * xi2 * y2 * y3 - 2 * xi1 * xi2 * y2 * y3 - 2 * xi2 * xi2 * y2 * y3 + y3 * y3 -
                         2 * xi1 * y3 * y3 +
                         xi1 * xi1 * y3 * y3 - 2 * xi2 * y3 * y3 + 2 * xi1 * xi2 * y3 * y3 + xi2 * xi2 * y3 * y3 +
                         xi1 * xi1 * z1 * z1 +
                         2 * xi1 * xi2 * z1 * z2 + xi2 * xi2 * z2 * z2 -
                         2 * (-1 + xi1 + xi2) * (xi1 * z1 + xi2 * z2) * z3 +
                         (-1 + xi1 + xi2) * (-1 + xi1 + xi2) * z3 * z3);

        return result;
    }

    // -----------------------------------------------------------------------
    // Integrate a function over a spherical triangle whose vertices are given
    KOKKOS_INLINE_FUNCTION
    double
    RadiationFEMN::IntegrateFunctionSphericaTriangle(double x1, double y1, double z1, double x2, double y2, double z2,
                                                     double x3, double y3, double z3,
                                                     std::function<double(double, double, double, double, double,
                                                                          double, double, double, double, double,
                                                                          double, double)> func) {
        double result{0.};
        for (size_t i = 0; i < scheme_num_points; i++) {
            result += CalculateDeterminantJacobian(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0),
                                                   scheme_points(i, 1), scheme_points(i, 2)) *
                      func(x1, y1, z1, x2, y2, z2, x3, y3, z3, scheme_points(i, 0), scheme_points(i, 1),
                           scheme_points(i, 2)) * scheme_weights(i);
        }

        result = 0.5 * result;

        return result;
    }

    KOKKOS_INLINE_FUNCTION
    double RadiationFEMN::IntegratePsiPsiAB(int a, int b) {
        double result{0.};

        bool is_edge{false};
        FindTriangles(a,b,is_edge);

        if(is_edge) {
            for (size_t i = 0; i < 6; i++) {
                if(edge_triangles(i,0) >= 0) {
                    int triangle_index_1 = edge_triangles(i, 0);
                    int triangle_index_2 = edge_triangles(i, 1);
                    int triangle_index_3 = edge_triangles(i, 2);

                    double triangle_vertex_1_x = pmy_pack->pradfemn->x(triangle_index_1);
                    double triangle_vertex_1_y = pmy_pack->pradfemn->y(triangle_index_1);
                    double triangle_vertex_1_z = pmy_pack->pradfemn->z(triangle_index_1);

                    double triangle_vertex_2_x = pmy_pack->pradfemn->x(triangle_index_2);
                    double triangle_vertex_2_y = pmy_pack->pradfemn->y(triangle_index_2);
                    double triangle_vertex_2_z = pmy_pack->pradfemn->z(triangle_index_2);

                    double triangle_vertex_3_x = pmy_pack->pradfemn->x(triangle_index_3);
                    double triangle_vertex_3_y = pmy_pack->pradfemn->y(triangle_index_3);
                    double triangle_vertex_3_z = pmy_pack->pradfemn->z(triangle_index_3);

                    /*double integrated_result = IntegrateFuncSphericalTriangle(triangle_vertex_1_x, triangle_vertex_1_y,
                                                                              triangle_vertex_1_z, triangle_vertex_2_x,
                                                                              triangle_vertex_2_y, triangle_vertex_2_z,
                                                                              triangle_vertex_3_x, triangle_vertex_3_y,
                                                                              triangle_vertex_3_z,
                                                                              PsiPsiAB());*/
                }
            }
        }
    }

    /*
    void RadiationFEMN::IntegrateBasisBasisAB(int a, int b, double x, double y, double z) {
        double result{0.};
        double resultx{0.};
        double resulty{0.};
        double resultz{0.};

        bool is_edge = false;

        FindTriangles(a, b, is_edge);

        if (is_edge) {
            for (size_t i = 0; i < 6; i++) {
                int triangle_index_1 = edge_triangles(i, 0);
                int triangle_index_2 = edge_triangles(i, 1);
                int triangle_index_3 = edge_triangles(i, 2);

                double triangle_vertex_1_x = pmy_pack->pradfemn->x(triangle_index_1);
                double triangle_vertex_1_y = pmy_pack->pradfemn->y(triangle_index_1);
                double triangle_vertex_1_z = pmy_pack->pradfemn->z(triangle_index_1);

                double triangle_vertex_2_x = pmy_pack->pradfemn->x(triangle_index_2);
                double triangle_vertex_2_y = pmy_pack->pradfemn->y(triangle_index_2);
                double triangle_vertex_2_z = pmy_pack->pradfemn->z(triangle_index_2);

                double triangle_vertex_3_x = pmy_pack->pradfemn->x(triangle_index_3);
                double triangle_vertex_3_y = pmy_pack->pradfemn->y(triangle_index_3);
                double triangle_vertex_3_z = pmy_pack->pradfemn->z(triangle_index_3);

                double integrated_result_x = IntegrateFuncSphericalTriangle(triangle_vertex_1_x, triangle_vertex_1_y,
                                                                            triangle_vertex_1_z, triangle_vertex_2_x,
                                                                            triangle_vertex_2_y, triangle_vertex_2_z,
                                                                            triangle_vertex_3_x, triangle_vertex_3_y,
                                                                            triangle_vertex_3_z,
                                                                            FEMBasisABasisB);
            }
        }

    }*/
}