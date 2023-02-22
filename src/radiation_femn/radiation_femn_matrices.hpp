#ifndef ATHENA_RADIATION_FEMN_MATRICES_HPP
#define ATHENA_RADIATION_FEMN_MATRICES_HPP

//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief Matrices for the FEM_N and FP_N schemes for GR Boltzmann transport

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

namespace radiationfemn {

    // -----------------------
    // Geodesic grid functions
    // -----------------------
    void
    GeodesicGridBaseGenerate(int &geogrid_level, int &geogrid_num_points, int &geogrid_num_edges, int &geogrid_num_triangles, HostArray1D<Real> &x, HostArray1D<Real> &y,
                             HostArray1D<Real> &z, HostArray1D<Real> &r, HostArray1D<Real> &theta, HostArray1D<Real> &phi, HostArray2D<int> &edges,
                             HostArray2D<int> &triangles);    // Generate base geodesic grid
    void GeodesicGridRefine(int &geogrid_level, int &geogrid_num_points, int &geogrid_num_edges, int &geogrid_num_triangles, HostArray1D<Real> &x, HostArray1D<Real> &y,
                            HostArray1D<Real> &z, HostArray1D<Real> &r, HostArray1D<Real> &theta, HostArray1D<Real> &phi, HostArray2D<int> &edges,
                            HostArray2D<int> &triangles);          // Refine geodesic grid by one level
    void CartesianToSpherical(double xvar, double yvar, double zvar, double &rvar, double &thetavar, double &phivar);   // Convert from Cartesian to spherical coordinates
    double FindEdgesIndex(int e1, int e2, HostArray2D<int> &edges);  // Given two edge indices, find

    // -------------------
    // FEM basis functions
    // -------------------
    // Type 1: 'Overlapping tent' (Default FEM_N choice)
    double FEMBasis1Type1(double xi1, double xi2, double xi3);
    double FEMBasis2Type1(double xi1, double xi2, double xi3);
    double FEMBasis3Type1(double xi1, double xi2, double xi3);
    // Type 2: 'Non-overlapping tent'
    double FEMBasis1Type2(double xi1, double xi2, double xi3);
    double FEMBasis2Type2(double xi1, double xi2, double xi3);
    double FEMBasis3Type2(double xi1, double xi2, double xi3);
    // Type 3: 'Overlapping honeycomb'
    double FEMBasis1Type3(double xi1, double xi2, double xi3);
    double FEMBasis2Type3(double xi1, double xi2, double xi3);
    double FEMBasis3Type3(double xi1, double xi2, double xi3);
    // Type 4: 'Non-overlapping honeycomb' (S_N choice)
    double FEMBasis1Type4(double xi1, double xi2, double xi3);
    double FEMBasis2Type4(double xi1, double xi2, double xi3);
    double FEMBasis3Type4(double xi1, double xi2, double xi3);


} // namespace radiationfemn

#endif //ATHENA_RADIATION_FEMN_MATRICES_HPP