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
    void GeodesicGridBaseGenerate();    // Generate base geodesic grid
    void GeodesicGridRefine();          // Refine geodesic grid by one level
    void GeodesicGridRefineN();         // Refine geodesic grid by N levels
    void CartesianToSpherical(double xvar, double yvar, double zvar, double &rvar, double &thetavar, double &phivar);   // Convert from Cartesian to spherical coordinates
    double FindEdgesIndex(int e1, int e2);  // Given two edge indices, find

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