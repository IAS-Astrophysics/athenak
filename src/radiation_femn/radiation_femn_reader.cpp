//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_reader.cpp
//! \brief function to read matrices from file

#include "athena.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include <string>
#include <fstream>

namespace radiationfemn {

//----------------------------------------------------------------------------------------
//! \fn void RadiationFEMN::LoadMatrix(
//  \brief Load matrices constructed from the geodesic grid

    void RadiationFEMN::LoadMatrix(int num_angles, int basis, const std::string& matname, DvceArray2D<Real> &mat, const std::string& path) {
        int nvar = nangles;
        HostArray2D<Real> matrix("host_temp_array", nangles, nangles);
        Real value(0);

        std::ifstream file;
        file.open(path +"/"+ matname + "_" +std::to_string(num_angles) + "_" + std::to_string(basis) + ".txt");
        if(!file) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                      << "The location of the matrix cannot be found. Please check that the filepath"
                      << " is entered correctly." << std::endl;
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < nvar; i++) {
            for (size_t j = 0; j < nvar; j++) {
                file >> value;
                matrix(i,j) = value;
            }
        }
        file.close();

        Kokkos::deep_copy(mat, matrix);

        return;
    }

    void RadiationFEMN::CalcIntPsi() {
        int nvar = nangles;
        HostArray1D<Real> matrix("host_temp_array_1d", nvar);
        Real value(0);

        Kokkos::deep_copy(matrix, 0.);
        for (size_t i = 0; i < nvar; i++) {
            for (size_t j = 0; j < nvar; j++) {
                matrix(i) = matrix(i) + mass_matrix(i,j);
            }
        }

        Kokkos::deep_copy(int_psi, matrix);

        return;
    }
}  //namespace radiationfemn