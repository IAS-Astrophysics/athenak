//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_compute_pmatrices.cpp
//  \brief generate final matrices for evolution equations. These are independent of angular scheme.

#include <iostream>

#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_linalg.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid.hpp"
#include "radiation_femn_matinv.hpp"

namespace radiationfemn {

// compute inverse of mass matrix (on host, then copy to device)
void RadiationFEMN::ComputeMassInverse() {

  std::cout << "Computing the inverse of the mass matrix ..." << std::endl;

  HostArray2D<Real> mass_matrix_host;
  HostArray2D<Real> mass_matrix_inv_host;
  HostArray2D<Real> lu_matrix_host;
  HostArray1D<Real> b_matrix_host;
  HostArray1D<Real> x_matrix_host;
  HostArray1D<int> pivots_host;

  Kokkos::realloc(mass_matrix_host, num_points, num_points);
  Kokkos::realloc(mass_matrix_inv_host, num_points, num_points);
  Kokkos::realloc(lu_matrix_host, num_points, num_points);
  Kokkos::realloc(b_matrix_host, num_points);
  Kokkos::realloc(x_matrix_host, num_points);
  Kokkos::realloc(pivots_host, num_points - 1);

  Kokkos::deep_copy(mass_matrix_host, mass_matrix);
  Kokkos::deep_copy(lu_matrix_host, mass_matrix);
  Kokkos::deep_copy(b_matrix_host, 0.);
  Kokkos::deep_copy(x_matrix_host, 0.);
  Kokkos::deep_copy(pivots_host, 0.);

  radiationfemn::LUInv<HostArray2D<Real>, HostArray1D<Real>, HostArray1D<int>>(mass_matrix_host, mass_matrix_inv_host, lu_matrix_host,
                                                                               x_matrix_host, b_matrix_host, pivots_host);
  Kokkos::deep_copy(mass_matrix_inv, mass_matrix_inv_host);

}

// compute P and Pmod matrices (multiplied by M^-1)
void RadiationFEMN::ComputePMatrices() {

  std::cout << "Computing P matrices and modified P matrices ..." << std::endl;

  DvceArray2D<Real> temp_array;
  HostArray2D<Real> temp_array_host;
  HostArray2D<Real> temp_array_corrected;
  HostArray5D<Real> temp_array_5d;

  Kokkos::realloc(temp_array, num_points, num_points);
  Kokkos::realloc(temp_array_host, num_points, num_points);
  Kokkos::realloc(temp_array_corrected, num_points, num_points);
  Kokkos::realloc(temp_array_5d, 4, 4, 3, num_points, num_points);

  // M^-1 M
  if (!multiply_massinv) {
    Kokkos::deep_copy(temp_array, mass_matrix);
  } else {
    radiationfemn::MatMatMultiply(mass_matrix_inv, mass_matrix, temp_array);
  }
  auto P_matrix_0 = Kokkos::subview(P_matrix, 0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_0, temp_array);
  auto Pmod_matrix_0 = Kokkos::subview(Pmod_matrix, 0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_0, temp_array);

  // stilde-x matrix
  if (!multiply_massinv) {
    Kokkos::deep_copy(temp_array, stiffness_matrix_x);
  } else {
    radiationfemn::MatMatMultiply(mass_matrix_inv, stiffness_matrix_x, temp_array);
  }
  auto P_matrix_1 = Kokkos::subview(P_matrix, 1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_1, temp_array);
  Kokkos::deep_copy(temp_array_host, temp_array);

  // stilde-x mod matrix
  radiationfemn::ZeroSpeedCorrection(temp_array_host, temp_array_corrected, 1. / Kokkos::sqrt(3.));
  auto Pmod_matrix_1 = Kokkos::subview(Pmod_matrix, 1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_1, temp_array_corrected);

  // stilde-y matrix
  if (!multiply_massinv) {
    Kokkos::deep_copy(temp_array, stiffness_matrix_y);
  } else {
    radiationfemn::MatMatMultiply(mass_matrix_inv, stiffness_matrix_y, temp_array);
  }
  auto P_matrix_2 = Kokkos::subview(P_matrix, 2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_2, temp_array);
  Kokkos::deep_copy(temp_array_host, temp_array);

  // stilde-y mod matrix
  radiationfemn::ZeroSpeedCorrection(temp_array_host, temp_array_corrected, 1. / Kokkos::sqrt(3.));
  auto Pmod_matrix_2 = Kokkos::subview(Pmod_matrix, 2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_2, temp_array_corrected);

  // stilde-z matrix
  if (!multiply_massinv) {
    Kokkos::deep_copy(temp_array, stiffness_matrix_z);
  } else {
    radiationfemn::MatMatMultiply(mass_matrix_inv, stiffness_matrix_z, temp_array);
  }
  auto P_matrix_3 = Kokkos::subview(P_matrix, 3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(P_matrix_3, temp_array);
  Kokkos::deep_copy(temp_array_host, temp_array);

  // stilde-z mod matrix
  radiationfemn::ZeroSpeedCorrection(temp_array_host, temp_array_corrected, 1. / Kokkos::sqrt(3.));
  auto Pmod_matrix_3 = Kokkos::subview(Pmod_matrix, 3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(Pmod_matrix_3, temp_array_corrected);

  // compute the G & F matrices
  HostArray2D<Real> mass_matrix_inv_host;
  Kokkos::realloc(mass_matrix_inv_host, num_points, num_points);
  Kokkos::deep_copy(mass_matrix_inv_host, mass_matrix_inv);

  for (int nu = 0; nu < 4; nu++) {
    for (int mu = 0; mu < 4; mu++) {
      for (int i = 0; i < 3; i++) {
        auto gm_host = Kokkos::subview(G_mat_host, nu, mu, i, Kokkos::ALL, Kokkos::ALL);
        auto gm_temp = Kokkos::subview(temp_array_5d, nu, mu, i, Kokkos::ALL, Kokkos::ALL);
        if (!multiply_massinv) {
          Kokkos::deep_copy(temp_array_host, gm_host);
        } else {
          radiationfemn::MatMultiplyHost(mass_matrix_inv_host, gm_host, temp_array_host);
        }
        Kokkos::deep_copy(gm_temp, temp_array_host);
      }
    }
  }
  Kokkos::deep_copy(G_matrix, temp_array_5d);

  for (int nu = 0; nu < 4; nu++) {
    for (int mu = 0; mu < 4; mu++) {
      for (int i = 0; i < 3; i++) {
        auto fm_host = Kokkos::subview(F_mat_host, nu, mu, i, Kokkos::ALL, Kokkos::ALL);
        auto fm_temp = Kokkos::subview(temp_array_5d, nu, mu, i, Kokkos::ALL, Kokkos::ALL);

        if (!multiply_massinv) {
          Kokkos::deep_copy(temp_array_host, fm_host);
        } else {
          radiationfemn::MatMultiplyHost(mass_matrix_inv_host, fm_host, temp_array_host);
        }
        Kokkos::deep_copy(fm_temp, temp_array_host);
      }
    }
  }
  Kokkos::deep_copy(F_matrix, temp_array_5d);

}

// compute the final source matrices (multiplied by M^-1)
void RadiationFEMN::ComputeSourceMatrices() {

  std::cout << "Constructing the final source matrices ..." << std::endl;

  DvceArray1D<Real> e_source_temp;
  DvceArray1D<Real> e_source_temp_mod;
  DvceArray2D<Real> S_source_temp;
  DvceArray2D<Real> S_source_temp_mod;

  Kokkos::realloc(e_source_temp, num_points);
  Kokkos::realloc(e_source_temp_mod, num_points);
  Kokkos::realloc(S_source_temp, num_points, num_points);
  Kokkos::realloc(S_source_temp_mod, num_points, num_points);

  Kokkos::deep_copy(e_source_temp, e_source);
  Kokkos::deep_copy(e_source_nominv, e_source);

  if (!multiply_massinv) {
    Kokkos::deep_copy(e_source_temp_mod, e_source);
  } else {
    radiationfemn::MatVecMultiply(mass_matrix_inv, e_source, e_source_temp_mod);
  }
  Kokkos::deep_copy(e_source, e_source_temp_mod);

  par_for("radiation_femn_compute_source_smatrix", DevExeSpace(), 0, num_points - 1, 0, num_points - 1,
          KOKKOS_LAMBDA(const int j, const int i) {
            S_source_temp(i, j) = e_source_temp(i) * e_source_temp(j);
          });

  if (!multiply_massinv) {
    Kokkos::deep_copy(S_source_temp_mod, S_source);
  } else {
    radiationfemn::MatMatMultiply(mass_matrix_inv, S_source_temp, S_source_temp_mod);
  }
  Kokkos::deep_copy(S_source, S_source_temp_mod);

}

}