//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tetrad.cpp
//  \brief Unit tests for checking tetrad transformation

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>
#include <cctype>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/rsolvers/flux_dyn_grmhd.hpp"
#include "eos/primitive-solver/geom_math.hpp"

template<class arr2D>
void ExtractADMMetric(Real g3d[NSPMETRIC], Real& alpha, Real beta_u[3], Real& detg,
                      arr2D& g_dd) {
  g3d[S11] = g_dd[1][1];
  g3d[S12] = g_dd[1][2];
  g3d[S13] = g_dd[1][3];
  g3d[S22] = g_dd[2][2];
  g3d[S23] = g_dd[2][3];
  g3d[S33] = g_dd[3][3];
  
  Real g3u[NSPMETRIC];
  detg = Primitive::GetDeterminant(g3d);
  Primitive::InvertMatrix(g3u, g3d, detg);
  Real beta_d[3] = {g_dd[0][1], g_dd[0][2], g_dd[0][3]};
  Primitive::RaiseForm(beta_u, beta_d, g3u);
  alpha = Kokkos::sqrt(-g_dd[0][0] + Primitive::Contract(beta_u, beta_d));
}

template<class arr2D>
bool CheckConsistency(arr2D& g_dd, arr2D& eta_dd, arr2D& e_ud, arr2D& e_dd,
                      Real det, Real tol) {
  // Check that we can accurately recover the spacetime metric.
  Real res[4][4] = {0.0};
  Real err = 0.0;
  for (int mu = 0; mu < 4; mu++) {
    for (int nu = 0; nu < 4; nu++) {
      for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
          res[mu][nu] += e_dd[a][mu]*e_dd[b][nu]*eta_dd[a][b];
        }
      }
      err += Kokkos::fabs(g_dd[mu][nu] - res[mu][nu]);
    }
  }
  err /= (16.0*det);
  // Check if the error is large.
  if (Kokkos::fabs(err) > tol) {
    std::cout << "Tetrad does not accurately recover input metric.\n";
    std::cout << "  " << res[0][0] << " " << res[0][1] << " " << res[0][2] << " "
                                                              << res[0][3] << "\n"
              << "  " << res[1][0] << " " << res[1][1] << " " << res[1][2] << " "
                                                              << res[1][3] << "\n"
              << "  " << res[2][0] << " " << res[2][1] << " " << res[2][2] << " "
                                                              << res[2][3] << "\n"
              << "  " << res[3][0] << " " << res[3][1] << " " << res[3][2] << " "
                                                              << res[3][3] << "\n";
    std::cout << "  Average relative error: " << err << "\n";
    return false;
  }

  // Check that the calculated metric is approximately symmetric.
  err = 0.0;
  for (int mu = 0; mu < 4; mu++) {
    for (int nu = 0; nu < mu; nu++) {
      err += Kokkos::fabs(res[mu][nu] - res[nu][mu]);
    }
  }

  err /= (6.0*det);
  
  if (Kokkos::fabs(err) > tol) {
    std::cout << "Tetrad does not compute a symmetric metric.\n"
              << "  Average relative error: " << err << "\n";
    return false;
  }

  // Check that we can accurately recover Minkowski space.
  for (int a = 0; a < 4; a++) {
    for (int b = 0; b < 4; b++) {
      res[a][b] = 0.0;
      for (int mu = 0; mu < 4; mu++) {
        for (int nu = 0; nu < 4; nu++) {
          res[a][b] += e_ud[mu][a]*e_ud[nu][b]*g_dd[mu][nu];
        }
      }
      err += Kokkos::fabs(eta_dd[a][b] - res[a][b]);
    }
  }

  err /= 16.0;
  // Check if the error is large.
  if (Kokkos::fabs(err) > tol) {
    std::cout << "Tetrad does not accurately transform to Minkowski space.\n";
    std::cout << "  " << res[0][0] << " " << res[0][1] << " " << res[0][2] << " "
                                                              << res[0][3] << "\n"
              << "  " << res[1][0] << " " << res[1][1] << " " << res[1][2] << " "
                                                              << res[1][3] << "\n"
              << "  " << res[2][0] << " " << res[2][1] << " " << res[2][2] << " "
                                                              << res[2][3] << "\n"
              << "  " << res[3][0] << " " << res[3][1] << " " << res[3][2] << " "
                                                              << res[3][3] << "\n";
    std::cout << "  Average relative error: " << err << "\n";
    return false;
  }

  // Check that the calculated metric is approximately symmetric.
  err = 0.0;
  for (int a = 0; a < 4; a++) {
    for (int b = 0; b < a; b++) {
      err += Kokkos::fabs(res[a][b] - res[b][a]);
    }
  }

  err /= 6.0;

  if (Kokkos::fabs(err) > tol) {
    std::cout << "Tetrad does not compute a symmetric eta.\n"
              << "Average relative error: " << err << "\n";
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  // Metric for Minkowski spacetime
  Real eta_dd[4][4] = {0.0};
  eta_dd[0][0] = -1.0;
  eta_dd[1][1] = 1.0;
  eta_dd[2][2] = 1.0;
  eta_dd[3][3] = 1.0;

  // Check that flat space is consistent.
  Real g_dd[4][4] = {0.0};
  g_dd[0][0] = -1.0;
  g_dd[1][1] = 1.0;
  g_dd[2][2] = 1.0;
  g_dd[3][3] = 1.0;
  Real g3d[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  Real alpha = 1.0;
  Real beta_u[3] = {0.0};
  Real e_ud[4][4], e_dd[4][4];
  dyngr::ComputeOrthonormalTetrad<1>(g3d, beta_u, alpha, 1.0, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -1.0, 1e-10)) {
    std::cout << "Minkowski Tetrad Test, x-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  dyngr::ComputeOrthonormalTetrad<2>(g3d, beta_u, alpha, 1.0, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -1.0, 1e-10)) {
    std::cout << "Minkowski Tetrad Test, y-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  dyngr::ComputeOrthonormalTetrad<3>(g3d, beta_u, alpha, 1.0, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -1.0, 1e-10)) {
    std::cout << "Minkowski Tetrad Test, z-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Check a spherical Schwarzschild spacetime located at r = 4M, phi = 0, theta = pi/4.
  Real r = 4.0;
  Real fac = 1.0 - 2.0/(r);
  Real theta = Kokkos::numbers::pi/4.0;
  g_dd[0][0] = -fac;
  g_dd[1][1] = 1.0/fac;
  g_dd[2][2] = r*r;
  g_dd[3][3] = Kokkos::sin(theta)*Kokkos::sin(theta)*r*r;
  Real detg;

  ExtractADMMetric(g3d, alpha, beta_u, detg, g_dd);

  dyngr::ComputeOrthonormalTetrad<1>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -alpha*alpha*detg, 1e-10)) {
    std::cout << "Schwarzschild Tetrad Test, x-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  dyngr::ComputeOrthonormalTetrad<2>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -alpha*alpha*detg, 1e-10)) {
    std::cout << "Schwarzschild Tetrad Test, y-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  dyngr::ComputeOrthonormalTetrad<3>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -alpha*alpha*detg, 1e-10)) {
    std::cout << "Schwarzschild Tetrad Test, z-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Check a Cartesian Kerr-Schild spacetime for Schwarzschild coordinates at
  // x = 3M, y = 4M, z = 5M
  Real x = 3.0;
  Real y = 4.0;
  Real z = 5.0;
  r = Kokkos::sqrt(x*x + y*y + z*z);
  Real f = 2.0/r;
  Real k_d[4] = {1.0, x/r, y/r, z/r};
  for (int mu = 0; mu < 4; mu++) {
    for (int nu = 0; nu < 4; nu++) {
      g_dd[mu][nu] = eta_dd[mu][nu] + f*k_d[mu]*k_d[nu];
    }
  }
  // Check that a four-vector is invariant under a tetrad transform
  Real u_u[4] = {3.0, 0.5, 1.2, 0.2};
  Real mag_g = 0;
  for (int mu = 0; mu < 4; mu++) {
    for (int nu = 0; nu < 4; nu++) {
      mag_g += u_u[mu]*u_u[nu]*g_dd[mu][nu];
    }
  }
  
  ExtractADMMetric(g3d, alpha, beta_u, detg, g_dd);

  dyngr::ComputeOrthonormalTetrad<1>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -alpha*alpha*detg, 1e-10)) {
    std::cout << "Kerr-Schild Tetrad Test, x-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Transform the four-vector and check that the magnitude is the same.
  Real u_t[4] = {0.0};
  for (int a = 0; a < 4; a++) {
    for (int mu = 0; mu < 4; mu++) {
      u_t[a] += e_dd[a][mu]*u_u[mu];
    }
  }
  Real mag_eta = -u_t[0]*u_t[0] + u_t[1]*u_t[1] + u_t[2]*u_t[2] + u_t[3]*u_t[3];
  if (Kokkos::fabs((mag_eta - mag_g)/mag_g) > 1e-10) {
    std::cout << "Kerr-Schild Four-Vector Test, x-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  dyngr::ComputeOrthonormalTetrad<2>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -alpha*alpha*detg, 1e-10)) {
    std::cout << "Kerr-Schild Tetrad Test, y-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Transform the four-vector and check that the magnitude is the same.
  for (int a = 0; a < 4; a++) {
    u_t[a] = 0.0;
    for (int mu = 0; mu < 4; mu++) {
      u_t[a] += e_dd[a][mu]*u_u[mu];
    }
  }
  mag_eta = -u_t[0]*u_t[0] + u_t[1]*u_t[1] + u_t[2]*u_t[2] + u_t[3]*u_t[3];
  if (Kokkos::fabs((mag_eta - mag_g)/mag_g) > 1e-10) {
    std::cout << "Kerr-Schild Four-Vector Test, y-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  dyngr::ComputeOrthonormalTetrad<3>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);

  if (!CheckConsistency(g_dd, eta_dd, e_ud, e_dd, -alpha*alpha*detg, 1e-10)) {
    std::cout << "Kerr-Schild Tetrad Test, z-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Transform the four-vector and check that the magnitude is the same.
  for (int a = 0; a < 4; a++) {
    u_t[a] = 0.0;
    for (int mu = 0; mu < 4; mu++) {
      u_t[a] += e_dd[a][mu]*u_u[mu];
    }
  }
  mag_eta = -u_t[0]*u_t[0] + u_t[1]*u_t[1] + u_t[2]*u_t[2] + u_t[3]*u_t[3];
  if (Kokkos::fabs((mag_eta - mag_g)/mag_g) > 1e-10) {
    std::cout << "Kerr-Schild Four-Vector Test, z-direction failed" << std::endl;
    exit(EXIT_FAILURE);
  }


  // Check that a four-velocity and spatial three-velocity transform consistently in
  // Kerr-Schild coordinates.
  dyngr::ComputeOrthonormalTetrad<1>(g3d, beta_u, alpha, 1.0/detg, e_ud, e_dd);
  Real wvi[3] = {1.0, 2.0, 3.0};
  Real W = 1.0;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      W += g_dd[j+1][i+1]*wvi[j]*wvi[i];
    }
  }
  W = Kokkos::sqrt(W);
  u_u[0] = W/alpha;
  u_u[1] = wvi[0] - W*beta_u[0]/alpha;
  u_u[2] = wvi[1] - W*beta_u[1]/alpha;
  u_u[3] = wvi[2] - W*beta_u[2]/alpha;

  Real wv_t[3] = {0.0};
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < 3; i++) {
      wv_t[a] += e_dd[a+1][i+1]*wvi[i];
    }
  }

  for (int a = 0; a < 4; a++) {
    u_t[a] = 0.0;
    for (int mu = 0; mu < 4; mu++) {
      u_t[a] += e_dd[a][mu]*u_u[mu];
    }
  }

  if (Kokkos::fabs((u_t[0] + W)/W) > 1e-10) {
    std::cout << "Kerr-Schild Velocity Test, x-direction failed:\n"
              << "  t-component is not correct" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (Kokkos::fabs((u_t[1] - wv_t[0])/W) > 1e-10) {
    std::cout << "Kerr-Schild Velocity Test, x-direction failed:\n"
              << " x-component is not correct" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (Kokkos::fabs((u_t[2] - wv_t[1])/W) > 1e-10) {
    std::cout << "Kerr-Schild Velocity Test, x-direction failed:\n"
              << " y-component is not correct" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (Kokkos::fabs((u_t[3] - wv_t[2])/W) > 1e-10) {
    std::cout << "Kerr-Schild Velocity Test, x-direction failed:\n"
              << " z-component is not correct" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "All tetrad tests passed!\n";

  return;
}
