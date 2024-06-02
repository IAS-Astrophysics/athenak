//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_beams.cpp
//  \brief set up beam sources

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_linalg.hpp"
#include "coordinates/cell_locations.hpp"
#include "radiation_femn_geodesic_grid.hpp"

namespace radiationfemn {

void ApplyBeamSourcesFEMN(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;
  //int &ks = indcs.ks;
  int npts1 = pmesh->pmb_pack->pradfemn->num_points_total - 1;
  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;

  int &ng = indcs.ng;
  //int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  auto &f0_ = pmesh->pmb_pack->pradfemn->f0;
  auto &num_beams_ = pmesh->pmb_pack->pradfemn->num_beams;
  auto &beam_source_1_y1_ = pmesh->pmb_pack->pradfemn->beam_source_1_y1;
  auto &beam_source_1_y2_ = pmesh->pmb_pack->pradfemn->beam_source_1_y2;
  auto &beam_source_2_y1_ = pmesh->pmb_pack->pradfemn->beam_source_2_y1;
  auto &beam_source_2_y2_ = pmesh->pmb_pack->pradfemn->beam_source_2_y2;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradfemn->beam_source_1_vals;
  auto &beam_source_2_vals_ = pmesh->pmb_pack->pradfemn->beam_source_2_vals;

  if (pmesh->one_d) {
    par_for("radiation_femn_beams_populate_1d", DevExeSpace(), 0, nmb1, 0, npts1, 0, (n3 - 1), 0, (n2 - 1),
            KOKKOS_LAMBDA(int m, int n, int k, int j) {

              switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
                case BoundaryFlag::outflow:
                  for (int i = 0; i < ng; ++i) {
                    f0_(m, n, k, j, is - i - 1) = beam_source_1_vals_(n);
                  }

                  break;

                default:break;
              }
            });
  } else if (pmesh->two_d) {
    par_for("radiation_femn_beams_populate", DevExeSpace(), 0, nmb1, 0, npts1, 0, (n3 - 1), 0, (n2 - 1),
            KOKKOS_LAMBDA(int m, int n, int k, int j) {

              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              int nx2 = indcs.nx2;
              Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

              switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
                case BoundaryFlag::outflow:
                  if (beam_source_1_y1_ <= x2 && x2 <= beam_source_1_y2_) {
                    for (int i = 0; i < ng; ++i) {
                      f0_(m, n, k, j, is - i - 1) = beam_source_1_vals_(n);
                    }
                  }

                  if (num_beams_ > 1 && beam_source_2_y1_ <= x2 && x2 <= beam_source_2_y2_) {
                    for (int i = 0; i < ng; ++i) {
                      f0_(m, n, k, j, is - i - 1) = beam_source_2_vals_(n);
                    }
                  }
                  break;

                default:break;
              }
            });
  } else {
    std::cout << "Beams for Radiation FEMN not implemented in 3d!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void RadiationFEMN::InitializeBeamsSourcesFPN() {

  std::cout << "Initializing beam sources for FPN" << std::endl;

  HostArray1D<Real> beam_source_1_vals_h;
  HostArray1D<Real> beam_source_2_vals_h;
  HostArray2D<Real> angular_grid_h;
  Kokkos::realloc(beam_source_1_vals_h, num_points);
  Kokkos::realloc(beam_source_2_vals_h, num_points);
  Kokkos::realloc(angular_grid_h, num_points, 2);
  Kokkos::deep_copy(angular_grid_h, angular_grid);

  for (int i = 0; i < num_points; i++) {
    beam_source_1_vals_h(i) = fpn_basis_lm((int) angular_grid_h(i, 0), (int) angular_grid_h(i, 1), beam_source_1_phi, beam_source_1_theta);
  }
  Kokkos::deep_copy(beam_source_1_vals, beam_source_1_vals_h);

  if (num_beams > 1) {
    for (int i = 0; i < num_points; i++) {
      beam_source_2_vals_h(i) = fpn_basis_lm((int) angular_grid_h(i, 0), (int) angular_grid_h(i, 1), beam_source_2_phi, beam_source_2_theta);
    }
    Kokkos::deep_copy(beam_source_2_vals, beam_source_2_vals_h);
  }
}

void RadiationFEMN::InitializeBeamsSourcesM1() {

  std::cout << "Initializing beam sources for M1" << std::endl;

  HostArray1D<Real> beam_source_1_vals_h;
  HostArray1D<Real> beam_source_2_vals_h;
  Kokkos::realloc(beam_source_1_vals_h, num_points);
  Kokkos::realloc(beam_source_2_vals_h, num_points);

  Real Sen = (Kokkos::pow(energy_max, 4) - 0.) / 4.0;
  Real Fnorm = 1. / Sen;
  Real E = Fnorm;
  Real Fx = Fnorm * Kokkos::sin(beam_source_1_theta) * Kokkos::cos(beam_source_1_phi);
  Real Fy = Fnorm * Kokkos::sin(beam_source_1_theta) * Kokkos::sin(beam_source_1_phi);
  Real Fz = Fnorm * Kokkos::cos(beam_source_1_theta);
  Real F2 = Fx * Fx + Fy * Fy + Fz * Fz;

  E = Kokkos::fmax(E, rad_E_floor);
  Real lim = E * E * (1. - rad_eps);
  if (F2 > lim) {
    Real fac = lim / F2;
    Fx = fac * Fx;
    Fy = fac * Fy;
    Fz = fac * Fz;
  }

  beam_source_1_vals_h(0) = (1. / Kokkos::sqrt(4. * M_PI)) * E;
  beam_source_1_vals_h(1) = -Kokkos::sqrt(3. / (4. * M_PI)) * Fy;
  beam_source_1_vals_h(2) = Kokkos::sqrt(3. / (4. * M_PI)) * Fz;
  beam_source_1_vals_h(3) = -Kokkos::sqrt(3. / (4. * M_PI)) * Fx;
  beam_source_1_vals_h(4) = 0;
  beam_source_1_vals_h(5) = 0;
  beam_source_1_vals_h(6) = 0;
  beam_source_1_vals_h(7) = 0;
  beam_source_1_vals_h(8) = 0;

  Kokkos::deep_copy(beam_source_1_vals, beam_source_1_vals_h);

  if (num_beams > 1) {
    Fnorm = 1. / Sen;
    E = Fnorm;
    Fx = Fnorm * Kokkos::sin(beam_source_2_theta) * Kokkos::cos(beam_source_2_phi);
    Fy = Fnorm * Kokkos::sin(beam_source_2_theta) * Kokkos::sin(beam_source_2_phi);
    Fz = Fnorm * Kokkos::cos(beam_source_2_theta);

    E = Kokkos::fmax(E, rad_E_floor);
    lim = E * E * (1. - rad_eps);
    if (F2 > lim) {
      Real fac = lim / F2;
      Fx = fac * Fx;
      Fy = fac * Fy;
      Fz = fac * Fz;
    }

    beam_source_2_vals_h(0) = (1. / Kokkos::sqrt(4. * M_PI)) * E;
    beam_source_2_vals_h(1) = -Kokkos::sqrt(3. / (4. * M_PI)) * Fy;
    beam_source_2_vals_h(2) = Kokkos::sqrt(3. / (4. * M_PI)) * Fz;
    beam_source_2_vals_h(3) = -Kokkos::sqrt(3. / (4. * M_PI)) * Fx;
    beam_source_2_vals_h(4) = 0;
    beam_source_2_vals_h(5) = 0;
    beam_source_2_vals_h(6) = 0;
    beam_source_2_vals_h(7) = 0;
    beam_source_2_vals_h(8) = 0;

    Kokkos::deep_copy(beam_source_2_vals, beam_source_2_vals_h);
  }
}

void RadiationFEMN::InitializeBeamsSourcesFEMN() {

  std::cout << "Initializing beam sources for FEM" << std::endl;

  HostArray1D<Real> psi_basis;
  Kokkos::realloc(psi_basis, num_points);
  Kokkos::deep_copy(psi_basis, 0);

  Real x0 = Kokkos::sin(beam_source_1_theta) * Kokkos::cos(beam_source_1_phi);
  Real y0 = Kokkos::sin(beam_source_1_theta) * Kokkos::sin(beam_source_1_phi);
  Real z0 = Kokkos::cos(beam_source_1_theta);

  for (int i = 0; i < num_triangles; i++) {
    Real x1 = angular_grid_cartesian(triangle_information(i, 0), 0);
    Real y1 = angular_grid_cartesian(triangle_information(i, 0), 1);
    Real z1 = angular_grid_cartesian(triangle_information(i, 0), 2);

    Real x2 = angular_grid_cartesian(triangle_information(i, 1), 0);
    Real y2 = angular_grid_cartesian(triangle_information(i, 1), 1);
    Real z2 = angular_grid_cartesian(triangle_information(i, 1), 2);

    Real x3 = angular_grid_cartesian(triangle_information(i, 2), 0);
    Real y3 = angular_grid_cartesian(triangle_information(i, 2), 1);
    Real z3 = angular_grid_cartesian(triangle_information(i, 2), 2);

    Real a = (x3 * y2 * z0 - x2 * y3 * z0 - x3 * y0 * z2 + x0 * y3 * z2 + x2 * y0 * z3 - x0 * y2 * z3)
             / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3);
    Real b = (x3 * y1 * z0 - x1 * y3 * z0 - x3 * y0 * z1 + x0 * y3 * z1 + x1 * y0 * z3 - x0 * y1 * z3)
             / (-(x3 * y2 * z1) + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 - x2 * y1 * z3 + x1 * y2 * z3);
    Real c = (x2 * y1 * z0 - x1 * y2 * z0 - x2 * y0 * z1 + x0 * y2 * z1 + x1 * y0 * z2 - x0 * y1 * z2)
             / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3);

    Real lam = 1. / (a + b + c);

    if (a >= 0 && b >= 0 && c >= 0 && lam > 0) {
      Real xi1 = a * lam;
      Real xi2 = b * lam;
      Real xi3 = c * lam;
      psi_basis(triangle_information(i, 0)) = 2. * xi1 + xi2 + xi3 - 1.;
      psi_basis(triangle_information(i, 1)) = xi1 + 2. * xi2 + xi3 - 1.;
      psi_basis(triangle_information(i, 2)) = xi1 + xi2 + 2. * xi3 - 1.;

    }
  }

  DvceArray1D<Real> psi_basis_dvce;
  Kokkos::realloc(psi_basis_dvce, num_points);
  Kokkos::deep_copy(psi_basis_dvce, psi_basis);

  MatVecMultiply(mass_matrix_inv, psi_basis_dvce, beam_source_1_vals);

  if (num_beams > 1) {
    Kokkos::deep_copy(psi_basis, 0);

    x0 = Kokkos::sin(beam_source_2_theta) * Kokkos::cos(beam_source_2_phi);
    y0 = Kokkos::sin(beam_source_2_theta) * Kokkos::sin(beam_source_2_phi);
    z0 = Kokkos::cos(beam_source_2_theta);

    for (int i = 0; i < num_triangles; i++) {
      Real x1 = angular_grid_cartesian(triangle_information(i, 0), 0);
      Real y1 = angular_grid_cartesian(triangle_information(i, 0), 1);
      Real z1 = angular_grid_cartesian(triangle_information(i, 0), 2);

      Real x2 = angular_grid_cartesian(triangle_information(i, 1), 0);
      Real y2 = angular_grid_cartesian(triangle_information(i, 1), 1);
      Real z2 = angular_grid_cartesian(triangle_information(i, 1), 2);

      Real x3 = angular_grid_cartesian(triangle_information(i, 2), 0);
      Real y3 = angular_grid_cartesian(triangle_information(i, 2), 1);
      Real z3 = angular_grid_cartesian(triangle_information(i, 2), 2);

      Real a = (x3 * y2 * z0 - x2 * y3 * z0 - x3 * y0 * z2 + x0 * y3 * z2 + x2 * y0 * z3 - x0 * y2 * z3)
               / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3);
      Real b = (x3 * y1 * z0 - x1 * y3 * z0 - x3 * y0 * z1 + x0 * y3 * z1 + x1 * y0 * z3 - x0 * y1 * z3)
               / (-(x3 * y2 * z1) + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 - x2 * y1 * z3 + x1 * y2 * z3);
      Real c = (x2 * y1 * z0 - x1 * y2 * z0 - x2 * y0 * z1 + x0 * y2 * z1 + x1 * y0 * z2 - x0 * y1 * z2)
               / (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3);

      Real lam = 1. / (a + b + c);

      if (a >= 0 && b >= 0 && c >= 0 && lam > 0) {
        Real xi1 = a * lam;
        Real xi2 = b * lam;
        Real xi3 = c * lam;
        psi_basis(triangle_information(i, 0)) = 2. * xi1 + xi2 + xi3 - 1.;
        psi_basis(triangle_information(i, 1)) = xi1 + 2. * xi2 + xi3 - 1.;
        psi_basis(triangle_information(i, 2)) = xi1 + xi2 + 2. * xi3 - 1.;

      }
    }

    Kokkos::deep_copy(psi_basis_dvce, psi_basis);
    MatVecMultiply(mass_matrix_inv, psi_basis_dvce, beam_source_2_vals);

  }

}

}