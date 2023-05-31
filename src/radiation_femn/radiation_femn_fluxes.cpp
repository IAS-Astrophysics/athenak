//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = num_points - 1;
  int npts1 = num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &mm_ = mass_matrix;
  auto &stiffnessx_ = stiffness_matrix_x;
  auto &stiffnessy_ = stiffness_matrix_y;
  auto &stiffnessz_ = stiffness_matrix_z;
  auto &stiffnessmodx_ = stiffness_matrix_x;
  auto &stiffnessmody_ = stiffness_matrix_y;
  auto &stiffnessmodz_ = stiffness_matrix_z;

  auto &f0_ = f0;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &flx1 = iflx.x1f;
  Kokkos::deep_copy(flx1, 0.);
  par_for("radiation_femn_flux_x", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, int(ie / 2) + 1, 0, npts1, 0,
          nang1, 0, 3,
          KOKKOS_LAMBDA(const int m,
                        const int k,
                        const int j,
                        const int i,
                        const int enang,
                        const int A,
                        const int muhat) {

            RadiationFEMNPhaseIndices idcs = Indices(enang);
            int en = idcs.eindex;
            int B = idcs.angindex;

            auto kk = k;
            auto jj = j;
            auto ii = 2 * i - 2;
            auto Ven = (1. / 3.) * (pow(energy_grid(en), 3) - pow(energy_grid(en - 1), 3));
            auto Favg = (0.5) * Ven *
                (P_matrix(muhat, A, B) * f0_(m, enang, kk, jj, ii) *
                    L_mu_muhat0(m, 1, muhat, kk, jj, ii) * sqrt_det_g(m, kk, jj, ii) +
                    P_matrix(muhat, A, B) * f0_(m, enang, kk, jj, ii + 1) *
                        L_mu_muhat0(m, 1, muhat, kk, jj, ii + 1) * sqrt_det_g(m, kk, jj, ii + 1));

            auto Fminus = (0.5) * (1.5 * sqrt_det_g(m, kk, jj, ii) * L_mu_muhat0(m, 1, muhat, kk, jj, ii) -
                0.5 * sqrt_det_g(m, kk, jj, ii + 1) *
                    L_mu_muhat0(m, 1, muhat, kk, jj, ii + 1)) * (P_matrix(muhat, A, B) *
                ((1.5) *
                    f0_(m, enang, kk, jj, ii) -
                    (0.5) *
                        f0_(m, enang, kk, jj, ii + 1) +
                    (1.5) *
                        f0_(m, enang, kk, jj, ii - 1) -
                    (0.5) *
                        f0_(m, enang, kk, jj, ii - 2))
                + std::signbit(
                    1.5 * L_mu_muhat0(m, 1, muhat, kk, jj, ii) -
                        0.5 * L_mu_muhat0(m, 1, muhat, kk, jj, ii + 1)) * P_matrix(muhat, A, B) *
                    ((1.5) * f0_(m, enang, kk, jj,
                                 ii - 1) - (0.5) *
                        f0_(m,
                            enang,
                            kk,
                            jj,
                            ii -
                                2) -
                        (1.5) *
                            f0_(m, enang, kk, jj, ii) +
                        (0.5) * f0_(m, enang, kk, jj,
                                    ii + 1)));

            auto Fplus = (0.5) * (-0.5 * sqrt_det_g(m, kk, jj, ii) * L_mu_muhat0(m, 1, muhat, kk, jj, ii) +
                1.5 * sqrt_det_g(m, kk, jj, ii + 1) *
                    L_mu_muhat0(m, 1, muhat, kk, jj, ii + 1)) * (P_matrix(muhat, A, B) *
                ((1.5) *
                    f0_(m, enang, kk, jj, ii + 2) -
                    (0.5) *
                        f0_(m, enang, kk, jj, ii + 3) +
                    (1.5) *
                        f0_(m, enang, kk, jj, ii + 1) -
                    (0.5) *
                        f0_(m, enang, kk, jj, ii))
                + std::signbit(
                    -0.5 * L_mu_muhat0(m, 1, muhat, kk, jj, ii) +
                        1.5 * L_mu_muhat0(m, 1, muhat, kk, jj, ii + 1)) * P_matrix(muhat, A, B) *
                    ((1.5) * f0_(m, enang, kk, jj,
                                 ii + 1) - (0.5) *
                        f0_(m,
                            enang,
                            kk,
                            jj,
                            ii) -
                        (1.5) * f0_(m, enang, kk, jj,
                                    ii + 2) +
                        (0.5) * f0_(m, enang, kk, jj,
                                    ii + 3)));

            int enangindex = en * num_points + A;
            flx1(m, enangindex, kk, jj, ii) += ((1.5) * Fminus - Favg - (0.5) * Fplus) / (2.0);
            flx1(m, enangindex, kk, jj, ii + 1) += ((0.5) * Fminus + Favg - (1.5) * Fplus) / (2.0);
          });


  //--------------------------------------------------------------------------------------
  // j-direction

  auto &flx2 = iflx.x2f;
  Kokkos::deep_copy(flx2, 0.);
  if (multi_d) {
    par_for("radiation_femn_flux_y", DevExeSpace(), 0, nmb1, ks, ke, js, int(je / 2) + 1, is, ie, 0, npts1, 0,
            nang1, 0, 3,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int enang,
                          const int A, const int muhat) {

              RadiationFEMNPhaseIndices idcs = this->Indices(enang);
              int en = idcs.eindex;
              int B = idcs.angindex;

              auto kk = k;
              auto jj = 2 * j - 2;
              auto ii = i;

              auto Ven = (1. / 3.) * (pow(energy_grid(en), 3) - pow(energy_grid(en - 1), 3));
              auto Favg = (0.5) * Ven *
                  (P_matrix(muhat, A, B) * f0_(m, enang, kk, jj, ii) *
                      L_mu_muhat0(m, 2, muhat, kk, jj, ii) * sqrt_det_g(m, kk, jj, ii) +
                      P_matrix(muhat, A, B) * f0_(m, enang, kk, jj + 1, ii) *
                          L_mu_muhat0(m, 2, muhat, kk, jj + 1, ii) * sqrt_det_g(m, kk, jj + 1, ii));

              auto Fminus = (0.5) * (1.5 * sqrt_det_g(m, kk, jj, ii) * L_mu_muhat0(m, 2, muhat, kk, jj, ii) -
                  0.5 * sqrt_det_g(m, kk, jj + 1, ii) *
                      L_mu_muhat0(m, 2, muhat, kk, jj + 1, ii)) * (P_matrix(muhat, A, B) *
                  ((1.5) *
                      f0_(m, enang, kk, jj, ii) -
                      (0.5) *
                          f0_(m, enang, kk, jj + 1,
                              ii) +
                      (1.5) *
                          f0_(m, enang, kk, jj - 1,
                              ii) -
                      (0.5) *
                          f0_(m, enang, kk, jj - 2,
                              ii))
                  + std::signbit(
                      1.5 * L_mu_muhat0(m, 2, muhat, kk, jj, ii) -
                          0.5 * L_mu_muhat0(m, 2, muhat, kk, jj + 1, ii)) * P_matrix(muhat, A, B) *
                      ((1.5) *
                          f0_(m, enang, kk, jj - 1,
                              ii) - (0.5) *
                          f0_(m, enang, kk,
                              jj - 2,
                              ii) -
                          (1.5) *
                              f0_(m, enang, kk, jj,
                                  ii) +
                          (0.5) *
                              f0_(m, enang, kk, jj + 1,
                                  ii)));

              auto Fplus = (0.5) * (-0.5 * sqrt_det_g(m, kk, jj, ii) * L_mu_muhat0(m, 2, muhat, kk, jj, ii) +
                  1.5 * sqrt_det_g(m, kk, jj + 1, ii) *
                      L_mu_muhat0(m, 2, muhat, kk, jj + 1, ii)) * (P_matrix(muhat, A, B) *
                  ((1.5) *
                      f0_(m, enang, kk, jj + 2,
                          ii) -
                      (0.5) *
                          f0_(m, enang, kk, jj + 3,
                              ii) +
                      (1.5) *
                          f0_(m, enang, kk, jj + 1,
                              ii) -
                      (0.5) *
                          f0_(m, enang, kk, jj, ii))
                  + std::signbit(
                      -0.5 * L_mu_muhat0(m, 2, muhat, kk, jj, ii) +
                          1.5 * L_mu_muhat0(m, 2, muhat, kk, jj + 1, ii)) * P_matrix(muhat, A, B) *
                      ((1.5) *
                          f0_(m, enang, kk, jj + 1,
                              ii) - (0.5) *
                          f0_(m, enang, kk,
                              jj,
                              ii) -
                          (1.5) *
                              f0_(m, enang, kk, jj + 2,
                                  ii) +
                          (0.5) *
                              f0_(m, enang, kk, jj + 3,
                                  ii)));

              int enangindex = en * num_points + A;
              flx2(m, enangindex, kk, jj, ii) += ((1.5) * Fminus - Favg - (0.5) * Fplus) / (2.0);
              flx2(m, enangindex, kk, jj + 1, ii) += ((0.5) * Fminus + Favg - (1.5) * Fplus) / (2.0);

            });
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  auto &flx3 = iflx.x3f;
  Kokkos::deep_copy(flx3, 0.);
  if (three_d) {
    par_for("radiation_femn_flux_z", DevExeSpace(), 0, nmb1, ks, int(ke / 2) + 1, js, je, is, ie, 0, npts1, 0,
            nang1, 0, 3,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int enang, const int A,
                          const int muhat) {

              RadiationFEMNPhaseIndices idcs = this->Indices(enang);
              int en = idcs.eindex;
              int B = idcs.angindex;

              auto kk = 2 * k - 1;
              auto jj = j;
              auto ii = i;

              auto Ven = (1. / 3.) * (pow(energy_grid(en), 3) - pow(energy_grid(en - 1), 3));
              auto Favg = (0.5) * Ven *
                  (P_matrix(muhat, A, B) * f0_(m, enang, kk, jj, ii) *
                      L_mu_muhat0(m, 3, muhat, kk, jj, ii) * sqrt_det_g(m, kk, jj, ii) +
                      P_matrix(muhat, A, B) * f0_(m, enang, kk + 1, jj, ii) *
                          L_mu_muhat0(m, 3, muhat, kk + 1, jj, ii) * sqrt_det_g(m, kk + 1, jj, ii));

              auto Fminus = (0.5) * (1.5 * sqrt_det_g(m, kk, jj, ii) * L_mu_muhat0(m, 3, muhat, kk, jj, ii) -
                  0.5 * sqrt_det_g(m, kk + 1, jj, ii) *
                      L_mu_muhat0(m, 3, muhat, kk + 1, jj, ii)) * (P_matrix(muhat, A, B) *
                  ((1.5) *
                      f0_(m, enang, kk, jj, ii) -
                      (0.5) *
                          f0_(m, enang, kk + 1, jj, ii) +
                      (1.5) *
                          f0_(m, enang, kk - 1, jj, ii) -
                      (0.5) *
                          f0_(m, enang, kk - 2, jj, ii))
                  + std::signbit(
                      1.5 * L_mu_muhat0(m, 3, muhat, kk, jj, ii) -
                          0.5 * L_mu_muhat0(m, 3, muhat, kk + 1, jj, ii)) * P_matrix(muhat, A, B) *
                      ((1.5) * f0_(m, enang, kk - 1, jj,
                                   ii) - (0.5) *
                          f0_(m,
                              enang,
                              kk - 2,
                              jj,
                              ii) -
                          (1.5) *
                              f0_(m, enang, kk, jj, ii) +
                          (0.5) * f0_(m, enang, kk + 1, jj,
                                      ii)));

              auto Fplus = (0.5) * (-0.5 * sqrt_det_g(m, kk, jj, ii) * L_mu_muhat0(m, 3, muhat, kk, jj, ii) +
                  1.5 * sqrt_det_g(m, kk + 1, jj, ii) *
                      L_mu_muhat0(m, 3, muhat, kk + 1, jj, ii)) * (P_matrix(muhat, A, B) *
                  ((1.5) *
                      f0_(m, enang, kk + 2, jj, ii) -
                      (0.5) *
                          f0_(m, enang, kk + 3, jj, ii) +
                      (1.5) *
                          f0_(m, enang, kk + 1, jj, ii) -
                      (0.5) *
                          f0_(m, enang, kk, jj, ii))
                  + std::signbit(
                      -0.5 * L_mu_muhat0(m, 3, muhat, kk, jj, ii) +
                          1.5 * L_mu_muhat0(m, 3, muhat, kk + 1, jj, ii)) * P_matrix(muhat, A, B) *
                      ((1.5) * f0_(m, enang, kk + 1, jj,
                                   ii) - (0.5) *
                          f0_(m,
                              enang,
                              kk,
                              jj,
                              ii) -
                          (1.5) * f0_(m, enang, kk + 2, jj,
                                      ii) +
                          (0.5) * f0_(m, enang, kk + 3, jj,
                                      ii)));

              int enangindex = en * num_points + A;
              flx1(m, enangindex, kk, jj, ii) += ((1.5) * Fminus - Favg - (0.5) * Fplus) / (2.0);
              flx1(m, enangindex, kk + 1, jj, ii) += ((0.5) * Fminus + Favg - (1.5) * Fplus) / (2.0);

            });
  }

  return TaskStatus::complete;
}

} // namespace radiationfemn