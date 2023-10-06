//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_update.cpp
//  \brief Performs update of Radiation conserved variables (f0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.
//  Explicit (not implicit) radiation source terms are included in this update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int npts1 = num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage - 1];
  Real &gam1 = pdriver->gam1[stage - 1];
  Real beta_dt = (pdriver->beta[stage - 1]) * (pmy_pack->pmesh->dt);

  auto f0_ = f0;
  auto f1_ = f1;
  auto &flx1 = iflx.x1f;
  auto &flx2 = iflx.x2f;
  auto &flx3 = iflx.x3f;
  auto &L_mu_muhat0_ = L_mu_muhat0;
  auto &L_mu_muhat1_ = L_mu_muhat1;
  auto &u_mu_ = u_mu;
  auto &Gamma_ = Gamma;

  // update the distribution function for radiation
  par_for("radiation_femn_update", DevExeSpace(), 0, nmb1, 0, npts1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int enang, int k, int j, int i) {

            // Compute Christoeffel in fluid frame
            double Gamma_fluid = 0;

            RadiationFEMNPhaseIndices idcs = IndicesComponent(enang);
            int en = idcs.eindex;
            int B = idcs.angindex;
            auto Ven = (1. / 3.) * (pow(energy_grid(en + 1), 3) - pow(energy_grid(en), 3));

            Real divf_s = flx1(m, enang, k, j, i) / (2. * mbsize.d_view(m).dx1 * Ven);
            if (multi_d) {
              divf_s += flx2(m, enang, k, j, i) / (2. * mbsize.d_view(m).dx2 * Ven);
            }
            if (three_d) {
              divf_s += flx3(m, enang, k, j, i) / (2. * mbsize.d_view(m).dx3 * Ven);
            }

            if (!rad_source) {
              f0_(m, enang, k, j, i) = gam0 * f0_(m, enang, k, j, i) + gam1 * f1_(m, enang, k, j, i) - beta_dt * divf_s;
            } else {
              f0_(m, enang, k, j, i) = gam0 * f0_(m, enang, k, j, i) + gam1 * f1_(m, enang, k, j, i) - beta_dt * divf_s
                  + sqrt_det_g(m, k, j, i) * beta_dt * eta(m, k, j, i) * e_source(B);
                  //- sqrt_det_g(m, k, j, i) * beta_dt * (kappa_s(m, k, j, i) + kappa_a(m, k, j, i)) * f0_(m, enang, k, j, i);
            }
          });


  int scr_size = ScrArray2D<Real>::shmem_size(num_points, num_points) * 2;
  int scr_level = 0;
  par_for_outer("radiation_femn_update_matinv", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, npts1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int enang, int k, int j, int i) {

                  RadiationFEMNPhaseIndices idcs = IndicesComponent(enang);
                  int en = idcs.eindex;
                  int B = idcs.angindex;

                  //ScrArray2D<Real> Q_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points, num_points);
                  //ScrArray2D<Real> Qinv_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points, num_points);
                  DvceArray2D<Real> Q_matrix;
                  DvceArray2D<Real> Qinv_matrix;
                  Kokkos::realloc(Q_matrix, num_points, num_points);
                  Kokkos::realloc(Qinv_matrix, num_points, num_points);

                  par_for_inner(member, 0, num_points * num_points - 1, [&](const int idx) {
                    int row = int(idx / num_points);
                    int col = idx - row * num_points;
                    Q_matrix(row, col) = sqrt_det_g(m, k, j, i) * (L_mu_muhat0_(m, 0, 0, k, j, i) * P_matrix(0, row, col)
                        + L_mu_muhat0_(m, 0, 1, k, j, i) * P_matrix(1, row, col) + L_mu_muhat0_(m, 0, 2, k, j, i) * P_matrix(2, row, col)
                        + L_mu_muhat0_(m, 0, 3, k, j, i) * P_matrix(3, row, col)
                        + beta_dt * (kappa_s(m, k, j, i) + kappa_a(m, k, j, i)) * (row == col)
                        + beta_dt * (1./(4.*M_PI)) * S_source(row, col));
                  });
                  member.team_barrier();
                  radiationfemn::LUInverse(Q_matrix, Qinv_matrix);

                  Real final_result;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, num_points), [&](const int A, Real &partial_sum) {
                      partial_sum += Qinv_matrix(B,A) * f0_(m, en*num_points+A, k, j, i);
                  }, final_result);
                  member.team_barrier();

                  f0_(m, enang, k, j, i) = final_result;
                });

  // update the tetrad quantities
  par_for("radiation_femn_tetrad_update", DevExeSpace(), 0, nmb1, 0, 3, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int muhat, int k, int j, int i) {
            Real tetr_rhs =
                (u_mu_(m, 1, k, j, i) / u_mu_(m, 0, k, j, i)) * (L_mu_muhat1_(m, mu, muhat, k, j, i + 1) - L_mu_muhat1_(m, mu, muhat, k, j, i)) / mbsize.d_view(m).dx1;
            if (multi_d) {
              tetr_rhs +=
                  (u_mu_(m, 2, k, j, i) / u_mu_(m, 0, k, j, i)) * (L_mu_muhat1_(m, mu, muhat, k, j + 1, i) - L_mu_muhat1_(m, mu, muhat, k, j, i)) / mbsize.d_view(m).dx2;
            }
            if (three_d) {
              tetr_rhs +=
                  (u_mu_(m, 3, k, j, i) / u_mu_(m, 0, k, j, i)) * (L_mu_muhat1_(m, mu, muhat, k + 1, j, i) - L_mu_muhat1_(m, mu, muhat, k, j, i)) / mbsize.d_view(m).dx3;
            }
            L_mu_muhat0_(m, mu, muhat, k, j, i) = gam0 * L_mu_muhat0_(m, mu, muhat, k, j, i) + gam1 * L_mu_muhat1_(m, mu, muhat, k, j, i) - beta_dt * tetr_rhs;
          });

  // ------------------------------------------------------------------------------------------------------
  // Dummy output, Remove later
  // ------------------------------------------------------------------------------------------------------
  /*std::cout << "Metric:" << std::endl;
  std::cout << g_dd(0, 0, 0, 0, 4, 4) << " " << g_dd(0, 0, 1, 0, 4, 4) << " " << g_dd(0, 0, 2, 0, 4, 4) << " " << g_dd(0, 0, 3, 0, 4, 4) << std::endl;
  std::cout << g_dd(0, 1, 0, 0, 4, 4) << " " << g_dd(0, 1, 1, 0, 4, 4) << " " << g_dd(0, 1, 2, 0, 4, 4) << " " << g_dd(0, 1, 3, 0, 4, 4) << std::endl;
  std::cout << g_dd(0, 2, 0, 0, 4, 4) << " " << g_dd(0, 2, 1, 0, 4, 4) << " " << g_dd(0, 2, 2, 0, 4, 4) << " " << g_dd(0, 2, 3, 0, 4, 4) << std::endl;
  std::cout << g_dd(0, 3, 0, 0, 4, 4) << " " << g_dd(0, 3, 1, 0, 4, 4) << " " << g_dd(0, 3, 2, 0, 4, 4) << " " << g_dd(0, 3, 3, 0, 4, 4) << std::endl;
  std::cout << std::endl;
  std::cout << "L^mu_0: (" << L_mu_muhat0_(0, 0, 0, 0, 4, 4) << ", " << L_mu_muhat0_(0, 1, 0, 0, 4, 4) << ", " << L_mu_muhat0_(0, 2, 0, 0, 4, 4) << ", "
            << L_mu_muhat0_(0, 3, 0, 0, 4, 4) << ")" << std::endl;
  std::cout << "L^mu_1: (" << L_mu_muhat0_(0, 0, 1, 0, 4, 4) << ", " << L_mu_muhat0_(0, 1, 1, 0, 4, 4) << ", " << L_mu_muhat0_(0, 2, 1, 0, 4, 4) << ", "
            << L_mu_muhat0_(0, 3, 1, 0, 4, 4) << ")" << std::endl;
  std::cout << "L^mu_2: (" << L_mu_muhat0_(0, 0, 2, 0, 4, 4) << ", " << L_mu_muhat0_(0, 1, 2, 0, 4, 4) << ", " << L_mu_muhat0_(0, 2, 2, 0, 4, 4) << ", "
            << L_mu_muhat0_(0, 3, 2, 0, 4, 4) << ")" << std::endl;
  std::cout << "L^mu_3: (" << L_mu_muhat0_(0, 0, 3, 0, 4, 4) << ", " << L_mu_muhat0_(0, 1, 3, 0, 4, 4) << ", " << L_mu_muhat0_(0, 2, 3, 0, 4, 4) << ", "
            << L_mu_muhat0_(0, 3, 3, 0, 4, 4) << ")" << std::endl;
  std::cout << std::endl;*/
  // ------------------------------------------------------------------------------------------------------

  // Add explicit source terms
  if (beam_source) {
    // @TODO: Add beam source support
    //AddBeamSource(f0_);
  }

  return TaskStatus::complete;
}
} // namespace radiationfemn