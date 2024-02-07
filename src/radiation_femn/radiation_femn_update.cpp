//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_update.cpp
//  \brief Performs update of distribution function and tetrad

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_matinv.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage - 1];
  Real &gam1 = pdriver->gam1[stage - 1];
  Real beta_dt = (pdriver->beta[stage - 1]) * (pmy_pack->pmesh->dt);

  int &num_points_ = pmy_pack->pradfemn->num_points;
  int &num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  int &num_species_ = pmy_pack->pradfemn->num_species;
  int &num_points_total_ = pmy_pack->pradfemn->num_points_total;
  int num_species_energy = num_species_ * num_energy_bins_;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &f1_ = pmy_pack->pradfemn->f1;
  auto &energy_grid_ = pmy_pack->pradfemn->energy_grid;
  auto &flx1 = pmy_pack->pradfemn->iflx.x1f;
  auto &flx2 = pmy_pack->pradfemn->iflx.x2f;
  auto &flx3 = pmy_pack->pradfemn->iflx.x3f;
  auto &L_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  //auto &L_mu_muhat1_ = pmy_pack->pradfemn->L_mu_muhat1;
  //auto &u_mu_ = pmy_pack->pradfemn->u_mu;
  auto &sqrt_det_g_ = pmy_pack->pradfemn->sqrt_det_g;
  auto &eta_ = pmy_pack->pradfemn->eta;
  auto &e_source_ = pmy_pack->pradfemn->e_source;
  auto &kappa_s_ = pmy_pack->pradfemn->kappa_s;
  auto &kappa_a_ = pmy_pack->pradfemn->kappa_a;
  auto &F_matrix_ = pmy_pack->pradfemn->F_matrix;
  auto &G_matrix_ = pmy_pack->pradfemn->G_matrix;
  auto &energy_par_ = pmy_pack->pradfemn->energy_par;
  auto &P_matrix_ = pmy_pack->pradfemn->P_matrix;
  auto &S_source_ = pmy_pack->pradfemn->S_source;

  size_t scr_size = ScrArray2D<Real>::shmem_size(num_points_, num_points_) * 5 + ScrArray1D<Real>::shmem_size(num_points_) * 5
      + ScrArray1D<int>::shmem_size(num_points_ - 1) * 1 + +ScrArray1D<Real>::shmem_size(4 * 4 * 4) * 2;
  int scr_level = 0;

  par_for_outer("radiation_femn_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int k, int j, int i) {

                  DvceArray3D<Real> Gamma_fluid_udd;
                  Kokkos::realloc(Gamma_fluid_udd, 4, 4, 4);
                  Kokkos::deep_copy(Gamma_fluid_udd, 0.);

                  // compute matrices F Gam and G_Gam
                  ScrArray2D<Real> F_Gamma = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> G_Gamma = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);

                  par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;

                    Real sum_nuhatmuhat_f = 0.;
                    Real sum_nuhatmuhat_g = 0.;
                    for (int nuhatmuhat = 0; nuhatmuhat < 16; nuhatmuhat++) {
                      int nuhat = int(nuhatmuhat / 4);
                      int muhat = nuhatmuhat - nuhat * 4;

                      sum_nuhatmuhat_f += F_matrix_(nuhat, muhat, 0, row, col) * Gamma_fluid_udd(1, nuhat, muhat)
                          + F_matrix_(nuhat, muhat, 1, row, col) * Gamma_fluid_udd(2, nuhat, muhat)
                          + F_matrix_(nuhat, muhat, 2, row, col) * Gamma_fluid_udd(3, nuhat, muhat);

                      sum_nuhatmuhat_g += G_matrix_(nuhat, muhat, 0, row, col) * Gamma_fluid_udd(1, nuhat, muhat)
                          + G_matrix_(nuhat, muhat, 1, row, col) * Gamma_fluid_udd(2, nuhat, muhat)
                          + G_matrix_(nuhat, muhat, 2, row, col) * Gamma_fluid_udd(3, nuhat, muhat);
                    }
                    F_Gamma(row, col) = sum_nuhatmuhat_f;
                    G_Gamma(row, col) = sum_nuhatmuhat_g;
                  });
                  member.team_barrier();

                  // compute Lax-Friedrichs constant
                  Real K = 0.;
                  for (int idx = 0; idx < num_points_ * num_points_; idx++) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;

                    K += F_Gamma(row, col) * F_Gamma(row, col);
                  }
                  K = sqrt(K);

                  ScrArray1D<Real> g_rhs_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_total_);
                  par_for_inner(member, 0, num_points_total_ - 1, [&](const int nuenangidx) {

                    RadiationFEMNPhaseIndices indices = IndicesComponent(nuenangidx, num_points_, num_energy_bins_, num_species_);
                    int nu = indices.nuidx;
                    int B = indices.angidx;
                    int en = indices.enidx;

                    Real Ven = (1. / 3.) * (energy_grid_(en + 1) * energy_grid_(en + 1) * energy_grid_(en + 1) - energy_grid_(en) * energy_grid_(en) * energy_grid_(en));

                    Real divf_s = flx1(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx1 * Ven);

                    if (multi_d) {
                      divf_s += flx2(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx2 * Ven);
                    }

                    if (three_d) {
                      divf_s += flx3(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx3 * Ven);
                    }

                    g_rhs_scratch(nuenangidx) = gam0 * f0_(m, nuenangidx, k, j, i) + gam1 * f1_(m, nuenangidx, k, j, i) - beta_dt * divf_s
                        + sqrt_det_g_(m, k, j, i) * beta_dt * eta_(m, k, j, i) * e_source_(B) / Ven;

                    if (moving_medium || gravity) {
                      Real FG_Gamma_f_B = 0;
                      for (int A = 0; A < num_points_; A++) {
                        int nuenangA = IndicesUnited(nu, en, A, num_species_, num_energy_bins_, num_points_);
                        FG_Gamma_f_B += (gam0 * f0_(m, nuenangA, k, j, i) + gam1 * f1_(m, nuenangA, k, j, i)) * (F_Gamma(A, B) + G_Gamma(A, B));
                      }
                      g_rhs_scratch(nuenangidx) = g_rhs_scratch(nuenangidx) - FG_Gamma_f_B;
                    }

                    if (pmy_pack->pradfemn->num_energy_bins > 1) {
                      Real multi_energy_terms = 0;
                      for (int A = 0; A < num_points_; A++) {

                        int nuenangA_nm2 = IndicesUnited(nu, en - 2, A, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_nm1 = IndicesUnited(nu, en - 1, A, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_n = IndicesUnited(nu, en, A, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_np1 = IndicesUnited(nu, en + 1, A, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_np2 = IndicesUnited(nu, en, A, num_species_, num_energy_bins_, num_points_);

                        Real f_nm2 = (en - 2 >= 0 && en - 2 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_nm2, k, j, i) + gam1 * f1_(m, nuenangA_nm2, k, j, i) : 0.;
                        Real f_nm1 = (en - 1 >= 0 && en - 1 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_nm1, k, j, i) + gam1 * f1_(m, nuenangA_nm1, k, j, i) : 0.;
                        Real f_n = gam0 * f0_(m, nuenangA_n, k, j, i) + gam1 * f1_(m, nuenangA_n, k, j, i);
                        Real f_np1 = (en + 1 >= 0 && en + 1 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_np1, k, j, i) + gam1 * f1_(m, nuenangA_np1, k, j, i) : 0.;
                        Real f_np2 = (en + 2 >= 0 && en + 2 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_np2, k, j, i) + gam1 * f1_(m, nuenangA_np2, k, j, i) : 0.;

                        // {F^A} for n and n+1 th bin
                        Real f_term1_np1 = 0.5 * (f_np1 + f_n);
                        Real f_term1_n = 0.5 * (f_n + f_nm1);

                        // [[F^A]] for n and n+1 th bin
                        Real f_term2_np1 = f_n - f_np1;
                        Real f_term2_n = (f_nm1 - f_n);

                        // width of n-th energy bin
                        Real den = energy_grid_(en + 1) - energy_grid_(en);

                        Real Dmfn = (f_n - f_nm1) / den;
                        Real Dpfn = (f_np1 - f_n) / den;
                        Real Dfn = (f_np1 - f_nm1) / (2. * den);

                        Real Dmfnm1 = (f_nm1 - f_nm2) / den;
                        Real Dpfnm1 = (f_n - f_nm1) / den;
                        Real Dfnm1 = (f_n - f_nm2) / (2. * den);

                        Real Dmfnp1 = (f_np1 - f_n) / den;
                        Real Dpfnp1 = (f_np2 - f_np1) / den;
                        Real Dfnp1 = (f_np2 - f_n) / (2. * den);

                        Real theta_np12 = (Dfn < energy_par_ * den || Dmfn * Dpfn > 0.) ? 0. : 1.;
                        Real theta_nm12 = (Dfnm1 < energy_par_ * den || Dmfnm1 * Dpfnm1 > 0.) ? 0. : 1.;
                        Real theta_np32 = (Dfnp1 < energy_par_ * den || Dmfnp1 * Dpfnp1 > 0.) ? 0. : 1.;

                        Real theta_n = (theta_nm12 > theta_np12) ? theta_nm12 : theta_np12;
                        Real theta_np1 = (theta_np12 > theta_np32) ? theta_np12 : theta_np32;

                        multi_energy_terms +=
                            (energy_grid(en + 1) * energy_grid(en + 1) * energy_grid(en + 1) * (F_Gamma(A, B) * f_term1_np1 - theta_np1 * K * f_term2_np1 / 2.)
                                - energy_grid(en) * energy_grid(en) * energy_grid(en) * (F_Gamma(A, B) * f_term1_n - theta_n * K * f_term2_n / 2.));
                      }
                      g_rhs_scratch(nuenangidx) = g_rhs_scratch(nuenangidx) + multi_energy_terms;
                    }

                    f0_(m, nuenangidx, k, j, i) = g_rhs_scratch(nuenangidx);

                  });

                });
  /*
  par_for_outer("radiation_femn_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, num_species_energy - 1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int nuen, int k, int j, int i) {

                  int nu = int(nuen / num_energy_bins_);
                  int en = nuen - nu * num_energy_bins_;

                  AthenaScratchTensor<Real, TensorSymm::NONE, 3, 3> Gamma_fluid_udd;
                  for (int ihat = 0; ihat < 3; ihat++) {
                    for (int nuhat = 0; nuhat < 3; nuhat++) {
                      for (int muhat = 0; muhat < 3; muhat++) {
                        Gamma_fluid_udd(ihat, nuhat, muhat) = 0.;
                      }
                    }
                  }

                  // compute matrices F Gam and G_Gam
                  ScrArray2D<Real> F_Gamma = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> G_Gamma = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);

                  par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;

                    Real sum_nuhatmuhat_f = 0.;
                    Real sum_nuhatmuhat_g = 0.;
                    for (int nuhatmuhat = 0; nuhatmuhat < 16; nuhatmuhat++) {
                      int nuhat = int(nuhatmuhat / 4);
                      int muhat = nuhatmuhat - nuhat * 4;

                      sum_nuhatmuhat_f += F_matrix_(nuhat, muhat, 0, row, col) * Gamma_fluid_udd(1, nuhat, muhat)
                          + F_matrix_(nuhat, muhat, 1, row, col) * Gamma_fluid_udd(2, nuhat, muhat)
                          + F_matrix_(nuhat, muhat, 2, row, col) * Gamma_fluid_udd(3, nuhat, muhat);

                      sum_nuhatmuhat_g += G_matrix_(nuhat, muhat, 0, row, col) * Gamma_fluid_udd(1, nuhat, muhat)
                          + G_matrix_(nuhat, muhat, 1, row, col) * Gamma_fluid_udd(2, nuhat, muhat)
                          + G_matrix_(nuhat, muhat, 2, row, col) * Gamma_fluid_udd(3, nuhat, muhat);
                    }
                    F_Gamma(row, col) = sum_nuhatmuhat_f;
                    G_Gamma(row, col) = sum_nuhatmuhat_g;
                  });
                  member.team_barrier();

                  // compute Lax-Friedrichs constant
                  Real K = 0.;
                  for (int idx = 0; idx < num_points_ * num_points_; idx++) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;

                    K += F_Gamma(row, col) * F_Gamma(row, col);
                  }
                  K = sqrt(K);

                  // compute rhs
                  ScrArray1D<Real> g_rhs_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  auto Ven = (1. / 3.) * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3));

                  par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {
                    int nuenangidx = IndicesUnited(nu, en, idx, num_species_, num_energy_bins_, num_points_);

                    Real divf_s = flx1(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx1 * Ven);

                    if (multi_d) {
                      divf_s += flx2(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx2 * Ven);
                    }

                    if (three_d) {
                      divf_s += flx3(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx3 * Ven);
                    }

                    g_rhs_scratch(idx) = gam0 * f0_(m, nuenangidx, k, j, i) + gam1 * f1_(m, nuenangidx, k, j, i) - beta_dt * divf_s
                        + sqrt_det_g_(m, k, j, i) * beta_dt * eta_(m, k, j, i) * e_source_(idx) / Ven;

                    // add momentum derivative terms
                    if (moving_medium || gravity) {
                      Real FG_Gamma_f_idx = 0;
                      for (int angle = 0; angle < num_points_; angle++) {
                        int nuenangA = IndicesUnited(nu, en, angle, num_species_, num_energy_bins_, num_points_);
                        FG_Gamma_f_idx += (gam0 * f0_(m, nuenangA, k, j, i) + gam1 * f1_(m, nuenangA, k, j, i)) * (F_Gamma(angle, idx) + G_Gamma(angle, idx));
                      }
                      g_rhs_scratch(idx) = g_rhs_scratch(idx) - FG_Gamma_f_idx;
                    }

                    if (pmy_pack->pradfemn->num_energy_bins > 1) {
                      Real multi_energy_terms = 0;
                      for (int angle = 0; angle < num_points_; angle++) {

                        int nuenangA_nm2 = IndicesUnited(nu, en - 2, angle, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_nm1 = IndicesUnited(nu, en - 1, angle, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_n = IndicesUnited(nu, en, angle, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_np1 = IndicesUnited(nu, en + 1, angle, num_species_, num_energy_bins_, num_points_);
                        int nuenangA_np2 = IndicesUnited(nu, en, angle, num_species_, num_energy_bins_, num_points_);

                        Real f_nm2 = (en - 2 >= 0 && en - 2 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_nm2, k, j, i) + gam1 * f1_(m, nuenangA_nm2, k, j, i) : 0.;
                        Real f_nm1 = (en - 1 >= 0 && en - 1 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_nm1, k, j, i) + gam1 * f1_(m, nuenangA_nm1, k, j, i) : 0.;
                        Real f_n = gam0 * f0_(m, nuenangA_n, k, j, i) + gam1 * f1_(m, nuenangA_n, k, j, i);
                        Real f_np1 = (en + 1 >= 0 && en + 1 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_np1, k, j, i) + gam1 * f1_(m, nuenangA_np1, k, j, i) : 0.;
                        Real f_np2 = (en + 2 >= 0 && en + 2 < num_energy_bins_) ? gam0 * f0_(m, nuenangA_np2, k, j, i) + gam1 * f1_(m, nuenangA_np2, k, j, i) : 0.;

                        // {F^A} for n and n+1 th bin
                        Real f_term1_np1 = 0.5 * (f_np1 + f_n);
                        Real f_term1_n = 0.5 * (f_n + f_nm1);

                        // [[F^A]] for n and n+1 th bin
                        Real f_term2_np1 = f_n - f_np1;
                        Real f_term2_n = (f_nm1 - f_n);

                        // width of n-th energy bin
                        Real den = energy_grid_(en + 1) - energy_grid_(en);

                        Real Dmfn = (f_n - f_nm1) / den;
                        Real Dpfn = (f_np1 - f_n) / den;
                        Real Dfn = (f_np1 - f_nm1) / (2. * den);

                        Real Dmfnm1 = (f_nm1 - f_nm2) / den;
                        Real Dpfnm1 = (f_n - f_nm1) / den;
                        Real Dfnm1 = (f_n - f_nm2) / (2. * den);

                        Real Dmfnp1 = (f_np1 - f_n) / den;
                        Real Dpfnp1 = (f_np2 - f_np1) / den;
                        Real Dfnp1 = (f_np2 - f_n) / (2. * den);

                        Real theta_np12 = (Dfn < energy_par_ * den || Dmfn * Dpfn > 0.) ? 0. : 1.;
                        Real theta_nm12 = (Dfnm1 < energy_par_ * den || Dmfnm1 * Dpfnm1 > 0.) ? 0. : 1.;
                        Real theta_np32 = (Dfnp1 < energy_par_ * den || Dmfnp1 * Dpfnp1 > 0.) ? 0. : 1.;

                        Real theta_n = (theta_nm12 > theta_np12) ? theta_nm12 : theta_np12;
                        Real theta_np1 = (theta_np12 > theta_np32) ? theta_np12 : theta_np32;

                        multi_energy_terms +=
                            (energy_grid(en + 1) * energy_grid(en + 1) * energy_grid(en + 1) * (F_Gamma(angle, idx) * f_term1_np1 - theta_np1 * K * f_term2_np1 / 2.)
                                - energy_grid(en) * energy_grid(en) * energy_grid(en) * (F_Gamma(angle, idx) * f_term1_n - theta_n * K * f_term2_n / 2.));
                      }
                      g_rhs_scratch(idx) = g_rhs_scratch(idx) + multi_energy_terms;
                    }

                  });
                  member.team_barrier();

                  ScrArray2D<Real> Q_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> Qinv_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> lu_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray1D<Real> x_array = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> b_array = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<int> pivots = ScrArray1D<int>(member.team_scratch(scr_level), num_points_ - 1);

                  par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;
                    Q_matrix(row, col) = sqrt_det_g_(m, k, j, i) * (L_mu_muhat0_(m, 0, 0, k, j, i) * P_matrix_(0, row, col)
                        + L_mu_muhat0_(m, 0, 1, k, j, i) * P_matrix_(1, row, col) + L_mu_muhat0_(m, 0, 2, k, j, i) * P_matrix_(2, row, col)
                        + L_mu_muhat0_(m, 0, 3, k, j, i) * P_matrix_(3, row, col))
                        + sqrt_det_g_(m, k, j, i) * beta_dt * (kappa_s_(m, k, j, i) + kappa_a_(m, k, j, i)) * (row == col) / Ven
                        - sqrt_det_g_(m, k, j, i) * beta_dt * (1. / (4. * M_PI)) * kappa_s_(m, k, j, i) * S_source_(row, col) / Ven;
                    lu_matrix(row, col) = Q_matrix(row, col);
                  });
                  member.team_barrier();

                  radiationfemn::LUInv<ScrArray2D<Real>, ScrArray1D<Real>, ScrArray1D<int>>(member, Q_matrix, Qinv_matrix, lu_matrix, x_array, b_array, pivots);
                  member.team_barrier();

                  par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {

                    Real final_result = 0.;
                    for (int A = 0; A < num_points_; A++) {
                      final_result += Qinv_matrix(idx, A) * g_rhs_scratch(A);
                    }

                    auto unifiedidx = IndicesUnited(nu, en, idx, num_species_, num_energy_bins_, num_points_);
                    f0_(m, unifiedidx, k, j, i) = final_result;
                  });

                }); */

// update the tetrad quantities
/*
par_for("radiation_femn_tetrad_update", DevExeSpace(), 0, nmb1, 0, 3, 1, 3, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int mu, int muhat, int k, int j, int i) {
          Real tetr_rhs =
              (u_mu_(m, 1, k, j, i) / u_mu_(m, 0, k, j, i)) * (L_mu_muhat1_(m, mu, muhat, k, j, i + 1) - L_mu_muhat1_(m, mu, muhat, k, j, i))
                  / mbsize.d_view(m).dx1;
          if (multi_d) {
            tetr_rhs +=
                (u_mu_(m, 2, k, j, i) / u_mu_(m, 0, k, j, i)) * (L_mu_muhat1_(m, mu, muhat, k, j + 1, i) - L_mu_muhat1_(m, mu, muhat, k, j, i))
                    / mbsize.d_view(m).dx2;
          }
          if (three_d) {
            tetr_rhs +=
                (u_mu_(m, 3, k, j, i) / u_mu_(m, 0, k, j, i)) * (L_mu_muhat1_(m, mu, muhat, k + 1, j, i) - L_mu_muhat1_(m, mu, muhat, k, j, i))
                    / mbsize.d_view(m).dx3;
          }
          L_mu_muhat0_(m, mu, muhat, k, j, i) =
              gam0 * L_mu_muhat0_(m, mu, muhat, k, j, i) + gam1 * L_mu_muhat1_(m, mu, muhat, k, j, i) - beta_dt * tetr_rhs;
        }); */

  return
      TaskStatus::complete;
}
} // namespace radiationfemn