//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
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
#include "radiation_femn/radiation_femn_matinv.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {
  const int NGHOST = 2;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  //int npts1 = num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage - 1];
  Real &gam1 = pdriver->gam1[stage - 1];
  Real beta_dt = (pdriver->beta[stage - 1]) * (pmy_pack->pmesh->dt);

  //int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  //int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
  //int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;

  int &num_points_ = pmy_pack->pradfemn->num_points;
  int &num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  int &num_species_ = pmy_pack->pradfemn->num_species;
  int num_species_energy = num_species_ * num_energy_bins_;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &f1_ = pmy_pack->pradfemn->f1;
  auto &energy_grid_ = pmy_pack->pradfemn->energy_grid;
  auto &flx1 = pmy_pack->pradfemn->iflx.x1f;
  auto &flx2 = pmy_pack->pradfemn->iflx.x2f;
  auto &flx3 = pmy_pack->pradfemn->iflx.x3f;
  auto &L_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  auto &u_mu_ = pmy_pack->pradfemn->u_mu;
  auto &eta_ = pmy_pack->pradfemn->eta;
  auto &e_source_ = pmy_pack->pradfemn->e_source;
  auto &kappa_s_ = pmy_pack->pradfemn->kappa_s;
  auto &kappa_a_ = pmy_pack->pradfemn->kappa_a;
  auto &F_matrix_ = pmy_pack->pradfemn->F_matrix;
  auto &G_matrix_ = pmy_pack->pradfemn->G_matrix;
  auto &energy_par_ = pmy_pack->pradfemn->energy_par;
  auto &P_matrix_ = pmy_pack->pradfemn->P_matrix;
  auto &S_source_ = pmy_pack->pradfemn->S_source;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  size_t scr_size = ScrArray2D<Real>::shmem_size(num_points_, num_points_) * 5 + ScrArray1D<Real>::shmem_size(num_points_) * 5
      + ScrArray1D<int>::shmem_size(num_points_ - 1) * 1 + +ScrArray1D<Real>::shmem_size(4 * 4 * 4) * 2;
  int scr_level = 0;
  par_for_outer("radiation_femn_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, num_species_energy - 1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int nuen, int k, int j, int i) {

                  int nu = int(nuen / num_energy_bins_);
                  int en = nuen - nu * num_energy_bins_;

                  // metric and inverse metric
                  Real g_dd[16];
                  Real g_uu[16];
                  adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                       adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                       adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);
                  adm::SpacetimeUpperMetric(adm.alpha(m, k, j, i),
                                            adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                            adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_uu);
                  Real sqrt_det_g_ijk = adm.alpha(m, k, j, i) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                                                                                     adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                                                                                     adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)));

                  // derivative terms
                  ScrArray1D<Real> g_rhs_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  auto Ven = (1. / 3.) * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3));

                  par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {
                    int nuenangidx = IndicesUnited(nu, en, idx, num_species_, num_energy_bins_, num_points_);

                    Real divf_s = flx1(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx1);

                    if (multi_d) {
                      divf_s += flx2(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx2);
                    }

                    if (three_d) {
                      divf_s += flx3(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx3);
                    }

                    g_rhs_scratch(idx) = gam0 * f0_(m, nuenangidx, k, j, i) + gam1 * f1_(m, nuenangidx, k, j, i) - beta_dt * divf_s
                        + sqrt_det_g_ijk * beta_dt * eta_(m, k, j, i) * e_source_(idx) / Ven;

                  });
                  member.team_barrier();

                  Real idx[] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2, 1 / mbsize.d_view(m).dx3};

                  // lapse derivatives (\p_mu alpha)
                  Real dtalpha_d = 0.;    // time derivatives, get from z4c
                  AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d; // spatial derivatives
                  dalpha_d(0) = Dx<NGHOST>(0, idx, adm.alpha, m, k, j, i);
                  dalpha_d(1) = (multi_d) ? Dx<NGHOST>(1, idx, adm.alpha, m, k, j, i) : 0.;
                  dalpha_d(2) = (three_d) ? Dx<NGHOST>(2, idx, adm.alpha, m, k, j, i) : 0.;

                  // shift derivatives (\p_mu beta^i)
                  Real dtbetax_du = 0.; // time derivatives, get from z4c
                  Real dtbetay_du = 0.;
                  Real dtbetaz_du = 0.;
                  AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du; // spatial derivatives
                  for (int a = 0; a < 3; ++a) {
                    dbeta_du(0, a) = Dx<NGHOST>(0, idx, adm.beta_u, m, a, k, j, i);
                    dbeta_du(1, a) = (multi_d) ? Dx<NGHOST>(1, idx, adm.beta_u, m, a, k, j, i) : 0.;
                    dbeta_du(1, a) = (three_d) ? Dx<NGHOST>(1, idx, adm.beta_u, m, a, k, j, i) : 0.;
                  }

                  // covariant shift (beta_i)
                  Real betax_d = adm.g_dd(m, 0, 0, k, j, i) * adm.beta_u(m, 0, k, j, i) + adm.g_dd(m, 0, 1, k, j, i) * adm.beta_u(m, 1, k, j, i)
                      + adm.g_dd(m, 0, 2, k, j, i) * adm.beta_u(m, 2, k, j, i);
                  Real betay_d = adm.g_dd(m, 1, 0, k, j, i) * adm.beta_u(m, 0, k, j, i) + adm.g_dd(m, 1, 1, k, j, i) * adm.beta_u(m, 1, k, j, i)
                      + adm.g_dd(m, 1, 2, k, j, i) * adm.beta_u(m, 2, k, j, i);
                  Real betaz_d = adm.g_dd(m, 2, 0, k, j, i) * adm.beta_u(m, 0, k, j, i) + adm.g_dd(m, 2, 1, k, j, i) * adm.beta_u(m, 1, k, j, i)
                      + adm.g_dd(m, 2, 2, k, j, i) * adm.beta_u(m, 2, k, j, i);

                  // derivatives of spatial metric (\p_mu g_ij)
                  AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> dtg_dd;
                  AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
                  for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                      dtg_dd(a, b) = 0.; // time derivatives, get from z4c

                      dg_ddd(0, a, b) = Dx<NGHOST>(0, idx, adm.g_dd, m, a, b, k, j, i); // spatial derivatives
                      dg_ddd(1, a, b) = (multi_d) ? Dx<NGHOST>(1, idx, adm.g_dd, m, a, b, k, j, i) : 0.;
                      dg_ddd(2, a, b) = (three_d) ? Dx<NGHOST>(2, idx, adm.g_dd, m, a, b, k, j, i) : 0.;

                    }
                  }

                  // derivatives of the 4-metric: time derivatives
                  AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> dg4_ddd; //f
                  dg4_ddd(0, 0, 0) = -2. * adm.alpha(m, k, j, i) * dtalpha_d + 2. * betax_d * dtbetax_du + 2. * betay_d * dtbetay_du + 2. * betaz_d * dtbetaz_du
                      + dtg_dd(0, 0) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 0, k, j, i) + 2. * dtg_dd(0, 1) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 1, k, j, i)
                      + 2. * dtg_dd(0, 2) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 2, k, j, i) + dtg_dd(1, 1) * adm.beta_u(m, 1, k, j, i) * adm.beta_u(m, 1, k, j, i)
                      + 2. * dtg_dd(1, 2) * adm.beta_u(m, 1, k, j, i) * adm.beta_u(m, 2, k, j, i) + dtg_dd(2, 2) * adm.beta_u(m, 2, k, j, i) * adm.beta_u(m, 2, k, j, i);
                  for (int a = 1; a < 4; ++a) {
                    dg4_ddd(0, a, 0) = adm.g_dd(m, 0, 0, k, j, i) * dtbetax_du + adm.g_dd(m, 0, 1, k, j, i) * dtbetay_du + adm.g_dd(m, 0, 2, k, j, i) * dtbetaz_du
                        + dtg_dd(a - 1, 0) * adm.beta_u(m, 0, k, j, i) + dtg_dd(a - 1, 1) * adm.beta_u(m, 1, k, j, i) + dtg_dd(a - 1, 2) * adm.beta_u(m, 2, k, j, i);
                  }
                  for (int a = 1; a < 4; ++a) {
                    for (int b = 1; b < 4; ++b) {
                      dg4_ddd(0, a, b) = 0.; // time derivatives, get from z4c
                    }
                  }

                  // derivatives of the 4-metric: spatial derivatives
                  for (int a = 1; a < 4; ++a) {
                    for (int b = 1; b < 4; ++b) {

                      dg4_ddd(1, a, b) = Dx<NGHOST>(0, idx, adm.g_dd, m, a - 1, b - 1, k, j, i);
                      dg4_ddd(2, a, b) = (multi_d) ? Dx<NGHOST>(1, idx, adm.g_dd, m, a - 1, b - 1, k, j, i) : 0.;
                      dg4_ddd(3, a, b) = (three_d) ? Dx<NGHOST>(2, idx, adm.g_dd, m, a - 1, b - 1, k, j, i) : 0.;

                      dg4_ddd(a, 0, b) = adm.g_dd(m, 0, 0, k, j, i) * dbeta_du(a - 1, 0) + adm.g_dd(m, 0, 1, k, j, i) * dbeta_du(a - 1, 1)
                          + adm.g_dd(m, 0, 2, k, j, i) * dbeta_du(a - 1, 2) + dg_ddd(a - 1, 0, b - 1) * adm.beta_u(m, 0, k, j, i)
                          + dg_ddd(a - 1, 1, b - 1) * adm.beta_u(m, 1, k, j, i) + dg_ddd(a - 1, 2, b - 1) * adm.beta_u(m, 2, k, j, i);
                    }
                    dg4_ddd(a, 0, 0) = -2. * adm.alpha(m, k, j, i) * dalpha_d(a - 1) + 2. * betax_d * dbeta_du(a - 1, 0) + 2. * betay_d * dbeta_du(a - 1, 1)
                        + 2. * betaz_d * dbeta_du(a - 1, 2) + dtg_dd(0, 0) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 0, k, j, i)
                        + 2. * dg_ddd(a - 1, 0, 1) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 1, k, j, i)
                        + 2. * dg_ddd(a - 1, 0, 2) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 2, k, j, i)
                        + dg_ddd(a - 1, 1, 1) * adm.beta_u(m, 1, k, j, i) * adm.beta_u(m, 1, k, j, i)
                        + 2. * dg_ddd(a - 1, 1, 2) * adm.beta_u(m, 1, k, j, i) * adm.beta_u(m, 2, k, j, i)
                        + dg_ddd(a - 1, 2, 2) * adm.beta_u(m, 2, k, j, i) * adm.beta_u(m, 2, k, j, i);
                  }

                  // Christoeffel symbols
                  AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> Gamma_udd;
                  for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                      for (int c = 0; c < 4; ++c) {
                        Gamma_udd(a, b, c) = 0.0;
                        for (int d = 0; d < 4; ++d) {
                          Gamma_udd(a, b, c) += 0.5 * g_uu[a + 4 * d] * ((dg4_ddd(b, d, c)) + (dg4_ddd(c, b, d)) - (dg4_ddd(d, b, c)));
                        }
                      }
                    }
                  }

                  // Ricci rotation coefficients
                  AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> Gamma_fluid_udd;
                  for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                      for (int c = 0; c < 4; ++c) {
                        Gamma_fluid_udd(a, b, c) = 0.0;
                        for (int d = 0; d < 64; ++d) {
                          int a_idx = int(d / (4 * 4));
                          int b_idx = int((d - 4 * 4 * a_idx) / 4);
                          int c_idx = d - a_idx * 4 * 4 - b_idx * 4;

                          Real l_sign = (a == 0) ? -1. : +1.;
                          Real L_ahat_aidx = l_sign * (g_dd[a_idx + 4 * 0] * L_mu_muhat0_(m, 0, a, k, j, i) + g_dd[a_idx + 4 * 1] * L_mu_muhat0_(m, 1, a, k, j, i)
                              + g_dd[a_idx + 4 * 2] * L_mu_muhat0_(m, 2, a, k, j, i) + g_dd[a_idx + 4 * 3] * L_mu_muhat0_(m, 3, a, k, j, i));
                          Gamma_fluid_udd(a, b, c) +=
                              L_mu_muhat0_(m, b_idx, b, k, j, i) * L_mu_muhat0_(m, c_idx, c, k, j, i) * L_ahat_aidx * Gamma_udd(a_idx, b_idx, c_idx);
                        }

                        for (int a_idx = 0; a_idx < 4; ++a_idx) {

                          Real l_sign = (a == 0) ? -1. : +1.;
                          Real L_ahat_aidx = l_sign * (g_dd[a_idx + 4 * 0] * L_mu_muhat0_(m, 0, a, k, j, i) + g_dd[a_idx + 4 * 1] * L_mu_muhat0_(m, 1, a, k, j, i)
                              + g_dd[a_idx + 4 * 2] * L_mu_muhat0_(m, 2, a, k, j, i) + g_dd[a_idx + 4 * 3] * L_mu_muhat0_(m, 3, a, k, j, i));

                          Gamma_fluid_udd(a, b, c) +=
                              L_ahat_aidx * (L_mu_muhat0_(m, 1, c, k, j, i) - (u_mu_(m, 1, k, j, i) / u_mu_(m, 0, k, j, i)) * L_mu_muhat0_(m, 0, c, k, j, i))
                                  * Dx<NGHOST>(0, idx, L_mu_muhat0_, m, a_idx, c, k, j, i);

                          if (multi_d) {
                            Gamma_fluid_udd(a, b, c) +=
                                L_ahat_aidx * (L_mu_muhat0_(m, 2, c, k, j, i) - (u_mu_(m, 2, k, j, i) / u_mu_(m, 0, k, j, i)) * L_mu_muhat0_(m, 0, c, k, j, i))
                                    * Dx<NGHOST>(1, idx, L_mu_muhat0_, m, a_idx, c, k, j, i);
                          }

                          if (three_d) {
                            Gamma_fluid_udd(a, b, c) +=
                                L_ahat_aidx * (L_mu_muhat0_(m, 3, c, k, j, i) - (u_mu_(m, 3, k, j, i) / u_mu_(m, 0, k, j, i)) * L_mu_muhat0_(m, 0, c, k, j, i))
                                    * Dx<NGHOST>(2, idx, L_mu_muhat0_, m, a_idx, c, k, j, i);
                          }
                        }
                      }
                    }
                  }

                  // Compute F Gam and G Gam matrices
                  ScrArray2D<Real> F_Gamma_AB = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> G_Gamma_AB = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);

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
                    F_Gamma_AB(row, col) = sum_nuhatmuhat_f;
                    G_Gamma_AB(row, col) = sum_nuhatmuhat_g;
                  });
                  member.team_barrier();

                  // Add Christoeffel terms to rhs and compute Lax Friedrich's const K
                  Real K = 0.;
                  for (int idx = 0; idx < num_points_ * num_points_; idx++) {
                    int idx_b = int(idx / num_points_);
                    int idx_a = idx - idx_b * num_points_;

                    int idx_united = IndicesUnited(nu, en, idx_a, num_species_, num_energy_bins_, num_points_);

                    g_rhs_scratch(idx_b) -=
                        (F_Gamma_AB(idx_b, idx_a) + G_Gamma_AB(idx_b, idx_a)) * (gam0 * f0_(m, idx_united, k, j, i) + gam1 * f1_(m, idx_united, k, j, i));

                    K += F_Gamma_AB(idx_b, idx_a) * F_Gamma_AB(idx_b, idx_a);
                  }
                  K = sqrt(K);

                  // adding energy coupling terms for multi-energy case
                  if (num_energy_bins_ > 1) {
                    ScrArray1D<Real> energy_terms = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {

                      Real part_sum_idx = 0.;
                      for (int A = 0; A < num_points_; A++) {
                        Real fn = f0_(m, en * num_points + A, k, j, i);
                        Real fnm1 = (en - 1 >= 0 && en - 1 < num_energy_bins_) ? f0_(m, (en - 1) * num_points_ + A, k, j, i) : 0.;
                        Real fnm2 = (en - 2 >= 0 && en - 2 < num_energy_bins_) ? f0_(m, (en - 2) * num_points_ + A, k, j, i) : 0.;
                        Real fnp1 = (en + 1 >= 0 && en + 1 < num_energy_bins_) ? f0_(m, (en + 1) * num_points_ + A, k, j, i) : 0.;
                        Real fnp2 = (en + 2 >= 0 && en + 2 < num_energy_bins_) ? f0_(m, (en + 2) * num_points_ + A, k, j, i) : 0.;

                        // {F^A} for n and n+1 th bin
                        Real f_term1_np1 = 0.5 * (fnp1 + fn);
                        Real f_term1_n = 0.5 * (fn + fnm1);

                        // [[F^A]] for n and n+1 th bin
                        Real f_term2_np1 = fn - fnp1;
                        Real f_term2_n = (fnm1 - fn);

                        // width of energy bin (uniform grid)
                        Real delta_energy = energy_grid_(1) - energy_grid_(0);

                        Real Dmfn = (fn - fnm1) / delta_energy;
                        Real Dpfn = (fnp1 - fn) / delta_energy;
                        Real Dfn = (fnp1 - fnm1) / (2. * delta_energy);

                        Real Dmfnm1 = (fnm1 - fnm2) / delta_energy;
                        Real Dpfnm1 = (fn - fnm1) / delta_energy;
                        Real Dfnm1 = (fn - fnm2) / (2. * delta_energy);

                        Real Dmfnp1 = (fnp1 - fn) / delta_energy;
                        Real Dpfnp1 = (fnp2 - fnp1) / delta_energy;
                        Real Dfnp1 = (fnp2 - fn) / (2. * delta_energy);

                        Real theta_np12 = (Dfn < energy_par_ * delta_energy || Dmfn * Dpfn > 0.) ? 0. : 1.;
                        Real theta_nm12 = (Dfnm1 < energy_par_ * delta_energy || Dmfnm1 * Dpfnm1 > 0.) ? 0. : 1.;
                        Real theta_np32 = (Dfnp1 < energy_par_ * delta_energy || Dmfnp1 * Dpfnp1 > 0.) ? 0. : 1.;

                        Real theta_n = (theta_nm12 > theta_np12) ? theta_nm12 : theta_np12;
                        Real theta_np1 = (theta_np12 > theta_np32) ? theta_np12 : theta_np32;

                        part_sum_idx +=
                            (energy_grid(en + 1) * energy_grid(en + 1) * energy_grid(en + 1) * (F_Gamma_AB(A, idx) * f_term1_np1 - theta_np1 * K * f_term2_np1 / 2.)
                                - energy_grid(en) * energy_grid(en) * energy_grid(en) * (F_Gamma_AB(A, idx) * f_term1_n - theta_n * K * f_term2_n / 2.));
                      }
                      energy_terms(idx) = part_sum_idx;
                    });
                    member.team_barrier();

                    for (int idx = 0; idx < num_points_; idx++) {
                      g_rhs_scratch(idx) += energy_terms(idx);
                    }

                  }
                  // matrix inverse
                  ScrArray2D<Real> Q_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> Qinv_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> lu_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray1D<Real> x_array = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> b_array = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<int> pivots = ScrArray1D<int>(member.team_scratch(scr_level), num_points_ - 1);

                  par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;
                    Q_matrix(row, col) = sqrt_det_g_ijk * (L_mu_muhat0_(m, 0, 0, k, j, i) * P_matrix_(0, row, col)
                        + L_mu_muhat0_(m, 0, 1, k, j, i) * P_matrix_(1, row, col) + L_mu_muhat0_(m, 0, 2, k, j, i) * P_matrix_(2, row, col)
                        + L_mu_muhat0_(m, 0, 3, k, j, i) * P_matrix_(3, row, col))
                        + sqrt_det_g_ijk * beta_dt * (kappa_s_(m, k, j, i) + kappa_a_(m, k, j, i)) * (row == col) / Ven
                        - sqrt_det_g_ijk * beta_dt * (1. / (4. * M_PI)) * kappa_s_(m, k, j, i) * S_source_(row, col) / Ven;
                    lu_matrix(row, col) = Q_matrix(row, col);
                  });
                  member.team_barrier();

                  radiationfemn::LUInv<ScrArray2D<Real>, ScrArray1D<Real>, ScrArray1D<int>>(member, Q_matrix, Qinv_matrix, lu_matrix, x_array, b_array, pivots);
                  member.team_barrier();

                  par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {
                    Real final_result = 0.;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [&](const int A, Real &partial_sum) {
                      partial_sum += Qinv_matrix(idx, A) * (g_rhs_scratch(A));
                    }, final_result);
                    member.team_barrier();

                    auto unifiedidx = IndicesUnited(nu, en, idx, num_species_, num_energy_bins_, num_points_);
                    f0_(m, unifiedidx, k, j, i) = final_result;
                  });
                  member.team_barrier();
                });

  return TaskStatus::complete;
}
} // namespace radiationfemn