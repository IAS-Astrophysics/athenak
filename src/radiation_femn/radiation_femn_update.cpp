//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_update.cpp
//  \brief Performs update of radiation variables (f0) using semi-implicit time stepping

#include <math.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_matinv.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"
#include "radiation_femn_closure.hpp"

namespace radiationfemn {
TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {
  const int NGHOST = 2;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  bool &m1_flag_ = pmy_pack->pradfemn->m1_flag;

  Real beta[2] = {0.5, 1.0};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  int &num_points_ = pmy_pack->pradfemn->num_points;
  int &num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  int &num_species_ = pmy_pack->pradfemn->num_species;
  int nnu1 = num_species_ - 1;
  Real &rad_E_floor_ = pmy_pack->pradfemn->rad_E_floor;
  Real &rad_eps_ = pmy_pack->pradfemn->rad_eps;

  auto &rad_mask_array_ = pmy_pack->pradfemn->radiation_mask;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &f1_ = pmy_pack->pradfemn->f1;
  auto &energy_grid_ = pmy_pack->pradfemn->energy_grid;
  auto &flx1 = pmy_pack->pradfemn->iflx.x1f;
  auto &flx2 = pmy_pack->pradfemn->iflx.x2f;
  auto &flx3 = pmy_pack->pradfemn->iflx.x3f;
  auto &tetr_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  auto &u_mu_ = pmy_pack->pradfemn->u_mu;
  auto &eta_ = pmy_pack->pradfemn->eta;
  auto &e_source_ = pmy_pack->pradfemn->e_source;
  auto &kappa_s_ = pmy_pack->pradfemn->kappa_s;
  auto &kappa_a_ = pmy_pack->pradfemn->kappa_a;
  auto &f_matrix = pmy_pack->pradfemn->F_matrix;
  auto &g_matrix = pmy_pack->pradfemn->G_matrix;
  auto &energy_par_ = pmy_pack->pradfemn->energy_par;
  auto &p_matrix = pmy_pack->pradfemn->P_matrix;
  auto &s_source = pmy_pack->pradfemn->S_source;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  size_t scr_size = ScrArray2D<Real>::shmem_size(num_points_, num_points_) * 5
      + ScrArray1D<Real>::shmem_size(num_points_) * 5
      + ScrArray1D<Real>::shmem_size(num_points_) * 8
      + ScrArray1D<Real>::shmem_size(4 * 4 * 4) * 2;
  int scr_level = 0;
  par_for_outer("radiation_femn_update", DevExeSpace(), scr_size, scr_level,
                0, nmb1, 0, nnu1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int nu, int k, int j, int i) {

                  if (rad_mask_array_(m, k, j, i)) {
                    // rhs array
                    auto g_rhs_scratch =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_energy_bins_, num_points_);

                    // compute metric, its inverse and sqrt(-determinant)
                    Real g_dd[16];
                    Real g_uu[16];
                    adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                         adm.beta_u(m, 0, k, j, i),
                                         adm.beta_u(m, 1, k, j, i),
                                         adm.beta_u(m, 2, k, j, i),
                                         adm.g_dd(m, 0, 0, k, j, i),
                                         adm.g_dd(m, 0, 1, k, j, i),
                                         adm.g_dd(m, 0, 2, k, j, i),
                                         adm.g_dd(m, 1, 1, k, j, i),
                                         adm.g_dd(m, 1, 2, k, j, i),
                                         adm.g_dd(m, 2, 2, k, j, i),
                                         g_dd);
                    adm::SpacetimeUpperMetric(adm.alpha(m, k, j, i),
                                              adm.beta_u(m, 0, k, j, i),
                                              adm.beta_u(m, 1, k, j, i),
                                              adm.beta_u(m, 2, k, j, i),
                                              adm.g_dd(m, 0, 0, k, j, i),
                                              adm.g_dd(m, 0, 1, k, j, i),
                                              adm.g_dd(m, 0, 2, k, j, i),
                                              adm.g_dd(m, 1, 1, k, j, i),
                                              adm.g_dd(m, 1, 2, k, j, i),
                                              adm.g_dd(m, 2, 2, k, j, i),
                                              g_uu);
                    Real sqrt_det_g_ijk = adm.alpha(m, k, j, i)
                        * Kokkos::sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i),
                                                       adm.g_dd(m, 0, 1, k, j, i),
                                                       adm.g_dd(m, 0, 2, k, j, i),
                                                       adm.g_dd(m, 1, 1, k, j, i),
                                                       adm.g_dd(m, 1, 2, k, j, i),
                                                       adm.g_dd(m, 2, 2, k, j, i)));

                    Real deltax[3] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2,
                                      1 / mbsize.d_view(m).dx3};

                    // lapse derivatives (\p_mu alpha)
                    Real dtalpha_d = 0.; // time derivatives, get from z4c
                    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
                    dalpha_d(0) = Dx<NGHOST>(0, deltax, adm.alpha, m, k, j, i);
                    dalpha_d(1) =
                        (multi_d) ? Dx<NGHOST>(1, deltax, adm.alpha, m, k, j, i) : 0.;
                    dalpha_d(2) =
                        (three_d) ? Dx<NGHOST>(2, deltax, adm.alpha, m, k, j, i) : 0.;

                    // shift derivatives (\p_mu beta^i)
                    Real dtbetax_du = 0.; // time derivatives, get from z4c
                    Real dtbetay_du = 0.;
                    Real dtbetaz_du = 0.;
                    Real dtbeta_du[3] = {dtbetax_du, dtbetay_du, dtbetaz_du};
                    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2>
                        dbeta_du; // spatial derivatives
                    for (int a = 0; a < 3; ++a) {
                      dbeta_du(0, a) = Dx<NGHOST>(0, deltax, adm.beta_u, m, a, k, j, i);
                      dbeta_du(1, a) =
                          (multi_d) ? Dx<NGHOST>(1, deltax, adm.beta_u, m, a, k, j, i)
                                    : 0.;
                      dbeta_du(2, a) =
                          (three_d) ? Dx<NGHOST>(2, deltax, adm.beta_u, m, a, k, j, i)
                                    : 0.;
                    }

                    // covariant shift (beta_i)
                    Real betax_d = 0;
                    Real betay_d = 0;
                    Real betaz_d = 0;
                    for (int a = 0; a < 3; ++a) {
                      betax_d += adm.g_dd(m, 0, a, k, j, i) * adm.beta_u(m, a, k, j, i);
                      betay_d += adm.g_dd(m, 1, a, k, j, i) * adm.beta_u(m, a, k, j, i);
                      betaz_d += adm.g_dd(m, 2, a, k, j, i) * adm.beta_u(m, a, k, j, i);
                    }
                    Real beta_d[3] = {betax_d, betay_d, betaz_d};

                    // derivatives of spatial metric (\p_mu g_ij)
                    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> dtg_dd;
                    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
                    for (int a = 0; a < 3; ++a) {
                      for (int b = 0; b < 3; ++b) {
                        dtg_dd(a, b) = 0.; // time derivatives, get from z4c
                        dg_ddd(0, a, b) =
                            Dx<NGHOST>(0, deltax, adm.g_dd, m, a, b, k, j, i);
                        dg_ddd(1, a, b) =
                            (multi_d) ? Dx<NGHOST>(1, deltax, adm.g_dd, m, a, b, k, j, i)
                                      : 0.;
                        dg_ddd(2, a, b) =
                            (three_d) ? Dx<NGHOST>(2, deltax, adm.g_dd, m, a, b, k, j, i)
                                      : 0.;
                      }
                    }

                    // derivatives of the 4-metric: time derivatives
                    AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> dg4_ddd;
                    dg4_ddd(0, 0, 0) = -2. * adm.alpha(m, k, j, i) * dtalpha_d;
                    for (int a = 0; a < 3; ++a) {
                      dg4_ddd(0, 0, 0) += 2. * beta_d[a] * dtbeta_du[a];
                      dg4_ddd(0, a + 1, 0) = 0;
                      for (int b = 0; b < 3; ++b) {
                        dg4_ddd(0, 0, 0) += dtg_dd(a, b)
                            * adm.beta_u(m, a, k, j, i)
                            * adm.beta_u(m, b, k, j, i);
                        dg4_ddd(0, a + 1, 0) += dtg_dd(a, b) * adm.beta_u(m, b, k, j, i)
                            + adm.g_dd(m, a, b, k, j, i) * dtbeta_du[b];
                        dg4_ddd(0, a + 1, b + 1) = dtg_dd(a, b);
                      }
                    }

                    // derivatives of the 4-metric: spatial derivatives
                    for (int c = 0; c < 3; c++) {
                      dg4_ddd(c + 1, 0, 0) = -2. * adm.alpha(m, k, j, i) * dalpha_d(c);
                      for (int a = 0; a < 3; ++a) {
                        dg4_ddd(c + 1, 0, 0) += 2. * beta_d[a] * dbeta_du(c, a);
                        dg4_ddd(c + 1, a + 1, 0) = 0;
                        for (int b = 0; b < 3; ++b) {
                          dg4_ddd(c + 1, 0, 0) += dg_ddd(c, a, b)
                              * adm.beta_u(m, a, k, j, i)
                              * adm.beta_u(m, b, k, j, i);
                          dg4_ddd(c + 1, a + 1, 0) +=
                              dg_ddd(c, a, b) * adm.beta_u(m, b, k, j, i)
                                  + adm.g_dd(m, a, b, k, j, i) * dbeta_du(c, b);
                          dg4_ddd(c + 1, a + 1, b + 1) = dg_ddd(c, a, b);
                        }
                      }
                    }

                    // 4-Christoeffel symbols
                    AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> Gamma_udd;
                    for (int a = 0; a < 4; ++a) {
                      for (int b = 0; b < 4; ++b) {
                        for (int c = 0; c < 4; ++c) {
                          Gamma_udd(a, b, c) = 0.0;
                          for (int d = 0; d < 4; ++d) {
                            Gamma_udd(a, b, c) += 0.5 * g_uu[a + 4 * d]
                                * (dg4_ddd(b, d, c)
                                    + dg4_ddd(c, b, d)
                                    - dg4_ddd(d, b, c));
                          }
                        }
                      }
                    }

                    // Ricci rotation coefficients
                    AthenaScratchTensor4d<Real, TensorSymm::NONE, 4, 3> Gamma_fluid_udd;
                    for (int a = 0; a < 4; ++a) {
                      for (int b = 0; b < 4; ++b) {
                        for (int c = 0; c < 4; ++c) {
                          Gamma_fluid_udd(a, b, c) = 0.0;
                          for (int ac_idx = 0; ac_idx < 4 * 4; ++ac_idx) {
                            const int a_idx = static_cast<int>(ac_idx / 4);
                            const int c_idx = ac_idx - a_idx * 4;

                            // compute inverse tetrad
                            Real sign_factor = (a == 0) ? -1. : +1.;
                            Real tetr_a_aidx = 0;
                            for (int d_idx = 0; d_idx < 4; ++d_idx) {
                              tetr_a_aidx += sign_factor * g_dd[a_idx + 4 * d_idx]
                                  * tetr_mu_muhat0_(m, d_idx, a, k, j, i);
                            }

                            for (int b_idx = 0; b_idx < 4; ++b_idx) {
                              Gamma_fluid_udd(a, b, c) += tetr_a_aidx
                                  * tetr_mu_muhat0_(m, c_idx, c, k, j, i)
                                  * tetr_mu_muhat0_(m, b_idx, b, k, j, i)
                                  * Gamma_udd(a_idx, b_idx, c_idx);
                            }

                            Real dtetr_mu_muhat_d[4];
                            dtetr_mu_muhat_d[0] = 0.; // no time derivatives currently
                            dtetr_mu_muhat_d[1] =
                                Dx<NGHOST>(0, deltax, tetr_mu_muhat0_, m, a_idx, b,
                                           k, j, i);
                            dtetr_mu_muhat_d[2] =
                                (multi_d) ? Dx<NGHOST>(1, deltax, tetr_mu_muhat0_, m,
                                                       a_idx, b, k, j, i) : 0.;
                            dtetr_mu_muhat_d[3] =
                                (three_d) ? Dx<NGHOST>(2, deltax, tetr_mu_muhat0_,
                                                       m, a_idx, b, k, j, i) : 0.;
                            Gamma_fluid_udd(a, b, c) += tetr_a_aidx
                                * tetr_mu_muhat0_(m, c_idx, c, k, j, i)
                                * dtetr_mu_muhat_d[c_idx];
                          }
                        }
                      }
                    }
                    member.team_barrier();

                    // Compute F Gam and G Gam matrices
                    ScrArray2D<Real> f_gam =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);
                    ScrArray2D<Real> g_gam =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);

                    par_for_inner(member, 0, num_points_ * num_points_ - 1,
                                  [&](const int idx) {
                                    const int row = static_cast<int>(idx / num_points_);
                                    const int col = idx - row * num_points_;

                                    Real sum_fgam = 0.;
                                    Real sum_ggam = 0.;
                                    for (int inumu = 0; inumu < 48; inumu++) {
                                      RadiationFEMNPhaseIndices idx_numui =
                                          IndicesComponent(inumu, 4, 4, 3);
                                      int id_i = idx_numui.nuidx;
                                      int id_nu = idx_numui.enidx;
                                      int id_mu = idx_numui.angidx;

                                      sum_fgam +=
                                          f_matrix(id_nu, id_mu, id_i, row, col)
                                              * Gamma_fluid_udd(id_i + 1, id_nu, id_mu);
                                      sum_ggam +=
                                          g_matrix(id_nu, id_mu, id_i, row, col)
                                              * Gamma_fluid_udd(id_i + 1, id_nu, id_mu);
                                    }
                                    f_gam(row, col) = sum_fgam;
                                    g_gam(row, col) = sum_ggam;
                                  });
                    member.team_barrier();

                    // calculate spatial derivative term
                    par_for_inner(member, 0, num_energy_bins_*num_points_ - 1,
                      [&](const int ennB){
                        const int enn = static_cast<int>(ennB / num_points_);
                        const int B = ennB - enn * num_points_;

                        const int nuenangidx = IndicesUnited(nu, enn, B,
                                                              num_species_, num_energy_bins_, num_points_);

                        Real divf_s = flx1(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx1);
                        if (multi_d) {
                          divf_s += flx2(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx2);
                        }
                        if (three_d) {
                          divf_s += flx3(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx3);
                        }

                        Real fval = 0;
                        for (int idx_a = 0; idx_a < num_points_; idx_a++) {
                          for (int idx_m = 0; idx_m < num_species_; idx_m++) {
                            for (int idx_muhat = 0; idx_muhat < 4; idx_muhat++) {
                              const int nuenangidx_a =
                                IndicesUnited(nu, idx_m, idx_a, num_species_,
                                              num_energy_bins_, num_points_);
                                fval += Ven_matrix(idx_m, enn)
                                        * tetr_mu_muhat0_(m, 0, idx_muhat, k, j, i)
                                        * p_matrix(idx_muhat, idx_a, B)
                                        * f1_(m, nuenangidx_a, k, j, i);
                            }
                          }
                        }

                        Real momentum_term = 0.;
                        for (int idx_a = 0; idx_a < num_points_; idx_a++) {
                          for (int idx_m = 0; idx_m < num_energy_bins_; idx_m++) {
                            int idxunited_am = IndicesUnited(nu, idx_m, idx_a,
                                                              num_species_,
                                                              num_energy_bins_, num_points_);
                            momentum_term += - g_gam(idx_a, B) * Ven_matrix(idx_m, enn) * f0_(m, idxunited_am, k, j, i)
                                             - f_gam(idx_a, B) * Wen_matrix(idx_m, enn) * f0_(m, idxunited_am, k, j, i);
                          }
                        }

                        g_rhs_scratch(enn, B) = fval + beta_dt * divf_s
                                                + beta_dt * momentum_term
                                                + sqrt_det_g_ijk * beta_dt * eta_(m, k, j, i) * e_source_(B);
                    });
                    member.team_barrier();

                    for (int idx_n=0; idx_n < num_energy_bins_; idx_n++) {
                      auto Q_matrix =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);
                      auto lu_matrix =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);
                      auto x_array =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      auto pivots =
                        ScrArray1D<int>(member.team_scratch(scr_level), num_points_ - 1);

                      // @TODO: fix from here
                      /*
                      par_for_inner(member, 0, num_points_ * num_points_ - 1,
                                    [&](const int idx)
                                    {
                                      int row = int(idx / num_points_);
                                      int col = idx - row * num_points_;

                                      Real sum_val = 0;
                                      for (int id_i = 0; id_i < 4; ++id_i) {
                                        sum_val += tetr_mu_muhat0_(m, 0, id_i, k, j, i)
                                          * p_matrix(id_i, row, col);
                                      }
                                      Q_matrix(row, col) = sum_val
                                        + beta_dt
                                        * (kappa_s_(m, k, j, i)
                                          + kappa_a_(m, k, j, i))
                                        * p_matrix(0, row, col) / Ven
                                        - beta_dt
                                        * (1. / (4. * M_PI))
                                        * kappa_s_(m, k, j, i)
                                        * s_source(row, col) / Ven;
                                      lu_matrix(row, col) = Q_matrix(row, col);
                                    });*/
                    }
                  }
                });
  /*
  par_for_outer("radiation_femn_update", DevExeSpace(), scr_size, scr_level,
                0, nmb1, 0, num_species_energy - 1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int nuen, int k, int j, int i) {
                  if (rad_mask_array_(m, k, j, i)) {
                    const int nu = static_cast<int>(nuen / num_energy_bins_);
                    const int en = nuen - nu * num_energy_bins_;

                    // rhs array and energy factor
                    ScrArray1D<Real> g_rhs_scratch =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    const Real Ven = (num_energy_bins_ > 1) ? (1. / 3.)
                        * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3)) : 1.;

                    // compute metric, its inverse and sqrt(-determinant)
                    Real g_dd[16];
                    Real g_uu[16];
                    adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                         adm.beta_u(m, 0, k, j, i),
                                         adm.beta_u(m, 1, k, j, i),
                                         adm.beta_u(m, 2, k, j, i),
                                         adm.g_dd(m, 0, 0, k, j, i),
                                         adm.g_dd(m, 0, 1, k, j, i),
                                         adm.g_dd(m, 0, 2, k, j, i),
                                         adm.g_dd(m, 1, 1, k, j, i),
                                         adm.g_dd(m, 1, 2, k, j, i),
                                         adm.g_dd(m, 2, 2, k, j, i),
                                         g_dd);
                    adm::SpacetimeUpperMetric(adm.alpha(m, k, j, i),
                                              adm.beta_u(m, 0, k, j, i),
                                              adm.beta_u(m, 1, k, j, i),
                                              adm.beta_u(m, 2, k, j, i),
                                              adm.g_dd(m, 0, 0, k, j, i),
                                              adm.g_dd(m, 0, 1, k, j, i),
                                              adm.g_dd(m, 0, 2, k, j, i),
                                              adm.g_dd(m, 1, 1, k, j, i),
                                              adm.g_dd(m, 1, 2, k, j, i),
                                              adm.g_dd(m, 2, 2, k, j, i),
                                              g_uu);
                    Real sqrt_det_g_ijk = adm.alpha(m, k, j, i)
                        * Kokkos::sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i),
                                                       adm.g_dd(m, 0, 1, k, j, i),
                                                       adm.g_dd(m, 0, 2, k, j, i),
                                                       adm.g_dd(m, 1, 1, k, j, i),
                                                       adm.g_dd(m, 1, 2, k, j, i),
                                                       adm.g_dd(m, 2, 2, k, j, i)));

                    // calculate spatial derivative term
                    par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {
                      const int nuenangidx = IndicesUnited(nu, en, idx, num_species_,
                                                           num_energy_bins_, num_points_);
                      Real divf_s =
                          flx1(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx1);
                      if (multi_d) {
                        divf_s +=
                            flx2(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx2);
                      }
                      if (three_d) {
                        divf_s +=
                            flx3(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx3);
                      }

                      Real fval = 0;
                      for (int idx_a = 0; idx_a < num_points_; idx_a++) {
                        const int
                            nuenangidx_a = IndicesUnited(nu, en, idx_a, num_species_,
                                                         num_energy_bins_, num_points_);
                        for (int muhat = 0; muhat < 4; muhat++) {
                          fval += tetr_mu_muhat0_(m, 0, muhat, k, j, i)
                              * p_matrix(muhat, idx_a, idx)
                              * f1_(m, nuenangidx_a, k, j, i);
                        }
                      }
                      g_rhs_scratch(idx) = fval + beta_dt * divf_s
                          + sqrt_det_g_ijk * beta_dt * eta_(m, k, j, i) * e_source_(idx)
                              / Ven;
                    });
                    member.team_barrier();

                    Real deltax[3] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2,
                                      1 / mbsize.d_view(m).dx3};

                    // lapse derivatives (\p_mu alpha)
                    Real dtalpha_d = 0.; // time derivatives, get from z4c
                    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
                    dalpha_d(0) = Dx<NGHOST>(0, deltax, adm.alpha, m, k, j, i);
                    dalpha_d(1) =
                        (multi_d) ? Dx<NGHOST>(1, deltax, adm.alpha, m, k, j, i) : 0.;
                    dalpha_d(2) =
                        (three_d) ? Dx<NGHOST>(2, deltax, adm.alpha, m, k, j, i) : 0.;

                    // shift derivatives (\p_mu beta^i)
                    Real dtbetax_du = 0.; // time derivatives, get from z4c
                    Real dtbetay_du = 0.;
                    Real dtbetaz_du = 0.;
                    Real dtbeta_du[3] = {dtbetax_du, dtbetay_du, dtbetaz_du};
                    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2>
                        dbeta_du; // spatial derivatives
                    for (int a = 0; a < 3; ++a) {
                      dbeta_du(0, a) = Dx<NGHOST>(0, deltax, adm.beta_u, m, a, k, j, i);
                      dbeta_du(1, a) =
                          (multi_d) ? Dx<NGHOST>(1, deltax, adm.beta_u, m, a, k, j, i)
                                    : 0.;
                      dbeta_du(2, a) =
                          (three_d) ? Dx<NGHOST>(2, deltax, adm.beta_u, m, a, k, j, i)
                                    : 0.;
                    }

                    // covariant shift (beta_i)
                    Real betax_d = 0;
                    Real betay_d = 0;
                    Real betaz_d = 0;
                    for (int a = 0; a < 3; ++a) {
                      betax_d += adm.g_dd(m, 0, a, k, j, i) * adm.beta_u(m, a, k, j, i);
                      betay_d += adm.g_dd(m, 1, a, k, j, i) * adm.beta_u(m, a, k, j, i);
                      betaz_d += adm.g_dd(m, 2, a, k, j, i) * adm.beta_u(m, a, k, j, i);
                    }
                    Real beta_d[3] = {betax_d, betay_d, betaz_d};

                    // derivatives of spatial metric (\p_mu g_ij)
                    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> dtg_dd;
                    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
                    for (int a = 0; a < 3; ++a) {
                      for (int b = 0; b < 3; ++b) {
                        dtg_dd(a, b) = 0.; // time derivatives, get from z4c
                        dg_ddd(0, a, b) =
                            Dx<NGHOST>(0, deltax, adm.g_dd, m, a, b, k, j, i);
                        dg_ddd(1, a, b) =
                            (multi_d) ? Dx<NGHOST>(1, deltax, adm.g_dd, m, a, b, k, j, i)
                                      : 0.;
                        dg_ddd(2, a, b) =
                            (three_d) ? Dx<NGHOST>(2, deltax, adm.g_dd, m, a, b, k, j, i)
                                      : 0.;
                      }
                    }

                    // derivatives of the 4-metric: time derivatives
                    AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> dg4_ddd;
                    dg4_ddd(0, 0, 0) = -2. * adm.alpha(m, k, j, i) * dtalpha_d;
                    for (int a = 0; a < 3; ++a) {
                      dg4_ddd(0, 0, 0) += 2. * beta_d[a] * dtbeta_du[a];
                      dg4_ddd(0, a + 1, 0) = 0;
                      for (int b = 0; b < 3; ++b) {
                        dg4_ddd(0, 0, 0) += dtg_dd(a, b)
                            * adm.beta_u(m, a, k, j, i)
                            * adm.beta_u(m, b, k, j, i);
                        dg4_ddd(0, a + 1, 0) += dtg_dd(a, b) * adm.beta_u(m, b, k, j, i)
                            + adm.g_dd(m, a, b, k, j, i) * dtbeta_du[b];
                        dg4_ddd(0, a + 1, b + 1) = dtg_dd(a, b);
                      }
                    }

                    // derivatives of the 4-metric: spatial derivatives
                    for (int c = 0; c < 3; c++) {
                      dg4_ddd(c + 1, 0, 0) = -2. * adm.alpha(m, k, j, i) * dalpha_d(c);
                      for (int a = 0; a < 3; ++a) {
                        dg4_ddd(c + 1, 0, 0) += 2. * beta_d[a] * dbeta_du(c, a);
                        dg4_ddd(c + 1, a + 1, 0) = 0;
                        for (int b = 0; b < 3; ++b) {
                          dg4_ddd(c + 1, 0, 0) += dg_ddd(c, a, b)
                              * adm.beta_u(m, a, k, j, i)
                              * adm.beta_u(m, b, k, j, i);
                          dg4_ddd(c + 1, a + 1, 0) +=
                              dg_ddd(c, a, b) * adm.beta_u(m, b, k, j, i)
                                  + adm.g_dd(m, a, b, k, j, i) * dbeta_du(c, b);
                          dg4_ddd(c + 1, a + 1, b + 1) = dg_ddd(c, a, b);
                        }
                      }
                    }

                    // 4-Christoeffel symbols
                    AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> Gamma_udd;
                    for (int a = 0; a < 4; ++a) {
                      for (int b = 0; b < 4; ++b) {
                        for (int c = 0; c < 4; ++c) {
                          Gamma_udd(a, b, c) = 0.0;
                          for (int d = 0; d < 4; ++d) {
                            Gamma_udd(a, b, c) += 0.5 * g_uu[a + 4 * d]
                                * (dg4_ddd(b, d, c)
                                    + dg4_ddd(c, b, d)
                                    - dg4_ddd(d, b, c));
                          }
                        }
                      }
                    }

                    // Ricci rotation coefficients
                    AthenaScratchTensor4d<Real, TensorSymm::NONE, 4, 3> Gamma_fluid_udd;
                    for (int a = 0; a < 4; ++a) {
                      for (int b = 0; b < 4; ++b) {
                        for (int c = 0; c < 4; ++c) {
                          Gamma_fluid_udd(a, b, c) = 0.0;
                          for (int ac_idx = 0; ac_idx < 4 * 4; ++ac_idx) {
                            const int a_idx = static_cast<int>(ac_idx / 4);
                            const int c_idx = ac_idx - a_idx * 4;

                            // compute inverse tetrad
                            Real sign_factor = (a == 0) ? -1. : +1.;
                            Real tetr_a_aidx = 0;
                            for (int d_idx = 0; d_idx < 4; ++d_idx) {
                              tetr_a_aidx += sign_factor * g_dd[a_idx + 4 * d_idx]
                                  * tetr_mu_muhat0_(m, d_idx, a, k, j, i);
                            }

                            for (int b_idx = 0; b_idx < 4; ++b_idx) {
                              Gamma_fluid_udd(a, b, c) += tetr_a_aidx
                                  * tetr_mu_muhat0_(m, c_idx, c, k, j, i)
                                  * tetr_mu_muhat0_(m, b_idx, b, k, j, i)
                                  * Gamma_udd(a_idx, b_idx, c_idx);
                            }

                            Real dtetr_mu_muhat_d[4];
                            dtetr_mu_muhat_d[0] = 0.; // no time derivatives currently
                            dtetr_mu_muhat_d[1] =
                                Dx<NGHOST>(0, deltax, tetr_mu_muhat0_, m, a_idx, b,
                                           k, j, i);
                            dtetr_mu_muhat_d[2] =
                                (multi_d) ? Dx<NGHOST>(1, deltax, tetr_mu_muhat0_, m,
                                                       a_idx, b, k, j, i) : 0.;
                            dtetr_mu_muhat_d[3] =
                                (three_d) ? Dx<NGHOST>(2, deltax, tetr_mu_muhat0_,
                                                       m, a_idx, b, k, j, i) : 0.;
                            Gamma_fluid_udd(a, b, c) += tetr_a_aidx
                                * tetr_mu_muhat0_(m, c_idx, c, k, j, i)
                                * dtetr_mu_muhat_d[c_idx];
                          }
                        }
                      }
                    }
                    member.team_barrier();
                    // Compute F Gam and G Gam matrices
                    ScrArray2D<Real> f_gam =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);
                    ScrArray2D<Real> g_gam =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);

                    par_for_inner(member, 0, num_points_ * num_points_ - 1,
                                  [&](const int idx) {
                                    const int row = static_cast<int>(idx / num_points_);
                                    const int col = idx - row * num_points_;

                                    Real sum_fgam = 0.;
                                    Real sum_ggam = 0.;
                                    for (int inumu = 0; inumu < 48; inumu++) {
                                      RadiationFEMNPhaseIndices idx_numui =
                                          IndicesComponent(inumu, 4, 4, 3);
                                      int id_i = idx_numui.nuidx;
                                      int id_nu = idx_numui.enidx;
                                      int id_mu = idx_numui.angidx;

                                      sum_fgam +=
                                          f_matrix(id_nu, id_mu, id_i, row, col)
                                              * Gamma_fluid_udd(id_i + 1, id_nu, id_mu);
                                      sum_ggam +=
                                          g_matrix(id_nu, id_mu, id_i, row, col)
                                              * Gamma_fluid_udd(id_i + 1, id_nu, id_mu);
                                    }
                                    f_gam(row, col) = sum_fgam;
                                    g_gam(row, col) = sum_ggam;
                                  });
                    member.team_barrier();

                    // Add Christoeffel terms to rhs and compute Lax Friedrich's const K
                    Real K = 0.;
                    for (int index_b = 0; index_b < num_points_; index_b++) {
                      Real sum_terms = 0.;
                      for (int index_a = 0; index_a < num_points_; index_a++) {
                        int index_a_united = IndicesUnited(nu, en, index_a, num_species_,
                                                           num_energy_bins_, num_points_);
                        sum_terms +=
                            (g_gam(index_a, index_b) - (1.5)*f_gam(index_a, index_b))
                                * f0_(m, index_a_united, k, j, i);
                        K += f_gam(index_a, index_b) * f_gam(index_a, index_b);
                      }
                      g_rhs_scratch(index_b) -= beta_dt * sum_terms;
                    }
                    member.team_barrier();
                    K = sqrt(K);

                    }

                    ScrArray2D<Real> Q_matrix =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);
                    ScrArray2D<Real> lu_matrix =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_points_,
                                         num_points_);
                    ScrArray1D<Real> x_array =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<int> pivots =
                        ScrArray1D<int>(member.team_scratch(scr_level), num_points_ - 1);

                    par_for_inner(member, 0, num_points_ * num_points_ - 1,
                                  [&](const int idx) {
                                    int row = int(idx / num_points_);
                                    int col = idx - row * num_points_;

                                    Real sum_val = 0;
                                    for (int id_i = 0; id_i < 4; ++id_i) {
                                      sum_val += tetr_mu_muhat0_(m, 0, id_i, k, j, i)
                                          * p_matrix(id_i, row, col);
                                    }
                                    Q_matrix(row, col) = sum_val
                                        + beta_dt
                                            * (kappa_s_(m, k, j, i)
                                                + kappa_a_(m, k, j, i))
                                            * p_matrix(0, row, col) / Ven
                                        - beta_dt
                                            * (1. / (4. * M_PI))
                                            * kappa_s_(m, k, j, i)
                                            * s_source(row, col) / Ven;
                                    lu_matrix(row, col) = Q_matrix(row, col);
                                  });
                    member.team_barrier();

                    for (int index_i = 0; index_i < num_points_; index_i++) {
                      x_array(index_i) = g_rhs_scratch(index_i);
                    }
                    radiationfemn::LUSolveAxb<ScrArray2D<Real>, ScrArray1D<Real>,
                                              ScrArray1D<int>>(member, Q_matrix,
                                                               lu_matrix, x_array,
                                                               g_rhs_scratch, pivots);
                    member.team_barrier();

                    if (m1_flag_) {
                      ApplyM1Floor(member, x_array, rad_E_floor_, rad_eps_);
                      member.team_barrier();
                    }

                    par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {
                      auto unifiedidx = IndicesUnited(nu, en, idx, num_species_,
                                                      num_energy_bins_, num_points_);
                      f0_(m, unifiedidx, k, j, i) = x_array(idx);
                    });
                    member.team_barrier();
                  }
                }
  ); */

  return TaskStatus::complete;
}
} // namespace radiationfemn
