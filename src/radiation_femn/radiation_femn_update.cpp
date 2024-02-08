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

namespace radiationfemn {

TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {
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
  //auto &L_mu_muhat1_ = pmy_pack->pradfemn->L_mu_muhat1;
  //auto &u_mu_ = pmy_pack->pradfemn->u_mu;
  auto &eta_ = pmy_pack->pradfemn->eta;
  auto &e_source_ = pmy_pack->pradfemn->e_source;
  auto &kappa_s_ = pmy_pack->pradfemn->kappa_s;
  auto &kappa_a_ = pmy_pack->pradfemn->kappa_a;
  //auto &F_matrix_ = pmy_pack->pradfemn->F_matrix;
  //auto &G_matrix_ = pmy_pack->pradfemn->G_matrix;
  //auto &energy_par_ = pmy_pack->pradfemn->energy_par;
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

                  Real sqrt_det_g_i = adm.alpha(m, k, j, i) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
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
                        + sqrt_det_g_i * beta_dt * eta_(m, k, j, i) * e_source_(idx) / Ven;

                  });
                  member.team_barrier();
                  /*
                  ScrArray2D<Real> Q_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> Qinv_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> lu_matrix = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray1D<Real> x_array = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> b_array = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<int> pivots = ScrArray1D<int>(member.team_scratch(scr_level), num_points_ - 1);

                  par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;
                    Q_matrix(row, col) = sqrt_det_g_i * (L_mu_muhat0_(m, 0, 0, k, j, i) * P_matrix_(0, row, col)
                        + L_mu_muhat0_(m, 0, 1, k, j, i) * P_matrix_(1, row, col) + L_mu_muhat0_(m, 0, 2, k, j, i) * P_matrix_(2, row, col)
                        + L_mu_muhat0_(m, 0, 3, k, j, i) * P_matrix_(3, row, col))
                        + sqrt_det_g_i * beta_dt * (kappa_s_(m, k, j, i) + kappa_a_(m, k, j, i)) * (row == col) / Ven
                        - sqrt_det_g_i * beta_dt * (1. / (4. * M_PI)) * kappa_s_(m, k, j, i) * S_source_(row, col) / Ven;
                    lu_matrix(row, col) = Q_matrix(row, col);
                  });
                  member.team_barrier();

                  radiationfemn::LUInv<ScrArray2D<Real>, ScrArray1D<Real>, ScrArray1D<int>>(member, Q_matrix, Qinv_matrix, lu_matrix, x_array, b_array, pivots);
                  member.team_barrier(); */

                  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, num_points_), [&](const int idx) {
                    /*
                    Real final_result = 0.;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [&](const int A, Real &partial_sum) {
                      partial_sum += Qinv_matrix(idx, A) * (g_rhs_scratch(A) + 0);
                    }, final_result);
                    member.team_barrier(); */

                    auto unifiedidx = IndicesUnited(nu, en, idx, num_species_, num_energy_bins_, num_points_);
                    f0_(m, unifiedidx, k, j, i) = g_rhs_scratch(idx);
                  });
                  member.team_barrier();
                });

  /*
  par_for_outer("radiation_femn_update_semi_implicit", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, num_energy_bins_ - 1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int en, int k, int j, int i) {


                  // (1) Compute derivative terms for all angles and store in scratch array g_rhs_scratch [num_angles]
                  ScrArray1D<Real> g_rhs_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  auto Ven = (1. / 3.) * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3));

                  par_for_inner(member, 0, num_points_ - 1, [&](const int idx) {
                    int enangidx = en * num_points_ + idx;

                    Real divf_s = flx1(m, enangidx, k, j, i) / (2. * mbsize.d_view(m).dx1 * Ven);

                    if (multi_d) {
                      divf_s += flx2(m, enangidx, k, j, i) / (2. * mbsize.d_view(m).dx2 * Ven);
                    }

                    if (three_d) {
                      divf_s += flx3(m, enangidx, k, j, i) / (2. * mbsize.d_view(m).dx3 * Ven);
                    }

                    g_rhs_scratch(idx) = gam0 * f0_(m, enangidx, k, j, i) + gam1 * f1_(m, enangidx, k, j, i) - beta_dt * divf_s
                        + sqrt_det_g_(m, k, j, i) * beta_dt * eta_(m, k, j, i) * e_source_(idx)
                        - sqrt_det_g_(m, k, j, i) * beta_dt * (kappa_s_(m, k, j, i) + kappa_a_(m, k, j, i)) * f0_(m, enangidx, k, j, i);

                  });
                  member.team_barrier();

                  // (2) Compute matrix FG_{A}^{B} = F_{ihat}^{nuhat muhat}_{A}^{B} Gamma^{ihat}_{nuhat muhat} [num_angles x num_angles]
                  //     and GG_{A}^{B} = G_{ihat}^{nuhat muhat}_{A}^{B} Gamma^{ihat}_{nuhat muhat} [num_angles x num_angles]
                  DvceArray3D<Real> Gamma_fluid_udd;
                  Kokkos::realloc(Gamma_fluid_udd, 4, 4, 4);
                  Kokkos::deep_copy(Gamma_fluid_udd, 0.);

                  // scratch arrays for F_{ihat}^{nuhat muhat}_{A}^{B} Gamma^{ihat}_{nuhat muhat} [num_angles x num_angles]
                  ScrArray2D<Real> F_Gamma_AB = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                  ScrArray2D<Real> G_Gamma_AB = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);

                  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, num_points_ * num_points_), [&](const int idx) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;

                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, 16), [&](const int nuhatmuhat, Real &sum_nuhatmuhat) {
                      int nuhat = int(nuhatmuhat / 4);
                      int muhat = nuhatmuhat - nuhat * 4;

                      sum_nuhatmuhat += F_matrix_(nuhat, muhat, 0, row, col) * Gamma_fluid_udd(1, nuhat, muhat)
                          + F_matrix_(nuhat, muhat, 1, row, col) * Gamma_fluid_udd(2, nuhat, muhat)
                          + F_matrix_(nuhat, muhat, 2, row, col) * Gamma_fluid_udd(3, nuhat, muhat);
                    }, F_Gamma_AB(row, col));

                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, 16), [&](const int nuhatmuhat, Real &sum_nuhatmuhat) {
                      int nuhat = int(nuhatmuhat / 4);
                      int muhat = nuhatmuhat - nuhat * 4;
                      sum_nuhatmuhat += G_matrix_(nuhat, muhat, 0, row, col) * Gamma_fluid_udd(1, nuhat, muhat)
                          + G_matrix_(nuhat, muhat, 1, row, col) * Gamma_fluid_udd(2, nuhat, muhat)
                          + G_matrix_(nuhat, muhat, 2, row, col) * Gamma_fluid_udd(3, nuhat, muhat);
                    }, G_Gamma_AB(row, col));

                  });
                  member.team_barrier();

                  // (4) Compute Lax Friedrich's const K from Frobenius norm of FG_{A}^{B}
                  Real K = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, num_points_ * num_points_), [&](const int idx, Real &frob_norm) {
                    int row = int(idx / num_points_);
                    int col = idx - row * num_points_;

                    frob_norm += F_Gamma_AB(row, col) * F_Gamma_AB(row, col);
                  }, K);
                  member.team_barrier();
                  K = sqrt(K);


                  // (5) Compute the coupling term
                  ScrArray1D<Real> energy_terms = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, num_points_), [&](const int idx) {

                    Real part_sum_idx = 0.;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [&](const int A, Real &part_sum) {

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

                      part_sum += G_Gamma_AB(A, idx) * f0_(m, en * num_points_ + A, k, j, i);
                      //-
                      //    (energy_grid(en + 1) * energy_grid(en + 1) * energy_grid(en + 1) * (F_Gamma_AB(A, idx) * f_term1_np1 - theta_np1 * K * f_term2_np1 / 2.)
                      //        - energy_grid(en) * energy_grid(en) * energy_grid(en) * (F_Gamma_AB(A, idx) * f_term1_n - theta_n * K * f_term2_n / 2.));
                    }, part_sum_idx);

                    energy_terms(idx) = part_sum_idx;
                  });
                  member.team_barrier();

                  // (6) Form the Q matrix and it's inverse, this is needed to go from G to F

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
                        + L_mu_muhat0_(m, 0, 3, k, j, i) * P_matrix_(3, row, col)
                        + beta_dt * (kappa_s_(m, k, j, i) + kappa_a_(m, k, j, i)) * (row == col)
                        + beta_dt * (1. / (4. * M_PI)) * S_source_(row, col));
                    lu_matrix(row, col) = Q_matrix(row, col);
                  });
                  member.team_barrier();

                  radiationfemn::LUInv<ScrArray2D<Real>, ScrArray1D<Real>, ScrArray1D<int>>(member, Q_matrix, Qinv_matrix, lu_matrix, x_array, b_array, pivots);
                  member.team_barrier();

                  // (7) Compute F from G
                  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, num_points_), [=](const int idx) {

                    Real final_result = 0.;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [&](const int A, Real &partial_sum) {
                      partial_sum += Qinv_matrix(idx, A) * (g_rhs_scratch(A) + energy_terms(A));
                    }, final_result);
                    member.team_barrier();

                    f0_(m, en * num_points_ + idx, k, j, i) = final_result;
                  });
                  member.team_barrier();
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

  return TaskStatus::complete;
}
} // namespace radiationfemn