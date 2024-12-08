//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_fluxes.cpp
//  \brief Calculate rhs side from 3D fluxes for radiation

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_closure.hpp"
#include "adm/adm.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  int num_points_ = pmy_pack->pradfemn->num_points;
  int num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  int num_species_ = pmy_pack->pradfemn->num_species;
  int nnu1 = num_species_ - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  bool &m1_flag_ = pmy_pack->pradfemn->m1_flag;
  M1Closure m1_closure_ = pmy_pack->pradfemn->m1_closure;
  ClosureFunc m1_closure_fun_ = pmy_pack->pradfemn->closure_fun;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &rad_mask_array_ = pmy_pack->pradfemn->radiation_mask;
  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &energy_grid_ = pmy_pack->pradfemn->energy_grid;
  auto &p_matrix = pmy_pack->pradfemn->P_matrix;
  auto &pmod_matrix = pmy_pack->pradfemn->Pmod_matrix;
  auto &tetr_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  // i-direction
  int scr_level = 0;
  int scr_size = ScrArray1D<Real>::shmem_size(num_points) * 6;
  auto &flx1 = iflx.x1f;
  par_for_outer("radiation_femn_flux_x", DevExeSpace(), scr_size, scr_level,
                0, nmb1, 0, nnu1, ks, ke, js, je, is, static_cast<int>(ie / 2) + 1,
                KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuidx,
                              const int k, const int j, const int i) {
                  const int kk = k;
                  const int jj = j;
                  const int ii = 2 * i - 2;

                  if (rad_mask_array_(m, kk, jj, ii)) {
                    // load scratch arrays using closure
                    auto f0_scratch = ScrArray2D<Real>(
                        member.team_scratch(scr_level), num_energy_bins_, num_points_);
                    auto f0_scratch_p1 =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_energy_bins_,
                                         num_points_);
                    auto f0_scratch_p2 =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_energy_bins_,
                                         num_points_);
                    auto f0_scratch_p3 =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_energy_bins_,
                                         num_points_);
                    auto f0_scratch_m1 =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_energy_bins_,
                                         num_points_);
                    auto f0_scratch_m2 =
                        ScrArray2D<Real>(member.team_scratch(scr_level), num_energy_bins_,
                                         num_points_);
                    ApplyClosure(1, member, num_species_, num_energy_bins_, num_points_,
                                  m, nuidx, kk, jj, ii,
                                  f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                  f0_scratch_p3, f0_scratch_m1, f0_scratch_m2,
                                  m1_flag_, m1_closure_, m1_closure_fun_);
                    member.team_barrier();

                    par_for_inner(member, 0, num_energy_bins_*num_points_ - 1, [&](const int ennB) {
                        const int enn = static_cast<int>(ennB / num_points_);
                        const int B = ennB - enn * num_points_;

                        Real favg = 0.;
                        Real fminus = 0.;
                        Real fplus = 0;

                        for (int muhat = 0; muhat < 4; muhat++) {
                            for (int enm = 0; enm < num_energy_bins_; enm++) {
                                for (int A = 0; A < num_points_; A++) {

                                    favg += (0.5) * Ven_matrix(enm, enn)
                                    * (p_matrix(muhat, B, A)
                                    * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii)
                                    * f0_scratch(enm, A)
                                    + p_matrix(muhat, B, A)
                                    * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1)
                                    * f0_scratch_p1(enm, A));

                                    const Real tetr_mu_muhat0_L = 0.5
                                        * (tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii - 1)
                                        + tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii));

                                    fminus += (0.5) * Ven_matrix(enm, enn)
                                        * (tetr_mu_muhat0_L)
                                        * (p_matrix(muhat, B, A)
                                        * ((1.5) * f0_scratch(enm, A)
                                        - (0.5) * f0_scratch_p1(enm, A)
                                        + (1.5) * f0_scratch_m1(enm, A)
                                        - (0.5) * f0_scratch_m2(enm, A))
                                        - Sgn(tetr_mu_muhat0_L)
                                        * pmod_matrix(muhat, B, A)
                                        * (-(1.5) * f0_scratch_m1(enm, A)
                                        + (0.5) * f0_scratch_m2(enm, A)
                                        + (1.5) * f0_scratch(enm, A)
                                        - (0.5) * f0_scratch_p1(enm, A)));

                                    const Real tetr_mu_muhat0_R = 0.5
                                        * (tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1)
                                        + tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 2));

                                    fplus += (0.5) * Ven_matrix(enm, enn)
                                    * (tetr_mu_muhat0_R)
                                    * (p_matrix(muhat, B, A)
                                    * ((1.5) * f0_scratch_p2(enm, A)
                                    - (0.5) * f0_scratch_p3(enm, A)
                                    + (1.5) * f0_scratch_p1(enm, A)
                                    - (0.5) * f0_scratch(enm, A))
                                    - Sgn(tetr_mu_muhat0_R) * pmod_matrix(muhat, B, A)
                                    * (-(1.5) * f0_scratch_p1(enm, A)
                                    + (0.5) * f0_scratch(enm, A)
                                    + (1.5) * f0_scratch_p2(enm, A)
                                    - (0.5) * f0_scratch_p3(enm, A)));
                                }
                            }
                        }

                        const int nuenang = IndicesUnited(
                            nuidx, enn, B, num_species_, num_energy_bins_, num_points_);

                        flx1(m, nuenang, kk, jj, ii) =
                            ((1.5) * fminus - favg - (0.5) * fplus);
                        flx1(m, nuenang, kk, jj, ii + 1) =
                            ((0.5) * fminus + favg - (1.5) * fplus);
                    });
                  }
                });

// j-direction
  auto &flx2 = iflx.x2f;
  if (multi_d) {
    par_for_outer("radiation_femn_flux_y", DevExeSpace(), scr_size, scr_level,
                  0, nmb1, 0, nnu1, ks, ke, js, static_cast<int>(je / 2) + 1, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuidx,
                                const int k, const int j, const int i) {
                    const int kk = k;
                    const int jj = 2 * j - 2;
                    const int ii = i;

                    if (rad_mask_array_(m, kk, jj, ii)) {
                        // load scratch arrays using closure
                        auto f0_scratch = ScrArray2D<Real>(
                            member.team_scratch(scr_level), num_energy_bins_,
                            num_points_);
                        auto f0_scratch_p1 =
                            ScrArray2D<Real>(member.team_scratch(scr_level),
                                             num_energy_bins_,
                                             num_points_);
                        auto f0_scratch_p2 =
                            ScrArray2D<Real>(member.team_scratch(scr_level),
                                             num_energy_bins_,
                                             num_points_);
                        auto f0_scratch_p3 =
                            ScrArray2D<Real>(member.team_scratch(scr_level),
                                             num_energy_bins_,
                                             num_points_);
                        auto f0_scratch_m1 =
                            ScrArray2D<Real>(member.team_scratch(scr_level),
                                             num_energy_bins_,
                                             num_points_);
                        auto f0_scratch_m2 =
                            ScrArray2D<Real>(member.team_scratch(scr_level),
                                             num_energy_bins_,
                                             num_points_);
                        ApplyClosure(2, member, num_species_, num_energy_bins_,
                                     num_points_, m, nuidx, kk, jj, ii,
                                     f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                     f0_scratch_p3, f0_scratch_m1, f0_scratch_m2,
                                     m1_flag_, m1_closure_, m1_closure_fun_);
                        member.team_barrier();

                      par_for_inner(member, 0, num_energy_bins_*num_points_ - 1,
                          [&](const int ennB) {
                              const int enn = static_cast<int>(ennB / num_points_);
                              const int B = ennB - enn * num_points_;

                              Real favg = 0.;
                              Real fminus = 0.;
                              Real fplus = 0;

                              for (int muhat = 0; muhat < 4; muhat++) {
                                  for (int enm = 0; enm < num_energy_bins_; enm++) {
                                      for (int A = 0; A < num_points_; A++) {
                                          favg += (0.5) * Ven_matrix(enm, enn)
                                              * (p_matrix(muhat, B, A)
                                              * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii)
                                              * f0_scratch(enm, A)
                                              + p_matrix(muhat, B, A)
                                              * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1)
                                              * f0_scratch_p1(enm, A));

                                          Real tetr_mu_muhat0_L =
                                              0.5 * (tetr_mu_muhat0_(m, 2, muhat, kk, jj - 1, ii)
                                              + tetr_mu_muhat0_(m, 2, muhat, kk, jj, ii));

                                          fminus += (0.5) * Ven_matrix(enm, enn)
                                              * (tetr_mu_muhat0_L)
                                              * (p_matrix(muhat, B, A)
                                              * ((1.5) * f0_scratch(enm, A)
                                              - (0.5) * f0_scratch_p1(enm, A)
                                              + (1.5) * f0_scratch_m1(enm, A)
                                              - (0.5) * f0_scratch_m2(enm, A))
                                              - Sgn(tetr_mu_muhat0_L)
                                              * pmod_matrix(muhat, B, A)
                                              * (-(1.5) * f0_scratch_m1(enm, A)
                                              + (0.5) * f0_scratch_m2(enm, A)
                                              + (1.5) * f0_scratch(enm, A)
                                              - (0.5) * f0_scratch_p1(enm, A)));

                                          Real tetr_mu_muhat0_R =
                                              0.5 * (tetr_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii)
                                              + tetr_mu_muhat0_(m, 2, muhat, kk, jj + 2, ii));

                                          fplus += (0.5) * Ven_matrix(enm, enn)
                                              * (tetr_mu_muhat0_R)
                                              * (p_matrix(muhat, B, A)
                                              * ((1.5) * f0_scratch_p2(enm, A)
                                              - (0.5) * f0_scratch_p3(enm, A)
                                              + (1.5) * f0_scratch_p1(enm, A)
                                              - (0.5) * f0_scratch(enm, A))
                                              - Sgn(tetr_mu_muhat0_R)
                                              * pmod_matrix(muhat, B, A)
                                              * (-(1.5) * f0_scratch_p1(enm, A)
                                              + (0.5) * f0_scratch(enm, A)
                                              + (1.5) * f0_scratch_p2(enm, A)
                                              - (0.5) * f0_scratch_p3(enm, A)));
                                      }
                                  }
                              }

                        const int nuenang =
                            IndicesUnited(nuidx, enn, B, num_species_, num_energy_bins_, num_points_);
                        flx2(m, nuenang, kk, jj, ii) =
                            ((1.5) * fminus - favg - (0.5) * fplus);
                        flx2(m, nuenang, kk, jj + 1, ii) =
                            ((0.5) * fminus + favg - (1.5) * fplus);
                      });
                    }
                  });
  }

//--------------------------------------------------------------------------------------
// k-direction

  auto &flx3 = iflx.x3f;
  if (three_d) {
    par_for_outer("radiation_femn_flux_z", DevExeSpace(), scr_size, scr_level,
                  0, nmb1, 0, nnu1, ks, static_cast<int>(ke / 2) + 1, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuidx,
                                const int k, const int j, const int i) {
                    const int kk = 2 * k - 2;
                    const int jj = j;
                    const int ii = i;

                    if (rad_mask_array_(m, kk, jj, ii)) {

                      // load scratch arrays using closure
                      auto f0_scratch = ScrArray2D<Real>(
                          member.team_scratch(scr_level), num_energy_bins_,
                          num_points_);
                      auto f0_scratch_p1 =
                          ScrArray2D<Real>(member.team_scratch(scr_level),
                                           num_energy_bins_,
                                           num_points_);
                      auto f0_scratch_p2 =
                          ScrArray2D<Real>(member.team_scratch(scr_level),
                                           num_energy_bins_,
                                           num_points_);
                      auto f0_scratch_p3 =
                          ScrArray2D<Real>(member.team_scratch(scr_level),
                                           num_energy_bins_,
                                           num_points_);
                      auto f0_scratch_m1 =
                          ScrArray2D<Real>(member.team_scratch(scr_level),
                                           num_energy_bins_,
                                           num_points_);
                      auto f0_scratch_m2 =
                          ScrArray2D<Real>(member.team_scratch(scr_level),
                                           num_energy_bins_,
                                           num_points_);
                        ApplyClosure(3, member, num_species_, num_energy_bins_,
                                     num_points_, m, nuidx, kk, jj, ii,
                                     f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                     f0_scratch_p3, f0_scratch_m1, f0_scratch_m2,
                                     m1_flag_, m1_closure_, m1_closure_fun_);
                      member.team_barrier();

                      par_for_inner(member, 0, num_energy_bins_*num_points_ - 1,
                          [&](const int ennB) {
                              const int enn = static_cast<int>(ennB / num_points_);
                              const int B = ennB - enn * num_points_;

                              Real favg = 0.;
                              Real fminus = 0.;
                              Real fplus = 0;

                              for (int muhat = 0; muhat < 4; muhat++) {
                                  for (int enm = 0; enm < num_energy_bins_; enm++) {
                                      for (int A = 0; A < num_points_; A++) {
                                          favg += (0.5) * Ven_matrix(enm, enn)
                                              * (p_matrix(muhat, B, A)
                                              * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii)
                                              * f0_scratch(enm, A)
                                              + p_matrix(muhat, B, A)
                                              * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1)
                                              * f0_scratch_p1(enm, A));

                                          Real tetr_mu_muhat0_L =
                                              0.5 * (tetr_mu_muhat0_(m, 3, muhat, kk - 1, jj, ii)
                                              + tetr_mu_muhat0_(m, 3, muhat, kk, jj, ii));

                                          fminus += (0.5) * Ven_matrix(enm, enn)
                                              * (tetr_mu_muhat0_L)
                                              * (p_matrix(muhat, B, A)
                                              * ((1.5) * f0_scratch(enm, A)
                                              - (0.5) * f0_scratch_p1(enm, A)
                                              + (1.5) * f0_scratch_m1(enm, A)
                                              - (0.5) * f0_scratch_m2(enm, A))
                                              - Sgn(tetr_mu_muhat0_L)
                                              * pmod_matrix(muhat, B, A)
                                              * (-(1.5) * f0_scratch_m1(enm, A)
                                              + (0.5) * f0_scratch_m2(enm, A)
                                              + (1.5) * f0_scratch(enm, A)
                                              - (0.5) * f0_scratch_p1(enm, A)));

                                          Real tetr_mu_muhat0_R =
                                              0.5 * (tetr_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii)
                                                  + tetr_mu_muhat0_(m, 3, muhat, kk + 2, jj, ii));

                                          fplus += (0.5) * Ven_matrix(enm, enn)
                                              * (tetr_mu_muhat0_R)
                                              * (p_matrix(muhat, B, A)
                                              * ((1.5) * f0_scratch_p2(enm, A)
                                              - (0.5) * f0_scratch_p3(enm, A)
                                              + (1.5) * f0_scratch_p1(enm, A)
                                              - (0.5) * f0_scratch(enm, A))
                                              - Sgn(tetr_mu_muhat0_R)
                                              * pmod_matrix(muhat, B, A)
                                              * (-(1.5) * f0_scratch_p1(enm, A)
                                              + (0.5) * f0_scratch(enm, A)
                                              + (1.5) * f0_scratch_p2(enm, A)
                                              - (0.5) * f0_scratch_p3(enm, A)));
                                      }
                                  }
                              }

                              const int nuenang =
                                  IndicesUnited(nuidx, enn, B, num_species_,
                                  num_energy_bins_, num_points_);
                              flx3(m, nuenang, kk, jj, ii) =
                                  ((1.5) * fminus - favg - (0.5) * fplus);
                              flx3(m, nuenang, kk + 1, jj, ii) =
                                  ((0.5) * fminus + favg - (1.5) * fplus);
                          });
                    }
                  });
  }
  return TaskStatus::complete;
}
} // namespace radiationfemn
