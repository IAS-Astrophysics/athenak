//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

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
  int nnuenpts1 = num_species_ * num_energy_bins_ - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  bool &m1_flag_ = pmy_pack->pradfemn->m1_flag;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &energy_grid_ = pmy_pack->pradfemn->energy_grid;
  auto &P_matrix_ = pmy_pack->pradfemn->P_matrix;
  auto &Pmod_matrix_ = pmy_pack->pradfemn->Pmod_matrix;
  auto &L_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  //--------------------------------------------------------------------------------------
  // i-direction

  int scr_level = 0;
  int scr_size = ScrArray1D<Real>::shmem_size(num_points) * 6;
  auto &flx1 = iflx.x1f;
  par_for_outer("radiation_femn_flux_x", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, nnuenpts1, ks, ke, js, je, is, int(ie / 2) + 1,
                KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuen, const int k, const int j, const int i) {

                  auto kk = k;
                  auto jj = j;
                  auto ii = 2 * i - 2;

                  RadiationFEMNPhaseIndices nuenidx = NuEnIndicesComponent(nuen, num_species_, num_energy_bins_);
                  int enidx = nuenidx.enidx;
                  int nuidx = nuenidx.nuidx;

                  Real sqrt_det_g_ii = adm.alpha(m, kk, jj, ii) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, kk, jj, ii), adm.g_dd(m, 0, 1, kk, jj, ii),
                                                                                       adm.g_dd(m, 0, 2, kk, jj, ii), adm.g_dd(m, 1, 1, kk, jj, ii),
                                                                                       adm.g_dd(m, 1, 2, kk, jj, ii), adm.g_dd(m, 2, 2, kk, jj, ii)));
                  Real sqrt_det_g_iip1 = adm.alpha(m, kk, jj, ii + 1) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, kk, jj, ii + 1), adm.g_dd(m, 0, 1, kk, jj, ii + 1),
                                                                                             adm.g_dd(m, 0, 2, kk, jj, ii + 1), adm.g_dd(m, 1, 1, kk, jj, ii + 1),
                                                                                             adm.g_dd(m, 1, 2, kk, jj, ii + 1), adm.g_dd(m, 2, 2, kk, jj, ii + 1)));

                  Real sqrt_det_g_L = 1.5 * sqrt_det_g_ii - 0.5 * sqrt_det_g_iip1;
                  Real sqrt_det_g_R = -0.5 * sqrt_det_g_ii + 1.5 * sqrt_det_g_iip1;

                  // load scratch arrays using closure
                  ScrArray1D<Real> f0_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_p1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_p2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_p3 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_m1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_m2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ApplyClosureX(member, num_species_, num_energy_bins_, num_points_, m, nuidx, enidx, kk, jj, ii, f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                f0_scratch_p3, f0_scratch_m1, f0_scratch_m2, m1_flag_);
                  member.team_barrier();

                  par_for_inner(member, 0, num_points_ - 1, [&](const int B) {

                    Real Favg = 0.;
                    Real Fminus = 0.;
                    Real Fplus = 0;

                    for (int muhatA = 0; muhatA < 4 * num_points_; muhatA++) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      Favg += (0.5) * (P_matrix_(muhat, B, A) * sqrt_det_g_ii * L_mu_muhat0_(m, 1, muhat, kk, jj, ii) * f0_scratch(A)
                          + P_matrix_(muhat, B, A) * sqrt_det_g_iip1 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1) * f0_scratch_p1(A));

                      Real L_mu_muhat0_L = 1.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii) - 0.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1);
                      Fminus += (0.5) * (sqrt_det_g_L * L_mu_muhat0_L) * (P_matrix_(muhat, B, A)
                          * ((1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A) + (1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A))
                          - Sgn(L_mu_muhat0_L) * Pmod_matrix_(muhat, B, A)
                              * (-(1.5) * f0_scratch_m1(A) + (0.5) * f0_scratch_m2(A) + (1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A)));

                      Real L_mu_muhat0_R = -0.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii) + 1.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1);
                      Fplus += (0.5) * (sqrt_det_g_R * L_mu_muhat0_R) * (P_matrix_(muhat, B, A) *
                          ((1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A) + (1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A))
                          - Sgn(L_mu_muhat0_R) * Pmod_matrix_(muhat, B, A) *
                              (-(1.5) * f0_scratch_p1(A) + (0.5) * f0_scratch(A) + (1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A)));
                    }

                    int nuenang = IndicesUnited(nuidx, enidx, B, num_species_, num_energy_bins_, num_points_);
                    flx1(m, nuenang, kk, jj, ii) = ((1.5) * Fminus - Favg - (0.5) * Fplus);
                    flx1(m, nuenang, kk, jj, ii + 1) = ((0.5) * Fminus + Favg - (1.5) * Fplus);

                  });

                });
//--------------------------------------------------------------------------------------
// j-direction

  auto &flx2 = iflx.x2f;
  if (multi_d) {
    par_for_outer("radiation_femn_flux_y", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, nnuenpts1, ks, ke, js, int(je / 2) + 1, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuen, const int k, const int j, const int i) {

                    auto kk = k;
                    auto jj = 2 * j - 2;
                    auto ii = i;

                    RadiationFEMNPhaseIndices nuenidx = NuEnIndicesComponent(nuen, num_species_, num_energy_bins_);
                    int enidx = nuenidx.enidx;
                    int nuidx = nuenidx.nuidx;

                    Real sqrt_det_g_jj = adm.alpha(m, kk, jj, ii) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, kk, jj, ii), adm.g_dd(m, 0, 1, kk, jj, ii),
                                                                                         adm.g_dd(m, 0, 2, kk, jj, ii), adm.g_dd(m, 1, 1, kk, jj, ii),
                                                                                         adm.g_dd(m, 1, 2, kk, jj, ii), adm.g_dd(m, 2, 2, kk, jj, ii)));
                    Real sqrt_det_g_jjp1 = adm.alpha(m, kk, jj + 1, ii) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, kk, jj + 1, ii), adm.g_dd(m, 0, 1, kk, jj + 1, ii),
                                                                                               adm.g_dd(m, 0, 2, kk, jj + 1, ii), adm.g_dd(m, 1, 1, kk, jj + 1, ii),
                                                                                               adm.g_dd(m, 1, 2, kk, jj + 1, ii), adm.g_dd(m, 2, 2, kk, jj + 1, ii)));

                    Real sqrt_det_g_L = 1.5 * sqrt_det_g_jj - 0.5 * sqrt_det_g_jjp1;
                    Real sqrt_det_g_R = -0.5 * sqrt_det_g_jj + 1.5 * sqrt_det_g_jjp1;

                    // load scratch arrays using closure
                    ScrArray1D<Real> f0_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p3 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ApplyClosureY(member, num_species_, num_energy_bins_, num_points_, m, nuidx, enidx, kk, jj, ii, f0_, f0_scratch,
                                  f0_scratch_p1, f0_scratch_p2, f0_scratch_p3, f0_scratch_m1, f0_scratch_m2, m1_flag_);
                    member.team_barrier();

                    par_for_inner(member, 0, num_points_ - 1, [&](const int B) {

                      Real Favg = 0.;
                      Real Fminus = 0.;
                      Real Fplus = 0;

                      for (int muhatA = 0; muhatA < 4 * num_points_; muhatA++) {
                        int muhat = int(muhatA / num_points_);
                        int A = muhatA - muhat * num_points_;

                        Favg += (0.5) * (P_matrix_(muhat, B, A) * sqrt_det_g_jj * L_mu_muhat0_(m, 2, muhat, kk, jj, ii) * f0_scratch(A)
                            + P_matrix_(muhat, B, A) * sqrt_det_g_jjp1 * L_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii) * f0_scratch_p1(A));

                        Real L_mu_muhat0_L = 1.5 * L_mu_muhat0_(m, 2, muhat, kk, jj, ii) - 0.5 * L_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii);
                        Fminus += (0.5) * (sqrt_det_g_L * L_mu_muhat0_L) * (P_matrix_(muhat, B, A)
                            * ((1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A) + (1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A))
                            - Sgn(L_mu_muhat0_L) * Pmod_matrix_(muhat, B, A)
                                * (-(1.5) * f0_scratch_m1(A) + (0.5) * f0_scratch_m2(A) + (1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A)));

                        Real L_mu_muhat0_R = -0.5 * L_mu_muhat0_(m, 2, muhat, kk, jj, ii) + 1.5 * L_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii);
                        Fplus += (0.5) * (sqrt_det_g_R * L_mu_muhat0_R) * (P_matrix_(muhat, B, A) *
                            ((1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A) + (1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A))
                            - Sgn(L_mu_muhat0_R) * Pmod_matrix_(muhat, B, A) *
                                (-(1.5) * f0_scratch_p1(A) + (0.5) * f0_scratch(A) + (1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A)));
                      }

                      int nuenang = IndicesUnited(nuidx, enidx, B, num_species_, num_energy_bins_, num_points_);
                      flx2(m, nuenang, kk, jj, ii) = ((1.5) * Fminus - Favg - (0.5) * Fplus);
                      flx2(m, nuenang, kk, jj + 1, ii) = ((0.5) * Fminus + Favg - (1.5) * Fplus);

                    });

                  });

  }

//--------------------------------------------------------------------------------------
// k-direction

  auto &flx3 = iflx.x3f;
  if (three_d) {
    par_for_outer("radiation_femn_flux_z", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, nnuenpts1, ks, int(ke / 2) + 1, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuen, const int k, const int j, const int i) {

                    auto kk = 2 * k - 2;
                    auto jj = j;
                    auto ii = i;

                    RadiationFEMNPhaseIndices nuenidx = NuEnIndicesComponent(nuen, num_species_, num_energy_bins_);
                    int enidx = nuenidx.enidx;
                    int nuidx = nuenidx.nuidx;

                    Real sqrt_det_g_kk = adm.alpha(m, kk, jj, ii) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, kk, jj, ii), adm.g_dd(m, 0, 1, kk, jj, ii),
                                                                                         adm.g_dd(m, 0, 2, kk, jj, ii), adm.g_dd(m, 1, 1, kk, jj, ii),
                                                                                         adm.g_dd(m, 1, 2, kk, jj, ii), adm.g_dd(m, 2, 2, kk, jj, ii)));
                    Real sqrt_det_g_kkp1 = adm.alpha(m, kk + 1, jj, ii) * sqrt(adm::SpatialDet(adm.g_dd(m, 0, 0, kk + 1, jj, ii), adm.g_dd(m, 0, 1, kk + 1, jj, ii),
                                                                                               adm.g_dd(m, 0, 2, kk + 1, jj, ii), adm.g_dd(m, 1, 1, kk + 1, jj, ii),
                                                                                               adm.g_dd(m, 1, 2, kk + 1, jj, ii), adm.g_dd(m, 2, 2, kk + 1, jj, ii)));

                    // compute quantities at the left and right boundaries
                    Real sqrt_det_g_L = 1.5 * sqrt_det_g_kk - 0.5 * sqrt_det_g_kkp1;
                    Real sqrt_det_g_R = -0.5 * sqrt_det_g_kk + 1.5 * sqrt_det_g_kkp1;

                    // load scratch arrays using closure
                    ScrArray1D<Real> f0_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p3 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ApplyClosureZ(member, num_species_, num_energy_bins_, num_points_, m, nuidx, enidx, kk, jj, ii, f0_, f0_scratch, f0_scratch_p1,
                                  f0_scratch_p2, f0_scratch_p3, f0_scratch_m1, f0_scratch_m2, m1_flag_);
                    member.team_barrier();

                    par_for_inner(member, 0, num_points_ - 1, [&](const int B) {

                      Real Favg = 0.;
                      Real Fminus = 0.;
                      Real Fplus = 0;

                      for (int muhatA = 0; muhatA < 4 * num_points_; muhatA++) {
                        int muhat = int(muhatA / num_points_);
                        int A = muhatA - muhat * num_points_;

                        Favg += (0.5) * (P_matrix_(muhat, B, A) * sqrt_det_g_kk * L_mu_muhat0_(m, 3, muhat, kk, jj, ii) * f0_scratch(A)
                            + P_matrix_(muhat, B, A) * sqrt_det_g_kk * L_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii) * f0_scratch_p1(A));

                        Real L_mu_muhat0_L = 1.5 * L_mu_muhat0_(m, 3, muhat, kk, jj, ii) - 0.5 * L_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii);

                        Fminus += (0.5) * (sqrt_det_g_L * L_mu_muhat0_L) * (P_matrix_(muhat, B, A)
                            * ((1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A) + (1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A))
                            - Sgn(L_mu_muhat0_L) * Pmod_matrix_(muhat, B, A)
                                * (-(1.5) * f0_scratch_m1(A) + (0.5) * f0_scratch_m2(A) + (1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A)));

                        Real L_mu_muhat0_R = -0.5 * L_mu_muhat0_(m, 3, muhat, kk, jj, ii) + 1.5 * L_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii);

                        Fplus += (0.5) * (sqrt_det_g_R * L_mu_muhat0_R) * (P_matrix_(muhat, B, A) *
                            ((1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A) + (1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A))
                            - Sgn(L_mu_muhat0_R) * Pmod_matrix_(muhat, B, A) *
                                (-(1.5) * f0_scratch_p1(A) + (0.5) * f0_scratch(A) + (1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A)));
                      }

                      int nuenang = IndicesUnited(nuidx, enidx, B, num_species_, num_energy_bins_, num_points_);
                      flx3(m, nuenang, kk, jj, ii) = ((1.5) * Fminus - Favg - (0.5) * Fplus);
                      flx3(m, nuenang, kk + 1, jj, ii) = ((0.5) * Fminus + Favg - (1.5) * Fplus);

                    });

                  });
  }

  return TaskStatus::complete;
}

} // namespace radiationfemn