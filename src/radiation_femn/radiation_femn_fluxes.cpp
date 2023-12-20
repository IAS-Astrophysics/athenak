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

namespace radiationfemn {

TaskStatus RadiationFEMN::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  int num_points_ = pmy_pack->pradfemn->num_points;
  int num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  int num_species_ = pmy_pack->pradfemn->num_species;
  int npts1 = pmy_pack->pradfemn->num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  bool &m1_flag_ = pmy_pack->pradfemn->m1_flag;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &energy_grid_ = pmy_pack->pradfemn->energy_grid;
  auto &P_matrix_ = pmy_pack->pradfemn->P_matrix;
  auto &Pmod_matrix_ = pmy_pack->pradfemn->Pmod_matrix;
  auto &sqrt_det_g_ = pmy_pack->pradfemn->sqrt_det_g;
  auto &L_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;

  //--------------------------------------------------------------------------------------
  // i-direction

  int scr_level = 0;
  int scr_size = ScrArray1D<Real>::shmem_size(num_points) * 6;
  auto &flx1 = iflx.x1f;
  Kokkos::deep_copy(flx1, 0.);
  par_for_outer("radiation_femn_flux_x", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, npts1, ks, ke, js, je, is, int(ie / 2) + 1,
                KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuenang, const int k, const int j, const int i) {

                  auto kk = k;
                  auto jj = j;
                  auto ii = 2 * i - 2;

                  RadiationFEMNPhaseIndices idcs = IndicesComponent(nuenang, num_points_, num_energy_bins_, num_species_);
                  int species = idcs.nuidx;
                  int en = idcs.enidx;
                  int B = idcs.angidx;

                  // load scratch arrays using closure
                  ScrArray1D<Real> f0_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_p1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_p2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_p3 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_m1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ScrArray1D<Real> f0_scratch_m2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                  ApplyClosureX(member, num_species_, num_energy_bins_, num_points_, m, species, en, kk, jj, ii, f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2, f0_scratch_p3, f0_scratch_m1, f0_scratch_m2, m1_flag_);
                  member.team_barrier();

                  // factor from energy contribution
                  Real Ven = (1. / 3.) * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3));

                  // compute quantities at the left and right boundaries
                  Real sqrt_det_g_L = 1.5 * sqrt_det_g_(m, kk, jj, ii) - 0.5 * sqrt_det_g_(m, kk, jj, ii + 1);
                  Real sqrt_det_g_R = -0.5 * sqrt_det_g_(m, kk, jj, ii) + 1.5 * sqrt_det_g_(m, kk, jj, ii + 1);

                  // average flux of element
                  Real Favg = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                    int muhat = int(muhatA / num_points_);
                    int A = muhatA - muhat * num_points_;

                    partial_sum += (0.5) * Ven * (P_matrix_(muhat, B, A) * sqrt_det_g_(m, kk, jj, ii) * L_mu_muhat0_(m, 1, muhat, kk, jj, ii) * f0_scratch(A)
                        + P_matrix_(muhat, B, A) * sqrt_det_g_(m, kk, jj, ii + 1) * L_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1) * f0_scratch_p1(A));
                  }, Favg);
                  member.team_barrier();

                  // flux at left boundary
                  Real Fminus = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                    int muhat = int(muhatA / num_points_);
                    int A = muhatA - muhat * num_points_;

                    Real L_mu_muhat0_L = 1.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii) - 0.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1);

                    partial_sum += (0.5) * Ven * (sqrt_det_g_L * L_mu_muhat0_L) * (P_matrix_(muhat, B, A)
                        * ((1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A) + (1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A))
                        - std::copysign(1.0, L_mu_muhat0_L) * Pmod_matrix_(muhat, B, A)
                            * ((1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A) - (1.5) * f0_scratch(A) + (0.5) * f0_scratch_p1(A)));
                  }, Fminus);
                  member.team_barrier();

                  // flux at right boundary
                  Real Fplus = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                    int muhat = int(muhatA / num_points_);
                    int A = muhatA - muhat * num_points_;

                    Real L_mu_muhat0_R = -0.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii) + 1.5 * L_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1);

                    partial_sum += (0.5) * Ven * (sqrt_det_g_R * L_mu_muhat0_R) * (P_matrix_(muhat, B, A) *
                        ((1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A) + (1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A))
                        - std::copysign(1.0, L_mu_muhat0_R) * Pmod_matrix_(muhat, B, A) *
                            ((1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A) - (1.5) * f0_scratch_p2(A) + (0.5) * f0_scratch_p3(A)));

                  }, Fplus);
                  member.team_barrier();

                  flx1(m, nuenang, kk, jj, ii) = ((1.5) * Fminus - Favg - (0.5) * Fplus);
                  flx1(m, nuenang, kk, jj, ii + 1) = ((0.5) * Fminus + Favg - (1.5) * Fplus);
                });

//--------------------------------------------------------------------------------------
// j-direction

  scr_level = 0;
  scr_size = ScrArray1D<Real>::shmem_size(num_points) * 6;
  auto &flx2 = iflx.x2f;
  Kokkos::deep_copy(flx2, 0.);
  if (multi_d) {
    par_for_outer("radiation_femn_flux_y", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, npts1, ks, ke, js, int(je / 2) + 1, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int enang, const int k, const int j, const int i) {

                    auto kk = k;
                    auto jj = 2 * j - 2;
                    auto ii = i;

                    RadiationFEMNPhaseIndices idcs = IndicesComponent(enang, num_points_, num_energy_bins_, num_species_);
                    int species = idcs.nuidx;
                    int en = idcs.enidx;
                    int B = idcs.angidx;

                    // load scratch arrays using closure
                    ScrArray1D<Real> f0_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p3 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ApplyClosureY(member, num_species_, num_energy_bins_, num_points_, m, species, en, kk, jj, ii, f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2, f0_scratch_p3, f0_scratch_m1, f0_scratch_m2, m1_flag_);
                    member.team_barrier();

                    // factor from energy contribution
                    Real Ven = (1. / 3.) * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3));

                    // compute quantities at the left and right boundaries
                    Real sqrt_det_g_L = 1.5 * sqrt_det_g_(m, kk, jj, ii) - 0.5 * sqrt_det_g_(m, kk, jj + 1, ii);
                    Real sqrt_det_g_R = -0.5 * sqrt_det_g_(m, kk, jj, ii) + 1.5 * sqrt_det_g_(m, kk, jj + 1, ii);

                    // average flux of element
                    Real Favg = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      partial_sum += (0.5) * Ven * (P_matrix_(muhat, B, A) * sqrt_det_g_(m, kk, jj, ii) * L_mu_muhat0_(m, 2, muhat, kk, jj, ii) * f0_scratch(A)
                          + P_matrix_(muhat, B, A) * sqrt_det_g_(m, kk, jj + 1, ii) * L_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii) * f0_scratch_p1(A));

                    }, Favg);
                    member.team_barrier();

                    // flux at left boundary
                    Real Fminus = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      Real L_mu_muhat0_L = 1.5 * L_mu_muhat0_(m, 2, muhat, kk, jj, ii) - 0.5 * L_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii);

                      partial_sum += (0.5) * Ven * (sqrt_det_g_L * L_mu_muhat0_L) * (P_matrix_(muhat, B, A)
                          * ((1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A) + (1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A))
                          - std::copysign(1.0, L_mu_muhat0_L) * Pmod_matrix_(muhat, B, A)
                              * ((1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A) - (1.5) * f0_scratch(A) + (0.5) * f0_scratch_p1(A)));

                    }, Fminus);
                    member.team_barrier();

                    // flux at right boundary
                    Real Fplus = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      Real L_mu_muhat0_R = -0.5 * L_mu_muhat0_(m, 2, muhat, kk, jj, ii) + 1.5 * L_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii);

                      partial_sum += (0.5) * Ven * (sqrt_det_g_R * L_mu_muhat0_R) * (P_matrix_(muhat, B, A) *
                          ((1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A) + (1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A))
                          - std::copysign(1.0, L_mu_muhat0_R) * Pmod_matrix_(muhat, B, A) *
                              ((1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A) - (1.5) * f0_scratch_p2(A) + (0.5) * f0_scratch_p3(A)));

                    }, Fplus);
                    member.team_barrier();

                    flx2(m, enang, kk, jj, ii) = ((1.5) * Fminus - Favg - (0.5) * Fplus);
                    flx2(m, enang, kk, jj + 1, ii) = ((0.5) * Fminus + Favg - (1.5) * Fplus);
                  });
  }

//--------------------------------------------------------------------------------------
// k-direction

  scr_level = 0;
  scr_size = ScrArray1D<Real>::shmem_size(num_points) * 6;
  auto &flx3 = iflx.x3f;
  Kokkos::deep_copy(flx3, 0.);
  if (three_d) {
    par_for_outer("radiation_femn_flux_z", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, npts1, ks, int(ke / 2) + 1, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int enang, const int k, const int j, const int i) {

                    auto kk = 2 * k - 2;
                    auto jj = j;
                    auto ii = i;

                    RadiationFEMNPhaseIndices idcs = IndicesComponent(enang, num_points_, num_energy_bins_, num_species_);
                    int species = idcs.nuidx;
                    int en = idcs.enidx;
                    int B = idcs.angidx;

                    // load scratch arrays using closure
                    ScrArray1D<Real> f0_scratch = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p3 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m1 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m2 = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ApplyClosureZ(member, num_species_, num_energy_bins_, num_points_, m, species, en, kk, jj, ii, f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2, f0_scratch_p3, f0_scratch_m1, f0_scratch_m2, m1_flag_);
                    member.team_barrier();

                    // factor from energy contribution
                    Real Ven = (1. / 3.) * (pow(energy_grid_(en + 1), 3) - pow(energy_grid_(en), 3));

                    // compute quantities at the left and right boundaries
                    Real sqrt_det_g_L = 1.5 * sqrt_det_g_(m, kk, jj, ii) - 0.5 * sqrt_det_g_(m, kk + 1, jj, ii);
                    Real sqrt_det_g_R = -0.5 * sqrt_det_g_(m, kk, jj, ii) + 1.5 * sqrt_det_g_(m, kk + 1, jj, ii);

                    // average flux of element
                    Real Favg = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      partial_sum += (0.5) * Ven * (P_matrix_(muhat, B, A) * sqrt_det_g_(m, kk, jj, ii) * L_mu_muhat0_(m, 3, muhat, kk, jj, ii) * f0_scratch(A)
                          + P_matrix_(muhat, B, A) * sqrt_det_g_(m, kk + 1, jj, ii) * L_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii) * f0_scratch_p1(A));

                    }, Favg);
                    member.team_barrier();

                    // flux at left boundary
                    Real Fminus = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      Real L_mu_muhat0_L = 1.5 * L_mu_muhat0_(m, 3, muhat, kk, jj, ii) - 0.5 * L_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii);

                      partial_sum += (0.5) * Ven * (sqrt_det_g_L * L_mu_muhat0_L) * (P_matrix_(muhat, B, A)
                          * ((1.5) * f0_scratch(A) - (0.5) * f0_scratch_p1(A) + (1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A))
                          - std::copysign(1.0, L_mu_muhat0_L) * Pmod_matrix_(muhat, B, A)
                              * ((1.5) * f0_scratch_m1(A) - (0.5) * f0_scratch_m2(A) - (1.5) * f0_scratch(A) + (0.5) * f0_scratch_p1(A)));
                    }, Fminus);
                    member.team_barrier();

                    // flux at right boundary
                    Real Fplus = 0.;
                    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 4 * num_points_), [&](const int muhatA, Real &partial_sum) {
                      int muhat = int(muhatA / num_points_);
                      int A = muhatA - muhat * num_points_;

                      Real L_mu_muhat0_R = -0.5 * L_mu_muhat0_(m, 3, muhat, kk, jj, ii) + 1.5 * L_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii);

                      partial_sum += (0.5) * Ven * (sqrt_det_g_R * L_mu_muhat0_R) * (P_matrix_(muhat, B, A) *
                          ((1.5) * f0_scratch_p2(A) - (0.5) * f0_scratch_p3(A) + (1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A))
                          - std::copysign(1.0, L_mu_muhat0_R) * Pmod_matrix_(muhat, B, A) *
                              ((1.5) * f0_scratch_p1(A) - (0.5) * f0_scratch(A) - (1.5) * f0_scratch_p2(A) + (0.5) * f0_scratch_p3(A)));

                    }, Fplus);
                    member.team_barrier();

                    flx3(m, enang, kk, jj, ii) = ((1.5) * Fminus - Favg - (0.5) * Fplus);
                    flx3(m, enang, kk + 1, jj, ii) = ((0.5) * Fminus + Favg - (1.5) * Fplus);
                  });
  }

  return TaskStatus::complete;
}

} // namespace radiationfemn