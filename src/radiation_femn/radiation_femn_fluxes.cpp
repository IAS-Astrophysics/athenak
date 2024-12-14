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
  int nnuenpts1 = num_species_ * num_energy_bins_ - 1;
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

  //--------------------------------------------------------------------------------------
  // i-direction
  int scr_level = 0;
  int scr_size = ScrArray1D<Real>::shmem_size(num_points) * 6;
  auto &flx1 = iflx.x1f;
  par_for_outer("radiation_femn_flux_x", DevExeSpace(), scr_size, scr_level,
                0, nmb1, 0, nnuenpts1, ks, ke, js, je, is, static_cast<int>(ie / 2) + 1,
                KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuen,
                              const int k, const int j, const int i) {
                  const int kk = k;
                  const int jj = j;
                  const int ii = 2 * i - 2;

                  if (rad_mask_array_(m, kk, jj, ii)) {
                    RadiationFEMNPhaseIndices nuenidx =
                        NuEnIndicesComponent(nuen, num_species_, num_energy_bins_);
                    const int enidx = nuenidx.enidx;
                    const int nuidx = nuenidx.nuidx;

                    // load scratch arrays using closure
                    ScrArray1D<Real> f0_scratch =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p1 =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p2 =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_p3 =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m1 =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ScrArray1D<Real> f0_scratch_m2 =
                        ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                    ApplyClosureX(member, num_species_, num_energy_bins_, num_points_,
                                  m, nuidx, enidx, kk, jj, ii,
                                  f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                  f0_scratch_p3, f0_scratch_m1, f0_scratch_m2,
                                  m1_flag_, m1_closure_, m1_closure_fun_);
                    member.team_barrier();

                    par_for_inner(member, 0, num_points_ - 1, [&](const int B) {
                      Real favg = 0.;
                      Real fminus = 0.;
                      Real fplus = 0;

                      for (int muhatA = 0; muhatA < 4 * num_points_; muhatA++) {
                        const int muhat = static_cast<int>(muhatA / num_points_);
                        const int A = muhatA - muhat * num_points_;

                        favg += (0.5) * (p_matrix(muhat, B, A)
                            * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii) * f0_scratch(A)
                            + p_matrix(muhat, B, A)
                                * tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1)
                                * f0_scratch_p1(A));

                        const Real tetr_mu_muhat0_L =
                            0.5 * (tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii - 1)
                                + tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii));
                        fminus += (0.5) * (tetr_mu_muhat0_L)
                            * (p_matrix(muhat, B, A) * ((1.5) * f0_scratch(A)
                                - (0.5) * f0_scratch_p1(A)
                                + (1.5) * f0_scratch_m1(A)
                                - (0.5) * f0_scratch_m2(A))
                                - Sgn(tetr_mu_muhat0_L) * pmod_matrix(muhat, B, A)
                                    * (-(1.5) * f0_scratch_m1(A)
                                        + (0.5) * f0_scratch_m2(A)
                                        + (1.5) * f0_scratch(A)
                                        - (0.5) * f0_scratch_p1(A)));

                        const Real tetr_mu_muhat0_R =
                            0.5 * (tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 1)
                                + tetr_mu_muhat0_(m, 1, muhat, kk, jj, ii + 2));
                        fplus += (0.5) * (tetr_mu_muhat0_R)
                            * (p_matrix(muhat, B, A) * ((1.5) * f0_scratch_p2(A)
                                - (0.5) * f0_scratch_p3(A)
                                + (1.5) * f0_scratch_p1(A)
                                - (0.5) * f0_scratch(A))
                                - Sgn(tetr_mu_muhat0_R) * pmod_matrix(muhat, B, A)
                                    * (-(1.5) * f0_scratch_p1(A)
                                        + (0.5) * f0_scratch(A)
                                        + (1.5) * f0_scratch_p2(A)
                                        - (0.5) * f0_scratch_p3(A)));
                      }
                      const int nuenang = IndicesUnited(nuidx, enidx, B, num_species_,
                                                        num_energy_bins_, num_points_);
                      flx1(m, nuenang, kk, jj, ii) =
                          ((1.5) * fminus - favg - (0.5) * fplus) * Ven_matrix(0,0); //@TODO: fix later!
                      flx1(m, nuenang, kk, jj, ii + 1) =
                          ((0.5) * fminus + favg - (1.5) * fplus) * Ven_matrix(0,0);
                    });
                  }
                });
//--------------------------------------------------------------------------------------
// j-direction
  auto &flx2 = iflx.x2f;
  if (multi_d) {
    par_for_outer("radiation_femn_flux_y", DevExeSpace(), scr_size, scr_level,
                  0, nmb1, 0, nnuenpts1, ks, ke, js, static_cast<int>(je / 2) + 1, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuen,
                                const int k, const int j, const int i) {
                    const int kk = k;
                    const int jj = 2 * j - 2;
                    const int ii = i;

                    if (rad_mask_array_(m, kk, jj, ii)) {
                      RadiationFEMNPhaseIndices nuenidx =
                          NuEnIndicesComponent(nuen, num_species_, num_energy_bins_);
                      const int enidx = nuenidx.enidx;
                      const int nuidx = nuenidx.nuidx;

                      // load scratch arrays using closure
                      ScrArray1D<Real> f0_scratch =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_p1 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_p2 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_p3 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_m1 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_m2 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ApplyClosureY(member, num_species_, num_energy_bins_, num_points_,
                                    m, nuidx, enidx, kk, jj, ii,
                                    f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                    f0_scratch_p3, f0_scratch_m1, f0_scratch_m2,
                                    m1_flag_, m1_closure_, m1_closure_fun_);
                      member.team_barrier();

                      par_for_inner(member, 0, num_points_ - 1, [&](const int B) {
                        Real favg = 0.;
                        Real fminus = 0.;
                        Real fplus = 0;

                        for (int muhatA = 0; muhatA < 4 * num_points_; muhatA++) {
                          const int muhat = static_cast<int>(muhatA / num_points_);
                          const int A = muhatA - muhat * num_points_;

                          favg += (0.5) * (p_matrix(muhat, B, A)
                              * tetr_mu_muhat0_(m, 2, muhat, kk, jj, ii) * f0_scratch(A)
                              + p_matrix(muhat, B, A)
                                  * tetr_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii)
                                  * f0_scratch_p1(A));

                          Real tetr_mu_muhat0_L =
                              0.5 * (tetr_mu_muhat0_(m, 2, muhat, kk, jj - 1, ii)
                                  + tetr_mu_muhat0_(m, 2, muhat, kk, jj, ii));
                          fminus += (0.5) * (tetr_mu_muhat0_L)
                              * (p_matrix(muhat, B, A)
                                  * ((1.5) * f0_scratch(A)
                                      - (0.5) * f0_scratch_p1(A)
                                      + (1.5) * f0_scratch_m1(A)
                                      - (0.5) * f0_scratch_m2(A))
                                  - Sgn(tetr_mu_muhat0_L) * pmod_matrix(muhat, B, A)
                                      * (-(1.5) * f0_scratch_m1(A)
                                          + (0.5) * f0_scratch_m2(A)
                                          + (1.5) * f0_scratch(A)
                                          - (0.5) * f0_scratch_p1(A)));

                          Real tetr_mu_muhat0_R =
                              0.5 * (tetr_mu_muhat0_(m, 2, muhat, kk, jj + 1, ii)
                                  + tetr_mu_muhat0_(m, 2, muhat, kk, jj + 2, ii));
                          fplus += (0.5) * (tetr_mu_muhat0_R)
                              * (p_matrix(muhat, B, A)
                                  * ((1.5) * f0_scratch_p2(A)
                                      - (0.5) * f0_scratch_p3(A)
                                      + (1.5) * f0_scratch_p1(A)
                                      - (0.5) * f0_scratch(A))
                                  - Sgn(tetr_mu_muhat0_R) * pmod_matrix(muhat, B, A) *
                                      (-(1.5) * f0_scratch_p1(A)
                                          + (0.5) * f0_scratch(A)
                                          + (1.5) * f0_scratch_p2(A)
                                          - (0.5) * f0_scratch_p3(A)));
                        }

                        const int nuenang = IndicesUnited(nuidx, enidx, B, num_species_,
                                                          num_energy_bins_, num_points_);
                        flx2(m, nuenang, kk, jj, ii) =
                            ((1.5) * fminus - favg - (0.5) * fplus) * Ven_matrix(0,0); //@TODO: fix later!
                        flx2(m, nuenang, kk, jj + 1, ii) =
                            ((0.5) * fminus + favg - (1.5) * fplus) * Ven_matrix(0,0);
                      });
                    }
                  });
  }

//--------------------------------------------------------------------------------------
// k-direction

  auto &flx3 = iflx.x3f;
  if (three_d) {
    par_for_outer("radiation_femn_flux_z", DevExeSpace(), scr_size, scr_level,
                  0, nmb1, 0, nnuenpts1, ks, static_cast<int>(ke / 2) + 1, js, je, is, ie,
                  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int nuen,
                                const int k, const int j, const int i) {
                    const int kk = 2 * k - 2;
                    const int jj = j;
                    const int ii = i;

                    if (rad_mask_array_(m, kk, jj, ii)) {
                      RadiationFEMNPhaseIndices nuenidx =
                          NuEnIndicesComponent(nuen, num_species_, num_energy_bins_);
                      int enidx = nuenidx.enidx;
                      int nuidx = nuenidx.nuidx;

                      // load scratch arrays using closure
                      ScrArray1D<Real> f0_scratch =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_p1 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_p2 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_p3 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_m1 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ScrArray1D<Real> f0_scratch_m2 =
                          ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                      ApplyClosureZ(member, num_species_, num_energy_bins_, num_points_,
                                    m, nuidx, enidx, kk, jj, ii,
                                    f0_, f0_scratch, f0_scratch_p1, f0_scratch_p2,
                                    f0_scratch_p3, f0_scratch_m1, f0_scratch_m2,
                                    m1_flag_, m1_closure_, m1_closure_fun_);
                      member.team_barrier();

                      par_for_inner(member, 0, num_points_ - 1, [&](const int B) {
                        Real favg = 0.;
                        Real fminus = 0.;
                        Real fplus = 0;

                        for (int muhatA = 0; muhatA < 4 * num_points_; muhatA++) {
                          const int muhat = static_cast<int>(muhatA / num_points_);
                          const int A = muhatA - muhat * num_points_;

                          favg += (0.5) * (p_matrix(muhat, B, A)
                              * tetr_mu_muhat0_(m, 3, muhat, kk, jj, ii) * f0_scratch(A)
                              + p_matrix(muhat, B, A)
                                  * tetr_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii)
                                  * f0_scratch_p1(A));

                          const Real tetr_mu_muhat0_L =
                              0.5 * (tetr_mu_muhat0_(m, 3, muhat, kk - 1, jj, ii)
                                  + tetr_mu_muhat0_(m, 3, muhat, kk, jj, ii));
                          fminus += (0.5) * (tetr_mu_muhat0_L)
                              * (p_matrix(muhat, B, A)
                                  * ((1.5) * f0_scratch(A)
                                      - (0.5) * f0_scratch_p1(A)
                                      + (1.5) * f0_scratch_m1(A)
                                      - (0.5) * f0_scratch_m2(A))
                                  - Sgn(tetr_mu_muhat0_L) * pmod_matrix(muhat, B, A)
                                      * (-(1.5) * f0_scratch_m1(A)
                                          + (0.5) * f0_scratch_m2(A)
                                          + (1.5) * f0_scratch(A)
                                          - (0.5) * f0_scratch_p1(A)));

                          const Real tetr_mu_muhat0_R = 0.5
                              * (tetr_mu_muhat0_(m, 3, muhat, kk + 1, jj, ii)
                                  + tetr_mu_muhat0_(m, 3, muhat, kk + 2, jj, ii));
                          fplus += (0.5) * (tetr_mu_muhat0_R)
                              * (p_matrix(muhat, B, A)
                                  * ((1.5) * f0_scratch_p2(A)
                                      - (0.5) * f0_scratch_p3(A)
                                      + (1.5) * f0_scratch_p1(A)
                                      - (0.5) * f0_scratch(A))
                                  - Sgn(tetr_mu_muhat0_R) * pmod_matrix(muhat, B, A)
                                      * (-(1.5) * f0_scratch_p1(A)
                                          + (0.5) * f0_scratch(A)
                                          + (1.5) * f0_scratch_p2(A)
                                          - (0.5) * f0_scratch_p3(A)));
                        }

                        const int nuenang = IndicesUnited(nuidx, enidx, B, num_species_,
                                                          num_energy_bins_, num_points_);
                        flx3(m, nuenang, kk, jj, ii) =
                            ((1.5) * fminus - favg - (0.5) * fplus)  * Ven_matrix(0,0); // @TODO: fix later!
                        flx3(m, nuenang, kk + 1, jj, ii) =
                            ((0.5) * fminus + favg - (1.5) * fplus)  * Ven_matrix(0,0);
                      });
                    }
                  });
  }
  return TaskStatus::complete;
}
} // namespace radiationfemn
