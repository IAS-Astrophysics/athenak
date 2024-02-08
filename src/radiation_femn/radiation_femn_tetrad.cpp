//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_tetrad.cpp
//  \brief construct tetrad for radiation FEM_N

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "adm/adm.hpp"

namespace radiationfemn {
TaskStatus RadiationFEMN::TetradOrthogonalize(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  //auto &mbsize = pmy_pack->pmb->mb_size;

  auto L_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
  auto u_mu_ = pmy_pack->pradfemn->u_mu;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  Kokkos::deep_copy(L_mu_muhat0_, 0.);

  // L^mu_0 = u^mu
  par_for("radiation_femn_tetrad_compute_L_mu_0", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {
            L_mu_muhat0_(m, mu, 0, k, j, i) = u_mu_(m, mu, k, j, i);
          });

  // L^mu_1 = p_x + (d_x.L^mu_0)L^mu_0, p_x = (0,1,0,0), d_x.L^mu_0 = g_mu_nu d_x^mu L^nu_0
  par_for("radiation_femn_tetrad_compute_L_mu_1", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {

            Real g_dd[16];
            adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                 adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                 adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);

            L_mu_muhat0_(m, mu, 1, k, j, i) =
                (mu == 1) + (g_dd[4] * L_mu_muhat0_(m, 0, 0, k, j, i) + g_dd[5] * L_mu_muhat0_(m, 1, 0, k, j, i)
                    + g_dd[6] * L_mu_muhat0_(m, 2, 0, k, j, i) + g_dd[7] * L_mu_muhat0_(m, 3, 0, k, j, i))
                    * L_mu_muhat0_(m, mu, 0, k, j, i);
          });

  // L^mu_1 = L^mu_1/||L^mu_1||, ||L^mu_1|| = sqrt(g_mu_mu L^mu_1 L^nu_1)
  int scr_level = 0;
  int scr_size = 1;
  par_for_outer("radiation_femn_tetrad_normalize_L_mu_1", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int k, int j, int i) {

                  Real g_dd[16];
                  adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                       adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                       adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);

                  Real L_mu_1_norm = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 16), [&](const int munu, Real &partial_sum) {
                    int mu = int(munu / 4);
                    int nu = munu - 4 * mu;
                    partial_sum += g_dd[munu] * L_mu_muhat0_(m, mu, 1, k, j, i) * L_mu_muhat0_(m, nu, 1, k, j, i);
                  }, L_mu_1_norm);
                  member.team_barrier();
                  L_mu_1_norm = sqrt(L_mu_1_norm);

                  L_mu_muhat0_(m, 0, 1, k, j, i) = L_mu_muhat0_(m, 0, 1, k, j, i) / L_mu_1_norm;
                  L_mu_muhat0_(m, 1, 1, k, j, i) = L_mu_muhat0_(m, 1, 1, k, j, i) / L_mu_1_norm;
                  L_mu_muhat0_(m, 2, 1, k, j, i) = L_mu_muhat0_(m, 2, 1, k, j, i) / L_mu_1_norm;
                  L_mu_muhat0_(m, 3, 1, k, j, i) = L_mu_muhat0_(m, 3, 1, k, j, i) / L_mu_1_norm;
                });

  // L^mu_2 = p_y - (d_y.L^mu_1)L^mu_1 + (d_x.L^mu_0)L^mu_0, d_x = (0,1,0,0), d_y = (0,0,1,0)
  par_for("radiation_femn_tetrad_compute_L_mu_2", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {

            Real g_dd[16];
            adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                 adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                 adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);

            L_mu_muhat0_(m, mu, 2, k, j, i) =
                (mu == 2) - (g_dd[8] * L_mu_muhat0_(m, 0, 1, k, j, i) + g_dd[9] * L_mu_muhat0_(m, 1, 1, k, j, i)
                    + g_dd[10] * L_mu_muhat0_(m, 2, 1, k, j, i) + g_dd[11] * L_mu_muhat0_(m, 3, 1, k, j, i))
                    * L_mu_muhat0_(m, mu, 1, k, j, i)
                    + (g_dd[8] * L_mu_muhat0_(m, 0, 0, k, j, i) + g_dd[9] * L_mu_muhat0_(m, 1, 0, k, j, i)
                        + g_dd[10] * L_mu_muhat0_(m, 2, 0, k, j, i) + g_dd[11] * L_mu_muhat0_(m, 3, 0, k, j, i))
                        * L_mu_muhat0_(m, mu, 0, k, j, i);
          });

  // L^mu_2 = L^mu_2/||L^mu_2||, ||L^mu_2|| = sqrt(g_mu_mu L^mu_2 L^nu_2)
  par_for_outer("radiation_femn_tetrad_normalize_L_mu_2", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int k, int j, int i) {

                  Real g_dd[16];
                  adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                       adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                       adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);

                  Real L_mu_2_norm = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 16), [&](const int munu, Real &partial_sum) {
                    int mu = int(munu / 4);
                    int nu = munu - 4 * mu;
                    partial_sum += g_dd[munu] * L_mu_muhat0_(m, mu, 2, k, j, i) * L_mu_muhat0_(m, nu, 2, k, j, i);
                  }, L_mu_2_norm);
                  member.team_barrier();
                  L_mu_2_norm = sqrt(L_mu_2_norm);

                  L_mu_muhat0_(m, 0, 2, k, j, i) = L_mu_muhat0_(m, 0, 2, k, j, i) / L_mu_2_norm;
                  L_mu_muhat0_(m, 1, 2, k, j, i) = L_mu_muhat0_(m, 1, 2, k, j, i) / L_mu_2_norm;
                  L_mu_muhat0_(m, 2, 2, k, j, i) = L_mu_muhat0_(m, 2, 2, k, j, i) / L_mu_2_norm;
                  L_mu_muhat0_(m, 3, 2, k, j, i) = L_mu_muhat0_(m, 3, 2, k, j, i) / L_mu_2_norm;
                });

  // L^mu_3 = p_y - (d_z.L^mu_1) L^mu_1 - (d_z.L^mu_2) L^mu_2 + (d_x.L^mu_0) L^mu_0, d_x = (0,1,0,0), d_y = (0,0,1,0), d_z = (0,0,0,1)
  par_for("radiation_femn_tetrad_compute_L_mu_3", DevExeSpace(), 0, nmb1, 0, 3, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int k, int j, int i) {

            Real g_dd[16];
            adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                 adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                 adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);

            L_mu_muhat0_(m, mu, 3, k, j, i) =
                (mu == 3) - (g_dd[12] * L_mu_muhat0_(m, 0, 1, k, j, i) + g_dd[13] * L_mu_muhat0_(m, 1, 1, k, j, i)
                    + g_dd[14] * L_mu_muhat0_(m, 2, 1, k, j, i) + g_dd[15] * L_mu_muhat0_(m, 3, 1, k, j, i))
                    * L_mu_muhat0_(m, mu, 1, k, j, i)
                    - (g_dd[12] * L_mu_muhat0_(m, 0, 2, k, j, i) + g_dd[13] * L_mu_muhat0_(m, 1, 2, k, j, i)
                        + g_dd[14] * L_mu_muhat0_(m, 2, 2, k, j, i) + g_dd[15] * L_mu_muhat0_(m, 3, 2, k, j, i))
                        * L_mu_muhat0_(m, mu, 2, k, j, i)
                    + (g_dd[12] * L_mu_muhat0_(m, 0, 0, k, j, i) + g_dd[13] * L_mu_muhat0_(m, 1, 0, k, j, i)
                        + g_dd[14] * L_mu_muhat0_(m, 2, 0, k, j, i) + g_dd[15] * L_mu_muhat0_(m, 3, 0, k, j, i))
                        * L_mu_muhat0_(m, mu, 0, k, j, i);
          });

  // L^mu_3 = L^mu_3/||L^mu_3||, ||L^mu_3|| = sqrt(g_mu_mu L^mu_3 L^nu_3)
  par_for_outer("radiation_femn_tetrad_normalize_L_mu_3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, int m, int k, int j, int i) {

                  Real g_dd[16];
                  adm::SpacetimeMetric(adm.alpha(m, k, j, i),
                                       adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                                       adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), g_dd);

                  Real L_mu_3_norm = 0.;
                  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, 0, 16), [&](const int munu, Real &partial_sum) {
                    int mu = int(munu / 4);
                    int nu = munu - 4 * mu;
                    partial_sum += g_dd[munu] * L_mu_muhat0_(m, mu, 3, k, j, i) * L_mu_muhat0_(m, nu, 3, k, j, i);
                  }, L_mu_3_norm);
                  member.team_barrier();
                  L_mu_3_norm = sqrt(L_mu_3_norm);

                  L_mu_muhat0_(m, 0, 3, k, j, i) = L_mu_muhat0_(m, 0, 3, k, j, i) / L_mu_3_norm;
                  L_mu_muhat0_(m, 1, 3, k, j, i) = L_mu_muhat0_(m, 1, 3, k, j, i) / L_mu_3_norm;
                  L_mu_muhat0_(m, 2, 3, k, j, i) = L_mu_muhat0_(m, 2, 3, k, j, i) / L_mu_3_norm;
                  L_mu_muhat0_(m, 3, 3, k, j, i) = L_mu_muhat0_(m, 3, 3, k, j, i) / L_mu_3_norm;
                });

  return TaskStatus::complete;
}
}