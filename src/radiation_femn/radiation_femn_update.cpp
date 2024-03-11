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

#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_matinv.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"

namespace radiationfemn
{
    TaskStatus RadiationFEMN::ExpRKUpdate(Driver* pdriver, int stage)
    {
        const int NGHOST = 2;

        auto& indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        //int npts1 = num_points_total - 1;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto& mbsize = pmy_pack->pmb->mb_size;

        bool& multi_d = pmy_pack->pmesh->multi_d;
        bool& three_d = pmy_pack->pmesh->three_d;

        Real& gam0 = pdriver->gam0[stage - 1];
        Real& gam1 = pdriver->gam1[stage - 1];
        Real beta_dt = (pdriver->beta[stage - 1]) * (pmy_pack->pmesh->dt);

        //int ncells1 = indcs.nx1 + 2 * (indcs.ng);
        //int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
        //int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;

        int& num_points_ = pmy_pack->pradfemn->num_points;
        int& num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
        int& num_species_ = pmy_pack->pradfemn->num_species;
        int num_species_energy = num_species_ * num_energy_bins_;

        auto& f0_ = pmy_pack->pradfemn->f0;
        auto& f1_ = pmy_pack->pradfemn->f1;
        auto& energy_grid_ = pmy_pack->pradfemn->energy_grid;
        auto& flx1 = pmy_pack->pradfemn->iflx.x1f;
        auto& flx2 = pmy_pack->pradfemn->iflx.x2f;
        auto& flx3 = pmy_pack->pradfemn->iflx.x3f;
        auto& L_mu_muhat0_ = pmy_pack->pradfemn->L_mu_muhat0;
        auto& u_mu_ = pmy_pack->pradfemn->u_mu;
        auto& eta_ = pmy_pack->pradfemn->eta;
        auto& e_source_ = pmy_pack->pradfemn->e_source;
        auto& kappa_s_ = pmy_pack->pradfemn->kappa_s;
        auto& kappa_a_ = pmy_pack->pradfemn->kappa_a;
        auto& F_matrix_ = pmy_pack->pradfemn->F_matrix;
        auto& G_matrix_ = pmy_pack->pradfemn->G_matrix;
        auto& energy_par_ = pmy_pack->pradfemn->energy_par;
        auto& P_matrix_ = pmy_pack->pradfemn->P_matrix;
        auto& S_source_ = pmy_pack->pradfemn->S_source;
        adm::ADM::ADM_vars& adm = pmy_pack->padm->adm;

        size_t scr_size = ScrArray2D<Real>::shmem_size(num_points_, num_points_) * 5 + ScrArray1D<Real>::shmem_size(num_points_) * 5
            + ScrArray1D<int>::shmem_size(num_points_ - 1) * 1 + +ScrArray1D<Real>::shmem_size(4 * 4 * 4) * 2;
        int scr_level = 0;
        par_for_outer("radiation_femn_update", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, num_species_energy - 1, ks, ke, js, je, is, ie,
                      KOKKOS_LAMBDA(TeamMember_t member, int m, int nuen, int k, int j, int i)
                      {
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

                          par_for_inner(member, 0, num_points_ - 1, [&](const int idx)
                          {
                              int nuenangidx = IndicesUnited(nu, en, idx, num_species_, num_energy_bins_, num_points_);

                              Real divf_s = flx1(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx1);

                              if (multi_d)
                              {
                                  divf_s += flx2(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx2);
                              }

                              if (three_d)
                              {
                                  divf_s += flx3(m, nuenangidx, k, j, i) / (2. * mbsize.d_view(m).dx3);
                              }

                              g_rhs_scratch(idx) = gam0 * f0_(m, nuenangidx, k, j, i) + gam1 * f1_(m, nuenangidx, k, j, i) - beta_dt * divf_s
                                  + sqrt_det_g_ijk * beta_dt * eta_(m, k, j, i) * e_source_(idx) / Ven;
                          });
                          member.team_barrier();

                          Real idx[] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2, 1 / mbsize.d_view(m).dx3};

                          //----------------------------------------------------------------------------------------
                          /*
                          Real& x1min = mbsize.d_view(m).x1min;
                          Real& x1max = mbsize.d_view(m).x1max;
                          int nx1 = indcs.nx1;
                          Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

                          Real& x2min = mbsize.d_view(m).x2min;
                          Real& x2max = mbsize.d_view(m).x2max;
                          int nx2 = indcs.nx2;
                          Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

                          Real M = 1.;
                          Real r = sqrt(x1 * x1 + x2 * x2);
                          std::cout << "r: " << r << std::endl;

                          // Start of metric check
                          Real gdd_0_0 = -pow(M - 2 * r, 2) / pow(M + 2 * r, 2);
                          Real gdd_0_1 = 0;
                          Real gdd_0_2 = 0;
                          Real gdd_0_3 = 0;
                          Real gdd_1_0 = 0;
                          Real gdd_1_1 = (1.0 / 16.0) * pow(M + 2 * r, 4) / pow(r, 4);
                          Real gdd_1_2 = 0;
                          Real gdd_1_3 = 0;
                          Real gdd_2_0 = 0;
                          Real gdd_2_1 = 0;
                          Real gdd_2_2 = (1.0 / 16.0) * pow(M + 2 * r, 4) / pow(r, 4);
                          Real gdd_2_3 = 0;
                          Real gdd_3_0 = 0;
                          Real gdd_3_1 = 0;
                          Real gdd_3_2 = 0;
                          Real gdd_3_3 = (1.0 / 16.0) * pow(M + 2 * r, 4) / pow(r, 4);

                          std::cout << "gdd_00:" << g_dd[0 + 4 * 0] << " " << gdd_0_0 << std::endl;
                          std::cout << "gdd_01:" << g_dd[0 + 4 * 1] << " " << gdd_0_1 << std::endl;
                          std::cout << "gdd_02:" << g_dd[0 + 4 * 2] << " " << gdd_0_2 << std::endl;
                          std::cout << "gdd_03:" << g_dd[0 + 4 * 3] << " " << gdd_0_3 << std::endl;
                          std::cout << "gdd_10:" << g_dd[1 + 4 * 0] << " " << gdd_1_0 << std::endl;
                          std::cout << "gdd_11:" << g_dd[1 + 4 * 1] << " " << gdd_1_1 << std::endl;
                          std::cout << "gdd_12:" << g_dd[1 + 4 * 2] << " " << gdd_1_2 << std::endl;
                          std::cout << "gdd_13:" << g_dd[1 + 4 * 3] << " " << gdd_1_3 << std::endl;
                          std::cout << "gdd_20:" << g_dd[2 + 4 * 0] << " " << gdd_2_0 << std::endl;
                          std::cout << "gdd_21:" << g_dd[2 + 4 * 1] << " " << gdd_2_1 << std::endl;
                          std::cout << "gdd_22:" << g_dd[2 + 4 * 2] << " " << gdd_2_2 << std::endl;
                          std::cout << "gdd_23:" << g_dd[2 + 4 * 3] << " " << gdd_2_3 << std::endl;
                          std::cout << "gdd_30:" << g_dd[3 + 4 * 0] << " " << gdd_3_0 << std::endl;
                          std::cout << "gdd_31:" << g_dd[3 + 4 * 1] << " " << gdd_3_1 << std::endl;
                          std::cout << "gdd_32:" << g_dd[3 + 4 * 2] << " " << gdd_3_2 << std::endl;
                          std::cout << "gdd_33:" << g_dd[3 + 4 * 3] << " " << gdd_3_3 << std::endl;
                          // End of metric check
                          *
                          /
                          // Start of metric inverse check
                          /*
                          Real guu_0_0 = (-pow(M, 2) - 4 * M * r - 4 * pow(r, 2)) / (pow(M, 2) - 4 * M * r + 4 * pow(r, 2));
                          Real guu_0_1 = 0;
                          Real guu_0_2 = 0;
                          Real guu_0_3 = 0;
                          Real guu_1_0 = 0;
                          Real guu_1_1 = 16 * pow(r, 4) / (pow(M, 4) + 8 * pow(M, 3) * r + 24 * pow(M, 2) * pow(r, 2) + 32 * M * pow(r, 3) + 16 * pow(r, 4));
                          Real guu_1_2 = 0;
                          Real guu_1_3 = 0;
                          Real guu_2_0 = 0;
                          Real guu_2_1 = 0;
                          Real guu_2_2 = 16 * pow(r, 4) / (pow(M, 4) + 8 * pow(M, 3) * r + 24 * pow(M, 2) * pow(r, 2) + 32 * M * pow(r, 3) + 16 * pow(r, 4));
                          Real guu_2_3 = 0;
                          Real guu_3_0 = 0;
                          Real guu_3_1 = 0;
                          Real guu_3_2 = 0;
                          Real guu_3_3 = 16 * pow(r, 4) / (pow(M, 4) + 8 * pow(M, 3) * r + 24 * pow(M, 2) * pow(r, 2) + 32 * M * pow(r, 3) + 16 * pow(r, 4));

                          std::cout << "guu_00:" << g_uu[0 + 4 * 0] << " " << guu_0_0 << std::endl;
                          std::cout << "guu_01:" << g_uu[0 + 4 * 1] << " " << guu_0_1 << std::endl;
                          std::cout << "guu_02:" << g_uu[0 + 4 * 2] << " " << guu_0_2 << std::endl;
                          std::cout << "guu_03:" << g_uu[0 + 4 * 3] << " " << guu_0_3 << std::endl;
                          std::cout << "guu_10:" << g_uu[1 + 4 * 0] << " " << guu_1_0 << std::endl;
                          std::cout << "guu_11:" << g_uu[1 + 4 * 1] << " " << guu_1_1 << std::endl;
                          std::cout << "guu_12:" << g_uu[1 + 4 * 2] << " " << guu_1_2 << std::endl;
                          std::cout << "guu_13:" << g_uu[1 + 4 * 3] << " " << guu_1_3 << std::endl;
                          std::cout << "guu_20:" << g_uu[2 + 4 * 0] << " " << guu_2_0 << std::endl;
                          std::cout << "guu_21:" << g_uu[2 + 4 * 1] << " " << guu_2_1 << std::endl;
                          std::cout << "guu_22:" << g_uu[2 + 4 * 2] << " " << guu_2_2 << std::endl;
                          std::cout << "guu_23:" << g_uu[2 + 4 * 3] << " " << guu_2_3 << std::endl;
                          std::cout << "guu_30:" << g_uu[3 + 4 * 0] << " " << guu_3_0 << std::endl;
                          std::cout << "guu_31:" << g_uu[3 + 4 * 1] << " " << guu_3_1 << std::endl;
                          std::cout << "guu_32:" << g_uu[3 + 4 * 2] << " " << guu_3_2 << std::endl;
                          std::cout << "guu_33:" << g_uu[3 + 4 * 3] << " " << guu_3_3 << std::endl;
                          */
                          // End of metric inverse check
                          // ---------------------------------------------------------------

                          // lapse derivatives (\p_mu alpha)
                          Real dtalpha_d = 0.; // time derivatives, get from z4c
                          AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d; // spatial derivatives
                          dalpha_d(0) = Dx<NGHOST>(0, idx, adm.alpha, m, k, j, i);
                          dalpha_d(1) = (multi_d) ? Dx<NGHOST>(1, idx, adm.alpha, m, k, j, i) : 0.;
                          dalpha_d(2) = (three_d) ? Dx<NGHOST>(2, idx, adm.alpha, m, k, j, i) : 0.;

                          //-----------------------------------------------
                          /*
                          Real dalpha_x = 4 * M * x1 / (pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 2) * sqrt(pow(x1, 2) + pow(x2, 2)));
                          Real dalpha_y = 4 * M * x2 / (pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 2) * sqrt(pow(x1, 2) + pow(x2, 2)));
                          Real dalpha_z = 0;
                          std::cout << "Lapse derivatives: spatial" << std::endl;
                          std::cout << "dalpha_x:" << dalpha_x << " " << dalpha_d(0) << std::endl;
                          std::cout << "dalpha_y:" << dalpha_y << " " << dalpha_d(1) << std::endl;
                          std::cout << "dalpha_z:" << dalpha_z << " " << dalpha_d(2) << std::endl;
                          */
                          // ----------------------------------------------

                          // shift derivatives (\p_mu beta^i)
                          Real dtbetax_du = 0.; // time derivatives, get from z4c
                          Real dtbetay_du = 0.;
                          Real dtbetaz_du = 0.;
                          AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du; // spatial derivatives
                          for (int a = 0; a < 3; ++a)
                          {
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

                          // ---------------------------------------
                          /*
                          std::cout << "Shift and derivatives: " << std::endl;
                          std::cout << "dx_beta_1: " << dbeta_du(0, 0) << std::endl;
                          std::cout << "dx_beta_2: " << dbeta_du(0, 1) << std::endl;
                          std::cout << "dx_beta_3: " << dbeta_du(0, 2) << std::endl;
                          std::cout << "dy_beta_1: " << dbeta_du(1, 0) << std::endl;
                          std::cout << "dy_beta_2: " << dbeta_du(1, 1) << std::endl;
                          std::cout << "dy_beta_3: " << dbeta_du(1, 2) << std::endl;
                          std::cout << "dz_beta_1: " << dbeta_du(2, 0) << std::endl;
                          std::cout << "dz_beta_2: " << dbeta_du(2, 1) << std::endl;
                          std::cout << "dz_beta_3: " << dbeta_du(2, 2) << std::endl;
                          std::cout << "betax_d: " << betax_d << std::endl;
                          std::cout << "betay_d: " << betay_d << std::endl;
                          std::cout << "betaz_d: " << betaz_d << std::endl;
                          */
                          // ---------------------------------------

                          // derivatives of spatial metric (\p_mu g_ij)
                          AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> dtg_dd;
                          AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
                          for (int a = 0; a < 3; ++a)
                          {
                              for (int b = a; b < 3; ++b)
                              {
                                  dtg_dd(a, b) = 0.; // time derivatives, get from z4c

                                  dg_ddd(0, a, b) = Dx<NGHOST>(0, idx, adm.g_dd, m, a, b, k, j, i); // spatial derivatives
                                  dg_ddd(1, a, b) = (multi_d) ? Dx<NGHOST>(1, idx, adm.g_dd, m, a, b, k, j, i) : 0.;
                                  dg_ddd(2, a, b) = (three_d) ? Dx<NGHOST>(2, idx, adm.g_dd, m, a, b, k, j, i) : 0.;
                              }
                          }

                          // --------------------
                          // print spatial derivatives of spatial metric
                          /*
                          Real dx_gdd_11 = -1.0 / 4.0 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dy_gdd_11 = -1.0 / 4.0 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dz_gdd_11 = 0;
                          Real dx_gdd_12 = 0;
                          Real dy_gdd_12 = 0;
                          Real dz_gdd_12 = 0;
                          Real dx_gdd_13 = 0;
                          Real dy_gdd_13 = 0;
                          Real dz_gdd_13 = 0;
                          Real dx_gdd_21 = 0;
                          Real dy_gdd_21 = 0;
                          Real dz_gdd_21 = 0;
                          Real dx_gdd_22 = -1.0 / 4.0 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dy_gdd_22 = -1.0 / 4.0 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dz_gdd_22 = 0;
                          Real dx_gdd_23 = 0;
                          Real dy_gdd_23 = 0;
                          Real dz_gdd_23 = 0;
                          Real dx_gdd_31 = 0;
                          Real dy_gdd_31 = 0;
                          Real dz_gdd_31 = 0;
                          Real dx_gdd_32 = 0;
                          Real dy_gdd_32 = 0;
                          Real dz_gdd_32 = 0;
                          Real dx_gdd_33 = -1.0 / 4.0 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dy_gdd_33 = -1.0 / 4.0 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dz_gdd_33 = 0;

                          std::cout << "dx_gdd_11: " << dx_gdd_11 << " " << dg_ddd(0, 0, 0) << std::endl;
                          std::cout << "dy_gdd_11: " << dy_gdd_11 << " " << dg_ddd(1, 0, 0) << std::endl;
                          std::cout << "dz_gdd_11: " << dz_gdd_11 << " " << dg_ddd(2, 0, 0) << std::endl;
                          std::cout << "dx_gdd_12: " << dx_gdd_12 << " " << dg_ddd(0, 0, 1) << std::endl;
                          std::cout << "dy_gdd_12: " << dy_gdd_12 << " " << dg_ddd(1, 0, 1) << std::endl;
                          std::cout << "dz_gdd_12: " << dz_gdd_12 << " " << dg_ddd(2, 0, 1) << std::endl;
                          std::cout << "dx_gdd_13: " << dx_gdd_13 << " " << dg_ddd(0, 0, 2) << std::endl;
                          std::cout << "dy_gdd_13: " << dy_gdd_13 << " " << dg_ddd(1, 0, 2) << std::endl;
                          std::cout << "dz_gdd_13: " << dz_gdd_13 << " " << dg_ddd(2, 0, 2) << std::endl;
                          std::cout << "dx_gdd_21: " << dx_gdd_21 << " " << dg_ddd(0, 1, 0) << std::endl;
                          std::cout << "dy_gdd_21: " << dy_gdd_21 << " " << dg_ddd(1, 1, 0) << std::endl;
                          std::cout << "dz_gdd_21: " << dz_gdd_21 << " " << dg_ddd(2, 1, 0) << std::endl;
                          std::cout << "dx_gdd_22: " << dx_gdd_22 << " " << dg_ddd(0, 1, 1) << std::endl;
                          std::cout << "dy_gdd_22: " << dy_gdd_22 << " " << dg_ddd(1, 1, 1) << std::endl;
                          std::cout << "dz_gdd_22: " << dz_gdd_22 << " " << dg_ddd(2, 1, 1) << std::endl;
                          std::cout << "dx_gdd_23: " << dx_gdd_23 << " " << dg_ddd(0, 1, 2) << std::endl;
                          std::cout << "dy_gdd_23: " << dy_gdd_23 << " " << dg_ddd(1, 1, 2) << std::endl;
                          std::cout << "dz_gdd_23: " << dz_gdd_23 << " " << dg_ddd(2, 1, 2) << std::endl;
                          std::cout << "dx_gdd_31: " << dx_gdd_31 << " " << dg_ddd(0, 2, 0) << std::endl;
                          std::cout << "dy_gdd_31: " << dy_gdd_31 << " " << dg_ddd(1, 2, 0) << std::endl;
                          std::cout << "dz_gdd_31: " << dz_gdd_31 << " " << dg_ddd(2, 2, 0) << std::endl;
                          std::cout << "dx_gdd_32: " << dx_gdd_32 << " " << dg_ddd(0, 2, 1) << std::endl;
                          std::cout << "dy_gdd_32: " << dy_gdd_32 << " " << dg_ddd(1, 2, 1) << std::endl;
                          std::cout << "dz_gdd_32: " << dz_gdd_32 << " " << dg_ddd(2, 2, 1) << std::endl;
                          std::cout << "dx_gdd_33: " << dx_gdd_33 << " " << dg_ddd(0, 2, 2) << std::endl;
                          std::cout << "dy_gdd_33: " << dy_gdd_33 << " " << dg_ddd(1, 2, 2) << std::endl;
                          std::cout << "dz_gdd_33: " << dz_gdd_33 << " " << dg_ddd(2, 2, 2) << std::endl;
                          std::cout << std::endl;
                          */
                          // end of spatial derivatives of spatial metric
                          // --------------------

                          // derivatives of the 4-metric: time derivatives
                          AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> dg4_ddd; //f
                          dg4_ddd(0, 0, 0) = -2. * adm.alpha(m, k, j, i) * dtalpha_d + 2. * betax_d * dtbetax_du + 2. * betay_d * dtbetay_du + 2. * betaz_d * dtbetaz_du
                              + dtg_dd(0, 0) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 0, k, j, i) + 2. * dtg_dd(0, 1) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 1, k, j, i)
                              + 2. * dtg_dd(0, 2) * adm.beta_u(m, 0, k, j, i) * adm.beta_u(m, 2, k, j, i) + dtg_dd(1, 1) * adm.beta_u(m, 1, k, j, i) * adm.beta_u(m, 1, k, j, i)
                              + 2. * dtg_dd(1, 2) * adm.beta_u(m, 1, k, j, i) * adm.beta_u(m, 2, k, j, i) + dtg_dd(2, 2) * adm.beta_u(m, 2, k, j, i) * adm.beta_u(m, 2, k, j, i);
                          for (int a = 1; a < 4; ++a)
                          {
                              dg4_ddd(0, a, 0) = adm.g_dd(m, 0, 0, k, j, i) * dtbetax_du + adm.g_dd(m, 0, 1, k, j, i) * dtbetay_du + adm.g_dd(m, 0, 2, k, j, i) * dtbetaz_du
                                  + dtg_dd(a - 1, 0) * adm.beta_u(m, 0, k, j, i) + dtg_dd(a - 1, 1) * adm.beta_u(m, 1, k, j, i) + dtg_dd(a - 1, 2) * adm.beta_u(m, 2, k, j, i);
                          }
                          for (int a = 1; a < 4; ++a)
                          {
                              for (int b = 1; b < 4; ++b)
                              {
                                  dg4_ddd(0, a, b) = 0.; // time derivatives, get from z4c
                              }
                          }

                          // -------------------------------
                          // print time derivatives of 4 metric
                          /*
                          for (int a = 0; a < 4; a++)
                          {
                              for (int b = 0; b < 4; b++)
                              {
                                  std::cout << "dt_gdd_" << a << b << ": " << dg4_ddd(0, a, b) << std::endl;
                              }
                          }
                          */
                          // end time derivatives of 4 metric
                          // --------------------------------

                          // derivatives of the 4-metric: spatial derivatives
                          for (int a = 1; a < 4; ++a)
                          {
                              for (int b = 1; b < 4; ++b)
                              {
                                  dg4_ddd(1, a, b) = dg_ddd(0, a - 1, b - 1);
                                  dg4_ddd(2, a, b) = dg_ddd(1, a - 1, b - 1);
                                  dg4_ddd(3, a, b) = dg_ddd(2, a - 1, b - 1);

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

                          // ------------------
                          // print spatial derivatives of 4-metric
                          /*
                          Real dx_gdd_00 = 8 * M * x1 * (M - 2 * sqrt(pow(x1, 2) + pow(x2, 2))) / (pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * sqrt(pow(x1, 2) + pow(x2, 2)));
                          Real dy_gdd_00 = 8 * M * x2 * (M - 2 * sqrt(pow(x1, 2) + pow(x2, 2))) / (pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * sqrt(pow(x1, 2) + pow(x2, 2)));
                          Real dz_gdd_00 = 0;
                          Real dx_gdd_01 = 0;
                          Real dy_gdd_01 = 0;
                          Real dz_gdd_01 = 0;
                          Real dx_gdd_02 = 0;
                          Real dy_gdd_02 = 0;
                          Real dz_gdd_02 = 0;
                          Real dx_gdd_03 = 0;
                          Real dy_gdd_03 = 0;
                          Real dz_gdd_03 = 0;
                          Real dx_gdd_10 = 0;
                          Real dy_gdd_10 = 0;
                          Real dz_gdd_10 = 0;
                          Real dx_gdd_11 = -1.0 / 4.0 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dy_gdd_11 = -1.0 / 4.0 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dz_gdd_11 = 0;
                          Real dx_gdd_12 = 0;
                          Real dy_gdd_12 = 0;
                          Real dz_gdd_12 = 0;
                          Real dx_gdd_13 = 0;
                          Real dy_gdd_13 = 0;
                          Real dz_gdd_13 = 0;
                          Real dx_gdd_20 = 0;
                          Real dy_gdd_20 = 0;
                          Real dz_gdd_20 = 0;
                          Real dx_gdd_21 = 0;
                          Real dy_gdd_21 = 0;
                          Real dz_gdd_21 = 0;
                          Real dx_gdd_22 = -1.0 / 4.0 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dy_gdd_22 = -1.0 / 4.0 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dz_gdd_22 = 0;
                          Real dx_gdd_23 = 0;
                          Real dy_gdd_23 = 0;
                          Real dz_gdd_23 = 0;
                          Real dx_gdd_30 = 0;
                          Real dy_gdd_30 = 0;
                          Real dz_gdd_30 = 0;
                          Real dx_gdd_31 = 0;
                          Real dy_gdd_31 = 0;
                          Real dz_gdd_31 = 0;
                          Real dx_gdd_32 = 0;
                          Real dy_gdd_32 = 0;
                          Real dz_gdd_32 = 0;
                          Real dx_gdd_33 = -1.0 / 4.0 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dy_gdd_33 = -1.0 / 4.0 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) / pow(pow(x1, 2) + pow(x2, 2), 3);
                          Real dz_gdd_33 = 0;

                          std::cout << "dt_gdd_00: " << 0 << " " << dg4_ddd(0, 0, 0) << std::endl;
                          std::cout << "dx_gdd_00: " << dx_gdd_00 << " " << dg4_ddd(1, 0, 0) << std::endl;
                          std::cout << "dy_gdd_00: " << dy_gdd_00 << " " << dg4_ddd(2, 0, 0) << std::endl;
                          std::cout << "dz_gdd_00: " << dz_gdd_00 << " " << dg4_ddd(3, 0, 0) << std::endl;
                          std::cout << "dt_gdd_01: " << 0 << " " << dg4_ddd(0, 0, 1) << std::endl;
                          std::cout << "dx_gdd_01: " << dx_gdd_01 << " " << dg4_ddd(1, 0, 1) << std::endl;
                          std::cout << "dy_gdd_01: " << dy_gdd_01 << " " << dg4_ddd(2, 0, 1) << std::endl;
                          std::cout << "dz_gdd_01: " << dz_gdd_01 << " " << dg4_ddd(3, 0, 1) << std::endl;
                          std::cout << "dt_gdd_02: " << 0 << " " << dg4_ddd(0, 0, 2) << std::endl;
                          std::cout << "dx_gdd_02: " << dx_gdd_02 << " " << dg4_ddd(1, 0, 2) << std::endl;
                          std::cout << "dy_gdd_02: " << dy_gdd_02 << " " << dg4_ddd(2, 0, 2) << std::endl;
                          std::cout << "dz_gdd_02: " << dz_gdd_02 << " " << dg4_ddd(3, 0, 2) << std::endl;
                          std::cout << "dt_gdd_03: " << 0 << " " << dg4_ddd(0, 0, 3) << std::endl;
                          std::cout << "dx_gdd_03: " << dx_gdd_03 << " " << dg4_ddd(1, 0, 3) << std::endl;
                          std::cout << "dy_gdd_03: " << dy_gdd_03 << " " << dg4_ddd(2, 0, 3) << std::endl;
                          std::cout << "dz_gdd_03: " << dz_gdd_03 << " " << dg4_ddd(3, 0, 3) << std::endl;
                          std::cout << "dt_gdd_10: " << 0 << " " << dg4_ddd(0, 1, 0) << std::endl;
                          std::cout << "dx_gdd_10: " << dx_gdd_10 << " " << dg4_ddd(1, 1, 0) << std::endl;
                          std::cout << "dy_gdd_10: " << dy_gdd_10 << " " << dg4_ddd(2, 1, 0) << std::endl;
                          std::cout << "dz_gdd_10: " << dz_gdd_10 << " " << dg4_ddd(3, 1, 0) << std::endl;
                          std::cout << "dt_gdd_11: " << 0 << " " << dg4_ddd(0, 1, 1) << std::endl;
                          std::cout << "dx_gdd_11: " << dx_gdd_11 << " " << dg4_ddd(1, 1, 1) << std::endl;
                          std::cout << "dy_gdd_11: " << dy_gdd_11 << " " << dg4_ddd(2, 1, 1) << std::endl;
                          std::cout << "dz_gdd_11: " << dz_gdd_11 << " " << dg4_ddd(3, 1, 1) << std::endl;
                          std::cout << "dt_gdd_12: " << 0 << " " << dg4_ddd(0, 1, 2) << std::endl;
                          std::cout << "dx_gdd_12: " << dx_gdd_12 << " " << dg4_ddd(1, 1, 2) << std::endl;
                          std::cout << "dy_gdd_12: " << dy_gdd_12 << " " << dg4_ddd(2, 1, 2) << std::endl;
                          std::cout << "dz_gdd_12: " << dz_gdd_12 << " " << dg4_ddd(3, 1, 2) << std::endl;
                          std::cout << "dt_gdd_13: " << 0 << " " << dg4_ddd(0, 1, 3) << std::endl;
                          std::cout << "dx_gdd_13: " << dx_gdd_13 << " " << dg4_ddd(1, 1, 3) << std::endl;
                          std::cout << "dy_gdd_13: " << dy_gdd_13 << " " << dg4_ddd(2, 1, 3) << std::endl;
                          std::cout << "dz_gdd_13: " << dz_gdd_13 << " " << dg4_ddd(3, 1, 3) << std::endl;
                          std::cout << "dt_gdd_20: " << 0 << " " << dg4_ddd(0, 2, 0) << std::endl;
                          std::cout << "dx_gdd_20: " << dx_gdd_20 << " " << dg4_ddd(1, 2, 0) << std::endl;
                          std::cout << "dy_gdd_20: " << dy_gdd_20 << " " << dg4_ddd(2, 2, 0) << std::endl;
                          std::cout << "dz_gdd_20: " << dz_gdd_20 << " " << dg4_ddd(3, 2, 0) << std::endl;
                          std::cout << "dt_gdd_21: " << 0 << " " << dg4_ddd(0, 2, 1) << std::endl;
                          std::cout << "dx_gdd_21: " << dx_gdd_21 << " " << dg4_ddd(1, 2, 1) << std::endl;
                          std::cout << "dy_gdd_21: " << dy_gdd_21 << " " << dg4_ddd(2, 2, 1) << std::endl;
                          std::cout << "dz_gdd_21: " << dz_gdd_21 << " " << dg4_ddd(3, 2, 1) << std::endl;
                          std::cout << "dt_gdd_22: " << 0 << " " << dg4_ddd(0, 2, 2) << std::endl;
                          std::cout << "dx_gdd_22: " << dx_gdd_22 << " " << dg4_ddd(1, 2, 2) << std::endl;
                          std::cout << "dy_gdd_22: " << dy_gdd_22 << " " << dg4_ddd(2, 2, 2) << std::endl;
                          std::cout << "dz_gdd_22: " << dz_gdd_22 << " " << dg4_ddd(3, 2, 2) << std::endl;
                          std::cout << "dt_gdd_23: " << 0 << " " << dg4_ddd(0, 2, 3) << std::endl;
                          std::cout << "dx_gdd_23: " << dx_gdd_23 << " " << dg4_ddd(1, 2, 3) << std::endl;
                          std::cout << "dy_gdd_23: " << dy_gdd_23 << " " << dg4_ddd(2, 2, 3) << std::endl;
                          std::cout << "dz_gdd_23: " << dz_gdd_23 << " " << dg4_ddd(3, 2, 3) << std::endl;
                          std::cout << "dt_gdd_30: " << 0 << " " << dg4_ddd(0, 3, 0) << std::endl;
                          std::cout << "dx_gdd_30: " << dx_gdd_30 << " " << dg4_ddd(1, 3, 0) << std::endl;
                          std::cout << "dy_gdd_30: " << dy_gdd_30 << " " << dg4_ddd(2, 3, 0) << std::endl;
                          std::cout << "dz_gdd_30: " << dz_gdd_30 << " " << dg4_ddd(3, 3, 0) << std::endl;
                          std::cout << "dt_gdd_31: " << 0 << " " << dg4_ddd(0, 3, 1) << std::endl;
                          std::cout << "dx_gdd_31: " << dx_gdd_31 << " " << dg4_ddd(1, 3, 1) << std::endl;
                          std::cout << "dy_gdd_31: " << dy_gdd_31 << " " << dg4_ddd(2, 3, 1) << std::endl;
                          std::cout << "dz_gdd_31: " << dz_gdd_31 << " " << dg4_ddd(3, 3, 1) << std::endl;
                          std::cout << "dt_gdd_32: " << 0 << " " << dg4_ddd(0, 3, 2) << std::endl;
                          std::cout << "dx_gdd_32: " << dx_gdd_32 << " " << dg4_ddd(1, 3, 2) << std::endl;
                          std::cout << "dy_gdd_32: " << dy_gdd_32 << " " << dg4_ddd(2, 3, 2) << std::endl;
                          std::cout << "dz_gdd_32: " << dz_gdd_32 << " " << dg4_ddd(3, 3, 2) << std::endl;
                          std::cout << "dt_gdd_33: " << 0 << " " << dg4_ddd(0, 3, 3) << std::endl;
                          std::cout << "dx_gdd_33: " << dx_gdd_33 << " " << dg4_ddd(1, 3, 3) << std::endl;
                          std::cout << "dy_gdd_33: " << dy_gdd_33 << " " << dg4_ddd(2, 3, 3) << std::endl;
                          std::cout << "dz_gdd_33: " << dz_gdd_33 << " " << dg4_ddd(3, 3, 3) << std::endl;
                          */
                          // end of spatial derivatives of spatial metric
                          // ------------------

                          // Christoeffel symbols
                          AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> Gamma_udd;
                          for (int a = 0; a < 4; ++a)
                          {
                              for (int b = 0; b < 4; ++b)
                              {
                                  for (int c = 0; c < 4; ++c)
                                  {
                                      Gamma_udd(a, b, c) = 0.0;
                                      for (int d = 0; d < 4; ++d)
                                      {
                                          Gamma_udd(a, b, c) += 0.5 * g_uu[a + 4 * d] * (dg4_ddd(b, d, c) + dg4_ddd(c, b, d) - dg4_ddd(d, b, c));
                                      }
                                  }
                              }
                          }

                          // Ricci rotation coefficients
                          AthenaScratchTensor4d<Real, TensorSymm::SYM2, 4, 3> Gamma_fluid_udd;
                          for (int a = 0; a < 4; ++a)
                          {
                              for (int b = 0; b < 4; ++b)
                              {
                                  for (int c = 0; c < 4; ++c)
                                  {
                                      Gamma_fluid_udd(a, b, c) = 0.0;
                                      for (int d = 0; d < 64; ++d)
                                      {
                                          int a_idx = int(d / (4 * 4));
                                          int b_idx = int((d - 4 * 4 * a_idx) / 4);
                                          int c_idx = d - a_idx * 4 * 4 - b_idx * 4;

                                          Real l_sign = (a == 0) ? -1. : +1.;
                                          Real L_ahat_aidx = l_sign * (g_dd[a_idx + 4 * 0] * L_mu_muhat0_(m, 0, a, k, j, i) + g_dd[a_idx + 4 * 1] * L_mu_muhat0_(m, 1, a, k, j, i)
                                              + g_dd[a_idx + 4 * 2] * L_mu_muhat0_(m, 2, a, k, j, i) + g_dd[a_idx + 4 * 3] * L_mu_muhat0_(m, 3, a, k, j, i));
                                          Gamma_fluid_udd(a, b, c) +=
                                              L_mu_muhat0_(m, b_idx, b, k, j, i) * L_mu_muhat0_(m, c_idx, c, k, j, i) * L_ahat_aidx * Gamma_udd(a_idx, b_idx, c_idx);
                                      }

                                      for (int a_idx = 0; a_idx < 4; ++a_idx)
                                      {
                                          Real l_sign = (a == 0) ? -1. : +1.;
                                          Real L_ahat_aidx = l_sign * (g_dd[a_idx + 4 * 0] * L_mu_muhat0_(m, 0, a, k, j, i) + g_dd[a_idx + 4 * 1] * L_mu_muhat0_(m, 1, a, k, j, i)
                                              + g_dd[a_idx + 4 * 2] * L_mu_muhat0_(m, 2, a, k, j, i) + g_dd[a_idx + 4 * 3] * L_mu_muhat0_(m, 3, a, k, j, i));

                                          Gamma_fluid_udd(a, b, c) +=
                                              L_ahat_aidx * (L_mu_muhat0_(m, 1, c, k, j, i) - (u_mu_(m, 1, k, j, i) / u_mu_(m, 0, k, j, i)) * L_mu_muhat0_(m, 0, c, k, j, i))
                                              * Dx<NGHOST>(0, idx, L_mu_muhat0_, m, a_idx, c, k, j, i);

                                          if (multi_d)
                                          {
                                              Gamma_fluid_udd(a, b, c) +=
                                                  L_ahat_aidx * (L_mu_muhat0_(m, 2, c, k, j, i) - (u_mu_(m, 2, k, j, i) / u_mu_(m, 0, k, j, i)) * L_mu_muhat0_(m, 0, c, k, j, i))
                                                  * Dx<NGHOST>(1, idx, L_mu_muhat0_, m, a_idx, c, k, j, i);
                                          }

                                          if (three_d)
                                          {
                                              Gamma_fluid_udd(a, b, c) +=
                                                  L_ahat_aidx * (L_mu_muhat0_(m, 3, c, k, j, i) - (u_mu_(m, 3, k, j, i) / u_mu_(m, 0, k, j, i)) * L_mu_muhat0_(m, 0, c, k, j, i))
                                                  * Dx<NGHOST>(2, idx, L_mu_muhat0_, m, a_idx, c, k, j, i);
                                          }
                                      }
                                  }
                              }
                          }

                          // -------------------------------
                          // start of 4-Christoeffel print
                          /*
                          Real gamma4d_0_0_0 = 0;
                          Real gamma4d_0_0_1 = 4 * M * x1 * (-pow(M, 3) + 6 * pow(M, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 12 * M * pow(x1, 2) - 12 * M * pow(x2, 2) + 8 * pow(x1, 2)
                              * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2))) / (pow(M, 5) * sqrt(pow(x1, 2) + pow(x2, 2)) - 6 * pow(M, 4) *
                              pow(x1, 2) - 6 * pow(M, 4) * pow(x2, 2) + 8 * pow(M, 3) * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(M, 3) * pow(x2, 2) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(M, 2) * pow(x1, 4) + 32 * pow(M, 2) * pow(x1, 2) * pow(x2, 2) + 16 * pow(M, 2) * pow(x2, 4) - 48 * M *
                              pow(x1, 4) * sqrt(pow(x1, 2) + pow(x2, 2)) - 96 * M * pow(x1, 2) * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 48 * M * pow(x2, 4) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * pow(x1, 6) + 96 * pow(x1, 4) * pow(x2, 2) + 96 * pow(x1, 2) * pow(x2, 4) + 32 * pow(x2, 6));
                          Real gamma4d_0_0_2 = 4 * M * x2 * (-pow(M, 3) + 6 * pow(M, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 12 * M * pow(x1, 2) - 12 * M * pow(x2, 2) + 8 * pow(x1, 2)
                              * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2))) / (pow(M, 5) * sqrt(pow(x1, 2) + pow(x2, 2)) - 6 * pow(M, 4) *
                              pow(x1, 2) - 6 * pow(M, 4) * pow(x2, 2) + 8 * pow(M, 3) * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(M, 3) * pow(x2, 2) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(M, 2) * pow(x1, 4) + 32 * pow(M, 2) * pow(x1, 2) * pow(x2, 2) + 16 * pow(M, 2) * pow(x2, 4) - 48 * M *
                              pow(x1, 4) * sqrt(pow(x1, 2) + pow(x2, 2)) - 96 * M * pow(x1, 2) * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 48 * M * pow(x2, 4) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * pow(x1, 6) + 96 * pow(x1, 4) * pow(x2, 2) + 96 * pow(x1, 2) * pow(x2, 4) + 32 * pow(x2, 6));
                          Real gamma4d_0_0_3 = 0;
                          Real gamma4d_0_1_0 = 4 * M * x1 * (-pow(M, 3) + 6 * pow(M, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 12 * M * pow(x1, 2) - 12 * M * pow(x2, 2) + 8 * pow(x1, 2)
                              * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2))) / (pow(M, 5) * sqrt(pow(x1, 2) + pow(x2, 2)) - 6 * pow(M, 4) *
                              pow(x1, 2) - 6 * pow(M, 4) * pow(x2, 2) + 8 * pow(M, 3) * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(M, 3) * pow(x2, 2) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(M, 2) * pow(x1, 4) + 32 * pow(M, 2) * pow(x1, 2) * pow(x2, 2) + 16 * pow(M, 2) * pow(x2, 4) - 48 * M *
                              pow(x1, 4) * sqrt(pow(x1, 2) + pow(x2, 2)) - 96 * M * pow(x1, 2) * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 48 * M * pow(x2, 4) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * pow(x1, 6) + 96 * pow(x1, 4) * pow(x2, 2) + 96 * pow(x1, 2) * pow(x2, 4) + 32 * pow(x2, 6));
                          Real gamma4d_0_1_1 = 0;
                          Real gamma4d_0_1_2 = 0;
                          Real gamma4d_0_1_3 = 0;
                          Real gamma4d_0_2_0 = 4 * M * x2 * (-pow(M, 3) + 6 * pow(M, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 12 * M * pow(x1, 2) - 12 * M * pow(x2, 2) + 8 * pow(x1, 2)
                              * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2))) / (pow(M, 5) * sqrt(pow(x1, 2) + pow(x2, 2)) - 6 * pow(M, 4) *
                              pow(x1, 2) - 6 * pow(M, 4) * pow(x2, 2) + 8 * pow(M, 3) * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 8 * pow(M, 3) * pow(x2, 2) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(M, 2) * pow(x1, 4) + 32 * pow(M, 2) * pow(x1, 2) * pow(x2, 2) + 16 * pow(M, 2) * pow(x2, 4) - 48 * M *
                              pow(x1, 4) * sqrt(pow(x1, 2) + pow(x2, 2)) - 96 * M * pow(x1, 2) * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) - 48 * M * pow(x2, 4) *
                              sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * pow(x1, 6) + 96 * pow(x1, 4) * pow(x2, 2) + 96 * pow(x1, 2) * pow(x2, 4) + 32 * pow(x2, 6));
                          Real gamma4d_0_2_1 = 0;
                          Real gamma4d_0_2_2 = 0;
                          Real gamma4d_0_2_3 = 0;
                          Real gamma4d_0_3_0 = 0;
                          Real gamma4d_0_3_1 = 0;
                          Real gamma4d_0_3_2 = 0;
                          Real gamma4d_0_3_3 = 0;
                          Real gamma4d_1_0_0 = -64 * M * x1 * (M - 2 * sqrt(pow(x1, 2) + pow(x2, 2))) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * sqrt(pow(x1, 2) + pow(x2, 2)) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 *
                                  pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) * pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) *
                                  sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 * pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_1_0_1 = 0;
                          Real gamma4d_1_0_2 = 0;
                          Real gamma4d_1_0_3 = 0;
                          Real gamma4d_1_1_0 = 0;
                          Real gamma4d_1_1_1 = -2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_1_1_2 = -2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_1_1_3 = 0;
                          Real gamma4d_1_2_0 = 0;
                          Real gamma4d_1_2_1 = -2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_1_2_2 = 2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_1_2_3 = 0;
                          Real gamma4d_1_3_0 = 0;
                          Real gamma4d_1_3_1 = 0;
                          Real gamma4d_1_3_2 = 0;
                          Real gamma4d_1_3_3 = 2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_2_0_0 = -64 * M * x2 * (M - 2 * sqrt(pow(x1, 2) + pow(x2, 2))) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * sqrt(pow(x1, 2) + pow(x2, 2)) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 *
                                  pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) * pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) *
                                  sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 * pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_2_0_1 = 0;
                          Real gamma4d_2_0_2 = 0;
                          Real gamma4d_2_0_3 = 0;
                          Real gamma4d_2_1_0 = 0;
                          Real gamma4d_2_1_1 = 2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_2_1_2 = -2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_2_1_3 = 0;
                          Real gamma4d_2_2_0 = 0;
                          Real gamma4d_2_2_1 = -2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_2_2_2 = -2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_2_2_3 = 0;
                          Real gamma4d_2_3_0 = 0;
                          Real gamma4d_2_3_1 = 0;
                          Real gamma4d_2_3_2 = 0;
                          Real gamma4d_2_3_3 = 2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_3_0_0 = 0;
                          Real gamma4d_3_0_1 = 0;
                          Real gamma4d_3_0_2 = 0;
                          Real gamma4d_3_0_3 = 0;
                          Real gamma4d_3_1_0 = 0;
                          Real gamma4d_3_1_1 = 0;
                          Real gamma4d_3_1_2 = 0;
                          Real gamma4d_3_1_3 = -2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_3_2_0 = 0;
                          Real gamma4d_3_2_1 = 0;
                          Real gamma4d_3_2_2 = 0;
                          Real gamma4d_3_2_3 = -2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_3_3_0 = 0;
                          Real gamma4d_3_3_1 = -2 * M * x1 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_3_3_2 = -2 * M * x2 * pow(M + 2 * sqrt(pow(x1, 2) + pow(x2, 2)), 3) * (pow(x1, 4) + 2 * pow(x1, 2) * pow(x2, 2) + pow(x2, 4)) / (
                              pow(pow(x1, 2) + pow(x2, 2), 3) * (pow(M, 4) + 8 * pow(M, 3) * sqrt(pow(x1, 2) + pow(x2, 2)) + 24 * pow(M, 2) * pow(x1, 2) + 24 * pow(M, 2) *
                                  pow(x2, 2) + 32 * M * pow(x1, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 32 * M * pow(x2, 2) * sqrt(pow(x1, 2) + pow(x2, 2)) + 16 * pow(x1, 4) + 32 *
                                  pow(x1, 2) * pow(x2, 2) + 16 * pow(x2, 4)));
                          Real gamma4d_3_3_3 = 0;

                          std::cout << "000:" << Gamma_udd(0, 0, 0) << " " << gamma4d_0_0_0 << std::endl;
                          std::cout << "001:" << Gamma_udd(0, 0, 1) << " " << gamma4d_0_0_1 << std::endl;
                          std::cout << "002:" << Gamma_udd(0, 0, 2) << " " << gamma4d_0_0_2 << std::endl;
                          std::cout << "003:" << Gamma_udd(0, 0, 3) << " " << gamma4d_0_0_3 << std::endl;
                          std::cout << "010:" << Gamma_udd(0, 1, 0) << " " << gamma4d_0_1_0 << std::endl;
                          std::cout << "011:" << Gamma_udd(0, 1, 1) << " " << gamma4d_0_1_1 << std::endl;
                          std::cout << "012:" << Gamma_udd(0, 1, 2) << " " << gamma4d_0_1_2 << std::endl;
                          std::cout << "013:" << Gamma_udd(0, 1, 3) << " " << gamma4d_0_1_3 << std::endl;
                          std::cout << "020:" << Gamma_udd(0, 2, 0) << " " << gamma4d_0_2_0 << std::endl;
                          std::cout << "021:" << Gamma_udd(0, 2, 1) << " " << gamma4d_0_2_1 << std::endl;
                          std::cout << "022:" << Gamma_udd(0, 2, 2) << " " << gamma4d_0_2_2 << std::endl;
                          std::cout << "023:" << Gamma_udd(0, 2, 3) << " " << gamma4d_0_2_3 << std::endl;
                          std::cout << "030:" << Gamma_udd(0, 3, 0) << " " << gamma4d_0_3_0 << std::endl;
                          std::cout << "031:" << Gamma_udd(0, 3, 1) << " " << gamma4d_0_3_1 << std::endl;
                          std::cout << "032:" << Gamma_udd(0, 3, 2) << " " << gamma4d_0_3_2 << std::endl;
                          std::cout << "033:" << Gamma_udd(0, 3, 3) << " " << gamma4d_0_3_3 << std::endl;
                          std::cout << "100:" << Gamma_udd(1, 0, 0) << " " << gamma4d_1_0_0 << std::endl;
                          std::cout << "101:" << Gamma_udd(1, 0, 1) << " " << gamma4d_1_0_1 << std::endl;
                          std::cout << "102:" << Gamma_udd(1, 0, 2) << " " << gamma4d_1_0_2 << std::endl;
                          std::cout << "103:" << Gamma_udd(1, 0, 3) << " " << gamma4d_1_0_3 << std::endl;
                          std::cout << "110:" << Gamma_udd(1, 1, 0) << " " << gamma4d_1_1_0 << std::endl;
                          std::cout << "111:" << Gamma_udd(1, 1, 1) << " " << gamma4d_1_1_1 << std::endl;
                          std::cout << "112:" << Gamma_udd(1, 1, 2) << " " << gamma4d_1_1_2 << std::endl;
                          std::cout << "113:" << Gamma_udd(1, 1, 3) << " " << gamma4d_1_1_3 << std::endl;
                          std::cout << "120:" << Gamma_udd(1, 2, 0) << " " << gamma4d_1_2_0 << std::endl;
                          std::cout << "121:" << Gamma_udd(1, 2, 1) << " " << gamma4d_1_2_1 << std::endl;
                          std::cout << "122:" << Gamma_udd(1, 2, 2) << " " << gamma4d_1_2_2 << std::endl;
                          std::cout << "123:" << Gamma_udd(1, 2, 3) << " " << gamma4d_1_2_3 << std::endl;
                          std::cout << "130:" << Gamma_udd(1, 3, 0) << " " << gamma4d_1_3_0 << std::endl;
                          std::cout << "131:" << Gamma_udd(1, 3, 1) << " " << gamma4d_1_3_1 << std::endl;
                          std::cout << "132:" << Gamma_udd(1, 3, 2) << " " << gamma4d_1_3_2 << std::endl;
                          std::cout << "133:" << Gamma_udd(1, 3, 3) << " " << gamma4d_1_3_3 << std::endl;
                          std::cout << "200:" << Gamma_udd(2, 0, 0) << " " << gamma4d_2_0_0 << std::endl;
                          std::cout << "201:" << Gamma_udd(2, 0, 1) << " " << gamma4d_2_0_1 << std::endl;
                          std::cout << "202:" << Gamma_udd(2, 0, 2) << " " << gamma4d_2_0_2 << std::endl;
                          std::cout << "203:" << Gamma_udd(2, 0, 3) << " " << gamma4d_2_0_3 << std::endl;
                          std::cout << "210:" << Gamma_udd(2, 1, 0) << " " << gamma4d_2_1_0 << std::endl;
                          std::cout << "211:" << Gamma_udd(2, 1, 1) << " " << gamma4d_2_1_1 << std::endl;
                          std::cout << "212:" << Gamma_udd(2, 1, 2) << " " << gamma4d_2_1_2 << std::endl;
                          std::cout << "213:" << Gamma_udd(2, 1, 3) << " " << gamma4d_2_1_3 << std::endl;
                          std::cout << "220:" << Gamma_udd(2, 2, 0) << " " << gamma4d_2_2_0 << std::endl;
                          std::cout << "221:" << Gamma_udd(2, 2, 1) << " " << gamma4d_2_2_1 << std::endl;
                          std::cout << "222:" << Gamma_udd(2, 2, 2) << " " << gamma4d_2_2_2 << std::endl;
                          std::cout << "223:" << Gamma_udd(2, 2, 3) << " " << gamma4d_2_2_3 << std::endl;
                          std::cout << "230:" << Gamma_udd(2, 3, 0) << " " << gamma4d_2_3_0 << std::endl;
                          std::cout << "231:" << Gamma_udd(2, 3, 1) << " " << gamma4d_2_3_1 << std::endl;
                          std::cout << "232:" << Gamma_udd(2, 3, 2) << " " << gamma4d_2_3_2 << std::endl;
                          std::cout << "233:" << Gamma_udd(2, 3, 3) << " " << gamma4d_2_3_3 << std::endl;
                          std::cout << "300:" << Gamma_udd(3, 0, 0) << " " << gamma4d_3_0_0 << std::endl;
                          std::cout << "301:" << Gamma_udd(3, 0, 1) << " " << gamma4d_3_0_1 << std::endl;
                          std::cout << "302:" << Gamma_udd(3, 0, 2) << " " << gamma4d_3_0_2 << std::endl;
                          std::cout << "303:" << Gamma_udd(3, 0, 3) << " " << gamma4d_3_0_3 << std::endl;
                          std::cout << "310:" << Gamma_udd(3, 1, 0) << " " << gamma4d_3_1_0 << std::endl;
                          std::cout << "311:" << Gamma_udd(3, 1, 1) << " " << gamma4d_3_1_1 << std::endl;
                          std::cout << "312:" << Gamma_udd(3, 1, 2) << " " << gamma4d_3_1_2 << std::endl;
                          std::cout << "313:" << Gamma_udd(3, 1, 3) << " " << gamma4d_3_1_3 << std::endl;
                          std::cout << "320:" << Gamma_udd(3, 2, 0) << " " << gamma4d_3_2_0 << std::endl;
                          std::cout << "321:" << Gamma_udd(3, 2, 1) << " " << gamma4d_3_2_1 << std::endl;
                          std::cout << "322:" << Gamma_udd(3, 2, 2) << " " << gamma4d_3_2_2 << std::endl;
                          std::cout << "323:" << Gamma_udd(3, 2, 3) << " " << gamma4d_3_2_3 << std::endl;
                          std::cout << "330:" << Gamma_udd(3, 3, 0) << " " << gamma4d_3_3_0 << std::endl;
                          std::cout << "331:" << Gamma_udd(3, 3, 1) << " " << gamma4d_3_3_1 << std::endl;
                          std::cout << "332:" << Gamma_udd(3, 3, 2) << " " << gamma4d_3_3_2 << std::endl;
                          std::cout << "333:" << Gamma_udd(3, 3, 3) << " " << gamma4d_3_3_3 << std::endl;
                          std::cout << std::endl;
                          std::cout << "000:" << Gamma_fluid_udd(0, 0, 0) << " " << gamma4d_0_0_0 << std::endl;
                          std::cout << "001:" << Gamma_fluid_udd(0, 0, 1) << " " << gamma4d_0_0_1 << std::endl;
                          std::cout << "002:" << Gamma_fluid_udd(0, 0, 2) << " " << gamma4d_0_0_2 << std::endl;
                          std::cout << "003:" << Gamma_fluid_udd(0, 0, 3) << " " << gamma4d_0_0_3 << std::endl;
                          std::cout << "010:" << Gamma_fluid_udd(0, 1, 0) << " " << gamma4d_0_1_0 << std::endl;
                          std::cout << "011:" << Gamma_fluid_udd(0, 1, 1) << " " << gamma4d_0_1_1 << std::endl;
                          std::cout << "012:" << Gamma_fluid_udd(0, 1, 2) << " " << gamma4d_0_1_2 << std::endl;
                          std::cout << "013:" << Gamma_fluid_udd(0, 1, 3) << " " << gamma4d_0_1_3 << std::endl;
                          std::cout << "020:" << Gamma_fluid_udd(0, 2, 0) << " " << gamma4d_0_2_0 << std::endl;
                          std::cout << "021:" << Gamma_fluid_udd(0, 2, 1) << " " << gamma4d_0_2_1 << std::endl;
                          std::cout << "022:" << Gamma_fluid_udd(0, 2, 2) << " " << gamma4d_0_2_2 << std::endl;
                          std::cout << "023:" << Gamma_fluid_udd(0, 2, 3) << " " << gamma4d_0_2_3 << std::endl;
                          std::cout << "030:" << Gamma_fluid_udd(0, 3, 0) << " " << gamma4d_0_3_0 << std::endl;
                          std::cout << "031:" << Gamma_fluid_udd(0, 3, 1) << " " << gamma4d_0_3_1 << std::endl;
                          std::cout << "032:" << Gamma_fluid_udd(0, 3, 2) << " " << gamma4d_0_3_2 << std::endl;
                          std::cout << "033:" << Gamma_fluid_udd(0, 3, 3) << " " << gamma4d_0_3_3 << std::endl;
                          std::cout << "100:" << Gamma_fluid_udd(1, 0, 0) << " " << gamma4d_1_0_0 << std::endl;
                          std::cout << "101:" << Gamma_fluid_udd(1, 0, 1) << " " << gamma4d_1_0_1 << std::endl;
                          std::cout << "102:" << Gamma_fluid_udd(1, 0, 2) << " " << gamma4d_1_0_2 << std::endl;
                          std::cout << "103:" << Gamma_fluid_udd(1, 0, 3) << " " << gamma4d_1_0_3 << std::endl;
                          std::cout << "110:" << Gamma_fluid_udd(1, 1, 0) << " " << gamma4d_1_1_0 << std::endl;
                          std::cout << "111:" << Gamma_fluid_udd(1, 1, 1) << " " << gamma4d_1_1_1 << std::endl;
                          std::cout << "112:" << Gamma_fluid_udd(1, 1, 2) << " " << gamma4d_1_1_2 << std::endl;
                          std::cout << "113:" << Gamma_fluid_udd(1, 1, 3) << " " << gamma4d_1_1_3 << std::endl;
                          std::cout << "120:" << Gamma_fluid_udd(1, 2, 0) << " " << gamma4d_1_2_0 << std::endl;
                          std::cout << "121:" << Gamma_fluid_udd(1, 2, 1) << " " << gamma4d_1_2_1 << std::endl;
                          std::cout << "122:" << Gamma_fluid_udd(1, 2, 2) << " " << gamma4d_1_2_2 << std::endl;
                          std::cout << "123:" << Gamma_fluid_udd(1, 2, 3) << " " << gamma4d_1_2_3 << std::endl;
                          std::cout << "130:" << Gamma_fluid_udd(1, 3, 0) << " " << gamma4d_1_3_0 << std::endl;
                          std::cout << "131:" << Gamma_fluid_udd(1, 3, 1) << " " << gamma4d_1_3_1 << std::endl;
                          std::cout << "132:" << Gamma_fluid_udd(1, 3, 2) << " " << gamma4d_1_3_2 << std::endl;
                          std::cout << "133:" << Gamma_fluid_udd(1, 3, 3) << " " << gamma4d_1_3_3 << std::endl;
                          std::cout << "200:" << Gamma_fluid_udd(2, 0, 0) << " " << gamma4d_2_0_0 << std::endl;
                          std::cout << "201:" << Gamma_fluid_udd(2, 0, 1) << " " << gamma4d_2_0_1 << std::endl;
                          std::cout << "202:" << Gamma_fluid_udd(2, 0, 2) << " " << gamma4d_2_0_2 << std::endl;
                          std::cout << "203:" << Gamma_fluid_udd(2, 0, 3) << " " << gamma4d_2_0_3 << std::endl;
                          std::cout << "210:" << Gamma_fluid_udd(2, 1, 0) << " " << gamma4d_2_1_0 << std::endl;
                          std::cout << "211:" << Gamma_fluid_udd(2, 1, 1) << " " << gamma4d_2_1_1 << std::endl;
                          std::cout << "212:" << Gamma_fluid_udd(2, 1, 2) << " " << gamma4d_2_1_2 << std::endl;
                          std::cout << "213:" << Gamma_fluid_udd(2, 1, 3) << " " << gamma4d_2_1_3 << std::endl;
                          std::cout << "220:" << Gamma_fluid_udd(2, 2, 0) << " " << gamma4d_2_2_0 << std::endl;
                          std::cout << "221:" << Gamma_fluid_udd(2, 2, 1) << " " << gamma4d_2_2_1 << std::endl;
                          std::cout << "222:" << Gamma_fluid_udd(2, 2, 2) << " " << gamma4d_2_2_2 << std::endl;
                          std::cout << "223:" << Gamma_fluid_udd(2, 2, 3) << " " << gamma4d_2_2_3 << std::endl;
                          std::cout << "230:" << Gamma_fluid_udd(2, 3, 0) << " " << gamma4d_2_3_0 << std::endl;
                          std::cout << "231:" << Gamma_fluid_udd(2, 3, 1) << " " << gamma4d_2_3_1 << std::endl;
                          std::cout << "232:" << Gamma_fluid_udd(2, 3, 2) << " " << gamma4d_2_3_2 << std::endl;
                          std::cout << "233:" << Gamma_fluid_udd(2, 3, 3) << " " << gamma4d_2_3_3 << std::endl;
                          std::cout << "300:" << Gamma_fluid_udd(3, 0, 0) << " " << gamma4d_3_0_0 << std::endl;
                          std::cout << "301:" << Gamma_fluid_udd(3, 0, 1) << " " << gamma4d_3_0_1 << std::endl;
                          std::cout << "302:" << Gamma_fluid_udd(3, 0, 2) << " " << gamma4d_3_0_2 << std::endl;
                          std::cout << "303:" << Gamma_fluid_udd(3, 0, 3) << " " << gamma4d_3_0_3 << std::endl;
                          std::cout << "310:" << Gamma_fluid_udd(3, 1, 0) << " " << gamma4d_3_1_0 << std::endl;
                          std::cout << "311:" << Gamma_fluid_udd(3, 1, 1) << " " << gamma4d_3_1_1 << std::endl;
                          std::cout << "312:" << Gamma_fluid_udd(3, 1, 2) << " " << gamma4d_3_1_2 << std::endl;
                          std::cout << "313:" << Gamma_fluid_udd(3, 1, 3) << " " << gamma4d_3_1_3 << std::endl;
                          std::cout << "320:" << Gamma_fluid_udd(3, 2, 0) << " " << gamma4d_3_2_0 << std::endl;
                          std::cout << "321:" << Gamma_fluid_udd(3, 2, 1) << " " << gamma4d_3_2_1 << std::endl;
                          std::cout << "322:" << Gamma_fluid_udd(3, 2, 2) << " " << gamma4d_3_2_2 << std::endl;
                          std::cout << "323:" << Gamma_fluid_udd(3, 2, 3) << " " << gamma4d_3_2_3 << std::endl;
                          std::cout << "330:" << Gamma_fluid_udd(3, 3, 0) << " " << gamma4d_3_3_0 << std::endl;
                          std::cout << "331:" << Gamma_fluid_udd(3, 3, 1) << " " << gamma4d_3_3_1 << std::endl;
                          std::cout << "332:" << Gamma_fluid_udd(3, 3, 2) << " " << gamma4d_3_3_2 << std::endl;
                          std::cout << "333:" << Gamma_fluid_udd(3, 3, 3) << " " << gamma4d_3_3_3 << std::endl;
                          exit(EXIT_FAILURE); */
                          // end of 4-Christoeffel print
                          // -----------------------------------

                          // Compute F Gam and G Gam matrices
                          ScrArray2D<Real> F_Gamma_AB = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);
                          ScrArray2D<Real> G_Gamma_AB = ScrArray2D<Real>(member.team_scratch(scr_level), num_points_, num_points_);

                          par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx)
                          {
                              int row = int(idx / num_points_);
                              int col = idx - row * num_points_;

                              Real sum_nuhatmuhat_f = 0.;
                              Real sum_nuhatmuhat_g = 0.;
                              for (int nuhatmuhat = 0; nuhatmuhat < 16; nuhatmuhat++)
                              {
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
                          for (int idx = 0; idx < num_points_ * num_points_; idx++)
                          {
                              int idx_b = int(idx / num_points_);
                              int idx_a = idx - idx_b * num_points_;

                              int idx_united = IndicesUnited(nu, en, idx_a, num_species_, num_energy_bins_, num_points_);

                              g_rhs_scratch(idx_b) -=
                                  (F_Gamma_AB(idx_b, idx_a) + G_Gamma_AB(idx_b, idx_a)) * (gam0 * f0_(m, idx_united, k, j, i) + gam1 * f1_(m, idx_united, k, j, i));

                              K += F_Gamma_AB(idx_b, idx_a) * F_Gamma_AB(idx_b, idx_a);
                          }
                          K = sqrt(K);

                          // adding energy coupling terms for multi-energy case
                          if (num_energy_bins_ > 1)
                          {
                              ScrArray1D<Real> energy_terms = ScrArray1D<Real>(member.team_scratch(scr_level), num_points_);
                              par_for_inner(member, 0, num_points_ - 1, [&](const int idx)
                              {
                                  Real part_sum_idx = 0.;
                                  for (int A = 0; A < num_points_; A++)
                                  {
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

                              for (int idx = 0; idx < num_points_; idx++)
                              {
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

                          par_for_inner(member, 0, num_points_ * num_points_ - 1, [&](const int idx)
                          {
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

                          par_for_inner(member, 0, num_points_ - 1, [&](const int idx)
                          {
                              Real final_result = 0.;
                              Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [&](const int A, Real& partial_sum)
                              {
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
