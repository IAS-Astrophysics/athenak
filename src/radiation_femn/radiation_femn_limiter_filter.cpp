//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_limiter_filter.cpp
//! \brief implementation of limiters and filters for FEM_N and FP_N

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cell_locations.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

KOKKOS_INLINE_FUNCTION
Real Lanczos(Real eta) {
  return eta == 0 ? 1 : Kokkos::sin(eta) / eta;
}

KOKKOS_INLINE_FUNCTION
Real minmod2(Real a, Real b, Real c) {
  auto signa = int((0 < a) - (a < 0));
  auto signb = int((0 < b) - (b < 0));
  auto signc = int((0 < c) - (c < 0));

  auto absa = Kokkos::fabs(a);
  auto absb = Kokkos::fabs(b);
  auto absc = Kokkos::fabs(c);

  auto s = signa * int(Kokkos::fabs(signa + signb + signc) == 3);
  auto min_abs_b_abs_c = Kokkos::fmin(absb, absc);

  auto result = (s * absa) * int(absa < 2.0 * min_abs_b_abs_c) +
                (s * min_abs_b_abs_c) * int(absa >= 2.0 * min_abs_b_abs_c);

  return result;
}

KOKKOS_INLINE_FUNCTION
Real minmod(Real a, Real b, Real c) {
  auto signa = int((0 < a) - (a < 0));
  auto signb = int((0 < b) - (b < 0));
  auto signc = int((0 < c) - (c < 0));

  auto absa = Kokkos::fabs(a);
  auto absb = Kokkos::fabs(b);
  auto absc = Kokkos::fabs(c);

  auto sign = int(Kokkos::abs(signa + signb + signc) == 3);
  auto min_abs_a_abs_b_abs_c = Kokkos::fmin(Kokkos::fmin(absa, absb), absc);

  return signa * sign * min_abs_a_abs_b_abs_c;
}

KOKKOS_INLINE_FUNCTION Real slope_limiter(const Real a, const Real b, const Real c, const LimiterDG limiter_dg_minmod_type) {
  if (limiter_dg_minmod_type == LimiterDG::minmod2) {
    return minmod2(a, b, c);
  } else {
    return minmod(a, b, c);
  }
}

/* \fn RadiationFEMN::ApplyFilterLanczos
 *
 * \brief Applies a Lanczos for all angles in each energy bin for
 * FP_N solutions.
 */
TaskStatus RadiationFEMN::ApplyFilterLanczos(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int npts1 = pmy_pack->pradfemn->num_points_total - 1;
  auto &num_points_ = pmy_pack->pradfemn->num_points;
  auto &num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  auto &num_species_ = pmy_pack->pradfemn->num_species;
  auto &lmax_ = pmy_pack->pradfemn->lmax;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &angular_grid_ = pmy_pack->pradfemn->angular_grid;
  Real filtstrength = -(pmy_pack->pmesh->dt) * filter_sigma_eff / log(Lanczos(Real(lmax) / (Real(lmax) + 1.0)));

  par_for("radiation_femn_filter_Lanczos", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, npts1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int nuenang) {

            RadiationFEMNPhaseIndices idcs = IndicesComponent(nuenang, num_points_, num_energy_bins_, num_species_);
            int B = idcs.angidx;
            auto lval = angular_grid_(B, 0);

            f0_(m, nuenang, k, j, i) = Kokkos::pow(Lanczos(Real(lval) / (Real(lmax_) + 1.0)), filtstrength) * f0_(m, nuenang, k, j, i);
          });

  return TaskStatus::complete;
}

/* \fn RadiationFEMN::ApplyLimiterFEM
 *
 * \brief Applies a clipping limiter pointwise to the distribution function for each
 * energy bin which conserves the particle number density in the Eulerian frame.
 */
TaskStatus RadiationFEMN::ApplyLimiterFEM(Driver *pdriver, int stage) {

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &num_points_ = pmy_pack->pradfemn->num_points;
  auto &num_energy_bins_ = pmy_pack->pradfemn->num_energy_bins;
  //auto &num_species_ = pmy_pack->pradfemn->num_species;
  int nouter1 = pmy_pack->pradfemn->num_species * pmy_pack->pradfemn->num_energy_bins - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &L_mu_muhat_ = pmy_pack->pradfemn->L_mu_muhat0;
  auto &Q_matrix_ = pmy_pack->pradfemn->Q_matrix;

  int scr_level = 0;
  int scr_size = 1;
  par_for_outer("rad_femn_compute_clp_femn", DevExeSpace(), scr_size, scr_level, 0, nmb1, 0, nouter1, ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA(TeamMember_t member, const int m, const int outervar, const int k, const int j, const int i) {

                  Real numerator = 0.;
                  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [=](const int A, Real &partial_sum_angle) {
                    partial_sum_angle += f0_(m, outervar * num_energy_bins_ + A, k, j, i) * (Q_matrix_(0, A) * L_mu_muhat_(m, 0, 0, k, j, i)
                                                                                             + Q_matrix_(1, A) * L_mu_muhat_(m, 0, 1, k, j, i) +
                                                                                             Q_matrix_(2, A) * L_mu_muhat_(m, 0, 2, k, j, i) +
                                                                                             Q_matrix_(3, A) * L_mu_muhat_(m, 0, 3, k, j, i));
                  }, numerator);

                  Real denominator = 0.;
                  Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, 0, num_points_), [=](const int A, Real &partial_sum_angle) {
                    Real tmp = f0_(m, outervar * num_energy_bins_ + A, k, j, i);
                    partial_sum_angle += Kokkos::fmax(0, tmp) * (Q_matrix_(0, A) * L_mu_muhat_(m, 0, 0, k, j, i)
                                                                 + Q_matrix_(1, A) * L_mu_muhat_(m, 0, 1, k, j, i) + Q_matrix_(2, A) * L_mu_muhat_(m, 0, 2, k, j, i) +
                                                                 Q_matrix_(3, A) * L_mu_muhat_(m, 0, 3, k, j, i));
                  }, denominator);
                  member.team_barrier();

                  Real correction_factor = (numerator > 0 && denominator != 0) ? (numerator / denominator) : 0.0;

                  par_for_inner(member, 0, num_points_ - 1, [&](const int A) {
                    Real tmp = f0_(m, outervar * num_energy_bins_ + A, k, j, i);
                    f0_(m, outervar * num_energy_bins_ + A, k, j, i) = correction_factor * Kokkos::fmax(tmp, 0.);
                  });
                  member.team_barrier();

                });

  return TaskStatus::complete;
}

TaskStatus RadiationFEMN::ApplyLimiterDG(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int npts1 = pmy_pack->pradfemn->num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  bool &one_d = pmy_pack->pmesh->one_d;
  bool &two_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &limiter_dg_minmod_type_ = pmy_pack->pradfemn->limiter_dg_minmod_type;

  auto &f0_ = pmy_pack->pradfemn->f0;
  auto &ftemp_ = pmy_pack->pradfemn->ftemp;

  Kokkos::deep_copy(ftemp_, 0.);

  if (one_d) {
    par_for("radiation_femn_limiter_dg1d", DevExeSpace(), 0, nmb1, 0, npts1, ks, ke, js, je, is, int(ie / 2) + 1,
            KOKKOS_LAMBDA(const int m, const int enang, const int k, const int j, const int i) {

              auto kk = k;
              auto jj = j;
              auto ii = 2 * i - 2;

              auto f0_cellavg = (0.5) * (f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj, ii + 1));
              auto f0_cellavg_mx = (0.5) * (f0_(m, enang, kk, jj, ii - 2) + f0_(m, enang, kk, jj, ii - 1));
              auto f0_cellavg_px = (0.5) * (f0_(m, enang, kk, jj, ii + 2) + f0_(m, enang, kk, jj, ii + 3));

              auto dminusx = (f0_cellavg - f0_cellavg_mx) / (2.0 * mbsize.d_view(m).dx1);
              auto dplusx = (f0_cellavg_px - f0_cellavg) / (2.0 * mbsize.d_view(m).dx1);
              auto islopex = 2.0 * (f0_(m, enang, kk, jj, ii + 1) - f0_(m, enang, kk, jj, ii)) / (2.0 * mbsize.d_view(m).dx1);

              auto sigmax = slope_limiter(islopex, dminusx, dplusx, limiter_dg_minmod_type_);

              Real &x1min = mbsize.d_view(m).x1min;
              Real &x1max = mbsize.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real xii = CellCenterX(ii - is, nx1, x1min, x1max);
              Real xiip1 = CellCenterX(ii - is + 1, nx1, x1min, x1max);

              auto xmean = 0.5 * (xii + xiip1);

              ftemp_(m, enang, kk, jj, ii) = f0_cellavg + sigmax * (xii - xmean);
              ftemp_(m, enang, kk, jj, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean);

            });
  }

  if (two_d) {
    par_for("radiation_femn_limiter_dg2d", DevExeSpace(), 0, nmb1, 0, npts1, ks, ke, js, int(je / 2) + 1, is, int(ie / 2) + 1,
            KOKKOS_LAMBDA(const int m, const int enang, const int k, const int j, const int i) {

              auto kk = k;
              auto jj = 2 * j - 2;
              auto ii = 2 * i - 2;

              auto f0_cellavg = (0.25) * (f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj, ii + 1) + f0_(m, enang, kk, jj + 1, ii) + f0_(m, enang, kk, jj + 1, ii + 1));
              auto f0_cellavg_mx =
                  (0.25) * (f0_(m, enang, kk, jj, ii - 2) + f0_(m, enang, kk, jj, ii - 1) + f0_(m, enang, kk, jj + 1, ii - 2) + f0_(m, enang, kk, jj + 1, ii - 1));
              auto f0_cellavg_px =
                  (0.25) * (f0_(m, enang, kk, jj, ii + 2) + f0_(m, enang, kk, jj, ii + 3) + f0_(m, enang, kk, jj + 1, ii + 2) + f0_(m, enang, kk, jj + 1, ii + 3));
              auto f0_cellavg_my =
                  (0.25) * (f0_(m, enang, kk, jj - 2, ii) + f0_(m, enang, kk, jj - 2, ii + 1) + f0_(m, enang, kk, jj - 1, ii) + f0_(m, enang, kk, jj - 1, ii + 1));
              auto f0_cellavg_py =
                  (0.25) * (f0_(m, enang, kk, jj + 2, ii) + f0_(m, enang, kk, jj + 2, ii + 1) + f0_(m, enang, kk, jj + 3, ii) + f0_(m, enang, kk, jj + 3, ii + 1));

              auto dminusx = (f0_cellavg - f0_cellavg_mx) / (2.0 * mbsize.d_view(m).dx1);
              auto dplusx = (f0_cellavg_px - f0_cellavg) / (2.0 * mbsize.d_view(m).dx1);
              auto islopex = 2.0 * (f0_(m, enang, kk, jj, ii + 1) - f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj + 1, ii + 1) - f0_(m, enang, kk, jj + 1, ii))
                             / (2.0 * 2.0 * mbsize.d_view(m).dx1);

              auto dminusy = (f0_cellavg - f0_cellavg_my) / (2.0 * mbsize.d_view(m).dx2);
              auto dplusy = (f0_cellavg_py - f0_cellavg) / (2.0 * mbsize.d_view(m).dx2);
              auto islopey = 2.0 * (f0_(m, enang, kk, jj + 1, ii) - f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj + 1, ii + 1) - f0_(m, enang, kk, jj, ii + 1))
                             / (2.0 * 2.0 * mbsize.d_view(m).dx2);

              auto sigmax = slope_limiter(islopex, dminusx, dplusx, limiter_dg_minmod_type_);
              auto sigmay = slope_limiter(islopey, dminusy, dplusy, limiter_dg_minmod_type_);

              Real &x1min = mbsize.d_view(m).x1min;
              Real &x1max = mbsize.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real xii = CellCenterX(ii - is, nx1, x1min, x1max);
              Real xiip1 = CellCenterX(ii - is + 1, nx1, x1min, x1max);

              Real &x2min = mbsize.d_view(m).x2min;
              Real &x2max = mbsize.d_view(m).x2max;
              int nx2 = indcs.nx2;
              Real xjj = CellCenterX(jj - js, nx2, x2min, x2max);
              Real xjjp1 = CellCenterX(jj - js + 1, nx2, x2min, x2max);

              auto xmean = 0.5 * (xii + xiip1);
              auto ymean = 0.5 * (xjj + xjjp1);

              ftemp_(m, enang, kk, jj, ii) = f0_cellavg + sigmax * (xii - xmean) + sigmay * (xjj - ymean);
              ftemp_(m, enang, kk, jj, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjj - ymean);
              ftemp_(m, enang, kk, jj + 1, ii) = f0_cellavg + sigmax * (xii - xmean) + sigmay * (xjjp1 - ymean);
              ftemp_(m, enang, kk, jj + 1, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjjp1 - ymean);
            });
  }

  if (three_d) {
    par_for("radiation_femn_limiter_dg3d", DevExeSpace(), 0, nmb1, 0, npts1, ks, int(ke / 2) + 1, js, int(je / 2) + 1, is, int(ie / 2) + 1,
            KOKKOS_LAMBDA(const int m, const int enang, const int k, const int j, const int i) {

              auto kk = 2 * k - 2;
              auto jj = 2 * j - 2;
              auto ii = 2 * i - 2;

              auto f0_cellavg = (1.0 / 8.0)
                                * (f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj, ii + 1) + f0_(m, enang, kk, jj + 1, ii) + f0_(m, enang, kk, jj + 1, ii + 1)
                                   + f0_(m, enang, kk + 1, jj, ii)
                                   + f0_(m, enang, kk + 1, jj, ii + 1) + f0_(m, enang, kk + 1, jj + 1, ii) + f0_(m, enang, kk + 1, jj + 1, ii + 1));
              auto f0_cellavg_mx =
                  (1.0 / 8.0) * (f0_(m, enang, kk, jj, ii - 2) + f0_(m, enang, kk, jj, ii - 1) + f0_(m, enang, kk, jj + 1, ii - 2) + f0_(m, enang, kk, jj + 1, ii - 1)
                                 + f0_(m, enang, kk + 1, jj, ii - 2) + f0_(m, enang, kk + 1, jj, ii - 1) + f0_(m, enang, kk + 1, jj + 1, ii - 2)
                                 + f0_(m, enang, kk + 1, jj + 1, ii - 1));
              auto f0_cellavg_px = (1.0 / 8.0)
                                   * (f0_(m, enang, kk, jj, ii + 2) + f0_(m, enang, kk, jj, ii + 3) + f0_(m, enang, kk, jj + 1, ii + 2) + f0_(m, enang, kk, jj + 1, ii + 3)
                                      + f0_(m, enang, kk + 1, jj, ii + 2) + f0_(m, enang, kk + 1, jj, ii + 3) + f0_(m, enang, kk + 1, jj + 1, ii + 2)
                                      + f0_(m, enang, kk + 1, jj + 1, ii + 3));
              auto f0_cellavg_my = (1.0 / 8.0)
                                   * (f0_(m, enang, kk, jj - 2, ii) + f0_(m, enang, kk, jj - 2, ii + 1) + f0_(m, enang, kk, jj - 1, ii) + f0_(m, enang, kk, jj - 1, ii + 1)
                                      + f0_(m, enang, kk + 1, jj - 2, ii) + f0_(m, enang, kk + 1, jj - 2, ii + 1) + f0_(m, enang, kk + 1, jj - 1, ii)
                                      + f0_(m, enang, kk + 1, jj - 1, ii + 1));
              auto f0_cellavg_py = (1.0 / 8.0)
                                   * (f0_(m, enang, kk, jj + 2, ii) + f0_(m, enang, kk, jj + 2, ii + 1) + f0_(m, enang, kk, jj + 3, ii) + f0_(m, enang, kk, jj + 3, ii + 1)
                                      + f0_(m, enang, kk + 1, jj + 2, ii) + f0_(m, enang, kk + 1, jj + 2, ii + 1) + f0_(m, enang, kk + 1, jj + 3, ii)
                                      + f0_(m, enang, kk + 1, jj + 3, ii + 1));
              auto f0_cellavg_mz = (1.0 / 8.0)
                                   * (f0_(m, enang, kk - 2, jj, ii) + f0_(m, enang, kk - 2, jj, ii + 1) + f0_(m, enang, kk - 2, jj + 1, ii) + f0_(m, enang, kk - 2, jj + 1, ii + 1)
                                      + f0_(m, enang, kk - 1, jj, ii) + f0_(m, enang, kk - 1, jj, ii + 1) + f0_(m, enang, kk - 1, jj + 1, ii) +
                                      f0_(m, enang, kk - 1, jj + 1, ii + 1));
              auto f0_cellavg_pz = (1.0 / 8.0)
                                   * (f0_(m, enang, kk + 2, jj, ii) + f0_(m, enang, kk + 2, jj, ii + 1) + f0_(m, enang, kk + 2, jj + 1, ii) + f0_(m, enang, kk + 2, jj + 1, ii + 1)
                                      + f0_(m, enang, kk + 3, jj, ii) + f0_(m, enang, kk + 3, jj, ii + 1) + f0_(m, enang, kk + 3, jj + 1, ii) +
                                      f0_(m, enang, kk + 3, jj + 1, ii + 1));

              auto dminusx = (f0_cellavg - f0_cellavg_mx) / (2.0 * mbsize.d_view(m).dx1);
              auto dplusx = (f0_cellavg_px - f0_cellavg) / (2.0 * mbsize.d_view(m).dx1);
              auto islopex = 2.0 * (f0_(m, enang, kk, jj, ii + 1) - f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj + 1, ii + 1) - f0_(m, enang, kk, jj + 1, ii)
                                    + f0_(m, enang, kk + 1, jj, ii + 1) - f0_(m, enang, kk + 1, jj, ii) + f0_(m, enang, kk + 1, jj + 1, ii + 1) - f0_(m, enang, kk + 1, jj + 1, ii))
                             / (2.0 * 2.0 * 2.0 * mbsize.d_view(m).dx1);

              auto dminusy = (f0_cellavg - f0_cellavg_my) / (2.0 * mbsize.d_view(m).dx2);
              auto dplusy = (f0_cellavg_py - f0_cellavg) / (2.0 * mbsize.d_view(m).dx2);
              auto islopey = 2.0 * (f0_(m, enang, kk, jj + 1, ii) - f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk, jj + 1, ii + 1) - f0_(m, enang, kk, jj, ii + 1)
                                    + f0_(m, enang, kk + 1, jj + 1, ii) - f0_(m, enang, kk + 1, jj, ii) + f0_(m, enang, kk + 1, jj + 1, ii + 1) - f0_(m, enang, kk + 1, jj, ii + 1))
                             / (2.0 * 2.0 * 2.0 * mbsize.d_view(m).dx2);

              auto dminusz = (f0_cellavg - f0_cellavg_mz) / (2.0 * mbsize.d_view(m).dx3);
              auto dplusz = (f0_cellavg_pz - f0_cellavg) / (2.0 * mbsize.d_view(m).dx3);
              auto islopez = 2.0 * (f0_(m, enang, kk + 1, jj, ii) - f0_(m, enang, kk, jj, ii) + f0_(m, enang, kk + 1, jj, ii + 1) - f0_(m, enang, kk, jj, ii + 1)
                                    + f0_(m, enang, kk + 1, jj + 1, ii) - f0_(m, enang, kk, jj + 1, ii) + f0_(m, enang, kk + 1, jj + 1, ii + 1) - f0_(m, enang, kk, jj, ii + 1))
                             / (2.0 * 2.0 * 2.0 * mbsize.d_view(m).dx3);

              auto sigmax = slope_limiter(islopex, dminusx, dplusx, limiter_dg_minmod_type_);
              auto sigmay = slope_limiter(islopey, dminusy, dplusy, limiter_dg_minmod_type_);
              auto sigmaz = slope_limiter(islopez, dminusz, dplusz, limiter_dg_minmod_type_);

              Real &x1min = mbsize.d_view(m).x1min;
              Real &x1max = mbsize.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real xii = CellCenterX(ii - is, nx1, x1min, x1max);
              Real xiip1 = CellCenterX(ii - is + 1, nx1, x1min, x1max);

              Real &x2min = mbsize.d_view(m).x2min;
              Real &x2max = mbsize.d_view(m).x2max;
              int nx2 = indcs.nx2;
              Real xjj = CellCenterX(jj - js, nx2, x2min, x2max);
              Real xjjp1 = CellCenterX(jj - js + 1, nx2, x2min, x2max);

              Real &x3min = mbsize.d_view(m).x3min;
              Real &x3max = mbsize.d_view(m).x3max;
              int nx3 = indcs.nx3;
              Real xkk = CellCenterX(kk - ks, nx3, x3min, x3max);
              Real xkkp1 = CellCenterX(kk - ks + 1, nx3, x3min, x3max);

              auto xmean = 0.5 * (xii + xiip1);
              auto ymean = 0.5 * (xjj + xjjp1);
              auto zmean = 0.5 * (xkk + xkkp1);

              ftemp_(m, enang, kk, jj, ii) = f0_cellavg + sigmax * (xii - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkk - zmean);
              ftemp_(m, enang, kk, jj, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkk - zmean);
              ftemp_(m, enang, kk, jj + 1, ii) = f0_cellavg + sigmax * (xii - xmean) + sigmay * (xjjp1 - ymean) + sigmaz * (xkk - zmean);
              ftemp_(m, enang, kk, jj + 1, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjjp1 - ymean) + sigmaz * (xkk - zmean);
              ftemp_(m, enang, kk + 1, jj, ii) = f0_cellavg + sigmax * (xii - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkkp1 - zmean);
              ftemp_(m, enang, kk + 1, jj, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkkp1 - zmean);
              ftemp_(m, enang, kk + 1, jj + 1, ii) = f0_cellavg + sigmax * (xii - xmean) + sigmay * (xjjp1 - ymean) + sigmaz * (xkkp1 - zmean);
              ftemp_(m, enang, kk + 1, jj + 1, ii + 1) = f0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjjp1 - ymean) + sigmaz * (xkkp1 - zmean);
            });
  }

  Kokkos::deep_copy(f0_, ftemp_);

  return TaskStatus::complete;
}

} // namespace radiationfemn
