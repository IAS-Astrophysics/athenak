//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==============================================================================================
//! \file radiation_femn_limiter_filter.cpp
//! \brief implementation of limiters and filters for FEM_N and FP_N

#include <cmath>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cell_locations.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

    KOKKOS_INLINE_FUNCTION
    double Lanczos(double eta) {
        return eta == 0 ? 1 : sin(eta) / eta;
    }

    KOKKOS_INLINE_FUNCTION
    Real minmod2(Real a, Real b, Real c) {
        auto signa = int((0 < a) - (a < 0));
        auto signb = int((0 < b) - (b < 0));
        auto signc = int((0 < c) - (c < 0));

        auto absa = abs(a);
        auto absb = abs(b);
        auto absc = abs(c);

        auto s = signa * int(abs(signa + signb + signc) == 3);
        auto min_abs_b_abs_c = fmin(absb, absc);

        auto result = (s * absa) * int(absa < 2.0 * min_abs_b_abs_c) +
                      (s * min_abs_b_abs_c) * int(absa >= 2.0 * min_abs_b_abs_c);

        return result;
    }

    TaskStatus RadiationFEMN::ApplyFilterLanczos(Driver *pdriver, int stage) {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nang1 = num_points - 1;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &i0_ = i0;

        auto filtstrength =
                -(pmy_pack->pmesh->dt) * filter_sigma_eff / log(Lanczos(double(lmax) / (double(lmax) + 1.0)));

        par_for("radiation_femn_filter_Lanczos", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, nang1,
                KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int A) {
                    auto lval = int(sqrt(A)); // TODO: check this to see if square root is not causing issues
                    i0_(m, A, k, j, i) =
                            pow(Lanczos(double(lval) / (double(lmax) + 1.0)), filtstrength) * i0_(m, A, k, j, i);

                });

        return TaskStatus::complete;
    }

    TaskStatus RadiationFEMN::ApplyLimiterFEM(Driver *pdriver, int stage) {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int &is = indcs.is, &ie = indcs.ie;
        int &js = indcs.js, &je = indcs.je;
        int &ks = indcs.ks, &ke = indcs.ke;
        int nang1 = num_points - 1;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &mbsize = pmy_pack->pmb->mb_size;
        auto &i0_ = i0;
        auto &mm_ = mass_matrix;
        auto &etemp0_ = etemp0;
        auto &etemp1_ = etemp1;

        Kokkos::deep_copy(etemp0_, 0.0);
        Kokkos::deep_copy(etemp1_, 0.0);

        par_for("radiation_femn_etemp_calculate", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, nang1, 0, nang1,
                KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int B, const int A) {
                    etemp0_(m, k, j, i) += mm_(B, A) * i0_(m, A, k, j, i);
                    etemp1_(m, k, j, i) += mm_(B, A) * fmax(i0_(m, A, k, j, i), 0.0);
                });

        par_for("radiation_femn_limiter_clp", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0, nang1,
                KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int A) {
                    auto theta = (etemp0_(m, k, j, i) > 0 && etemp1_(m, k, j, i) != 0) ? (etemp0_(m, k, j, i) /
                                                                                          etemp1_(m, k, j, i)) : 0.0;

                    i0_(m, A, k, j, i) = theta * fmax(i0_(m, A, k, j, i), 0.0);

                });

        return TaskStatus::complete;
    }

    TaskStatus RadiationFEMN::ApplyLimiterDG(Driver *pdriver, int stage) {
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int is = indcs.is, ie = indcs.ie;
        int js = indcs.js, je = indcs.je;
        int ks = indcs.ks, ke = indcs.ke;
        int nang1 = num_points - 1;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &mbsize = pmy_pack->pmb->mb_size;

        bool &one_d = pmy_pack->pmesh->one_d;
        bool &two_d = pmy_pack->pmesh->multi_d;
        bool &three_d = pmy_pack->pmesh->three_d;

        auto &i0_ = i0;
        auto &itemp_ = itemp;

        Kokkos::deep_copy(itemp_, 0.);

        if (one_d) {
            par_for("radiation_femn_limiter_dg1d", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je, is,
                    int(ie / 2) + 1,
                    KOKKOS_LAMBDA(const int m, const int A, const int k, const int j, const int i) {
                        auto kk = k;
                        auto jj = j;
                        auto ii = 2 * i - 2;

                        auto u0_cellavg = (0.5) * (i0_(m, A, kk, jj, ii) + i0_(m, A, kk, jj, ii + 1));
                        auto u0_cellavg_mx = (0.5) * (i0_(m, A, kk, jj, ii - 2) + i0_(m, A, kk, jj, ii - 1));
                        auto u0_cellavg_px = (0.5) * (i0_(m, A, kk, jj, ii + 2) + i0_(m, A, kk, jj, ii + 3));

                        auto dminusx = (u0_cellavg - u0_cellavg_mx) / (2.0 * mbsize.d_view(m).dx1);
                        auto dplusx = (u0_cellavg_px - u0_cellavg) / (2.0 * mbsize.d_view(m).dx1);
                        auto islopex = 2.0 * (i0_(m, A, kk, jj, ii + 1) - i0_(m, A, kk, jj, ii)) /
                                       (2.0 * mbsize.d_view(m).dx1);

                        auto sigmax = minmod2(islopex, dminusx, dplusx);

                        Real &x1min = mbsize.d_view(m).x1min;
                        Real &x1max = mbsize.d_view(m).x1max;
                        int nx1 = indcs.nx1;
                        Real xii = CellCenterX(ii - is, nx1, x1min, x1max);
                        Real xiip1 = CellCenterX(ii - is + 1, nx1, x1min, x1max);

                        auto xmean = 0.5 * (xii + xiip1);

                        itemp_(m, A, kk, jj, ii) = u0_cellavg + sigmax * (xii - xmean);
                        itemp_(m, A, kk, jj, ii + 1) = u0_cellavg + sigmax * (xiip1 - xmean);

                    });
        }

        if (two_d) {
            par_for("radiation_femn_limiter_dg2d", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, int(je / 2) + 1, is,
                    int(ie / 2) + 1,
                    KOKKOS_LAMBDA(const int m, const int A, const int k, const int j, const int i) {
                        auto kk = k;
                        auto jj = 2 * j - 2;
                        auto ii = 2 * i - 2;

                        auto i0_cellavg = (0.25) * (i0_(m, A, kk, jj, ii) + i0_(m, A, kk, jj, ii + 1) \
 + i0_(m, A, kk, jj + 1, ii) + i0_(m, A, kk, jj + 1, ii + 1));
                        auto i0_cellavg_mx = (0.25) * (i0_(m, A, kk, jj, ii - 2) + i0_(m, A, kk, jj, ii - 1) \
 + i0_(m, A, kk, jj + 1, ii - 2) + i0_(m, A, kk, jj + 1, ii - 1));
                        auto i0_cellavg_px = (0.25) * (i0_(m, A, kk, jj, ii + 2) + i0_(m, A, kk, jj, ii + 3) \
 + i0_(m, A, kk, jj + 1, ii + 2) + i0_(m, A, kk, jj + 1, ii + 3));
                        auto i0_cellavg_my = (0.25) * (i0_(m, A, kk, jj - 2, ii) + i0_(m, A, kk, jj - 2, ii + 1) \
 + i0_(m, A, kk, jj - 1, ii) + i0_(m, A, kk, jj - 1, ii + 1));
                        auto i0_cellavg_py = (0.25) * (i0_(m, A, kk, jj + 2, ii) + i0_(m, A, kk, jj + 2, ii + 1) \
 + i0_(m, A, kk, jj + 3, ii) + i0_(m, A, kk, jj + 3, ii + 1));

                        auto dminusx = (i0_cellavg - i0_cellavg_mx) / (2.0 * mbsize.d_view(m).dx1);
                        auto dplusx = (i0_cellavg_px - i0_cellavg) / (2.0 * mbsize.d_view(m).dx1);
                        auto islopex = 2.0 * (i0_(m, A, kk, jj, ii + 1) - i0_(m, A, kk, jj, ii) \
 + i0_(m, A, kk, jj + 1, ii + 1) - i0_(m, A, kk, jj + 1, ii)) /
                                       (2.0 * 2.0 * mbsize.d_view(m).dx1);

                        auto dminusy = (i0_cellavg - i0_cellavg_my) / (2.0 * mbsize.d_view(m).dx2);
                        auto dplusy = (i0_cellavg_py - i0_cellavg) / (2.0 * mbsize.d_view(m).dx2);
                        auto islopey = 2.0 * (i0_(m, A, kk, jj + 1, ii) - i0_(m, A, kk, jj, ii) \
 + i0_(m, A, kk, jj + 1, ii + 1) - i0_(m, A, kk, jj, ii + 1)) /
                                       (2.0 * 2.0 * mbsize.d_view(m).dx2);

                        auto sigmax = minmod2(islopex, dminusx, dplusx);
                        auto sigmay = minmod2(islopey, dminusy, dplusy);

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

                        itemp_(m, A, kk, jj, ii) = i0_cellavg + sigmax * (xii - xmean) + sigmay * (xjj - ymean);
                        itemp_(m, A, kk, jj, ii + 1) = i0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjj - ymean);
                        itemp_(m, A, kk, jj + 1, ii) = i0_cellavg + sigmax * (xii - xmean) + sigmay * (xjjp1 - ymean);
                        itemp_(m, A, kk, jj + 1, ii + 1) =
                                i0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjjp1 - ymean);

                    });
        }

        if (three_d) {
            par_for("radiation_femn_limiter_dg3d", DevExeSpace(), 0, nmb1, 0, nang1, ks, int(ke / 2) + 1, js,
                    int(je / 2) + 1, is, int(ie / 2) + 1,
                    KOKKOS_LAMBDA(const int m, const int A, const int k, const int j, const int i) {
                        auto kk = 2 * k - 2;
                        auto jj = 2 * j - 2;
                        auto ii = 2 * i - 2;

                        auto i0_cellavg = (1.0 / 8.0) * (i0_(m, A, kk, jj, ii) + i0_(m, A, kk, jj, ii + 1) \
 + i0_(m, A, kk, jj + 1, ii) + i0_(m, A, kk, jj + 1, ii + 1) \
 + i0_(m, A, kk + 1, jj, ii) + i0_(m, A, kk + 1, jj, ii + 1) \
 + i0_(m, A, kk + 1, jj + 1, ii) + i0_(m, A, kk + 1, jj + 1, ii + 1));
                        auto i0_cellavg_mx = (1.0 / 8.0) * (i0_(m, A, kk, jj, ii - 2) + i0_(m, A, kk, jj, ii - 1) \
 + i0_(m, A, kk, jj + 1, ii - 2) + i0_(m, A, kk, jj + 1, ii - 1) \
 + i0_(m, A, kk + 1, jj, ii - 2) + i0_(m, A, kk + 1, jj, ii - 1) \
 + i0_(m, A, kk + 1, jj + 1, ii - 2) + i0_(m, A, kk + 1, jj + 1, ii - 1));
                        auto i0_cellavg_px = (1.0 / 8.0) * (i0_(m, A, kk, jj, ii + 2) + i0_(m, A, kk, jj, ii + 3) \
 + i0_(m, A, kk, jj + 1, ii + 2) + i0_(m, A, kk, jj + 1, ii + 3) \
 + i0_(m, A, kk + 1, jj, ii + 2) + i0_(m, A, kk + 1, jj, ii + 3) \
 + i0_(m, A, kk + 1, jj + 1, ii + 2) + i0_(m, A, kk + 1, jj + 1, ii + 3));
                        auto i0_cellavg_my = (1.0 / 8.0) * (i0_(m, A, kk, jj - 2, ii) + i0_(m, A, kk, jj - 2, ii + 1) \
 + i0_(m, A, kk, jj - 1, ii) + i0_(m, A, kk, jj - 1, ii + 1) \
 + i0_(m, A, kk + 1, jj - 2, ii) + i0_(m, A, kk + 1, jj - 2, ii + 1) \
 + i0_(m, A, kk + 1, jj - 1, ii) + i0_(m, A, kk + 1, jj - 1, ii + 1));
                        auto i0_cellavg_py = (1.0 / 8.0) * (i0_(m, A, kk, jj + 2, ii) + i0_(m, A, kk, jj + 2, ii + 1) \
 + i0_(m, A, kk, jj + 3, ii) + i0_(m, A, kk, jj + 3, ii + 1) \
 + i0_(m, A, kk + 1, jj + 2, ii) + i0_(m, A, kk + 1, jj + 2, ii + 1) \
 + i0_(m, A, kk + 1, jj + 3, ii) + i0_(m, A, kk + 1, jj + 3, ii + 1));
                        auto i0_cellavg_mz = (1.0 / 8.0) * (i0_(m, A, kk - 2, jj, ii) + i0_(m, A, kk - 2, jj, ii + 1) \
 + i0_(m, A, kk - 2, jj + 1, ii) + i0_(m, A, kk - 2, jj + 1, ii + 1) \
 + i0_(m, A, kk - 1, jj, ii) + i0_(m, A, kk - 1, jj, ii + 1) \
 + i0_(m, A, kk - 1, jj + 1, ii) + i0_(m, A, kk - 1, jj + 1, ii + 1));
                        auto i0_cellavg_pz = (1.0 / 8.0) * (i0_(m, A, kk + 2, jj, ii) + i0_(m, A, kk + 2, jj, ii + 1) \
 + i0_(m, A, kk + 2, jj + 1, ii) + i0_(m, A, kk + 2, jj + 1, ii + 1) \
 + i0_(m, A, kk + 3, jj, ii) + i0_(m, A, kk + 3, jj, ii + 1) \
 + i0_(m, A, kk + 3, jj + 1, ii) + i0_(m, A, kk + 3, jj + 1, ii + 1));

                        auto dminusx = (i0_cellavg - i0_cellavg_mx) / (2.0 * mbsize.d_view(m).dx1);
                        auto dplusx = (i0_cellavg_px - i0_cellavg) / (2.0 * mbsize.d_view(m).dx1);
                        auto islopex = 2.0 * (i0_(m, A, kk, jj, ii + 1) - i0_(m, A, kk, jj, ii) \
 + i0_(m, A, kk, jj + 1, ii + 1) - i0_(m, A, kk, jj + 1, ii) \
 + i0_(m, A, kk + 1, jj, ii + 1) - i0_(m, A, kk + 1, jj, ii) \
 + i0_(m, A, kk + 1, jj + 1, ii + 1) - i0_(m, A, kk + 1, jj + 1, ii)) /
                                       (2.0 * 2.0 * 2.0 * mbsize.d_view(m).dx1);

                        auto dminusy = (i0_cellavg - i0_cellavg_my) / (2.0 * mbsize.d_view(m).dx2);
                        auto dplusy = (i0_cellavg_py - i0_cellavg) / (2.0 * mbsize.d_view(m).dx2);
                        auto islopey = 2.0 * (i0_(m, A, kk, jj + 1, ii) - i0_(m, A, kk, jj, ii) \
 + i0_(m, A, kk, jj + 1, ii + 1) - i0_(m, A, kk, jj, ii + 1) \
 + i0_(m, A, kk + 1, jj + 1, ii) - i0_(m, A, kk + 1, jj, ii) \
 + i0_(m, A, kk + 1, jj + 1, ii + 1) - i0_(m, A, kk + 1, jj, ii + 1)) /
                                       (2.0 * 2.0 * 2.0 * mbsize.d_view(m).dx2);

                        auto dminusz = (i0_cellavg - i0_cellavg_mz) / (2.0 * mbsize.d_view(m).dx3);
                        auto dplusz = (i0_cellavg_pz - i0_cellavg) / (2.0 * mbsize.d_view(m).dx3);
                        auto islopez = 2.0 * (i0_(m, A, kk + 1, jj, ii) - i0_(m, A, kk, jj, ii) \
 + i0_(m, A, kk + 1, jj, ii + 1) - i0_(m, A, kk, jj, ii + 1) \
 + i0_(m, A, kk + 1, jj + 1, ii) - i0_(m, A, kk, jj + 1, ii) \
 + i0_(m, A, kk + 1, jj + 1, ii + 1) - i0_(m, A, kk, jj, ii + 1)) /
                                       (2.0 * 2.0 * 2.0 * mbsize.d_view(m).dx3);

                        auto sigmax = minmod2(islopex, dminusx, dplusx);
                        auto sigmay = minmod2(islopey, dminusy, dplusy);
                        auto sigmaz = minmod2(islopez, dminusz, dplusz);

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

                        itemp_(m, A, kk, jj, ii) =
                                i0_cellavg + sigmax * (xii - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkk - zmean);
                        itemp_(m, A, kk, jj, ii + 1) =
                                i0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkk - zmean);
                        itemp_(m, A, kk, jj + 1, ii) =
                                i0_cellavg + sigmax * (xii - xmean) + sigmay * (xjjp1 - ymean) + sigmaz * (xkk - zmean);
                        itemp_(m, A, kk, jj + 1, ii + 1) =
                                i0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjjp1 - ymean) +
                                sigmaz * (xkk - zmean);
                        itemp_(m, A, kk + 1, jj, ii) =
                                i0_cellavg + sigmax * (xii - xmean) + sigmay * (xjj - ymean) + sigmaz * (xkkp1 - zmean);
                        itemp_(m, A, kk + 1, jj, ii + 1) =
                                i0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjj - ymean) +
                                sigmaz * (xkkp1 - zmean);
                        itemp_(m, A, kk + 1, jj + 1, ii) =
                                i0_cellavg + sigmax * (xii - xmean) + sigmay * (xjjp1 - ymean) +
                                sigmaz * (xkkp1 - zmean);
                        itemp_(m, A, kk + 1, jj + 1, ii + 1) =
                                i0_cellavg + sigmax * (xiip1 - xmean) + sigmay * (xjjp1 - ymean) +
                                sigmaz * (xkkp1 - zmean);

                    });
        }

        Kokkos::deep_copy(i0_, itemp_);

        return TaskStatus::complete;
    }

} // namespace radiationfemn
