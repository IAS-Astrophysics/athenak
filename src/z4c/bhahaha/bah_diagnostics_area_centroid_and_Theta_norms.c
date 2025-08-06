#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * BHaHAHA apparent horizon diagnostics: compute area, centroid location, and Theta (L2 and Linfinity) norms.
 */
void bah_diagnostics_area_centroid_and_Theta_norms(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  const int grid = 0;
  const params_struct *restrict params = &griddata[grid].params;
  REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  const REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];
#include "set_CodeParameters.h"

  // Set integration weights.
  const REAL *restrict weights;
  int weight_stencil_size;
  bah_diagnostics_integration_weights(Nxx1, Nxx2, &weights, &weight_stencil_size);

  // Compute Linfinity and L2 norms of Theta
  REAL min_radius = 1e10, max_radius = -1e10;
  REAL sum_Theta_squared_for_L2_norm = 0.0;
  REAL sum_curr_area = 0;
  REAL sum_x_centroid = 0, sum_y_centroid = 0, sum_z_centroid = 0;
  REAL max_Theta_squared_for_Linf_norm = -1e30;
#pragma omp parallel
  {
#pragma omp for
    for (int i2 = NGHOSTS; i2 < NGHOSTS + Nxx2; i2++) {
      const REAL weight2 = weights[(i2 - NGHOSTS) % weight_stencil_size];
      const REAL xx2 = xx[2][i2];
      for (int i1 = NGHOSTS; i1 < NGHOSTS + Nxx1; i1++) {
        const REAL weight1 = weights[(i1 - NGHOSTS) % weight_stencil_size];
        const REAL xx1 = xx[1][i1];
        for (int i0 = NGHOSTS; i0 < NGHOSTS + Nxx0; i0++) {
          /*
           * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
           * Read gridfunction(s) from main memory and compute FD stencils as needed.
           */
          const REAL WW = auxevol_gfs[IDX4(WWGF, i0, i1, i2)];
          const REAL aDD00 = auxevol_gfs[IDX4(ADD00GF, i0, i1, i2)];
          const REAL aDD01 = auxevol_gfs[IDX4(ADD01GF, i0, i1, i2)];
          const REAL aDD02 = auxevol_gfs[IDX4(ADD02GF, i0, i1, i2)];
          const REAL aDD11 = auxevol_gfs[IDX4(ADD11GF, i0, i1, i2)];
          const REAL aDD12 = auxevol_gfs[IDX4(ADD12GF, i0, i1, i2)];
          const REAL aDD22 = auxevol_gfs[IDX4(ADD22GF, i0, i1, i2)];
          const REAL hDD00 = auxevol_gfs[IDX4(HDD00GF, i0, i1, i2)];
          const REAL hDD01 = auxevol_gfs[IDX4(HDD01GF, i0, i1, i2)];
          const REAL hDD02 = auxevol_gfs[IDX4(HDD02GF, i0, i1, i2)];
          const REAL hDD11 = auxevol_gfs[IDX4(HDD11GF, i0, i1, i2)];
          const REAL hDD12 = auxevol_gfs[IDX4(HDD12GF, i0, i1, i2)];
          const REAL hDD22 = auxevol_gfs[IDX4(HDD22GF, i0, i1, i2)];
          const REAL hh_i1m3_i2m3 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2 - 3)];
          const REAL hh_i1m2_i2m3 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2 - 3)];
          const REAL hh_i1m1_i2m3 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2 - 3)];
          const REAL hh_i2m3 = in_gfs[IDX4(HHGF, i0, i1, i2 - 3)];
          const REAL hh_i1p1_i2m3 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2 - 3)];
          const REAL hh_i1p2_i2m3 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2 - 3)];
          const REAL hh_i1p3_i2m3 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2 - 3)];
          const REAL hh_i1m3_i2m2 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2 - 2)];
          const REAL hh_i1m2_i2m2 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2 - 2)];
          const REAL hh_i1m1_i2m2 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2 - 2)];
          const REAL hh_i2m2 = in_gfs[IDX4(HHGF, i0, i1, i2 - 2)];
          const REAL hh_i1p1_i2m2 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2 - 2)];
          const REAL hh_i1p2_i2m2 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2 - 2)];
          const REAL hh_i1p3_i2m2 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2 - 2)];
          const REAL hh_i1m3_i2m1 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2 - 1)];
          const REAL hh_i1m2_i2m1 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2 - 1)];
          const REAL hh_i1m1_i2m1 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2 - 1)];
          const REAL hh_i2m1 = in_gfs[IDX4(HHGF, i0, i1, i2 - 1)];
          const REAL hh_i1p1_i2m1 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2 - 1)];
          const REAL hh_i1p2_i2m1 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2 - 1)];
          const REAL hh_i1p3_i2m1 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2 - 1)];
          const REAL hh_i1m3 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2)];
          const REAL hh_i1m2 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2)];
          const REAL hh_i1m1 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2)];
          const REAL hh = in_gfs[IDX4(HHGF, i0, i1, i2)];
          const REAL hh_i1p1 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2)];
          const REAL hh_i1p2 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2)];
          const REAL hh_i1p3 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2)];
          const REAL hh_i1m3_i2p1 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2 + 1)];
          const REAL hh_i1m2_i2p1 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2 + 1)];
          const REAL hh_i1m1_i2p1 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2 + 1)];
          const REAL hh_i2p1 = in_gfs[IDX4(HHGF, i0, i1, i2 + 1)];
          const REAL hh_i1p1_i2p1 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2 + 1)];
          const REAL hh_i1p2_i2p1 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2 + 1)];
          const REAL hh_i1p3_i2p1 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2 + 1)];
          const REAL hh_i1m3_i2p2 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2 + 2)];
          const REAL hh_i1m2_i2p2 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2 + 2)];
          const REAL hh_i1m1_i2p2 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2 + 2)];
          const REAL hh_i2p2 = in_gfs[IDX4(HHGF, i0, i1, i2 + 2)];
          const REAL hh_i1p1_i2p2 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2 + 2)];
          const REAL hh_i1p2_i2p2 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2 + 2)];
          const REAL hh_i1p3_i2p2 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2 + 2)];
          const REAL hh_i1m3_i2p3 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2 + 3)];
          const REAL hh_i1m2_i2p3 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2 + 3)];
          const REAL hh_i1m1_i2p3 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2 + 3)];
          const REAL hh_i2p3 = in_gfs[IDX4(HHGF, i0, i1, i2 + 3)];
          const REAL hh_i1p1_i2p3 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2 + 3)];
          const REAL hh_i1p2_i2p3 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2 + 3)];
          const REAL hh_i1p3_i2p3 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2 + 3)];
          const REAL partial_D_WW0 = auxevol_gfs[IDX4(PARTIAL_D_WW0GF, i0, i1, i2)];
          const REAL partial_D_WW1 = auxevol_gfs[IDX4(PARTIAL_D_WW1GF, i0, i1, i2)];
          const REAL partial_D_WW2 = auxevol_gfs[IDX4(PARTIAL_D_WW2GF, i0, i1, i2)];
          const REAL partial_D_hDD000 = auxevol_gfs[IDX4(PARTIAL_D_HDD000GF, i0, i1, i2)];
          const REAL partial_D_hDD001 = auxevol_gfs[IDX4(PARTIAL_D_HDD001GF, i0, i1, i2)];
          const REAL partial_D_hDD002 = auxevol_gfs[IDX4(PARTIAL_D_HDD002GF, i0, i1, i2)];
          const REAL partial_D_hDD011 = auxevol_gfs[IDX4(PARTIAL_D_HDD011GF, i0, i1, i2)];
          const REAL partial_D_hDD012 = auxevol_gfs[IDX4(PARTIAL_D_HDD012GF, i0, i1, i2)];
          const REAL partial_D_hDD022 = auxevol_gfs[IDX4(PARTIAL_D_HDD022GF, i0, i1, i2)];
          const REAL partial_D_hDD100 = auxevol_gfs[IDX4(PARTIAL_D_HDD100GF, i0, i1, i2)];
          const REAL partial_D_hDD101 = auxevol_gfs[IDX4(PARTIAL_D_HDD101GF, i0, i1, i2)];
          const REAL partial_D_hDD102 = auxevol_gfs[IDX4(PARTIAL_D_HDD102GF, i0, i1, i2)];
          const REAL partial_D_hDD111 = auxevol_gfs[IDX4(PARTIAL_D_HDD111GF, i0, i1, i2)];
          const REAL partial_D_hDD112 = auxevol_gfs[IDX4(PARTIAL_D_HDD112GF, i0, i1, i2)];
          const REAL partial_D_hDD122 = auxevol_gfs[IDX4(PARTIAL_D_HDD122GF, i0, i1, i2)];
          const REAL partial_D_hDD200 = auxevol_gfs[IDX4(PARTIAL_D_HDD200GF, i0, i1, i2)];
          const REAL partial_D_hDD201 = auxevol_gfs[IDX4(PARTIAL_D_HDD201GF, i0, i1, i2)];
          const REAL partial_D_hDD202 = auxevol_gfs[IDX4(PARTIAL_D_HDD202GF, i0, i1, i2)];
          const REAL partial_D_hDD211 = auxevol_gfs[IDX4(PARTIAL_D_HDD211GF, i0, i1, i2)];
          const REAL partial_D_hDD212 = auxevol_gfs[IDX4(PARTIAL_D_HDD212GF, i0, i1, i2)];
          const REAL partial_D_hDD222 = auxevol_gfs[IDX4(PARTIAL_D_HDD222GF, i0, i1, i2)];
          const REAL trK = auxevol_gfs[IDX4(TRKGF, i0, i1, i2)];
          static const REAL FDPart1_Rational_3_4 = 3.0 / 4.0;
          static const REAL FDPart1_Rational_3_20 = 3.0 / 20.0;
          static const REAL FDPart1_Rational_1_60 = 1.0 / 60.0;
          static const REAL FDPart1_Rational_49_18 = 49.0 / 18.0;
          static const REAL FDPart1_Rational_1_90 = 1.0 / 90.0;
          static const REAL FDPart1_Rational_3_2 = 3.0 / 2.0;
          static const REAL FDPart1_Rational_9_16 = 9.0 / 16.0;
          static const REAL FDPart1_Rational_9_80 = 9.0 / 80.0;
          static const REAL FDPart1_Rational_9_400 = 9.0 / 400.0;
          static const REAL FDPart1_Rational_1_80 = 1.0 / 80.0;
          static const REAL FDPart1_Rational_1_400 = 1.0 / 400.0;
          static const REAL FDPart1_Rational_1_3600 = 1.0 / 3600.0;
          const REAL FDPart1tmp1 = -hh_i2p2;
          const REAL FDPart1tmp2 = -FDPart1_Rational_49_18 * hh;
          const REAL hh_dD1 = invdxx1 * (FDPart1_Rational_1_60 * (-hh_i1m3 + hh_i1p3) + FDPart1_Rational_3_20 * (hh_i1m2 - hh_i1p2) +
                                         FDPart1_Rational_3_4 * (-hh_i1m1 + hh_i1p1));
          const REAL hh_dD2 = invdxx2 * (FDPart1_Rational_1_60 * (-hh_i2m3 + hh_i2p3) + FDPart1_Rational_3_20 * (FDPart1tmp1 + hh_i2m2) +
                                         FDPart1_Rational_3_4 * (-hh_i2m1 + hh_i2p1));
          const REAL hh_dDD11 = ((invdxx1) * (invdxx1)) * (FDPart1_Rational_1_90 * (hh_i1m3 + hh_i1p3) + FDPart1_Rational_3_2 * (hh_i1m1 + hh_i1p1) +
                                                           FDPart1_Rational_3_20 * (-hh_i1m2 - hh_i1p2) + FDPart1tmp2);
          const REAL hh_dDD12 =
              invdxx1 * invdxx2 *
              (FDPart1_Rational_1_3600 * (hh_i1m3_i2m3 - hh_i1m3_i2p3 - hh_i1p3_i2m3 + hh_i1p3_i2p3) +
               FDPart1_Rational_1_400 *
                   (-hh_i1m2_i2m3 + hh_i1m2_i2p3 - hh_i1m3_i2m2 + hh_i1m3_i2p2 + hh_i1p2_i2m3 - hh_i1p2_i2p3 + hh_i1p3_i2m2 - hh_i1p3_i2p2) +
               FDPart1_Rational_1_80 *
                   (hh_i1m1_i2m3 - hh_i1m1_i2p3 + hh_i1m3_i2m1 - hh_i1m3_i2p1 - hh_i1p1_i2m3 + hh_i1p1_i2p3 - hh_i1p3_i2m1 + hh_i1p3_i2p1) +
               FDPart1_Rational_9_16 * (hh_i1m1_i2m1 - hh_i1m1_i2p1 - hh_i1p1_i2m1 + hh_i1p1_i2p1) +
               FDPart1_Rational_9_400 * (hh_i1m2_i2m2 - hh_i1m2_i2p2 - hh_i1p2_i2m2 + hh_i1p2_i2p2) +
               FDPart1_Rational_9_80 *
                   (-hh_i1m1_i2m2 + hh_i1m1_i2p2 - hh_i1m2_i2m1 + hh_i1m2_i2p1 + hh_i1p1_i2m2 - hh_i1p1_i2p2 + hh_i1p2_i2m1 - hh_i1p2_i2p1));
          const REAL hh_dDD22 = ((invdxx2) * (invdxx2)) * (FDPart1_Rational_1_90 * (hh_i2m3 + hh_i2p3) + FDPart1_Rational_3_2 * (hh_i2m1 + hh_i2p1) +
                                                           FDPart1_Rational_3_20 * (FDPart1tmp1 - hh_i2m2) + FDPart1tmp2);

          /*
           * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
           * Evaluate SymPy expressions and write to main memory.
           */
          const REAL FDPart3tmp0 = (1.0 / ((WW) * (WW)));
          const REAL FDPart3tmp1 = ((hh) * (hh));
          const REAL FDPart3tmp2 = sin(xx1);
          const REAL FDPart3tmp7 = (1.0 / 3.0) * trK;
          const REAL FDPart3tmp9 = ((hh) * (hh) * (hh) * (hh));
          const REAL FDPart3tmp10 = pow(WW, -6);
          const REAL FDPart3tmp14 = hDD00 + 1;
          const REAL FDPart3tmp28 = (1.0 / ((WW) * (WW) * (WW) * (WW)));
          const REAL FDPart3tmp35 = ((hh) * (hh) * (hh));
          const REAL FDPart3tmp47 = ((hh_dD2) * (hh_dD2));
          const REAL FDPart3tmp57 = ((hh_dD1) * (hh_dD1));
          const REAL FDPart3tmp77 = (1.0 / ((WW) * (WW) * (WW)));
          const REAL FDPart3tmp78 = 2 * hh;
          const REAL FDPart3tmp101 = cos(xx1);
          const REAL FDPart3tmp166 = 2 * hh_dD1;
          const REAL FDPart3tmp173 = 2 * hh_dD2;
          const REAL FDPart3tmp179 = ((hh) * (hh));
          const REAL FDPart3tmp3 = ((FDPart3tmp2) * (FDPart3tmp2));
          const REAL FDPart3tmp15 = FDPart3tmp9 * ((hDD12) * (hDD12));
          const REAL FDPart3tmp17 = FDPart3tmp1 * hDD11 + FDPart3tmp1;
          const REAL FDPart3tmp20 = FDPart3tmp1 * ((hDD01) * (hDD01));
          const REAL FDPart3tmp27 = FDPart3tmp2 * hDD02;
          const REAL FDPart3tmp30 = FDPart3tmp1 * FDPart3tmp2;
          const REAL FDPart3tmp36 = FDPart3tmp2 * hDD12;
          const REAL FDPart3tmp65 = FDPart3tmp0 * FDPart3tmp1;
          const REAL FDPart3tmp69 = FDPart3tmp0 * FDPart3tmp14;
          const REAL FDPart3tmp72 = FDPart3tmp2 * hh;
          const REAL FDPart3tmp74 = FDPart3tmp0 * hh;
          const REAL FDPart3tmp84 = 2 * FDPart3tmp77;
          const REAL FDPart3tmp4 = FDPart3tmp1 * FDPart3tmp3;
          const REAL FDPart3tmp31 = FDPart3tmp30 * hDD12;
          const REAL FDPart3tmp37 = -FDPart3tmp17 * FDPart3tmp27 * FDPart3tmp28 * hh + FDPart3tmp28 * FDPart3tmp35 * FDPart3tmp36 * hDD01;
          const REAL FDPart3tmp71 = FDPart3tmp0 * FDPart3tmp30;
          const REAL FDPart3tmp80 = FDPart3tmp27 * FDPart3tmp77 * FDPart3tmp78;
          const REAL FDPart3tmp85 = FDPart3tmp84 * partial_D_WW2;
          const REAL FDPart3tmp88 = FDPart3tmp77 * FDPart3tmp78 * hDD01;
          const REAL FDPart3tmp104 = FDPart3tmp84 * partial_D_WW1;
          const REAL FDPart3tmp111 = 2 * FDPart3tmp101 * FDPart3tmp30;
          const REAL FDPart3tmp118 = FDPart3tmp84 * partial_D_WW0;
          const REAL FDPart3tmp177 = FDPart3tmp0 * hDD01 * hh;
          const REAL FDPart3tmp178 = FDPart3tmp0 * FDPart3tmp27 * hh;
          const REAL FDPart3tmp6 = FDPart3tmp4 * hDD22 + FDPart3tmp4;
          const REAL FDPart3tmp13 = FDPart3tmp10 * FDPart3tmp3 * FDPart3tmp9 * hDD01 * hDD02 * hDD12;
          const REAL FDPart3tmp16 = FDPart3tmp10 * FDPart3tmp14 * FDPart3tmp15 * FDPart3tmp3;
          const REAL FDPart3tmp18 = FDPart3tmp4 * ((hDD02) * (hDD02));
          const REAL FDPart3tmp32 = FDPart3tmp1 * FDPart3tmp27 * FDPart3tmp28 * hDD01 - FDPart3tmp14 * FDPart3tmp28 * FDPart3tmp31;
          const REAL FDPart3tmp39 = FDPart3tmp14 * FDPart3tmp17 * FDPart3tmp28 - FDPart3tmp20 * FDPart3tmp28;
          const REAL FDPart3tmp81 = FDPart3tmp0 * FDPart3tmp72 * partial_D_hDD202 - FDPart3tmp80 * partial_D_WW2;
          const REAL FDPart3tmp82 = 2 * FDPart3tmp37;
          const REAL FDPart3tmp86 = -FDPart3tmp31 * FDPart3tmp85 + FDPart3tmp71 * partial_D_hDD212;
          const REAL FDPart3tmp89 = FDPart3tmp74 * partial_D_hDD201 - FDPart3tmp88 * partial_D_WW2;
          const REAL FDPart3tmp92 = FDPart3tmp0 * partial_D_hDD200 - FDPart3tmp14 * FDPart3tmp85;
          const REAL FDPart3tmp96 = -FDPart3tmp17 * FDPart3tmp85 + FDPart3tmp65 * partial_D_hDD211;
          const REAL FDPart3tmp99 = FDPart3tmp74 * partial_D_hDD101 - FDPart3tmp88 * partial_D_WW1;
          const REAL FDPart3tmp102 = FDPart3tmp0 * (FDPart3tmp101 * hDD02 * hh + FDPart3tmp72 * partial_D_hDD102) - FDPart3tmp80 * partial_D_WW1;
          const REAL FDPart3tmp105 =
              FDPart3tmp0 * (FDPart3tmp1 * FDPart3tmp101 * hDD12 + FDPart3tmp30 * partial_D_hDD112) - FDPart3tmp104 * FDPart3tmp31;
          const REAL FDPart3tmp107 = FDPart3tmp0 * partial_D_hDD100 - FDPart3tmp104 * FDPart3tmp14;
          const REAL FDPart3tmp109 = -FDPart3tmp104 * FDPart3tmp17 + FDPart3tmp65 * partial_D_hDD111;
          const REAL FDPart3tmp114 = FDPart3tmp0 * (FDPart3tmp27 + FDPart3tmp72 * partial_D_hDD002) - FDPart3tmp80 * partial_D_WW0;
          const REAL FDPart3tmp116 = FDPart3tmp0 * (hDD01 + hh * partial_D_hDD001) - FDPart3tmp88 * partial_D_WW0;
          const REAL FDPart3tmp119 = FDPart3tmp0 * (FDPart3tmp30 * partial_D_hDD012 + FDPart3tmp36 * FDPart3tmp78) - FDPart3tmp118 * FDPart3tmp31;
          const REAL FDPart3tmp121 = FDPart3tmp0 * partial_D_hDD000 - FDPart3tmp118 * FDPart3tmp14;
          const REAL FDPart3tmp123 =
              FDPart3tmp0 * (FDPart3tmp1 * partial_D_hDD011 + FDPart3tmp78 * hDD11 + FDPart3tmp78) - FDPart3tmp118 * FDPart3tmp17;
          const REAL FDPart3tmp19 = FDPart3tmp10 * FDPart3tmp17 * FDPart3tmp18;
          const REAL FDPart3tmp43 = 2 * FDPart3tmp32;
          const REAL FDPart3tmp52 = FDPart3tmp28 * FDPart3tmp6;
          const REAL FDPart3tmp83 = FDPart3tmp81 * FDPart3tmp82;
          const REAL FDPart3tmp94 = FDPart3tmp0 * FDPart3tmp4 * partial_D_hDD222 - FDPart3tmp6 * FDPart3tmp85;
          const REAL FDPart3tmp103 = FDPart3tmp102 * FDPart3tmp82;
          const REAL FDPart3tmp112 =
              FDPart3tmp0 * (FDPart3tmp111 * hDD22 + FDPart3tmp111 + FDPart3tmp4 * partial_D_hDD122) - FDPart3tmp104 * FDPart3tmp6;
          const REAL FDPart3tmp115 = FDPart3tmp114 * FDPart3tmp82;
          const REAL FDPart3tmp126 =
              FDPart3tmp0 * (FDPart3tmp3 * FDPart3tmp78 * hDD22 + FDPart3tmp3 * FDPart3tmp78 + FDPart3tmp4 * partial_D_hDD022) -
              FDPart3tmp118 * FDPart3tmp6;
          const REAL FDPart3tmp22 = FDPart3tmp10 * FDPart3tmp20 * FDPart3tmp6;
          const REAL FDPart3tmp24 = FDPart3tmp10 * FDPart3tmp14 * FDPart3tmp17 * FDPart3tmp6;
          const REAL FDPart3tmp53 = FDPart3tmp28 * FDPart3tmp3 * FDPart3tmp35 * hDD02 * hDD12 - FDPart3tmp52 * hDD01 * hh;
          const REAL FDPart3tmp58 = FDPart3tmp14 * FDPart3tmp52 - FDPart3tmp18 * FDPart3tmp28;
          const REAL FDPart3tmp61 = -FDPart3tmp15 * FDPart3tmp28 * FDPart3tmp3 + FDPart3tmp17 * FDPart3tmp52;
          const REAL FDPart3tmp87 = FDPart3tmp43 * FDPart3tmp86;
          const REAL FDPart3tmp95 = FDPart3tmp39 * FDPart3tmp94;
          const REAL FDPart3tmp106 = FDPart3tmp105 * FDPart3tmp43;
          const REAL FDPart3tmp113 = FDPart3tmp112 * FDPart3tmp39;
          const REAL FDPart3tmp120 = FDPart3tmp119 * FDPart3tmp43;
          const REAL FDPart3tmp127 = FDPart3tmp126 * FDPart3tmp39;
          const REAL FDPart3tmp25 = 2 * FDPart3tmp13 - FDPart3tmp16 - FDPart3tmp19 - FDPart3tmp22 + FDPart3tmp24;
          const REAL FDPart3tmp90 = 2 * FDPart3tmp53;
          const REAL FDPart3tmp93 = FDPart3tmp61 * FDPart3tmp92;
          const REAL FDPart3tmp97 = FDPart3tmp58 * FDPart3tmp96;
          const REAL FDPart3tmp108 = FDPart3tmp107 * FDPart3tmp61;
          const REAL FDPart3tmp110 = FDPart3tmp109 * FDPart3tmp58;
          const REAL FDPart3tmp122 = FDPart3tmp121 * FDPart3tmp61;
          const REAL FDPart3tmp124 = FDPart3tmp123 * FDPart3tmp58;
          const REAL FDPart3tmp26 = (1.0 / (FDPart3tmp25));
          const REAL FDPart3tmp91 = FDPart3tmp89 * FDPart3tmp90;
          const REAL FDPart3tmp100 = FDPart3tmp90 * FDPart3tmp99;
          const REAL FDPart3tmp117 = FDPart3tmp116 * FDPart3tmp90;
          const REAL FDPart3tmp128 = (1.0 / ((FDPart3tmp25) * (FDPart3tmp25)));
          const REAL FDPart3tmp38 = FDPart3tmp26 * FDPart3tmp37;
          const REAL FDPart3tmp40 = FDPart3tmp26 * FDPart3tmp39;
          const REAL FDPart3tmp44 = FDPart3tmp26 * FDPart3tmp43;
          const REAL FDPart3tmp54 = FDPart3tmp26 * FDPart3tmp53;
          const REAL FDPart3tmp59 = FDPart3tmp26 * FDPart3tmp58;
          const REAL FDPart3tmp62 = FDPart3tmp26 * FDPart3tmp61;
          const REAL FDPart3tmp129 = FDPart3tmp128 * ((FDPart3tmp53) * (FDPart3tmp53));
          const REAL FDPart3tmp130 = FDPart3tmp128 * ((FDPart3tmp58) * (FDPart3tmp58));
          const REAL FDPart3tmp131 = FDPart3tmp128 * ((FDPart3tmp32) * (FDPart3tmp32));
          const REAL FDPart3tmp132 = FDPart3tmp128 * FDPart3tmp32;
          const REAL FDPart3tmp134 = FDPart3tmp128 * FDPart3tmp58;
          const REAL FDPart3tmp136 = FDPart3tmp128 * FDPart3tmp53;
          const REAL FDPart3tmp139 = FDPart3tmp128 * FDPart3tmp61;
          const REAL FDPart3tmp141 = FDPart3tmp128 * FDPart3tmp37;
          const REAL FDPart3tmp149 = FDPart3tmp128 * ((FDPart3tmp37) * (FDPart3tmp37));
          const REAL FDPart3tmp150 = FDPart3tmp128 * ((FDPart3tmp39) * (FDPart3tmp39));
          const REAL FDPart3tmp152 = FDPart3tmp128 * FDPart3tmp39;
          const REAL FDPart3tmp163 = FDPart3tmp128 * ((FDPart3tmp61) * (FDPart3tmp61));
          const REAL FDPart3tmp34 = FDPart3tmp26 * FDPart3tmp32 * hh_dD1;
          const REAL FDPart3tmp41 = FDPart3tmp40 * hh_dD2;
          const REAL FDPart3tmp46 = FDPart3tmp44 * hh_dD1 * hh_dD2;
          const REAL FDPart3tmp49 = FDPart3tmp38 * hh_dD2;
          const REAL FDPart3tmp55 = FDPart3tmp54 * hh_dD1;
          const REAL FDPart3tmp66 = FDPart3tmp26 * FDPart3tmp32 * hh_dD2;
          const REAL FDPart3tmp67 = FDPart3tmp59 * hh_dD1;
          const REAL FDPart3tmp133 = FDPart3tmp132 * FDPart3tmp90;
          const REAL FDPart3tmp137 = FDPart3tmp136 * FDPart3tmp37;
          const REAL FDPart3tmp138 = FDPart3tmp136 * FDPart3tmp32;
          const REAL FDPart3tmp142 = FDPart3tmp141 * FDPart3tmp58;
          const REAL FDPart3tmp144 = FDPart3tmp141 * FDPart3tmp32;
          const REAL FDPart3tmp146 = FDPart3tmp136 * FDPart3tmp39;
          const REAL FDPart3tmp151 = FDPart3tmp132 * FDPart3tmp82;
          const REAL FDPart3tmp157 = FDPart3tmp44 * hh_dDD12;
          const REAL FDPart3tmp158 = FDPart3tmp139 * FDPart3tmp32;
          const REAL FDPart3tmp164 = FDPart3tmp141 * FDPart3tmp90;
          const REAL FDPart3tmp42 = -FDPart3tmp34 + FDPart3tmp38 - FDPart3tmp41;
          const REAL FDPart3tmp68 = FDPart3tmp54 - FDPart3tmp66 - FDPart3tmp67;
          const REAL FDPart3tmp70 = -FDPart3tmp49 - FDPart3tmp55 + FDPart3tmp62;
          const REAL FDPart3tmp135 = -FDPart3tmp100 * FDPart3tmp134 - FDPart3tmp102 * FDPart3tmp133 - FDPart3tmp106 * FDPart3tmp134 -
                                     FDPart3tmp107 * FDPart3tmp129 - FDPart3tmp109 * FDPart3tmp130 - FDPart3tmp112 * FDPart3tmp131;
          const REAL FDPart3tmp145 =
              hh_dD1 * (-FDPart3tmp114 * FDPart3tmp137 - FDPart3tmp114 * FDPart3tmp139 * FDPart3tmp32 - FDPart3tmp116 * FDPart3tmp129 -
                        FDPart3tmp116 * FDPart3tmp139 * FDPart3tmp58 - FDPart3tmp119 * FDPart3tmp138 - FDPart3tmp119 * FDPart3tmp142 -
                        FDPart3tmp122 * FDPart3tmp136 - FDPart3tmp124 * FDPart3tmp136 - FDPart3tmp126 * FDPart3tmp144);
          const REAL FDPart3tmp148 =
              hh_dD1 * (-FDPart3tmp131 * FDPart3tmp86 - FDPart3tmp132 * FDPart3tmp95 - FDPart3tmp132 * FDPart3tmp97 -
                        FDPart3tmp134 * FDPart3tmp39 * FDPart3tmp86 - FDPart3tmp137 * FDPart3tmp92 - FDPart3tmp138 * FDPart3tmp89 -
                        FDPart3tmp142 * FDPart3tmp89 - FDPart3tmp144 * FDPart3tmp81 - FDPart3tmp146 * FDPart3tmp81);
          const REAL FDPart3tmp153 = -FDPart3tmp131 * FDPart3tmp96 - FDPart3tmp149 * FDPart3tmp92 - FDPart3tmp150 * FDPart3tmp94 -
                                     FDPart3tmp151 * FDPart3tmp89 - FDPart3tmp152 * FDPart3tmp83 - FDPart3tmp152 * FDPart3tmp87;
          const REAL FDPart3tmp155 =
              hh_dD2 * (-FDPart3tmp102 * FDPart3tmp144 - FDPart3tmp102 * FDPart3tmp146 - FDPart3tmp105 * FDPart3tmp131 -
                        FDPart3tmp105 * FDPart3tmp134 * FDPart3tmp39 - FDPart3tmp107 * FDPart3tmp137 - FDPart3tmp110 * FDPart3tmp132 -
                        FDPart3tmp113 * FDPart3tmp132 - FDPart3tmp138 * FDPart3tmp99 - FDPart3tmp142 * FDPart3tmp99);
          const REAL FDPart3tmp156 =
              hh_dD2 * (-FDPart3tmp114 * FDPart3tmp139 * FDPart3tmp39 - FDPart3tmp114 * FDPart3tmp149 - FDPart3tmp116 * FDPart3tmp137 -
                        FDPart3tmp116 * FDPart3tmp139 * FDPart3tmp32 - FDPart3tmp119 * FDPart3tmp144 - FDPart3tmp119 * FDPart3tmp146 -
                        FDPart3tmp122 * FDPart3tmp141 - FDPart3tmp123 * FDPart3tmp138 - FDPart3tmp127 * FDPart3tmp141);
          const REAL FDPart3tmp160 = -FDPart3tmp102 * FDPart3tmp137 - FDPart3tmp102 * FDPart3tmp158 - FDPart3tmp105 * FDPart3tmp138 -
                                     FDPart3tmp105 * FDPart3tmp142 - FDPart3tmp108 * FDPart3tmp136 - FDPart3tmp110 * FDPart3tmp136 -
                                     FDPart3tmp112 * FDPart3tmp144 - FDPart3tmp129 * FDPart3tmp99 - FDPart3tmp139 * FDPart3tmp58 * FDPart3tmp99;
          const REAL FDPart3tmp162 = -FDPart3tmp136 * FDPart3tmp39 * FDPart3tmp86 - FDPart3tmp137 * FDPart3tmp89 - FDPart3tmp138 * FDPart3tmp96 -
                                     FDPart3tmp139 * FDPart3tmp39 * FDPart3tmp81 - FDPart3tmp141 * FDPart3tmp93 - FDPart3tmp141 * FDPart3tmp95 -
                                     FDPart3tmp144 * FDPart3tmp86 - FDPart3tmp149 * FDPart3tmp81 - FDPart3tmp158 * FDPart3tmp89;
          const REAL FDPart3tmp165 = -FDPart3tmp115 * FDPart3tmp139 - FDPart3tmp117 * FDPart3tmp139 - FDPart3tmp119 * FDPart3tmp164 -
                                     FDPart3tmp121 * FDPart3tmp163 - FDPart3tmp123 * FDPart3tmp129 - FDPart3tmp126 * FDPart3tmp149;
          const REAL FDPart3tmp63 =
              FDPart3tmp40 * FDPart3tmp47 + FDPart3tmp46 - 2 * FDPart3tmp49 - 2 * FDPart3tmp55 + FDPart3tmp57 * FDPart3tmp59 + FDPart3tmp62;
          const REAL FDPart3tmp64 = (1.0 / (FDPart3tmp63));
          const REAL FDPart3tmp98 = (1.0 / sqrt(FDPart3tmp63));
          const REAL FDPart3tmp76 = 2 * FDPart3tmp64 * FDPart3tmp70;
          const REAL FDPart3tmp167 = (1.0 / 2.0) * FDPart3tmp98;
          const REAL FDPart3tmp168 =
              FDPart3tmp167 * (-2 * FDPart3tmp145 - 2 * FDPart3tmp156 + FDPart3tmp165 +
                               FDPart3tmp166 * hh_dD2 *
                                   (-FDPart3tmp114 * FDPart3tmp144 - FDPart3tmp114 * FDPart3tmp146 - FDPart3tmp116 * FDPart3tmp138 -
                                    FDPart3tmp116 * FDPart3tmp142 - FDPart3tmp119 * FDPart3tmp131 - FDPart3tmp119 * FDPart3tmp134 * FDPart3tmp39 -
                                    FDPart3tmp121 * FDPart3tmp137 - FDPart3tmp124 * FDPart3tmp132 - FDPart3tmp127 * FDPart3tmp132) +
                               FDPart3tmp47 * (-FDPart3tmp115 * FDPart3tmp152 - FDPart3tmp116 * FDPart3tmp151 - FDPart3tmp120 * FDPart3tmp152 -
                                               FDPart3tmp121 * FDPart3tmp149 - FDPart3tmp123 * FDPart3tmp131 - FDPart3tmp126 * FDPart3tmp150) +
                               FDPart3tmp57 * (-FDPart3tmp114 * FDPart3tmp133 - FDPart3tmp117 * FDPart3tmp134 - FDPart3tmp120 * FDPart3tmp134 -
                                               FDPart3tmp121 * FDPart3tmp129 - FDPart3tmp123 * FDPart3tmp130 - FDPart3tmp126 * FDPart3tmp131));
          const REAL FDPart3tmp174 =
              FDPart3tmp167 *
              (-FDPart3tmp129 * FDPart3tmp96 - FDPart3tmp139 * FDPart3tmp83 - FDPart3tmp139 * FDPart3tmp91 + FDPart3tmp148 * FDPart3tmp173 -
               FDPart3tmp149 * FDPart3tmp94 + FDPart3tmp153 * FDPart3tmp47 + FDPart3tmp157 * hh_dD2 - FDPart3tmp162 * FDPart3tmp173 -
               FDPart3tmp163 * FDPart3tmp92 - FDPart3tmp164 * FDPart3tmp86 -
               FDPart3tmp166 * (-FDPart3tmp129 * FDPart3tmp89 - FDPart3tmp136 * FDPart3tmp93 - FDPart3tmp136 * FDPart3tmp97 -
                                FDPart3tmp137 * FDPart3tmp81 - FDPart3tmp138 * FDPart3tmp86 - FDPart3tmp139 * FDPart3tmp58 * FDPart3tmp89 -
                                FDPart3tmp142 * FDPart3tmp86 - FDPart3tmp144 * FDPart3tmp94 - FDPart3tmp158 * FDPart3tmp81) -
               2 * FDPart3tmp38 * hh_dDD22 + 2 * FDPart3tmp41 * hh_dDD22 + FDPart3tmp44 * hh_dD1 * hh_dDD22 - 2 * FDPart3tmp54 * hh_dDD12 +
               FDPart3tmp57 * (-FDPart3tmp129 * FDPart3tmp92 - FDPart3tmp130 * FDPart3tmp96 - FDPart3tmp131 * FDPart3tmp94 -
                               FDPart3tmp133 * FDPart3tmp81 - FDPart3tmp134 * FDPart3tmp87 - FDPart3tmp134 * FDPart3tmp91) +
               2 * FDPart3tmp67 * hh_dDD12);
          const REAL FDPart3tmp175 =
              FDPart3tmp167 *
              (-FDPart3tmp100 * FDPart3tmp139 - FDPart3tmp103 * FDPart3tmp139 - FDPart3tmp105 * FDPart3tmp164 - FDPart3tmp107 * FDPart3tmp163 -
               FDPart3tmp109 * FDPart3tmp129 - FDPart3tmp112 * FDPart3tmp149 + FDPart3tmp135 * FDPart3tmp57 + FDPart3tmp155 * FDPart3tmp166 +
               FDPart3tmp157 * hh_dD1 - FDPart3tmp160 * FDPart3tmp166 -
               FDPart3tmp173 * (-FDPart3tmp102 * FDPart3tmp139 * FDPart3tmp39 - FDPart3tmp102 * FDPart3tmp149 - FDPart3tmp105 * FDPart3tmp144 -
                                FDPart3tmp105 * FDPart3tmp146 - FDPart3tmp108 * FDPart3tmp141 - FDPart3tmp109 * FDPart3tmp138 -
                                FDPart3tmp113 * FDPart3tmp141 - FDPart3tmp137 * FDPart3tmp99 - FDPart3tmp158 * FDPart3tmp99) -
               2 * FDPart3tmp38 * hh_dDD12 + 2 * FDPart3tmp41 * hh_dDD12 + FDPart3tmp44 * hh_dD2 * hh_dDD11 +
               FDPart3tmp47 * (-FDPart3tmp103 * FDPart3tmp152 - FDPart3tmp106 * FDPart3tmp152 - FDPart3tmp107 * FDPart3tmp149 -
                               FDPart3tmp109 * FDPart3tmp131 - FDPart3tmp112 * FDPart3tmp150 - FDPart3tmp151 * FDPart3tmp99) -
               2 * FDPart3tmp54 * hh_dDD11 + 2 * FDPart3tmp67 * hh_dDD11);
          const REAL Theta =
              ((FDPart3tmp42) * (FDPart3tmp42)) * FDPart3tmp64 * (FDPart3tmp0 * FDPart3tmp4 * aDD22 + FDPart3tmp0 * FDPart3tmp6 * FDPart3tmp7) +
              2 * FDPart3tmp42 * FDPart3tmp64 * FDPart3tmp68 * (FDPart3tmp7 * FDPart3tmp71 * hDD12 + FDPart3tmp71 * aDD12) +
              FDPart3tmp42 * FDPart3tmp76 * (FDPart3tmp0 * FDPart3tmp72 * aDD02 + FDPart3tmp27 * FDPart3tmp7 * FDPart3tmp74) +
              FDPart3tmp64 * ((FDPart3tmp68) * (FDPart3tmp68)) * (FDPart3tmp0 * FDPart3tmp17 * FDPart3tmp7 + FDPart3tmp65 * aDD11) +
              FDPart3tmp64 * ((FDPart3tmp70) * (FDPart3tmp70)) * (FDPart3tmp0 * aDD00 + FDPart3tmp69 * FDPart3tmp7) +
              FDPart3tmp68 * FDPart3tmp76 * (FDPart3tmp7 * FDPart3tmp74 * hDD01 + FDPart3tmp74 * aDD01) +
              FDPart3tmp98 * (-FDPart3tmp135 * hh_dD1 - FDPart3tmp145 - FDPart3tmp148 - FDPart3tmp153 * hh_dD2 - FDPart3tmp155 - FDPart3tmp156 -
                              FDPart3tmp157 + FDPart3tmp160 + FDPart3tmp162 + FDPart3tmp165 - FDPart3tmp40 * hh_dDD22 - FDPart3tmp59 * hh_dDD11) -
              trK +
              (FDPart3tmp42 * FDPart3tmp98 * (FDPart3tmp83 + FDPart3tmp87 + FDPart3tmp91 + FDPart3tmp93 + FDPart3tmp95 + FDPart3tmp97) +
               FDPart3tmp68 * FDPart3tmp98 * (FDPart3tmp100 + FDPart3tmp103 + FDPart3tmp106 + FDPart3tmp108 + FDPart3tmp110 + FDPart3tmp113) +
               FDPart3tmp70 * FDPart3tmp98 * (FDPart3tmp115 + FDPart3tmp117 + FDPart3tmp120 + FDPart3tmp122 + FDPart3tmp124 + FDPart3tmp127)) /
                  (4 * FDPart3tmp13 - 2 * FDPart3tmp16 - 2 * FDPart3tmp19 - 2 * FDPart3tmp22 + 2 * FDPart3tmp24) +
              (-FDPart3tmp168 * FDPart3tmp49 - FDPart3tmp168 * FDPart3tmp55 + FDPart3tmp168 * FDPart3tmp62 - FDPart3tmp174 * FDPart3tmp34 +
               FDPart3tmp174 * FDPart3tmp38 - FDPart3tmp174 * FDPart3tmp41 + FDPart3tmp175 * FDPart3tmp54 - FDPart3tmp175 * FDPart3tmp66 -
               FDPart3tmp175 * FDPart3tmp67) /
                  (-FDPart3tmp40 * FDPart3tmp47 - FDPart3tmp46 + 2 * FDPart3tmp49 + 2 * FDPart3tmp55 - FDPart3tmp57 * FDPart3tmp59 - FDPart3tmp62);
          const REAL area_element =
              sqrt((FDPart3tmp0 * (FDPart3tmp179 * hDD11 + FDPart3tmp179) + FDPart3tmp166 * FDPart3tmp177 + FDPart3tmp57 * FDPart3tmp69) *
                       (FDPart3tmp0 * (FDPart3tmp179 * FDPart3tmp3 * hDD22 + FDPart3tmp179 * FDPart3tmp3) + FDPart3tmp173 * FDPart3tmp178 +
                        FDPart3tmp47 * FDPart3tmp69) -
                   ((FDPart3tmp0 * FDPart3tmp179 * FDPart3tmp36 + FDPart3tmp177 * hh_dD2 + FDPart3tmp178 * hh_dD1 + FDPart3tmp69 * hh_dD1 * hh_dD2) *
                    (FDPart3tmp0 * FDPart3tmp179 * FDPart3tmp36 + FDPart3tmp177 * hh_dD2 + FDPart3tmp178 * hh_dD1 + FDPart3tmp69 * hh_dD1 * hh_dD2)));

#pragma omp critical
          {
            sum_curr_area += area_element * weight1 * weight2;
            sum_Theta_squared_for_L2_norm += Theta * Theta * area_element * weight1 * weight2;
            {
              const REAL tmp0 = hh * sin(xx1);
              sum_x_centroid += (Cart_originx + tmp0 * cos(xx2)) * area_element * weight1 * weight2;
              sum_y_centroid += (Cart_originy + tmp0 * sin(xx2)) * area_element * weight1 * weight2;
              sum_z_centroid += (Cart_originz + hh * cos(xx1)) * area_element * weight1 * weight2;
            } // END centroid sums
            if (Theta * Theta > max_Theta_squared_for_Linf_norm)
              max_Theta_squared_for_Linf_norm = Theta * Theta;
            if (hh > max_radius)
              max_radius = hh;
            if (hh < min_radius)
              min_radius = hh;
          } // END OMP CRITICAL
        } // END LOOP over i0
      } // END LOOP over i1
    } // END LOOP over i2
  } // END OMP PARALLEL

  // Store diagnostics in commondata->bhahaha_diagnostics struct.
  {

    // {min,max}_radius_wrt_grid_center are strictly internal diagnostics; these will
    //   exhibit a kink the second time a horizon is found, as the initial guess for
    //   the horizon centroid might be a little bit off from the guess.
    // Users should view the radii wrt the AH centroid location, which will be smooth
    //   over time.
    commondata->min_radius_wrt_grid_center = min_radius;
    commondata->max_radius_wrt_grid_center = max_radius;

    bhahaha_diagnostics_struct *restrict bhahaha_diags = commondata->bhahaha_diagnostics;

    // Theta was just evaluated at Nxx1*Nxx2 gridpoints; update counter:
    bhahaha_diags->Theta_eval_points_counter += params->Nxx1 * params->Nxx2;

    // Update area, compute irreducible mass.
    bhahaha_diags->area = sum_curr_area * params->dxx1 * params->dxx2;

    // Compute area-weighted norms of Theta, normalized properly by M_scale, as [Theta] ~ 1/length.
    const REAL M_scale = commondata->bhahaha_params_and_data->M_scale;
    bhahaha_diags->Theta_Linf_times_M = M_scale * sqrt(max_Theta_squared_for_Linf_norm);
    bhahaha_diags->Theta_L2_times_M = M_scale * sqrt(sum_Theta_squared_for_L2_norm * params->dxx1 * params->dxx2 / bhahaha_diags->area);

    bhahaha_diags->x_centroid_wrt_coord_origin = sum_x_centroid * params->dxx1 * params->dxx2 / bhahaha_diags->area;
    bhahaha_diags->y_centroid_wrt_coord_origin = sum_y_centroid * params->dxx1 * params->dxx2 / bhahaha_diags->area;
    bhahaha_diags->z_centroid_wrt_coord_origin = sum_z_centroid * params->dxx1 * params->dxx2 / bhahaha_diags->area;
  } // END store diagnostics in commondata->bhahaha_diagnostics struct.
} // END FUNCTION bah_diagnostics_area_centroid_and_Theta_norms
