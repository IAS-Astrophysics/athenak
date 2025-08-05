#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Evaluate RHSs
 */
void bah_rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
                  const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
#include "set_CodeParameters.h"
#pragma omp parallel for
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
      MAYBE_UNUSED const REAL f1_of_xx1 = rfmstruct->f1_of_xx1[i1];
      MAYBE_UNUSED const REAL f1_of_xx1__D1 = rfmstruct->f1_of_xx1__D1[i1];
      MAYBE_UNUSED const REAL f1_of_xx1__DD11 = rfmstruct->f1_of_xx1__DD11[i1];

      for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
        MAYBE_UNUSED const REAL f0_of_xx0 = rfmstruct->f0_of_xx0[i0];

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
        const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
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
        const REAL FDPart3tmp1 = ((f1_of_xx1) * (f1_of_xx1));
        const REAL FDPart3tmp2 = ((hh) * (hh));
        const REAL FDPart3tmp6 = (1.0 / 3.0) * trK;
        const REAL FDPart3tmp8 = ((hh) * (hh) * (hh) * (hh));
        const REAL FDPart3tmp9 = pow(WW, -6);
        const REAL FDPart3tmp13 = hDD00 + 1;
        const REAL FDPart3tmp26 = f1_of_xx1 * hDD02;
        const REAL FDPart3tmp27 = (1.0 / ((WW) * (WW) * (WW) * (WW)));
        const REAL FDPart3tmp34 = ((hh) * (hh) * (hh));
        const REAL FDPart3tmp48 = ((hh_dD2) * (hh_dD2));
        const REAL FDPart3tmp56 = ((hh_dD1) * (hh_dD1));
        const REAL FDPart3tmp70 = f1_of_xx1 * hh;
        const REAL FDPart3tmp75 = (1.0 / ((WW) * (WW) * (WW)));
        const REAL FDPart3tmp76 = 2 * hh;
        const REAL FDPart3tmp163 = 2 * hh_dD1;
        const REAL FDPart3tmp170 = 2 * hh_dD2;
        const REAL FDPart3tmp3 = FDPart3tmp1 * FDPart3tmp2;
        const REAL FDPart3tmp7 = FDPart3tmp0 * FDPart3tmp6;
        const REAL FDPart3tmp14 = FDPart3tmp8 * ((hDD12) * (hDD12));
        const REAL FDPart3tmp16 = FDPart3tmp2 * hDD11 + FDPart3tmp2;
        const REAL FDPart3tmp19 = FDPart3tmp2 * ((hDD01) * (hDD01));
        const REAL FDPart3tmp29 = FDPart3tmp2 * f1_of_xx1;
        const REAL FDPart3tmp64 = FDPart3tmp0 * FDPart3tmp2;
        const REAL FDPart3tmp72 = FDPart3tmp0 * hh;
        const REAL FDPart3tmp82 = 2 * FDPart3tmp75;
        const REAL FDPart3tmp5 = FDPart3tmp3 * hDD22 + FDPart3tmp3;
        const REAL FDPart3tmp12 = FDPart3tmp1 * FDPart3tmp8 * FDPart3tmp9 * hDD01 * hDD02 * hDD12;
        const REAL FDPart3tmp15 = FDPart3tmp1 * FDPart3tmp13 * FDPart3tmp14 * FDPart3tmp9;
        const REAL FDPart3tmp17 = FDPart3tmp3 * ((hDD02) * (hDD02));
        const REAL FDPart3tmp30 = FDPart3tmp29 * hDD12;
        const REAL FDPart3tmp36 = -FDPart3tmp16 * FDPart3tmp26 * FDPart3tmp27 * hh + FDPart3tmp27 * FDPart3tmp34 * f1_of_xx1 * hDD01 * hDD12;
        const REAL FDPart3tmp69 = FDPart3tmp0 * FDPart3tmp29;
        const REAL FDPart3tmp78 = FDPart3tmp26 * FDPart3tmp75 * FDPart3tmp76;
        const REAL FDPart3tmp83 = FDPart3tmp82 * partial_D_WW2;
        const REAL FDPart3tmp86 = FDPart3tmp75 * FDPart3tmp76 * hDD01;
        const REAL FDPart3tmp101 = FDPart3tmp82 * partial_D_WW1;
        const REAL FDPart3tmp108 = 2 * FDPart3tmp29 * f1_of_xx1__D1;
        const REAL FDPart3tmp113 = FDPart3tmp82 * partial_D_WW0;
        const REAL FDPart3tmp18 = FDPart3tmp16 * FDPart3tmp17 * FDPart3tmp9;
        const REAL FDPart3tmp31 = -FDPart3tmp13 * FDPart3tmp27 * FDPart3tmp30 + FDPart3tmp2 * FDPart3tmp26 * FDPart3tmp27 * hDD01;
        const REAL FDPart3tmp38 = FDPart3tmp13 * FDPart3tmp16 * FDPart3tmp27 - FDPart3tmp19 * FDPart3tmp27;
        const REAL FDPart3tmp51 = FDPart3tmp27 * FDPart3tmp5;
        const REAL FDPart3tmp79 = FDPart3tmp0 * FDPart3tmp70 * partial_D_hDD202 - FDPart3tmp78 * partial_D_WW2;
        const REAL FDPart3tmp80 = 2 * FDPart3tmp36;
        const REAL FDPart3tmp84 = -FDPart3tmp30 * FDPart3tmp83 + FDPart3tmp69 * partial_D_hDD212;
        const REAL FDPart3tmp87 = FDPart3tmp72 * partial_D_hDD201 - FDPart3tmp86 * partial_D_WW2;
        const REAL FDPart3tmp90 = FDPart3tmp0 * partial_D_hDD200 - FDPart3tmp13 * FDPart3tmp83;
        const REAL FDPart3tmp92 = FDPart3tmp0 * FDPart3tmp3 * partial_D_hDD222 - FDPart3tmp5 * FDPart3tmp83;
        const REAL FDPart3tmp94 = -FDPart3tmp16 * FDPart3tmp83 + FDPart3tmp64 * partial_D_hDD211;
        const REAL FDPart3tmp97 = FDPart3tmp0 * (FDPart3tmp70 * partial_D_hDD102 + f1_of_xx1__D1 * hDD02 * hh) - FDPart3tmp78 * partial_D_WW1;
        const REAL FDPart3tmp99 = FDPart3tmp72 * partial_D_hDD101 - FDPart3tmp86 * partial_D_WW1;
        const REAL FDPart3tmp102 =
            FDPart3tmp0 * (FDPart3tmp2 * f1_of_xx1__D1 * hDD12 + FDPart3tmp29 * partial_D_hDD112) - FDPart3tmp101 * FDPart3tmp30;
        const REAL FDPart3tmp104 = FDPart3tmp0 * partial_D_hDD100 - FDPart3tmp101 * FDPart3tmp13;
        const REAL FDPart3tmp106 = -FDPart3tmp101 * FDPart3tmp16 + FDPart3tmp64 * partial_D_hDD111;
        const REAL FDPart3tmp109 =
            FDPart3tmp0 * (FDPart3tmp108 * hDD22 + FDPart3tmp108 + FDPart3tmp3 * partial_D_hDD122) - FDPart3tmp101 * FDPart3tmp5;
        const REAL FDPart3tmp111 = FDPart3tmp0 * (FDPart3tmp26 + FDPart3tmp70 * partial_D_hDD002) - FDPart3tmp78 * partial_D_WW0;
        const REAL FDPart3tmp114 = FDPart3tmp0 * (FDPart3tmp29 * partial_D_hDD012 + FDPart3tmp76 * f1_of_xx1 * hDD12) - FDPart3tmp113 * FDPart3tmp30;
        const REAL FDPart3tmp116 = FDPart3tmp0 * (hDD01 + hh * partial_D_hDD001) - FDPart3tmp86 * partial_D_WW0;
        const REAL FDPart3tmp118 = FDPart3tmp0 * partial_D_hDD000 - FDPart3tmp113 * FDPart3tmp13;
        const REAL FDPart3tmp120 =
            FDPart3tmp0 * (FDPart3tmp2 * partial_D_hDD011 + FDPart3tmp76 * hDD11 + FDPart3tmp76) - FDPart3tmp113 * FDPart3tmp16;
        const REAL FDPart3tmp123 = FDPart3tmp0 * (FDPart3tmp1 * FDPart3tmp76 * hDD22 + FDPart3tmp1 * FDPart3tmp76 + FDPart3tmp3 * partial_D_hDD022) -
                                   FDPart3tmp113 * FDPart3tmp5;
        const REAL FDPart3tmp21 = FDPart3tmp19 * FDPart3tmp5 * FDPart3tmp9;
        const REAL FDPart3tmp23 = FDPart3tmp13 * FDPart3tmp16 * FDPart3tmp5 * FDPart3tmp9;
        const REAL FDPart3tmp42 = 2 * FDPart3tmp31;
        const REAL FDPart3tmp52 = FDPart3tmp1 * FDPart3tmp27 * FDPart3tmp34 * hDD02 * hDD12 - FDPart3tmp51 * hDD01 * hh;
        const REAL FDPart3tmp57 = FDPart3tmp13 * FDPart3tmp51 - FDPart3tmp17 * FDPart3tmp27;
        const REAL FDPart3tmp60 = -FDPart3tmp1 * FDPart3tmp14 * FDPart3tmp27 + FDPart3tmp16 * FDPart3tmp51;
        const REAL FDPart3tmp81 = FDPart3tmp79 * FDPart3tmp80;
        const REAL FDPart3tmp93 = FDPart3tmp38 * FDPart3tmp92;
        const REAL FDPart3tmp98 = FDPart3tmp80 * FDPart3tmp97;
        const REAL FDPart3tmp110 = FDPart3tmp109 * FDPart3tmp38;
        const REAL FDPart3tmp112 = FDPart3tmp111 * FDPart3tmp80;
        const REAL FDPart3tmp124 = FDPart3tmp123 * FDPart3tmp38;
        const REAL FDPart3tmp24 = 2 * FDPart3tmp12 - FDPart3tmp15 - FDPart3tmp18 - FDPart3tmp21 + FDPart3tmp23;
        const REAL FDPart3tmp85 = FDPart3tmp42 * FDPart3tmp84;
        const REAL FDPart3tmp88 = 2 * FDPart3tmp52;
        const REAL FDPart3tmp91 = FDPart3tmp60 * FDPart3tmp90;
        const REAL FDPart3tmp95 = FDPart3tmp57 * FDPart3tmp94;
        const REAL FDPart3tmp103 = FDPart3tmp102 * FDPart3tmp42;
        const REAL FDPart3tmp105 = FDPart3tmp104 * FDPart3tmp60;
        const REAL FDPart3tmp107 = FDPart3tmp106 * FDPart3tmp57;
        const REAL FDPart3tmp115 = FDPart3tmp114 * FDPart3tmp42;
        const REAL FDPart3tmp119 = FDPart3tmp118 * FDPart3tmp60;
        const REAL FDPart3tmp121 = FDPart3tmp120 * FDPart3tmp57;
        const REAL FDPart3tmp25 = (1.0 / (FDPart3tmp24));
        const REAL FDPart3tmp89 = FDPart3tmp87 * FDPart3tmp88;
        const REAL FDPart3tmp100 = FDPart3tmp88 * FDPart3tmp99;
        const REAL FDPart3tmp117 = FDPart3tmp116 * FDPart3tmp88;
        const REAL FDPart3tmp125 = (1.0 / ((FDPart3tmp24) * (FDPart3tmp24)));
        const REAL FDPart3tmp37 = FDPart3tmp25 * FDPart3tmp36;
        const REAL FDPart3tmp39 = FDPart3tmp25 * FDPart3tmp38;
        const REAL FDPart3tmp43 = FDPart3tmp25 * FDPart3tmp42;
        const REAL FDPart3tmp53 = FDPart3tmp25 * FDPart3tmp52;
        const REAL FDPart3tmp58 = FDPart3tmp25 * FDPart3tmp57;
        const REAL FDPart3tmp61 = FDPart3tmp25 * FDPart3tmp60;
        const REAL FDPart3tmp126 = FDPart3tmp125 * ((FDPart3tmp52) * (FDPart3tmp52));
        const REAL FDPart3tmp127 = FDPart3tmp125 * ((FDPart3tmp57) * (FDPart3tmp57));
        const REAL FDPart3tmp128 = FDPart3tmp125 * ((FDPart3tmp31) * (FDPart3tmp31));
        const REAL FDPart3tmp129 = FDPart3tmp125 * FDPart3tmp31;
        const REAL FDPart3tmp131 = FDPart3tmp125 * FDPart3tmp57;
        const REAL FDPart3tmp133 = FDPart3tmp125 * FDPart3tmp52;
        const REAL FDPart3tmp136 = FDPart3tmp125 * FDPart3tmp60;
        const REAL FDPart3tmp138 = FDPart3tmp125 * FDPart3tmp36;
        const REAL FDPart3tmp146 = FDPart3tmp125 * ((FDPart3tmp36) * (FDPart3tmp36));
        const REAL FDPart3tmp147 = FDPart3tmp125 * ((FDPart3tmp38) * (FDPart3tmp38));
        const REAL FDPart3tmp149 = FDPart3tmp125 * FDPart3tmp38;
        const REAL FDPart3tmp160 = FDPart3tmp125 * ((FDPart3tmp60) * (FDPart3tmp60));
        const REAL FDPart3tmp33 = FDPart3tmp25 * FDPart3tmp31 * hh_dD1;
        const REAL FDPart3tmp40 = FDPart3tmp39 * hh_dD2;
        const REAL FDPart3tmp46 = FDPart3tmp37 * hh_dD2;
        const REAL FDPart3tmp54 = FDPart3tmp53 * hh_dD1;
        const REAL FDPart3tmp65 = FDPart3tmp25 * FDPart3tmp31 * hh_dD2;
        const REAL FDPart3tmp66 = FDPart3tmp58 * hh_dD1;
        const REAL FDPart3tmp130 = FDPart3tmp129 * FDPart3tmp88;
        const REAL FDPart3tmp134 = FDPart3tmp133 * FDPart3tmp36;
        const REAL FDPart3tmp135 = FDPart3tmp133 * FDPart3tmp31;
        const REAL FDPart3tmp139 = FDPart3tmp138 * FDPart3tmp57;
        const REAL FDPart3tmp141 = FDPart3tmp138 * FDPart3tmp31;
        const REAL FDPart3tmp143 = FDPart3tmp133 * FDPart3tmp38;
        const REAL FDPart3tmp148 = FDPart3tmp129 * FDPart3tmp80;
        const REAL FDPart3tmp154 = FDPart3tmp43 * hh_dDD12;
        const REAL FDPart3tmp155 = FDPart3tmp136 * FDPart3tmp31;
        const REAL FDPart3tmp161 = FDPart3tmp138 * FDPart3tmp88;
        const REAL FDPart3tmp41 = -FDPart3tmp33 + FDPart3tmp37 - FDPart3tmp40;
        const REAL FDPart3tmp45 = FDPart3tmp43 * hh_dD1 * hh_dD2;
        const REAL FDPart3tmp67 = FDPart3tmp53 - FDPart3tmp65 - FDPart3tmp66;
        const REAL FDPart3tmp68 = -FDPart3tmp46 - FDPart3tmp54 + FDPart3tmp61;
        const REAL FDPart3tmp132 = -FDPart3tmp100 * FDPart3tmp131 - FDPart3tmp103 * FDPart3tmp131 - FDPart3tmp104 * FDPart3tmp126 -
                                   FDPart3tmp106 * FDPart3tmp127 - FDPart3tmp109 * FDPart3tmp128 - FDPart3tmp130 * FDPart3tmp97;
        const REAL FDPart3tmp142 =
            hh_dD1 * (-FDPart3tmp111 * FDPart3tmp134 - FDPart3tmp111 * FDPart3tmp136 * FDPart3tmp31 - FDPart3tmp114 * FDPart3tmp135 -
                      FDPart3tmp114 * FDPart3tmp139 - FDPart3tmp116 * FDPart3tmp126 - FDPart3tmp116 * FDPart3tmp136 * FDPart3tmp57 -
                      FDPart3tmp119 * FDPart3tmp133 - FDPart3tmp121 * FDPart3tmp133 - FDPart3tmp123 * FDPart3tmp141);
        const REAL FDPart3tmp145 =
            hh_dD1 * (-FDPart3tmp128 * FDPart3tmp84 - FDPart3tmp129 * FDPart3tmp93 - FDPart3tmp129 * FDPart3tmp95 -
                      FDPart3tmp131 * FDPart3tmp38 * FDPart3tmp84 - FDPart3tmp134 * FDPart3tmp90 - FDPart3tmp135 * FDPart3tmp87 -
                      FDPart3tmp139 * FDPart3tmp87 - FDPart3tmp141 * FDPart3tmp79 - FDPart3tmp143 * FDPart3tmp79);
        const REAL FDPart3tmp150 = -FDPart3tmp128 * FDPart3tmp94 - FDPart3tmp146 * FDPart3tmp90 - FDPart3tmp147 * FDPart3tmp92 -
                                   FDPart3tmp148 * FDPart3tmp87 - FDPart3tmp149 * FDPart3tmp81 - FDPart3tmp149 * FDPart3tmp85;
        const REAL FDPart3tmp152 =
            hh_dD2 * (-FDPart3tmp102 * FDPart3tmp128 - FDPart3tmp102 * FDPart3tmp131 * FDPart3tmp38 - FDPart3tmp104 * FDPart3tmp134 -
                      FDPart3tmp107 * FDPart3tmp129 - FDPart3tmp110 * FDPart3tmp129 - FDPart3tmp135 * FDPart3tmp99 - FDPart3tmp139 * FDPart3tmp99 -
                      FDPart3tmp141 * FDPart3tmp97 - FDPart3tmp143 * FDPart3tmp97);
        const REAL FDPart3tmp153 =
            hh_dD2 * (-FDPart3tmp111 * FDPart3tmp136 * FDPart3tmp38 - FDPart3tmp111 * FDPart3tmp146 - FDPart3tmp114 * FDPart3tmp141 -
                      FDPart3tmp114 * FDPart3tmp143 - FDPart3tmp116 * FDPart3tmp134 - FDPart3tmp116 * FDPart3tmp136 * FDPart3tmp31 -
                      FDPart3tmp119 * FDPart3tmp138 - FDPart3tmp120 * FDPart3tmp135 - FDPart3tmp124 * FDPart3tmp138);
        const REAL FDPart3tmp157 = -FDPart3tmp102 * FDPart3tmp135 - FDPart3tmp102 * FDPart3tmp139 - FDPart3tmp105 * FDPart3tmp133 -
                                   FDPart3tmp107 * FDPart3tmp133 - FDPart3tmp109 * FDPart3tmp141 - FDPart3tmp126 * FDPart3tmp99 -
                                   FDPart3tmp134 * FDPart3tmp97 - FDPart3tmp136 * FDPart3tmp57 * FDPart3tmp99 - FDPart3tmp155 * FDPart3tmp97;
        const REAL FDPart3tmp159 = -FDPart3tmp133 * FDPart3tmp38 * FDPart3tmp84 - FDPart3tmp134 * FDPart3tmp87 - FDPart3tmp135 * FDPart3tmp94 -
                                   FDPart3tmp136 * FDPart3tmp38 * FDPart3tmp79 - FDPart3tmp138 * FDPart3tmp91 - FDPart3tmp138 * FDPart3tmp93 -
                                   FDPart3tmp141 * FDPart3tmp84 - FDPart3tmp146 * FDPart3tmp79 - FDPart3tmp155 * FDPart3tmp87;
        const REAL FDPart3tmp162 = -FDPart3tmp112 * FDPart3tmp136 - FDPart3tmp114 * FDPart3tmp161 - FDPart3tmp117 * FDPart3tmp136 -
                                   FDPart3tmp118 * FDPart3tmp160 - FDPart3tmp120 * FDPart3tmp126 - FDPart3tmp123 * FDPart3tmp146;
        const REAL FDPart3tmp62 =
            FDPart3tmp39 * FDPart3tmp48 + FDPart3tmp45 - 2 * FDPart3tmp46 - 2 * FDPart3tmp54 + FDPart3tmp56 * FDPart3tmp58 + FDPart3tmp61;
        const REAL FDPart3tmp63 = (1.0 / (FDPart3tmp62));
        const REAL FDPart3tmp96 = (1.0 / sqrt(FDPart3tmp62));
        const REAL FDPart3tmp74 = 2 * FDPart3tmp63 * FDPart3tmp68;
        const REAL FDPart3tmp164 = (1.0 / 2.0) * FDPart3tmp96;
        const REAL FDPart3tmp165 =
            FDPart3tmp164 * (-2 * FDPart3tmp142 - 2 * FDPart3tmp153 + FDPart3tmp162 +
                             FDPart3tmp163 * hh_dD2 *
                                 (-FDPart3tmp111 * FDPart3tmp141 - FDPart3tmp111 * FDPart3tmp143 - FDPart3tmp114 * FDPart3tmp128 -
                                  FDPart3tmp114 * FDPart3tmp131 * FDPart3tmp38 - FDPart3tmp116 * FDPart3tmp135 - FDPart3tmp116 * FDPart3tmp139 -
                                  FDPart3tmp118 * FDPart3tmp134 - FDPart3tmp121 * FDPart3tmp129 - FDPart3tmp124 * FDPart3tmp129) +
                             FDPart3tmp48 * (-FDPart3tmp112 * FDPart3tmp149 - FDPart3tmp115 * FDPart3tmp149 - FDPart3tmp116 * FDPart3tmp148 -
                                             FDPart3tmp118 * FDPart3tmp146 - FDPart3tmp120 * FDPart3tmp128 - FDPart3tmp123 * FDPart3tmp147) +
                             FDPart3tmp56 * (-FDPart3tmp111 * FDPart3tmp130 - FDPart3tmp115 * FDPart3tmp131 - FDPart3tmp117 * FDPart3tmp131 -
                                             FDPart3tmp118 * FDPart3tmp126 - FDPart3tmp120 * FDPart3tmp127 - FDPart3tmp123 * FDPart3tmp128));
        const REAL FDPart3tmp171 =
            FDPart3tmp164 *
            (-FDPart3tmp126 * FDPart3tmp94 - FDPart3tmp136 * FDPart3tmp81 - FDPart3tmp136 * FDPart3tmp89 + FDPart3tmp145 * FDPart3tmp170 -
             FDPart3tmp146 * FDPart3tmp92 + FDPart3tmp150 * FDPart3tmp48 + FDPart3tmp154 * hh_dD2 - FDPart3tmp159 * FDPart3tmp170 -
             FDPart3tmp160 * FDPart3tmp90 - FDPart3tmp161 * FDPart3tmp84 -
             FDPart3tmp163 * (-FDPart3tmp126 * FDPart3tmp87 - FDPart3tmp133 * FDPart3tmp91 - FDPart3tmp133 * FDPart3tmp95 -
                              FDPart3tmp134 * FDPart3tmp79 - FDPart3tmp135 * FDPart3tmp84 - FDPart3tmp136 * FDPart3tmp57 * FDPart3tmp87 -
                              FDPart3tmp139 * FDPart3tmp84 - FDPart3tmp141 * FDPart3tmp92 - FDPart3tmp155 * FDPart3tmp79) -
             2 * FDPart3tmp37 * hh_dDD22 + 2 * FDPart3tmp40 * hh_dDD22 + FDPart3tmp43 * hh_dD1 * hh_dDD22 - 2 * FDPart3tmp53 * hh_dDD12 +
             FDPart3tmp56 * (-FDPart3tmp126 * FDPart3tmp90 - FDPart3tmp127 * FDPart3tmp94 - FDPart3tmp128 * FDPart3tmp92 -
                             FDPart3tmp130 * FDPart3tmp79 - FDPart3tmp131 * FDPart3tmp85 - FDPart3tmp131 * FDPart3tmp89) +
             2 * FDPart3tmp66 * hh_dDD12);
        const REAL FDPart3tmp172 =
            FDPart3tmp164 *
            (-FDPart3tmp100 * FDPart3tmp136 - FDPart3tmp102 * FDPart3tmp161 - FDPart3tmp104 * FDPart3tmp160 - FDPart3tmp106 * FDPart3tmp126 -
             FDPart3tmp109 * FDPart3tmp146 + FDPart3tmp132 * FDPart3tmp56 - FDPart3tmp136 * FDPart3tmp98 + FDPart3tmp152 * FDPart3tmp163 +
             FDPart3tmp154 * hh_dD1 - FDPart3tmp157 * FDPart3tmp163 -
             FDPart3tmp170 * (-FDPart3tmp102 * FDPart3tmp141 - FDPart3tmp102 * FDPart3tmp143 - FDPart3tmp105 * FDPart3tmp138 -
                              FDPart3tmp106 * FDPart3tmp135 - FDPart3tmp110 * FDPart3tmp138 - FDPart3tmp134 * FDPart3tmp99 -
                              FDPart3tmp136 * FDPart3tmp38 * FDPart3tmp97 - FDPart3tmp146 * FDPart3tmp97 - FDPart3tmp155 * FDPart3tmp99) -
             2 * FDPart3tmp37 * hh_dDD12 + 2 * FDPart3tmp40 * hh_dDD12 + FDPart3tmp43 * hh_dD2 * hh_dDD11 +
             FDPart3tmp48 * (-FDPart3tmp103 * FDPart3tmp149 - FDPart3tmp104 * FDPart3tmp146 - FDPart3tmp106 * FDPart3tmp128 -
                             FDPart3tmp109 * FDPart3tmp147 - FDPart3tmp148 * FDPart3tmp99 - FDPart3tmp149 * FDPart3tmp98) -
             2 * FDPart3tmp53 * hh_dDD11 + 2 * FDPart3tmp66 * hh_dDD11);
        rhs_gfs[IDX4(HHGF, i0, i1, i2)] = -eta_damping * hh + vv;
        rhs_gfs[IDX4(VVGF, i0, i1, i2)] =
            -((FDPart3tmp41) * (FDPart3tmp41)) * FDPart3tmp63 * (FDPart3tmp0 * FDPart3tmp3 * aDD22 + FDPart3tmp5 * FDPart3tmp7) -
            2 * FDPart3tmp41 * FDPart3tmp63 * FDPart3tmp67 * (FDPart3tmp6 * FDPart3tmp69 * hDD12 + FDPart3tmp69 * aDD12) -
            FDPart3tmp41 * FDPart3tmp74 * (FDPart3tmp0 * FDPart3tmp70 * aDD02 + FDPart3tmp26 * FDPart3tmp6 * FDPart3tmp72) -
            FDPart3tmp63 * ((FDPart3tmp67) * (FDPart3tmp67)) * (FDPart3tmp16 * FDPart3tmp7 + FDPart3tmp64 * aDD11) -
            FDPart3tmp63 * ((FDPart3tmp68) * (FDPart3tmp68)) * (FDPart3tmp0 * aDD00 + FDPart3tmp13 * FDPart3tmp7) -
            FDPart3tmp67 * FDPart3tmp74 * (FDPart3tmp6 * FDPart3tmp72 * hDD01 + FDPart3tmp72 * aDD01) -
            FDPart3tmp96 * (-FDPart3tmp132 * hh_dD1 - FDPart3tmp142 - FDPart3tmp145 - FDPart3tmp150 * hh_dD2 - FDPart3tmp152 - FDPart3tmp153 -
                            FDPart3tmp154 + FDPart3tmp157 + FDPart3tmp159 + FDPart3tmp162 - FDPart3tmp39 * hh_dDD22 - FDPart3tmp58 * hh_dDD11) +
            trK -
            (FDPart3tmp41 * FDPart3tmp96 * (FDPart3tmp81 + FDPart3tmp85 + FDPart3tmp89 + FDPart3tmp91 + FDPart3tmp93 + FDPart3tmp95) +
             FDPart3tmp67 * FDPart3tmp96 * (FDPart3tmp100 + FDPart3tmp103 + FDPart3tmp105 + FDPart3tmp107 + FDPart3tmp110 + FDPart3tmp98) +
             FDPart3tmp68 * FDPart3tmp96 * (FDPart3tmp112 + FDPart3tmp115 + FDPart3tmp117 + FDPart3tmp119 + FDPart3tmp121 + FDPart3tmp124)) /
                (4 * FDPart3tmp12 - 2 * FDPart3tmp15 - 2 * FDPart3tmp18 - 2 * FDPart3tmp21 + 2 * FDPart3tmp23) -
            (-FDPart3tmp165 * FDPart3tmp46 - FDPart3tmp165 * FDPart3tmp54 + FDPart3tmp165 * FDPart3tmp61 - FDPart3tmp171 * FDPart3tmp33 +
             FDPart3tmp171 * FDPart3tmp37 - FDPart3tmp171 * FDPart3tmp40 + FDPart3tmp172 * FDPart3tmp53 - FDPart3tmp172 * FDPart3tmp65 -
             FDPart3tmp172 * FDPart3tmp66) /
                (-FDPart3tmp39 * FDPart3tmp48 - FDPart3tmp45 + 2 * FDPart3tmp46 + 2 * FDPart3tmp54 - FDPart3tmp56 * FDPart3tmp58 - FDPart3tmp61);

      } // END LOOP: for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
} // END FUNCTION bah_rhs_eval
