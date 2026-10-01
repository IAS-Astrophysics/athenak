#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Apply KO
 */
void bah_KO_apply(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
                  const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
#include "set_CodeParameters.h"
  if (commondata->KO_diss_strength == 0.0)
    return;
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
        const REAL hh_i2m3 = in_gfs[IDX4(HHGF, i0, i1, i2 - 3)];
        const REAL hh_i2m2 = in_gfs[IDX4(HHGF, i0, i1, i2 - 2)];
        const REAL hh_i2m1 = in_gfs[IDX4(HHGF, i0, i1, i2 - 1)];
        const REAL hh_i1m3 = in_gfs[IDX4(HHGF, i0, i1 - 3, i2)];
        const REAL hh_i1m2 = in_gfs[IDX4(HHGF, i0, i1 - 2, i2)];
        const REAL hh_i1m1 = in_gfs[IDX4(HHGF, i0, i1 - 1, i2)];
        const REAL hh = in_gfs[IDX4(HHGF, i0, i1, i2)];
        const REAL hh_i1p1 = in_gfs[IDX4(HHGF, i0, i1 + 1, i2)];
        const REAL hh_i1p2 = in_gfs[IDX4(HHGF, i0, i1 + 2, i2)];
        const REAL hh_i1p3 = in_gfs[IDX4(HHGF, i0, i1 + 3, i2)];
        const REAL hh_i2p1 = in_gfs[IDX4(HHGF, i0, i1, i2 + 1)];
        const REAL hh_i2p2 = in_gfs[IDX4(HHGF, i0, i1, i2 + 2)];
        const REAL hh_i2p3 = in_gfs[IDX4(HHGF, i0, i1, i2 + 3)];
        const REAL vv_i2m3 = in_gfs[IDX4(VVGF, i0, i1, i2 - 3)];
        const REAL vv_i2m2 = in_gfs[IDX4(VVGF, i0, i1, i2 - 2)];
        const REAL vv_i2m1 = in_gfs[IDX4(VVGF, i0, i1, i2 - 1)];
        const REAL vv_i1m3 = in_gfs[IDX4(VVGF, i0, i1 - 3, i2)];
        const REAL vv_i1m2 = in_gfs[IDX4(VVGF, i0, i1 - 2, i2)];
        const REAL vv_i1m1 = in_gfs[IDX4(VVGF, i0, i1 - 1, i2)];
        const REAL vv = in_gfs[IDX4(VVGF, i0, i1, i2)];
        const REAL vv_i1p1 = in_gfs[IDX4(VVGF, i0, i1 + 1, i2)];
        const REAL vv_i1p2 = in_gfs[IDX4(VVGF, i0, i1 + 2, i2)];
        const REAL vv_i1p3 = in_gfs[IDX4(VVGF, i0, i1 + 3, i2)];
        const REAL vv_i2p1 = in_gfs[IDX4(VVGF, i0, i1, i2 + 1)];
        const REAL vv_i2p2 = in_gfs[IDX4(VVGF, i0, i1, i2 + 2)];
        const REAL vv_i2p3 = in_gfs[IDX4(VVGF, i0, i1, i2 + 3)];
        static const REAL FDPart1_Rational_5_16 = 5.0 / 16.0;
        static const REAL FDPart1_Rational_3_32 = 3.0 / 32.0;
        static const REAL FDPart1_Rational_1_64 = 1.0 / 64.0;
        static const REAL FDPart1_Rational_15_64 = 15.0 / 64.0;
        const REAL FDPart1tmp0 = -FDPart1_Rational_5_16 * hh;
        const REAL FDPart1tmp1 = -FDPart1_Rational_5_16 * vv;
        const REAL hh_dKOD1 = invdxx1 * (FDPart1_Rational_15_64 * (hh_i1m1 + hh_i1p1) + FDPart1_Rational_1_64 * (hh_i1m3 + hh_i1p3) +
                                         FDPart1_Rational_3_32 * (-hh_i1m2 - hh_i1p2) + FDPart1tmp0);
        const REAL hh_dKOD2 = invdxx2 * (FDPart1_Rational_15_64 * (hh_i2m1 + hh_i2p1) + FDPart1_Rational_1_64 * (hh_i2m3 + hh_i2p3) +
                                         FDPart1_Rational_3_32 * (-hh_i2m2 - hh_i2p2) + FDPart1tmp0);
        const REAL vv_dKOD1 = invdxx1 * (FDPart1_Rational_15_64 * (vv_i1m1 + vv_i1p1) + FDPart1_Rational_1_64 * (vv_i1m3 + vv_i1p3) +
                                         FDPart1_Rational_3_32 * (-vv_i1m2 - vv_i1p2) + FDPart1tmp1);
        const REAL vv_dKOD2 = invdxx2 * (FDPart1_Rational_15_64 * (vv_i2m1 + vv_i2p1) + FDPart1_Rational_1_64 * (vv_i2m3 + vv_i2p3) +
                                         FDPart1_Rational_3_32 * (-vv_i2m2 - vv_i2p2) + FDPart1tmp1);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL FDPart3tmp0 = KO_diss_strength / hh;
        const REAL FDPart3tmp1 = FDPart3tmp0 / f1_of_xx1;
        rhs_gfs[IDX4(HHGF, i0, i1, i2)] = FDPart3tmp0 * hh_dKOD1 + FDPart3tmp1 * hh_dKOD2 + rhs_gfs[IDX4(HHGF, i0, i1, i2)];
        rhs_gfs[IDX4(VVGF, i0, i1, i2)] = FDPart3tmp0 * vv_dKOD1 + FDPart3tmp1 * vv_dKOD2 + rhs_gfs[IDX4(VVGF, i0, i1, i2)];

      } // END LOOP: for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
} // END FUNCTION bah_KO_apply
