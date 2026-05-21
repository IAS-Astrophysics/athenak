#include "BHaH_defines.h"
/**
 * Compute h_{ij,k}.
 */
void bah_hDD_dD_and_W_dD_in_interp_src_grid_interior(commondata_struct *restrict commondata) {

  int i0_min_shift = 0;
  if (commondata->bhahaha_params_and_data->r_min_external_input == 0)
    i0_min_shift = NGHOSTS;

  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;

  const REAL invdxx0 = commondata->interp_src_invdxx0;
  const REAL invdxx1 = commondata->interp_src_invdxx1;
  const REAL invdxx2 = commondata->interp_src_invdxx2;

  // PART 1 OF 2: Compute angular derivatives h_{ij,k} and W_{,k} (k = 1, 2) at all active radial points.
#pragma omp parallel for
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
      for (int i0 = i0_min_shift; i0 < Nxx_plus_2NGHOSTS0; i0++) {
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL WW_i2m3 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1, i2 - 3)];
        const REAL WW_i2m2 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1, i2 - 2)];
        const REAL WW_i2m1 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1, i2 - 1)];
        const REAL WW_i1m3 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1 - 3, i2)];
        const REAL WW_i1m2 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1 - 2, i2)];
        const REAL WW_i1m1 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1 - 1, i2)];
        const REAL WW_i1p1 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1 + 1, i2)];
        const REAL WW_i1p2 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1 + 2, i2)];
        const REAL WW_i1p3 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1 + 3, i2)];
        const REAL WW_i2p1 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1, i2 + 1)];
        const REAL WW_i2p2 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1, i2 + 2)];
        const REAL WW_i2p3 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0, i1, i2 + 3)];
        const REAL hDD00_i2m3 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1, i2 - 3)];
        const REAL hDD00_i2m2 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1, i2 - 2)];
        const REAL hDD00_i2m1 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1, i2 - 1)];
        const REAL hDD00_i1m3 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1 - 3, i2)];
        const REAL hDD00_i1m2 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1 - 2, i2)];
        const REAL hDD00_i1m1 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1 - 1, i2)];
        const REAL hDD00_i1p1 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1 + 1, i2)];
        const REAL hDD00_i1p2 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1 + 2, i2)];
        const REAL hDD00_i1p3 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1 + 3, i2)];
        const REAL hDD00_i2p1 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1, i2 + 1)];
        const REAL hDD00_i2p2 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1, i2 + 2)];
        const REAL hDD00_i2p3 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0, i1, i2 + 3)];
        const REAL hDD01_i2m3 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1, i2 - 3)];
        const REAL hDD01_i2m2 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1, i2 - 2)];
        const REAL hDD01_i2m1 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1, i2 - 1)];
        const REAL hDD01_i1m3 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1 - 3, i2)];
        const REAL hDD01_i1m2 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1 - 2, i2)];
        const REAL hDD01_i1m1 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1 - 1, i2)];
        const REAL hDD01_i1p1 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1 + 1, i2)];
        const REAL hDD01_i1p2 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1 + 2, i2)];
        const REAL hDD01_i1p3 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1 + 3, i2)];
        const REAL hDD01_i2p1 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1, i2 + 1)];
        const REAL hDD01_i2p2 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1, i2 + 2)];
        const REAL hDD01_i2p3 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0, i1, i2 + 3)];
        const REAL hDD02_i2m3 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1, i2 - 3)];
        const REAL hDD02_i2m2 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1, i2 - 2)];
        const REAL hDD02_i2m1 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1, i2 - 1)];
        const REAL hDD02_i1m3 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1 - 3, i2)];
        const REAL hDD02_i1m2 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1 - 2, i2)];
        const REAL hDD02_i1m1 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1 - 1, i2)];
        const REAL hDD02_i1p1 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1 + 1, i2)];
        const REAL hDD02_i1p2 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1 + 2, i2)];
        const REAL hDD02_i1p3 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1 + 3, i2)];
        const REAL hDD02_i2p1 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1, i2 + 1)];
        const REAL hDD02_i2p2 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1, i2 + 2)];
        const REAL hDD02_i2p3 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0, i1, i2 + 3)];
        const REAL hDD11_i2m3 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2 - 3)];
        const REAL hDD11_i2m2 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2 - 2)];
        const REAL hDD11_i2m1 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2 - 1)];
        const REAL hDD11_i1m3 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1 - 3, i2)];
        const REAL hDD11_i1m2 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1 - 2, i2)];
        const REAL hDD11_i1m1 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1 - 1, i2)];
        const REAL hDD11_i1p1 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1 + 1, i2)];
        const REAL hDD11_i1p2 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1 + 2, i2)];
        const REAL hDD11_i1p3 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1 + 3, i2)];
        const REAL hDD11_i2p1 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2 + 1)];
        const REAL hDD11_i2p2 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2 + 2)];
        const REAL hDD11_i2p3 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2 + 3)];
        const REAL hDD12_i2m3 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1, i2 - 3)];
        const REAL hDD12_i2m2 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1, i2 - 2)];
        const REAL hDD12_i2m1 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1, i2 - 1)];
        const REAL hDD12_i1m3 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1 - 3, i2)];
        const REAL hDD12_i1m2 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1 - 2, i2)];
        const REAL hDD12_i1m1 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1 - 1, i2)];
        const REAL hDD12_i1p1 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1 + 1, i2)];
        const REAL hDD12_i1p2 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1 + 2, i2)];
        const REAL hDD12_i1p3 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1 + 3, i2)];
        const REAL hDD12_i2p1 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1, i2 + 1)];
        const REAL hDD12_i2p2 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1, i2 + 2)];
        const REAL hDD12_i2p3 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0, i1, i2 + 3)];
        const REAL hDD22_i2m3 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1, i2 - 3)];
        const REAL hDD22_i2m2 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1, i2 - 2)];
        const REAL hDD22_i2m1 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1, i2 - 1)];
        const REAL hDD22_i1m3 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1 - 3, i2)];
        const REAL hDD22_i1m2 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1 - 2, i2)];
        const REAL hDD22_i1m1 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1 - 1, i2)];
        const REAL hDD22_i1p1 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1 + 1, i2)];
        const REAL hDD22_i1p2 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1 + 2, i2)];
        const REAL hDD22_i1p3 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1 + 3, i2)];
        const REAL hDD22_i2p1 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1, i2 + 1)];
        const REAL hDD22_i2p2 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1, i2 + 2)];
        const REAL hDD22_i2p3 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0, i1, i2 + 3)];
        static const REAL FDPart1_Rational_3_4 = 3.0 / 4.0;
        static const REAL FDPart1_Rational_3_20 = 3.0 / 20.0;
        static const REAL FDPart1_Rational_1_60 = 1.0 / 60.0;
        const REAL WW_dD1 = invdxx1 * (FDPart1_Rational_1_60 * (-WW_i1m3 + WW_i1p3) + FDPart1_Rational_3_20 * (WW_i1m2 - WW_i1p2) +
                                       FDPart1_Rational_3_4 * (-WW_i1m1 + WW_i1p1));
        const REAL WW_dD2 = invdxx2 * (FDPart1_Rational_1_60 * (-WW_i2m3 + WW_i2p3) + FDPart1_Rational_3_20 * (WW_i2m2 - WW_i2p2) +
                                       FDPart1_Rational_3_4 * (-WW_i2m1 + WW_i2p1));
        const REAL hDD_dD001 = invdxx1 * (FDPart1_Rational_1_60 * (-hDD00_i1m3 + hDD00_i1p3) + FDPart1_Rational_3_20 * (hDD00_i1m2 - hDD00_i1p2) +
                                          FDPart1_Rational_3_4 * (-hDD00_i1m1 + hDD00_i1p1));
        const REAL hDD_dD002 = invdxx2 * (FDPart1_Rational_1_60 * (-hDD00_i2m3 + hDD00_i2p3) + FDPart1_Rational_3_20 * (hDD00_i2m2 - hDD00_i2p2) +
                                          FDPart1_Rational_3_4 * (-hDD00_i2m1 + hDD00_i2p1));
        const REAL hDD_dD011 = invdxx1 * (FDPart1_Rational_1_60 * (-hDD01_i1m3 + hDD01_i1p3) + FDPart1_Rational_3_20 * (hDD01_i1m2 - hDD01_i1p2) +
                                          FDPart1_Rational_3_4 * (-hDD01_i1m1 + hDD01_i1p1));
        const REAL hDD_dD012 = invdxx2 * (FDPart1_Rational_1_60 * (-hDD01_i2m3 + hDD01_i2p3) + FDPart1_Rational_3_20 * (hDD01_i2m2 - hDD01_i2p2) +
                                          FDPart1_Rational_3_4 * (-hDD01_i2m1 + hDD01_i2p1));
        const REAL hDD_dD021 = invdxx1 * (FDPart1_Rational_1_60 * (-hDD02_i1m3 + hDD02_i1p3) + FDPart1_Rational_3_20 * (hDD02_i1m2 - hDD02_i1p2) +
                                          FDPart1_Rational_3_4 * (-hDD02_i1m1 + hDD02_i1p1));
        const REAL hDD_dD022 = invdxx2 * (FDPart1_Rational_1_60 * (-hDD02_i2m3 + hDD02_i2p3) + FDPart1_Rational_3_20 * (hDD02_i2m2 - hDD02_i2p2) +
                                          FDPart1_Rational_3_4 * (-hDD02_i2m1 + hDD02_i2p1));
        const REAL hDD_dD111 = invdxx1 * (FDPart1_Rational_1_60 * (-hDD11_i1m3 + hDD11_i1p3) + FDPart1_Rational_3_20 * (hDD11_i1m2 - hDD11_i1p2) +
                                          FDPart1_Rational_3_4 * (-hDD11_i1m1 + hDD11_i1p1));
        const REAL hDD_dD112 = invdxx2 * (FDPart1_Rational_1_60 * (-hDD11_i2m3 + hDD11_i2p3) + FDPart1_Rational_3_20 * (hDD11_i2m2 - hDD11_i2p2) +
                                          FDPart1_Rational_3_4 * (-hDD11_i2m1 + hDD11_i2p1));
        const REAL hDD_dD121 = invdxx1 * (FDPart1_Rational_1_60 * (-hDD12_i1m3 + hDD12_i1p3) + FDPart1_Rational_3_20 * (hDD12_i1m2 - hDD12_i1p2) +
                                          FDPart1_Rational_3_4 * (-hDD12_i1m1 + hDD12_i1p1));
        const REAL hDD_dD122 = invdxx2 * (FDPart1_Rational_1_60 * (-hDD12_i2m3 + hDD12_i2p3) + FDPart1_Rational_3_20 * (hDD12_i2m2 - hDD12_i2p2) +
                                          FDPart1_Rational_3_4 * (-hDD12_i2m1 + hDD12_i2p1));
        const REAL hDD_dD221 = invdxx1 * (FDPart1_Rational_1_60 * (-hDD22_i1m3 + hDD22_i1p3) + FDPart1_Rational_3_20 * (hDD22_i1m2 - hDD22_i1p2) +
                                          FDPart1_Rational_3_4 * (-hDD22_i1m1 + hDD22_i1p1));
        const REAL hDD_dD222 = invdxx2 * (FDPart1_Rational_1_60 * (-hDD22_i2m3 + hDD22_i2p3) + FDPart1_Rational_3_20 * (hDD22_i2m2 - hDD22_i2p2) +
                                          FDPart1_Rational_3_4 * (-hDD22_i2m1 + hDD22_i2p1));

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD100GF, i0, i1, i2)] = hDD_dD001;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD101GF, i0, i1, i2)] = hDD_dD011;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD102GF, i0, i1, i2)] = hDD_dD021;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD111GF, i0, i1, i2)] = hDD_dD111;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD112GF, i0, i1, i2)] = hDD_dD121;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD122GF, i0, i1, i2)] = hDD_dD221;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD200GF, i0, i1, i2)] = hDD_dD002;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD201GF, i0, i1, i2)] = hDD_dD012;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD202GF, i0, i1, i2)] = hDD_dD022;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD211GF, i0, i1, i2)] = hDD_dD112;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD212GF, i0, i1, i2)] = hDD_dD122;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD222GF, i0, i1, i2)] = hDD_dD222;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_WW1GF, i0, i1, i2)] = WW_dD1;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_WW2GF, i0, i1, i2)] = WW_dD2;

      } // END LOOP over non-inner-boundary points

  // PART 2 OF 2: Compute radial derivatives h_{ij,k} and W_{,k} (k = 0) at ALL interior points.
#pragma omp parallel for
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
      for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL WW_i0m3 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0 - 3, i1, i2)];
        const REAL WW_i0m2 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0 - 2, i1, i2)];
        const REAL WW_i0m1 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0 - 1, i1, i2)];
        const REAL WW_i0p1 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0 + 1, i1, i2)];
        const REAL WW_i0p2 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0 + 2, i1, i2)];
        const REAL WW_i0p3 = commondata->interp_src_gfs[IDX4(SRC_WWGF, i0 + 3, i1, i2)];
        const REAL hDD00_i0m3 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0 - 3, i1, i2)];
        const REAL hDD00_i0m2 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0 - 2, i1, i2)];
        const REAL hDD00_i0m1 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0 - 1, i1, i2)];
        const REAL hDD00_i0p1 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0 + 1, i1, i2)];
        const REAL hDD00_i0p2 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0 + 2, i1, i2)];
        const REAL hDD00_i0p3 = commondata->interp_src_gfs[IDX4(SRC_HDD00GF, i0 + 3, i1, i2)];
        const REAL hDD01_i0m3 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0 - 3, i1, i2)];
        const REAL hDD01_i0m2 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0 - 2, i1, i2)];
        const REAL hDD01_i0m1 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0 - 1, i1, i2)];
        const REAL hDD01_i0p1 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0 + 1, i1, i2)];
        const REAL hDD01_i0p2 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0 + 2, i1, i2)];
        const REAL hDD01_i0p3 = commondata->interp_src_gfs[IDX4(SRC_HDD01GF, i0 + 3, i1, i2)];
        const REAL hDD02_i0m3 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0 - 3, i1, i2)];
        const REAL hDD02_i0m2 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0 - 2, i1, i2)];
        const REAL hDD02_i0m1 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0 - 1, i1, i2)];
        const REAL hDD02_i0p1 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0 + 1, i1, i2)];
        const REAL hDD02_i0p2 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0 + 2, i1, i2)];
        const REAL hDD02_i0p3 = commondata->interp_src_gfs[IDX4(SRC_HDD02GF, i0 + 3, i1, i2)];
        const REAL hDD11_i0m3 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0 - 3, i1, i2)];
        const REAL hDD11_i0m2 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0 - 2, i1, i2)];
        const REAL hDD11_i0m1 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0 - 1, i1, i2)];
        const REAL hDD11_i0p1 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0 + 1, i1, i2)];
        const REAL hDD11_i0p2 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0 + 2, i1, i2)];
        const REAL hDD11_i0p3 = commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0 + 3, i1, i2)];
        const REAL hDD12_i0m3 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0 - 3, i1, i2)];
        const REAL hDD12_i0m2 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0 - 2, i1, i2)];
        const REAL hDD12_i0m1 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0 - 1, i1, i2)];
        const REAL hDD12_i0p1 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0 + 1, i1, i2)];
        const REAL hDD12_i0p2 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0 + 2, i1, i2)];
        const REAL hDD12_i0p3 = commondata->interp_src_gfs[IDX4(SRC_HDD12GF, i0 + 3, i1, i2)];
        const REAL hDD22_i0m3 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0 - 3, i1, i2)];
        const REAL hDD22_i0m2 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0 - 2, i1, i2)];
        const REAL hDD22_i0m1 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0 - 1, i1, i2)];
        const REAL hDD22_i0p1 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0 + 1, i1, i2)];
        const REAL hDD22_i0p2 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0 + 2, i1, i2)];
        const REAL hDD22_i0p3 = commondata->interp_src_gfs[IDX4(SRC_HDD22GF, i0 + 3, i1, i2)];
        static const REAL FDPart1_Rational_3_4 = 3.0 / 4.0;
        static const REAL FDPart1_Rational_3_20 = 3.0 / 20.0;
        static const REAL FDPart1_Rational_1_60 = 1.0 / 60.0;
        const REAL WW_dD0 = invdxx0 * (FDPart1_Rational_1_60 * (-WW_i0m3 + WW_i0p3) + FDPart1_Rational_3_20 * (WW_i0m2 - WW_i0p2) +
                                       FDPart1_Rational_3_4 * (-WW_i0m1 + WW_i0p1));
        const REAL hDD_dD000 = invdxx0 * (FDPart1_Rational_1_60 * (-hDD00_i0m3 + hDD00_i0p3) + FDPart1_Rational_3_20 * (hDD00_i0m2 - hDD00_i0p2) +
                                          FDPart1_Rational_3_4 * (-hDD00_i0m1 + hDD00_i0p1));
        const REAL hDD_dD010 = invdxx0 * (FDPart1_Rational_1_60 * (-hDD01_i0m3 + hDD01_i0p3) + FDPart1_Rational_3_20 * (hDD01_i0m2 - hDD01_i0p2) +
                                          FDPart1_Rational_3_4 * (-hDD01_i0m1 + hDD01_i0p1));
        const REAL hDD_dD020 = invdxx0 * (FDPart1_Rational_1_60 * (-hDD02_i0m3 + hDD02_i0p3) + FDPart1_Rational_3_20 * (hDD02_i0m2 - hDD02_i0p2) +
                                          FDPart1_Rational_3_4 * (-hDD02_i0m1 + hDD02_i0p1));
        const REAL hDD_dD110 = invdxx0 * (FDPart1_Rational_1_60 * (-hDD11_i0m3 + hDD11_i0p3) + FDPart1_Rational_3_20 * (hDD11_i0m2 - hDD11_i0p2) +
                                          FDPart1_Rational_3_4 * (-hDD11_i0m1 + hDD11_i0p1));
        const REAL hDD_dD120 = invdxx0 * (FDPart1_Rational_1_60 * (-hDD12_i0m3 + hDD12_i0p3) + FDPart1_Rational_3_20 * (hDD12_i0m2 - hDD12_i0p2) +
                                          FDPart1_Rational_3_4 * (-hDD12_i0m1 + hDD12_i0p1));
        const REAL hDD_dD220 = invdxx0 * (FDPart1_Rational_1_60 * (-hDD22_i0m3 + hDD22_i0p3) + FDPart1_Rational_3_20 * (hDD22_i0m2 - hDD22_i0p2) +
                                          FDPart1_Rational_3_4 * (-hDD22_i0m1 + hDD22_i0p1));

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD000GF, i0, i1, i2)] = hDD_dD000;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD001GF, i0, i1, i2)] = hDD_dD010;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD002GF, i0, i1, i2)] = hDD_dD020;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD011GF, i0, i1, i2)] = hDD_dD110;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD012GF, i0, i1, i2)] = hDD_dD120;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD022GF, i0, i1, i2)] = hDD_dD220;
        commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_WW0GF, i0, i1, i2)] = WW_dD0;

      } // END LOOP over ALL interior points
} // END FUNCTION bah_hDD_dD_and_W_dD_in_interp_src_grid_interior
