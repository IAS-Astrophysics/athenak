#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * BHaHAHA apparent horizon diagnostics: Compute Theta L2 and Linfinity norms.
 */
void bah_diagnostics_min_max_mean_radii_wrt_centroid(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  const int grid = 0;
  bhahaha_diagnostics_struct *restrict bhahaha_diags = commondata->bhahaha_diagnostics;
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

  // Compute radii quantities. Mean radius is area-weighted, since physical gridspacing is quite uneven.
  REAL sum_curr_area = 0.0;
  REAL sum_mean_radius = 0.0;
  REAL min_radius_squared = +1e30;
  REAL max_radius_squared = -1e30;
#pragma omp parallel
  {
#pragma omp for
    for (int i2 = NGHOSTS; i2 < NGHOSTS + Nxx2; i2++) {
      const REAL weight2 = weights[(i2 - NGHOSTS) % weight_stencil_size];
      const MAYBE_UNUSED REAL xx2 = xx[2][i2];
      for (int i1 = NGHOSTS; i1 < NGHOSTS + Nxx1; i1++) {
        const REAL weight1 = weights[(i1 - NGHOSTS) % weight_stencil_size];
        const MAYBE_UNUSED REAL xx1 = xx[1][i1];
        for (int i0 = NGHOSTS; i0 < NGHOSTS + Nxx0; i0++) {
          /*
           * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
           * Read gridfunction(s) from main memory and compute FD stencils as needed.
           */
          const REAL WW = auxevol_gfs[IDX4(WWGF, i0, i1, i2)];
          const REAL hDD00 = auxevol_gfs[IDX4(HDD00GF, i0, i1, i2)];
          const REAL hDD01 = auxevol_gfs[IDX4(HDD01GF, i0, i1, i2)];
          const REAL hDD02 = auxevol_gfs[IDX4(HDD02GF, i0, i1, i2)];
          const REAL hDD11 = auxevol_gfs[IDX4(HDD11GF, i0, i1, i2)];
          const REAL hDD12 = auxevol_gfs[IDX4(HDD12GF, i0, i1, i2)];
          const REAL hDD22 = auxevol_gfs[IDX4(HDD22GF, i0, i1, i2)];
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
          static const REAL FDPart1_Rational_3_4 = 3.0 / 4.0;
          static const REAL FDPart1_Rational_3_20 = 3.0 / 20.0;
          static const REAL FDPart1_Rational_1_60 = 1.0 / 60.0;
          const REAL hh_dD1 = invdxx1 * (FDPart1_Rational_1_60 * (-hh_i1m3 + hh_i1p3) + FDPart1_Rational_3_20 * (hh_i1m2 - hh_i1p2) +
                                         FDPart1_Rational_3_4 * (-hh_i1m1 + hh_i1p1));
          const REAL hh_dD2 = invdxx2 * (FDPart1_Rational_1_60 * (-hh_i2m3 + hh_i2p3) + FDPart1_Rational_3_20 * (hh_i2m2 - hh_i2p2) +
                                         FDPart1_Rational_3_4 * (-hh_i2m1 + hh_i2p1));

          /*
           * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
           * Evaluate SymPy expressions and write to main memory.
           */
          const REAL FDPart3tmp0 = (1.0 / ((WW) * (WW)));
          const REAL FDPart3tmp4 = sin(xx1);
          const REAL FDPart3tmp6 = ((hh) * (hh));
          const REAL FDPart3tmp3 = FDPart3tmp0 * (hDD00 + 1);
          const REAL FDPart3tmp7 = ((FDPart3tmp4) * (FDPart3tmp4)) * FDPart3tmp6;
          const REAL FDPart3tmp2 = FDPart3tmp0 * hDD01 * hh;
          const REAL FDPart3tmp5 = FDPart3tmp0 * FDPart3tmp4 * hDD02 * hh;
          const REAL area_element =
              sqrt((FDPart3tmp0 * (FDPart3tmp6 * hDD11 + FDPart3tmp6) + 2 * FDPart3tmp2 * hh_dD1 + FDPart3tmp3 * ((hh_dD1) * (hh_dD1))) *
                       (FDPart3tmp0 * (FDPart3tmp7 * hDD22 + FDPart3tmp7) + FDPart3tmp3 * ((hh_dD2) * (hh_dD2)) + 2 * FDPart3tmp5 * hh_dD2) -
                   ((FDPart3tmp0 * FDPart3tmp4 * FDPart3tmp6 * hDD12 + FDPart3tmp2 * hh_dD2 + FDPart3tmp3 * hh_dD1 * hh_dD2 + FDPart3tmp5 * hh_dD1) *
                    (FDPart3tmp0 * FDPart3tmp4 * FDPart3tmp6 * hDD12 + FDPart3tmp2 * hh_dD2 + FDPart3tmp3 * hh_dD1 * hh_dD2 + FDPart3tmp5 * hh_dD1)));

#pragma omp critical
          {
            sum_curr_area += area_element * weight1 * weight2;
            const REAL r = in_gfs[IDX4(HHGF, NGHOSTS, i1, i2)];
            const REAL theta = commondata->interp_src_r_theta_phi[1][i1];
            const REAL phi = commondata->interp_src_r_theta_phi[2][i2];
            const REAL xx = r * sin(theta) * cos(phi);
            const REAL yy = r * sin(theta) * sin(phi);
            const REAL zz = r * cos(theta);
            // Radius as measured from AH centroid:
            REAL radius_squared = ((xx - bhahaha_diags->x_centroid_wrt_coord_origin) * (xx - bhahaha_diags->x_centroid_wrt_coord_origin) + //
                                   (yy - bhahaha_diags->y_centroid_wrt_coord_origin) * (yy - bhahaha_diags->y_centroid_wrt_coord_origin) + //
                                   (zz - bhahaha_diags->z_centroid_wrt_coord_origin) * (zz - bhahaha_diags->z_centroid_wrt_coord_origin));
            sum_mean_radius += sqrt(radius_squared) * area_element * weight1 * weight2;
            if (radius_squared < min_radius_squared)
              min_radius_squared = radius_squared;
            if (radius_squared > max_radius_squared)
              max_radius_squared = radius_squared;
          } // END OMP CRITICAL
        } // END LOOP over i0
      } // END LOOP over i1
    } // END LOOP over i2
  } // END OMP PARALLEL

  // Store commondata diagnostic parameters.
  const REAL curr_area = sum_curr_area * params->dxx1 * params->dxx2;
  bhahaha_diags->min_coord_radius_wrt_centroid = sqrt(min_radius_squared);
  bhahaha_diags->max_coord_radius_wrt_centroid = sqrt(max_radius_squared);
  bhahaha_diags->mean_coord_radius_wrt_centroid = sum_mean_radius * params->dxx1 * params->dxx2 / curr_area;
} // END FUNCTION bah_diagnostics_min_max_mean_radii_wrt_centroid
