#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Read 3D metric data (in Cartesian basis) from file, basis transform, apply BCs, compute h_{ij,k}, then set up initial guess h(r,theta)
 */
int bah_initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  const int grid = 0;
  const params_struct *restrict params = &griddata[grid].params;

#include "set_CodeParameters.h"
  const int NUM_THETA = Nxx1;
  REAL *restrict coarse_to_fine = NULL;
  if (commondata->use_coarse_horizon) {
    const int num_dst_pts = Nxx1 * Nxx2;
    REAL(*dst_pts)[2] = malloc(num_dst_pts * sizeof(*dst_pts));
    coarse_to_fine = malloc(sizeof(REAL) * Nxx1 * Nxx2);
    if (dst_pts == NULL || coarse_to_fine == NULL)
      return INITIAL_DATA_MALLOC_ERROR;

#pragma omp parallel for
    for (int i2 = NGHOSTS; i2 < Nxx2 + NGHOSTS; i2++)
      for (int i1 = NGHOSTS; i1 < Nxx1 + NGHOSTS; i1++) {
        dst_pts[IDX2(i1 - NGHOSTS, i2 - NGHOSTS)][0] = griddata->xx[1][i1];
        dst_pts[IDX2(i1 - NGHOSTS, i2 - NGHOSTS)][1] = griddata->xx[2][i2];
      }
    // Choosing an NGHOSTS stencil half-width significantly speeds up BHaHAHA finds.
    bah_interpolation_2d_general__uniform_src_grid(NGHOSTS, commondata->coarse_horizon_dxx1, commondata->coarse_horizon_dxx2,
                                                   commondata->coarse_horizon_Nxx_plus_2NGHOSTS1, commondata->coarse_horizon_Nxx_plus_2NGHOSTS2,
                                                   commondata->coarse_horizon_r_theta_phi, commondata->coarse_horizon, num_dst_pts, dst_pts,
                                                   coarse_to_fine);
    free(dst_pts);
    free(commondata->coarse_horizon);
    for (int ii = 0; ii < 3; ii++)
      free(commondata->coarse_horizon_r_theta_phi[ii]);
  }

  // Step 2.a: Use OpenMP to parallelize the loop over the entire grid, initializing the h(theta, phi) scalar.
  const REAL times[3] = {commondata->bhahaha_params_and_data->t_m1, //
                         commondata->bhahaha_params_and_data->t_m2, //
                         commondata->bhahaha_params_and_data->t_m3};
  LOOP_OMP("omp parallel for",            //
           i0, NGHOSTS, Nxx0 + NGHOSTS,   // Loop over radial grid points
           i1, NGHOSTS, Nxx1 + NGHOSTS,   // Loop over polar grid points
           i2, NGHOSTS, Nxx2 + NGHOSTS) { // Loop over azimuthal grid points
    // Step 2.b: Set an initial guess value for the h(theta, phi) field at each grid point.
    //      NOTE: coarse_to_fine (interpolated above) & horizon_guess contain Nxx1 x Nxx2 points.
    if (commondata->use_coarse_horizon) {
      griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, i0, i1, i2)] = coarse_to_fine[IDX2(i1 - NGHOSTS, i2 - NGHOSTS)];
    } else if (commondata->bhahaha_params_and_data->use_fixed_radius_guess_on_full_sphere) {
      // r_max_interior = r_min_external_input + ((Nr_external_input-BHAHAHA_NGHOSTS) + 0.5) * dr
      const REAL r_max_interior =
          commondata->bhahaha_params_and_data->r_min_external_input +
          ((commondata->bhahaha_params_and_data->Nr_external_input - BHAHAHA_NGHOSTS) + 0.5) * commondata->bhahaha_params_and_data->dr_external_input;
      griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, i0, i1, i2)] = 0.8 * r_max_interior;
    } else {
      griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, i0, i1, i2)] =
          bah_quadratic_extrapolation(times, //
                                      commondata->bhahaha_params_and_data->prev_horizon_m1[IDX2(i1 - NGHOSTS, i2 - NGHOSTS)],
                                      commondata->bhahaha_params_and_data->prev_horizon_m2[IDX2(i1 - NGHOSTS, i2 - NGHOSTS)],
                                      commondata->bhahaha_params_and_data->prev_horizon_m3[IDX2(i1 - NGHOSTS, i2 - NGHOSTS)],
                                      commondata->bhahaha_params_and_data->time_external_input);
    }
    // set VVGF = eta * HHGF,
    //  so that partial_t h = VVGF - eta * HHGF = 0 at t=0. Otherwise we get really ugly dynamics.
    griddata[grid].gridfuncs.y_n_gfs[IDX4(VVGF, i0, i1, i2)] = eta_damping * griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, i0, i1, i2)];
  } // END LOOP over all gridpoints
  if (commondata->use_coarse_horizon)
    free(coarse_to_fine);

  bah_apply_bcs_inner_only(commondata, &griddata[grid].params, &griddata[grid].bcstruct, griddata[grid].gridfuncs.y_n_gfs);

  commondata->use_coarse_horizon = 1; // for next time initial_data() is called

  return BHAHAHA_SUCCESS;
} // END FUNCTION bah_initial_data
