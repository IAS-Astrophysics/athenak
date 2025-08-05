#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 *
 * Manages the over-relaxation process for the horizon function h(theta, phi) (denoted by HHGF) in the apparent horizon algorithm.
 * The function uses linear extrapolation based on the current horizon guess h and a previously computed guess h_p.
 * Over-relaxation accelerates convergence toward the steady-state solution where the expansion function Theta = 0,
 * which defines the apparent horizon.
 *
 * After over-relaxation, the time derivative v(theta, phi) (denoted by VVGF) is reset to stabilize the dynamics
 * by enforcing partial_t h = 0.
 *
 * @param commondata - Pointer to the common data structure containing global and diagnostic information.
 * @param griddata - Pointer to the grid data structure containing grid-level parameters and functions.
 * @note This function uses OpenMP for parallelization and incorporates grid-level diagnostics for monitoring and adjustments.
 *
 */
void bah_over_relaxation(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  const int grid = 0;
  const params_struct *restrict params = &griddata[grid].params;
  REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
  bhahaha_diagnostics_struct *restrict bhahaha_diags = commondata->bhahaha_diagnostics;

  // Dimensions of the computational grid including ghost zones.
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Initialize the previous horizon guess h_p and its associated time t_p on the first iteration.
  if (commondata->nn == 0) {
    commondata->h_p = malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2);
    LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
      commondata->h_p[IDX3(i0, i1, i2)] = 0.0; //
    }
    commondata->time_of_h_p = 0.0; // Initialize the time for h_p.
  } // END initialization of h_p and t_p.

  const int iterations_per_M_scale = commondata->bhahaha_params_and_data->M_scale / commondata->dt;
  const int relax_every_nn = 0.25 * M_PI * iterations_per_M_scale;
  // Attempt over-relaxation every relax_every_nn iterations, provided t_p is valid.
  if (commondata->nn >= 3 * relax_every_nn && commondata->nn % relax_every_nn == 0 && commondata->time_of_h_p != 0) {
    // Interpolate data on the source grid using radial spokes for extrapolation.
    commondata->error_flag = bah_interpolation_1d_radial_spokes_on_3d_src_grid(
        &griddata[0].params, commondata, &griddata[0].gridfuncs.y_n_gfs[IDX4pt(HHGF, 0)], griddata[0].gridfuncs.auxevol_gfs);

    // Compute horizon diagnostics (area, centroid, and Theta norms).
    bah_diagnostics_area_centroid_and_Theta_norms(commondata, griddata);

    REAL M_irr_orig = bhahaha_diags->Theta_Linf_times_M; // Original irreducible mass diagnostic value.
    REAL min_Theta_Linf_times_M = M_irr_orig;            // Track the minimum irreducible mass diagnostic value.
    REAL best_overstep = 0.0;                            // Track the optimal overstep value.

    // Explore potential overstep values to accelerate convergence.
    for (REAL overstep = 2.0; overstep <= 1e4; overstep *= 1.2) {
      LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
        const int idx3 = IDX3(i0, i1, i2);
        in_gfs[IDX4(HHGF, i0 + 1, i1, i2)] = in_gfs[IDX4(HHGF, i0, i1, i2)]; // Backup the current horizon value.

        // Perform linear extrapolation to compute an overstep for h(theta, phi).
        const REAL y_prev = commondata->h_p[IDX3(i0, i1, i2)], y_curr = in_gfs[IDX4(HHGF, i0, i1, i2)];
        const REAL t_prev = commondata->time_of_h_p, t_curr = commondata->time;
        const REAL dst_time = t_prev + overstep * (t_curr - t_prev);
        in_gfs[IDX4pt(HHGF, idx3)] = y_curr + (y_prev - y_curr) * (dst_time - t_curr) / (t_prev - t_curr);
      } // END LOOP: Apply overstep to gridpoints.

      // Recompute diagnostics after applying the overstep.
      commondata->error_flag = bah_interpolation_1d_radial_spokes_on_3d_src_grid(
          &griddata[0].params, commondata, &griddata[0].gridfuncs.y_n_gfs[IDX4pt(HHGF, 0)], griddata[0].gridfuncs.auxevol_gfs);

      if (commondata->error_flag == 0) {
        bah_diagnostics_area_centroid_and_Theta_norms(commondata, griddata); // Update diagnostics.
      } else {
        // Reset any errors that are raised.
        commondata->error_flag = BHAHAHA_SUCCESS;
      }

      LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
        in_gfs[IDX4(HHGF, i0, i1, i2)] = in_gfs[IDX4(HHGF, i0 + 1, i1, i2)]; // Revert horizon values to their original state.
      } // END LOOP: Revert horizon values to their original state.

      // Update the best overstep if diagnostics improve significantly.
      if (bhahaha_diags->Theta_Linf_times_M < min_Theta_Linf_times_M) {
        min_Theta_Linf_times_M = bhahaha_diags->Theta_Linf_times_M;
        best_overstep = overstep;
        if (commondata->bhahaha_params_and_data->verbosity_level == 2)
          printf("# Iteration %d: Best overstep factor = %e, Improvement ratio = %.15e\n", commondata->nn, best_overstep,
                 fabs(M_irr_orig - min_Theta_Linf_times_M) / M_irr_orig);
      } else {
        break; // Exit loop early if no further improvement is found.
      } // END IF: overstep resulted in a reduced error.
    } // END LOOP: Iterate over oversteps.

    // Apply the best overstep if it significantly improves convergence.
    if (best_overstep != 0.0 && fabs(M_irr_orig - min_Theta_Linf_times_M) / M_irr_orig > 5e-5) {
      LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
        const int idx3 = IDX3(i0, i1, i2);
        // Perform linear extrapolation to compute an overstep for h(theta, phi).
        const REAL y_prev = commondata->h_p[IDX3(i0, i1, i2)], y_curr = in_gfs[IDX4(HHGF, i0, i1, i2)];
        const REAL t_prev = commondata->time_of_h_p, t_curr = commondata->time;
        const REAL dst_time = t_prev + best_overstep * (t_curr - t_prev);
        in_gfs[IDX4pt(HHGF, idx3)] = y_curr + (y_prev - y_curr) * (dst_time - t_curr) / (t_prev - t_curr);

        // Update v(theta, phi) to reset time dynamics.
        in_gfs[IDX4(VVGF, i0, i1, i2)] = commondata->eta_damping * in_gfs[IDX4(HHGF, i0, i1, i2)];
      } // END LOOP over grid interior.
    } // END LOOP: Apply the optimal overstep.

    // Since over-relaxation was successful, we need to update the surface on which the metric
    //    is interpolated, and the CFL-limited timestep.
    // Interpolate metric to over-relaxed surface.
    commondata->error_flag = bah_interpolation_1d_radial_spokes_on_3d_src_grid(
        &griddata[0].params, commondata, &griddata[grid].gridfuncs.y_n_gfs[IDX4pt(HHGF, 0)], griddata[0].gridfuncs.auxevol_gfs);
    // Update the timestep based on the CFL condition on the current surface.
    bah_cfl_limited_timestep_based_on_h_equals_r(commondata, griddata);

    // Reset the stored horizon guess h_p and its time t_p.
    LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
      commondata->h_p[IDX3(i0, i1, i2)] = 0.0; //
    }
    commondata->time_of_h_p = 0.0; // Reset time for h_p.
    return;
  } // END IF: Overstep logic.

  // Update the stored horizon guess h_p every relax_every_nn iterations.
  if (commondata->nn % relax_every_nn == 0) {
    LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
      commondata->h_p[IDX3(i0, i1, i2)] = in_gfs[IDX4pt(HHGF, IDX3(i0, i1, i2))];
    } // END LOOP over all gridpoints
    commondata->time_of_h_p = commondata->time; // Update time for h_p.
  } // END IF nn % relax_every_nn == 0
} // END FUNCTION bah_over_relaxation
