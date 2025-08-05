#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

#include <sys/time.h> // Include sys/time.h for timing functions like gettimeofday()

/**
 * Converts two timeval structures to milliseconds and returns the elapsed time.
 *
 * @param start - The starting time.
 * @param end - The ending time.
 * @return - The elapsed time in milliseconds.
 */
static REAL timeval_to_milliseconds(struct timeval start, struct timeval end) {
  double start_ms = start.tv_sec * 1000.0 + start.tv_usec / 1000.0;
  double end_ms = end.tv_sec * 1000.0 + end.tv_usec / 1000.0;
  return end_ms - start_ms;
}

/**
 * Frees all dynamically allocated memory associated with griddata,
 * except for external input grid functions.
 *
 * @param commondata - Pointer to the common data structure containing shared parameters.
 * @param griddata - Pointer to the grid data structure to be freed.
 */
static void free_all_but_external_input_gfs(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  const int grid = 0;

  // Free precomputed reference metric arrays.
  bah_rfm_precompute_free(commondata, &griddata[grid].params, griddata[grid].rfmstruct);
  free(griddata[grid].rfmstruct);

  // Free inner boundary condition array.
  free(griddata[grid].bcstruct.inner_bc_array);

  // Free pure outer boundary condition arrays.
  for (int ng = 0; ng < NGHOSTS * 3; ng++) {
    free(griddata[grid].bcstruct.pure_outer_bc_array[ng]);
  } // END LOOP: freeing pure outer boundary condition arrays

  // Free grid functions allocated by the Method of Lines.
  bah_MoL_free_memory_y_n_gfs(&griddata[grid].gridfuncs);
  bah_MoL_free_memory_non_y_n_gfs(&griddata[grid].gridfuncs);

  // Free coordinate arrays for each dimension.
  for (int i = 0; i < 3; i++) {
    free(griddata[grid].xx[i]);
  } // END LOOP: freeing coordinate arrays

  // Free the griddata structure itself.
  free(griddata);

  // Free interpolation source grid functions.
  free(commondata->interp_src_gfs);

  // Free interpolation source coordinate arrays for each dimension.
  for (int i = 0; i < 3; i++) {
    free(commondata->interp_src_r_theta_phi[i]);
  } // END LOOP: freeing interpolation source coordinate arrays

  // Free previous horizon guess array, used for overstep.
  if (commondata->h_p != NULL)
    free(commondata->h_p);
} // END FUNCTION: free_all_but_external_input_gfs

/**
 *
 * Finds the apparent horizon using BHaHAHA.
 *
 * This driver function initializes necessary data structures, sets up grids, and runs the main simulation loop
 * to identify the apparent horizon with progressively refined grid resolutions.
 *
 * @param bhahaha_params_and_data - Input parameters and data for the algorithm.
 * @param bhahaha_diags - Diagnostics data structure to be updated during execution.
 * @return - Returns BHaHAHA (0) on success or a nonzero error code on failure.
 *
 */
int bah_find_horizon(bhahaha_params_and_data_struct *restrict bhahaha_params_and_data, bhahaha_diagnostics_struct *restrict bhahaha_diags) {

  // Step 1.a: Start global timer.
  struct timeval start_time;
  {
    // Verify that gettimeofday() is functional before proceeding.
    if (gettimeofday(&start_time, NULL) != 0) {
      return FIND_HORIZON_GETTIMEOFDAY_BROKEN;
    }
  } // END BLOCK: gettimeofday() sanity check.

  commondata_struct commondata; // Structure containing parameters common to all grids.

  // Assign input diagnostics and parameters to commondata for widespread access.
  commondata.bhahaha_diagnostics = bhahaha_diags;
  commondata.bhahaha_params_and_data = bhahaha_params_and_data;

  // Step 1.b: Initialize commondata parameters to their default values.
  bah_commondata_struct_set_to_default(&commondata);
  commondata.eta_damping = bhahaha_params_and_data->eta_damping_times_M / bhahaha_params_and_data->M_scale;
  commondata.CFL_FACTOR = bhahaha_params_and_data->cfl_factor;
  commondata.KO_diss_strength = commondata.bhahaha_params_and_data->KO_strength;
  // Initialize counter for total number of points at which Theta is evaluated.
  commondata.bhahaha_diagnostics->Theta_eval_points_counter = 0;

  // Set internal flags for horizon refinement and final iteration diagnostics.
  commondata.use_coarse_horizon = 0;
  commondata.is_final_iteration = 0;

  // Step 1.c: Define angular resolution patterns to optimize the solving process.
  const int n_resolutions = bhahaha_params_and_data->num_resolutions_multigrid;
  int Ntheta[MAX_RESOLUTIONS], Nphi[MAX_RESOLUTIONS];
  memcpy(Ntheta, bhahaha_params_and_data->Ntheta_array_multigrid, sizeof(int) * MAX_RESOLUTIONS);
  memcpy(Nphi, bhahaha_params_and_data->Nphi_array_multigrid, sizeof(int) * MAX_RESOLUTIONS);

  // Step 1.d: Set up external input grids by adding inner ghost zones and applying boundary conditions.
  commondata.external_input_gfs_Cart_basis_no_gzs = bhahaha_params_and_data->input_metric_data;
  commondata.error_flag = bah_numgrid__external_input_set_up(&commondata, n_resolutions, Ntheta, Nphi);
  if (commondata.error_flag != BHAHAHA_SUCCESS) {
    return commondata.error_flag;
  }

  // Step 2: Iterate over different grid resolutions to refine the apparent horizon.
  commondata.error_flag = BHAHAHA_SUCCESS; // Assume success initially.

  for (int resolution = 0; resolution < n_resolutions; resolution++) {
    // Start timing for the current resolution.
    struct timeval res_start_time;
    gettimeofday(&res_start_time, NULL);

    // Adjust diagnostics output frequency based on grid resolution.
    commondata.output_diagnostics_every_nn = 4 * (int)pow(Ntheta[n_resolutions - 1] / Ntheta[resolution], 2.0);

    int Nx_evol_grid[3];
    griddata_struct *restrict griddata; // Structure containing data specific to the current grid.
    const int grid = 0;

    // Define the evolution grid size for the current resolution.
    Nx_evol_grid[0] = 1;
    Nx_evol_grid[1] = Ntheta[resolution];
    Nx_evol_grid[2] = Nphi[resolution];

    // Step 2.b: Set up interpolation source grid and allocate grid functions.
    commondata.error_flag = bah_numgrid__interp_src_set_up(&commondata, Nx_evol_grid);
    if (commondata.error_flag != BHAHAHA_SUCCESS) {
      for (int i = 0; i < 3; i++) {
        free(commondata.external_input_r_theta_phi[i]);
      }
      free(commondata.external_input_gfs);
      return commondata.error_flag;
    }

    // Step 2.c: Allocate memory for griddata structure.
    griddata = (griddata_struct *restrict)malloc(sizeof(griddata_struct));

    // Step 2.d: Initialize griddata parameters to their default values.
    bah_params_struct_set_to_default(&commondata, griddata);

    // Step 2.e: Configure the 2D numerical grid for the Apparent Horizon finder.
    bah_numgrid__evol_set_up(&commondata, griddata, Nx_evol_grid);

    const params_struct *restrict params = &griddata[grid].params;
    const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

    {
      const int grid = 0;

      // Step 3.a: Allocate storage for initial grid functions (y_n_gfs).
      bah_MoL_malloc_y_n_gfs(&commondata, params, &griddata[grid].gridfuncs);

      // Step 3.b: Allocate storage for additional grid functions required for time-stepping.
      bah_MoL_malloc_non_y_n_gfs(&commondata, params, &griddata[grid].gridfuncs);

      // Step 3.c: Initialize commondata.h_p = NULL, so that if Step 5.a (interp 1D) fails,
      //   it doesn't trigger a double free() of h_p.
      commondata.h_p = NULL;
    } // END BLOCK: Allocation of grid functions

    // Step 4: Initialize initial data for the simulation.
    if (bah_initial_data(&commondata, griddata) != BHAHAHA_SUCCESS) {
      free_all_but_external_input_gfs(&commondata, griddata);
      for (int i = 0; i < 3; i++) {
        free(commondata.external_input_r_theta_phi[i]);
      }
      free(commondata.external_input_gfs);
      return INITIAL_DATA_MALLOC_ERROR;
    }

    // Step 5: Execute the main simulation loop to evolve the horizon over time.
    int stop_condition = 0;
    while (commondata.time < commondata.t_final) { // Main loop to advance the simulation.

      // Step 5.a: Interpolate metric to current best surface. This is done at the start of each
      //           timestep instead of at each RK substep for efficiency reasons.
      commondata.error_flag = bah_interpolation_1d_radial_spokes_on_3d_src_grid(
          &griddata[0].params, &commondata, &griddata[grid].gridfuncs.y_n_gfs[IDX4pt(HHGF, 0)], griddata[0].gridfuncs.auxevol_gfs);
      if (commondata.error_flag != BHAHAHA_SUCCESS)
        break;

      // Step 5.b: Update the timestep based on the CFL condition.
      bah_cfl_limited_timestep_based_on_h_equals_r(&commondata, griddata);

      // Step 5.c: Attempt over-relaxation every once in a while. If an
      //           over-relaxation is performed, the CFL timestep
      //           and metric will be updated to the new surface.
      bah_over_relaxation(&commondata, griddata);
      if (commondata.error_flag != BHAHAHA_SUCCESS) // could fail due to metric interpolation.
        break;

      // Step 5.d: Time-varying eta prescription -- reduce eta with residual
      if (bhahaha_params_and_data->enable_eta_varying_alg_for_precision_common_horizon && commondata.nn % 10000 == 0) {
        commondata.error_flag = bah_interpolation_1d_radial_spokes_on_3d_src_grid(
            &griddata[0].params, &commondata, &griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, 0, 0, 0)], griddata[0].gridfuncs.auxevol_gfs);
        bah_diagnostics_area_centroid_and_Theta_norms(&commondata, griddata);
        REAL eta_min_times_M = 0.15;
        REAL eta_max_times_M = 3.0;
        if (resolution == 1)
          eta_max_times_M = 3.0;
        if (resolution >= 2)
          eta_max_times_M = 30.0;

        const REAL eta_damping_times_M = MAX(eta_min_times_M, eta_max_times_M * sqrt(bhahaha_diags->Theta_Linf_times_M));
        commondata.eta_damping = eta_damping_times_M / bhahaha_params_and_data->M_scale;

        LOOP_OMP("omp parallel for", i0, NGHOSTS, NGHOSTS + 1, i1, 0, Nxx_plus_2NGHOSTS1, i2, 0, Nxx_plus_2NGHOSTS2) {
          griddata[grid].gridfuncs.y_n_gfs[IDX4(VVGF, i0, i1, i2)] =
              commondata.eta_damping * griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, i0, i1, i2)];
        } // END LOOP over all gridpoints on horizon surface.
      } // END time-varying eta prescription.

      // Step 5.e: Output diagnostic information.
      bah_diagnostics(&commondata, griddata);
      if (commondata.error_flag != BHAHAHA_SUCCESS) {
        break;
      } // END IF: Check for diagnostic errors

      // Step 5.f: Determine if stop conditions are met to exit the simulation loop.
      if (commondata.nn > bhahaha_params_and_data->max_iterations) {
        commondata.error_flag = FIND_HORIZON_MAX_ITERATIONS_EXCEEDED;
        stop_condition = 1;
        break;
      } else if (commondata.nn > 2 && // Ensure a minimum number of iterations.
                 bhahaha_diags->Theta_Linf_times_M <= bhahaha_params_and_data->Theta_Linf_times_M_tolerance &&
                 bhahaha_diags->Theta_L2_times_M <= bhahaha_params_and_data->Theta_L2_times_M_tolerance) {
        stop_condition = 1;
        break;
      } // END IF: Check multiple stop conditions

      // Step 5.g: Advance the simulation using the Method of Lines with Runge-Kutta-like integration.
      if (!stop_condition)
        bah_MoL_step_forward_in_time(&commondata, griddata);
      if (commondata.error_flag != BHAHAHA_SUCCESS) {
        break;
      } // END IF: Check for time-stepping errors
    } // END LOOP: Main simulation loop

    {
      // End timing for the current resolution and display elapsed time.
      struct timeval end_time;
      gettimeofday(&end_time, NULL);

      if (bhahaha_params_and_data->verbosity_level == 2) {
        printf("#Nth x Nph = %d x %d elapsed time = %.1f ms / %.1f ms so far...\n", params->Nxx1, params->Nxx2,
               timeval_to_milliseconds(res_start_time, end_time), timeval_to_milliseconds(start_time, end_time));
      }
    } // END BLOCK: Timing and logging

    if (commondata.error_flag == BHAHAHA_SUCCESS) {
      // Step 6: Save the coarse horizon for subsequent resolutions or output final diagnostics.
      if (resolution < n_resolutions - 1) {
        // Allocate memory for storing coarse horizon data.
        const int total_points = Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
        commondata.coarse_horizon = malloc(sizeof(REAL) * total_points);

        // Store horizon data including ghost zones for interpolation in the next resolution.
        const int NUM_THETA = Nxx_plus_2NGHOSTS1; // NUM_THETA needed for IDX2() macro.
#pragma omp parallel for
        for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
          for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
            commondata.coarse_horizon[IDX2(i1, i2)] = griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, NGHOSTS, i1, i2)];
          } // END LOOP: theta indices
        } // END LOOP: phi indices

        // Save grid parameters for the coarse horizon to maintain consistency.
        commondata.coarse_horizon_dxx1 = params->dxx1;
        commondata.coarse_horizon_dxx2 = params->dxx2;
        commondata.coarse_horizon_Nxx_plus_2NGHOSTS1 = Nxx_plus_2NGHOSTS1;
        commondata.coarse_horizon_Nxx_plus_2NGHOSTS2 = Nxx_plus_2NGHOSTS2;

        // Allocate and store coordinate arrays for the coarse horizon.
        commondata.coarse_horizon_r_theta_phi[0] = malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
        for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
          commondata.coarse_horizon_r_theta_phi[0][i0] = griddata[grid].xx[0][i0];
        } // END LOOP: radial coordinates

        commondata.coarse_horizon_r_theta_phi[1] = malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
        for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
          commondata.coarse_horizon_r_theta_phi[1][i1] = griddata[grid].xx[1][i1];
        } // END LOOP: theta coordinates

        commondata.coarse_horizon_r_theta_phi[2] = malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS2);
        for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
          commondata.coarse_horizon_r_theta_phi[2][i2] = griddata[grid].xx[2][i2];
        } // END LOOP: phi coordinates
      } else { // IF: Horizon found at final resolution

        // Store the final horizon data and perform a last diagnostic output.
        const int NUM_THETA = params->Nxx1; // Required for IDX2() macro.
        memcpy(commondata.bhahaha_params_and_data->prev_horizon_m3, commondata.bhahaha_params_and_data->prev_horizon_m2,
               sizeof(REAL) * NUM_THETA * params->Nxx2);
        memcpy(commondata.bhahaha_params_and_data->prev_horizon_m2, commondata.bhahaha_params_and_data->prev_horizon_m1,
               sizeof(REAL) * NUM_THETA * params->Nxx2);
#pragma omp parallel for
        for (int i2 = 0; i2 < params->Nxx2; i2++) {
          for (int i1 = 0; i1 < params->Nxx1; i1++) {
            commondata.bhahaha_params_and_data->prev_horizon_m1[IDX2(i1, i2)] =
                griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, NGHOSTS, i1 + NGHOSTS, i2 + NGHOSTS)];
          } // END LOOP: theta indices
        } // END LOOP: phi indices

        // Adjust setting for the final iteration, to trigger diagnostics and compute additional diagnostics.
        commondata.is_final_iteration = 1;

      } // END IF/ELSE: Handling final resolution
    } else if (commondata.error_flag == INTERP1D_HORIZON_TOO_LARGE) {
      // Handle specific error when the horizon exceeds interpolation limits.
      REAL max_radius = -1e10;
#pragma omp parallel for reduction(max : max_radius)
      for (int i2 = 0; i2 < params->Nxx2; i2++) {
        for (int i1 = 0; i1 < params->Nxx1; i1++) {
          REAL current_radius = griddata[grid].gridfuncs.y_n_gfs[IDX4(HHGF, NGHOSTS, i1 + NGHOSTS, i2 + NGHOSTS)];
          if (current_radius > max_radius) {
            max_radius = current_radius;
          }
        } // END LOOP: theta indices
      } // END LOOP: phi indices

      if (commondata.bhahaha_params_and_data->verbosity_level > 0) {
        // r_max_interior = r_min_external_input + ((Nr_external_input-BHAHAHA_NGHOSTS) + 0.5)*dr
        const REAL r_max_interior =
            commondata.bhahaha_params_and_data->r_min_external_input +
            ((commondata.bhahaha_params_and_data->Nr_external_input - BHAHAHA_NGHOSTS) + 0.5) * commondata.bhahaha_params_and_data->dr_external_input;
        printf("ERROR: h_max = %#.4g too close to r_max_search = %#.4g. "
               "Try either increasing search radius or decreasing cfl_factor.\n",
               max_radius, r_max_interior);
      }
    } // END IF: Handling specific error conditions

    // Step 7: Horizon found! Compute final diagnostics.
    if (commondata.error_flag == BHAHAHA_SUCCESS) {
      const int orig_output_diagnostics_every_nn = commondata.output_diagnostics_every_nn;
      commondata.output_diagnostics_every_nn = 1;
      // Compute diagnostics, cycling _m{3,2} and storing _m1 data while we're at it for {x,y,z}_center
      //   as they depend on centroids being computed, and r_{min,max} for good measure.
      bah_diagnostics(&commondata, griddata);
      commondata.output_diagnostics_every_nn = orig_output_diagnostics_every_nn;
    } // END BLOCK: Freeing current grid resolution memory

    // Step 8: Release all allocated memory for the current grid resolution.
    free_all_but_external_input_gfs(&commondata, griddata);

    if (commondata.error_flag != BHAHAHA_SUCCESS) {
      break;
    } // END IF: Check for errors after freeing memory
  } // END LOOP: Iterating over grid resolutions

  // Step 9: After processing all resolutions, release external input memory.
  for (int i = 0; i < 3; i++) {
    free(commondata.external_input_r_theta_phi[i]);
  }
  free(commondata.external_input_gfs);

  // Display final timing information if verbosity is enabled.
  if (commondata.bhahaha_params_and_data->verbosity_level > 0) {
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    printf("#-={ BHaHAHA finished horizon %d / %d in %#.4g seconds }=-\n", commondata.bhahaha_params_and_data->which_horizon,
           commondata.bhahaha_params_and_data->num_horizons, timeval_to_milliseconds(start_time, end_time) / 1000.0);
  }

  return commondata.error_flag;
} // END FUNCTION bah_find_horizon
