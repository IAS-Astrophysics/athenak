#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Initializes the interp_src numerical grid, i.e., the source grid for 1D radial-spoke
 * interpolations during the hyperbolic relaxation.
 *
 * This function sets up the numerical grid used as the "interpolation source" for metric data
 * on the evolved grids. It configures grid parameters, allocates memory for grid functions
 * and coordinate arrays, performs interpolation from external input, applies boundary conditions,
 * and computes necessary spatial derivatives.
 *
 * @param commondata Pointer to the common data structure containing simulation parameters and data.
 * @param Nx_evol_grid Array specifying the number of grid points in each dimension for the evolved grid.
 * @return Returns BHAHAHA_SUCCESS on successful setup, or an error code if memory allocation fails.
 */
int bah_numgrid__interp_src_set_up(commondata_struct *restrict commondata, const int Nx_evol_grid[3]) {

  int i0_min_shift = 0;
  if (commondata->bhahaha_params_and_data->r_min_external_input == 0)
    i0_min_shift = NGHOSTS;

  // Step 1: Configure grid parameters for the interpolation source.
  {
    // Align the radial grid with external input data.
    commondata->interp_src_Nxx0 = commondata->external_input_Nxx0;
    commondata->interp_src_Nxx1 = Nx_evol_grid[1];
    commondata->interp_src_Nxx2 = Nx_evol_grid[2];

    // Calculate grid sizes including ghost zones.
    commondata->interp_src_Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx0 + 2 * NGHOSTS;
    commondata->interp_src_Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx1 + 2 * NGHOSTS;
    commondata->interp_src_Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx2 + 2 * NGHOSTS;

    // Set grid spacing based on external input and predefined angular ranges.
    commondata->interp_src_dxx0 = commondata->external_input_dxx0;
    const REAL xxmin1 = 0.0, xxmax1 = M_PI;
    const REAL xxmin2 = -M_PI, xxmax2 = M_PI;
    commondata->interp_src_dxx1 = (xxmax1 - xxmin1) / ((REAL)commondata->interp_src_Nxx1);
    commondata->interp_src_dxx2 = (xxmax2 - xxmin2) / ((REAL)commondata->interp_src_Nxx2);

    // Precompute inverse grid spacings for efficiency in derivative calculations.
    commondata->interp_src_invdxx0 = 1.0 / commondata->interp_src_dxx0;
    commondata->interp_src_invdxx1 = 1.0 / commondata->interp_src_dxx1;
    commondata->interp_src_invdxx2 = 1.0 / commondata->interp_src_dxx2;

    // Allocate memory for interpolation source grid functions.
    commondata->interp_src_gfs = malloc(sizeof(REAL) * commondata->interp_src_Nxx_plus_2NGHOSTS0 * commondata->interp_src_Nxx_plus_2NGHOSTS1 *
                                        commondata->interp_src_Nxx_plus_2NGHOSTS2 * NUM_INTERP_SRC_GFS);
    if (commondata->interp_src_gfs == NULL) {
      // Memory allocation failed for grid functions.
      return NUMGRID_INTERP_MALLOC_ERROR_GFS;
    }
  } // END STEP 1: Configure grid parameters for the interpolation source.

  // Step 2: Initialize coordinate arrays for the interpolation source grid.
  {
    // Step 2.a: Allocate memory for radial, theta, and phi coordinate arrays.
    commondata->interp_src_r_theta_phi[0] = (REAL *)malloc(sizeof(REAL) * commondata->interp_src_Nxx_plus_2NGHOSTS0);
    commondata->interp_src_r_theta_phi[1] = (REAL *)malloc(sizeof(REAL) * commondata->interp_src_Nxx_plus_2NGHOSTS1);
    commondata->interp_src_r_theta_phi[2] = (REAL *)malloc(sizeof(REAL) * commondata->interp_src_Nxx_plus_2NGHOSTS2);
    if (commondata->interp_src_r_theta_phi[0] == NULL || commondata->interp_src_r_theta_phi[1] == NULL ||
        commondata->interp_src_r_theta_phi[2] == NULL) {
      // Free previously allocated grid functions before exiting due to memory allocation failure.
      free(commondata->interp_src_gfs);
      return NUMGRID_INTERP_MALLOC_ERROR_RTHETAPHI;
    } // END IF memory allocation for coordinate arrays failed

    // Step 2.b: Populate coordinate arrays for a uniform, cell-centered spherical grid.
    const REAL xxmin1 = 0.0;
    const REAL xxmin2 = -M_PI;

    // Initialize radial coordinates by copying from external input.
    for (int j = 0; j < commondata->interp_src_Nxx_plus_2NGHOSTS0; j++)
      commondata->interp_src_r_theta_phi[0][j] = commondata->external_input_r_theta_phi[0][j];

    // Initialize theta coordinates with cell-centered values.
    for (int j = 0; j < commondata->interp_src_Nxx_plus_2NGHOSTS1; j++)
      commondata->interp_src_r_theta_phi[1][j] = xxmin1 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * commondata->interp_src_dxx1;

    // Initialize phi coordinates with cell-centered values.
    for (int j = 0; j < commondata->interp_src_Nxx_plus_2NGHOSTS2; j++)
      commondata->interp_src_r_theta_phi[2][j] = xxmin2 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * commondata->interp_src_dxx2;
  } // END STEP 2: Initialize coordinate arrays for the interpolation source grid.

  // Step 2.c: Extract grid sizes for use in indexing macros.
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;

  // Step 3: Perform interpolation from external input to the interpolation source grid.
  // This involves multiple 2D interpolations corresponding to the radial grid and ghost zones.
  bah_interpolation_2d_external_input_to_interp_src_grid(commondata);

  // Step 4: Transfer interpolated data from external grid functions to interpolation source grid functions.
  {
    const REAL *restrict r_theta_phi[3] = {commondata->interp_src_r_theta_phi[0], commondata->interp_src_r_theta_phi[1],
                                           commondata->interp_src_r_theta_phi[2]};
    REAL *restrict in_gfs = commondata->interp_src_gfs;

#pragma omp parallel for
    for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
      const MAYBE_UNUSED REAL xx2 = r_theta_phi[2][i2];
      for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
        const MAYBE_UNUSED REAL xx1 = r_theta_phi[1][i1];
        for (int i0 = i0_min_shift; i0 < Nxx_plus_2NGHOSTS0; i0++) {
          const MAYBE_UNUSED REAL xx0 = r_theta_phi[0][i0];
          // We perform this transformation in place; data read in will be written to the same points.
          const REAL external_Sph_W = in_gfs[IDX4(EXTERNAL_SPHERICAL_WWGF, i0, i1, i2)];
          const REAL external_Sph_trK = in_gfs[IDX4(EXTERNAL_SPHERICAL_TRKGF, i0, i1, i2)];
          const REAL external_Sph_hDD00 = in_gfs[IDX4(EXTERNAL_SPHERICAL_HDD00GF, i0, i1, i2)];
          const REAL external_Sph_hDD01 = in_gfs[IDX4(EXTERNAL_SPHERICAL_HDD01GF, i0, i1, i2)];
          const REAL external_Sph_hDD02 = in_gfs[IDX4(EXTERNAL_SPHERICAL_HDD02GF, i0, i1, i2)];
          const REAL external_Sph_hDD11 = in_gfs[IDX4(EXTERNAL_SPHERICAL_HDD11GF, i0, i1, i2)];
          const REAL external_Sph_hDD12 = in_gfs[IDX4(EXTERNAL_SPHERICAL_HDD12GF, i0, i1, i2)];
          const REAL external_Sph_hDD22 = in_gfs[IDX4(EXTERNAL_SPHERICAL_HDD22GF, i0, i1, i2)];
          const REAL external_Sph_aDD00 = in_gfs[IDX4(EXTERNAL_SPHERICAL_ADD00GF, i0, i1, i2)];
          const REAL external_Sph_aDD01 = in_gfs[IDX4(EXTERNAL_SPHERICAL_ADD01GF, i0, i1, i2)];
          const REAL external_Sph_aDD02 = in_gfs[IDX4(EXTERNAL_SPHERICAL_ADD02GF, i0, i1, i2)];
          const REAL external_Sph_aDD11 = in_gfs[IDX4(EXTERNAL_SPHERICAL_ADD11GF, i0, i1, i2)];
          const REAL external_Sph_aDD12 = in_gfs[IDX4(EXTERNAL_SPHERICAL_ADD12GF, i0, i1, i2)];
          const REAL external_Sph_aDD22 = in_gfs[IDX4(EXTERNAL_SPHERICAL_ADD22GF, i0, i1, i2)];

          in_gfs[IDX4(SRC_WWGF, i0, i1, i2)] = external_Sph_W;
          in_gfs[IDX4(SRC_TRKGF, i0, i1, i2)] = external_Sph_trK;
          in_gfs[IDX4(SRC_HDD00GF, i0, i1, i2)] = external_Sph_hDD00;
          in_gfs[IDX4(SRC_HDD01GF, i0, i1, i2)] = external_Sph_hDD01;
          in_gfs[IDX4(SRC_HDD02GF, i0, i1, i2)] = external_Sph_hDD02;
          in_gfs[IDX4(SRC_HDD11GF, i0, i1, i2)] = external_Sph_hDD11;
          in_gfs[IDX4(SRC_HDD12GF, i0, i1, i2)] = external_Sph_hDD12;
          in_gfs[IDX4(SRC_HDD22GF, i0, i1, i2)] = external_Sph_hDD22;
          in_gfs[IDX4(SRC_ADD00GF, i0, i1, i2)] = external_Sph_aDD00;
          in_gfs[IDX4(SRC_ADD01GF, i0, i1, i2)] = external_Sph_aDD01;
          in_gfs[IDX4(SRC_ADD02GF, i0, i1, i2)] = external_Sph_aDD02;
          in_gfs[IDX4(SRC_ADD11GF, i0, i1, i2)] = external_Sph_aDD11;
          in_gfs[IDX4(SRC_ADD12GF, i0, i1, i2)] = external_Sph_aDD12;
          in_gfs[IDX4(SRC_ADD22GF, i0, i1, i2)] = external_Sph_aDD22;

        } // END LOOP over i0
      } // END LOOP over i1
    } // END LOOP over i2
  } // END STEP 4: Transfer interpolated data to interpolation source grid functions.

  // Step 5: Initialize boundary condition structure for the interpolation source grid.
  bc_struct interp_src_bcstruct;
  {
    // Assign grid spacing and sizes to the boundary condition structure.
    commondata->bcstruct_dxx0 = commondata->interp_src_dxx0;
    commondata->bcstruct_dxx1 = commondata->interp_src_dxx1;
    commondata->bcstruct_dxx2 = commondata->interp_src_dxx2;
    commondata->bcstruct_Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
    commondata->bcstruct_Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
    commondata->bcstruct_Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;

    // Set up boundary conditions based on the initialized grid.
    bah_bcstruct_set_up(commondata, commondata->interp_src_r_theta_phi, &interp_src_bcstruct);
  } // END STEP 5: Initialize boundary condition structure.

  // Step 6: Apply inner boundary conditions to specific grid functions to ensure smoothness.
  {
    // Step 6.a: Access boundary condition information from the boundary condition structure.
    const bc_info_struct *restrict bc_info = &interp_src_bcstruct.bc_info;

    // Step 6.b: Iterate over relevant grid functions and apply inner boundary conditions.
#pragma omp parallel
    for (int which_gf = 0; which_gf < NUM_INTERP_SRC_GFS; which_gf++) {
      switch (which_gf) {
      case SRC_WWGF:
      case SRC_HDD00GF:
      case SRC_HDD01GF:
      case SRC_HDD02GF:
      case SRC_HDD11GF:
      case SRC_HDD12GF:
      case SRC_HDD22GF: {
#pragma omp for
        for (int pt = 0; pt < bc_info->num_inner_boundary_points; pt++) {
          const int dstpt = interp_src_bcstruct.inner_bc_array[pt].dstpt;
          const int srcpt = interp_src_bcstruct.inner_bc_array[pt].srcpt;

          // Apply boundary condition by copying and adjusting with parity.
          commondata->interp_src_gfs[IDX4pt(which_gf, dstpt)] =
              interp_src_bcstruct.inner_bc_array[pt].parity[interp_src_gf_parity[which_gf]] * commondata->interp_src_gfs[IDX4pt(which_gf, srcpt)];
        } // END LOOP over inner boundary points
        break;
      }
      default:
        // No boundary conditions needed for other grid functions.
        break;
      } // END SWITCH
    } // END LOOP over gridfunctions
  } // END STEP 6: Apply inner boundary conditions to specific grid functions.

  // Step 7: Compute spatial derivatives of h_{ij} within the interior of the interpolation source grid.
  bah_hDD_dD_and_W_dD_in_interp_src_grid_interior(commondata);

  // Step 8: Calculate radial derivatives at the outer boundaries using upwinding for stability.
  // If r_min is non-zero, apply the same procedure at the inner radial boundary.
  bah_apply_bcs_r_maxmin_partial_r_hDD_upwinding(commondata, commondata->interp_src_r_theta_phi, commondata->interp_src_gfs,
                                                 commondata->bhahaha_params_and_data->r_min_external_input != 0);

  // Step 9: Enforce boundary conditions on all interpolation source grid functions.
  {
    // Step 9.a: Access boundary condition information.
    const bc_info_struct *restrict bc_info = &interp_src_bcstruct.bc_info;

    // Step 9.b: Apply boundary conditions across all grid functions and boundary points.
#pragma omp parallel for collapse(2)
    for (int which_gf = 0; which_gf < NUM_INTERP_SRC_GFS; which_gf++) {
      for (int pt = 0; pt < bc_info->num_inner_boundary_points; pt++) {
        const int dstpt = interp_src_bcstruct.inner_bc_array[pt].dstpt;
        const int srcpt = interp_src_bcstruct.inner_bc_array[pt].srcpt;

        // Apply boundary condition with parity correction for derivative calculations.
        commondata->interp_src_gfs[IDX4pt(which_gf, dstpt)] =
            interp_src_bcstruct.inner_bc_array[pt].parity[interp_src_gf_parity[which_gf]] * commondata->interp_src_gfs[IDX4pt(which_gf, srcpt)];
      } // END LOOP over inner boundary points
    } // END LOOP over gridfunctions
  } // END STEP 9: Enforce boundary conditions on all interpolation source grid functions.

  // Step 10: Release allocated memory for boundary condition structures.
  {
    free(interp_src_bcstruct.inner_bc_array);
    for (int ng = 0; ng < NGHOSTS * 3; ng++)
      free(interp_src_bcstruct.pure_outer_bc_array[ng]);
  } // END STEP 10: Free allocated memory for boundary condition structures.

  return BHAHAHA_SUCCESS;
} // END FUNCTION bah_numgrid__interp_src_set_up
