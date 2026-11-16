#include "BHaH_defines.h"
/**
 * Computes the first derivative of a grid function along the x0 direction using finite-difference
 * schemes with arbitrary upwind offsets. The offset determines which stencil to use:
 * - Negative offsets correspond to backward (upwind) stencils.
 * - Zero offset corresponds to a centered stencil.
 * - Positive offsets correspond to forward (downwind) stencils.
 * Returns the computed first derivative at the given grid point.
 */
static inline REAL FD1_arbitrary_upwind_x0_dirn(const commondata_struct *restrict commondata, const REAL *restrict gf, const int i0, const int i1,
                                                const int i2, const int offset) {

  const MAYBE_UNUSED int Nxx_plus_2NGHOSTS0 = commondata->bcstruct_Nxx_plus_2NGHOSTS0;
  const MAYBE_UNUSED int Nxx_plus_2NGHOSTS1 = commondata->bcstruct_Nxx_plus_2NGHOSTS1;
  const MAYBE_UNUSED int Nxx_plus_2NGHOSTS2 = commondata->bcstruct_Nxx_plus_2NGHOSTS2;
  const REAL invdxx0 = 1.0 / (commondata->bcstruct_dxx0);
  switch (offset) {
  case 0:
    return (-1.0 / 60.0 * gf[IDX3(i0 - 3, i1, i2)] + 3.0 / 20.0 * gf[IDX3(i0 - 2, i1, i2)] - 3.0 / 4.0 * gf[IDX3(i0 - 1, i1, i2)] +
            3.0 / 4.0 * gf[IDX3(i0 + 1, i1, i2)] - 3.0 / 20.0 * gf[IDX3(i0 + 2, i1, i2)] + 1.0 / 60.0 * gf[IDX3(i0 + 3, i1, i2)]) *
           invdxx0;
  case 1:
    return (+1.0 / 30.0 * gf[IDX3(i0 - 2, i1, i2)] - 2.0 / 5.0 * gf[IDX3(i0 - 1, i1, i2)] - 7.0 / 12.0 * gf[IDX3(i0, i1, i2)] +
            4.0 / 3.0 * gf[IDX3(i0 + 1, i1, i2)] - 1.0 / 2.0 * gf[IDX3(i0 + 2, i1, i2)] + 2.0 / 15.0 * gf[IDX3(i0 + 3, i1, i2)] -
            1.0 / 60.0 * gf[IDX3(i0 + 4, i1, i2)]) *
           invdxx0;
  case -1:
    return (+1.0 / 60.0 * gf[IDX3(i0 - 4, i1, i2)] - 2.0 / 15.0 * gf[IDX3(i0 - 3, i1, i2)] + 1.0 / 2.0 * gf[IDX3(i0 - 2, i1, i2)] -
            4.0 / 3.0 * gf[IDX3(i0 - 1, i1, i2)] + 7.0 / 12.0 * gf[IDX3(i0, i1, i2)] + 2.0 / 5.0 * gf[IDX3(i0 + 1, i1, i2)] -
            1.0 / 30.0 * gf[IDX3(i0 + 2, i1, i2)]) *
           invdxx0;
  case 2:
    return (-1.0 / 6.0 * gf[IDX3(i0 - 1, i1, i2)] - 77.0 / 60.0 * gf[IDX3(i0, i1, i2)] + 5.0 / 2.0 * gf[IDX3(i0 + 1, i1, i2)] -
            5.0 / 3.0 * gf[IDX3(i0 + 2, i1, i2)] + 5.0 / 6.0 * gf[IDX3(i0 + 3, i1, i2)] - 1.0 / 4.0 * gf[IDX3(i0 + 4, i1, i2)] +
            1.0 / 30.0 * gf[IDX3(i0 + 5, i1, i2)]) *
           invdxx0;
  case -2:
    return (-1.0 / 30.0 * gf[IDX3(i0 - 5, i1, i2)] + 1.0 / 4.0 * gf[IDX3(i0 - 4, i1, i2)] - 5.0 / 6.0 * gf[IDX3(i0 - 3, i1, i2)] +
            5.0 / 3.0 * gf[IDX3(i0 - 2, i1, i2)] - 5.0 / 2.0 * gf[IDX3(i0 - 1, i1, i2)] + 77.0 / 60.0 * gf[IDX3(i0, i1, i2)] +
            1.0 / 6.0 * gf[IDX3(i0 + 1, i1, i2)]) *
           invdxx0;
  case 3:
    return (-49.0 / 20.0 * gf[IDX3(i0, i1, i2)] + 6 * gf[IDX3(i0 + 1, i1, i2)] - 15.0 / 2.0 * gf[IDX3(i0 + 2, i1, i2)] +
            20.0 / 3.0 * gf[IDX3(i0 + 3, i1, i2)] - 15.0 / 4.0 * gf[IDX3(i0 + 4, i1, i2)] + 6.0 / 5.0 * gf[IDX3(i0 + 5, i1, i2)] -
            1.0 / 6.0 * gf[IDX3(i0 + 6, i1, i2)]) *
           invdxx0;
  case -3:
    return (+1.0 / 6.0 * gf[IDX3(i0 - 6, i1, i2)] - 6.0 / 5.0 * gf[IDX3(i0 - 5, i1, i2)] + 15.0 / 4.0 * gf[IDX3(i0 - 4, i1, i2)] -
            20.0 / 3.0 * gf[IDX3(i0 - 3, i1, i2)] + 15.0 / 2.0 * gf[IDX3(i0 - 2, i1, i2)] - 6 * gf[IDX3(i0 - 1, i1, i2)] +
            49.0 / 20.0 * gf[IDX3(i0, i1, i2)]) *
           invdxx0;

  default:
    // Return NaN if offset is invalid
    return 0.0 / 0.0;
  }
} // END FUNCTION FD1_arbitrary_upwind_x0_dirn

/**
 * Applies boundary conditions to r_max and possibly r_min (outer) boundaries of the computational grid by computing
 * the radial derivative partial_r hDD using upwind finite-difference stencils. The function iterates over the boundary points
 * in the radial (x0) direction and computes the derivative for each component of hDD, ensuring that the chosen stencils do not
 * access grid points outside the computational domain. OpenMP parallelization is employed to optimize computations over
 * the angular directions (x1 and x2).
 *
 * @param commondata - Pointer to common data structure containing boundary condition and grid information.
 * @param xx - Array of pointers to grid coordinate arrays.
 * @param gfs - Pointer to the grid functions array where derivatives are stored.
 * @param fill_r_min_ghosts - Boolean flag indicating if r_min boundary ghost zones should be filled.
 * @return - Void.
 * @note - Parallelizes angular computations to enhance performance and reduce computation time.
 *
 */
void bah_apply_bcs_r_maxmin_partial_r_hDD_upwinding(const commondata_struct *restrict commondata, REAL *restrict xx[3], REAL *restrict gfs,
                                                    const bool fill_r_min_ghosts) {

  const int Nxx_plus_2NGHOSTS0 = commondata->bcstruct_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->bcstruct_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->bcstruct_Nxx_plus_2NGHOSTS2;
  const int Nxxtot012 = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  //////////////////////////////////////////////////////////////
  // Evaluate partial_r hDD at r = r_max boundary

  // Iterate over the r_max boundary points in the radial (x0) direction.
  for (int i0 = Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 < Nxx_plus_2NGHOSTS0; i0++) {
    // i0 = Nxx_plus_2NGHOSTS0 - 1 --> offset = -NGHOSTS
    // i0 = Nxx_plus_2NGHOSTS0 - 2 --> offset = -NGHOSTS + 1
    // -> i0 = Nxx_plus_2NGHOSTS0 - j --> offset = -NGHOSTS + (j-1)
    // -> j = Nxx_plus_2NGHOSTS0 - i0 --> offset = -NGHOSTS + ((Nxx_plus_2NGHOSTS0-i0) - 1)
    const int offset = -NGHOSTS + ((Nxx_plus_2NGHOSTS0 - i0) - 1);

    // Parallelize computations over the angular (x1 and x2) directions to leverage multi-core processing.
#pragma omp parallel for
    for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
      for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
        for (int which_gf = 0; which_gf < NUM_INTERP_SRC_GFS; which_gf++) {
          int base_gf = -1;

          // Map the interpolation source grid function to its corresponding base grid function.
          switch (which_gf) {
          case SRC_PARTIAL_D_HDD000GF:
            base_gf = SRC_HDD00GF;
            break;
          case SRC_PARTIAL_D_HDD001GF:
            base_gf = SRC_HDD01GF;
            break;
          case SRC_PARTIAL_D_HDD002GF:
            base_gf = SRC_HDD02GF;
            break;
          case SRC_PARTIAL_D_HDD011GF:
            base_gf = SRC_HDD11GF;
            break;
          case SRC_PARTIAL_D_HDD012GF:
            base_gf = SRC_HDD12GF;
            break;
          case SRC_PARTIAL_D_HDD022GF:
            base_gf = SRC_HDD22GF;
            break;
          case SRC_PARTIAL_D_WW0GF:
            base_gf = SRC_WWGF;
            break;
          default:
            // Skip processing for undefined grid function indices to maintain data integrity.
            break;
          } // END SWITCH to set base_gf

          if (base_gf != -1) {
            // Compute the radial derivative using the appropriate upwind stencil based on the offset.
            const REAL partial_x0_f = FD1_arbitrary_upwind_x0_dirn(commondata, &gfs[base_gf * Nxxtot012], i0, i1, i2, offset);
            // Store the computed derivative in the target grid function array.
            gfs[IDX4(which_gf, i0, i1, i2)] = partial_x0_f;
          } // END IF the derivative gridfunction needs to be set
        } // END LOOP over gridfunctions
      } // END LOOP over i1
    } // END LOOP over i2
  } // END LOOP over i0: iterating through r_max radial boundary points

  /////////////////////////////////////////////////////////////////
  // Evaluate partial_r hDD at r = r_min boundary if required
  if (fill_r_min_ghosts) {
    // Iterate over the r_min boundary points in the radial (x0) direction.
    for (int i0 = 0; i0 < NGHOSTS; i0++) {
      const int offset = NGHOSTS - i0;

      // Parallelize computations over the angular (x1 and x2) directions to leverage multi-core processing.
#pragma omp parallel for
      for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
        for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
          for (int which_gf = 0; which_gf < NUM_INTERP_SRC_GFS; which_gf++) {
            int base_gf = -1;

            // Map the interpolation source grid function to its corresponding base grid function.
            switch (which_gf) {
            case SRC_PARTIAL_D_HDD000GF:
              base_gf = SRC_HDD00GF;
              break;
            case SRC_PARTIAL_D_HDD001GF:
              base_gf = SRC_HDD01GF;
              break;
            case SRC_PARTIAL_D_HDD002GF:
              base_gf = SRC_HDD02GF;
              break;
            case SRC_PARTIAL_D_HDD011GF:
              base_gf = SRC_HDD11GF;
              break;
            case SRC_PARTIAL_D_HDD012GF:
              base_gf = SRC_HDD12GF;
              break;
            case SRC_PARTIAL_D_HDD022GF:
              base_gf = SRC_HDD22GF;
              break;
            case SRC_PARTIAL_D_WW0GF:
              base_gf = SRC_WWGF;
              break;
            default:
              // Skip processing for undefined grid function indices to maintain data integrity.
              break;
            } // END SWITCH to set base_gf

            if (base_gf != -1) {
              // Compute the radial derivative using the appropriate upwind stencil based on the offset.
              const REAL partial_x0_f = FD1_arbitrary_upwind_x0_dirn(commondata, &gfs[base_gf * Nxxtot012], i0, i1, i2, offset);
              // Store the computed derivative in the target grid function array.
              gfs[IDX4(which_gf, i0, i1, i2)] = partial_x0_f;
            } // END IF the derivative gridfunction needs to be set
          } // END LOOP over gridfunctions
        } // END LOOP over i1
      } // END LOOP over i2
    } // END LOOP: over i0, iterating through r_min radial boundary points
  } // END IF: fill_r_min_ghosts flag check
} // END FUNCTION bah_apply_bcs_r_maxmin_partial_r_hDD_upwinding
