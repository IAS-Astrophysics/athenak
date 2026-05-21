#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 *
 * Performs adaptive extrapolation to determine the center coordinates and minimum/maximum radii
 * at a specified time based on available horizon finds. The extrapolation order adjusts up to quadratic
 * based on the number of available data points:
 * - Quadratic extrapolation if three horizons are available.
 * - Linear extrapolation if two horizons are available.
 * - Zeroth order extrapolation if only one horizon is available.
 *
 * After extrapolation, the function adjusts the minimum and maximum radii to ensure appropriate search
 * volume adjustments based on specific conditions.
 *
 * @param pars     Pointer to a structure containing time points, center coordinates,
 * and radius values for previous observations and the target time.
 * @param x_center Pointer to the variable where the extrapolated x-coordinate center will be stored.
 * @param y_center Pointer to the variable where the extrapolated y-coordinate center will be stored.
 * @param z_center Pointer to the variable where the extrapolated z-coordinate center will be stored.
 * @param r_min    Pointer to the variable where the extrapolated minimum radius will be stored.
 * @param r_max    Pointer to the variable where the extrapolated maximum radius will be stored.
 * @return         This function does not return a value but updates the provided variables with the extrapolated values.
 *
 * @note This function uses adaptive extrapolation methods to predict values at the current simulation time
 * using known data from up to the most recent three horizon finds. It adjusts the extrapolation order
 * based on the number of available data points to ensure flexibility and accuracy.
 *
 */
void bah_xyz_center_r_minmax(const bhahaha_params_and_data_struct *restrict pars, REAL *restrict x_center, REAL *restrict y_center,
                             REAL *restrict z_center, REAL *restrict r_min, REAL *restrict r_max) {

  // Initialize time points for extrapolation.
  const REAL times[3] = {pars->t_m1, pars->t_m2, pars->t_m3};
  // Destination time for extrapolation.
  const REAL dst_time = pars->time_external_input;

  // Extrapolate center coordinates using quadratic extrapolation.
  *x_center = bah_quadratic_extrapolation(times, pars->x_center_m1, pars->x_center_m2, pars->x_center_m3, dst_time);
  *y_center = bah_quadratic_extrapolation(times, pars->y_center_m1, pars->y_center_m2, pars->y_center_m3, dst_time);
  *z_center = bah_quadratic_extrapolation(times, pars->z_center_m1, pars->z_center_m2, pars->z_center_m3, dst_time);

  // Extrapolate minimum and maximum radii using quadratic extrapolation.
  *r_min = bah_quadratic_extrapolation(times, pars->r_min_m1, pars->r_min_m2, pars->r_min_m3, dst_time);
  *r_max = bah_quadratic_extrapolation(times, pars->r_max_m1, pars->r_max_m2, pars->r_max_m3, dst_time);

  // The BHaHAHA spherical grid must enclose the horizon with some buffer
  //    below r_min and above r_max, so that the hyperbolic relaxation
  //    has space to explore. Immediately after this function,
  //    bah_radial_grid_cell_centered_set_up() is called, which will set
  //    up the actual grid based on these r_min and r_max values that are
  //    input. As its first step, it will floor r_min to zero, and ceiling
  //    r_max to the user-specifiable max_search_radius for this horizon.
  //    So no worries if r_max > max_search_radius below!
  if (times[2] == -1) {
    // Adjust radii to expand search volume when the third horizon find is missing.
    *r_min *= 0.8;
    *r_max *= 1.2;
  } // END IF: checking if the horizon has not been found three times in a row.
  else {
    if ((pars->r_max_m1 - pars->r_max_m3) > 0.05 * (*r_max)) {
      // Increase r_max by 20% if the maximum radius is growing rapidly.
      *r_max *= 1.2;
    } // END IF: increasing r_max due to rapid growth
    else {
      // If the BH's minimum AH radius isn't changing rapidly,
      //   adjust r_min to 5% above the extrapolated value.
      *r_max *= 1.05;
    } // END ELSE: moderate adjustment of r_max

    if ((pars->r_min_m3 - pars->r_min_m1) > 0.05 * (*r_min)) {
      // Decrease r_min by 20% if the minimum radius is shrinking rapidly.
      *r_min *= 0.8;
    } // END IF: decreasing r_min due to rapid shrinkage
    else {
      // If the BH's minimum AH radius isn't changing rapidly,
      //   adjust r_min to 5% below the extrapolated value.
      *r_min *= 0.95;
    } // END ELSE: moderate adjustment of r_min
  } // END ELSE: adjusting radii based on growth/shrinkage rates
} // END FUNCTION bah_xyz_center_r_minmax
