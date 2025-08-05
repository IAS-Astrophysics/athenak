#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

// Parameter to set a min number of interior points for the BHaHAHA destination grid;
//   for example, if output_r_min = output_r_max, this will ensure at least 4 interior
//   points to enable the horizon to deform.
static const int MIN_INTERIOR_PTS = 4;

static void setup_grid_r_min_interior_zero(const REAL r_min_interior, const REAL r_max_interior, const REAL max_search_radius, const int max_Nr,
                                           int *restrict output_Nr_interp, REAL *restrict output_r_min, REAL *restrict output_dr) {
  // r_min_interior = r_min = 0
  // r_max_interior = r_min + Nr_interior * dr
  //                = Nr_interior * dr
  // Nr_interior = r_max_interior / max_search_radius * (max_Nr - BHAHAHA_NGHOSTS)
  *output_r_min = 0.0; // by assumption
  // Account for rounding in typecast:
  int Nr_interior = (int)((1.0 + 1e-7) * r_max_interior / max_search_radius * ((REAL)(max_Nr - BHAHAHA_NGHOSTS)));
  if (Nr_interior < MIN_INTERIOR_PTS) {
    Nr_interior = MIN_INTERIOR_PTS;
  }
  *output_dr = r_max_interior / ((REAL)Nr_interior);
  *output_Nr_interp = Nr_interior + BHAHAHA_NGHOSTS;
} // END FUNCTION setup_grid_r_min_interior_zero()

static void setup_grid_r_min_interior_gt_zero(const REAL r_min_interior, const REAL r_max_interior, const REAL max_search_radius, const int max_Nr,
                                              int *restrict output_Nr_interp, REAL *restrict output_r_min, REAL *restrict output_dr) {
  // r_min_interior = r_min + BHAHAHA_NGHOSTS * dr
  // r_max_interior = r_min + (BHAHAHA_NGHOSTS + Nr_interior) * dr
  // Nr_interior = (r_max_interior - r_min_interior) / max_search_radius * (max_Nr - BHAHAHA_NGHOSTS)
  // Account for rounding in typecast:
  int Nr_interior = (int)(((1.0 + 1e-7) * r_max_interior - r_min_interior) / max_search_radius * ((REAL)(max_Nr - BHAHAHA_NGHOSTS)));
  if (Nr_interior >= MIN_INTERIOR_PTS) {
    *output_dr = (r_max_interior - r_min_interior) / ((REAL)Nr_interior);
    *output_r_min = r_min_interior - BHAHAHA_NGHOSTS * (*output_dr);
  } else {
    const int Nr_interior_pts_to_add = MIN_INTERIOR_PTS - Nr_interior;
    Nr_interior = MIN_INTERIOR_PTS;
    const REAL dr_min = 0.25 * r_max_interior / ((REAL)(max_Nr - BHAHAHA_NGHOSTS));
    const REAL dr_fiducial = (r_max_interior - r_min_interior) / ((REAL)Nr_interior);
    if (dr_fiducial >= dr_min) {
      *output_dr = dr_fiducial;
      *output_r_min = r_min_interior - (BHAHAHA_NGHOSTS) * (*output_dr);
    } else {
      *output_dr = dr_min;
      *output_r_min = r_min_interior - (BHAHAHA_NGHOSTS + (Nr_interior_pts_to_add / 2)) * (*output_dr);
      REAL implied_r_max_interior = *output_r_min + (BHAHAHA_NGHOSTS + Nr_interior - 1) * (*output_dr);
      while (implied_r_max_interior > max_search_radius) {
        *output_r_min -= (*output_dr);
        // Update implied_r_max_interior.
        implied_r_max_interior = *output_r_min + (BHAHAHA_NGHOSTS + Nr_interior - 1) * (*output_dr);
      }
    }
  }
  *output_Nr_interp = Nr_interior + 2 * BHAHAHA_NGHOSTS;
} // END FUNCTION setup_grid_r_min_interior_gt_zero()

/**
 *
 * Initializes a cell-centered radial grid for interpolation.
 *
 * This function:
 * - Clamps `input_r_min` and `input_r_max` within [0, `max_search_radius`].
 * - Calculates the radial step size (`dr`).
 * - Determines the number of interpolation points, ensuring a minimum of 4 interior points.
 * - Adjusts `output_r_min` and `output_r_max` to accommodate ghost cells.
 * - Populates the `radii` array with computed radial coordinates.
 *
 * @param Nr_interp_max               Maximum number of radial interpolation points.
 * @param max_search_radius           Upper limit for the search radius; caps the adjusted maximum radius.
 * @param input_r_min                 Initial minimum radius; may equal `input_r_max`.
 * @param input_r_max                 Initial maximum radius; may equal `input_r_min`.
 * @param output_Nr_interp            Pointer to store the adjusted number of interpolation points.
 * @param output_r_min_interior       Pointer to store the minimum interior radius of the cell-centered radial grid.
 * @param output_dr                   Pointer to store the output grid spacing.
 * @param radii                       Array to store the computed radial coordinates.
 *
 * @return void
 *
 * @note Ensures the radial grid includes ghost cells and maintains non-negative radii.
 *
 */
void bah_radial_grid_cell_centered_set_up(const int Nr_interp_max, const REAL max_search_radius, const REAL input_r_min, const REAL input_r_max,
                                          int *restrict output_Nr_interp, REAL *restrict output_r_min_interior, REAL *restrict output_dr,
                                          REAL radii[Nr_interp_max]) {

  // Adjust radii to be within permissible range
  REAL r_min_interior = input_r_min < 0.0 ? 0.0 : input_r_min;
  REAL r_max_interior = input_r_max > max_search_radius ? max_search_radius : input_r_max;

  // Case 1: if r_min_interior == 0
  REAL output_r_min;
  if (r_min_interior == 0) {
    setup_grid_r_min_interior_zero(r_min_interior, r_max_interior, max_search_radius, Nr_interp_max, output_Nr_interp, &output_r_min, output_dr);
  } else {
    setup_grid_r_min_interior_gt_zero(r_min_interior, r_max_interior, max_search_radius, Nr_interp_max, output_Nr_interp, &output_r_min, output_dr);
    if (output_r_min < 0.0)
      setup_grid_r_min_interior_zero(r_min_interior, r_max_interior, max_search_radius, Nr_interp_max, output_Nr_interp, &output_r_min, output_dr);
  }
  // Display the radial step size and adjusted radii for debugging purposes.
  // printf("Radial step size dr: %e | Adjusted r_min: %e, r_max: %e, Interior points: %d\n", (*output_dr), , output_Nr_interior);

  // Set radii[] array for convenience; could be constructed from
  //   output_r_min, output_Nr_interp, and output_dr.
  for (int i = 0; i < (*output_Nr_interp); i++) {
    radii[i] = (output_r_min) + ((REAL)(i) + 0.5) * (*output_dr);
  } // END LOOP: populating radii array for r_min > 0
  *output_r_min_interior = output_r_min + ((REAL)(BHAHAHA_NGHOSTS)) * (*output_dr);
} // END FUNCTION bah_radial_grid_cell_centered_set_up

#ifdef STANDALONE

/**
 * Displays the input parameters for a given test case.
 *
 * @param Nr_interp_max        Maximum number of radial interpolation points.
 * @param max_search_radius    Initial maximum radius for grid scaling.
 * @param input_r_min          Initial minimum radius.
 * @param input_r_max          Maximum radius.
 *
 * @return void
 */
void print_input_parameters(const int Nr_interp_max, const REAL max_search_radius, const REAL input_r_min, const REAL input_r_max) {
  printf("Input parameters:\n");
  printf("  Nr_interp_max = %d\n", Nr_interp_max);
  printf("  max_search_radius = %f\n", max_search_radius);
  printf("  input_r_min = %f\n", input_r_min);
  printf("  input_r_max = %f\n", input_r_max);
} // END FUNCTION: print_input_parameters

/**
 * Displays the output parameters after setting up the radial grid.
 *
 * @param output_Nr_interp     Adjusted number of interpolation points.
 * @param output_dr            Adjusted minimum radius.
 * @param output_r_min_interior         Adjusted minimum radius.
 * @param output_r_max         Adjusted maximum radius.
 *
 * @return void
 */
void print_output_parameters(const int output_Nr_interp, const REAL output_dr, const REAL output_r_min_interior, const REAL output_r_max_interior) {
  printf("Output parameters for cell-centered radial grid:\n");
  printf("  output_Nr_interp = %d\n", output_Nr_interp);
  printf("  output_dr = %f\n", output_dr);
  printf("  output_r_min_interior = %f\n", output_r_min_interior);
  printf("  output_r_max_interior = %f\n", output_r_max_interior);
} // END FUNCTION: print_output_parameters

/**
 * Displays the computed radial coordinates.
 *
 * @param radii                Array of computed radial coordinates.
 * @param output_Nr_interp     Number of interpolation points.
 *
 * @return void
 */
void print_radii(const REAL radii[], const int output_Nr_interp) {
  printf("Radii:\n");
  for (int i = 0; i < output_Nr_interp; i++) {
    printf("  radii[%d] = %f\n", i, radii[i]);
  } // END LOOP: printing each radial coordinate
  printf("\n");
} // END FUNCTION: print_radii

/**
 * Executes a test case by displaying inputs, setting up the radial grid, and displaying outputs.
 *
 * @param test_case_description Description of the test case scenario.
 * @param Nr_interp_max         Maximum number of radial interpolation points.
 * @param max_search_radius     Maximum search radius.
 * @param input_r_min           Input minimum radius.
 * @param input_r_max           Input maximum radius.
 *
 * @return void
 */
void run_test_case(const char *test_case_description, int Nr_interp_max, REAL max_search_radius, REAL input_r_min, REAL input_r_max) {
  printf("%s\n", test_case_description);

  // Display the input parameters for the current test case.
  print_input_parameters(Nr_interp_max, max_search_radius, input_r_min, input_r_max);

  int output_Nr_interp;
  REAL output_r_min_interior, output_dr;
  REAL radii[Nr_interp_max];

  // Set up the radial grid based on the input parameters.
  bah_radial_grid_cell_centered_set_up(Nr_interp_max, max_search_radius, input_r_min, input_r_max, &output_Nr_interp, &output_r_min_interior,
                                       &output_dr, radii);

  // Display the adjusted output parameters after grid setup.
  const REAL output_r_max_interior = output_r_min_interior + output_dr * ((REAL)output_Nr_interp - BHAHAHA_NGHOSTS);
  print_output_parameters(output_Nr_interp, output_dr, output_r_min_interior, output_r_max_interior);

  // Display the computed radial coordinates.
  print_radii(radii, output_Nr_interp);
} // END FUNCTION: run_test_case

int main(void) {

  // Execute Test case 1: input_r_min > 0, ensuring actual_r_min remains non-negative.
  run_test_case("Test case 1: input_r_min > 0, ensuring actual_r_min remains non-negative.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                1.0,  // input_r_min
                10.0  // input_r_max
  );                  // END TEST CASE 1

  // Execute Test case 2: input_r_min > 0 with a different minimum radius.
  run_test_case("Test case 2: input_r_min > 0 with a different minimum radius.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                2.0,  // input_r_min
                10.0  // input_r_max
  );                  // END TEST CASE 2

  // Execute Test case 3: input_r_min <= 0, verifying grid starts at zero.
  run_test_case("Test case 3: input_r_min <= 0, verifying grid starts at zero.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                0.0,  // input_r_min
                10.0  // input_r_max
  );                  // END TEST CASE 3

  // Execute Test case 4: Adjust interpolation points to meet minimum grid size.
  run_test_case("Test case 4: Adjust interpolation points to meet minimum grid size.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                1.0,  // input_r_min
                1.5   // input_r_max
  );                  // END TEST CASE 4

  // Execute Test case 5: input_r_min slightly above zero, expecting grid to start at zero.
  run_test_case("Test case 5: input_r_min slightly above zero, expecting grid to start at zero.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                0.02, // input_r_min
                10.0  // input_r_max
  );                  // END TEST CASE 5

  // Execute Test case 6: High input_r_min nearing input_r_max, verifying minimal interpolation points.
  run_test_case("Test case 6: input_r_min nearing input_r_max, expecting Nr_interp = 4 + 2*BHAHAHA_NGHOSTS.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                9.0,  // input_r_min
                10.0  // input_r_max
  );                  // END TEST CASE 6

  // Execute Test case 7: input_r_min = input_r_max, expecting Nr_interp = 4 + 2*BHAHAHA_NGHOSTS.
  run_test_case("Test case 7: input_r_min = input_r_max, expecting Nr_interp = 4 + 2*BHAHAHA_NGHOSTS.",
                20,   // Nr_interp_max
                10.0, // max_search_radius
                10.0, // input_r_min
                10.0  // input_r_max
  );                  // END TEST CASE 7

  // Execute Test case 8: input_r_min = input_r_max < max_search_radius, expecting Nr_interp = 4 + 2*BHAHAHA_NGHOSTS and dr = min_dr
  run_test_case("Test case 8: input_r_min = input_r_max < max_search_radius, expecting Nr_interp = 4 + 2*BHAHAHA_NGHOSTS and dr = min_dr.",
                20,  // Nr_interp_max
                1.0, // max_search_radius
                0.5, // input_r_min
                0.5  // input_r_max
  );                 // END TEST CASE 8

  return 0;
} // END FUNCTION: main

#endif // END STANDALONE
