#include "BHaH_defines.h"
#include "intrinsics/simd_intrinsics.h"

// #define STANDALONE
// Enables print statements:
// #define DEBUG
#define DST_IDX4(g, i, j, k) ((i) + dst_Nxx_plus_2NGHOSTS0 * ((j) + dst_Nxx_plus_2NGHOSTS1 * ((k) + dst_Nxx_plus_2NGHOSTS2 * (g))))
#define SRC_IDX4(g, i, j, k) ((i) + src_Nxx_plus_2NGHOSTS0 * ((j) + src_Nxx_plus_2NGHOSTS1 * ((k) + src_Nxx_plus_2NGHOSTS2 * (g))))
#define DST_IDX3(i, j, k) ((i) + dst_Nxx_plus_2NGHOSTS0 * ((j) + dst_Nxx_plus_2NGHOSTS1 * (k)))

#pragma GCC optimize("unroll-loops")
/**
 *
 * Perform 1D radial Lagrange interpolation along the radial spokes of a 3D
 * spherical grid. It computes the interpolated values using Lagrange polynomials.
 *
 * @param params               Pointer to destination grid parameters.
 * @param commondata           Pointer to common data, including source grid parameters and source grid functions.
 * @param dst_radii_aka_src_h_gf Array of destination radial points where interpolation is performed.
 * @param dst_interp_gfs       Array to store the interpolated results.
 *
 * @return                     BHAHAHA_SUCCESS on success, or an appropriate error code on failure.
 *
 * @note
 * - Assumes that the source grid has uniform grid spacing in r, theta, and phi.
 * - The interpolation destination is radial index i = NGHOSTS.
 *
 */
int bah_interpolation_1d_radial_spokes_on_3d_src_grid(const params_struct *restrict params, const commondata_struct *restrict commondata,
                                                      const REAL *restrict dst_radii_aka_src_h_gf, REAL *restrict dst_interp_gfs) {

  // Define the interpolation order based on the number of ghost zones.
  const int INTERP_ORDER = (2 * NinterpGHOSTS + 1); // Interpolation order corresponds to the number of points in the stencil per dimension.

  // UNPACK PARAMETERS
  // Source grid parameters from commondata.
  const int src_Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int src_Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int src_Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;

  // Destination grid parameters from params.
  const int dst_Nxx1 = params->Nxx1;
  const int dst_Nxx2 = params->Nxx2;
  const int dst_Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int dst_Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int dst_Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  // Source grid coordinates for r, theta, and phi.
  const REAL *restrict interp_src_r_theta_phi[3] = {commondata->interp_src_r_theta_phi[0], commondata->interp_src_r_theta_phi[1],
                                                    commondata->interp_src_r_theta_phi[2]};
  const REAL xxmin_incl_ghosts0 = interp_src_r_theta_phi[0][0];

  // Precompute (1/(src_dr))^(INTERP_ORDER-1) normalization factor.
  const REAL src_invdxx0 = commondata->interp_src_invdxx0;
  const REAL src_invdxx0_INTERP_ORDERm1 = pow(commondata->interp_src_invdxx0, (INTERP_ORDER - 1));

  // Source grid functions.
  const REAL *restrict interp_src_gfs = commondata->interp_src_gfs;

  // Perform sanity checks to ensure all required pointers are valid.
  if (interp_src_r_theta_phi[0] == NULL || interp_src_gfs == NULL || dst_radii_aka_src_h_gf == NULL || dst_interp_gfs == NULL)
    return INTERP1D_NULL_PTRS; // Return error if any pointer is NULL.

  // Ensure that the interpolation order does not exceed the source grid size.
  if (INTERP_ORDER > commondata->interp_src_Nxx0 + 2 * NinterpGHOSTS)
    return INTERP1D_INTERP_ORDER_GT_NXX_PLUS_2NINTERPGHOSTS0; // Return error if interpolation order is too high.

  // Precompute inverse denominators for Lagrange interpolation coefficients to optimize performance.
  REAL inv_denom[INTERP_ORDER];
  for (int i = 0; i < INTERP_ORDER; i++) {
    REAL denom = 1.0;
    for (int j = 0; j < i; j++)
      denom *= (REAL)(i - j);
    for (int j = i + 1; j < INTERP_ORDER; j++)
      denom *= (REAL)(i - j);
    inv_denom[i] = 1.0 / denom; // Store the inverse to avoid repeated division operations.
  } // END LOOP: Precompute inverse denominators.

  // Initialize the error flag to track any interpolation issues.
  int error_flag = BHAHAHA_SUCCESS;

  // Parallelize the outer loops using OpenMP for better performance.
#pragma omp parallel for
  for (int iphi = NGHOSTS; iphi < dst_Nxx2 + NGHOSTS; iphi++) {         // Iterate over phi indices, ignoring ghost zones.
    for (int itheta = NGHOSTS; itheta < dst_Nxx1 + NGHOSTS; itheta++) { // Iterate over theta indices, ignoring ghost zones.
      // Perform interpolation only at radial index i = NGHOSTS.
      const REAL r_dst = dst_radii_aka_src_h_gf[DST_IDX3(NGHOSTS, itheta, iphi)];
      // Calculate the central index for the stencil in the radial direction.
      const int idx_center0 = (int)((r_dst - xxmin_incl_ghosts0) * src_invdxx0 + 0.5);

      // Determine the base index for the interpolation stencil.
      const int base_idx_x = idx_center0 - NinterpGHOSTS;

      {
        // Ensure the stencil is within valid grid bounds.
        if ((idx_center0 - NinterpGHOSTS < 0) || (idx_center0 + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS0)) {
#ifdef DEBUG
          // Provide detailed error messages in debug mode for easier troubleshooting.
          fprintf(stderr, "ERROR: Interpolation stencil exceeds grid boundaries for r_dst = %.6f. itheta, iphi = %d %d.\n", r_dst, itheta, iphi);
          fprintf(stderr, "Grid bounds along radial direction: [0, %d], idx_center = %d, stencil indices: [%d, %d]\n", src_Nxx_plus_2NGHOSTS0 - 1,
                  idx_center0, idx_center0 - NinterpGHOSTS, idx_center0 + NinterpGHOSTS);
          fprintf(stderr, "Ensure that the destination point is within grid bounds or adjust the interpolation stencil.\n");
#endif // DEBUG
#pragma omp critical
          {
            error_flag = INTERP1D_HORIZON_TOO_LARGE; // Set error flag if stencil is out of bounds.
            if (idx_center0 - NinterpGHOSTS < 0)
              error_flag = INTERP1D_HORIZON_TOO_SMALL; // Adjust error flag if stencil is too small.
          }
          continue; // Skip further work for this iteration.
        } // END IF stencil in bounds

#ifdef DEBUG
        // Verify that the central index is the closest grid point to the destination radius.
        if (fabs(interp_src_r_theta_phi[0][idx_center0] - r_dst) > commondata->interp_src_dxx0 * 0.5) {
          fprintf(stderr, "ERROR: Radial center index too far from destination point!\n");
        } // END IF central index is properly centered.
#endif // DEBUG
      } // END SANITY CHECKS.

      // Arrays to store Lagrange interpolation coefficients and differences.
      REAL coeff_r[INTERP_ORDER];
      REAL diffs[INTERP_ORDER];

      // Step 1: Precompute all differences between destination radius and source grid points within the stencil.
#pragma omp simd
      for (int j = 0; j < INTERP_ORDER; j++) {
        diffs[j] = r_dst - interp_src_r_theta_phi[0][base_idx_x + j];
      } // END LOOP over j.

      // Step 2: Compute Lagrange basis coefficients for the radial direction.
#pragma omp simd
      for (int i = 0; i < INTERP_ORDER; i++) {
        REAL numer_i = 1.0;
        // Compute product for j < i (scalar loop).
        for (int j = 0; j < i; j++) {
          numer_i *= diffs[j];
        } // END LOOP over j < i.
        // Compute product for j > i (scalar loop).
        for (int j = i + 1; j < INTERP_ORDER; j++) {
          numer_i *= diffs[j];
        } // END LOOP over j > i.

        coeff_r[i] = numer_i * inv_denom[i]; // Scale by the inverse denominator.
      } // END LOOP over i.

      // Perform the 1D Lagrange interpolation along the radial direction with SIMD optimizations.
      for (int gf = 0; gf < NUM_INTERP_SRC_GFS; gf++) {
        REAL sum = 0.0;
        REAL_SIMD_ARRAY vec_sum = SetZeroSIMD; // Initialize vector sum to zero

        // Vectorized loop using SIMD
        int i = 0;
        for (; i <= INTERP_ORDER - simd_width; i += simd_width) {
          REAL_SIMD_ARRAY vec_src = ReadSIMD(&interp_src_gfs[SRC_IDX4(gf, base_idx_x + i, itheta, iphi)]);
          REAL_SIMD_ARRAY vec_coeff = ReadSIMD(&coeff_r[i]);
          vec_sum = FusedMulAddSIMD(vec_src, vec_coeff, vec_sum);
        }

        // Accumulate SIMD result
        sum += HorizAddSIMD(vec_sum);

        // Handle remaining elements that don't fit into a full SIMD register
        for (; i < INTERP_ORDER; i++) {
          const int i_gf_idx = base_idx_x + i;
          sum += interp_src_gfs[SRC_IDX4(gf, i_gf_idx, itheta, iphi)] * coeff_r[i];
        }

        dst_interp_gfs[DST_IDX4(gf, NGHOSTS, itheta, iphi)] = sum * src_invdxx0_INTERP_ORDERm1;
      } // END LOOP over grid functions.
    } // END LOOP over theta.
  } // END LOOP over phi.

  return error_flag; // Return the status of the interpolation process.
} // END FUNCTION bah_interpolation_1d_radial_spokes_on_3d_src_grid
#pragma GCC reset_options // Reset compiler optimizations after the function

#ifdef STANDALONE

const REAL RADIAL_EXTENT = 10.0; // Radial domain [0, RADIAL_EXTENT].

/**
 * Initializes the coordinate arrays for the source grid.
 *
 * Allocates and sets up the coordinate values for r, theta, and phi based on the number of ghost zones and grid dimensions.
 *
 * @param N_r                        Number of grid points in the radial direction.
 * @param N_theta                    Number of grid points in the theta direction.
 * @param N_phi                      Number of grid points in the phi direction.
 * @param src_r_theta_phi            Array to store coordinate values for r, theta, and phi.
 * @param src_dxx                    Array to store grid spacings in r, theta, and phi directions.
 * @param src_Nxx_plus_2NGHOSTS      Array containing the total number of grid points in each direction, including ghost zones.
 */
void initialize_coordinates(const int N_r, const int N_theta, const int N_phi, REAL *src_r_theta_phi[3], REAL src_dxx[3],
                            const int src_Nxx_plus_2NGHOSTS[3]) {
  // Calculate grid spacings based on the number of grid points.
  src_dxx[0] = RADIAL_EXTENT / N_r; // Radial grid spacing for domain [0, RADIAL_EXTENT].
  src_dxx[1] = M_PI / N_theta;      // Theta grid spacing for domain [0, pi).
  src_dxx[2] = 2.0 * M_PI / N_phi;  // Phi grid spacing for domain [-pi, pi).

  // Allocate memory for the coordinate arrays.
  src_r_theta_phi[0] = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS[0]);
  src_r_theta_phi[1] = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS[1]);
  src_r_theta_phi[2] = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS[2]);

  if (src_r_theta_phi[0] == NULL || src_r_theta_phi[1] == NULL || src_r_theta_phi[2] == NULL) {
    fprintf(stderr, "Memory allocation failed for coordinate arrays.\n");
    exit(EXIT_FAILURE);
  }

  // Initialize radial coordinate with ghost zones.
  for (int i = 0; i < src_Nxx_plus_2NGHOSTS[0]; i++) {
    src_r_theta_phi[0][i] = (i - NinterpGHOSTS) * src_dxx[0];
  }

  // Initialize theta coordinate with ghost zones.
  const REAL xxmin1 = 0.0;
  for (int j = 0; j < src_Nxx_plus_2NGHOSTS[1]; j++) {
    src_r_theta_phi[1][j] = xxmin1 + ((REAL)(j - NGHOSTS) + 0.5) * src_dxx[1];
  }

  // Initialize phi coordinate with ghost zones.
  const REAL xxmin2 = -M_PI;
  for (int k = 0; k < src_Nxx_plus_2NGHOSTS[2]; k++) {
    src_r_theta_phi[2][k] = xxmin2 + ((REAL)(k - NGHOSTS) + 0.5) * src_dxx[2];
  }
} // END FUNCTION: initialize_coordinates.

/**
 * Initializes the source grid function using a provided analytic function.
 *
 * Evaluates the analytic function at each grid point and stores the result in the source grid function array.
 *
 * @param src_Nxx_plus_2NGHOSTS  Array containing the total number of grid points in each direction, including ghost zones.
 * @param src_r_theta_phi        Array containing coordinate values for r, theta, and phi.
 * @param src_gf                 Array to store the initialized source grid function values.
 */
void initialize_src_gf(const int src_Nxx_plus_2NGHOSTS[3], REAL *src_r_theta_phi[3], REAL *src_gf) {
  // Declare the variables used in SRC_IDX4 macro
  const int src_Nxx_plus_2NGHOSTS0 = src_Nxx_plus_2NGHOSTS[0];
  const int src_Nxx_plus_2NGHOSTS1 = src_Nxx_plus_2NGHOSTS[1];
  const int src_Nxx_plus_2NGHOSTS2 = src_Nxx_plus_2NGHOSTS[2];
  // Allocate memory to precompute r_func values for each radial position
  REAL *r_func_values = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS0);
  if (r_func_values == NULL) {
    fprintf(stderr, "Memory allocation failed for r_func_values.\n");
    exit(EXIT_FAILURE);
  }
  // Precompute r_func for each radial position
  for (int i = 0; i < src_Nxx_plus_2NGHOSTS0; i++) {
    const REAL r = src_r_theta_phi[0][i];
    r_func_values[i] = sin(r) * r * r;
  }
  // Set the analytic grid functions
#pragma omp parallel for
  for (int gf = 0; gf < NUM_INTERP_SRC_GFS; gf++) {
    for (int k = 0; k < src_Nxx_plus_2NGHOSTS2; k++) {
      const REAL phi = src_r_theta_phi[2][k];
      const REAL cosphi = cos(phi);
      for (int j = 0; j < src_Nxx_plus_2NGHOSTS1; j++) {
        const REAL theta = src_r_theta_phi[1][j];
        const REAL sintheta = sin(theta);
        for (int i = 0; i < src_Nxx_plus_2NGHOSTS0; i++) {
          src_gf[SRC_IDX4(gf, i, j, k)] = r_func_values[i] * sintheta * cosphi;
        }
      }
    }
  }
  free(r_func_values);
} // END FUNCTION: initialize_src_gf.

/**
 * Main function to execute the standalone 1D interpolation and perform convergence validation tests.
 *
 * Sets up different grid resolutions, performs interpolation,
 * computes the L2 norm of interpolation errors, and evaluates the observed order of convergence.
 *
 * @return EXIT_SUCCESS on successful execution.
 *         EXIT_FAILURE or specific error codes if memory allocation or interpolation fails.
 */
int main() {
  const int INTERP_ORDER = (2 * NinterpGHOSTS + 1); // Defines the interpolation order based on ghost zones.
  const int num_resolutions = 3;                    // Number of grid resolutions to test.

  int N_r_arr[num_resolutions];        // Array to hold the number of radial grid points for each resolution.
  REAL h_arr[num_resolutions];         // Array to store grid spacings for convergence analysis.
  REAL error_L2_norm[num_resolutions]; // Array to store L2 norms of interpolation errors.

  // Define grid resolutions for the radial direction.
  N_r_arr[0] = 16;
  N_r_arr[1] = 32;
  N_r_arr[2] = 64;

  // Fixed resolutions for theta and phi directions.
  const int N_theta = 640;
  const int N_phi = 1280;

  // Iterate over each grid resolution to perform interpolation and error analysis.
  for (int res = 0; res < num_resolutions; res++) {
    int N_r = N_r_arr[res]; // Current resolution in radial direction.

    // Calculate grid spacing based on current resolution.
    h_arr[res] = (N_r > 0) ? (RADIAL_EXTENT / N_r) : 0.0; // Radial domain [0, RADIAL_EXTENT].

    // Determine the total number of grid points including ghost zones.
    int src_Nxx_plus_2NGHOSTS[3];
    src_Nxx_plus_2NGHOSTS[0] = N_r + 2 * NinterpGHOSTS;
    src_Nxx_plus_2NGHOSTS[1] = N_theta + 2 * NGHOSTS;
    src_Nxx_plus_2NGHOSTS[2] = N_phi + 2 * NGHOSTS;

    // Initialize coordinate arrays for the source grid.
    REAL *src_r_theta_phi[3] = {NULL, NULL, NULL};
    REAL src_dxx[3]; // Array to store grid spacings in r, theta, phi.

    initialize_coordinates(N_r, N_theta, N_phi, src_r_theta_phi, src_dxx, src_Nxx_plus_2NGHOSTS); // Set up grid coordinates.

    // Define destination grid parameters (same as source for theta and phi).
    params_struct params;
    params.Nxx0 = 1; // Only one radial point at i = NGHOSTS.
    params.Nxx1 = N_theta;
    params.Nxx2 = N_phi;
    params.Nxx_plus_2NGHOSTS0 = 1 + 2 * NGHOSTS; // Only one radial point.
    params.Nxx_plus_2NGHOSTS1 = N_theta + 2 * NGHOSTS;
    params.Nxx_plus_2NGHOSTS2 = N_phi + 2 * NGHOSTS;

    // Allocate memory for the source grid function data.
    int total_src_grid_points = src_Nxx_plus_2NGHOSTS[0] * src_Nxx_plus_2NGHOSTS[1] * src_Nxx_plus_2NGHOSTS[2];
    REAL *src_gf = (REAL *)malloc(sizeof(REAL) * NUM_INTERP_SRC_GFS * total_src_grid_points);
    if (src_gf == NULL) {
      fprintf(stderr, "Memory allocation failed for source grid function.\n");
      return EXIT_FAILURE;
    }

    // Initialize the source grid function using the analytic function.
    initialize_src_gf(src_Nxx_plus_2NGHOSTS, src_r_theta_phi, src_gf);

    // Prepare common data structure required by the interpolation function.
    commondata_struct commondata;
    commondata.interp_src_Nxx0 = N_r;
    commondata.interp_src_Nxx_plus_2NGHOSTS0 = src_Nxx_plus_2NGHOSTS[0];
    commondata.interp_src_Nxx_plus_2NGHOSTS1 = src_Nxx_plus_2NGHOSTS[1];
    commondata.interp_src_Nxx_plus_2NGHOSTS2 = src_Nxx_plus_2NGHOSTS[2];
    commondata.interp_src_r_theta_phi[0] = src_r_theta_phi[0];
    commondata.interp_src_r_theta_phi[1] = src_r_theta_phi[1];
    commondata.interp_src_r_theta_phi[2] = src_r_theta_phi[2];
    commondata.interp_src_invdxx0 = 1.0 / src_dxx[0];
    commondata.interp_src_dxx0 = src_dxx[0];
    commondata.interp_src_gfs = src_gf;

    // Allocate memory for destination radii grid function and interpolated grid functions.
    int dst_Nxx_plus_2NGHOSTS0 = 1 + 2 * NGHOSTS; // Only one radial point.
    int dst_Nxx_plus_2NGHOSTS1 = N_theta + 2 * NGHOSTS;
    int dst_Nxx_plus_2NGHOSTS2 = N_phi + 2 * NGHOSTS;
    int total_dst_grid_points = dst_Nxx_plus_2NGHOSTS0 * dst_Nxx_plus_2NGHOSTS1 * dst_Nxx_plus_2NGHOSTS2;

    // Define a safe domain within the source grid to ensure all destination points lie within bounds.
    REAL r_min_safe = src_r_theta_phi[0][NinterpGHOSTS] + 1e-6;
    REAL r_max_safe = src_r_theta_phi[0][src_Nxx_plus_2NGHOSTS[0] - NinterpGHOSTS - 1] - 1e-6;

    // Destination grid has the same number of gridpoints as the src grid.
    REAL *dst_radii = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS[0] * src_Nxx_plus_2NGHOSTS[1] * src_Nxx_plus_2NGHOSTS[2]);
    REAL *dst_interp_gfs =
        (REAL *)malloc(sizeof(REAL) * NUM_INTERP_SRC_GFS * src_Nxx_plus_2NGHOSTS[0] * src_Nxx_plus_2NGHOSTS[1] * src_Nxx_plus_2NGHOSTS[2]);
    if (dst_radii == NULL || dst_interp_gfs == NULL) {
      fprintf(stderr, "Memory allocation failed for destination radii.\n");
      return EXIT_FAILURE;
    }

    // Assign destination radii with random values between r_min_safe and r_max_safe.
    for (int j = NGHOSTS; j < N_theta + NGHOSTS; j++) {
      for (int k = NGHOSTS; k < N_phi + NGHOSTS; k++) {
        // Generate a random value between r_min_safe and r_max_safe
        dst_radii[DST_IDX3(NGHOSTS, j, k)] = r_min_safe + ((REAL)rand() / (REAL)RAND_MAX) * (r_max_safe - r_min_safe);
      }
    }

    // Perform the interpolation.
    int error_code = bah_interpolation_1d_radial_spokes_on_3d_src_grid(&params, &commondata, dst_radii, dst_interp_gfs);

    // Check for any interpolation errors and handle them appropriately.
    if (error_code != BHAHAHA_SUCCESS) {
      fprintf(stderr, "Interpolation failed with error code: %d\n", error_code);
      return error_code; // Exit with the specific error code encountered.
    }

    // Calculate the L2 norm of the interpolation error to assess accuracy.
    REAL error_sum = 0.0;

    // Compute the exact solution and error at each (theta, phi) point.
#pragma omp parallel for reduction(+ : error_sum)
    for (int k = NGHOSTS; k < N_phi + NGHOSTS; k++) {
      const REAL phi = src_r_theta_phi[2][k];
      const REAL cosphi = cos(phi);
      for (int j = NGHOSTS; j < N_theta + NGHOSTS; j++) {
        const REAL theta = src_r_theta_phi[1][j];
        const REAL sintheta = sin(theta);
        const REAL r_dst = dst_radii[DST_IDX3(NGHOSTS, j, k)];
        const REAL r_func = sin(r_dst) * r_dst * r_dst;

        REAL exact_value = r_func * sintheta * cosphi;
        // Only need to check the zeroth gridfunction; all gridfunctions are the same.
        REAL interp_value = dst_interp_gfs[DST_IDX4(0, NGHOSTS, j, k)];
        REAL error = interp_value - exact_value;
        error_sum += error * error;
      }
    }

    int num_points = N_theta * N_phi;                  // Total number of points where error is computed.
    error_L2_norm[res] = sqrt(error_sum / num_points); // Store the L2 norm for the current resolution.

    // Output the interpolation error for the current grid resolution.
    printf("Resolution %d: N_r = %d, h = %.5e, L2 error = %.5e\n", res, N_r, h_arr[res], error_L2_norm[res]);
  } // END LOOP: Iterate over all grid resolutions.

  // Analyze and report the observed order of convergence between successive resolutions.
  for (int res = 1; res < num_resolutions; res++) {
    REAL observed_order = log(error_L2_norm[res - 1] / error_L2_norm[res]) / log(h_arr[res - 1] / h_arr[res]);
    printf("Observed order of convergence between resolutions %d and %d: %.2f\n", res - 1, res, observed_order);
  }

  // Inform the expected order of convergence based on the interpolation order.
  printf("Expected order of convergence: %d\n", INTERP_ORDER);

  return EXIT_SUCCESS; // Indicate successful execution.
} // END FUNCTION: main.

#endif // STANDALONE
