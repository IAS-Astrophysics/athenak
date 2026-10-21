#include "intrinsics/simd_intrinsics.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#ifndef REAL
#define REAL double
#endif
#define DEBUG

#ifdef STANDALONE
// Remove the bah_ prefix if compiling as a standalone code, as this function goes by other names in other codes.
#define bah_interpolation_2d_general__uniform_src_grid interpolation_2d_general__uniform_src_grid
#endif

// In case this code is compiled as C++:
#ifdef __cplusplus
#ifndef restrict
#define restrict __restrict__
#endif
#endif
//===============================================
// BHaHAHA Error Handling
//===============================================
// Error codes, set in error_message.py
typedef enum {
  BHAHAHA_SUCCESS,
  FIND_HORIZON_GETTIMEOFDAY_BROKEN,
  FIND_HORIZON_MAX_ITERATIONS_EXCEEDED,
  FIND_HORIZON_HORIZON_TOO_SMALL,
  BCSTRUCT_EIGENCOORD_FAILURE,
  BCSTRUCT_SET_PARITY_ERROR,
  INITIAL_DATA_MALLOC_ERROR,
  NUMGRID_EXTERN_MALLOC_ERROR_GFS,
  NUMGRID_EXTERN_MALLOC_ERROR_RTHETAPHI,
  NUMGRID_INTERP_MALLOC_ERROR_GFS,
  NUMGRID_INTERP_MALLOC_ERROR_RTHETAPHI,
  INTERP1D_NULL_PTRS,
  INTERP1D_INTERP_ORDER_GT_NXX_PLUS_2NINTERPGHOSTS0,
  INTERP1D_HORIZON_TOO_LARGE,
  INTERP1D_HORIZON_TOO_SMALL,
  INTERP2D_EXT_TO_INTERPSRC_NULL_PTRS,
  INTERP2D_EXT_TO_INTERPSRC_INTERP_ORDER_GT_NXX_PLUS_2NINTERPGHOSTS12,
  INTERP2D_EXT_TO_INTERPSRC_HORIZON_OUT_OF_BOUNDS,
  INTERP2D_GENERAL_NULL_PTRS,
  INTERP2D_GENERAL_INTERP_ORDER_GT_NXX_PLUS_2NGHOSTS12,
  INTERP2D_GENERAL_HORIZON_OUT_OF_BOUNDS,
  DIAG_PROPER_CIRCUM_MALLOC_ERROR,
} bhahaha_error_codes;
#pragma GCC optimize("unroll-loops")
/**
 *
 * Performs 2D Lagrange interpolation on a uniform source grid.
 *
 * This function interpolates a scalar grid function from a uniform source grid defined in theta and phi
 * to a set of arbitrary destination points using Lagrange interpolation of a specified order.
 *
 * @param N_interp_GHOSTS          Number of ghost zones around each source point. Determines the interpolation order as 2 * N_interp_GHOSTS + 1.
 * @param src_dxx1                 Grid spacing in the theta direction on the source grid.
 * @param src_dxx2                 Grid spacing in the phi direction on the source grid.
 * @param src_Nxx_plus_2NGHOSTS1  Total number of grid points in the theta direction, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS2  Total number of grid points in the phi direction, including ghost zones.
 * @param src_r_theta_phi          Arrays containing coordinate values for r, theta, and phi on the source grid.
 * @param src_gf                   Pointer to the source grid function data, organized as a flattened 2D array.
 * @param num_dst_pts              Number of destination points where interpolation is performed.
 * @param dst_pts                  Array of destination points' coordinates, each consisting of (theta, phi).
 * @param dst_data                 Output array to store interpolated values at each destination point.
 *
 * @return                         BHAHAHA_SUCCESS on successful interpolation.
 * Appropriate error code if an error is encountered.
 *
 * @note
 * - Assumes that the source and destination grids are uniform in theta and phi directions.
 * - Ensures that destination points lie within the bounds of the source grid to prevent memory access violations.
 *
 */
int bah_interpolation_2d_general__uniform_src_grid(const int N_interp_GHOSTS, const REAL src_dxx1, const REAL src_dxx2,
                                                   const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2,
                                                   REAL *restrict src_r_theta_phi[3], const REAL *restrict src_gf, const int num_dst_pts,
                                                   const REAL dst_pts[][2], REAL *restrict dst_data) {

  // Define the interpolation order based on the number of ghost zones.
  const int INTERP_ORDER = (2 * N_interp_GHOSTS + 1); // Interpolation order corresponds to the number of points in the stencil per dimension.

  // Calculate inverse grid spacings for efficient index calculations.
  const REAL src_invdxx1 = 1.0 / src_dxx1;
  const REAL src_invdxx2 = 1.0 / src_dxx2;

  // Compute normalization factor to scale the interpolated result appropriately.
  const REAL src_invdxx12_INTERP_ORDERm1 = pow(src_dxx1 * src_dxx2, -(INTERP_ORDER - 1));

  // Validate input pointers to prevent segmentation faults.
  if (src_r_theta_phi[1] == NULL || src_r_theta_phi[2] == NULL || src_gf == NULL || dst_data == NULL)
    return INTERP2D_GENERAL_NULL_PTRS; // Exit if any required pointer is NULL.

  // Ensure that the interpolation order does not exceed the grid dimensions in either direction.
  if (INTERP_ORDER > src_Nxx_plus_2NGHOSTS1 || INTERP_ORDER > src_Nxx_plus_2NGHOSTS2)
    return INTERP2D_GENERAL_INTERP_ORDER_GT_NXX_PLUS_2NGHOSTS12; // Exit if interpolation order is too high.

  // Precompute inverse denominators for Lagrange interpolation coefficients to reduce redundant calculations.
  REAL inv_denom[INTERP_ORDER];
  for (int i = 0; i < INTERP_ORDER; i++) {
    REAL denom = 1.0;
    for (int j = 0; j < i; j++)
      denom *= (REAL)(i - j);
    for (int j = i + 1; j < INTERP_ORDER; j++)
      denom *= (REAL)(i - j);
    inv_denom[i] = 1.0 / denom; // Store the inverse to avoid repeated division operations.
  } // END LOOP: Precompute inverse denominators.

  // Define the minimum coordinate values including ghost zones for both theta and phi.
  const REAL xxmin_incl_ghosts1 = src_r_theta_phi[1][0];
  const REAL xxmin_incl_ghosts2 = src_r_theta_phi[2][0];

  // Initialize the error flag to track any interpolation issues.
  int error_flag = BHAHAHA_SUCCESS;

#pragma omp parallel for
  for (int dst_pt = 0; dst_pt < num_dst_pts; dst_pt++) {
    const REAL theta_dst = dst_pts[dst_pt][0]; // Destination point's theta coordinate.
    const REAL phi_dst = dst_pts[dst_pt][1];   // Destination point's phi coordinate.

    // Determine the central grid indices in theta and phi for the interpolation stencil.
    const int idx_center_th = (int)((theta_dst - xxmin_incl_ghosts1) * src_invdxx1 + 0.5);
    const int idx_center_ph = (int)((phi_dst - xxmin_incl_ghosts2) * src_invdxx2 + 0.5);

    {
      // Verify that the interpolation stencil does not exceed grid boundaries.
      if ((idx_center_th - N_interp_GHOSTS < 0) || (idx_center_th + N_interp_GHOSTS >= src_Nxx_plus_2NGHOSTS1) ||
          (idx_center_ph - N_interp_GHOSTS < 0) || (idx_center_ph + N_interp_GHOSTS >= src_Nxx_plus_2NGHOSTS2)) {
#ifdef DEBUG
        // Provide detailed error messages in debug mode for easier troubleshooting.
        fprintf(stderr, "ERROR: Interpolation stencil exceeds grid boundaries. %d %d %d %d\n", (idx_center_th - N_interp_GHOSTS < 0),
                (idx_center_th + N_interp_GHOSTS >= src_Nxx_plus_2NGHOSTS1), (idx_center_ph - N_interp_GHOSTS < 0),
                (idx_center_ph + N_interp_GHOSTS >= src_Nxx_plus_2NGHOSTS2));
        fprintf(stderr, "(th_dst, ph_dst) = (%.6f, %.6f). (dst_pt) = %d.\n", theta_dst, phi_dst, dst_pt);
        fprintf(stderr, "Grid bounds along theta direction: [0, %d], stencil indices: [%d, %d]\n", src_Nxx_plus_2NGHOSTS1 - 1,
                idx_center_th - N_interp_GHOSTS, idx_center_th + N_interp_GHOSTS);
        fprintf(stderr, "Grid bounds along phi direction: [0, %d], stencil indices: [%d, %d]\n", src_Nxx_plus_2NGHOSTS2 - 1,
                idx_center_ph - N_interp_GHOSTS, idx_center_ph + N_interp_GHOSTS);
        fprintf(stderr, "Ensure that the destination point is within grid bounds or adjust the interpolation stencil.\n");
#endif // DEBUG
#pragma omp critical
        {
          error_flag = INTERP2D_GENERAL_HORIZON_OUT_OF_BOUNDS; // Set error flag if stencil is out of bounds.
        }
        continue; // Skip interpolation for this destination point to prevent invalid memory access.
      } // END IF: Check stencil boundaries.

      // Additional sanity checks to ensure central index is correctly positioned.
#ifdef DEBUG
      const REAL TOLERANCE = 1e-13; // Tolerance to account for floating-point precision.
      if (fabs(src_r_theta_phi[1][idx_center_th] - theta_dst) > src_dxx1 * (0.5 + TOLERANCE)) {
        fprintf(stderr, "ERROR: theta center index too far from destination point! %.15e > %.15e\n",
                fabs(src_r_theta_phi[1][idx_center_th] - theta_dst), src_dxx1 * (0.5 + TOLERANCE));
      }
      if (fabs(src_r_theta_phi[2][idx_center_ph] - phi_dst) > src_dxx2 * (0.5 + TOLERANCE)) {
        fprintf(stderr, "ERROR: phi center index too far from destination point! %.15e > %.15e\n", fabs(src_r_theta_phi[2][idx_center_ph] - phi_dst),
                src_dxx2 * (0.5 + TOLERANCE));
      } // END IF: Central index is properly centered
#endif // DEBUG
    } // END SANITY CHECKS: Ensure stencil is valid and central index is correct.

    // Calculate the starting indices for the interpolation stencil in theta and phi directions.
    const int base_idx_th = idx_center_th - N_interp_GHOSTS;
    const int base_idx_ph = idx_center_ph - N_interp_GHOSTS;

    // Initialize arrays to store Lagrange interpolation coefficients and differences for theta.
    REAL coeff_th[INTERP_ORDER];
    REAL diffs_th[INTERP_ORDER];

    // Step 1: Precompute differences between destination theta and source grid theta points within the stencil.
#pragma omp simd
    for (int j = 0; j < INTERP_ORDER; j++) {
      diffs_th[j] = theta_dst - src_r_theta_phi[1][base_idx_th + j];
    } // END LOOP: Compute theta differences.

    // Step 2: Compute Lagrange basis coefficients for theta using precomputed inverse denominators.
#pragma omp simd
    for (int i = 0; i < INTERP_ORDER; i++) {
      REAL numer_i = 1.0;
      // Compute product for j < i (scalar loop).
      for (int j = 0; j < i; j++) {
        numer_i *= diffs_th[j];
      } // END LOOP: Product for j < i
      // Compute product for j > i (scalar loop).
      for (int j = i + 1; j < INTERP_ORDER; j++) {
        numer_i *= diffs_th[j];
      } // END LOOP: Product for j > i
      coeff_th[i] = numer_i * inv_denom[i]; // Scale by the inverse denominator.
    } // END LOOP: Compute theta coefficients.

    // Initialize arrays to store Lagrange interpolation coefficients and differences for phi.
    REAL coeff_ph[INTERP_ORDER];
    REAL diffs_ph[INTERP_ORDER];

    // Step 1: Precompute differences between destination phi and source grid phi points within the stencil.
#pragma omp simd
    for (int j = 0; j < INTERP_ORDER; j++) {
      diffs_ph[j] = phi_dst - src_r_theta_phi[2][base_idx_ph + j];
    } // END LOOP: Compute phi differences.

    // Step 2: Compute Lagrange basis coefficients for phi using precomputed inverse denominators.
#pragma omp simd
    for (int i = 0; i < INTERP_ORDER; i++) {
      REAL numer_i = 1.0;
      // Compute product for j < i (scalar loop).
      for (int j = 0; j < i; j++) {
        numer_i *= diffs_ph[j];
      } // END LOOP: Product for j < i
      // Compute product for j > i (scalar loop).
      for (int j = i + 1; j < INTERP_ORDER; j++) {
        numer_i *= diffs_ph[j];
      } // END LOOP: Product for j > i
      coeff_ph[i] = numer_i * inv_denom[i]; // Scale by the inverse denominator.
    } // END LOOP: Compute phi coefficients.

    // Precompute combined Lagrange coefficients to reduce computational overhead.
    REAL coeff_2d[INTERP_ORDER][INTERP_ORDER];
    for (int iph = 0; iph < INTERP_ORDER; iph++) {
      const REAL coeff_ph_i = coeff_ph[iph];
      for (int ith = 0; ith < INTERP_ORDER; ith++) {
        coeff_2d[iph][ith] = coeff_th[ith] * coeff_ph_i; // Combine theta and phi coefficients.
      }
    } // END LOOP: Combine theta and phi coefficients.

// Define a macro to calculate the flattened index for accessing the source grid function.
#define SRC_IDX2(j, k) ((j) + src_Nxx_plus_2NGHOSTS1 * (k))
    REAL sum = 0.0; // Initialize the accumulator for the interpolated value.

    // Perform the 2D Lagrange interpolation by summing over the stencil.
    for (int iph = 0; iph < INTERP_ORDER; iph++) {
      const int idx_ph = base_idx_ph + iph;
      const int base_offset = base_idx_th + src_Nxx_plus_2NGHOSTS1 * idx_ph;

      int ith = 0;
      REAL_SIMD_ARRAY vec_sum = SetZeroSIMD; // Initialize vector sum to zero

      // Vectorized loop using SIMD with FMA (Fused Multiply-Add), if available
      for (; ith <= INTERP_ORDER - simd_width; ith += simd_width) { // Process simd_width doubles at a time
        // Calculate the flat index for the current set of ith
        const int current_idx_th = base_offset + ith;

        // Load simd_width elements from src_gf and coeff_2d
        REAL_SIMD_ARRAY vec_src = ReadSIMD(&src_gf[current_idx_th]);
        REAL_SIMD_ARRAY vec_coeff = ReadSIMD(&coeff_2d[iph][ith]); // Corrected indexing

        // Use FMA to multiply src and coeff and add to vec_sum
        vec_sum = FusedMulAddSIMD(vec_src, vec_coeff, vec_sum);
      }

      // Horizontally add the elements of the vector sum and accumulate into the scalar sum
      sum += HorizAddSIMD(vec_sum);

      // Handle remaining elements that don't fit into a full SIMD register
      for (; ith < INTERP_ORDER; ith++) {
        const int current_idx_th = base_offset + ith;
        sum += src_gf[current_idx_th] * coeff_2d[iph][ith]; // Corrected indexing
      }
    } // END LOOP: Iterate over phi stencil points.

    // Scale the interpolated sum by the normalization factor and store the result.
    dst_data[dst_pt] = sum * src_invdxx12_INTERP_ORDERm1;
  } // END LOOP: Interpolate all destination points.

  return error_flag; // Return the status of the interpolation process.
} // END FUNCTION bah_interpolation_2d_general__uniform_src_grid

#pragma GCC reset_options // Reset compiler optimizations after the function.

#ifdef STANDALONE

// Define the number of grid functions to interpolate as a compile-time constant.
#define NUM_INTERP_GFS 1 // Number of grid functions to interpolate.

/**
 * Analytic function used for initializing the source grid function.
 *
 * Calculates the product of sine(theta) and cosine(phi).
 *
 * @param theta  Angle in the theta direction.
 * @param phi    Angle in the phi direction.
 *
 * @return       The computed value of sin(theta) * cos(phi).
 */
static inline REAL analytic_function(REAL theta, REAL phi) { return sin(theta) * cos(phi); }

/**
 * Initializes the coordinate arrays for the source grid.
 *
 * Allocates and sets up the theta and phi coordinate values based on the number of ghost zones and grid dimensions.
 *
 * @param N_interp_GHOSTS         Number of ghost zones around each grid point.
 * @param N_theta                 Number of grid points in the theta direction.
 * @param N_phi                   Number of grid points in the phi direction.
 * @param src_r_theta_phi         Arrays to store coordinate values for r, theta, and phi.
 * @param src_dxx1                Pointer to store the grid spacing in the theta direction.
 * @param src_dxx2                Pointer to store the grid spacing in the phi direction.
 * @param src_Nxx_plus_2NGHOSTS1 Total number of grid points in the theta direction, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS2 Total number of grid points in the phi direction, including ghost zones.
 */
void initialize_coordinates(const int N_interp_GHOSTS, const int N_theta, const int N_phi, REAL *src_r_theta_phi[3], REAL *src_dxx1, REAL *src_dxx2,
                            const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2) {
  // Calculate grid spacings based on the number of grid points.
  *src_dxx1 = (M_PI) / N_theta;     // Grid spacing in the theta direction.
  *src_dxx2 = (2.0 * M_PI) / N_phi; // Grid spacing in the phi direction.

  // Allocate memory for theta and phi coordinate arrays.
  src_r_theta_phi[1] = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS1);
  src_r_theta_phi[2] = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS2);

  // Populate the theta coordinate array with uniformly spaced values.
  for (int i = 0; i < src_Nxx_plus_2NGHOSTS1; i++) {
    src_r_theta_phi[1][i] = (i - N_interp_GHOSTS) * (*src_dxx1);
  } // END LOOP: Populate theta coordinates.

  // Populate the phi coordinate array with uniformly spaced values.
  for (int i = 0; i < src_Nxx_plus_2NGHOSTS2; i++) {
    src_r_theta_phi[2][i] = (i - N_interp_GHOSTS) * (*src_dxx2);
  } // END LOOP: Populate phi coordinates.
} // END FUNCTION: initialize_coordinates.

/**
 * Initializes the source grid function using a provided analytic function.
 *
 * Evaluates the analytic function at each grid point and stores the result in the source grid function array.
 *
 * @param src_Nxx_plus_2NGHOSTS1 Total number of grid points in the theta direction, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS2 Total number of grid points in the phi direction, including ghost zones.
 * @param src_r_theta_phi         Arrays containing coordinate values for r, theta, and phi.
 * @param src_gf                  Array to store the initialized source grid function values.
 * @param func                    Pointer to the analytic function used for initialization.
 */
void initialize_src_gf(const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2, REAL *src_r_theta_phi[3], REAL *src_gf,
                       REAL (*func)(REAL, REAL)) {
  // Iterate over each grid point in the phi direction.
  for (int i2 = 0; i2 < src_Nxx_plus_2NGHOSTS2; i2++) {
    // Iterate over each grid point in the theta direction.
    for (int i1 = 0; i1 < src_Nxx_plus_2NGHOSTS1; i1++) {
      const int idx = i1 + src_Nxx_plus_2NGHOSTS1 * i2;                   // Calculate the flattened index for the grid function array.
      src_gf[idx] = func(src_r_theta_phi[1][i1], src_r_theta_phi[2][i2]); // Evaluate and store the analytic function value.
    } // END LOOP: Iterate over theta grid points.
  } // END LOOP: Iterate over phi grid points.
} // END FUNCTION: initialize_src_gf.

/**
 * Main function to execute the standalone 2D interpolation and perform convergence validation tests.
 *
 * Sets up different grid resolutions, generates random destination points, performs interpolation,
 * computes the L2 norm of interpolation errors, and evaluates the observed order of convergence.
 *
 * @return EXIT_SUCCESS on successful execution.
 *         EXIT_FAILURE or specific error codes if memory allocation or interpolation fails.
 */
int main() {
  const int N_interp_GHOSTS = 4;                    // Number of ghost zones for 9th order interpolation.
  const int INTERP_ORDER = 2 * N_interp_GHOSTS + 1; // Defines the interpolation order based on ghost zones.
  const int num_resolutions = 3;                    // Number of grid resolutions to test.
  const int num_dst_pts = 10000000;                 // Number of destination points for interpolation.

  int N_theta_arr[num_resolutions];    // Array to hold the number of theta grid points for each resolution.
  int N_phi_arr[num_resolutions];      // Array to hold the number of phi grid points for each resolution.
  REAL h_arr[num_resolutions];         // Array to store grid spacings for convergence analysis.
  REAL error_L2_norm[num_resolutions]; // Array to store L2 norms of interpolation errors.

  // Define grid resolutions for theta and phi directions.
  N_theta_arr[0] = 16;
  N_phi_arr[0] = 16;

  N_theta_arr[1] = 32;
  N_phi_arr[1] = 32;

  N_theta_arr[2] = 64;
  N_phi_arr[2] = 64;

  // Allocate memory for destination points coordinates.
  REAL(*dst_pts)[2] = (REAL(*)[2])malloc(sizeof(REAL) * num_dst_pts * 2);
  if (dst_pts == NULL) {
    fprintf(stderr, "Memory allocation failed for destination points.\n");
    return EXIT_FAILURE; // EXIT_FAILURE indicates unsuccessful execution due to memory allocation error.
  }

  // Allocate memory for exact solution values at destination points.
  REAL *f_exact = (REAL *)malloc(sizeof(REAL) * num_dst_pts);
  if (f_exact == NULL) {
    fprintf(stderr, "Memory allocation failed for exact solution array.\n");
    return EXIT_FAILURE;
  }

  // Iterate over each grid resolution to perform interpolation and error analysis.
  for (int res = 0; res < num_resolutions; res++) {
    int N_theta = N_theta_arr[res]; // Current resolution in theta direction.
    int N_phi = N_phi_arr[res];     // Current resolution in phi direction.

    // Calculate grid spacing based on current resolution.
    h_arr[res] = (N_theta > 0) ? ((REAL)(M_PI) / N_theta) : 0.0; // Grid spacing in theta.

    // Determine the total number of grid points including ghost zones.
    int src_Nxx_plus_2NGHOSTS1 = N_theta + 2 * N_interp_GHOSTS;
    int src_Nxx_plus_2NGHOSTS2 = N_phi + 2 * N_interp_GHOSTS;

    // Initialize coordinate arrays for the source grid.
    REAL *src_r_theta_phi[3] = {NULL, NULL, NULL}; // r coordinate is unused in 2D interpolation.
    REAL src_dxx1_val, src_dxx2_val;               // Variables to store grid spacings in theta and phi.

    initialize_coordinates(N_interp_GHOSTS, N_theta, N_phi, src_r_theta_phi, &src_dxx1_val, &src_dxx2_val, src_Nxx_plus_2NGHOSTS1,
                           src_Nxx_plus_2NGHOSTS2); // Set up grid coordinates.

    // Define a safe domain within the source grid to ensure all destination points lie within bounds.
    REAL theta_min_safe = src_r_theta_phi[1][N_interp_GHOSTS] + 1e-6;
    REAL theta_max_safe = src_r_theta_phi[1][src_Nxx_plus_2NGHOSTS1 - N_interp_GHOSTS - 1] - 1e-6;
    REAL phi_min_safe = src_r_theta_phi[2][N_interp_GHOSTS] + 1e-6;
    REAL phi_max_safe = src_r_theta_phi[2][src_Nxx_plus_2NGHOSTS2 - N_interp_GHOSTS - 1] - 1e-6;

    // Seed the random number generator to ensure reproducibility across runs.
    srand(42 + res); // Different seed for each resolution.

    // Generate random destination points within the safe domain and compute exact function values.
    for (int i = 0; i < num_dst_pts; i++) {
      REAL theta = theta_min_safe + ((REAL)rand() / RAND_MAX) * (theta_max_safe - theta_min_safe);
      REAL phi = phi_min_safe + ((REAL)rand() / RAND_MAX) * (phi_max_safe - phi_min_safe);
      dst_pts[i][0] = theta;
      dst_pts[i][1] = phi;
      f_exact[i] = analytic_function(theta, phi); // Compute the exact function value for comparison.
    }

    // Allocate memory for the source grid function data.
    REAL *src_gf = (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS1 * src_Nxx_plus_2NGHOSTS2);
    if (src_gf == NULL) {
      fprintf(stderr, "Memory allocation failed for source grid function.\n");
      return EXIT_FAILURE;
    }

    // Initialize the source grid function using the analytic function.
    initialize_src_gf(src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, src_r_theta_phi, src_gf, analytic_function);

    // Allocate memory for storing interpolated data at destination points.
    REAL *dst_data = (REAL *)malloc(sizeof(REAL) * num_dst_pts);
    if (dst_data == NULL) {
      fprintf(stderr, "Memory allocation failed for interpolated data.\n");
      return EXIT_FAILURE;
    }

    // Perform the interpolation for all destination points.
    int error_code = bah_interpolation_2d_general__uniform_src_grid(N_interp_GHOSTS, src_dxx1_val, src_dxx2_val, src_Nxx_plus_2NGHOSTS1,
                                                                    src_Nxx_plus_2NGHOSTS2, src_r_theta_phi, src_gf, num_dst_pts, dst_pts, dst_data);

    // Check for any interpolation errors and handle them appropriately.
    if (error_code != BHAHAHA_SUCCESS) {
      fprintf(stderr, "Interpolation failed with error code: %d\n", error_code);
      return error_code; // Exit with the specific error code encountered.
    }

    // Calculate the L2 norm of the interpolation error to assess accuracy.
    REAL error_sum = 0.0;
    for (int i = 0; i < num_dst_pts; i++) {
      REAL error = dst_data[i] - f_exact[i];
      error_sum += error * error;
    }
    error_L2_norm[res] = sqrt(error_sum / num_dst_pts); // Store the L2 norm for the current resolution.

    // Output the interpolation error for the current grid resolution.
    printf("Resolution %d: N_theta = %d, N_phi = %d, h = %.5e, L2 error = %.5e\n", res, N_theta, N_phi, h_arr[res], error_L2_norm[res]);
  } // END LOOP: Iterate over all grid resolutions.

  // Analyze and report the observed order of convergence between successive resolutions.
  for (int res = 1; res < num_resolutions; res++) {
    REAL observed_order = log(error_L2_norm[res - 1] / error_L2_norm[res]) / log(h_arr[res - 1] / h_arr[res]);
    printf("Observed order of convergence between resolutions %d and %d: %.2f\n", res - 1, res, observed_order);
  }

  // Inform the expected order of convergence based on the interpolation order.
  printf("Expected order of convergence: %d\n", INTERP_ORDER);

  return 0; // Indicate successful execution.
} // END FUNCTION: main.

#endif // STANDALONE
