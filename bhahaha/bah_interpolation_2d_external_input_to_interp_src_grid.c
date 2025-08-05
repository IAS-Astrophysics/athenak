#include "BHaH_defines.h"

#define DEBUG
#define INTERP_ORDER (2 * NinterpGHOSTS + 1) // Interpolation order (== number of points in stencil, in each dimension).
#define DST_IDX4(g, i, j, k) ((i) + dst_Nxx_plus_2NGHOSTS0 * ((j) + dst_Nxx_plus_2NGHOSTS1 * ((k) + dst_Nxx_plus_2NGHOSTS2 * (g))))
#define SRC_IDX4(g, i, j, k) ((i) + src_Nxx_plus_2NGHOSTS0 * ((j) + src_Nxx_plus_2NGHOSTS1 * ((k) + src_Nxx_plus_2NGHOSTS2 * (g))))
// #define STANDALONE

#pragma GCC optimize("unroll-loops")
/**
 *
 * Interpolates data from the source grid (external input) to the destination grid (interp_src) using 2D Lagrange interpolation.
 *
 * Perform 2D interpolation on a uniform grid in the theta and phi directions,
 * at all (perfectly overlapping) radial points on the src and dst grids.
 *
 * @param commondata A structure containing grid parameters and data arrays for both source and destination grids.
 *
 * @return Returns an error code if any pointer is NULL or if the interpolation order exceeds grid boundaries. Returns 0 on success.
 *
 * Notes:
 * - Assumes that theta and phi values are uniform and consistent between source and destination grids.
 * - The stencil size for interpolation is defined by the INTERP_ORDER macro.
 * - The destination and src grid radial points overlap.
 *
 */
int bah_interpolation_2d_external_input_to_interp_src_grid(commondata_struct *restrict commondata) {

  // UNPACK PARAMETERS:
  // src = external_input
  // dst = interp_src
  const REAL src_dxx1 = commondata->external_input_dxx1;
  const REAL src_dxx2 = commondata->external_input_dxx2;
  const REAL src_invdxx1 = commondata->external_input_invdxx1;
  const REAL src_invdxx2 = commondata->external_input_invdxx2;
  const int src_Nxx_plus_2NGHOSTS0 = commondata->external_input_Nxx_plus_2NGHOSTS0;
  const int src_Nxx_plus_2NGHOSTS1 = commondata->external_input_Nxx_plus_2NGHOSTS1;
  const int src_Nxx_plus_2NGHOSTS2 = commondata->external_input_Nxx_plus_2NGHOSTS2;
  const REAL *restrict src_gfs = commondata->external_input_gfs;

  const int dst_Nxx1 = commondata->interp_src_Nxx1;
  const int dst_Nxx2 = commondata->interp_src_Nxx2;
  const int dst_Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int dst_Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int dst_Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;
  REAL *restrict dst_gfs = commondata->interp_src_gfs;

  // Compute (1/(src_dtheta * src_dphi))^(INTERP_ORDER-1) normalization factor, as pow() is expensive.
  // Orig code: params.inv_dxx0_dxx1_ORDERm1 = pow(1.0 / (dxx0 * dxx1), ORDER - 1);
  const REAL src_invdxx12_INTERP_ORDERm1 = pow(commondata->external_input_dxx1 * commondata->external_input_dxx2, -(INTERP_ORDER - 1));

  const REAL *restrict src_r_theta_phi[3] = {commondata->external_input_r_theta_phi[0], commondata->external_input_r_theta_phi[1],
                                             commondata->external_input_r_theta_phi[2]};

  // Perform debug checks if DEBUG is defined
  if (src_r_theta_phi[1] == NULL || src_r_theta_phi[2] == NULL || commondata->external_input_gfs == NULL ||
      commondata->interp_src_r_theta_phi[1] == NULL || commondata->interp_src_r_theta_phi[2] == NULL || commondata->interp_src_gfs == NULL)
    return INTERP2D_EXT_TO_INTERPSRC_NULL_PTRS;

  if (INTERP_ORDER > commondata->external_input_Nxx1 + 2 * NinterpGHOSTS || INTERP_ORDER > commondata->external_input_Nxx2 + 2 * NinterpGHOSTS)
    return INTERP2D_EXT_TO_INTERPSRC_INTERP_ORDER_GT_NXX_PLUS_2NINTERPGHOSTS12;

  // Precompute inverse denominators for Lagrange interpolation coefficients to optimize performance.
  REAL inv_denom[INTERP_ORDER];
  for (int i = 0; i < INTERP_ORDER; i++) {
    REAL denom = 1.0;
    for (int j = 0; j < i; j++)
      denom *= (REAL)(i - j);
    for (int j = i + 1; j < INTERP_ORDER; j++)
      denom *= (REAL)(i - j);
    inv_denom[i] = 1.0 / denom; // Divisions are expensive, so we do them only once.
  } // END LOOP: Precompute inverse denominators.

  // Perform interpolation for each destination angular point (theta, phi)
  const REAL xxmin_incl_ghosts1 = src_r_theta_phi[1][0];
  const REAL xxmin_incl_ghosts2 = src_r_theta_phi[2][0];
  int error_flag = BHAHAHA_SUCCESS;
  int i0_min_shift = 0;
  if (commondata->bhahaha_params_and_data->r_min_external_input == 0)
    i0_min_shift = NGHOSTS;
#pragma omp parallel for
  for (int ir = i0_min_shift; ir < dst_Nxx_plus_2NGHOSTS0; ir++) {
    for (int iphi = NGHOSTS; iphi < dst_Nxx2 + NGHOSTS; iphi++) { // Ignore ghost zones; these can be handled separately.
      const REAL phi_dst = commondata->interp_src_r_theta_phi[2][iphi];
      for (int itheta = NGHOSTS; itheta < dst_Nxx1 + NGHOSTS; itheta++) {
        const REAL theta_dst = commondata->interp_src_r_theta_phi[1][itheta];

        // printf("destination: (%e, %e, %e)\n", commondata->interp_src_r_theta_phi[0][ir], commondata->interp_src_r_theta_phi[1][itheta],
        //        commondata->interp_src_r_theta_phi[2][iphi]);

        // Calculate the central indices for the stencil in the theta & phi directions
        const int idx_center_th = (int)((theta_dst - xxmin_incl_ghosts1) * src_invdxx1 + 0.5);
        const int idx_center_ph = (int)((phi_dst - xxmin_incl_ghosts2) * src_invdxx2 + 0.5);

        {
          // Ensure the stencil is within valid grid bounds
          if ((idx_center_th - NinterpGHOSTS < 0) || (idx_center_th + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS1) ||
              (idx_center_ph - NinterpGHOSTS < 0) || (idx_center_ph + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS2)) {
#ifdef DEBUG
            fprintf(stderr, "ERROR: Interpolation stencil exceeds grid boundaries. %d %d %d %d\n", (idx_center_th - NinterpGHOSTS < 0),
                    (idx_center_th + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS1), (idx_center_ph - NinterpGHOSTS < 0),
                    (idx_center_ph + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS2));
            fprintf(stderr, "(th_dst,ph_dst) = (%.6f,%.6f). (itheta,iphi) = (%d,%d).\n", theta_dst, phi_dst, itheta, iphi);
            fprintf(stderr, "Grid bounds along theta direction: [0, %d], stencil indices: [%d, %d]\n", src_Nxx_plus_2NGHOSTS1 - 1,
                    idx_center_th - NinterpGHOSTS, idx_center_th + NinterpGHOSTS);
            fprintf(stderr, "Grid bounds along phi direction: [0, %d], stencil indices: [%d, %d]\n", src_Nxx_plus_2NGHOSTS2 - 1,
                    idx_center_ph - NinterpGHOSTS, idx_center_ph + NinterpGHOSTS);
            fprintf(stderr, "Ensure that the destination point is within grid bounds or adjust the interpolation stencil.\n");
#endif // DEBUG
#pragma omp critical
            {
              error_flag = INTERP2D_EXT_TO_INTERPSRC_HORIZON_OUT_OF_BOUNDS;
            }
            continue; // Skip further work for this iteration
          } // END IF stencil in bounds

#ifdef DEBUG
          // Verify that the central index is the closest grid point to the destination radius
          const REAL TOLERANCE = 1e-13; // The TOLERANCE is needed, as we often interpolate to points exactly midway between src points.
          if (fabs(src_r_theta_phi[1][idx_center_th] - theta_dst) > src_dxx1 * (0.5 + TOLERANCE)) {
            fprintf(stderr, "ERROR: theta center index too far from destination point! %.15e > %.15e\n",
                    fabs(src_r_theta_phi[1][idx_center_th] - theta_dst), src_dxx1 * (0.5 + TOLERANCE));
          }
          if (fabs(src_r_theta_phi[2][idx_center_ph] - phi_dst) > src_dxx2 * (0.5 + TOLERANCE)) {
            fprintf(stderr, "ERROR: phi center index too far from destination point! %.15e > %.15e\n",
                    fabs(src_r_theta_phi[2][idx_center_ph] - phi_dst), src_dxx2 * (0.5 + TOLERANCE));
          } // END IF central index is properly centered
#endif // DEBUG
        } // END sanity checks

        const int base_idx_th = idx_center_th - NinterpGHOSTS;
        const int base_idx_ph = idx_center_ph - NinterpGHOSTS;

        // THETA:
        REAL coeff_th[INTERP_ORDER];
        REAL diffs_th[INTERP_ORDER];
// Step 1: Precompute all differences (vectorized)
#pragma omp simd
        for (int j = 0; j < INTERP_ORDER; j++) {
          diffs_th[j] = theta_dst - src_r_theta_phi[1][base_idx_th + j];
        } // END LOOP over j
// Step 2: Vectorize the outer loop over i
#pragma omp simd
        for (int i = 0; i < INTERP_ORDER; i++) {
          REAL numer_i = 1.0;
          // Compute product for j < i (scalar loop)
          for (int j = 0; j < i; j++) {
            numer_i *= diffs_th[j];
          } // END LOOP over j < i
          // Compute product for j > i (scalar loop)
          for (int j = i + 1; j < INTERP_ORDER; j++) {
            numer_i *= diffs_th[j];
          } // END LOOP over j > i
          coeff_th[i] = numer_i * inv_denom[i];
        } // END LOOP over i

        // PHI:
        REAL coeff_ph[INTERP_ORDER];
        REAL diffs_ph[INTERP_ORDER];
// Step 1: Precompute all differences (vectorized)
#pragma omp simd
        for (int j = 0; j < INTERP_ORDER; j++) {
          diffs_ph[j] = phi_dst - src_r_theta_phi[2][base_idx_ph + j];
        } // END LOOP over j
// Step 2: Vectorize the outer loop over i
#pragma omp simd
        for (int i = 0; i < INTERP_ORDER; i++) {
          REAL numer_i = 1.0;
          // Compute product for j < i (scalar loop)
          for (int j = 0; j < i; j++) {
            numer_i *= diffs_ph[j];
          } // END LOOP over j < i
          // Compute product for j > i (scalar loop)
          for (int j = i + 1; j < INTERP_ORDER; j++) {
            numer_i *= diffs_ph[j];
          } // END LOOP over j > i
          coeff_ph[i] = numer_i * inv_denom[i];
        } // END LOOP over i

        // Precompute combined Lagrange coefficients to reduce computations
        REAL coeff_2d[INTERP_ORDER][INTERP_ORDER];
        for (int iph = 0; iph < INTERP_ORDER; iph++) {
          const REAL coeff_ph_i = coeff_ph[iph];
          for (int ith = 0; ith < INTERP_ORDER; ith++) {
            coeff_2d[iph][ith] = coeff_ph_i * coeff_th[ith];
          }
        }

        // Perform the 2D Lagrange interpolation, optimizing memory accesses and enabling vectorization
        REAL sum[NUM_EXT_INPUT_CONFORMAL_GFS];
        for (int gf = 0; gf < NUM_EXT_INPUT_CONFORMAL_GFS; gf++) {
          sum[gf] = 0.0;
        }
        for (int iph = 0; iph < INTERP_ORDER; iph++) {
          const int idx_ph = base_idx_ph + iph;
          for (int ith = 0; ith < INTERP_ORDER; ith++) {
            const int idx_th = base_idx_th + ith;
            const REAL coeff = coeff_2d[iph][ith];
#pragma omp simd
            for (int gf = 0; gf < NUM_EXT_INPUT_CONFORMAL_GFS; gf++) {
              sum[gf] += src_gfs[SRC_IDX4(gf, ir, idx_th, idx_ph)] * coeff;
            } // END LOOP over src gridfunctions
          } // END LOOP over theta
        } // END LOOP over phi
        for (int gf = 0; gf < NUM_EXT_INPUT_CONFORMAL_GFS; gf++) {
          dst_gfs[DST_IDX4(gf, ir, itheta, iphi)] = sum[gf] * src_invdxx12_INTERP_ORDERm1;
        } // END LOOP over dst gridfunctions
      } // END LOOP over theta
    } // END LOOP over phi
  } // END LOOP over r
  return error_flag;
} // END FUNCTION bah_interpolation_2d_external_input_to_interp_src_grid

#pragma GCC reset_options // Reset compiler optimizations after the function

#ifdef STANDALONE
/**
 * Analytic function used for initializing the source grid function.
 *
 * Calculates the product of \( r^2 \sin(r) \sin(\theta) \cos(\phi) \).
 *
 * @param r       Radial coordinate.
 * @param theta   Angle in the theta direction.
 * @param phi     Angle in the phi direction.
 *
 * @return        The computed value of \( r^2 \sin(r) \sin(\theta) \cos(\phi) \).
 */
static inline REAL analytic_function(REAL r, REAL theta, REAL phi) { return r * r * sin(r) * sin(theta) * cos(phi); }

/**
 * Initializes the coordinate arrays for a uniform, cell-centered grid.
 *
 * Allocates and sets up the r, theta, and phi coordinate values based on the number of ghost zones and grid dimensions.
 *
 * @param N_r                    Number of grid points in the radial direction (excluding ghost zones).
 * @param N_theta                Number of grid points in the theta direction (excluding ghost zones).
 * @param N_phi                  Number of grid points in the phi direction (excluding ghost zones).
 * @param r_theta_phi            Arrays to store coordinate values for r, theta, and phi.
 * @param dxx0                   Pointer to store the grid spacing in the radial direction.
 * @param dxx1                   Pointer to store the grid spacing in the theta direction.
 * @param dxx2                   Pointer to store the grid spacing in the phi direction.
 * @param Nxx_plus_2NGHOSTS0     Total number of grid points in the radial direction, including ghost zones.
 * @param Nxx_plus_2NGHOSTS1     Total number of grid points in the theta direction, including ghost zones.
 * @param Nxx_plus_2NGHOSTS2     Total number of grid points in the phi direction, including ghost zones.
 */
void initialize_coordinates(const int N_r, const int N_theta, const int N_phi, REAL *r_theta_phi[3], REAL *dxx0, REAL *dxx1, REAL *dxx2,
                            const int Nxx_plus_2NGHOSTS0, const int Nxx_plus_2NGHOSTS1, const int Nxx_plus_2NGHOSTS2) {
  // Calculate grid spacings based on the number of grid points.
  *dxx0 = 1.0 / N_r;          // Grid spacing in the radial direction.
  *dxx1 = M_PI / N_theta;     // Grid spacing in the theta direction.
  *dxx2 = 2.0 * M_PI / N_phi; // Grid spacing in the phi direction.

  // Allocate memory for r, theta, and phi coordinate arrays.
  r_theta_phi[0] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
  r_theta_phi[1] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
  r_theta_phi[2] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS2);

  if (r_theta_phi[0] == NULL || r_theta_phi[1] == NULL || r_theta_phi[2] == NULL) {
    fprintf(stderr, "Memory allocation failed for coordinate arrays.\n");
    exit(EXIT_FAILURE);
  }

  // Populate the r coordinate array with uniformly spaced values.
  for (int i = 0; i < Nxx_plus_2NGHOSTS0; i++) {
    r_theta_phi[0][i] = (i - NGHOSTS + 0.5) * (*dxx0);
  }

  // Populate the theta coordinate array with uniformly spaced, cell-centered values.
  for (int i = 0; i < Nxx_plus_2NGHOSTS1; i++) {
    r_theta_phi[1][i] = (i - NGHOSTS + 0.5) * (*dxx1);
  }

  // Populate the phi coordinate array with uniformly spaced, cell-centered values.
  for (int i = 0; i < Nxx_plus_2NGHOSTS2; i++) {
    r_theta_phi[2][i] = (-M_PI) + (i - NGHOSTS + 0.5) * (*dxx2);
  }
}

/**
 * Initializes the source grid function using a provided analytic function.
 *
 * Sets the grid functions at all radial indices.
 *
 * @param src_Nxx_plus_2NGHOSTS0 Total number of grid points in the radial direction, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS1 Total number of grid points in the theta direction, including ghost zones.
 * @param src_Nxx_plus_2NGHOSTS2 Total number of grid points in the phi direction, including ghost zones.
 * @param r_theta_phi            Arrays containing coordinate values for r, theta, and phi.
 * @param src_gf                 Array to store the initialized source grid function values.
 * @param func                   Pointer to the analytic function used for initialization.
 */
void initialize_src_gf(const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2, REAL *r_theta_phi[3],
                       REAL *src_gf, REAL (*func)(REAL, REAL, REAL)) {
  // Initialize all grid functions to zero.
#pragma omp parallel for
  for (int i = 0; i < src_Nxx_plus_2NGHOSTS0 * src_Nxx_plus_2NGHOSTS1 * src_Nxx_plus_2NGHOSTS2 * NUM_EXT_INPUT_CONFORMAL_GFS; i++) {
    src_gf[i] = 0.0;
  }

  // Set the grid function at all radial indices.
#pragma omp parallel for collapse(4)
  for (int gf = 0; gf < NUM_EXT_INPUT_CONFORMAL_GFS; gf++) {
    for (int i0 = 0; i0 < src_Nxx_plus_2NGHOSTS0; i0++) {
      for (int i1 = 0; i1 < src_Nxx_plus_2NGHOSTS1; i1++) {
        for (int i2 = 0; i2 < src_Nxx_plus_2NGHOSTS2; i2++) {
          int src_idx = SRC_IDX4(gf, i0, i1, i2);
          src_gf[src_idx] = func(r_theta_phi[0][i0], r_theta_phi[1][i1], r_theta_phi[2][i2]);
        }
      }
    }
  }
}

/**
 * Main function to execute the standalone 2D interpolation and perform convergence validation tests.
 *
 * Validates the 2D interpolation by interpolating from a low-resolution source grid to a high-resolution destination grid,
 * comparing the interpolated values against the exact analytic function, and computing the L2 norm of the errors.
 *
 * @return EXIT_SUCCESS on successful execution.
 *         EXIT_FAILURE or specific error codes if memory allocation or interpolation fails.
 */
int main() {
  // Define grid sizes (excluding ghost zones).
  const int N_r = 64; // Number of radial grid points

#define num_resolutions 3
  // Define source grid sizes for each resolution.
  const int Ntheta_src_arr[num_resolutions] = {16, 32, 64};
  const int Nphi_src_arr[num_resolutions] = {32, 64, 128};

  // Define destination grid sizes (fixed as per user instruction).
  const int Ntheta_dst = 320;
  const int Nphi_dst = 640;

  // Define total grid sizes including ghost zones for destination.
  const int dst_Nxx_plus_2NGHOSTS0 = N_r + 2 * NGHOSTS;        // r-direction
  const int dst_Nxx_plus_2NGHOSTS1 = Ntheta_dst + 2 * NGHOSTS; // theta-direction
  const int dst_Nxx_plus_2NGHOSTS2 = Nphi_dst + 2 * NGHOSTS;   // phi-direction

  // Allocate memory for destination grid coordinates.
  REAL *dst_r_theta_phi[3] = {NULL, NULL, NULL};
  REAL dst_dxx0_val, dst_dxx1_val, dst_dxx2_val;

  // Initialize destination grid coordinates once, as they are fixed across resolutions.
  initialize_coordinates(N_r, Ntheta_dst, Nphi_dst, dst_r_theta_phi, &dst_dxx0_val, &dst_dxx1_val, &dst_dxx2_val, dst_Nxx_plus_2NGHOSTS0,
                         dst_Nxx_plus_2NGHOSTS1, dst_Nxx_plus_2NGHOSTS2);

  // Allocate memory for destination grid function data.
  REAL *dst_gfs =
      (REAL *)malloc(sizeof(REAL) * dst_Nxx_plus_2NGHOSTS0 * dst_Nxx_plus_2NGHOSTS1 * dst_Nxx_plus_2NGHOSTS2 * NUM_EXT_INPUT_CONFORMAL_GFS);
  if (dst_gfs == NULL) {
    fprintf(stderr, "Memory allocation failed for destination grid functions.\n");
    return EXIT_FAILURE;
  }

  // Initialize the destination grid functions to zero.
#pragma omp parallel for
  for (int i = 0; i < dst_Nxx_plus_2NGHOSTS0 * dst_Nxx_plus_2NGHOSTS1 * dst_Nxx_plus_2NGHOSTS2 * NUM_EXT_INPUT_CONFORMAL_GFS; i++) {
    dst_gfs[i] = 0.0;
  }

  // Prepare arrays to store L2 errors and grid spacings for each resolution.
  REAL error_L2_norm[num_resolutions];
  REAL h_arr[num_resolutions];

  // Loop over each source grid resolution.
  for (int res = 0; res < num_resolutions; res++) {
    const int Ntheta_src = Ntheta_src_arr[res];
    const int Nphi_src = Nphi_src_arr[res];

    // Define total grid sizes including ghost zones for source.
    const int src_Nxx_plus_2NGHOSTS0 = N_r + 2 * NGHOSTS;        // r-direction
    const int src_Nxx_plus_2NGHOSTS1 = Ntheta_src + 2 * NGHOSTS; // theta-direction
    const int src_Nxx_plus_2NGHOSTS2 = Nphi_src + 2 * NGHOSTS;   // phi-direction

    // Allocate memory for source grid coordinates.
    REAL *src_r_theta_phi[3] = {NULL, NULL, NULL};
    REAL src_dxx0, src_dxx1, src_dxx2;

    // Initialize source grid coordinates.
    initialize_coordinates(N_r, Ntheta_src, Nphi_src, src_r_theta_phi, &src_dxx0, &src_dxx1, &src_dxx2, src_Nxx_plus_2NGHOSTS0,
                           src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2);

    // Allocate memory for source grid function data.
    REAL *src_gf =
        (REAL *)malloc(sizeof(REAL) * src_Nxx_plus_2NGHOSTS0 * src_Nxx_plus_2NGHOSTS1 * src_Nxx_plus_2NGHOSTS2 * NUM_EXT_INPUT_CONFORMAL_GFS);
    if (src_gf == NULL) {
      fprintf(stderr, "Memory allocation failed for source grid function at resolution %d.\n", res);
      return EXIT_FAILURE;
    }

    // Initialize the source grid function using the analytic function.
    initialize_src_gf(src_Nxx_plus_2NGHOSTS0, src_Nxx_plus_2NGHOSTS1, src_Nxx_plus_2NGHOSTS2, src_r_theta_phi, src_gf, analytic_function);

    // Prepare commondata_struct with necessary parameters for interpolation.
    commondata_struct commondata;

    // Assign source grid parameters.
    commondata.bhahaha_params_and_data = malloc(sizeof(bhahaha_params_and_data_struct));
    commondata.bhahaha_params_and_data->r_min_external_input = 0.0;
    commondata.external_input_dxx1 = src_dxx1;
    commondata.external_input_dxx2 = src_dxx2;
    commondata.external_input_invdxx1 = 1.0 / src_dxx1;
    commondata.external_input_invdxx2 = 1.0 / src_dxx2;
    commondata.external_input_Nxx1 = Ntheta_src;
    commondata.external_input_Nxx2 = Nphi_src;
    commondata.external_input_Nxx_plus_2NGHOSTS0 = src_Nxx_plus_2NGHOSTS0;
    commondata.external_input_Nxx_plus_2NGHOSTS1 = src_Nxx_plus_2NGHOSTS1;
    commondata.external_input_Nxx_plus_2NGHOSTS2 = src_Nxx_plus_2NGHOSTS2;
    commondata.external_input_gfs = src_gf;
    commondata.external_input_r_theta_phi[0] = src_r_theta_phi[0];
    commondata.external_input_r_theta_phi[1] = src_r_theta_phi[1];
    commondata.external_input_r_theta_phi[2] = src_r_theta_phi[2];

    // Assign destination grid parameters (fixed across resolutions).
    commondata.interp_src_Nxx1 = Ntheta_dst;
    commondata.interp_src_Nxx2 = Nphi_dst;
    commondata.interp_src_Nxx_plus_2NGHOSTS0 = dst_Nxx_plus_2NGHOSTS0;
    commondata.interp_src_Nxx_plus_2NGHOSTS1 = dst_Nxx_plus_2NGHOSTS1;
    commondata.interp_src_Nxx_plus_2NGHOSTS2 = dst_Nxx_plus_2NGHOSTS2;
    commondata.interp_src_gfs = dst_gfs;
    commondata.interp_src_r_theta_phi[0] = dst_r_theta_phi[0];
    commondata.interp_src_r_theta_phi[1] = dst_r_theta_phi[1];
    commondata.interp_src_r_theta_phi[2] = dst_r_theta_phi[2];

    // Perform the interpolation.
    int error_code = bah_interpolation_2d_external_input_to_interp_src_grid(&commondata);

    // Check for any interpolation errors and handle them appropriately.
    if (error_code != BHAHAHA_SUCCESS) {
      fprintf(stderr, "Interpolation failed with error code: %d at resolution %d.\n", error_code, res);
      continue;
    }

    // Store the grid spacing h for this resolution based on the source grid.
    h_arr[res] = src_dxx1;

    // Calculate the L2 norm of the interpolation error to assess accuracy.
    REAL error_sum = 0.0;
#pragma omp parallel for reduction(+ : error_sum)
    for (int gf = 0; gf < NUM_EXT_INPUT_CONFORMAL_GFS; gf++) {
      for (int i0 = NGHOSTS; i0 < dst_Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
        const REAL r = dst_r_theta_phi[0][i0];
        for (int i1 = NGHOSTS; i1 < dst_Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
          const REAL theta = dst_r_theta_phi[1][i1];
          for (int i2 = NGHOSTS; i2 < dst_Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
            const REAL phi = dst_r_theta_phi[2][i2];
            int idx = DST_IDX4(gf, i0, i1, i2);
            REAL interpolated = dst_gfs[idx];
            REAL exact = analytic_function(r, theta, phi);
            REAL error = interpolated - exact;
            error_sum += error * error;
          }
        }
      }
    }
    REAL L2_error = sqrt(
        error_sum / (dst_Nxx_plus_2NGHOSTS0 * dst_Nxx_plus_2NGHOSTS1 * dst_Nxx_plus_2NGHOSTS2 * NUM_EXT_INPUT_CONFORMAL_GFS)); // Compute the L2 norm.

    // Store the L2 error for this resolution.
    error_L2_norm[res] = L2_error;

    // Output the interpolation error for the current grid resolution.
    printf("Resolution %d: N_theta = %d, N_phi = %d, h = %.5e, L2 error = %.5e\n", res, Ntheta_src, Nphi_src, h_arr[res], L2_error);
  } // END LOOP over resolutions

  // Compute and report the observed order of convergence between successive resolutions.
  for (int res = 1; res < num_resolutions; res++) {
    REAL observed_order = log(error_L2_norm[res - 1] / error_L2_norm[res]) / log(h_arr[res - 1] / h_arr[res]);
    printf("Observed order of convergence between resolutions %d and %d: %.2f\n", res - 1, res, observed_order);
  }
  printf("Expected order of convergence: %d\n", INTERP_ORDER);

  return EXIT_SUCCESS; // Indicate successful execution.
} // END FUNCTION: main.
#endif // STANDALONE
