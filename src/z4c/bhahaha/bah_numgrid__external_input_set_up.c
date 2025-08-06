#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

// Indexing macros
#define EX_IDX4(g, i, j, k)                                                                                                                          \
  ((i) + commondata->external_input_Nxx_plus_2NGHOSTS0 *                                                                                             \
             ((j) + commondata->external_input_Nxx_plus_2NGHOSTS1 * ((k) + commondata->external_input_Nxx_plus_2NGHOSTS2 * (g))))
#define EX_NOGZ_IDX4(g, i, j, k)                                                                                                                     \
  ((i) + bhahaha_params_and_data->Nr_external_input * ((j) + commondata->external_input_Nxx1 * ((k) + commondata->external_input_Nxx2 * (g))))

/**
 * Initializes and processes the external input metric data (gamma_{ij}, K_{ij}).
 *
 * This function performs the following steps:
 * 1. Unpacks input parameters from the common data structure.
 * 2. Adds ghost zones to the external input arrays to facilitate boundary condition application.
 * 3. Allocates memory for the external input gridfunctions with ghost zones and assigns it to the common data structure.
 * 4. Transfers metric data from arrays without ghost zones into the newly allocated arrays with ghost zones.
 * 5. Sets up coordinate arrays for a uniform, cell-centered spherical grid.
 * 6. Transforms the metric components (gamma_{ij}, K_{ij}) from Cartesian to spherical coordinates, including necessary rescaling.
 * 7. Sets up boundary condition structures and applies inner boundary conditions, including parity corrections for all gridfunctions.
 *
 * @param commondata - Pointer to the common data structure containing simulation parameters and data.
 * @param n_resolutions - Number of angular resolutions.
 * @param Ntheta - Array containing the number of theta points for each resolution.
 * @param Nphi - Array containing the number of phi points for each resolution.
 *
 * @return BHAHAHA_SUCCESS on successful setup, or an error code indicating the failure reason.
 *
 */
int bah_numgrid__external_input_set_up(commondata_struct *restrict commondata, const int n_resolutions, const int *restrict Ntheta,
                                       const int *restrict Nphi) {

  // Step 1: Unpack input parameters from the common data structure.
  const bhahaha_params_and_data_struct *restrict bhahaha_params_and_data = commondata->bhahaha_params_and_data;

  // Calculate the number of interior (non-ghost) radial points by subtracting ghost zones.
  // Nr from external includes r ~ r_max NGHOSTS.
  commondata->external_input_Nxx0 = bhahaha_params_and_data->Nr_external_input - NGHOSTS;
  if (bhahaha_params_and_data->r_min_external_input > 0) {
    commondata->external_input_Nxx0 = bhahaha_params_and_data->Nr_external_input - 2 * NGHOSTS;
  }
  int i0_min_shift = 0;
  if (bhahaha_params_and_data->r_min_external_input == 0)
    i0_min_shift = NGHOSTS;

  // Set fixed angular resolutions for theta and phi directions.
  {
    const int max_resolution_i = bhahaha_params_and_data->num_resolutions_multigrid - 1;
    commondata->external_input_Nxx1 = bhahaha_params_and_data->Ntheta_array_multigrid[max_resolution_i];
    commondata->external_input_Nxx2 = bhahaha_params_and_data->Nphi_array_multigrid[max_resolution_i];
  }

  // Step 1.a: Calculate grid spacing in each coordinate direction based on the simulation domain and resolution.
  // x_i = min_i + (j + 0.5) * dx_i, where dx_i = (max_i - min_i) / N_i

  commondata->external_input_dxx0 = bhahaha_params_and_data->dr_external_input;
  commondata->external_input_dxx1 = M_PI / ((REAL)commondata->external_input_Nxx1);
  commondata->external_input_dxx2 = 2 * M_PI / ((REAL)commondata->external_input_Nxx2);

  // Precompute inverse grid spacings for performance optimization in calculations.
  commondata->external_input_invdxx0 = 1.0 / commondata->external_input_dxx0;
  commondata->external_input_invdxx1 = 1.0 / commondata->external_input_dxx1;
  commondata->external_input_invdxx2 = 1.0 / commondata->external_input_dxx2;

  // Step 2: Add ghost zones to the external input data.
  // Ghost zones are added to so that inner boundary conditions may be applied; 2 * NGHOSTS in each angular direction and NGHOSTS in the radial
  // direction.
  commondata->external_input_Nxx_plus_2NGHOSTS0 = commondata->external_input_Nxx0 + 2 * NGHOSTS;
  commondata->external_input_Nxx_plus_2NGHOSTS1 = commondata->external_input_Nxx1 + 2 * NGHOSTS;
  commondata->external_input_Nxx_plus_2NGHOSTS2 = commondata->external_input_Nxx2 + 2 * NGHOSTS;

  // Calculate the total number of grid points including ghost zones.
  const int total_elements_incl_gzs =
      commondata->external_input_Nxx_plus_2NGHOSTS0 * commondata->external_input_Nxx_plus_2NGHOSTS1 * commondata->external_input_Nxx_plus_2NGHOSTS2;

  // Pointers to the external input gridfunctions without ghost zones.
  REAL *restrict external_input_gfs_no_gzs = commondata->external_input_gfs_Cart_basis_no_gzs;

  // Allocate memory for the external input gridfunctions with ghost zones.
  REAL *restrict external_input_gfs = (REAL *)malloc(NUM_EXT_INPUT_CONFORMAL_GFS * total_elements_incl_gzs * sizeof(REAL));
  if (external_input_gfs == NULL) {
    return NUMGRID_EXTERN_MALLOC_ERROR_GFS;
  } // END IF memory allocation for external_input_gfs failed

  // Step 3: Assign the allocated array to commondata for use outside this function.
  commondata->external_input_gfs = external_input_gfs;

  // Step 4: Transfer data from the no-ghost zones array to the array with ghost zones.
  // This involves copying metric data (gamma_{ij} and K_{ij}) from the Cartesian basis into the newly allocated arrays with ghost zones.
  LOOP_OMP("omp parallel for",                                //
           i0, 0, bhahaha_params_and_data->Nr_external_input, //
           i1, 0, commondata->external_input_Nxx1,            //
           i2, 0, commondata->external_input_Nxx2) {
    for (int gf = 0; gf < NUM_EXT_INPUT_CARTESIAN_GFS; gf++) {
      external_input_gfs[EX_IDX4(gf, i0 + i0_min_shift, i1 + NGHOSTS, i2 + NGHOSTS)] = external_input_gfs_no_gzs[EX_NOGZ_IDX4(gf, i0, i1, i2)];
    }
  } // END LOOP: iterating through the external input grid points

  // Step 5: Set up coordinate arrays for a uniform, cell-centered spherical grid.
  {
    const int Nxx_plus_2NGHOSTS0 = commondata->external_input_Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = commondata->external_input_Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = commondata->external_input_Nxx_plus_2NGHOSTS2;

    // Step 5.a: Allocate memory for coordinate arrays in radial, theta, and phi directions.
    commondata->external_input_r_theta_phi[0] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS0);
    commondata->external_input_r_theta_phi[1] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS1);
    commondata->external_input_r_theta_phi[2] = (REAL *)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS2);
    if (commondata->external_input_r_theta_phi[0] == NULL || commondata->external_input_r_theta_phi[1] == NULL ||
        commondata->external_input_r_theta_phi[2] == NULL) {
      free(external_input_gfs);
      return NUMGRID_EXTERN_MALLOC_ERROR_RTHETAPHI;
    } // END IF memory allocation for external_input_r_theta_phi arrays failed

    // Step 5.b: Initialize coordinate arrays for a uniform, cell-centered spherical grid.
    // The coordinates are centered within each cell by adding 0.5 to the index before scaling.
    const REAL xxmin0 = bhahaha_params_and_data->r_min_external_input;
    const REAL xxmin1 = 0.0;
    const REAL xxmin2 = -M_PI;

    for (int j = 0; j < Nxx_plus_2NGHOSTS0; j++)
      commondata->external_input_r_theta_phi[0][j] = xxmin0 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * commondata->external_input_dxx0;
    for (int j = 0; j < Nxx_plus_2NGHOSTS1; j++)
      commondata->external_input_r_theta_phi[1][j] = xxmin1 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * commondata->external_input_dxx1;
    for (int j = 0; j < Nxx_plus_2NGHOSTS2; j++)
      commondata->external_input_r_theta_phi[2][j] = xxmin2 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * commondata->external_input_dxx2;
  } // END BLOCK: setting up coordinate arrays

  // Step 6: Transform the metric components (gamma_{ij}, K_{ij}) from Cartesian to spherical coordinates,
  // including necessary rescaling.
  {
    const int Nxx_plus_2NGHOSTS0 = commondata->external_input_Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = commondata->external_input_Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = commondata->external_input_Nxx_plus_2NGHOSTS2;

    // Step 6.a: Extract coordinate arrays for easier access during transformation.
    REAL *restrict external_input_r_theta_phi[3];
    for (int ww = 0; ww < 3; ww++)
      external_input_r_theta_phi[ww] = commondata->external_input_r_theta_phi[ww];

    // Step 6.b: Metric components: basis transform from Cartesian to Spherical & convert ADM->rescaled BSSN.
#pragma omp parallel for
    for (int i2 = NGHOSTS; i2 < commondata->external_input_Nxx2 + NGHOSTS; i2++) {
      const REAL xx2 = external_input_r_theta_phi[2][i2];
      for (int i1 = NGHOSTS; i1 < commondata->external_input_Nxx1 + NGHOSTS; i1++) {
        const REAL xx1 = external_input_r_theta_phi[1][i1];
        // Include all valid points, including those near r ~ r_max.
        for (int i0 = i0_min_shift; i0 < commondata->external_input_Nxx_plus_2NGHOSTS0; i0++) {
          const REAL xx0 = external_input_r_theta_phi[0][i0];

          // Read Cartesian metric components at the current grid point.
          const REAL Cart_gammaDD00 = external_input_gfs[IDX4(INTERP_GAMMADDXXGF, i0, i1, i2)];
          const REAL Cart_gammaDD01 = external_input_gfs[IDX4(INTERP_GAMMADDXYGF, i0, i1, i2)];
          const REAL Cart_gammaDD02 = external_input_gfs[IDX4(INTERP_GAMMADDXZGF, i0, i1, i2)];
          const REAL Cart_gammaDD11 = external_input_gfs[IDX4(INTERP_GAMMADDYYGF, i0, i1, i2)];
          const REAL Cart_gammaDD12 = external_input_gfs[IDX4(INTERP_GAMMADDYZGF, i0, i1, i2)];
          const REAL Cart_gammaDD22 = external_input_gfs[IDX4(INTERP_GAMMADDZZGF, i0, i1, i2)];
          const REAL Cart_KDD00 = external_input_gfs[IDX4(INTERP_KDDXXGF, i0, i1, i2)];
          const REAL Cart_KDD01 = external_input_gfs[IDX4(INTERP_KDDXYGF, i0, i1, i2)];
          const REAL Cart_KDD02 = external_input_gfs[IDX4(INTERP_KDDXZGF, i0, i1, i2)];
          const REAL Cart_KDD11 = external_input_gfs[IDX4(INTERP_KDDYYGF, i0, i1, i2)];
          const REAL Cart_KDD12 = external_input_gfs[IDX4(INTERP_KDDYZGF, i0, i1, i2)];
          const REAL Cart_KDD22 = external_input_gfs[IDX4(INTERP_KDDZZGF, i0, i1, i2)];
          const REAL tmp0 = ((xx0) * (xx0) * (xx0) * (xx0));
          const REAL tmp1 = sin(xx1);
          const REAL tmp4 = sin(xx2);
          const REAL tmp6 = ((xx0) * (xx0));
          const REAL tmp10 = cos(xx2);
          const REAL tmp14 = 2 * Cart_gammaDD01;
          const REAL tmp18 = cos(xx1);
          const REAL tmp36 = Cart_gammaDD01 * xx0;
          const REAL tmp84 = 2 * Cart_KDD01;
          const REAL tmp110 = (2.0 / 3.0) * Cart_gammaDD01;
          const REAL tmp2 = ((tmp1) * (tmp1));
          const REAL tmp5 = ((tmp4) * (tmp4));
          const REAL tmp11 = ((tmp10) * (tmp10));
          const REAL tmp19 = tmp1 * tmp18;
          const REAL tmp22 = ((tmp18) * (tmp18));
          const REAL tmp24 = Cart_gammaDD02 * tmp10;
          const REAL tmp26 = Cart_gammaDD12 * tmp4;
          const REAL tmp37 = tmp10 * tmp4;
          const REAL tmp81 = Cart_KDD02 * tmp10;
          const REAL tmp82 = Cart_KDD12 * tmp4;
          const REAL tmp103 = (1.0 / (tmp1));
          const REAL tmp105 = (1.0 / (tmp6));
          const REAL tmp3 = (1.0 / (tmp2));
          const REAL tmp7 = tmp2 * tmp6;
          const REAL tmp20 = tmp19 * xx0;
          const REAL tmp23 = tmp22 * xx0;
          const REAL tmp28 = tmp2 * xx0;
          const REAL tmp35 = 2 * tmp19;
          const REAL tmp38 = tmp36 * tmp37;
          const REAL tmp43 = tmp11 * tmp2;
          const REAL tmp45 = tmp2 * tmp5;
          const REAL tmp47 = tmp2 * tmp37;
          const REAL tmp52 = tmp19 * tmp6;
          const REAL tmp72 = tmp22 * tmp6;
          const REAL tmp109 = (2.0 / 3.0) * tmp19;
          const REAL tmp15 = tmp10 * tmp7;
          const REAL tmp21 = Cart_gammaDD22 * tmp20;
          const REAL tmp25 = tmp23 * tmp24;
          const REAL tmp27 = tmp23 * tmp26;
          const REAL tmp29 = tmp24 * tmp28;
          const REAL tmp30 = tmp26 * tmp28;
          const REAL tmp32 = Cart_gammaDD00 * tmp11 * tmp20;
          const REAL tmp55 = tmp37 * tmp52;
          const REAL tmp64 = tmp36 * tmp45;
          const REAL tmp65 = tmp28 * tmp37;
          const REAL tmp67 = tmp36 * tmp43;
          const REAL tmp71 = 2 * tmp52;
          const REAL tmp73 = Cart_gammaDD00 * tmp11 * tmp72;
          const REAL tmp76 = tmp37 * tmp72;
          const REAL tmp92 = Cart_KDD00 * tmp43 + Cart_KDD11 * tmp45 + Cart_KDD22 * tmp22 + tmp35 * tmp81 + tmp35 * tmp82 + tmp47 * tmp84;
          const REAL tmp9 = Cart_gammaDD00 * tmp5 * tmp7;
          const REAL tmp13 = Cart_gammaDD11 * tmp11 * tmp7;
          const REAL tmp16 = tmp15 * tmp4;
          const REAL tmp34 = Cart_gammaDD11 * tmp20 * tmp5;
          const REAL tmp48 = Cart_gammaDD00 * tmp43 + Cart_gammaDD11 * tmp45 + Cart_gammaDD22 * tmp22 + tmp14 * tmp47 + tmp24 * tmp35 + tmp26 * tmp35;
          const REAL tmp49 = Cart_gammaDD12 * tmp15;
          const REAL tmp51 = Cart_gammaDD02 * tmp4 * tmp7;
          const REAL tmp54 = Cart_gammaDD01 * tmp5 * tmp52;
          const REAL tmp56 = Cart_gammaDD00 * tmp55;
          const REAL tmp57 = Cart_gammaDD01 * tmp11 * tmp52;
          const REAL tmp58 = Cart_gammaDD11 * tmp55;
          const REAL tmp61 = Cart_gammaDD02 * tmp20 * tmp4;
          const REAL tmp63 = Cart_gammaDD12 * tmp10 * tmp20;
          const REAL tmp66 = Cart_gammaDD00 * tmp65;
          const REAL tmp68 = Cart_gammaDD11 * tmp65;
          const REAL tmp75 = Cart_gammaDD11 * tmp5 * tmp72;
          const REAL tmp85 =
              Cart_KDD00 * tmp11 * tmp72 + Cart_KDD11 * tmp5 * tmp72 + Cart_KDD22 * tmp7 - tmp71 * tmp81 - tmp71 * tmp82 + tmp76 * tmp84;
          const REAL tmp94 = Cart_KDD00 * tmp11 * tmp20 + Cart_KDD11 * tmp20 * tmp5 - Cart_KDD22 * tmp20 + tmp20 * tmp37 * tmp84 + tmp23 * tmp81 +
                             tmp23 * tmp82 - tmp28 * tmp81 - tmp28 * tmp82;
          const REAL tmp97 = -Cart_KDD00 * tmp55 + Cart_KDD01 * tmp11 * tmp52 - Cart_KDD01 * tmp5 * tmp52 + Cart_KDD02 * tmp4 * tmp7 +
                             Cart_KDD11 * tmp55 - Cart_KDD12 * tmp15;
          const REAL tmp99 = -Cart_KDD00 * tmp65 + Cart_KDD01 * tmp43 * xx0 - Cart_KDD01 * tmp45 * xx0 - Cart_KDD02 * tmp20 * tmp4 +
                             Cart_KDD11 * tmp65 + Cart_KDD12 * tmp10 * tmp20;
          const REAL tmp17 = tmp13 - tmp14 * tmp16 + tmp9;
          const REAL tmp39 = -tmp21 + tmp25 + tmp27 - tmp29 - tmp30 + tmp32 + tmp34 + tmp35 * tmp38;
          const REAL tmp59 = tmp49 - tmp51 + tmp54 + tmp56 - tmp57 - tmp58;
          const REAL tmp69 = tmp61 - tmp63 + tmp64 + tmp66 - tmp67 - tmp68;
          const REAL tmp77 = Cart_gammaDD22 * tmp7 + tmp14 * tmp76 - tmp24 * tmp71 - tmp26 * tmp71 + tmp73 + tmp75;
          const REAL tmp80 = 4 * tmp19 * tmp38 - 2 * tmp21 + 2 * tmp25 + 2 * tmp27 - 2 * tmp29 - 2 * tmp30 + 2 * tmp32 + 2 * tmp34;
          const REAL tmp86 = -tmp61 + tmp63 - tmp64 - tmp66 + tmp67 + tmp68;
          const REAL tmp88 = -tmp49 + tmp51 - tmp54 - tmp56 + tmp57 + tmp58;
          const REAL tmp93 = Cart_KDD00 * tmp5 * tmp7 + Cart_KDD11 * tmp11 * tmp7 - tmp16 * tmp84;
          const REAL tmp41 = tmp17 * ((tmp39) * (tmp39));
          const REAL tmp79 = tmp17 * tmp48 * tmp77;
          const REAL tmp91 = (1.0 / (-tmp41 - tmp48 * ((tmp88) * (tmp88)) - tmp77 * ((tmp86) * (tmp86)) + tmp79 + tmp80 * tmp86 * tmp88));
          const REAL tmp95 = 2 * tmp91;
          const REAL tmp101 = cbrt(tmp0 * tmp2 * tmp91);
          const REAL tmp100 = tmp85 * tmp91 * (tmp17 * tmp48 - ((tmp86) * (tmp86))) + tmp91 * tmp92 * (tmp17 * tmp77 - ((tmp88) * (tmp88))) +
                              tmp91 * tmp93 * (-((tmp39) * (tmp39)) + tmp48 * tmp77) + tmp94 * tmp95 * (-tmp17 * tmp39 + tmp86 * tmp88) +
                              tmp95 * tmp97 * (tmp39 * tmp86 - tmp48 * tmp88) + tmp95 * tmp99 * (tmp39 * tmp88 + tmp69 * tmp77);
          const REAL tmp102 = tmp101 / xx0;
          const REAL tmp107 = tmp101 * tmp103 * tmp105;
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_WWGF, i0, i1, i2)] =
              (1.0 /
               (sqrt(cbrt(fabs(tmp3 * (tmp41 + tmp48 * ((tmp59) * (tmp59)) - tmp59 * tmp69 * tmp80 + ((tmp69) * (tmp69)) * tmp77 - tmp79) / tmp0)))));
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_TRKGF, i0, i1, i2)] = tmp100;
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_HDD00GF, i0, i1, i2)] = tmp101 * tmp48 - 1;
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_HDD01GF, i0, i1, i2)] = tmp102 * tmp39;
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_HDD02GF, i0, i1, i2)] = tmp102 * tmp103 * tmp86;
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_HDD11GF, i0, i1, i2)] = tmp105 * (tmp101 * tmp77 - tmp6);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_HDD12GF, i0, i1, i2)] = tmp107 * tmp88;
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_HDD22GF, i0, i1, i2)] = tmp105 * tmp3 * (tmp101 * tmp17 - tmp7);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_ADD00GF, i0, i1, i2)] =
              tmp101 * (-tmp100 * ((1.0 / 3.0) * Cart_gammaDD00 * tmp43 + (1.0 / 3.0) * Cart_gammaDD11 * tmp45 +
                                   (1.0 / 3.0) * Cart_gammaDD22 * tmp22 + tmp109 * tmp24 + tmp109 * tmp26 + tmp110 * tmp47) +
                        tmp92);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_ADD01GF, i0, i1, i2)] =
              tmp102 * (-tmp100 * (tmp109 * tmp38 - 1.0 / 3.0 * tmp21 + (1.0 / 3.0) * tmp25 + (1.0 / 3.0) * tmp27 - 1.0 / 3.0 * tmp29 -
                                   1.0 / 3.0 * tmp30 + (1.0 / 3.0) * tmp32 + (1.0 / 3.0) * tmp34) +
                        tmp94);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_ADD02GF, i0, i1, i2)] =
              tmp102 * tmp103 *
              (-tmp100 *
                   (-1.0 / 3.0 * tmp61 + (1.0 / 3.0) * tmp63 - 1.0 / 3.0 * tmp64 - 1.0 / 3.0 * tmp66 + (1.0 / 3.0) * tmp67 + (1.0 / 3.0) * tmp68) +
               tmp99);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_ADD11GF, i0, i1, i2)] =
              tmp101 * tmp105 *
              (-tmp100 * ((1.0 / 3.0) * Cart_gammaDD22 * tmp7 + tmp110 * tmp76 - 2.0 / 3.0 * tmp24 * tmp52 - 2.0 / 3.0 * tmp26 * tmp52 +
                          (1.0 / 3.0) * tmp73 + (1.0 / 3.0) * tmp75) +
               tmp85);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_ADD12GF, i0, i1, i2)] =
              tmp107 * (-tmp100 * (-1.0 / 3.0 * tmp49 + (1.0 / 3.0) * tmp51 - 1.0 / 3.0 * tmp54 - 1.0 / 3.0 * tmp56 + (1.0 / 3.0) * tmp57 +
                                   (1.0 / 3.0) * tmp58) +
                        tmp97);
          external_input_gfs[IDX4(EXTERNAL_SPHERICAL_ADD22GF, i0, i1, i2)] =
              tmp101 * tmp105 * tmp3 * (-tmp100 * (-tmp110 * tmp16 + (1.0 / 3.0) * tmp13 + (1.0 / 3.0) * tmp9) + tmp93);

        } // END LOOP over i0
      } // END LOOP over i1
    } // END LOOP over i2
  } // END BLOCK: transformation and rescaling

  // Step 7: Set up boundary condition structures and apply inner boundary conditions.
  {
    const int Nxx_plus_2NGHOSTS0 = commondata->external_input_Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = commondata->external_input_Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = commondata->external_input_Nxx_plus_2NGHOSTS2;

    // Assign grid spacings and sizes to the boundary condition structure within commondata.
    commondata->bcstruct_dxx0 = commondata->external_input_dxx0;
    commondata->bcstruct_dxx1 = commondata->external_input_dxx1;
    commondata->bcstruct_dxx2 = commondata->external_input_dxx2;

    commondata->bcstruct_Nxx_plus_2NGHOSTS0 = Nxx_plus_2NGHOSTS0;
    commondata->bcstruct_Nxx_plus_2NGHOSTS1 = Nxx_plus_2NGHOSTS1;
    commondata->bcstruct_Nxx_plus_2NGHOSTS2 = Nxx_plus_2NGHOSTS2;

    // Initialize the boundary condition structure for external input data.
    bc_struct external_input_bcstruct;
    bah_bcstruct_set_up(commondata, commondata->external_input_r_theta_phi, &external_input_bcstruct);

    // Step 7.a: Unpack boundary condition information from the boundary condition structure.
    const bc_info_struct *restrict bc_info = &external_input_bcstruct.bc_info;

    // Step 7.b: Apply inner boundary conditions to all gridfunctions.
    // This involves copying values from source points to destination points with parity corrections.
#pragma omp parallel
    for (int which_gf = 0; which_gf < NUM_EXT_INPUT_CONFORMAL_GFS; which_gf++) {
#pragma omp for
      for (int pt = 0; pt < bc_info->num_inner_boundary_points; pt++) {
        const int dstpt = external_input_bcstruct.inner_bc_array[pt].dstpt;
        const int srcpt = external_input_bcstruct.inner_bc_array[pt].srcpt;
        // Apply the boundary condition by copying values from the source point to the destination point,
        // applying the appropriate parity correction for the gridfunction.
        commondata->external_input_gfs[IDX4pt(which_gf, dstpt)] =
            external_input_bcstruct.inner_bc_array[pt].parity[external_input_gf_parity[which_gf]] *
            commondata->external_input_gfs[IDX4pt(which_gf, srcpt)];
      } // END LOOP over inner boundary points
    } // END LOOP over gridfunctions

    // Step 7.c: Free allocated memory for boundary condition structures to prevent memory leaks.
    free(external_input_bcstruct.inner_bc_array);
    for (int ng = 0; ng < NGHOSTS * 3; ng++)
      free(external_input_bcstruct.pure_outer_bc_array[ng]);
  } // END BLOCK: applying boundary conditions

  return BHAHAHA_SUCCESS;
} // END FUNCTION bah_numgrid__external_input_set_up
