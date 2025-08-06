#ifdef __cplusplus
extern "C" {
#endif

#ifndef BHAHAHA_HEADER_H
#define BHAHAHA_HEADER_H

// Definition of REAL data type, using double by default.
#ifndef REAL
#define REAL double
#endif

// Number of external input Cartesian grid functions.
// gamma_{ij} and K_{ij} each have 6 independent components.
#ifndef NUM_EXT_INPUT_CARTESIAN_GFS
#define NUM_EXT_INPUT_CARTESIAN_GFS 12
#endif
// Sets the ordering of the 12 external input Cartesian grid functions.
enum {
  INTERP_GAMMADDXXGF,
  INTERP_GAMMADDXYGF,
  INTERP_GAMMADDXZGF,
  INTERP_GAMMADDYYGF,
  INTERP_GAMMADDYZGF,
  INTERP_GAMMADDZZGF,
  INTERP_KDDXXGF,
  INTERP_KDDXYGF,
  INTERP_KDDXZGF,
  INTERP_KDDYYGF,
  INTERP_KDDYZGF,
  INTERP_KDDZZGF
};

// Definition of PI if not already defined (for portability).
#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif

// IDX2(): Indexing macro for horizon_guess, assuming a 1D array layout where phi varies fastest.
// Usage example: to print h(theta, phi):
// const int NUM_THETA = max(Ntheta);  // Number of theta points.
// const int NUM_PHI = max(Nphi);    // Number of phi points.
// for (int iphi = 0; iphi < NUM_PHI; iphi++) {
//   // Notice that phi and theta are on cell-centered grids, running from (-pi, pi) and (0, pi), respectively.
//   const double phi = -M_PI + ((double)iphi + 0.5) * 2.0 * M_PI / ((double)NUM_PHI);
//   for (int itheta = 0; itheta < NUM_THETA; itheta++) {
//     const double theta = ((double)itheta + 0.5) * M_PI / ((double)NUM_THETA);
//     printf("%e %e %e\n", theta, phi, horizon_guess[IDX2(itheta, iphi)]);
//   } // END LOOP over theta (inner loop)
// } // END LOOP over phi (outer loop)
// Macro definition:
#define IDX2(itheta, iphi) ((itheta) + NUM_THETA * (iphi))
#define MAX_RESOLUTIONS 16

//===============================================
// C struct: bhahaha_params_and_data_struct
// Contains input parameters and data that must be set by the external code.
//===============================================
typedef struct {
  //==========================
  // Metric and grid setup
  //==========================
  REAL *restrict input_metric_data; // Stores gamma_{ij} and K_{ij} in Cartesian
                                    // basis on Nr x Ntheta x Nphi grid.
                                    // Indexing: input_metric_data[(i + Nr * (j
                                    // + Ntheta * (k + Nphi * gf)))], where gf
                                    // is the gridfunction (gxx=0, gxy=1, gxz=2,
                                    // gyy=3, gyz=4, gzz=5, kxx=6, kxy=7, etc.).

  //==========================
  // External Input Numerical Grid: Radial parameters
  //==========================
  // Free parameters for grid providing input data to BHaHAHA from external NR code.
  //
  // The grid is uniform and cell-centered covering the following volume
  // (excluding ghost zones needed for hyperbolic relaxation).
  //   r:      [r_min, r_max]
  //   theta:  [0, pi]
  //   phi:    [-pi, pi]
  //
  // Gridpoint x_{i,j} are computed as:
  //   x_{i,j} = min_i + (j + 0.5) * dx_i, where dx_i = (max_i - min_i) / N_i
  //   i = {0=r, 1=theta, 2=phi}; j ranges from 0 to N_i-1.
  //
  // Implementation notes:
  // - REQUIRED to have N_1 = Ntheta = max(Ntheta); N_2 = Nphi = max(Nphi), thus
  //   only radial coordinate need be set, by bah_radial_grid_cell_centered_set_up().
  // - bah_radial_grid_cell_centered_set_up() takes the min and max radii of the
  //   horizon as input and sets a uniform, cell-centered radial grid.
  // - On the first horizon find or when horizon finding fails, pass
  //   r_min=0 and r_max=max_search_radius to bah_radial_grid_cell_centered_set_up().
  //
  // Time from external input grid, needed for quadratic extrapolation & diagnostic file output.
  REAL time_external_input;
  int iteration_external_input; // External NR code iteration; needed for diagnostic file output.
  //                               Set to dummy value if not applicable.

  // Radial grid parameters
  int Nr_external_input; // Recommended default: 48
  REAL r_min_external_input, dr_external_input;
  // Angular multigrid parameters
  // Defaults:
  /*
  num_resolutions_multigrid = 3;
  Ntheta_array_multigrid = {8, 16, 32};
  Nphi_array_multigrid  = {16, 32, 64};
  */
  int num_resolutions_multigrid;
  int Ntheta_array_multigrid[MAX_RESOLUTIONS], Nphi_array_multigrid[MAX_RESOLUTIONS];

  //==========================
  // BHaHAHA hyperbolic relaxation parameters
  //==========================
  // Use h=0.8R as initial guess on full sphere of data? 0=no, 1=yes.
  // IMPORTANT: set to 1 until horizon is found, or if horizon is lost.
  int use_fixed_radius_guess_on_full_sphere;

  // Time Stepping Courant-Friedrichs-Lewy (CFL) Factor,
  //   for SSPRK3: Strong-Stability-Preserving RK3.
  // - max(cfl_factor) = 1.05.
  // - The timestep is set from the cfl_factor at the start of each timestep.
  // - Strongly recommended value: 1.05.
  REAL cfl_factor; // Courant-Friedrichs-Lewy (CFL) factor for time stepping.

  REAL M_scale; // Mass scale for this horizon, in code units. This is the mass used for
  //               eta_damping_times_M, Theta_Linf_times_M, Theta_Linf_times_M_tolerance, etc.
  REAL eta_damping_times_M; // Damping parameter for hyperbolic relaxation, recommended value = 1.6.
  REAL KO_strength;         // Kreiss-Oliger dissipation strength, strongly recommended: set to 0 (disables KO).

  // Convergence thresholds for horizon finding: Horizon finding ceases when
  // EITHER max_iterations is hit, OR:
  // (Theta_Linf_times_M < Theta_Linf_times_M_tolerance AND Theta_L2_times_M < Theta_L2_times_M_tolerance)
  int max_iterations;                // Maximum iterations, recommended value = 10000.
  REAL Theta_Linf_times_M_tolerance; // Theta Linf norm (max(|Theta|)), times mass scale. Recommended value = 1e-2,
  //                                    to defer to L2 norm for general horizon finding, as L2 norm is holistic,
  //                                    and general horizon finding cares mostly about holistic quantities:
  //                                    area, M_irr, and spins.
  REAL Theta_L2_times_M_tolerance; // Theta L2 norm (max(|Theta|)), times mass scale. Recommended value = 2e-5.

  //==========================
  // Diagnostic parameters
  //==========================
  // Used in diagnostics to print which horizon this one is, out of num_horizons.
  int which_horizon, num_horizons;

  // Diagnostic settings
  int verbosity_level;                                     // 0 = essential only; 1 = physical summary of found horizons;
                                                           // 2 = full algorithmic details.
  int enable_eta_varying_alg_for_precision_common_horizon; // 0 = no; 1 = yes. Janky, unless trying to find R_crit for a common horizon.

  //==========================
  // Persistent previous horizon data, used for setting up external input grid & horizon initial guess.
  // THESE ARE NOT SET BY NR CODE. However, NR code must *allocate* max(Ntheta) x max(Nphi) doubles for previous horizons.
  //==========================
  // Previous horizons found:
  //    m1 = "minus 1" = most recent found; m2 = next-to-most-recent found; etc.
  // EXTERNAL NR CODE MUST ALLOCATE: max(Ntheta) x max(Nphi) doubles for each of these, but does not set them!
  REAL *restrict prev_horizon_m1, *restrict prev_horizon_m2, *restrict prev_horizon_m3;

  // DO NOT TOUCH: Persistent quantities set by BHaHAHA.
  // External NR code times at which previous horizons were found.
  REAL t_m1, t_m2, t_m3;
  // Horizon min & max radii.
  REAL r_min_m1, r_min_m2, r_min_m3, r_max_m1, r_max_m2, r_max_m3;
  // Horizon coordinate centers.
  REAL x_center_m1, x_center_m2, x_center_m3;
  REAL y_center_m1, y_center_m2, y_center_m3;
  REAL z_center_m1, z_center_m2, z_center_m3;
} bhahaha_params_and_data_struct;

//===============================================
// C struct: bhahaha_diagnostics_struct
// Contains diagnostic quantities computed by BHaHAHA.
//===============================================
typedef struct {
  //==========================
  // Convergence-related quantities
  //==========================
  REAL Theta_Linf_times_M; // L-infinity norm of Theta (i.e., max(Theta)), times mass scale.
  REAL Theta_L2_times_M;   // L2 of Theta, times mass scale.
  REAL area;               // Surface area of the horizon.

  //==========================
  // Horizon centroid coordinates (relative to input coordinate origin)
  //==========================
  REAL x_centroid_wrt_coord_origin; // X-coordinate of the horizon's centroid.
  REAL y_centroid_wrt_coord_origin; // Y-coordinate of the horizon's centroid.
  REAL z_centroid_wrt_coord_origin; // Z-coordinate of the horizon's centroid.

  //==========================
  // Horizon radius quantities (centroid-based)
  //==========================
  REAL min_coord_radius_wrt_centroid;  // Minimum coordinate radius from the centroid.
  REAL max_coord_radius_wrt_centroid;  // Maximum coordinate radius from the centroid.
  REAL mean_coord_radius_wrt_centroid; // Mean coordinate radius from the centroid.

  //==========================
  // Horizon proper circumferences (planes based on input coordinate origin)
  //==========================
  REAL xy_plane_circumference; // Proper circumference in xy-plane
  REAL xz_plane_circumference; // Proper circumference in xz-plane
  REAL yz_plane_circumference; // Proper circumference in yz-plane

  //==========================
  // Dimensionless spin parameters chi^i = |a^i/m|, based on proper circumferences in various planes.
  // These parameters are computed from specific proper circumference ratios as per
  //  Eq 5.2 of Alcubierre et al. (arXiv:gr-qc/0411149) and related references.
  // If the ratio is > 1 or the spin cannot be computed, -10.0 is returned as a placeholder value.
  REAL spin_a_x_from_xz_over_yz_prop_circumfs;
  REAL spin_a_x_from_xy_over_yz_prop_circumfs;
  REAL spin_a_y_from_yz_over_xz_prop_circumfs;
  REAL spin_a_y_from_xy_over_xz_prop_circumfs;
  REAL spin_a_z_from_xz_over_xy_prop_circumfs;
  REAL spin_a_z_from_yz_over_xy_prop_circumfs;

  // Benchmarking: Counts number of points where Theta is evaluated.
  long Theta_eval_points_counter;
  //==========================
} bhahaha_diagnostics_struct;

//==================
// PUBLIC FUNCTIONS
//==================
// bah_poisoning_*(): Poison inputs into BHaHAHA and check whether the values have been set properly:
// (highly recommended) Call bah_poisoning_set_inputs() before external NR code sets BHaHAHA inputs.
void bah_poisoning_set_inputs(bhahaha_params_and_data_struct *restrict params);
// (highly recommended) Call bah_poisoning_check_inputs() right before bah_find_horizon(), to check whether external NR code has set inputs properly.
void bah_poisoning_check_inputs(const bhahaha_params_and_data_struct *restrict params);
// (required): Set up the (holey) spherical grid for BHaHAHA
void bah_radial_grid_cell_centered_set_up(const int Nr_interp_max, const REAL max_search_radius, const REAL input_r_min, const REAL input_r_max,
                                          int *restrict output_Nr_interp, REAL *restrict output_r_min, REAL *restrict output_dr,
                                          REAL *radii);
void bah_xyz_center_r_minmax(const bhahaha_params_and_data_struct *restrict pars, REAL *restrict x_center, REAL *restrict y_center,
                             REAL *restrict z_center, REAL *restrict r_min, REAL *restrict r_max);
// (required): Core BHaHAHA horizon finder
int bah_find_horizon(bhahaha_params_and_data_struct *restrict bhahaha_params_and_data, bhahaha_diagnostics_struct *restrict bhahaha_diags);
// (optional): Diagnostic file output, similar to AHFinderDirect output files.
void bah_diagnostics_file_output(const bhahaha_diagnostics_struct *diags, const bhahaha_params_and_data_struct *bhahaha_params_and_data,
                                 int N_horizons, const REAL x_center_input, const REAL y_center_input, const REAL z_center_input,
                                 const char *output_directory);

//===============================================
// Set the number of (finite-difference) ghostzones in BHaHAHA
//===============================================
#define BHAHAHA_NGHOSTS 3

//===============================================
// BHaHAHA error handling
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

// Function: bah_error_message
// Interprets bah_find_horizon() error codes & returns a useful string.
const char *bah_error_message(const bhahaha_error_codes error_code);
//===============================================

#endif // BHAHAHA_HEADER_H

#ifdef __cplusplus
}
#endif