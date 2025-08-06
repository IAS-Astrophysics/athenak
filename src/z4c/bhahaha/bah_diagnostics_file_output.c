#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Outputs BHaHAHA diagnostic data to two files:
 * 1. A diagnostics file (appended) for the given horizon
 * 2. A gnuplot-friendly file of horizon-surface data
 *
 * Operations performed:
 * 1. Generates filenames based on the horizon index and iteration, placing them
 * in the given output directory.
 * 2. Opens/creates the diagnostics file in append mode.
 * 3. Writes headers during the first simulation iteration (i.e., when file is new).
 * 4. Outputs diagnostic metrics, including geometric properties and spin magnitudes.
 * 5. Writes horizon surface data in a separate file, in a gnuplot-compatible format.
 *
 * @param diags                     Pointer to the BHaHAHA data structure containing diagnostic info.
 * @param bhahaha_params_and_data   Pointer to the BHaHAHA data structure containing horizon parameters and data.
 * @param N_horizons                Total number of horizons being tracked.
 * @param x_center_input            x-position on the global grid input into BHaHAHA.
 * @param y_center_input            y-position on the global grid input into BHaHAHA.
 * @param z_center_input            z-position on the global grid input into BHaHAHA.
 * @param output_directory          Path where all output files will be written (e.g., ".", "/path/to/dir/", etc.).
 *
 */
void bah_diagnostics_file_output(const bhahaha_diagnostics_struct *diags, const bhahaha_params_and_data_struct *bhahaha_params_and_data,
                                 int N_horizons, const REAL x_center_input, const REAL y_center_input, const REAL z_center_input,
                                 const char *output_directory) {

  // For safety, ensure output_directory is valid; fallback to "."
  if (!output_directory || output_directory[0] == '\0') {
    output_directory = ".";
  }

  // Current horizon radial data.
  const REAL *restrict curr_horizon_h_of_theta_phi = bhahaha_params_and_data->prev_horizon_m1;

  char file_name_buffer[1024]; // Buffer for diagnostics & surface-data file names.
  FILE *fileptr = NULL;        // File pointer for output.

  // ----------------------------------------------
  // (1) BHaHAHA diagnostics file output
  // ----------------------------------------------
  // Generate the diagnostics file name based on the horizon index, under output_directory.
  snprintf(file_name_buffer, sizeof(file_name_buffer), "%s/BHaHAHA_diagnostics.ah%d.gp", output_directory, bhahaha_params_and_data->which_horizon);

  // Open the diagnostics file in append mode. "a+" so we can read if we need to check file size.
  fileptr = fopen(file_name_buffer, "a+");
  if (fileptr == NULL) {
    fprintf(stderr, "Can't open BH-diagnostics output file \"%s\" for writing/appending!\n", file_name_buffer);
    return;
  } // END IF problem opening file

  // Determine if file is newly created by checking its size.
  fseek(fileptr, 0, SEEK_END);
  long file_size = ftell(fileptr);
  if (file_size == 0) {
    // File is new/empty -> write header once.
    fprintf(fileptr, "# apparent horizon %d/%d\n", bhahaha_params_and_data->which_horizon, N_horizons);
    fprintf(fileptr, "#\n");
    fprintf(fileptr, "# column  1 = Current simulation iteration\n");
    fprintf(fileptr, "# column  2 = Current simulation time\n");
    fprintf(fileptr, "# column  3 = x-coordinate of the centroid\n");
    fprintf(fileptr, "# column  4 = y-coordinate of the centroid\n");
    fprintf(fileptr, "# column  5 = z-coordinate of the centroid\n");
    fprintf(fileptr, "# column  6 = Minimum coordinate radius (centroid-based)\n");
    fprintf(fileptr, "# column  7 = Maximum coordinate radius (centroid-based)\n");
    fprintf(fileptr, "# column  8 = Mean coordinate radius (centroid-based)\n");
    fprintf(fileptr, "# column  9 = Circumference in the xy-plane\n");
    fprintf(fileptr, "# column 10 = Circumference in the xz-plane\n");
    fprintf(fileptr, "# column 11 = Circumference in the yz-plane\n");
    fprintf(fileptr, "# column 12 = Apparent horizon area\n");
    fprintf(fileptr, "# column 13 = Irreducible mass\n");
    fprintf(fileptr, "# column 14 = Theta (L-infinity norm) times mass\n");
    fprintf(fileptr, "# column 15 = Theta (L2 norm) times mass\n");
    fprintf(fileptr, "# column 16 = Spin x-component (based on xy/yz)\n");
    fprintf(fileptr, "# column 17 = Spin x-component (based on xz/yz)\n");
    fprintf(fileptr, "# column 18 = Spin y-component (based on yz/xz)\n");
    fprintf(fileptr, "# column 19 = Spin y-component (based on xy/xz)\n");
    fprintf(fileptr, "# column 20 = Spin z-component (based on xz/xy)\n");
    fprintf(fileptr, "# column 21 = Spin z-component (based on yz/xy)\n");
    fflush(fileptr);
  } // END IF file size zero -> need to write header

  // Calculate irreducible mass from horizon area.
  REAL M_irr = sqrt(diags->area / (16.0 * M_PI));

  // Assign spin magnitudes, using NaN to indicate undefined values.
  const REAL a_x_xy_over_yz_spin = (diags->spin_a_x_from_xy_over_yz_prop_circumfs != -10.0) ? diags->spin_a_x_from_xy_over_yz_prop_circumfs : NAN;
  const REAL a_x_xz_over_yz_spin = (diags->spin_a_x_from_xz_over_yz_prop_circumfs != -10.0) ? diags->spin_a_x_from_xz_over_yz_prop_circumfs : NAN;
  const REAL a_y_yz_over_xz_spin = (diags->spin_a_y_from_yz_over_xz_prop_circumfs != -10.0) ? diags->spin_a_y_from_yz_over_xz_prop_circumfs : NAN;
  const REAL a_y_xy_over_xz_spin = (diags->spin_a_y_from_xy_over_xz_prop_circumfs != -10.0) ? diags->spin_a_y_from_xy_over_xz_prop_circumfs : NAN;
  const REAL a_z_xz_over_xy_spin = (diags->spin_a_z_from_xz_over_xy_prop_circumfs != -10.0) ? diags->spin_a_z_from_xz_over_xy_prop_circumfs : NAN;
  const REAL a_z_yz_over_xy_spin = (diags->spin_a_z_from_yz_over_xy_prop_circumfs != -10.0) ? diags->spin_a_z_from_yz_over_xy_prop_circumfs : NAN;

  // Output diagnostic metrics to the diagnostics file.
  fprintf(fileptr,
          "%d\t%.3f\t%f\t%f\t%f\t%#.10g\t%#.10g\t%#.10g\t%#.10g\t%#.10g\t"
          "%#.10g\t%.15e\t%.15e\t%#.10g\t%#.10g\t%#.10g\t%#.10g\t%#.10g\t%#.10g\t%#.10g\t%#.10g\n",
          bhahaha_params_and_data->iteration_external_input, // (1) iteration
          bhahaha_params_and_data->time_external_input,      // (2) time
          bhahaha_params_and_data->x_center_m1,              // (3) centroid x
          bhahaha_params_and_data->y_center_m1,              // (4) centroid y
          bhahaha_params_and_data->z_center_m1,              // (5) centroid z
          diags->min_coord_radius_wrt_centroid,              // (6) min coord radius
          diags->max_coord_radius_wrt_centroid,              // (7) max coord radius
          diags->mean_coord_radius_wrt_centroid,             // (8) mean coord radius
          diags->xy_plane_circumference,                     // (9) xy-plane circumference
          diags->xz_plane_circumference,                     // (10) xz-plane circumference
          diags->yz_plane_circumference,                     // (11) yz-plane circumference
          diags->area,                                       // (12) horizon area
          M_irr,                                             // (13) irreducible mass
          diags->Theta_Linf_times_M,                         // (14) Theta (L∞ norm)*M
          diags->Theta_L2_times_M,                           // (15) Theta (L2 norm)*M
          a_x_xy_over_yz_spin,                               // (16) Spin x (xy/yz)
          a_x_xz_over_yz_spin,                               // (17) Spin x (xz/yz)
          a_y_yz_over_xz_spin,                               // (18) Spin y (yz/xz)
          a_y_xy_over_xz_spin,                               // (19) Spin y (xy/xz)
          a_z_xz_over_xy_spin,                               // (20) Spin z (xz/xy)
          a_z_yz_over_xy_spin                                // (21) Spin z (yz/xy)
  );

  fflush(fileptr);
  fclose(fileptr);

  // ----------------------------------------------
  // (2) Horizon surface data (h.t) file output
  // ----------------------------------------------
  snprintf(file_name_buffer, sizeof(file_name_buffer), "%s/h.t%07d.ah%d.gp", output_directory, bhahaha_params_and_data->iteration_external_input,
           bhahaha_params_and_data->which_horizon);

  fileptr = fopen(file_name_buffer, "w");
  if (fileptr == NULL) {
    fprintf(stderr, "Can't open horizon surface data output file \"%s\" for writing!\n", file_name_buffer);
    return;
  } // END IF problem opening file

  // Write headers for the gnuplot-compatible horizon surface data file.
  fprintf(fileptr, "# gnuplot-compatible horizon surface data, at time %.3f.\n", bhahaha_params_and_data->time_external_input);
  fprintf(fileptr, "# Column 1: x coordinate\n");
  fprintf(fileptr, "# Column 2: y coordinate\n");
  fprintf(fileptr, "# Column 3: z coordinate\n");

  // Retrieve the maximum theta/phi resolution from the last multigrid index.
  const int Ntheta_max = bhahaha_params_and_data->Ntheta_array_multigrid[bhahaha_params_and_data->num_resolutions_multigrid - 1];
  const int Nphi_max = bhahaha_params_and_data->Nphi_array_multigrid[bhahaha_params_and_data->num_resolutions_multigrid - 1];

  // Number of divisions for phi and theta.
  const int NUM_PHI = Nphi_max;
  const int NUM_THETA = Ntheta_max;

  // Loop over theta and phi, writing out Cartesian coordinates of the horizon surface.
  for (int itheta = 0; itheta < NUM_THETA; itheta++) {
    const REAL theta = ((REAL)itheta + 0.5) * M_PI / (REAL)NUM_THETA;
    const REAL sintheta = sin(theta);
    const REAL costheta = cos(theta);

    REAL first_x = NAN, first_y = NAN, first_z = NAN;

    for (int iphi = 0; iphi < NUM_PHI; iphi++) {
      const REAL phi = -M_PI + ((REAL)iphi + 0.5) * (2.0 * M_PI / (REAL)NUM_PHI);
      const REAL sinphi = sin(phi);
      const REAL cosphi = cos(phi);

      // Retrieve the radial coordinate from horizon data (prev_horizon_m1).
      const REAL r = curr_horizon_h_of_theta_phi[IDX2(itheta, iphi)];

      // Compute Cartesian coords; shift by input (global) centroid if desired.
      const REAL x = r * sintheta * cosphi + x_center_input;
      const REAL y = r * sintheta * sinphi + y_center_input;
      const REAL z = r * costheta + z_center_input;

      // Store first point, so we can close the loop at the end of the phi sweep.
      if (iphi == 0) {
        first_x = x;
        first_y = y;
        first_z = z;
      } // END IF iphi==0

      // Output the Cartesian coordinates.
      fprintf(fileptr, "%.10e %.10e %.10e\n", x, y, z);
    } // END LOOP over phi

    // Close the loop by reprinting the first point of this theta-ring.
    fprintf(fileptr, "%.10e %.10e %.10e\n", first_x, first_y, first_z);
    fprintf(fileptr, "\n"); // Blank line to separate slices in gnuplot format
  } // END LOOP over theta

  fflush(fileptr);
  fclose(fileptr);
} // END FUNCTION bah_diagnostics_file_output
