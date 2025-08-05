#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 *
 * Function: bah_error_handling()
 *
 * Description:
 * - Driver function for BHaHAHA error reporting, including when horizon not found!
 * - This function interprets error messages from throughout BHaHAHA.
 *
 * Parameter:
 * - error_code - error code from BHaHAHA.
 *
 * Returns:
 * - Error message string.
 *
 */
const char *bah_error_message(const bhahaha_error_codes error_code) {
  switch (error_code) {
  case BHAHAHA_SUCCESS:
    return "Success.";
  case FIND_HORIZON_GETTIMEOFDAY_BROKEN:
    return "bah_find_horizon(): gettimeofday() returned an error exit code, indicating a non-POSIX-compatible or broken system. Sorry.";
  case FIND_HORIZON_MAX_ITERATIONS_EXCEEDED:
    return "bah_find_horizon(): maximum iterations exceeded. Set verbosity to 2, or increase max_iterations.";
  case FIND_HORIZON_HORIZON_TOO_SMALL:
    return "bah_find_horizon(): Horizon radius < 3 * dr of BHaHAHA input grid. Set verbosity to 2, and/or try a smaller/higher-resolution grid.";
  case BCSTRUCT_EIGENCOORD_FAILURE:
    return "bah_bcstruct_set_up(): problem setting up Eigen-coordinates.";
  case BCSTRUCT_SET_PARITY_ERROR:
    return "bah_bcstruct_set_up(): problem computing parity conditions.";
  case INITIAL_DATA_MALLOC_ERROR:
    return "bah_initial_data(): Failed to allocate memory to coarse_to_fine or dst_pts[][].";
  case NUMGRID_EXTERN_MALLOC_ERROR_GFS:
    return "bah_numgrid__external_input_set_up(): Failed to allocate memory to external_input_gfs.";
  case NUMGRID_EXTERN_MALLOC_ERROR_RTHETAPHI:
    return "bah_numgrid__external_input_set_up(): Failed to allocate memory to commondata->external_input_r_theta_phi[][].";
  case NUMGRID_INTERP_MALLOC_ERROR_GFS:
    return "bah_numgrid__interp_src_set_up(): Failed to allocate memory to commondata->interp_src_gfs.";
  case NUMGRID_INTERP_MALLOC_ERROR_RTHETAPHI:
    return "bah_numgrid__interp_src_set_up(): Failed to allocate memory to commondata->interp_src_r_theta_phi.";
  case INTERP1D_NULL_PTRS:
    return "bah_interpolation_1d_radial_spokes_on_3d_src_grid(): Found NULL pointer(s); one or more input arrays not malloc'ed.";
  case INTERP1D_INTERP_ORDER_GT_NXX_PLUS_2NINTERPGHOSTS0:
    return "bah_interpolation_1d_radial_spokes_on_3d_src_grid(): Interpolation order > Nxx0 + 2*NinterpGHOSTS.";
  case INTERP1D_HORIZON_TOO_LARGE:
    return "bah_interpolation_1d_radial_spokes_on_3d_src_grid(): Horizon extends beyond input grid. Set verbosity to 2, and/or try a larger grid.";
  case INTERP1D_HORIZON_TOO_SMALL:
    return "bah_interpolation_1d_radial_spokes_on_3d_src_grid(): Horizon radius dropped below r_min_external_input. Set verbosity to 2, and/or try a "
           "larger/higher-resolution grid.";
  case INTERP2D_EXT_TO_INTERPSRC_NULL_PTRS:
    return "bah_interpolation_2d_external_input_to_interp_src_grid(): Found NULL pointer(s); one or more input arrays not malloc'ed.";
  case INTERP2D_EXT_TO_INTERPSRC_INTERP_ORDER_GT_NXX_PLUS_2NINTERPGHOSTS12:
    return "bah_interpolation_2d_external_input_to_interp_src_grid(): Interpolation order > Nxx{1,2} + 2*NinterpGHOSTS.";
  case INTERP2D_EXT_TO_INTERPSRC_HORIZON_OUT_OF_BOUNDS:
    return "bah_interpolation_2d_external_input_to_interp_src_grid(): Horizon extends beyond angular directions theta,phi. Should never happen...";
  case INTERP2D_GENERAL_NULL_PTRS:
    return "bah_interpolation_2d_general__uniform_src_grid(): Found NULL pointer(s); one or more input arrays not malloc'ed.";
  case INTERP2D_GENERAL_INTERP_ORDER_GT_NXX_PLUS_2NGHOSTS12:
    return "bah_interpolation_2d_general__uniform_src_grid(): Interpolation order > Nxx{1,2} + 2*NGHOSTS.";
  case INTERP2D_GENERAL_HORIZON_OUT_OF_BOUNDS:
    return "bah_interpolation_2d_general__uniform_src_grid(): Horizon extends beyond angular directions theta,phi. Should never happen...";
  case DIAG_PROPER_CIRCUM_MALLOC_ERROR:
    return "diagnostics_proper_circumferences(): One or more malloc's failed.";
  }
  fprintf(stderr, "BHaHAHA error code %d not defined!\n", error_code);
  return NULL;
} // END FUNCTION bah_error_message
