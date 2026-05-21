#include "BHaH_defines.h"
/**
 * Set commondata_struct to default values specified within NRPy+.
 */
void bah_commondata_struct_set_to_default(commondata_struct *restrict commondata) {

  // Set commondata_struct variables to default
  commondata->CFL_FACTOR = 0.5;          // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::CFL_FACTOR
  commondata->KO_diss_strength = 0.5;    // nrpy.infrastructures.BHaH.BHaHAHA.rhs_eval_KO_apply::KO_diss_strength
  commondata->NUMGRIDS = 1;              // nrpy.grid::NUMGRIDS
  commondata->eta_damping = 2.0;         // nrpy.infrastructures.BHaH.BHaHAHA.rhs_eval_KO_apply::eta_damping
  commondata->external_input_Nxx0 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx0
  commondata->external_input_Nxx1 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx1
  commondata->external_input_Nxx2 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx2
  commondata->external_input_Nxx_plus_2NGHOSTS0 =
      128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx_plus_2NGHOSTS0
  commondata->external_input_Nxx_plus_2NGHOSTS1 =
      128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx_plus_2NGHOSTS1
  commondata->external_input_Nxx_plus_2NGHOSTS2 =
      128;                                  // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx_plus_2NGHOSTS2
  commondata->external_input_dxx0 = 128;    // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx0
  commondata->external_input_dxx1 = 128;    // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx1
  commondata->external_input_dxx2 = 128;    // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx2
  commondata->external_input_invdxx0 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx0
  commondata->external_input_invdxx1 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx1
  commondata->external_input_invdxx2 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx2
  commondata->interp_src_Nxx0 = 128;        // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx0
  commondata->interp_src_Nxx1 = 128;        // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx1
  commondata->interp_src_Nxx2 = 128;        // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx2
  commondata->interp_src_Nxx_plus_2NGHOSTS0 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx_plus_2NGHOSTS0
  commondata->interp_src_Nxx_plus_2NGHOSTS1 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx_plus_2NGHOSTS1
  commondata->interp_src_Nxx_plus_2NGHOSTS2 = 128; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx_plus_2NGHOSTS2
  commondata->interp_src_dxx0 = 128;               // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx0
  commondata->interp_src_dxx1 = 128;               // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx1
  commondata->interp_src_dxx2 = 128;               // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx2
  commondata->interp_src_invdxx0 = 128;            // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx0
  commondata->interp_src_invdxx1 = 128;            // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx1
  commondata->interp_src_invdxx2 = 128;            // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx2
  commondata->max_radius_wrt_grid_center =
      -1.0; // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics_area_centroid_and_Theta_norms::max_radius_wrt_grid_center
  commondata->min_radius_wrt_grid_center =
      -1.0; // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics_area_centroid_and_Theta_norms::min_radius_wrt_grid_center
  commondata->output_diagnostics_every_nn = 500; // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics::output_diagnostics_every_nn
  commondata->t_final = 1e+30;                   // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_final
} // END FUNCTION bah_commondata_struct_set_to_default
