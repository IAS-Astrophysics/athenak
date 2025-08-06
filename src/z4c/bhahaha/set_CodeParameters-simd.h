const REAL NOSIMDCart_originx = params->Cart_originx;                            // nrpy.grid::Cart_originx
MAYBE_UNUSED const REAL_SIMD_ARRAY Cart_originx = ConstSIMD(NOSIMDCart_originx); // nrpy.grid::Cart_originx
const REAL NOSIMDCart_originy = params->Cart_originy;                            // nrpy.grid::Cart_originy
MAYBE_UNUSED const REAL_SIMD_ARRAY Cart_originy = ConstSIMD(NOSIMDCart_originy); // nrpy.grid::Cart_originy
const REAL NOSIMDCart_originz = params->Cart_originz;                            // nrpy.grid::Cart_originz
MAYBE_UNUSED const REAL_SIMD_ARRAY Cart_originz = ConstSIMD(NOSIMDCart_originz); // nrpy.grid::Cart_originz
const REAL NOSIMDCFL_FACTOR = commondata->CFL_FACTOR;                        // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::CFL_FACTOR
MAYBE_UNUSED const REAL_SIMD_ARRAY CFL_FACTOR = ConstSIMD(NOSIMDCFL_FACTOR); // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::CFL_FACTOR
const REAL NOSIMDdt = commondata->dt;                                        // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::dt
MAYBE_UNUSED const REAL_SIMD_ARRAY dt = ConstSIMD(NOSIMDdt);                 // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::dt
const REAL NOSIMDdxx0 = params->dxx0;                                        // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::dxx0
MAYBE_UNUSED const REAL_SIMD_ARRAY dxx0 = ConstSIMD(NOSIMDdxx0);             // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::dxx0
const REAL NOSIMDdxx1 = params->dxx1;                                        // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::dxx1
MAYBE_UNUSED const REAL_SIMD_ARRAY dxx1 = ConstSIMD(NOSIMDdxx1);             // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::dxx1
const REAL NOSIMDdxx2 = params->dxx2;                                        // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::dxx2
MAYBE_UNUSED const REAL_SIMD_ARRAY dxx2 = ConstSIMD(NOSIMDdxx2);             // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::dxx2
const REAL NOSIMDeta_damping = commondata->eta_damping;                      // nrpy.infrastructures.BHaH.BHaHAHA.rhs_eval_KO_apply::eta_damping
MAYBE_UNUSED const REAL_SIMD_ARRAY eta_damping = ConstSIMD(NOSIMDeta_damping); // nrpy.infrastructures.BHaH.BHaHAHA.rhs_eval_KO_apply::eta_damping
const REAL NOSIMDexternal_input_dxx0 =
    commondata->external_input_dxx0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx0
MAYBE_UNUSED const REAL_SIMD_ARRAY external_input_dxx0 =
    ConstSIMD(NOSIMDexternal_input_dxx0); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx0
const REAL NOSIMDexternal_input_dxx1 =
    commondata->external_input_dxx1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx1
MAYBE_UNUSED const REAL_SIMD_ARRAY external_input_dxx1 =
    ConstSIMD(NOSIMDexternal_input_dxx1); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx1
const REAL NOSIMDexternal_input_dxx2 =
    commondata->external_input_dxx2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx2
MAYBE_UNUSED const REAL_SIMD_ARRAY external_input_dxx2 =
    ConstSIMD(NOSIMDexternal_input_dxx2); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_dxx2
const REAL NOSIMDexternal_input_invdxx0 =
    commondata->external_input_invdxx0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx0
MAYBE_UNUSED const REAL_SIMD_ARRAY external_input_invdxx0 =
    ConstSIMD(NOSIMDexternal_input_invdxx0); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx0
const REAL NOSIMDexternal_input_invdxx1 =
    commondata->external_input_invdxx1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx1
MAYBE_UNUSED const REAL_SIMD_ARRAY external_input_invdxx1 =
    ConstSIMD(NOSIMDexternal_input_invdxx1); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx1
const REAL NOSIMDexternal_input_invdxx2 =
    commondata->external_input_invdxx2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx2
MAYBE_UNUSED const REAL_SIMD_ARRAY external_input_invdxx2 =
    ConstSIMD(NOSIMDexternal_input_invdxx2); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_invdxx2
MAYBE_UNUSED const int external_input_Nxx0 =
    commondata->external_input_Nxx0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx0
MAYBE_UNUSED const int external_input_Nxx1 =
    commondata->external_input_Nxx1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx1
MAYBE_UNUSED const int external_input_Nxx2 =
    commondata->external_input_Nxx2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx2
MAYBE_UNUSED const int external_input_Nxx_plus_2NGHOSTS0 =
    commondata
        ->external_input_Nxx_plus_2NGHOSTS0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx_plus_2NGHOSTS0
MAYBE_UNUSED const int external_input_Nxx_plus_2NGHOSTS1 =
    commondata
        ->external_input_Nxx_plus_2NGHOSTS1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx_plus_2NGHOSTS1
MAYBE_UNUSED const int external_input_Nxx_plus_2NGHOSTS2 =
    commondata
        ->external_input_Nxx_plus_2NGHOSTS2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__external_input_set_up::external_input_Nxx_plus_2NGHOSTS2
const REAL NOSIMDgrid_hole_radius = params->grid_hole_radius;                                // nrpy.reference_metric::grid_hole_radius
MAYBE_UNUSED const REAL_SIMD_ARRAY grid_hole_radius = ConstSIMD(NOSIMDgrid_hole_radius);     // nrpy.reference_metric::grid_hole_radius
const REAL NOSIMDgrid_physical_size = params->grid_physical_size;                            // nrpy.reference_metric::grid_physical_size
MAYBE_UNUSED const REAL_SIMD_ARRAY grid_physical_size = ConstSIMD(NOSIMDgrid_physical_size); // nrpy.reference_metric::grid_physical_size
MAYBE_UNUSED const bool grid_rotates = params->grid_rotates;                                 // nrpy.grid::grid_rotates
const REAL NOSIMDinterp_src_dxx0 = commondata->interp_src_dxx0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx0
MAYBE_UNUSED const REAL_SIMD_ARRAY interp_src_dxx0 =
    ConstSIMD(NOSIMDinterp_src_dxx0);                           // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx0
const REAL NOSIMDinterp_src_dxx1 = commondata->interp_src_dxx1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx1
MAYBE_UNUSED const REAL_SIMD_ARRAY interp_src_dxx1 =
    ConstSIMD(NOSIMDinterp_src_dxx1);                           // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx1
const REAL NOSIMDinterp_src_dxx2 = commondata->interp_src_dxx2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx2
MAYBE_UNUSED const REAL_SIMD_ARRAY interp_src_dxx2 =
    ConstSIMD(NOSIMDinterp_src_dxx2); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_dxx2
const REAL NOSIMDinterp_src_invdxx0 =
    commondata->interp_src_invdxx0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx0
MAYBE_UNUSED const REAL_SIMD_ARRAY interp_src_invdxx0 =
    ConstSIMD(NOSIMDinterp_src_invdxx0); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx0
const REAL NOSIMDinterp_src_invdxx1 =
    commondata->interp_src_invdxx1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx1
MAYBE_UNUSED const REAL_SIMD_ARRAY interp_src_invdxx1 =
    ConstSIMD(NOSIMDinterp_src_invdxx1); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx1
const REAL NOSIMDinterp_src_invdxx2 =
    commondata->interp_src_invdxx2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx2
MAYBE_UNUSED const REAL_SIMD_ARRAY interp_src_invdxx2 =
    ConstSIMD(NOSIMDinterp_src_invdxx2); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_invdxx2
MAYBE_UNUSED const int interp_src_Nxx0 = commondata->interp_src_Nxx0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx0
MAYBE_UNUSED const int interp_src_Nxx1 = commondata->interp_src_Nxx1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx1
MAYBE_UNUSED const int interp_src_Nxx2 = commondata->interp_src_Nxx2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx2
MAYBE_UNUSED const int interp_src_Nxx_plus_2NGHOSTS0 =
    commondata->interp_src_Nxx_plus_2NGHOSTS0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx_plus_2NGHOSTS0
MAYBE_UNUSED const int interp_src_Nxx_plus_2NGHOSTS1 =
    commondata->interp_src_Nxx_plus_2NGHOSTS1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx_plus_2NGHOSTS1
MAYBE_UNUSED const int interp_src_Nxx_plus_2NGHOSTS2 =
    commondata->interp_src_Nxx_plus_2NGHOSTS2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__interp_src_set_up::interp_src_Nxx_plus_2NGHOSTS2
const REAL NOSIMDinvdxx0 = params->invdxx0;    // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::invdxx0
MAYBE_UNUSED const REAL_SIMD_ARRAY invdxx0 = ConstSIMD(NOSIMDinvdxx0); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::invdxx0
const REAL NOSIMDinvdxx1 = params->invdxx1;                            // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::invdxx1
MAYBE_UNUSED const REAL_SIMD_ARRAY invdxx1 = ConstSIMD(NOSIMDinvdxx1); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::invdxx1
const REAL NOSIMDinvdxx2 = params->invdxx2;                            // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::invdxx2
MAYBE_UNUSED const REAL_SIMD_ARRAY invdxx2 = ConstSIMD(NOSIMDinvdxx2); // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::invdxx2
const REAL NOSIMDKO_diss_strength = commondata->KO_diss_strength;      // nrpy.infrastructures.BHaH.BHaHAHA.rhs_eval_KO_apply::KO_diss_strength
MAYBE_UNUSED const REAL_SIMD_ARRAY KO_diss_strength =
    ConstSIMD(NOSIMDKO_diss_strength); // nrpy.infrastructures.BHaH.BHaHAHA.rhs_eval_KO_apply::KO_diss_strength
const REAL NOSIMDmax_radius_wrt_grid_center =
    commondata->max_radius_wrt_grid_center; // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics_area_centroid_and_Theta_norms::max_radius_wrt_grid_center
MAYBE_UNUSED const REAL_SIMD_ARRAY max_radius_wrt_grid_center = ConstSIMD(
    NOSIMDmax_radius_wrt_grid_center); // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics_area_centroid_and_Theta_norms::max_radius_wrt_grid_center
const REAL NOSIMDmin_radius_wrt_grid_center =
    commondata->min_radius_wrt_grid_center; // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics_area_centroid_and_Theta_norms::min_radius_wrt_grid_center
MAYBE_UNUSED const REAL_SIMD_ARRAY min_radius_wrt_grid_center = ConstSIMD(
    NOSIMDmin_radius_wrt_grid_center);      // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics_area_centroid_and_Theta_norms::min_radius_wrt_grid_center
MAYBE_UNUSED const int nn = commondata->nn; // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::nn
MAYBE_UNUSED const int nn_0 = commondata->nn_0;                         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::nn_0
MAYBE_UNUSED const int NUMGRIDS = commondata->NUMGRIDS;                 // nrpy.grid::NUMGRIDS
MAYBE_UNUSED const int Nxx0 = params->Nxx0;                             // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::Nxx0
MAYBE_UNUSED const int Nxx1 = params->Nxx1;                             // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::Nxx1
MAYBE_UNUSED const int Nxx2 = params->Nxx2;                             // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::Nxx2
MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::Nxx_plus_2NGHOSTS0
MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::Nxx_plus_2NGHOSTS1
MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2; // nrpy.infrastructures.BHaH.BHaHAHA.numgrid__evol_set_up::Nxx_plus_2NGHOSTS2
MAYBE_UNUSED const int output_diagnostics_every_nn =
    commondata->output_diagnostics_every_nn;                           // nrpy.infrastructures.BHaH.BHaHAHA.diagnostics::output_diagnostics_every_nn
const REAL NOSIMDPI = params->PI;                                      // nrpy.reference_metric::PI
MAYBE_UNUSED const REAL_SIMD_ARRAY PI = ConstSIMD(NOSIMDPI);           // nrpy.reference_metric::PI
const REAL NOSIMDRMAX = params->RMAX;                                  // nrpy.reference_metric::RMAX
MAYBE_UNUSED const REAL_SIMD_ARRAY RMAX = ConstSIMD(NOSIMDRMAX);       // nrpy.reference_metric::RMAX
const REAL NOSIMDt_0 = commondata->t_0;                                // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_0
MAYBE_UNUSED const REAL_SIMD_ARRAY t_0 = ConstSIMD(NOSIMDt_0);         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_0
const REAL NOSIMDt_final = commondata->t_final;                        // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_final
MAYBE_UNUSED const REAL_SIMD_ARRAY t_final = ConstSIMD(NOSIMDt_final); // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_final
const REAL NOSIMDtime = commondata->time;                              // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::time
MAYBE_UNUSED const REAL_SIMD_ARRAY time = ConstSIMD(NOSIMDtime);       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::time
