#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

#define LOOP_ALL_GFS_GPS(ii)                                                                                                                         \
  _Pragma("omp parallel for") for (int(ii) = 0;                                                                                                      \
                                   (ii) < params->Nxx_plus_2NGHOSTS0 * params->Nxx_plus_2NGHOSTS1 * params->Nxx_plus_2NGHOSTS2 * NUM_EVOL_GFS;       \
                                   (ii)++)
/**
 * Kernel: rk_substep_1_host.
 * Compute RK substep 1.
 */
static void rk_substep_1_host(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs,
                              const REAL dt) {
  LOOP_ALL_GFS_GPS(i) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    next_y_input_gfs[i] = dt * k1_gfsL + y_n_gfsL;
  }
} // END FUNCTION rk_substep_1_host

/**
 * Runge-Kutta function for substep 1.
 */
static void rk_substep_1__launcher(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict y_n_gfs, REAL *restrict next_y_input_gfs,
                                   const REAL dt) {
  rk_substep_1_host(params, k1_gfs, y_n_gfs, next_y_input_gfs, dt);
} // END FUNCTION rk_substep_1__launcher

/**
 * Kernel: rk_substep_2_host.
 * Compute RK substep 2.
 */
static void rk_substep_2_host(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict y_n_gfs,
                              REAL *restrict next_y_input_gfs, const REAL dt) {
  LOOP_ALL_GFS_GPS(i) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    const REAL RK_Rational_1_4 = 1.0 / 4.0;
    next_y_input_gfs[i] = RK_Rational_1_4 * (dt * k1_gfsL + dt * k2_gfsL) + y_n_gfsL;
  }
} // END FUNCTION rk_substep_2_host

/**
 * Runge-Kutta function for substep 2.
 */
static void rk_substep_2__launcher(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict y_n_gfs,
                                   REAL *restrict next_y_input_gfs, const REAL dt) {
  rk_substep_2_host(params, k1_gfs, k2_gfs, y_n_gfs, next_y_input_gfs, dt);
} // END FUNCTION rk_substep_2__launcher

/**
 * Kernel: rk_substep_3_host.
 * Compute RK substep 3.
 */
static void rk_substep_3_host(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs,
                              REAL *restrict y_n_gfs, const REAL dt) {
  LOOP_ALL_GFS_GPS(i) {
    const REAL k1_gfsL = k1_gfs[i];
    const REAL k2_gfsL = k2_gfs[i];
    const REAL k3_gfsL = k3_gfs[i];
    const REAL y_n_gfsL = y_n_gfs[i];
    const REAL RK_Rational_1_6 = 1.0 / 6.0;
    const REAL RK_Rational_2_3 = 2.0 / 3.0;
    y_n_gfs[i] = RK_Rational_1_6 * (dt * k1_gfsL + dt * k2_gfsL) + RK_Rational_2_3 * dt * k3_gfsL + y_n_gfsL;
  }
} // END FUNCTION rk_substep_3_host

/**
 * Runge-Kutta function for substep 3.
 */
static void rk_substep_3__launcher(params_struct *restrict params, REAL *restrict k1_gfs, REAL *restrict k2_gfs, REAL *restrict k3_gfs,
                                   REAL *restrict y_n_gfs, const REAL dt) {
  rk_substep_3_host(params, k1_gfs, k2_gfs, k3_gfs, y_n_gfs, dt);
} // END FUNCTION rk_substep_3__launcher

/**
 * Method of Lines (MoL) for "SSPRK33" method: Step forward one full timestep.
 *
 */
void bah_MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  // C code implementation of -={ SSPRK33 }=- Method of Lines timestepping.

  // First set the initial time:
  const REAL time_start = commondata->time;
  // -={ START k1 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 0.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases, from griddata[].gridfuncs.
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    // Set pointers to this grid's params, rfm_struct/xx, bc_struct, etc.
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED const rfm_struct *restrict rfmstruct = griddata[grid].rfmstruct;
    MAYBE_UNUSED const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;

    bah_rhs_eval(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, k1_gfs);
    // Theta was just evaluated at Nxx1*Nxx2 gridpoints; update counter:
    commondata->bhahaha_diagnostics->Theta_eval_points_counter += params->Nxx1 * params->Nxx2;
    if (commondata->KO_diss_strength > 0.0)
      bah_KO_apply(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, k1_gfs);

    rk_substep_1__launcher(params, k1_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    bah_apply_bcs_inner_only(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 1.00000000000000000e+00 * commondata->dt;
    // Set gridfunction aliases, from griddata[].gridfuncs.
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    // Set pointers to this grid's params, rfm_struct/xx, bc_struct, etc.
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED const rfm_struct *restrict rfmstruct = griddata[grid].rfmstruct;
    MAYBE_UNUSED const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;

    bah_rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k2_gfs);
    // Theta was just evaluated at Nxx1*Nxx2 gridpoints; update counter:
    commondata->bhahaha_diagnostics->Theta_eval_points_counter += params->Nxx1 * params->Nxx2;
    if (commondata->KO_diss_strength > 0.0)
      bah_KO_apply(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k2_gfs);

    rk_substep_2__launcher(params, k1_gfs, k2_gfs, y_n_gfs, next_y_input_gfs, commondata->dt);
    bah_apply_bcs_inner_only(commondata, params, bcstruct, next_y_input_gfs);
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    commondata->time = time_start + 5.00000000000000000e-01 * commondata->dt;
    // Set gridfunction aliases, from griddata[].gridfuncs.
    MAYBE_UNUSED REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
    MAYBE_UNUSED REAL *restrict next_y_input_gfs = griddata[grid].gridfuncs.next_y_input_gfs;
    MAYBE_UNUSED REAL *restrict k1_gfs = griddata[grid].gridfuncs.k1_gfs;
    MAYBE_UNUSED REAL *restrict k2_gfs = griddata[grid].gridfuncs.k2_gfs;
    MAYBE_UNUSED REAL *restrict k3_gfs = griddata[grid].gridfuncs.k3_gfs;
    MAYBE_UNUSED REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
    // Set pointers to this grid's params, rfm_struct/xx, bc_struct, etc.
    MAYBE_UNUSED params_struct *restrict params = &griddata[grid].params;
    MAYBE_UNUSED const rfm_struct *restrict rfmstruct = griddata[grid].rfmstruct;
    MAYBE_UNUSED const bc_struct *restrict bcstruct = &griddata[grid].bcstruct;

    bah_rhs_eval(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k3_gfs);
    // Theta was just evaluated at Nxx1*Nxx2 gridpoints; update counter:
    commondata->bhahaha_diagnostics->Theta_eval_points_counter += params->Nxx1 * params->Nxx2;
    if (commondata->KO_diss_strength > 0.0)
      bah_KO_apply(commondata, params, rfmstruct, auxevol_gfs, next_y_input_gfs, k3_gfs);

    rk_substep_3__launcher(params, k1_gfs, k2_gfs, k3_gfs, y_n_gfs, commondata->dt);
    bah_apply_bcs_inner_only(commondata, params, bcstruct, y_n_gfs);
  }
  // -={ END k3 substep }=-

  // Adding dt to commondata->time many times will induce roundoff error,
  // so here we set time based on the iteration number:
  commondata->time = (REAL)(commondata->nn + 1) * commondata->dt;

  // Increment the timestep n:
  commondata->nn++;
} // END FUNCTION bah_MoL_step_forward_in_time
