#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Method of Lines (MoL) for "SSPRK33" method: Allocate memory for "y_n_gfs" gridfunctions
 * - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
 * - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
 */
void bah_MoL_malloc_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                            MoL_gridfunctions_struct *restrict gridfuncs) {
  const int Nxx_plus_2NGHOSTS_tot = params->Nxx_plus_2NGHOSTS0 * params->Nxx_plus_2NGHOSTS1 * params->Nxx_plus_2NGHOSTS2;
  BHAH_MALLOC(gridfuncs->y_n_gfs, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);

  gridfuncs->diagnostic_output_gfs = gridfuncs->k1_gfs;
  gridfuncs->diagnostic_output_gfs2 = gridfuncs->k2_gfs;
} // END FUNCTION bah_MoL_malloc_y_n_gfs
