#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Method of Lines (MoL) for "SSPRK33" method: Free memory for "non_y_n_gfs" gridfunctions
 * - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
 * - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
 *
 */
void bah_MoL_free_memory_non_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs) {
  BHAH_FREE(gridfuncs->next_y_input_gfs);
  BHAH_FREE(gridfuncs->k1_gfs);
  BHAH_FREE(gridfuncs->k2_gfs);
  BHAH_FREE(gridfuncs->k3_gfs);
  if (NUM_AUXEVOL_GFS > 0)
    BHAH_FREE(gridfuncs->auxevol_gfs);
} // END FUNCTION bah_MoL_free_memory_non_y_n_gfs
