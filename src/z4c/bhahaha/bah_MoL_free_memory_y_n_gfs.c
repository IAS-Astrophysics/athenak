#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Method of Lines (MoL) for "SSPRK33" method: Free memory for "y_n_gfs" gridfunctions
 * - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
 * - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
 *
 */
void bah_MoL_free_memory_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs) {
  BHAH_FREE(gridfuncs->y_n_gfs);
} // END FUNCTION bah_MoL_free_memory_y_n_gfs
