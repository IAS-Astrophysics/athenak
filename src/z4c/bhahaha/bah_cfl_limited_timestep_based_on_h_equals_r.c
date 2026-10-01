#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Compute minimum timestep dt = CFL_FACTOR * ds_min on a 2D spherical numerical grid.
 */
void bah_cfl_limited_timestep_based_on_h_equals_r(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  commondata->dt = 1e30;
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    const params_struct *restrict params = &griddata[grid].params;
    const REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
    REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++) {
      xx[ww] = griddata[grid].xx[ww];
    }

#include "set_CodeParameters.h"

    REAL ds_min = 1e38;
#pragma omp parallel for reduction(min : ds_min)
    LOOP_NOOMP(i0, NGHOSTS, Nxx0 + NGHOSTS, i1, NGHOSTS, Nxx1 + NGHOSTS, i2, NGHOSTS, Nxx2 + NGHOSTS) {
      const REAL hh = in_gfs[IDX4(HHGF, i0, i1, i2)];
      const REAL xx1 = xx[1][i1];
      REAL dsmin1, dsmin2;

      dsmin1 = fabs(hh * dxx1);
      dsmin2 = fabs(hh * dxx2 * sin(xx1));
      ds_min = MIN(ds_min, MIN(dsmin1, dsmin2));
    }
    commondata->dt = MIN(commondata->dt, ds_min * commondata->CFL_FACTOR);
  }
} // END FUNCTION bah_cfl_limited_timestep_based_on_h_equals_r
