#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Set up numerical grid for BHaHAHA 2D evolution grids; Nxx0 = Nr = 1.
 */
void bah_numgrid__evol_set_up(commondata_struct *restrict commondata, griddata_struct *restrict griddata, const int Nx_evol_grid[3]) {

  // Step 1: Set default parameters in griddata.params
  bah_params_struct_set_to_default(commondata, griddata);

  // Step 2: Set number of grids to 1 for BHaHAHA
  commondata->NUMGRIDS = 1;

  // Step 3: Initialize grid parameters
  const int grid = 0;
  params_struct *restrict params = &griddata[grid].params;

  params->grid_physical_size = 1.0; // Unused, since h sets the actual radius

  // Set grid sizes from Nx_evol_grid
  params->Nxx0 = Nx_evol_grid[0];
  params->Nxx1 = Nx_evol_grid[1];
  params->Nxx2 = Nx_evol_grid[2];

  // Include ghost zones
  params->Nxx_plus_2NGHOSTS0 = params->Nxx0 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS1 = params->Nxx1 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS2 = params->Nxx2 + 2 * NGHOSTS;

  // Step 4: Set grid boundaries
  params->RMAX = 1.0; // Completely arbitrary; must be set to some reasonable-sized number > 0, just so that BC set up doesn't error out.
  const REAL xxmin0 = 0.0;
  const REAL xxmax0 = params->RMAX;
  const REAL xxmin1 = 0.0;
  const REAL xxmax1 = M_PI;
  const REAL xxmin2 = -M_PI;
  const REAL xxmax2 = M_PI;

  // Step 5: Compute dxx and invdxx
  params->dxx0 = (xxmax0 - xxmin0) / ((REAL)params->Nxx0);
  params->dxx1 = (xxmax1 - xxmin1) / ((REAL)params->Nxx1);
  params->dxx2 = (xxmax2 - xxmin2) / ((REAL)params->Nxx2);

  params->invdxx0 = 1.0 / params->dxx0;
  params->invdxx1 = 1.0 / params->dxx1;
  params->invdxx2 = 1.0 / params->dxx2;

  // Arrays for simplifying loops
  const int Nxx_plus_2NGHOSTS[3] = {params->Nxx_plus_2NGHOSTS0, params->Nxx_plus_2NGHOSTS1, params->Nxx_plus_2NGHOSTS2};
  const REAL xxmin[3] = {xxmin0, xxmin1, xxmin2};
  const REAL dxx[3] = {params->dxx0, params->dxx1, params->dxx2};

  // Step 6: Allocate and initialize cell-centered grid coordinate arrays xx[0], xx[1], xx[2]
  for (int dir = 0; dir < 3; dir++) {
    griddata[grid].xx[dir] = (REAL *restrict)malloc(sizeof(REAL) * Nxx_plus_2NGHOSTS[dir]);
    for (int i = 0; i < Nxx_plus_2NGHOSTS[dir]; i++)
      griddata[grid].xx[dir][i] = xxmin[dir] + ((REAL)(i - NGHOSTS) + (1.0 / 2.0)) * dxx[dir];
  }

  // Step 7: Allocate and define reference-metric precompute lookup arrays.
  griddata[grid].rfmstruct = (rfm_struct *)malloc(sizeof(rfm_struct));
  bah_rfm_precompute_malloc(commondata, params, griddata[grid].rfmstruct);
  bah_rfm_precompute_defines(commondata, params, griddata[grid].rfmstruct, griddata[grid].xx);

  // Step 8: Set up bcstruct, for setting inner boundary conditions (i.e., BCs in theta & phi ghost zones, like theta < 0).
  {
    commondata->bcstruct_dxx0 = params->dxx0;
    commondata->bcstruct_dxx1 = params->dxx1;
    commondata->bcstruct_dxx2 = params->dxx2;
    commondata->bcstruct_Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
    commondata->bcstruct_Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
    commondata->bcstruct_Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
    bah_bcstruct_set_up(commondata, griddata[grid].xx, &griddata[grid].bcstruct);
  }

  // Step 9: Initialize time-stepping parameters
  commondata->nn = 0;
  commondata->nn_0 = 0;
  commondata->t_0 = 0.0;
  commondata->time = 0.0;
} // END FUNCTION bah_numgrid__evol_set_up
