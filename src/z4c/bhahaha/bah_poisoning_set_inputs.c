#include "BHaHAHA.h"
#include "math.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"

#define POISON_REAL(x) ((x) = NAN)
#define POISON_PTR(x) ((x) = NULL)
#define POISON_INT(x) ((x) = -100)

#define POISON_REAL_ARRAY(arr, size)                                                                                                                 \
  for (int i = 0; i < (size); i++)                                                                                                                   \
  POISON_REAL((arr)[i])

#define POISON_INT_ARRAY(arr, size)                                                                                                                  \
  for (int i = 0; i < (size); i++)                                                                                                                   \
  POISON_INT((arr)[i])

/**
 * Poison the inputs of bhahaha_params_and_data_struct by setting REALs to NaN, pointers to NULL, and ints to -1.
 */
void bah_poisoning_set_inputs(bhahaha_params_and_data_struct *restrict params) {

  if (!params) {
    fprintf(stderr, "poisoning_set_inputs: Received NULL pointer.\\n");
    exit(EXIT_FAILURE); // Exits the program with failure status
  }

  // Poison the inputs by setting REALs to NaN, pointers to NULL, and ints to -1
  POISON_PTR(params->input_metric_data);

  POISON_REAL(params->time_external_input);
  POISON_INT(params->iteration_external_input);
  POISON_INT(params->Nr_external_input);
  POISON_REAL(params->r_min_external_input);
  POISON_REAL(params->dr_external_input);

  POISON_INT(params->num_resolutions_multigrid);

  POISON_INT_ARRAY(params->Ntheta_array_multigrid, MAX_RESOLUTIONS);
  POISON_INT_ARRAY(params->Nphi_array_multigrid, MAX_RESOLUTIONS);

  POISON_INT(params->use_fixed_radius_guess_on_full_sphere);

  POISON_REAL(params->cfl_factor);
  POISON_REAL(params->M_scale);
  POISON_REAL(params->eta_damping_times_M);
  POISON_REAL(params->KO_strength);

  POISON_INT(params->max_iterations);
  POISON_REAL(params->Theta_Linf_times_M_tolerance);

  POISON_INT(params->which_horizon);
  POISON_INT(params->num_horizons);

  POISON_INT(params->verbosity_level);
  POISON_INT(params->enable_eta_varying_alg_for_precision_common_horizon);
} // END FUNCTION bah_poisoning_set_inputs
