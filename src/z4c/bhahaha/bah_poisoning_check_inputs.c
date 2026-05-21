#include "BHaHAHA.h"
#include "math.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"

// -------------------------------------------------
// is_nan() is needed for ffast-math compatibility;
//   isnan() does not function with ffast-math.
// -------------------------------------------------
static inline int is_nan(double x) {
  union {
    double d;
    uint64_t u;
  } value;
  value.d = x;
  uint64_t exponent = (value.u >> 52) & 0x7FF;
  uint64_t mantissa = value.u & 0xFFFFFFFFFFFFF;
  // NaN: exponent=2047 (0x7FF) and mantissa!=0
  return (exponent == 0x7FF) && (mantissa != 0);
}

// -------------------
// Checking macros
// -------------------
#define CHECK_REAL(x) (is_nan(x))
#define CHECK_PTR(x) ((x) == NULL)
#define CHECK_INT(x) ((x) == -100)

#define CHECK_REAL_ARRAY(arr, size, name, errflag)                                                                                                   \
  for (int i = 0; i < (size); i++) {                                                                                                                 \
    if (CHECK_REAL((arr)[i])) {                                                                                                                      \
      fprintf(stderr, "Error: %s[%d] is NaN.\n", (name), i);                                                                                         \
      (errflag) = 1;                                                                                                                                 \
    }                                                                                                                                                \
  }

#define CHECK_INT_ARRAY(arr, size, name, errflag)                                                                                                    \
  for (int i = 0; i < (size); i++) {                                                                                                                 \
    if (CHECK_INT((arr)[i])) {                                                                                                                       \
      fprintf(stderr, "Error: %s[%d] is poisoned (-100).\n", (name), i);                                                                             \
      (errflag) = 1;                                                                                                                                 \
    }                                                                                                                                                \
  }

// Helper macros to reduce repeated checks in poisoning_check_inputs:
#define CHECK_AND_PRINT_PTR(var, varname, errflag)                                                                                                   \
  do {                                                                                                                                               \
    if (CHECK_PTR(var)) {                                                                                                                            \
      fprintf(stderr, "Error: %s is NULL.\n", varname);                                                                                              \
      (errflag) = 1;                                                                                                                                 \
    }                                                                                                                                                \
  } while (0)

#define CHECK_AND_PRINT_REAL(var, varname, errflag)                                                                                                  \
  do {                                                                                                                                               \
    if (CHECK_REAL(var)) {                                                                                                                           \
      fprintf(stderr, "Error: %s is NaN.\n", varname);                                                                                               \
      (errflag) = 1;                                                                                                                                 \
    }                                                                                                                                                \
  } while (0)

#define CHECK_AND_PRINT_INT(var, varname, errflag)                                                                                                   \
  do {                                                                                                                                               \
    if (CHECK_INT(var)) {                                                                                                                            \
      fprintf(stderr, "Error: %s is poisoned (-1).\n", varname);                                                                                     \
      (errflag) = 1;                                                                                                                                 \
    }                                                                                                                                                \
  } while (0)

/**
 * Check the inputs of bhahaha_params_and_data_struct for poisoned values. Errors out with a useful message if any poisoned values are detected.
 */
void bah_poisoning_check_inputs(const bhahaha_params_and_data_struct *restrict params) {

  if (!params) {
    fprintf(stderr, "poisoning_check_inputs: Received NULL pointer.\n");
    exit(EXIT_FAILURE); // Exits the program with failure status
  }

  int error_flag = 0;

  // Check Metric and grid setup
  CHECK_AND_PRINT_PTR(params->input_metric_data, "input_metric_data", error_flag);

  // Check External Input Numerical Grid: Radial parameters
  CHECK_AND_PRINT_REAL(params->time_external_input, "time_external_input", error_flag);
  CHECK_AND_PRINT_INT(params->iteration_external_input, "iteration_external_input", error_flag);
  CHECK_AND_PRINT_INT(params->Nr_external_input, "Nr_external_input", error_flag);
  CHECK_AND_PRINT_REAL(params->r_min_external_input, "r_min_external_input", error_flag);
  CHECK_AND_PRINT_REAL(params->dr_external_input, "dr_external_input", error_flag);

  CHECK_AND_PRINT_INT(params->num_resolutions_multigrid, "num_resolutions_multigrid", error_flag);

  CHECK_INT_ARRAY(params->Ntheta_array_multigrid, params->num_resolutions_multigrid, "Ntheta_array_multigrid", error_flag);
  CHECK_INT_ARRAY(params->Nphi_array_multigrid, params->num_resolutions_multigrid, "Nphi_array_multigrid", error_flag);

  CHECK_AND_PRINT_INT(params->use_fixed_radius_guess_on_full_sphere, "use_fixed_radius_guess_on_full_sphere", error_flag);

  CHECK_AND_PRINT_REAL(params->cfl_factor, "cfl_factor", error_flag);
  CHECK_AND_PRINT_REAL(params->M_scale, "M_scale", error_flag);
  CHECK_AND_PRINT_REAL(params->eta_damping_times_M, "eta_damping_times_M", error_flag);
  CHECK_AND_PRINT_REAL(params->KO_strength, "KO_strength", error_flag);

  CHECK_AND_PRINT_INT(params->max_iterations, "max_iterations", error_flag);
  CHECK_AND_PRINT_REAL(params->Theta_Linf_times_M_tolerance, "Theta_Linf_times_M_tolerance", error_flag);

  CHECK_AND_PRINT_INT(params->which_horizon, "which_horizon", error_flag);
  CHECK_AND_PRINT_INT(params->num_horizons, "num_horizons", error_flag);

  CHECK_AND_PRINT_INT(params->verbosity_level, "verbosity_level", error_flag);
  CHECK_AND_PRINT_INT(params->enable_eta_varying_alg_for_precision_common_horizon, "enable_eta_varying_alg_for_precision_common_horizon", error_flag);

  if (error_flag) {
    fprintf(stderr, "poisoning_check_inputs: Poisoned inputs detected. Exiting.\n");
    exit(EXIT_FAILURE); // Exits the program with failure status
  }
} // END FUNCTION bah_poisoning_check_inputs
