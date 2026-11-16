#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 *
 * Performs up to quadratic extrapolation using a Lagrange-based formula.
 *
 * This function calculates the y-value at a specified destination time
 * (dst_time) based on the given y-values and their corresponding time points.
 * It uses a quadratic formula derived from Lagrange polynomials when all
 * three time points are available, falls back to linear extrapolation when
 * only two points are available, and defaults to the value at the earliest
 * time point if insufficient data is provided.
 *
 */
REAL bah_quadratic_extrapolation(const REAL times[3], const REAL y_tm1, const REAL y_tm2, const REAL y_tm3, const REAL dst_time) {

  const REAL tm1 = times[0];
  const REAL tm2 = times[1];
  const REAL tm3 = times[2];

  // Check if all three time points are populated for quadratic extrapolation
  if (tm1 != -1.0 && tm2 != -1.0 && tm3 != -1.0) {
    return (y_tm1 * (dst_time - tm2) * (dst_time - tm3) / ((tm1 - tm2) * (tm1 - tm3))) +
           (y_tm2 * (dst_time - tm1) * (dst_time - tm3) / ((tm2 - tm1) * (tm2 - tm3))) +
           (y_tm3 * (dst_time - tm1) * (dst_time - tm2) / ((tm3 - tm1) * (tm3 - tm2)));
  }
  // Check if only tm1 and tm2 are populated for linear extrapolation
  else if (tm1 != -1.0 && tm2 != -1.0 && tm3 == -1.0) {
    // Linear extrapolation formula
    return y_tm1 + (y_tm2 - y_tm1) * (dst_time - tm1) / (tm2 - tm1);
  }
  // Return zeroth-order estimate if insufficient data for extrapolation
  else {
    return y_tm1;
  }
} // END FUNCTION bah_quadratic_extrapolation
