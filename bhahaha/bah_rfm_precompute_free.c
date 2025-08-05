#include "BHaH_defines.h"
/**
 * rfm_precompute_free: reference metric precomputed lookup arrays: free
 */
void bah_rfm_precompute_free(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct) {
  BHAH_FREE__PtrMember(rfmstruct, f0_of_xx0);
  BHAH_FREE__PtrMember(rfmstruct, f1_of_xx1);
  BHAH_FREE__PtrMember(rfmstruct, f1_of_xx1__D1);
  BHAH_FREE__PtrMember(rfmstruct, f1_of_xx1__DD11);
} // END FUNCTION bah_rfm_precompute_free
