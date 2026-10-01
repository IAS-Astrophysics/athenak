#include "BHaH_defines.h"
/**
 * rfm_precompute_malloc: reference metric precomputed lookup arrays: malloc
 */
void bah_rfm_precompute_malloc(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct) {
  BHAH_MALLOC__PtrMember(rfmstruct, f0_of_xx0, sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
  BHAH_MALLOC__PtrMember(rfmstruct, f1_of_xx1, sizeof(REAL) * params->Nxx_plus_2NGHOSTS1);
  BHAH_MALLOC__PtrMember(rfmstruct, f1_of_xx1__D1, sizeof(REAL) * params->Nxx_plus_2NGHOSTS1);
  BHAH_MALLOC__PtrMember(rfmstruct, f1_of_xx1__DD11, sizeof(REAL) * params->Nxx_plus_2NGHOSTS1);
} // END FUNCTION bah_rfm_precompute_malloc
