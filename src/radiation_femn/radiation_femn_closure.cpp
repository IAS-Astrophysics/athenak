//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_closure.cpp
//  \brief Calculate closures for M1

#include "athena.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

void ApplyClosure(TeamMember_t member, int num_points, int m, int en, int kk, int jj, int ii, DvceArray5D<Real> f0, ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1,
                                         ScrArray1D<Real> f0_scratch_p2, ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1,
                                         ScrArray1D<Real> f0_scratch_m2) {

  int nang1 = num_points - 1;

  par_for_inner(member, 0, nang1, [&](const int idx) {
    f0_scratch(idx) = f0(m, en * num_points + idx, kk, jj, ii);
    f0_scratch_p1(idx) = f0(m, en * num_points + idx, kk, jj, ii + 1);
    f0_scratch_p2(idx) = f0(m, en * num_points + idx, kk, jj, ii + 2);
    f0_scratch_p3(idx) = f0(m, en * num_points + idx, kk, jj, ii + 3);
    f0_scratch_m1(idx) = f0(m, en * num_points + idx, kk, jj, ii - 1);
    f0_scratch_m2(idx) = f0(m, en * num_points + idx, kk, jj, ii - 2);
  });

};

KOKKOS_INLINE_FUNCTION void ApplyRestriction() {

};

}