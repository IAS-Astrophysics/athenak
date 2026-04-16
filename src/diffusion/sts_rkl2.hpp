#ifndef DIFFUSION_STS_RKL2_HPP_
#define DIFFUSION_STS_RKL2_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sts_rkl2.hpp
//! \brief Host-side helpers for RKL2 super time stepping stage-count and coefficient
//! evaluation.

#include "athena.hpp"

namespace parabolic {

struct RKL2Coefficients {
  Real muj = 0.0;
  Real nuj = 0.0;
  Real muj_tilde = 0.0;
  Real gammaj_tilde = 0.0;
};

int ComputeRKL2StageCount(Real dt_sweep, Real dt_parabolic_min);
RKL2Coefficients ComputeRKL2Coefficients(int stage, int nstages);

} // namespace parabolic

#endif // DIFFUSION_STS_RKL2_HPP_
