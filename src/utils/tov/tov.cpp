//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov.cpp
//  \brief Implementation for TOVStar class.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tov.hpp"
#include "tov_utils.hpp"

namespace tov {

TOVStar::TOVStar(ParameterInput* pin) {
  rhoc = pin->GetReal("problem", "rhoc");
  npoints = pin->GetReal("problem", "npoints");
  dr = pin->GetReal("problem", "dr");

  dfloor = pin->GetOrAddReal("problem", "rho_cut", rhoc*1e-10);
}

} // namespace tov
