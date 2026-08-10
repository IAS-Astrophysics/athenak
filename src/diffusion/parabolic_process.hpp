#ifndef DIFFUSION_PARABOLIC_PROCESS_HPP_
#define DIFFUSION_PARABOLIC_PROCESS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parabolic_process.hpp
//! \brief Metadata for diffusion processes that may use STS.

#include <cassert>
#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"

namespace parabolic {

struct ParabolicProcessDescriptor {
  std::string name;
  ParabolicProcessOwner owner;
  DiffusionSelection selection;
  const Real *explicit_dt_ptr;

  bool UsesSTS() const {
    return selection == DiffusionSelection::sts_only;
  }

  Real ExplicitDt() const {
    assert(explicit_dt_ptr != nullptr);
    return *explicit_dt_ptr;
  }
};

} // namespace parabolic

#endif // DIFFUSION_PARABOLIC_PROCESS_HPP_
