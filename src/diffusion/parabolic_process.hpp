#ifndef DIFFUSION_PARABOLIC_PROCESS_HPP_
#define DIFFUSION_PARABOLIC_PROCESS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parabolic_process.hpp
//! \brief Metadata container describing parabolic processes that may later participate in
//! super time stepping.

#include <cassert>
#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"

namespace parabolic {

struct ParabolicProcessDescriptor {
  std::string name;
  ParabolicProcessOwner owner;
  ParabolicIntegratorMode mode;
  ParabolicUpdateShape update_shape;
  const Real* explicit_dt_ptr;

  bool UsesSTS() const {
    return mode == ParabolicIntegratorMode::sts;
  }

  Real ExplicitDt() const {
    assert(explicit_dt_ptr != nullptr);
    return *explicit_dt_ptr;
  }
};

} // namespace parabolic

#endif // DIFFUSION_PARABOLIC_PROCESS_HPP_
