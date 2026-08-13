#ifndef DIFFUSION_STS_TYPES_HPP_
#define DIFFUSION_STS_TYPES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sts_types.hpp
//! \brief Shared types for selecting parabolic integration methods.

namespace parabolic {

enum class STSIntegrator {none, rkl2};
enum class DiffusionSelection {explicit_only, sts_only};
enum class ParabolicProcessOwner {hydro, mhd};

} // namespace parabolic

#endif // DIFFUSION_STS_TYPES_HPP_
