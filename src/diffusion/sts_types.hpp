#ifndef DIFFUSION_STS_TYPES_HPP_
#define DIFFUSION_STS_TYPES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sts_types.hpp
//! \brief Shared STS/parabolic metadata types used by diffusion-related infrastructure.

namespace parabolic {

enum class STSIntegrator {none, rkl2};

enum class ParabolicIntegratorMode {explicit_mode, sts};

enum class ParabolicProcessOwner {hydro, mhd, radiation, other};

enum class ParabolicUpdateShape {cell_centered, face_centered, cell_and_face};

} // namespace parabolic

#endif // DIFFUSION_STS_TYPES_HPP_
