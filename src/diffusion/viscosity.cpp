//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.cpp
//  \brief implements ctor and fns for Viscosity abstract base class

#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "viscosity.hpp"

//----------------------------------------------------------------------------------------
// Viscosity constructor

Viscosity::Viscosity(MeshBlockPack* pp, ParameterInput *pin)
   : pmy_pack(pp)
{
  dtnew = std::numeric_limits<float>::max();
}
