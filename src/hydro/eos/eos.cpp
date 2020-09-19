//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//  \brief implements ctor and fns for EquationOfState abstract base class

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "eos.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(Mesh* pm,ParameterInput *pin,int igid)
   : pmesh_(pm), my_mbgid_(igid)
{
  density_floor_ = pin->GetOrAddReal("eos","density_floor",(FLT_MIN));
  pressure_floor_ = pin->GetOrAddReal("eos","pressure_floor",(FLT_MIN));
}

//----------------------------------------------------------------------------------------
// EquationOfState::SoundSpeed()

Real EquationOfState::SoundSpeed(Real prim[5])
{
  return (0.0);
}

//----------------------------------------------------------------------------------------
// EquationOfState::SoundSpeed()

Real EquationOfState::SoundSpeed()
{
  return (0.0);
}

} // namespace hydro

