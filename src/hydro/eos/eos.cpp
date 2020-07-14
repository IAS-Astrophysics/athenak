//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//  \brief implements constructors for all EOS base and derived classes

#include <float.h>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "parameter_input.hpp"
#include "eos.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(Hydro *phyd, std::unique_ptr<ParameterInput> &pin) :
    pmy_hydro(phyd) {

  density_floor_ = pin->GetOrAddReal("eos","density_floor",(FLT_MIN));
  pressure_floor_ = pin->GetOrAddReal("eos","pressure_floor",(FLT_MIN));
}

//----------------------------------------------------------------------------------------
// AdiabaticHydro constructor

AdiabaticHydro::AdiabaticHydro(Hydro *phyd, std::unique_ptr<ParameterInput> &pin)
  : EquationOfState(phyd, pin) {

  gamma_ = pin->GetReal("eos", "gamma");

}

} // namespace hydro
