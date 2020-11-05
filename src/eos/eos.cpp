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

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(Mesh* pm, ParameterInput *pin, int igid)
   : pmesh_(pm), my_mbgid_(igid)
{
  eos_data.density_floor = pin->GetOrAddReal("eos","density_floor",(FLT_MIN));
  eos_data.pressure_floor = pin->GetOrAddReal("eos","pressure_floor",(FLT_MIN));

  // construct EOS type (no default)
  std::string eqn_of_state = pin->GetString("hydro","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {
    eos_type_ = EOS_Type::adiabatic_hydro;
    eos_data.is_adiabatic = true;
    eos_data.gamma = pin->GetReal("eos","gamma");
    eos_data.iso_cs = 0.0;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    eos_type_ = EOS_Type::isothermal_hydro;
    eos_data.is_adiabatic = false;
    eos_data.iso_cs = pin->GetReal("eos","iso_sound_speed");
    eos_data.gamma = 0.0;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// ConservedToPrimitive() 

void EquationOfState::ConservedToPrimitive(const AthenaArray4D<Real> &cons,
                                           AthenaArray4D<Real> &prim)
{                  
  switch (eos_type_) {
    case EOS_Type::adiabatic_hydro:
      ConsToPrimAdiHydro(cons, prim);
      break;
    case EOS_Type::isothermal_hydro:
      ConsToPrimIsoHydro(cons, prim);
      break;
    default:
      break; 
  }
  return;
} 
