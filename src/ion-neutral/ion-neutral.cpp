//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion-neutral.cpp
//  \brief

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "hydro/hydro.hpp"
#include "ion-neutral.hpp"

namespace ion_neutral {
//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

IonNeutral::IonNeutral(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp) {
  // Read various coefficients
  drag_coeff = pin->GetReal("ion-neutral","drag_coeff");
  ionization_coeff = pin->GetOrAddReal("ion-neutral","ionization_coeff",0.0);
  recombination_coeff = pin->GetOrAddReal("ion-neutral","recombination_coeff",0.0);
}
} // namespace ion_neutral
