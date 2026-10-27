//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.cpp
//  \brief implementation of the nuclear integrationn class

#include "nuclear/nuclear.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"

namespace nuclear {
  Nuclear::Nuclear(MeshBlockPack *ppack, ParameterInput *pin)
    : pmy_pack(ppack) {

    ishydro = pin->DoesBlockExist("hydro");
    isunits = pin->DoesBlockExist("units");
    ismhd = pin->DoesBlockExist("mhd");

    // Maximum number of network substeps per hydro step. If dt_nuc asks for more, the
    // substeps are enlarged to dt/max_substeps (and a warning is printed).
    max_substeps = pin->GetOrAddInteger("nuclear", "max_substeps", 100);
  }

  Nuclear::~Nuclear() {}
}

