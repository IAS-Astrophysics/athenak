//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nuclear.cpp
//  \brief implementation of the Nuclear class

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

    params.max_substeps = pin->GetOrAddInteger("nuclear", "max_substeps", 100);
    params.dt_nuc_factor = pin->GetOrAddReal("nuclear", "dt_nuc_factor", 0.5);

    // Network integration parameters. Converged when, for every species,
    // |d(rho*X)| <= y_atol + y_rtol*|rho*X| (rho*X in g/cm^3).
    params.y_atol        = pin->GetOrAddReal("nuclear", "y_atol", 1.0e-10);
    params.y_rtol        = pin->GetOrAddReal("nuclear", "y_rtol", 1.0e-10);
    params.max_iter      = pin->GetOrAddInteger("nuclear", "max_iter", 20);
    params.be_n_refactor = pin->GetOrAddInteger("nuclear", "n_refactor", 3);

    // Check the EOS and set the cgs <-> code conversion factors (DynGRMHD is built first)
    InitBurn();
  }

  Nuclear::~Nuclear() {}
}

