//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file interface_physics.cpp
//  \brief 

#include <iostream>

#include "parameter_input.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------

void Mesh::SelectPhysics(std::unique_ptr<ParameterInput> &pin) {

  bool hydro_defined = pin->DoesBlockExist("hydro");
  if (hydro_defined) {
    for (auto it = mblocks.begin(); it < mblocks.end(); ++it) {
      it->phydro = new Hydro(&*it,pin);
    }
  } else {
    std::cout << "Hydro block not found in input file" << std::endl;
  }
}
