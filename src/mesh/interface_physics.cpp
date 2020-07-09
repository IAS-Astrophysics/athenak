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


using namespace hydro;
//----------------------------------------------------------------------------------------

void Mesh::SelectPhysics(std::unique_ptr<ParameterInput> &pin) {

  // parse input blocks to see which physics defined
  bool hydro_defined = pin->DoesBlockExist("hydro");

  // Construct hydro module
  if (hydro_defined) {
    // loop through MeshBlocks on this rank and initialize Hydro
    for (auto it = mblocks.begin(); it < mblocks.end(); ++it) {
      it->phydro = new hydro::Hydro(&*it, pin);
    }
  } else {
    std::cout << "Hydro block not found in input file" << std::endl;
  }
}
