//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "hydro/eos/eos.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin) : pmy_mblock(pmb) {

  // allocate memory for conserved and primitive variables
  u.SetSize(5, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);
  w.SetSize(5, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);

  // construct EOS object
  peos = new AdiabaticHydro(this, pin);

  // construct reconstruction object

  // construct Riemann solver object


}

} // namespace hydro
