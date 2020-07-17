//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rsolver.cpp
//  \brief implements ctor and fns for RiemannSolver base class

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "parameter_input.hpp"
#include "hydro/rsolver/rsolver.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// RSolver constructor

RiemannSolver::RiemannSolver(Hydro *phyd, std::unique_ptr<ParameterInput> &pin) :
   pmy_hydro(phyd) {

}

} // namespace hydro
