//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.cpp
//  \brief implements constructors for all Reconstruction base and derived classes

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
// Reconstruction constructor

Reconstruction::Reconstruction(ParameterInput *pin, int nvar, int ncells1) :
  nvar_(nvar), ncells1_(ncells1)
{
}
