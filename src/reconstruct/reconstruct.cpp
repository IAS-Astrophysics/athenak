//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.cpp
//  \brief implements constructors for all Reconstruction base and derived classes

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "parameter_input.hpp"
#include "reconstruct.hpp"


//----------------------------------------------------------------------------------------
// Reconstruction constructor

Reconstruction::Reconstruction(std::unique_ptr<ParameterInput> &pin) {

}

//----------------------------------------------------------------------------------------
// DonorCell constructor

DonorCell::DonorCell(std::unique_ptr<ParameterInput> &pin) : Reconstruction(pin) {

}
