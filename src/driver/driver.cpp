//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.cpp
//  \brief implementation of functions in class Driver

#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "driver.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Driver::Driver(std::unique_ptr<ParameterInput> &pin, std::unique_ptr<Mesh> &pmesh,
     std::unique_ptr<Outputs> &pout) {

}

//----------------------------------------------------------------------------------------
// dtor

//----------------------------------------------------------------------------------------
// Driver::Execute()

void Driver::Execute(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout) {

  // cycle through output Types and load data
  pout->poutput_list_.front()->LoadOutputData(pmesh);

  // cycle through output Types and write files.  This design allows for asynchronous 
  // outputs to implemented in the future.
  pout->poutput_list_.front()->WriteOutputFile(pmesh);

}
