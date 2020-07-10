//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//  \brief implementation of functions in class ProblemGenerator

#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ProblemGenerator::ProblemGenerator(std::unique_ptr<Mesh> &pmesh,
   std::unique_ptr<ParameterInput> &pin) {

  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  // Set problem generator function to name specied on cmake command line
  //  ... code here ..

  // else, set pgen function to name read from <problem> block in input file
  // only predefined names of functions in pgen.hpp allowed

  if (pgen_fun_name.compare("shock_tube") == 0) {
    pgen_func_ = &ProblemGenerator::ShockTube_; 
  } else if (pgen_fun_name.compare("advection") == 0) {
    pgen_func_ = &ProblemGenerator::Advection_;

  // else, name not set on command line or input file, print warning and quit
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl << "and it was not set by -DPROBLEM option on command line to cmake."
        << std::endl << "Rerun cmake with -DPROBLEM=name to enable custom problem "
        << "generator names" << std::endl;;
    std::exit(EXIT_FAILURE);
  }

  // now cycle through MeshBlocks and call appropriate pgen function
  for (auto it = pmesh->mblocks.begin(); it < pmesh->mblocks.end(); ++it) {
    (this->*pgen_func_)(&*it, pin);
  }
}

//----------------------------------------------------------------------------------------
// dtor

//ProblemGenerator::~ProblemGenerator() {
//}
