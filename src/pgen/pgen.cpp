//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//  \brief implementation of functions in class ProblemGenerator

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, Driver *pd)
 : pmy_mesh_(pm), pmy_driver_(pd) 
{
#if USER_PROBLEM_ENABLED
  // call user-defined problem generator
  UserProblem(pm->pmb_pack, pin);
#else

  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("shock_tube") == 0) {
    pgen_func_ = &ProblemGenerator::ShockTube_; 
  } else if (pgen_fun_name.compare("advection") == 0) {
    pgen_func_ = &ProblemGenerator::Advection_;
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    pgen_func_ = &ProblemGenerator::LinearWave_;
  } else if (pgen_fun_name.compare("implode") == 0) {
    pgen_func_ = &ProblemGenerator::LWImplode_;
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    pgen_func_ = &ProblemGenerator::OrszagTang_;

  // else, name not set on command line or input file, print warning and quit
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl << "and it was not set by -D PROBLEM option on command line to cmake"
        << std::endl << "Rerun cmake with -D PROBLEM=file to specify custom problem "
        << "generator file" << std::endl;;
    std::exit(EXIT_FAILURE);
  }

  // now call appropriate pgen function
  (this->*pgen_func_)(pm->pmb_pack, pin);
#endif
}

//----------------------------------------------------------------------------------------
// dtor

//ProblemGenerator::~ProblemGenerator() {
//}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ProblemGeneratorFinalize()
//  \brief calls any final work to be done after execution of main loop, for example
//  compute errors in linear wave test

void ProblemGenerator::ProblemGeneratorFinalize(ParameterInput *pin, Mesh *pm)
{
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");
  if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWaveErrors_(pm->pmb_pack, pin);
  }
  return;
}
