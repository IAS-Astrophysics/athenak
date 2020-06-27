//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file show_config.cpp

#include <iostream>
#include <sstream>

#include "athena.hpp"

//--------------------------------------------------------------------------------------------------
//! \fn void ShowConfig()
//  \brief prints diagnostic messages about the configuration of an Athena++ executable

void ShowConfig() {
  // To match configure.py output: use 2 space indent for option, value output starts on column 30
  std::cout<<"This Athena++ executable is configured with:" << std::endl;
  std::cout<<"  Problem generator:          " << PROBLEM_GENERATOR << std::endl;
  if (SINGLE_PRECISION_ENABLED) {
    std::cout<<"  Floating-point precision:   single" << std::endl;
  } else {
    std::cout<<"  Floating-point precision:   double" << std::endl;
  }
  std::cout<<"  Number of ghost cells:      " << NGHOST << std::endl;
#ifdef MPI_PARALLEL_ON
  std::cout<<"  MPI parallelism:            ON" << std::endl;
#else
  std::cout<<"  MPI parallelism:            OFF" << std::endl;
#endif
#ifdef OPENMP_PARALLEL_ON
  std::cout<<"  OpenMP parallelism:         ON" << std::endl;
#else
  std::cout<<"  OpenMP parallelism:         OFF" << std::endl;
#endif

//  std::cout<<"  Compiler:                   " << COMPILED_WITH << std::endl;
//  std::cout<<"  Compilation command:        " << COMPILER_COMMAND
//           << COMPILED_WITH_OPTIONS << std::endl;
  return;
}
