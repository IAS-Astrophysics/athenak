//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//  \brief implementation of functions in class ProblemGenerator

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm)
 : pmy_mesh_(pm)
{
#if USER_PROBLEM_ENABLED
  // call user-defined problem generator
  UserProblem(pm->pmb_pack, pin);
#else

  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    pgen_func_ = &ProblemGenerator::Advection_;
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    pgen_func_ = &ProblemGenerator::LinearWave_;
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    pgen_func_ = &ProblemGenerator::ShockTube_; 
  } else if (pgen_fun_name.compare("implode") == 0) {
    pgen_func_ = &ProblemGenerator::LWImplode_;
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    pgen_func_ = &ProblemGenerator::OrszagTang_;

  // else, name not set on command line or input file, print warning and quit
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl 
        << "and it was not set by -D PROBLEM option on cmake command line during build"
        << std::endl
        << "Rerun cmake with -D PROBLEM=file to specify custom problem generator file"
        << std::endl;;
    std::exit(EXIT_FAILURE);
  }

  // now call appropriate pgen function
  (this->*pgen_func_)(pm->pmb_pack, pin);
#endif
}

//----------------------------------------------------------------------------------------
// constructor for restarts

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile)
 : pmy_mesh_(pm)
{
  IOWrapperSizeT datasize;
  // Read size of data arrays from restart file
  if (global_variable::my_rank == 0) { // the master process reads the header data
    if (resfile.Read(&datasize, 1, sizeof(IOWrapperSizeT)) != sizeof(IOWrapperSizeT)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Data size read from restart file is corrupted, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // get file offset for reading data arrays 
  IOWrapperSizeT headeroffset = resfile.GetPosition();

std::cout << "datasize = " << datasize << std::endl;

  // get size of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // read Hydro arrays
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  if (phydro != nullptr) {
    int nvar = phydro->nhydro + phydro->nscalars;
    HostArray5D<Real> indata("restart-in", nmb, nvar, nout3, nout2, nout1);
    if (indata.size()*sizeof(Real) != datasize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Data size read from restart file not equal to size of "
                << "Hydro array, restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }

    // now read Hydro restart data in parallel
    int mygids = pm->gidslist[global_variable::my_rank];
    IOWrapperSizeT myoffset = headeroffset + datasize*mygids;
    if (resfile.Read_at_all(indata.data(), datasize, 1, myoffset) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input hydro data not read correctly from restart file, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(phydro->u0, indata);
  }

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
