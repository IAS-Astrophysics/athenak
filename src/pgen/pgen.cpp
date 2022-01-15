//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ogen.cpp
//  \brief implementation of functions in class ProblemGenerator

#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm) :
  user_bcs(false),
  pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }

#if USER_PROBLEM_ENABLED
  // call user-defined problem generator
  UserProblem(pm->pmb_pack, pin);
#else
  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pm->pmb_pack, pin);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pm->pmb_pack, pin);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pm->pmb_pack, pin);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pm->pmb_pack, pin);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pm->pmb_pack, pin);
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
#endif

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "by problem generator." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// constructor for restarts

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile) :
    user_bcs(false),
    pmy_mesh_(pm) {
  // Read size of CC and FC data arrays from restart file
  IOWrapperSizeT ccdata_size, fcdata_size;
  if (global_variable::my_rank == 0) { // the master process reads the header data
    if (resfile.Read(&ccdata_size, 1, sizeof(IOWrapperSizeT)) != sizeof(IOWrapperSizeT)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "CC data size read from restart file is corrupted, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (resfile.Read(&fcdata_size, 1, sizeof(IOWrapperSizeT)) != sizeof(IOWrapperSizeT)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "FC data size read from restart file is corrupted, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // get file offset for reading data arrays
  IOWrapperSizeT headeroffset = resfile.GetPosition();

std::cout << "ccdata_size = "<<ccdata_size<<"  fcdata_size = "<<fcdata_size<<std::endl;

  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  int nhydro=0, nmhd=0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }

  // read CC data into host array
  HostArray5D<Real> ccin("rst-cc-in", nmb, (nhydro+nmhd), nout3, nout2, nout1);
  if (ccin.size()*sizeof(Real) != ccdata_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro and MHD arrays, restart file is broken." << std::endl;
    exit(EXIT_FAILURE);
  }
  int mygids = pm->gidslist[global_variable::my_rank];
  IOWrapperSizeT myoffset = headeroffset + (ccdata_size+fcdata_size)*mygids;
  if (resfile.Read_at_all(ccin.data(), ccdata_size, 1, myoffset) != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Input hydro data not read correctly from restart file, "
              << "restart file is broken." << std::endl;
    exit(EXIT_FAILURE);
  }
  myoffset += ccdata_size;

  // copy CC Hydro data to device
  if (phydro != nullptr) {
    auto array_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro),
                                       Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(phydro->u0, array_slice);
  }

  // copy CC MHD data to device
  if (pmhd != nullptr) {
    auto array_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(nhydro,nmhd),
                                       Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(pmhd->u0, array_slice);
  }

  // read MHD face-centered fields and copy to device
  if (pmhd != nullptr) {
    HostFaceFld4D<Real> fcin("rst-fc-in", nmb, nout3, nout2, nout1);
    if ((fcin.x1f.size() +fcin.x2f.size() +fcin.x3f.size())*sizeof(Real) != fcdata_size) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "FC data size read from restart file not equal to size "
                << "of MHD field arrays, restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }

    IOWrapperSizeT fcin_size = fcin.x1f.size()*sizeof(Real);
    if (resfile.Read_at_all(fcin.x1f.data(), fcin_size, 1, myoffset) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x1f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(pmhd->b0.x1f, fcin.x1f);

    myoffset += fcin_size;
    fcin_size = fcin.x2f.size()*sizeof(Real);
    if (resfile.Read_at_all(fcin.x2f.data(), fcin_size, 1, myoffset) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x2f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(pmhd->b0.x2f, fcin.x2f);

    myoffset += fcin_size;
    fcin_size = fcin.x3f.size()*sizeof(Real);
    if (resfile.Read_at_all(fcin.x3f.data(), fcin_size, 1, myoffset) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x3f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(pmhd->b0.x3f, fcin.x3f);
  }
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ProblemGeneratorFinalize()
//  \brief calls any final work to be done after execution of main loop, for example
//  compute errors in linear wave test

void ProblemGenerator::ProblemGeneratorFinalize(ParameterInput *pin, Mesh *pm) {
  if (pgen_error_func != nullptr) {
    (pgen_error_func)(pm->pmb_pack, pin);
  }
  return;
}
