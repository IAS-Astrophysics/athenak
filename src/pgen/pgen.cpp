//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//! \brief Implementation of constructors and functions in class ProblemGenerator.
//! Default constructor calls problem generator function, while  constructor for restarts
//! reads data from restart file, as well as re-initializing problem-specific data.

#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
// default constructor, calls pgen function.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm) :
    user_bcs(false),
    user_srcs(false),
    user_hist(false),
    pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }

  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);

#if USER_PROBLEM_ENABLED
  // call user-defined problem generator
  UserProblem(pin, false);
#else
  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, false);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, false);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, false);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, false);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, false);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, false);
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
                << "by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, but "
                << "not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// constructor for restarts
// When called, data needed to rebuild mesh has been read from restart file by
// Mesh::BuildTreeFromRestart() function. This constructor reads from the restart file and
// initializes all the dependent variables (u0,b0,etc) stored in each Physics class. It
// also calls ProblemGenerator::SetProblemData() function to set any user-defined BCs,
// and any data necessary for restart runs to continue correctly.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile) :
    user_bcs(false),
    user_srcs(false),
    user_hist(false),
    pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }
  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);
  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = 2*sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0) { // the master process reads the variables data
    if (resfile.Read_bytes(variabledata, 1, variablesize) != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // Read number of CC variables and FC fields per MeshBlock in restart file
  IOWrapperSizeT ccdata_cnt, fcdata_cnt, hdos = 0;
  std::memcpy(&ccdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT);
  std::memcpy(&fcdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));

  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  int nhydro_tot = 0, nmhd_tot = 0;
  if (phydro != nullptr) {
    nhydro_tot = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd_tot = pmhd->nmhd + pmhd->nscalars;
  }

  IOWrapperSizeT headeroffset;
  // master process gets file offset
  if (global_variable::my_rank == 0) {
    headeroffset = resfile.GetPosition();
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // allocate arrays for CC data
  HostArray5D<Real> ccin("pgen-ccin", nmb, (nhydro_tot + nmhd_tot), nout3, nout2, nout1);
  if (ccin.size() != (nmb*ccdata_cnt)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro and MHD arrays, restart file is broken." << std::endl;
    exit(EXIT_FAILURE);
  }

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmblist[0];
  int noutmbs_min = pm->nmblist[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmblist[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmblist[i]);
  }

  // read CC data into host array, one MeshBlock at a time to avoid exceeding 2^31 limit
  // on each read call for very large grids per MPI rank
  int mygids = pm->gidslist[global_variable::my_rank];
  IOWrapperSizeT myoffset = headeroffset + (ccdata_cnt+fcdata_cnt)*mygids*sizeof(Real);
  for (int m=0;  m<noutmbs_max; ++m) {
    // every rank has a MB to read, so read collectively
    if (m < noutmbs_min) {
      // get ptr to cell-centered MeshBlock data
      auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset) != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC data not read correctly from restart file, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += mbcnt*sizeof(Real);

    // some ranks are finished writing, so use non-collective write
    } else if (m < pm->nmb_thisrank) {
      // get ptr to MeshBlock data
      auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset) != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC data not read correctly from restart file, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += mbcnt*sizeof(Real);
    }
  }

  // copy CC Hydro data to device
  if (phydro != nullptr) {
    DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(phydro->u0);
    auto hst_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(host_u0, hst_slice);
    Kokkos::deep_copy(phydro->u0, host_u0);
  }

  // copy CC MHD data to device
  if (pmhd != nullptr) {
    DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(pmhd->u0);
    auto hst_slice = Kokkos::subview(ccin,Kokkos::ALL,std::make_pair(nhydro_tot,nmhd_tot),
                                     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(host_u0, hst_slice);
    Kokkos::deep_copy(pmhd->u0, host_u0);
  }

  // allocate arrays for FC data, read face-centered fields
  if (pmhd != nullptr) {
    HostFaceFld4D<Real> fcin("pgen-fcin", nmb, nout3, nout2, nout1);
    if ((fcin.x1f.size() +fcin.x2f.size() +fcin.x3f.size()) != (nmb*fcdata_cnt)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "FC data size read from restart file not equal to size "
                << "of MHD field arrays, restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }

    // read FC data into host array, again one MeshBlock at a time
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        int fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at_all(x1fptr.data(), fldcnt, myoffset) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x1f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at_all(x2fptr.data(), fldcnt, myoffset) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x2f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at_all(x3fptr.data(), fldcnt, myoffset) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x3f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        int fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, myoffset) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x1f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, myoffset) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x2f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, myoffset) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input x3f field not read correctly from restart file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);
      }
    }
    Kokkos::deep_copy(pmhd->b0.x1f, fcin.x1f);
    Kokkos::deep_copy(pmhd->b0.x2f, fcin.x2f);
    Kokkos::deep_copy(pmhd->b0.x3f, fcin.x3f);
  }

  // call problem generator again to re-initialize data, fn ptrs, as needed
#if USER_PROBLEM_ENABLED
  UserProblem(pin, true);
#else
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, true);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, true);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, true);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, true);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, true);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, true);
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
                << "during restart by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, "
                << "but not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
