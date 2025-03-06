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
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
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
  } else if (pgen_fun_name.compare("cpaw") == 0) {
    AlfvenWave(pin, false);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, false);
  } else if (pgen_fun_name.compare("tetrad") == 0) {
    CheckOrthonormalTetrad(pin, false);
  } else if (pgen_fun_name.compare("hohlraum") == 0) {
    Hohlraum(pin, false);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, false);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, false);
  } else if (pgen_fun_name.compare("gr_monopole") == 0) {
    Monopole(pin, false);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, false);
  } else if (pgen_fun_name.compare("rad_linear_wave") == 0) {
    RadiationLinearWave(pin, false);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, false);
  } else if (pgen_fun_name.compare("z4c_linear_wave") == 0) {
    Z4cLinearWave(pin, false);
  } else if (pgen_fun_name.compare("spherical_collapse") == 0) {
    SphericalCollapse(pin, false);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, false);
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

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile,
                                   bool single_file_per_rank) :
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

  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;
  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  adm::ADM* padm = pm->pmb_pack->padm;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  radiation::Radiation* prad=pm->pmb_pack->prad;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  int nrad = 0, nhydro = 0, nmhd = 0, nforce = 3, nadm = 0, nz4c = 0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
  } else if (padm != nullptr) {
    nadm = padm->nadm;
  }

  // root process reads z4c last_output_time and tracker data
  if (pz4c != nullptr) {
    Real last_output_time;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_Reals(&last_output_time, 1,single_file_per_rank) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "z4c::last_output_time data size read from restart "
                  << "file is incorrect, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(&last_output_time, sizeof(Real), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    pz4c->last_output_time = last_output_time;

    for (auto &pt : pz4c->ptracker) {
      Real pos[3];
      if (global_variable::my_rank == 0 || single_file_per_rank) {
        if (resfile.Read_Reals(&pos[0], 3, single_file_per_rank) != 3) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "compact object tracker data size read from restart "
                    << "file is incorrect, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
#if MPI_PARALLEL_ENABLED
      if (!single_file_per_rank) {
        MPI_Bcast(&pos[0], 3*sizeof(Real), MPI_CHAR, 0, MPI_COMM_WORLD);
      }
#endif
      pt->SetPos(&pos[0]);
    }
  }

  if (pturb != nullptr) {
    // root process reads size the random seed
    char *rng_data = new char[sizeof(RNG_State)];
    // the master process reads the variables data
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(rng_data, 1, sizeof(RNG_State), single_file_per_rank)
          != sizeof(RNG_State)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "RNG data size read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      // then broadcast the RNG information
      MPI_Bcast(rng_data, sizeof(RNG_State), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    std::memcpy(&(pturb->rstate), &(rng_data[0]), sizeof(RNG_State));
  }

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    if (resfile.Read_bytes(variabledata, 1, variablesize, single_file_per_rank)
        != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  if (!single_file_per_rank) {
    MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif
  IOWrapperSizeT data_size;
  std::memcpy(&data_size, &(variabledata[0]), sizeof(IOWrapperSizeT));

  // calculate total number of CC variables
  IOWrapperSizeT headeroffset;
  // master process gets file offset
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    headeroffset = resfile.GetPosition(single_file_per_rank);
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  if (!single_file_per_rank) {
    MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  IOWrapperSizeT data_size_ = 0;
  if (phydro != nullptr) {
    data_size_ += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
  }
  if (pmhd != nullptr) {
    data_size_ += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    data_size_ += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    data_size_ += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    data_size_ += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
  }
  if (prad != nullptr) {
    data_size_ += nout1*nout2*nout3*nrad*sizeof(Real);   // rad i0
  }
  if (pturb != nullptr) {
    data_size_ += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
  }
  if (pz4c != nullptr) {
    data_size_ += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
  } else if (padm != nullptr) {
    data_size_ += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
  }

  if (data_size_ != data_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro, MHD, Rad, and/or Z4c arrays, restart file is broken."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // read CC data into host array
  int mygids = pm->gids_eachrank[global_variable::my_rank];
  IOWrapperSizeT offset_myrank = headeroffset;
  if (!single_file_per_rank) {
    offset_myrank += data_size_ * pm->gids_eachrank[global_variable::my_rank];
  }
  IOWrapperSizeT myoffset = offset_myrank;

  HostArray5D<Real> ccin("rst-cc-in", 1, 1, 1, 1, 1);
  HostFaceFld4D<Real> fcin("rst-fc-in", 1, 1, 1, 1);

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmb_eachrank[0];
  int noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
  }

  if (phydro != nullptr) {
    Kokkos::realloc(ccin, nmb, nhydro, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(phydro->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
    myoffset = offset_myrank;
  }

  if (pmhd != nullptr) {
    Kokkos::realloc(ccin, nmb, nmhd, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    myoffset = offset_myrank;

    Kokkos::realloc(fcin.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(fcin.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(fcin.x3f, nmb, nout3+1, nout2, nout1);
    // read FC data into host array, again one MeshBlock at a time
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        int fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at_all(x1fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at_all(x2fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at_all(x3fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      } else if (m < pm->nmb_thisrank) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        int fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x1f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x2f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x3f);
    offset_myrank += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    offset_myrank += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    offset_myrank += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
    myoffset = offset_myrank;
  }

  if (prad != nullptr) {
    Kokkos::realloc(ccin, nmb, nrad, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(prad->i0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nrad*sizeof(Real);   // radiation i0
    myoffset = offset_myrank;
  }

  if (pturb != nullptr) {
    Kokkos::realloc(ccin, nmb, nforce, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pturb->force, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
    myoffset = offset_myrank;
  }

  if (pz4c != nullptr) {
    Kokkos::realloc(ccin, nmb, nz4c, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pz4c->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
    myoffset = offset_myrank;

    // We also need to reinitialize the ADM data.
    pz4c->Z4cToADM(pmy_mesh_->pmb_pack);
  } else if (padm != nullptr) {
    Kokkos::realloc(ccin, nmb, nadm, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        int mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(padm->u_adm, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
    myoffset = offset_myrank;
  }

  // call problem generator again to re-initialize data, fn ptrs, as needed
#if USER_PROBLEM_ENABLED
  UserProblem(pin, true);
#else
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, true);
  } else if (pgen_fun_name.compare("cpaw") == 0) {
    AlfvenWave(pin, true);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, true);
  } else if (pgen_fun_name.compare("tetrad") == 0) {
    CheckOrthonormalTetrad(pin, true);
  } else if (pgen_fun_name.compare("hohlraum") == 0) {
    Hohlraum(pin, true);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, true);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, true);
  } else if (pgen_fun_name.compare("gr_monopole") == 0) {
    Monopole(pin, true);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, true);
  } else if (pgen_fun_name.compare("rad_linear_wave") == 0) {
    RadiationLinearWave(pin, true);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, true);
  } else if (pgen_fun_name.compare("z4c_linear_wave") == 0) {
    Z4cLinearWave(pin, true);
  } else if (pgen_fun_name.compare("spherical_collapse") == 0) {
    SphericalCollapse(pin, true);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, true);
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
