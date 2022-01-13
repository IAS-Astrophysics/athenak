//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
/////////////////////////////////// AthenaXXX Main Program ///////////////////////////////
//! \file main.cpp
//  \brief AthenaXXX main program
//
// Based on the Athena (Cambridge version) and Athena++ MHD codes. Athena was originally
// written in 2002-2005 by Jim Stone, Tom Gardiner, and Peter Teuben, with many important
// contributions by many other developers after that, i.e. 2005-2014.
//
// Athena++ was started in Jan 2014.  The core design was finished during 4-7/2014 at the
// KITP by Jim Stone.  GR was implemented by Chris White and AMR by Kengo Tomida during
// 2014-2016.  Contributions from many others have continued to the present.
//
// AthenaXXX is an outgrowth of the Athena-Parthenon collaboration, and is a completely
// new implementation based on the Kokkos performance-portability library (an external
// module for this version of the code, required to run on GPUs)
//========================================================================================

// C/C++ headers
#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "utils/utils.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "driver/driver.hpp"

// MPI/OpenMP headers
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#if OPENMP_PARALLEL_ENABLED
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief AthenaK main program

int main(int argc, char *argv[]) {
  std::string input_file, restart_file, run_dir;
  int  res_flag = 0;  // set to 1 if -r        argument is on cmdline
  int narg_flag = 0;  // set to 1 if -n        argument is on cmdline
  int marg_flag = 0;  // set to 1 if -m        argument is on cmdline
  int iarg_flag = 0;  // set to 1 if -i <file> argument is on cmdline

  //--- Step 1. --------------------------------------------------------------------------
  // Initialize environment (must initialize MPI first, then Kokkos)

#if MPI_PARALLEL_ENABLED
#if OPENMP_PARALLEL_ENABLED
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
#endif  // OPENMP_PARALLEL_ENABLED
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(global_variable::my_rank))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return(0);
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no MPI
  global_variable::my_rank = 0;
  global_variable::nranks  = 1;
#endif  // MPI_PARALLEL_ENABLED

  Kokkos::initialize(argc, argv);

  //--- Step 2. --------------------------------------------------------------------------
  // Check for command line options and respond.

  for (int i=1; i<argc; i++) {
    // If argv[i] is a 2 character string of the form "-?" then:
    if (*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {
      // check that command line options that require arguments actually have them:
      char opt_letter = *(argv[i]+1);
      switch(opt_letter) {
        case 'c':
        case 'h':
        case 'm':
        case 'n':
          break;
        default:
          if ((i+1 >= argc) // flag is at the end of the command line options
              || (*argv[i+1] == '-') ) { // flag is followed by another flag
            if (global_variable::my_rank == 0) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                        << std::endl << "-" << opt_letter
                        << " must be followed by a valid argument" << std::endl;
              Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
              MPI_Finalize();
#endif
              return(0);
            }
          }
      }

      // set arguments, flags, or execute tasks specified by options
      switch(*(argv[i]+1)) {
        case 'i':                      // -i <input_file>
          input_file.assign(argv[++i]);
          iarg_flag = 1;
          break;
        case 'r':                      // -r <restart_file>
          res_flag = 1;
          restart_file.assign(argv[++i]);
          break;
        case 'd':                      // -d <run_directory>
          run_dir.assign(argv[++i]);
          break;
        case 'n':
          narg_flag = 1;
          break;
        case 'm':
          marg_flag = 1;
          break;
        case 'c':
          if (global_variable::my_rank == 0) ShowConfig();
          Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
        case 'h':
        default:
          if (global_variable::my_rank == 0) {
            std::cout << "Athena++ v" << ATHENA_VERSION_MAJOR << "."
                                      << ATHENA_VERSION_MINOR << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m              output mesh structure and quit\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
          Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
      }
    } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
  }

  // print error if input or restart file not given
  if (restart_file.empty() && input_file.empty()) {
    // no input file is given
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Either an input or restart file must be specified." << std::endl
              << "See " << argv[0] << " -h for options and usage." << std::endl;
    Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 3. --------------------------------------------------------------------------
  // Construct ParameterInput object and load data either from restart or input file.
  // With MPI, the input is read by every rank in parallel using MPI-IO.

  ParameterInput* pinput = new ParameterInput;
  IOWrapper infile, restartfile;
  // read parameters from restart file
  if (res_flag == 1) {
    restartfile.Open(restart_file.c_str(), IOWrapper::FileMode::read);
    pinput->LoadFromFile(restartfile);
  }
  // read parameters from input file.  If both -r and -i are specified, this will
  // override parameters from the restart file
  if (iarg_flag == 1) {
    infile.Open(input_file.c_str(), IOWrapper::FileMode::read);
    pinput->LoadFromFile(infile);
    infile.Close();
  }
  pinput->ModifyFromCmdline(argc, argv);

  // Dump input parameters and quit if code was run with -n option.
  if (narg_flag) {
    if (global_variable::my_rank == 0) pinput->ParameterDump(std::cout);
    if (res_flag == 1) restartfile.Close();
    delete pinput;
    Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 4. --------------------------------------------------------------------------
  // Construct Mesh.  Then build MeshBlockTree and add MeshBlockPack containing MeshBlocks
  // on this rank.  Latter cannot be performed in Mesh constructor since it requires
  // pointer to Mesh.

  Mesh* pmesh = new Mesh(pinput);
  if (res_flag == 0) {
    pmesh->BuildTreeFromScratch(pinput);
  } else {
    pmesh->BuildTreeFromRestart(pinput, restartfile);
  }

  //  If code was run with -m option, write mesh structure to file and quit.
  if (marg_flag) {
    if (global_variable::my_rank == 0) {pmesh->WriteMeshStructure();}
    if (res_flag == 1) {restartfile.Close();}
    delete pmesh;
    delete pinput;
    Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 5. --------------------------------------------------------------------------
  // Add physics modules to MeshBlockPack, and set initial conditions either by calling
  // problem generator or by reading restart file. Note these steps must occur after Mesh
  // (including MeshBlocks and MeshBlockPack) is fully constructed.

  pmesh->pmb_pack->AddPhysics(pinput);
  if (res_flag == 0) {
    // set ICs using ProblemGenerator constructor for new runs
    pmesh->pgen = std::make_unique<ProblemGenerator>(pinput, pmesh);
  } else {
    // read ICs from restart file using ProblemGenerator constructor for restarts
    pmesh->pgen = std::make_unique<ProblemGenerator>(pinput, pmesh, restartfile);
    restartfile.Close();
  }

  //--- Step 6. --------------------------------------------------------------------------
  // Construct Driver and Outputs. Actual outputs (including initial conditions) are made
  // in Driver.Initialize()

  Driver* pdriver = new Driver(pinput, pmesh);
  ChangeRunDir(run_dir);
  Outputs* pout = new Outputs(pinput, pmesh);

  //--- Step 7. --------------------------------------------------------------------------
  // Execute Driver.
  //    1. Initial conditions set in Driver::Initialize()
  //    2. TaskList(s) executed in Driver::Execute()
  //    3. Any final analysis or diagnostics run in Driver::Finalize()

  pdriver->Initialize(pmesh, pinput, pout);
  pdriver->Execute(pmesh, pinput, pout);
  pdriver->Finalize(pmesh, pinput, pout);

  //--- Step 8. -------------------------------------------------------------------------
  // clean up, and terminate
  // Note anything containing a Kokkos::view must be deleted before Kokkos::finalize()

  delete pout;
  delete pdriver;
  delete pmesh;
  delete pinput;
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return(0);
}
