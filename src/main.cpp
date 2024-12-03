//========================================================================================
// Athena astrophysical MHD code (Kokkos version)
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//////////////////////////////////// Athena Main Program /////////////////////////////////
//! \file main.cpp
//! \brief Athena main program
//!
//! Based on the Athena (Cambridge version) and Athena++ MHD codes. Athena originally was
//! written in 2002-2005 by Jim Stone, Tom Gardiner, and Peter Teuben, with many important
//! contributions by many other developers after that, i.e. 2005-2014.
//!
//! Athena++ was started in Jan 2014, with the core design developed 4-7/2014 during an
//! extended visit to the KITP at UCSB by J. Stone. GR was implemented by Chris White and
//! AMR by Kengo Tomida 2014-2016, with contributions from many others (esp. K. Felker)
//! continuing after that.
//!
//! Athena (Kokkos version) is an outgrowth of the Athena-Parthenon collaboration, and is
//! a completely new implementation based on the Kokkos performance-portability library
//! (which is an external dependency required for this version). It was started 6/2020
//! during the pandemic. As part of the keep-it-simple design, only a fraction of the
//! features of the C++ version are implemented.
//========================================================================================

// C/C++ headers
#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <cstdio> // sscanf
#include <fstream>  // Include this for std::ifstream

// Athena headers
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

#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//! \brief Athena main program

int main(int argc, char *argv[]) {
  std::string input_file, restart_file, run_dir;
  bool iarg_flag = false;  // set to true if -i <file> argument is on cmdline
  bool marg_flag = false;  // set to true if -m        argument is on cmdline
  bool narg_flag = false;  // set to true if -n        argument is on cmdline
  bool  res_flag = false;  // set to true if -r <file> argument is on cmdline
  Real wtlim = 0;

  //--- Step 1. --------------------------------------------------------------------------
  // Initialize environment (must initialize MPI first, then Kokkos)

#if MPI_PARALLEL_ENABLED
#if defined(KOKKOS_ENABLE_HIP)
  // JMF: This is a bizarre workaround to avoid segmentation faults on Frontier.
  // See OLCFDEV-1655: Occasional seg-fault during MPI_Init inside the Frontier
  // documentation.
  (void) hipInit(0);
#endif
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
          if ((i+1 >= argc) // no argument after option
              || (*argv[i+1] == '-') ) { // option is followed by another option
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
          iarg_flag = true;
          break;
        case 'r':                      // -r <restart_file>
          restart_file.assign(argv[++i]);
          res_flag = true;
          break;
        case 'd':                      // -d <run_directory>
          run_dir.assign(argv[++i]);
          break;
        case 'n':
          narg_flag = true;
          break;
        case 'm':
          marg_flag = true;
          break;
        case 't':                      // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          wtlim = static_cast<Real>(wth*3600 + wtm*60 + wts);
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
            std::cout << "Athena v" << ATHENA_VERSION_MAJOR << "."
                                    << ATHENA_VERSION_MINOR << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m              output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
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

  // Start the wall clock timer. This is done here rather than in the Driver to ensure
  // that the time taken in ProblemGenerator is also captured.
  Kokkos::Timer timer;

  //--- Step 3. --------------------------------------------------------------------------
  // Construct ParameterInput object and load data either from restart or input file.
  // With MPI, the input is read by every rank in parallel using MPI-IO.

  ParameterInput* pinput = new ParameterInput;
  IOWrapper infile, restartfile;
  // read parameters from restart file
  bool single_file_per_rank = false; // DBF: flag for single_file_per_rank for rst files
  if (res_flag) {
    // Check if the path contains "rank_" directory
    size_t rank_pos = restart_file.find("/rank_");
    single_file_per_rank = (rank_pos != std::string::npos);

    // If single_file_per_rank is true, modify the path for the current rank
    if (single_file_per_rank) {
        // Extract the base directory and file name
        size_t last_slash = restart_file.rfind('/');
        std::string base_dir = restart_file.substr(0, rank_pos);
        std::string file_name = restart_file.substr(last_slash + 1);

        // Construct the path for the current rank
        char rank_dir[20];
        std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d", global_variable::my_rank);
        restart_file = base_dir + "/" + rank_dir + "/" + file_name;
    }

    // Now use restart_file for opening the file
    std::ifstream file_check(restart_file);
    if (!file_check.good()) {
        std::cerr << "Error: Unable to open restart file: " << restart_file << std::endl;
        // Handle the error (e.g., exit the program or use a default configuration)
    }

    // read parameters from restart file
    restartfile.Open(restart_file.c_str(),IOWrapper::FileMode::read,single_file_per_rank);
    pinput->LoadFromFile(restartfile, single_file_per_rank);
    IOWrapperSizeT headeroffset = restartfile.GetPosition(single_file_per_rank);
  }

  // read parameters from input file.  If both -r and -i are specified, this will
  // override parameters from the restart file
  if (iarg_flag) {
    infile.Open(input_file.c_str(), IOWrapper::FileMode::read);
    pinput->LoadFromFile(infile);
    infile.Close();
  }
  pinput->ModifyFromCmdline(argc, argv);

  // Dump input parameters and quit if code was run with -n option.
  if (narg_flag) {
    if (global_variable::my_rank == 0) pinput->ParameterDump(std::cout);
    if (res_flag) restartfile.Close(single_file_per_rank);
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
  if (!res_flag) {
    pmesh->BuildTreeFromScratch(pinput);
  } else {
    pmesh->BuildTreeFromRestart(pinput, restartfile, single_file_per_rank);
  }

  //  If code was run with -m option, write mesh structure to file and quit.
  if (marg_flag) {
    if (global_variable::my_rank == 0) {pmesh->WriteMeshStructure();}
    if (res_flag) {restartfile.Close(single_file_per_rank);}
    delete pmesh;
    delete pinput;
    Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Step 5. --------------------------------------------------------------------------
  // Add coordinates and physics modules to MeshBlockPack, and set initial conditions.
  // Note these steps must occur after Mesh (including MeshBlocks and MeshBlockPack)
  // is fully constructed.

  pmesh->AddCoordinatesAndPhysics(pinput);
  if (!res_flag) {
    // set ICs using ProblemGenerator constructor for new runs
    pmesh->pgen = std::make_unique<ProblemGenerator>(pinput, pmesh);
  } else {
    // read ICs from restart file using ProblemGenerator constructor for restarts
    pmesh->pgen = std::make_unique<ProblemGenerator>(pinput,
                                                     pmesh,
                                                     restartfile,
                                                     single_file_per_rank);
    restartfile.Close(single_file_per_rank);
  }
  //--- Step 6. --------------------------------------------------------------------------
  // Construct Driver and Outputs. Actual outputs (including initial conditions) are made
  // in Driver.Initialize(). Add wall clock timer to Driver if necessary.

  ChangeRunDir(run_dir);
  Driver* pdriver = new Driver(pinput, pmesh, wtlim, &timer);
  Outputs* pout = new Outputs(pinput, pmesh);



  //--- Step 7. --------------------------------------------------------------------------
  // Execute Driver.
  //    1. Initial conditions set in Driver::Initialize()
  //    2. TaskList(s) executed in Driver::Execute()
  //    3. Any final analysis or diagnostics run in Driver::Finalize()

  pdriver->Initialize(pmesh, pinput, pout, res_flag);
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
