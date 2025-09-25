//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eventlog.cpp
//! \brief writes diagnostic data collected by various event counters implemented
//! throughout the code to a log file.  Checks whether there is data to be written
//! every time step, but only writes data if one or more counters are non-zero

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

EventLogOutput::EventLogOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  header_written = false;
}

//----------------------------------------------------------------------------------------
//! \fn void EventLogOutput::LoadOutputData()
//! \brief sums event counter data across MPI ranks

void EventLogOutput::LoadOutputData(Mesh *pm) {
#if MPI_PARALLEL_ENABLED
  // perform in-place sum or max over all MPI ranks, depending on counter
  int* pdfloor = &(pm->ecounter.neos_dfloor);
  int* pefloor = &(pm->ecounter.neos_efloor);
  int* ptfloor = &(pm->ecounter.neos_tfloor);
  int* pvceil  = &(pm->ecounter.neos_vceil);
  int* pfail   = &(pm->ecounter.neos_fail);
  int* pmaxit  = &(pm->ecounter.maxit_c2p);
  int* pfofc   = &(pm->ecounter.nfofc);
  MPI_Allreduce(MPI_IN_PLACE, pdfloor, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pefloor, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ptfloor, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pvceil,  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pfail,   1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pmaxit,  1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pfofc,   1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  // check if there is any data to be written
  no_output = true;
  if (pm->ecounter.neos_dfloor > 0 ||
      pm->ecounter.neos_efloor > 0 ||
      pm->ecounter.neos_tfloor > 0 ||
      pm->ecounter.neos_vceil  > 0 ||
      pm->ecounter.neos_fail   > 0 ||
      pm->ecounter.nfofc > 0 ||
      pm->ecounter.maxit_c2p > 0) {
    no_output=false;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void EventLogOutput::WriteOutputFile()
//! \brief writes event counter data to log file

void EventLogOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  if (header_written && no_output) return;

  // only the master rank writes the file
  if (global_variable::my_rank == 0) {
    // create filename: "file_basename" + ".log"
    // There is no file number or id in event log output filenames.
    std::string fname;
    fname.assign(out_params.file_basename);
    fname.append(".log");

    // open file for output
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }

    // Write header, if it has not been written already
    if (!(header_written)) {
      std::fprintf(pfile,"# Athena event counter data\n");
      std::fprintf(pfile,"#  cycle eos_dfloor eos_efloor eos_tfloor eos_vceil");
      std::fprintf(pfile," eos_fail c2p_it fofc");
      std::fprintf(pfile,"\n");  // terminate line
      header_written = true;
    }

    // write event counters
    if (!(no_output)) {
      std::fprintf(pfile, "%8d", pm->ncycle);
      std::fprintf(pfile, " %8d", pm->ecounter.neos_dfloor);
      std::fprintf(pfile, " %8d", pm->ecounter.neos_efloor);
      std::fprintf(pfile, " %8d", pm->ecounter.neos_tfloor);
      std::fprintf(pfile, " %8d", pm->ecounter.neos_vceil);
      std::fprintf(pfile, " %8d", pm->ecounter.neos_fail);
      std::fprintf(pfile, " %6d", pm->ecounter.maxit_c2p);
      std::fprintf(pfile, " %8d", pm->ecounter.nfofc);
      std::fprintf(pfile,"\n"); // terminate line
    }
    std::fclose(pfile);
  }

  // reset counters
  pm->ecounter.neos_dfloor = 0;
  pm->ecounter.neos_efloor = 0;
  pm->ecounter.neos_tfloor = 0;
  pm->ecounter.neos_vceil = 0;
  pm->ecounter.neos_fail = 0;
  pm->ecounter.maxit_c2p = 0;
  pm->ecounter.nfofc = 0;

  // increment output time, clean up
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
