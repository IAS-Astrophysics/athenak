//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rst_prtcl.cpp
//! \brief writes restart files for particles

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// constructor: also calls BaseTypeOutput base class constructor

ParticleRestartOutput::ParticleRestartOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create directories for particle restart outputs
  mkdir("rst_prtcl", 0775);

  // Check if we're starting from a particle restart and adjust file number
  int prtcl_rst_flag = pin->GetOrAddInteger("problem","prtcl_rst_flag",0);
  if (prtcl_rst_flag) {
    std::string prst_fname = pin->GetString("problem","prtcl_res_file");
    size_t last_period = prst_fname.rfind('.');
    std::string outnumber_str = prst_fname.substr(last_period-5,5);
    int outnumber = std::stoi(outnumber_str);
    out_params.file_number = outnumber + 1;
    out_params.last_time = pm->time;
  }

}

//----------------------------------------------------------------------------------------
// ParticleRestartOutput::LoadOutputData()
// Loads particle data from device to host for output

void ParticleRestartOutput::LoadOutputData(Mesh *pm) {
  nprtcl_thisrank = 0; 
  nrdata = 0;
  nidata = 0;

  // Check if particles exist on this rank
  if (pm->pmb_pack->ppart == nullptr) {
    return;
  }

  auto ppart = pm->pmb_pack->ppart;
  nprtcl_thisrank = ppart->nprtcl_thispack;

  // Get dimensions of particle data
  nrdata = ppart->nrdata;
  nidata = ppart->nidata;

  if (nprtcl_thisrank == 0) return;

  // Allocate host arrays
  Kokkos::realloc(outpart_rdata, nrdata, nprtcl_thisrank);
  Kokkos::realloc(outpart_idata, nidata, nprtcl_thisrank);

  // Copy particle data from device to host
  Kokkos::deep_copy(outpart_rdata, ppart->prtcl_rdata);
  Kokkos::deep_copy(outpart_idata, ppart->prtcl_idata);
}

//----------------------------------------------------------------------------------------
// ParticleRestartOutput::WriteOutputFile()
// Writes particle restart file for this rank

void ParticleRestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  std::string fname;
  char number[7];
  std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
  fname = std::string("rst_prtcl/") + out_params.file_basename + number + ".rst_prtcl";         
  // increment counters for next output
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  // Gather particle counts from all ranks
  std::vector<int> nprtcl_eachrank(global_variable::nranks);
  nprtcl_eachrank[global_variable::my_rank] = nprtcl_thisrank;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&nprtcl_thisrank, 1, MPI_INT, nprtcl_eachrank.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif

  int total_particles = 0;
  for (int i = 0; i < global_variable::nranks; ++i) {
    total_particles += nprtcl_eachrank[i];
  }

  // Get consistent nrdata/nidata across all ranks
  int max_nrdata = nrdata;
  int max_nidata = nidata;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&nrdata, &max_nrdata, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nidata, &max_nidata, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  nrdata = max_nrdata;
  nidata = max_nidata;

  // Rank 0 creates file
  if (global_variable::my_rank == 0) {
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Restart file '" << fname << "' could not be opened" << std::endl;
      exit(EXIT_FAILURE);
    }
    
    std::fwrite(&(pm->time), sizeof(Real), 1, pfile);
    std::fwrite(&(pm->ncycle), sizeof(int), 1, pfile);
    std::fwrite(&total_particles, sizeof(int), 1, pfile);
    
    if (total_particles > 0) {
      std::fwrite(&nrdata, sizeof(int), 1, pfile);
      std::fwrite(&nidata, sizeof(int), 1, pfile);
    }
    
    std::fclose(pfile);
  }

//#if MPI_PARALLEL_ENABLED
//  MPI_Barrier(MPI_COMM_WORLD);
//#endif

  // All ranks open the file with IOWrapper in append mode
  if (total_particles > 0) {
    IOWrapper prtcl_file;
    prtcl_file.Open(fname.c_str(), IOWrapper::FileMode::append);
    std::size_t header_offset = prtcl_file.GetPosition();

    // Calculate displacements
    int my_rdata_offset = 0;
    int my_idata_offset = 0;
    for (int i = 0; i < global_variable::my_rank; ++i) {
      my_rdata_offset += nprtcl_eachrank[i] * nrdata;
      my_idata_offset += nprtcl_eachrank[i] * nidata;
    }

    std::size_t rdata_start = header_offset;
    std::size_t idata_start = header_offset + total_particles * nrdata * sizeof(Real);

    // Each rank writes its own data at computed offset
    if (nprtcl_thisrank > 0) {
      prtcl_file.Write_any_type_at(outpart_rdata.data(), 
                                   nprtcl_thisrank * nrdata,
                                   rdata_start + my_rdata_offset * sizeof(Real),
				   "byte");
      prtcl_file.Write_any_type_at(outpart_idata.data(),
                                   nprtcl_thisrank * nidata, 
                                   idata_start + my_idata_offset * sizeof(int),
				   "byte");
    }

    prtcl_file.Close();
  }
}
