//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file track_prtcl.cpp
//! \brief writes data for tracked particles in unformatted binary

#include <sys/stat.h>  // mkdir
#include <vector>

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
// ctor: also calls BaseTypeOutput base class constructor

TrackedParticleOutput::TrackedParticleOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("trk",0775);
  // allocate arrays
  npout_eachrank.resize(global_variable::nranks);
  ntrack = pin->GetInteger(op.block_name,"nparticles");
  // TODO(@user) improve guess below?
  ntrack_thisrank = ntrack;
}

//----------------------------------------------------------------------------------------
// TrackedParticleOutput::LoadOutputData()
// Copies data for tracked particles on this rank to host outpart array

void TrackedParticleOutput::LoadOutputData(Mesh *pm) {
  // Load data for tracked particles on this rank into new device array
  DualArray1D<TrackedParticleData> tracked_prtcl("d_trked",ntrack_thisrank);
  int npart = pm->nprtcl_thisrank;
  auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
  auto &pi = pm->pmb_pack->ppart->prtcl_idata;
  int ntrack_ = ntrack;

  // Create device-side counter
  Kokkos::View<int> d_counter("counter");

  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    if (pi(PTAG,p) < ntrack_) {
      int index = Kokkos::atomic_fetch_add(&d_counter(),1);
      tracked_prtcl.d_view(index).tag = pi(PTAG,p);
      tracked_prtcl.d_view(index).x   = pr(IPX,p);
      tracked_prtcl.d_view(index).y   = pr(IPY,p);
      tracked_prtcl.d_view(index).z   = pr(IPZ,p);
      tracked_prtcl.d_view(index).vx  = pr(IPVX,p);
      tracked_prtcl.d_view(index).vy  = pr(IPVY,p);
      tracked_prtcl.d_view(index).vz  = pr(IPVZ,p);
    }
  });
  
  Kokkos::fence();
  Kokkos::deep_copy(npout, d_counter);
  
  // share number of tracked particles to be output across all ranks
  npout_eachrank[global_variable::my_rank] = npout;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npout, 1, MPI_INT, npout_eachrank.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif

  // sync tracked particle device array with host
  tracked_prtcl.template modify<DevExeSpace>();
  tracked_prtcl.template sync<HostMemSpace>();

  // copy host view into host outpart array
  Kokkos::realloc(outpart, npout);
  if (npout > 0) {
    auto tracked_slice = Kokkos::subview(tracked_prtcl.h_view, std::make_pair(0, npout));
    Kokkos::deep_copy(outpart, tracked_slice);
  }

}

//----------------------------------------------------------------------------------------
//! \fn void TrackedParticleOutput:::WriteOutputFile(Mesh *pm)
//! \brief Cycles over all tracked particles on this rank and writes ouput data
//! With MPI, all particles are written to the same file.

void TrackedParticleOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  int big_end = IsBigEndian(); // =1 on big endian machine

  // create filename: "trk/file_basename".trk
  std::string fname;
  fname.assign("trk/");
  fname.append(out_params.file_basename);
  fname.append(".trk");

  // Root process opens/creates file and appends string
  if (global_variable::my_rank == 0) {
    std::stringstream msg;
    msg << std::endl << "# AthenaK tracked particle data at time= " << pm->time
        << "  nranks= " << global_variable::nranks
        << "  cycle=" << pm->ncycle
        << "  ntracked_prtcls=" << ntrack << std::endl;
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }
    std::fprintf(pfile,"%s \n",msg.str().c_str());
    std::fclose(pfile);
  }

  // Now all ranks open file and append data
  IOWrapper partfile;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::append);
  std::size_t header_offset = partfile.GetPosition();

  // allocate arrays for tags (ints) and position/velocity data (floats)
  int *tags = new int[npout];
  float *data = new float[6*npout];
  // Loop over particles, load tag and positions into arrays
  for (int p=0; p<npout; ++p) {
    tags[p] = outpart(p).tag;
    data[ 6*p   ] = static_cast<float>(outpart(p).x);
    data[(6*p)+1] = static_cast<float>(outpart(p).y);
    data[(6*p)+2] = static_cast<float>(outpart(p).z);
    data[(6*p)+3] = static_cast<float>(outpart(p).vx);
    data[(6*p)+4] = static_cast<float>(outpart(p).vy);
    data[(6*p)+5] = static_cast<float>(outpart(p).vz);
  }

  // calculate local data offset
  std::vector<int> rank_offset(global_variable::nranks, 0);
  int npout_min = npout_eachrank[0];
  for (int n=1; n<global_variable::nranks; ++n) {
    rank_offset[n] = rank_offset[n-1] + npout_eachrank[n-1];
    npout_min = std::min(npout_min, npout_eachrank[n]);
  }

  // Write particle tags and position/velocity data
  // First write tags for particles collectively over minimum shared number of prtcls
  for (int p=0; p<npout_min; ++p) {
    // tag offset: each tag is an int at position based on particle tag
    std::size_t tag_offset = header_offset + outpart(p).tag * sizeof(int);
    if (partfile.Write_any_type_at_all(&(tags[p]),1,tag_offset,"int") != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle tag not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Write tags individually for remaining particles on each rank
  for (int p=npout_min; p<npout; ++p) {
    std::size_t tag_offset = header_offset + outpart(p).tag * sizeof(int);
    if (partfile.Write_any_type_at(&(tags[p]),1,tag_offset,"int") != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle tag not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Calculate offset for position/velocity data (after all tags)
  std::size_t data_section_offset = header_offset + ntrack * sizeof(int);

  // Write position/velocity data collectively over minimum shared number of prtcls
  for (int p=0; p<npout_min; ++p) {
    // data offset: 6 floats per particle, positioned by particle tag
    std::size_t data_offset = data_section_offset + 6 * outpart(p).tag * sizeof(float);
    if (partfile.Write_any_type_at_all(&(data[6*p]),6,data_offset,"float") != 6) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Write position/velocity data individually for remaining particles on each rank
  for (int p=npout_min; p<npout; ++p) {
    std::size_t data_offset = data_section_offset + 6 * outpart(p).tag * sizeof(float);
    if (partfile.Write_any_type_at(&(data[6*p]),6,data_offset,"float") != 6) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // close the output file and clean up
  partfile.Close();
  delete[] tags;
  delete[] data;

  // increment counters
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
