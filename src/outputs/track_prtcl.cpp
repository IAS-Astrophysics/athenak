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
#include <limits>
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
  DvceArray1D<int> counter("tracked_particle_counter", 1);
  Kokkos::deep_copy(counter, 0);
  auto tracked_prtcl_d = tracked_prtcl.d_view;
  int ntrack_thisrank_ = ntrack_thisrank;
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    if ((pi(PTAG,p) >= 0) && (pi(PTAG,p) < ntrack)) {
      int index = Kokkos::atomic_fetch_add(&counter(0), 1);
      if (index < ntrack_thisrank_) {
        tracked_prtcl_d(index).tag = pi(PTAG,p);
        tracked_prtcl_d(index).x   = pr(IPX,p);
        tracked_prtcl_d(index).y   = pr(IPY,p);
        tracked_prtcl_d(index).z   = pr(IPZ,p);
        tracked_prtcl_d(index).vx  = pr(IPVX,p);
        tracked_prtcl_d(index).vy  = pr(IPVY,p);
        tracked_prtcl_d(index).vz  = pr(IPVZ,p);
      }
    }
  });
  auto counter_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), counter);
  npout = std::min(counter_h(0), ntrack_thisrank);
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
  for (int p=0; p<npout; ++p) {
    outpart(p) = tracked_prtcl.h_view(p);
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
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Now all ranks open file and append data
  IOWrapper partfile;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::append);
  IOWrapperSizeT header_offset = partfile.GetSize();

  // allocate 1D vector of floats used to convert and output particle data
  std::sort(outpart.data(), outpart.data() + npout,
            [](const TrackedParticleData &a, const TrackedParticleData &b) {
              return a.tag < b.tag;
            });
  float *data = new float[6*npout];
  // Loop over particles, load positions into data[]
  for (int p=0; p<npout; ++p) {
    data[ 6*p   ] = static_cast<float>(outpart(p).x);
    data[(6*p)+1] = static_cast<float>(outpart(p).y);
    data[(6*p)+2] = static_cast<float>(outpart(p).z);
    data[(6*p)+3] = static_cast<float>(outpart(p).vx);
    data[(6*p)+4] = static_cast<float>(outpart(p).vy);
    data[(6*p)+5] = static_cast<float>(outpart(p).vz);
  }
  // Initialize a tag-indexed block so readers can locate particle tag p at record p.
  const IOWrapperSizeT datasize = sizeof(float);
  if (global_variable::my_rank == 0) {
    std::vector<float> empty(6*ntrack, std::numeric_limits<float>::quiet_NaN());
    if (partfile.Write_any_type_at(empty.data(), empty.size(), header_offset, "float") !=
        empty.size()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "particle data not initialized correctly in tracked particle "
                << "file" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  #if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  // Write each local tracked particle into its global tag slot.
  for (int p=0; p<npout; ++p) {
    if (outpart(p).tag < 0 || outpart(p).tag >= ntrack) {
      continue;
    }
    IOWrapperSizeT myoffset =
        header_offset + 6*outpart(p).tag*datasize;
    if (partfile.Write_any_type_at(&(data[6*p]),6,myoffset,"float") !=
        static_cast<std::size_t>(6)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // close the output file and clean up
  partfile.Close();
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
