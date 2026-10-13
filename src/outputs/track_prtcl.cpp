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
#include <utility>

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
  // create filename: "trk/file_basename".trk
  std::string fname;
  fname.assign("trk/");
  fname.append(out_params.file_basename);
  fname.append(".trk");

  std::sort(outpart.data(), outpart.data() + npout,
            [](const TrackedParticleData &a, const TrackedParticleData &b) {
              return a.tag < b.tag;
            });
  std::vector<int> local_tags(npout);
  std::vector<float> local_data(6*npout);
  for (int p=0; p<npout; ++p) {
    local_tags[p] = outpart(p).tag;
    local_data[ 6*p   ] = static_cast<float>(outpart(p).x);
    local_data[(6*p)+1] = static_cast<float>(outpart(p).y);
    local_data[(6*p)+2] = static_cast<float>(outpart(p).z);
    local_data[(6*p)+3] = static_cast<float>(outpart(p).vx);
    local_data[(6*p)+4] = static_cast<float>(outpart(p).vy);
    local_data[(6*p)+5] = static_cast<float>(outpart(p).vz);
  }

  std::vector<int> all_tags;
  std::vector<float> all_data;
#if MPI_PARALLEL_ENABLED
  std::vector<int> tag_displs(global_variable::nranks);
  std::vector<int> data_counts(global_variable::nranks);
  std::vector<int> data_displs(global_variable::nranks);
  int total_npout = 0;
  for (int n=0; n<global_variable::nranks; ++n) {
    tag_displs[n] = total_npout;
    data_counts[n] = 6*npout_eachrank[n];
    data_displs[n] = 6*total_npout;
    total_npout += npout_eachrank[n];
  }
  if (global_variable::my_rank == 0) {
    all_tags.resize(total_npout);
    all_data.resize(6*total_npout);
  }
  MPI_Gatherv(local_tags.empty() ? nullptr : local_tags.data(), npout, MPI_INT,
              all_tags.empty() ? nullptr : all_tags.data(), npout_eachrank.data(),
              tag_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(local_data.empty() ? nullptr : local_data.data(), 6*npout, MPI_FLOAT,
              all_data.empty() ? nullptr : all_data.data(), data_counts.data(),
              data_displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
#else
  all_tags = std::move(local_tags);
  all_data = std::move(local_data);
#endif

  if (global_variable::my_rank == 0) {
    std::vector<float> tag_ordered_data(6*ntrack,
                                        std::numeric_limits<float>::quiet_NaN());
    for (std::size_t p=0; p<all_tags.size(); ++p) {
      int tag = all_tags[p];
      if (tag < 0 || tag >= ntrack) {
        continue;
      }
      for (int n=0; n<6; ++n) {
        tag_ordered_data[6*tag + n] = all_data[6*p + n];
      }
    }

    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Output file '" << fname << "' could not be opened"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    std::stringstream msg;
    msg << std::endl << "# AthenaK tracked particle data at time= " << pm->time
        << "  nranks= " << global_variable::nranks
        << "  cycle=" << pm->ncycle
        << "  ntracked_prtcls=" << ntrack << std::endl;
    std::fprintf(pfile,"%s \n",msg.str().c_str());
    if (std::fwrite(tag_ordered_data.data(), sizeof(float),
                    tag_ordered_data.size(), pfile) != tag_ordered_data.size()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "particle data not written correctly to tracked particle "
                << "file" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::fclose(pfile);
  }

  // increment counters
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
