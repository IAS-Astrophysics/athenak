//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vtk_prtcl.cpp
//! \brief writes particle data in (legacy) vtk format.
//! Data is written in UNSTRUCTURED_GRID geometry, in BINARY format, and in FLOAT type
//! Data over multiple MeehBlocks and MPI ranks is written to a single file using MPI-IO.

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
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor
// Checks compatibility options for VTK outputs

ParticleVTKOutput::ParticleVTKOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("pvtk",0775);
}

//----------------------------------------------------------------------------------------
// ParticleVTKOutput::LoadOutputData()
// Copies real and integer particle data to host for outputs

void ParticleVTKOutput::LoadOutputData(Mesh *pm) {
  particles::Particles *pp = pm->pmb_pack->ppart;
  npout_thisrank = pm->nprtcl_thisrank;
  npout_total = pm->nprtcl_total;
  Kokkos::realloc(outpart_rdata, pp->nrdata, npout_thisrank);
  Kokkos::realloc(outpart_idata, pp->nidata, npout_thisrank);

  // Create mirror view on device of host view of output particle real/int data
  auto d_outpart_rdata = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(),
                                                    outpart_rdata);
  auto d_outpart_idata = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(),
                                                    outpart_idata);
  // Copy particle positions into device mirrors
  Kokkos::deep_copy(d_outpart_rdata, pp->prtcl_rdata);
  Kokkos::deep_copy(d_outpart_idata, pp->prtcl_idata);
  // Copy particle positions from device mirror to host output array
  Kokkos::deep_copy(outpart_rdata, d_outpart_rdata);
  Kokkos::deep_copy(outpart_idata, d_outpart_idata);
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleVTKOutput:::WriteOutputFile(Mesh *pm)
//! \brief Cycles over all particles and writes ouput data in (legacy) vtk format.
//! With MPI, all particles are written to the same file.
//!
//! There are seven basic parts to the VTK "legacy" file format for particles
//! (unstructured points):
//!  1. File version and identifier
//!  2. Header (time, cycle, variables, etc.)
//!  3. File format
//!  4. Dataset structure.
//!  5. Point (x,y,z) positions (written in BINARY format in this implementation)
//!  6. Arbitrary number of SCALARS data at each point (BINARY format)
//!  7. Arbitrary number of VECTORS data at each point (BINARY format)

void ParticleVTKOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  int big_end = IsBigEndian(); // =1 on big endian machine

  // create filename: "vtk/file_basename"."file_id"."XXXXX".part.vtk
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("pvtk/");
  fname.append(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  if (out_params.gid >= 0) {
    fname.append(std::to_string(out_params.gid));
    fname.append(".");
  }
  fname.append(number);
  fname.append(".part.vtk");

  IOWrapper partfile;
  std::size_t header_offset=0;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::write);

  //  Write parts 1-4: Create string with header text.
  {
    std::stringstream msg;
    msg << "# vtk DataFile Version 2.0" << std::endl
        << "# AthenaK particle data at time= " << pm->time
        << "  nranks= " << global_variable::nranks
        << "  cycle=" << pm->ncycle
        << "  variables=" << out_params.variable << std::endl
        << "BINARY" << std::endl
        << "DATASET UNSTRUCTURED_GRID" << std::endl;

    if (global_variable::my_rank == 0) {
      partfile.Write_any_type(msg.str().c_str(),msg.str().size(),"byte");
    }
    header_offset += msg.str().size();
  }

  // Write Part 5: Write (x,y,z) positions of points
  {
    std::stringstream msg;
    msg << std::endl << "POINTS " << npout_total << " float" << std::endl;
    if (global_variable::my_rank == 0) {
      partfile.Write_any_type(msg.str().c_str(),msg.str().size(),"byte");
    }
    header_offset += msg.str().size();
  }
  // allocate 1D vector of floats used to convert and output particle data
  float *data = new float[3*npout_thisrank];
  // Loop over particles, load positions into data[]
  for (int p=0; p<npout_thisrank; ++p) {
    data[3*p] = static_cast<float>(outpart_rdata(IPX,p));
    if (pm->multi_d) {
      data[(3*p)+1] = static_cast<float>(outpart_rdata(IPY,p));
    } else {
      data[(3*p)+1] = static_cast<float>(pm->mesh_size.x2min);
    }
    if (pm->three_d) {
      data[(3*p)+2] = static_cast<float>(outpart_rdata(IPZ,p));
    } else {
      data[(3*p)+2] = static_cast<float>(pm->mesh_size.x3min);
    }
  }
  // swap data for this variable into big endian order
  if (!big_end) {
    for (int i=0; i<(3*npout_thisrank); ++i) { Swap4Bytes(&data[i]); }
  }
  // calculate local data offset
  std::vector<int> rank_offset(global_variable::nranks, 0);
  int npout_min = pm->nprtcl_eachrank[0];
  for (int n=1; n<global_variable::nranks; ++n) {
    rank_offset[n] = rank_offset[n-1] + pm->nprtcl_eachrank[n-1];
    npout_min = std::min(npout_min, pm->nprtcl_eachrank[n]);
  }

  // Write particle positions
  {
    std::size_t datasize = sizeof(float);
    std::size_t myoffset=header_offset + 3*rank_offset[global_variable::my_rank]*datasize;
    // collective writes for minimum number of particles across ranks
    if (partfile.Write_any_type_at_all(&(data[0]),3*npout_min,myoffset,"float")
          != 3*npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to vtk particle file, "
          << "vtk file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    // individual writes for remaining particles on each rank
    myoffset += datasize*3*npout_min;
    int nremain = pm->nprtcl_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(data[3*npout_min]),3*nremain,myoffset,"float")
            != 3*nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "particle data not written correctly to vtk particle file, "
            << "vtk file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    header_offset += 3*pm->nprtcl_total*datasize;
  }

  // Write Part 6: scalar particle data
  bool have_written_pointdata_header = false;

  // Write gid of points
  for (int n=0; n<(pm->pmb_pack->ppart->nidata); ++n) {
    std::stringstream msg;

    if (!have_written_pointdata_header) {
      have_written_pointdata_header = true;
      msg << std::endl << std::endl << "POINT_DATA " << npout_total << std::endl;
    }

    if (n == static_cast<int>(PGID)) {
      msg << std::endl << "SCALARS gid float" << std::endl
          << "LOOKUP_TABLE default" << std::endl;
    } else if (n == static_cast<int>(PTAG)) {
      msg << std::endl << "SCALARS ptag float" << std::endl
          << "LOOKUP_TABLE default" << std::endl;
    }

    if (global_variable::my_rank == 0) {
      partfile.Write_any_type_at(msg.str().c_str(),msg.str().size(),header_offset,"byte");
    }

    header_offset += msg.str().size();

    // Loop over particles, load gid into data[]
    for (int p=0; p<npout_thisrank; ++p) {
      data[p] = static_cast<float>(outpart_idata(n,p));
    }
    // swap data for this variable into big endian order
    if (!big_end) {
      for (int i=0; i<npout_thisrank; ++i) { Swap4Bytes(&data[i]); }
    }

    // calculate local data offset and write gid
    std::size_t datasize = sizeof(float);
    std::size_t myoffset=header_offset + rank_offset[global_variable::my_rank]*datasize;
    // collective writes for minimum number of particles across ranks
    if (partfile.Write_any_type_at_all(&(data[0]),npout_min,myoffset,"float")
          != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to vtk particle file, "
          << "vtk file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    // individual writes for remaining particles on each rank
    myoffset += datasize*npout_min;
    int nremain = pm->nprtcl_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(data[npout_min]),nremain,myoffset,"float")
            != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "particle data not written correctly to vtk particle file, "
            << "vtk file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    header_offset += pm->nprtcl_total*datasize;
  }

  // Add output of vectors here with header:
  // VECTORS vectors float
  // [then binary vx,vy,vz data .... ]

  // close the output file and clean up
  partfile.Close();
  delete[] data;

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  return;
}
