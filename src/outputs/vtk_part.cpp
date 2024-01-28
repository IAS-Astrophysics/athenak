//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vtk_part.cpp
//! \brief writes particle data in (legacy) vtk format.
//! Data is written in UNSTRUCTURED_GRID geometry, in BINARY format, and in FLOAT type
//! Data over multiple MeehBlocks and MPI ranks is written to a single file using MPI-IO.

#include <sys/stat.h>  // mkdir

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

ParticleVTKOutput::ParticleVTKOutput(OutputParameters op, Mesh *pm) :
  BaseTypeOutput(op, pm) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("pvtk",0775);
}

//----------------------------------------------------------------------------------------
// ParticleVTKOutput::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void ParticleVTKOutput::LoadOutputData(Mesh *pm) {
  int ndim=1;
  if (pm->multi_d) {ndim++;}
  if (pm->three_d) {ndim++;}
  particles::Particles *pp = pm->pmb_pack->ppart;
  nout_part = pp->nparticles_thispack;
  Kokkos::realloc(outpart_pos, nout_part, ndim);
  Kokkos::realloc(outpart_gid, nout_part);

  // Create mirror view on device of host view of output particle positions
  auto d_outpart_pos = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(), outpart_pos);
  // Copy particle positions into device mirror
  Kokkos::deep_copy(d_outpart_pos, pp->prtcl_pos);
  // Copy particle positions from device mirror to host output array
  Kokkos::deep_copy(outpart_pos, d_outpart_pos);

  // Copy particles gids from h_view to host output array
  Kokkos::deep_copy(outpart_gid, pp->prtcl_gid.h_view);
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

  //  Write parts 1-4: Create string with header text.
  std::stringstream msg;
  msg << "# vtk DataFile Version 2.0" << std::endl
      << "# AthenaK particle data at time= " << pm->time
      << "  nranks= " << global_variable::nranks
      << "  cycle=" << pm->ncycle
      << "  variables=" << out_params.variable << std::endl
      << "BINARY" << std::endl
      << "DATASET UNSTRUCTURED_GRID" << std::endl;

  //----- WRITE SERIAL FILES: -----
  // For serial (non-mpi) runs, use standard Unix-I/O functions.
  // open file and write header
  FILE *pfile;
  if ((pfile = std::fopen(fname.c_str(),"w")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
  }
  std::fprintf(pfile,"%s",msg.str().c_str());

  // Write Part 5: Write (x,y,z) positions of points
  // allocate 1D vector of floats used to convert and output particle data
  float *data = new float[3*nout_part];
  {
    std::stringstream pts_msg;
    pts_msg << std::endl << "POINTS " << nout_part << " float" << std::endl;
    std::fprintf(pfile,"%s",pts_msg.str().c_str());

    // Loop over particles, load positions into data[]
    for (int p=0; p<(3*nout_part); (p+=3)) {
      data[p] = static_cast<float>(outpart_pos(p,0));
      if (pm->multi_d) {
        data[p+1] = static_cast<float>(outpart_pos(p,1));
      } else {
        data[p+1] = static_cast<float>(pm->mesh_size.x2min);
      }
      if (pm->three_d) {
        data[p+2] = static_cast<float>(outpart_pos(p,2));
      } else {
        data[p+2] = static_cast<float>(pm->mesh_size.x3min);
      }
    }
    // swap data for this variable into big endian order
    if (!big_end) {
      for (int i=0; i<(3*nout_part); ++i) { Swap4Bytes(&data[i]); }
    }
    // now write the data as unformatted binary
    std::fwrite(&(data[0]), sizeof(float), (3*nout_part), pfile);
  }

  // Write Part 6: Write gid of points
  {
    std::stringstream gid_msg;
    gid_msg << std::endl << "POINT_DATA " << nout_part << std::endl
            << "SCALARS gid float 1 " << "LOOKUP_TABLE default" << std::endl;
    std::fprintf(pfile,"%s",gid_msg.str().c_str());

    // Loop over particles, load gid into data[]
    for (int p=0; p<nout_part; ++p) {
      data[p] = static_cast<float>(outpart_gid(p));
    }
    // swap data for this variable into big endian order
    if (!big_end) {
      for (int i=0; i<nout_part; ++i) { Swap4Bytes(&data[i]); }
    }
    // now write the data as unformatted binary
    std::fwrite(&(data[0]), sizeof(float), nout_part, pfile);
  }

  // Add output of vectors here with header:
  // VECTORS vectors float
  // [then binary vx,vy,vz data .... ]

  // close the output file and clean up
  std::fclose(pfile);
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
