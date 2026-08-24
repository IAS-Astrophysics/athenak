//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vtk_prtcl.cpp
//! \brief writes particle data in (legacy) vtk format.
//! Data is written in UNSTRUCTURED_GRID geometry and BINARY format.
//! Data over multiple MeehBlocks and MPI ranks is written to a single file using MPI-IO.

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
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

[[noreturn]] void FatalParticleVTKOutput(const std::string &message) {
  std::cout << "### FATAL ERROR in vtk_prtcl.cpp" << std::endl
            << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor
// Checks compatibility options for VTK outputs

ParticleVTKOutput::ParticleVTKOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op), npout_thisrank(0), npout_total(0) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("pvtk",0775);
  if (pm->pmb_pack->ppart == nullptr) {
    FatalParticleVTKOutput("file_type=pvtk requires particles");
  }
  particles::ParticlePopulation *population =
      pm->pmb_pack->ppart->FindPopulation("particles");
  if (population == nullptr) {
    FatalParticleVTKOutput("particle population 'particles' was not found");
  }
  if (pm->pgen != nullptr) {
    for (const auto &variable : pm->pgen->user_particle_vtk_output_variables) {
      for (int n=0; n<population->nidata; ++n) {
        if (population->int_output[n] && variable.name == population->int_names[n]) {
          FatalParticleVTKOutput("user particle VTK output variable '" +
                                 variable.name +
                                 "' conflicts with a stored particle field");
        }
      }
      for (int n=0; n<population->nrdata; ++n) {
        if (population->real_output[n] && variable.name == population->real_names[n]) {
          FatalParticleVTKOutput("user particle VTK output variable '" +
                                 variable.name +
                                 "' conflicts with a stored particle field");
        }
      }
      user_real_names.push_back(variable.name);
    }
  }
}

//----------------------------------------------------------------------------------------
// ParticleVTKOutput::LoadOutputData()
// Copies particle data to host and evaluates output callbacks on the live device views.

void ParticleVTKOutput::LoadOutputData(Mesh *pm) {
  particles::ParticlePopulation *pp =
      pm->pmb_pack->ppart->FindPopulation("particles");
  npout_thisrank = pm->nprtcl_thisrank;
  npout_total = pm->nprtcl_total;
  Kokkos::realloc(outpart_rdata, pp->nrdata, npout_thisrank);
  Kokkos::realloc(outpart_idata, pp->nidata, npout_thisrank);
  Kokkos::realloc(outpart_user_rdata, user_real_names.size(), npout_thisrank);

  // Copy the live device particle views through host-accessible mirrors.
  auto h_outpart_rdata = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(),
                                                    outpart_rdata);
  auto h_outpart_idata = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(),
                                                    outpart_idata);
  Kokkos::deep_copy(h_outpart_rdata, pp->prtcl_rdata);
  Kokkos::deep_copy(h_outpart_idata, pp->prtcl_idata);
  Kokkos::deep_copy(outpart_rdata, h_outpart_rdata);
  Kokkos::deep_copy(outpart_idata, h_outpart_idata);

  const std::size_t registered_user_fields = (pm->pgen == nullptr) ? 0 :
      pm->pgen->user_particle_vtk_output_variables.size();
  if (registered_user_fields != user_real_names.size()) {
    FatalParticleVTKOutput("user particle VTK output registry changed after output setup");
  }
  if (!user_real_names.empty()) {
    DvceArray2D<Real> device_user_rdata(
        "particle_vtk_user_rdata", user_real_names.size(), npout_thisrank);
    if (npout_thisrank > 0) {
      Kokkos::deep_copy(device_user_rdata,
                        std::numeric_limits<Real>::quiet_NaN());
    }
    for (std::size_t n=0; n<user_real_names.size(); ++n) {
      const auto &variable = pm->pgen->user_particle_vtk_output_variables[n];
      if (variable.name != user_real_names[n] || variable.function == nullptr) {
        FatalParticleVTKOutput(
            "user particle VTK output registry changed after output setup");
      }
      particles::ParticleOutputData output(
          pm->pmb_pack, pp, pp->prtcl_rdata, pp->prtcl_idata, device_user_rdata,
          static_cast<int>(n), npout_thisrank);
      variable.function(&output);
    }
    if (npout_thisrank > 0) {
      Kokkos::deep_copy(outpart_user_rdata, device_user_rdata);
    }
  }
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
  const particles::ParticlePopulation *pp =
      pm->pmb_pack->ppart->FindPopulation("particles");
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
  float *data = new float[std::max(3*npout_thisrank, 1)];
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
    if (npout_min > 0 &&
        partfile.Write_any_type_at_all(&(data[0]),3*npout_min,myoffset,"float")
          != static_cast<size_t>(3*npout_min)) {
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
            != static_cast<size_t>(3*nremain)) {
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

  // Write all integer particle data
  int *idata = new int[std::max(npout_thisrank, 1)];
  for (int n=0; n<pp->nidata; ++n) {
    if (!pp->int_output[n]) continue;
    std::stringstream msg;

    if (!have_written_pointdata_header) {
      have_written_pointdata_header = true;
      msg << std::endl << std::endl << "POINT_DATA " << npout_total << std::endl;
    }

    msg << std::endl << "SCALARS " << pp->int_names[n] << " int" << std::endl
        << "LOOKUP_TABLE default" << std::endl;

    if (global_variable::my_rank == 0) {
      partfile.Write_any_type_at(msg.str().c_str(),msg.str().size(),header_offset,"byte");
    }

    header_offset += msg.str().size();

    // Loop over particles, load integer field into idata[]
    for (int p=0; p<npout_thisrank; ++p) {
      idata[p] = outpart_idata(n,p);
    }
    // swap data for this variable into big endian order
    if (!big_end) {
      for (int i=0; i<npout_thisrank; ++i) { Swap4Bytes(&idata[i]); }
    }

    // calculate local data offset and write integer field
    std::size_t datasize = sizeof(int);
    std::size_t myoffset=header_offset + rank_offset[global_variable::my_rank]*datasize;
    // collective writes for minimum number of particles across ranks
    if (npout_min > 0 &&
        partfile.Write_any_type_at_all(&(idata[0]),npout_min,myoffset,"int")
          != static_cast<size_t>(npout_min)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to vtk particle file, "
          << "vtk file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    // individual writes for remaining particles on each rank
    myoffset += datasize*npout_min;
    int nremain = pm->nprtcl_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(idata[npout_min]),nremain,myoffset,"int")
            != static_cast<size_t>(nremain)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "particle data not written correctly to vtk particle file, "
            << "vtk file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    header_offset += pm->nprtcl_total*datasize;
  }
  delete[] idata;

  // Positions are already stored as VTK points. Write all remaining real particle data,
  // followed by pgen-supplied output quantities.
  std::vector<int> real_fields;
  for (int n=3; n<pp->nrdata; ++n) {
    if (pp->real_output[n]) real_fields.push_back(n);
  }
  const int nstored_real = static_cast<int>(real_fields.size());
  const int noutput_real = nstored_real + static_cast<int>(user_real_names.size());
  for (int output_field=0; output_field<noutput_real; ++output_field) {
    const bool is_user_field = (output_field >= nstored_real);
    const int n = is_user_field ? output_field-nstored_real : real_fields[output_field];
    const std::string name = is_user_field ? user_real_names[n] : pp->real_names[n];
    std::stringstream msg;
    if (!have_written_pointdata_header) {
      have_written_pointdata_header = true;
      msg << std::endl << std::endl << "POINT_DATA " << npout_total << std::endl;
    }
    msg << std::endl << "SCALARS " << name << " float" << std::endl
        << "LOOKUP_TABLE default" << std::endl;

    if (global_variable::my_rank == 0) {
      partfile.Write_any_type_at(msg.str().c_str(),msg.str().size(),header_offset,"byte");
    }
    header_offset += msg.str().size();

    for (int p=0; p<npout_thisrank; ++p) {
      data[p] = static_cast<float>(is_user_field ? outpart_user_rdata(n,p) :
                                                   outpart_rdata(n,p));
    }
    if (!big_end) {
      for (int i=0; i<npout_thisrank; ++i) { Swap4Bytes(&data[i]); }
    }

    std::size_t datasize = sizeof(float);
    std::size_t myoffset=header_offset + rank_offset[global_variable::my_rank]*datasize;
    if (npout_min > 0 &&
        partfile.Write_any_type_at_all(&(data[0]),npout_min,myoffset,"float")
          != static_cast<size_t>(npout_min)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to vtk particle file, "
          << "vtk file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    myoffset += datasize*npout_min;
    int nremain = pm->nprtcl_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(data[npout_min]),nremain,myoffset,"float")
            != static_cast<size_t>(nremain)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "particle data not written correctly to vtk particle file, "
            << "vtk file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    header_offset += pm->nprtcl_total*datasize;
  }

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
