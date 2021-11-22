//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart.cpp
//! \brief writes restart files

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>  // mkdir

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls OutputType base class constructor

RestartOutput::RestartOutput(OutputParameters op, Mesh *pm)
  : OutputType(op, pm)
{
  // create directories for outputs. Comments in binary.cpp constructor explain why
  mkdir("rst",0775);
}

//----------------------------------------------------------------------------------------
//! \fn void RestartOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes everything to a single restart file

void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin)
{
  // create filename: "rst/file_basename" + "." + XXXXX + ".rst"
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("rst/");
  fname.append(out_params.file_basename);
  fname.append(".");
  fname.append(number);
  fname.append(".rst");

  // increment counters now so values for *next* dump are stored in restart file
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  // create string holding input parameters (copy of input file)
  std::stringstream ost;
  pin->ParameterDump(ost);
  std::string sbuf = ost.str();

  // calculate size of header
  IOWrapperSizeT headeroffset = sbuf.size()*sizeof(char) + 3*sizeof(int) + 2*sizeof(Real)
    + sizeof(RegionSize) + 3*sizeof(RegionIndcs) + sizeof(IOWrapperSizeT);
  // the size of variables stored in each MeshBlockPack
  size_t datasize = pm->pmb_pack->GetMeshBlockPackArraySizeInBytes();

  // open file and  write the header; this part is serial
  IOWrapper rstfile;
  rstfile.Open(fname.c_str(), IOWrapper::FileMode::write);
  if (global_variable::my_rank == 0) {
    // output the input parameters (input file)
    rstfile.Write(sbuf.c_str(),sizeof(char),sbuf.size());

    // output Mesh information
    rstfile.Write(&(pm->nmb_total), sizeof(int), 1);
    rstfile.Write(&(pm->root_level), sizeof(int), 1);
    rstfile.Write(&(pm->mesh_size), sizeof(RegionSize), 1);
    rstfile.Write(&(pm->mesh_indcs), sizeof(RegionIndcs), 1);
    rstfile.Write(&(pm->mb_indcs), sizeof(RegionIndcs), 1);
    rstfile.Write(&(pm->mb_cindcs), sizeof(RegionIndcs), 1);
    rstfile.Write(&(pm->time), sizeof(Real), 1);
    rstfile.Write(&(pm->dt), sizeof(Real), 1);
    rstfile.Write(&(pm->ncycle), sizeof(int), 1);
    rstfile.Write(&(datasize), sizeof(IOWrapperSizeT), 1);
  }

  // allocate memory for the ID list
  IOWrapperSizeT listsize = sizeof(LogicalLocation) + sizeof(Real);
  int mynmb = pm->nmblist[global_variable::my_rank];
  char *idlist = new char[listsize*mynmb];

  // Loop over MeshBlockPack and pack ID and cost lists
  int os=0;
  for (int id=(pm->pmb_pack->gids); id<=(pm->pmb_pack->gide); ++id) {
    std::memcpy(&(idlist[os]), &(pm->lloclist[id]), sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
    std::memcpy(&(idlist[os]), &(pm->costlist[id]), sizeof(double));
    os += sizeof(double);
  }

  // write the ID list collectively
  int mygids = pm->gidslist[global_variable::my_rank];
  IOWrapperSizeT myoffset = headeroffset + listsize*mygids;
  rstfile.Write_at_all(idlist, listsize, mynmb, myoffset);

  // deallocate the idlist array
  delete [] idlist;

  rstfile.Close();

  return;
}
