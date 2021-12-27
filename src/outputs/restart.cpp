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
// RestartOutput::LoadOutputData()
// overload of standard load data function specific to restarts.  Loads dependent
// variables, including ghost zones.

void RestartOutput::LoadOutputData(Mesh *pm)
{ 
  // get size of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // load Hydro data
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  if (phydro != nullptr) {
    // resize host array to proper dimensions
    int nvar = phydro->nhydro + phydro->nscalars; 
    Kokkos::realloc(outdata, nmb, nvar, nout3, nout2, nout1);
  
    // Now copy data Hydro conserved variables to host, using device mirror
//    DvceArray5D<Real>::HostMirror hst_data = Kokkos::create_mirror(u0_);
    Kokkos::deep_copy(outdata,phydro->u0);
  }
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

  //--- STEP 1.  Root process writes header data (input file, critical variables)
  // Input file data is read by ParameterInput on restart, and the remaining header
  // variables are read in Mesh::BuildTreeFromRestart()

  // open file and  write the header; this part is serial
  IOWrapper resfile;
  resfile.Open(fname.c_str(), IOWrapper::FileMode::write);
  if (global_variable::my_rank == 0) {
    // output the input parameters (input file)
    resfile.Write(sbuf.c_str(),sizeof(char),sbuf.size());

    // output Mesh information
    resfile.Write(&(pm->nmb_total), sizeof(int), 1);
    resfile.Write(&(pm->root_level), sizeof(int), 1);
    resfile.Write(&(pm->mesh_size), sizeof(RegionSize), 1);
    resfile.Write(&(pm->mesh_indcs), sizeof(RegionIndcs), 1);
    resfile.Write(&(pm->mb_indcs), sizeof(RegionIndcs), 1);
    resfile.Write(&(pm->time), sizeof(Real), 1);
    resfile.Write(&(pm->dt), sizeof(Real), 1);
    resfile.Write(&(pm->ncycle), sizeof(int), 1);
  }

  //--- STEP 2.  Root process writes list of logical locations and cost of MeshBlocks
  // This data read in Mesh::BuildTreeFromRestart()

  if (global_variable::my_rank == 0) {
    resfile.Write(&(pm->lloclist[0]), (pm->nmb_total)*sizeof(LogicalLocation), 1);
    resfile.Write(&(pm->costlist[0]), (pm->nmb_total)*sizeof(float), 1);
  }

  //--- STEP 3.  All ranks write data over all MeshBlocks (5D arrays) in parallel
  // This data read in ProblemGenerator constructor for restarts

  // write the size of variables stored in each MeshBlockPack
  IOWrapperSizeT datasize = outdata.size()*sizeof(Real);
  if (global_variable::my_rank == 0) {
    resfile.Write(&(datasize), sizeof(IOWrapperSizeT), 1);
  }

  // now write restart data in parallel
  IOWrapperSizeT step1size = sbuf.size()*sizeof(char) + 3*sizeof(int) + 2*sizeof(Real) +
                             sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  IOWrapperSizeT step2size = (pm->nmb_total)*(sizeof(LogicalLocation) + sizeof(float));
  int mygids = pm->gidslist[global_variable::my_rank];
  IOWrapperSizeT myoffset  = step1size + step2size + sizeof(IOWrapperSizeT) +
                             datasize*(pm->gidslist[global_variable::my_rank]);
  resfile.Write_at_all(outdata.data(), datasize, 1, myoffset);

  // close file, clean up
  resfile.Close();

  return;
}
