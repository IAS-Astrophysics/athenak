//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart.cpp
//! \brief writes restart files

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
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

RestartOutput::RestartOutput(OutputParameters op, Mesh *pm) :
  BaseTypeOutput(op, pm) {
  // create directories for outputs. Comments in binary.cpp constructor explain why
  mkdir("rst",0775);
}

//----------------------------------------------------------------------------------------
// RestartOutput::LoadOutputData()
// overload of standard load data function specific to restarts.  Loads dependent
// variables, including ghost zones.

void RestartOutput::LoadOutputData(Mesh *pm) {
  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  int nhydro=0, nmhd=0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  Kokkos::realloc(outarray, nmb, (nhydro+nmhd), nout3, nout2, nout1);

  // load hydro (CC) data (copy to host)
  if (phydro != nullptr) {
    DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(phydro->u0);
    Kokkos::deep_copy(host_u0,phydro->u0);
    auto hst_slice = Kokkos::subview(outarray, Kokkos::ALL, std::make_pair(0,nhydro),
                                     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(hst_slice,host_u0);
  }

  // load MHD (CC and FC) data (copy to host)
  if (pmhd != nullptr) {
    DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(pmhd->u0);
    Kokkos::deep_copy(host_u0,pmhd->u0);
    auto hst_slice = Kokkos::subview(outarray, Kokkos::ALL, std::make_pair(nhydro,nmhd),
                                     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(hst_slice,host_u0);

    Kokkos::realloc(outfield.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(outfield.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(outfield.x3f, nmb, nout3+1, nout2, nout1);

    Kokkos::deep_copy(outfield.x1f,pmhd->b0.x1f);
    Kokkos::deep_copy(outfield.x2f,pmhd->b0.x2f);
    Kokkos::deep_copy(outfield.x3f,pmhd->b0.x3f);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RestartOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes everything to a single restart file

void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
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

  // total size of all cell-centered variables and face-centered fields to be written by
  // this rank
  IOWrapperSizeT ccdata_size = outarray.size()*sizeof(Real);
  IOWrapperSizeT fcdata_size = 0;
  if (pm->pmb_pack->pmhd != nullptr) {
    fcdata_size = (outfield.x1f.size() + outfield.x2f.size() + outfield.x3f.size())*
                  (sizeof(Real));
  }
  if (global_variable::my_rank == 0) {
    resfile.Write(&(ccdata_size), sizeof(IOWrapperSizeT), 1);
    resfile.Write(&(fcdata_size), sizeof(IOWrapperSizeT), 1);
  }

  // calculate size of data written in Steps 1-2 above
  IOWrapperSizeT step1size = sbuf.size()*sizeof(char) + 3*sizeof(int) + 2*sizeof(Real) +
                             sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  IOWrapperSizeT step2size = (pm->nmb_total)*(sizeof(LogicalLocation) + sizeof(float));

  // write cell-centered variables in parallel
  int mygids = pm->gidslist[global_variable::my_rank];
  IOWrapperSizeT myoffset  = step1size + step2size + 2*sizeof(IOWrapperSizeT) +
                (ccdata_size + fcdata_size)*(pm->gidslist[global_variable::my_rank]);
  resfile.Write_at_all(outarray.data(), ccdata_size, 1, myoffset);
  myoffset += ccdata_size;

  if (fcdata_size > 0) {
    resfile.Write_at_all(outfield.x1f.data(),outfield.x1f.size()*sizeof(Real),1,myoffset);
    myoffset += outfield.x1f.size()*sizeof(Real);

    resfile.Write_at_all(outfield.x2f.data(),outfield.x2f.size()*sizeof(Real),1,myoffset);
    myoffset += outfield.x2f.size()*sizeof(Real);

    resfile.Write_at_all(outfield.x3f.data(),outfield.x3f.size()*sizeof(Real),1,myoffset);
  }

  // close file, clean up
  resfile.Close();

  return;
}
