//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their history.

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "outputs.hpp"

#define NHISTORY_VARIABLES 8

//----------------------------------------------------------------------------------------
// ctor: also calls OutputType base class constructor

HistoryOutput::HistoryOutput(OutputParameters op, Mesh *pm) : OutputType(op, pm)
{
  AthenaArray<Real> new_data;
  new_data.SetSize(pm->nmbthisrank, NHISTORY_VARIABLES);
  new_data.SetLabel("hst_data");
  out_data_.push_back(new_data);
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadOutputData()
//  \brief Compute and store history data over all MeshBlocks on this rank in a single
//  AthenaArray

void HistoryOutput::LoadOutputData(Mesh *pm)
{ 
  // l op over all MeshBlocks
  MeshBlock &pmb = pm->mblocks.front();
  int is = pmb.mb_cells.is, ie = pmb.mb_cells.ie;
  int js = pmb.mb_cells.js, je = pmb.mb_cells.je;
  int ks = pmb.mb_cells.ks, ke = pmb.mb_cells.ke;
  int imb=0;
  for (auto &mb : pm->mblocks) {
    // initialize variable sums to 0.0
    for (int n=0; n<NHISTORY_VARIABLES; ++n) out_data_[0](imb,n) = 0.0;

    hydro::Hydro *phyd = mb.phydro;
    
    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=ks; k<=ke; ++k) { 
      for (int j=js; j<=je; ++j) { 
        for (int i=is; i<=ie; ++i) {
          
          // Hydro conserved variables:
          Real& u_d  = phyd->u0(hydro::IDN,k,j,i);
          Real& u_mx = phyd->u0(hydro::IM1,k,j,i);
          Real& u_my = phyd->u0(hydro::IM2,k,j,i);
          Real& u_mz = phyd->u0(hydro::IM3,k,j,i);
          
          out_data_[0](imb,0) += u_d;
          out_data_[0](imb,1) += u_mx;
          out_data_[0](imb,2) += u_my;
          out_data_[0](imb,3) += u_mz;
          out_data_[0](imb,4) += 0.5*SQR(u_mx)/u_d;
          out_data_[0](imb,5) += 0.5*SQR(u_my)/u_d;
          out_data_[0](imb,6) += 0.5*SQR(u_mz)/u_d;
          
          if (phyd->peos->adiabatic_eos) {
            Real& u_e = phyd->u0(hydro::IEN,k,j,i);;
            out_data_[0](imb,7) += u_e;
          }
        }
      }
    }

    // normalize sums by volume of this MeshBlock
    for (int n=0; n<NHISTORY_VARIABLES; ++n) {
      out_data_[0](imb,n) /= (mb.mb_cells.nx1)*(mb.mb_cells.nx2)*(mb.mb_cells.nx3);
      out_data_[0](imb,n) *= (mb.mb_size.x1max - mb.mb_size.x1min)*
         (mb.mb_size.x2max - mb.mb_size.x2min)*(mb.mb_size.x3max - mb.mb_size.x3min);
    }
    ++imb;
  }  // end loop over MeshBlocks

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::WriteOutputFile()
//  \brief Writes history file

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin)
{
  // sume history data in each MeshBlock
  // TODO add MPI communications for global sum
  AthenaArray<Real> hst_data;
  hst_data.SetSize(NHISTORY_VARIABLES);

  int imb=0;
  for (auto &mb : pm->mblocks) {
    for (int n=0; n<NHISTORY_VARIABLES; ++n) {
      hst_data(n) += out_data_[0](imb,n);
    }
    ++imb;
  }

  // only the master rank writes the file
  if (global_variable::my_rank == 0) {

    // create filename: "file_basename" + ".hst".  There is no file number.
    std::string fname;
    fname.assign(out_params.file_basename);
    fname.append(".hst");

    // open file for output
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" << std::endl;
      exit(EXIT_FAILURE);
    }

    // If this is the first output, write header
    if (out_params.file_number == 0) {
      int iout = 1;
      std::fprintf(pfile,"# Athena++ history data\n");
      std::fprintf(pfile,"# [%d]=time     ", iout++);
      std::fprintf(pfile,"[%d]=dt       ", iout++);
      std::fprintf(pfile,"[%d]=mass     ", iout++);
      std::fprintf(pfile,"[%d]=1-mom    ", iout++);
      std::fprintf(pfile,"[%d]=2-mom    ", iout++);
      std::fprintf(pfile,"[%d]=3-mom    ", iout++);
      std::fprintf(pfile,"[%d]=1-KE     ", iout++);
      std::fprintf(pfile,"[%d]=2-KE     ", iout++);
      std::fprintf(pfile,"[%d]=3-KE     ", iout++);
      if (pm->mblocks.begin()->phydro->peos->adiabatic_eos) {
        std::fprintf(pfile,"[%d]=tot-E   ", iout++);
      }
      std::fprintf(pfile,"\n");                              // terminate line
      // increment counters so headers are not written again
      out_params.file_number++;
      pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
    }

    // write history variables
    std::fprintf(pfile, out_params.data_format.c_str(), pm->time);
    std::fprintf(pfile, out_params.data_format.c_str(), pm->dt);
    for (int n=0; n<(NHISTORY_VARIABLES); ++n)
      std::fprintf(pfile, out_params.data_format.c_str(), hst_data(n));
    std::fprintf(pfile,"\n"); // terminate line
    std::fclose(pfile);
  }

  // increment counters, clean up
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
