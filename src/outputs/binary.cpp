//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file binary.cpp
//! \brief writes output data in binary format, which simply consists of each MeshBlock
//! written contiguously in order of "gid" in binary format.

#include <sys/stat.h>  // mkdir

#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

BinaryOutput::BinaryOutput(OutputParameters op, Mesh *pm) :
  BaseTypeOutput(op, pm) {
  // create directories for outputs
  // useful for mpiio-based outputs because on some supercomputers you may need to
  // set different stripe counts depending on whether mpiio is used in order to
  // achieve the best performance and not to crash the filesystem
  mkdir("bin",0775);
}

//----------------------------------------------------------------------------------------
//! \fn void BinaryOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes OutputData in binary format
//   All MeshBlocks are written to the same file.

void BinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // create filename: "bin/file_basename" + "." + "file_id" + "." + XXXXX + ".bin"
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("bin/");
  fname.append(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  fname.append(number);
  fname.append(".bin");

  IOWrapper binfile;
  std::size_t header_offset=0;
  binfile.Open(fname.c_str(), IOWrapper::FileMode::write);

  // Basic parts of the format:
  // 1. Size of the header
  // 2. Current time
  // 3. List of variables in the file
  // 4. Header (input file information)
  {std::stringstream msg;
  msg << "Athena binary output version=1.1" << std::endl
      // preheader size includes "size of preheader" line up to "number of variables"
      << "  size of preheader=5" << std::endl
      << "  time=" << pm->time << std::endl
      << "  cycle=" << pm->ncycle << std::endl
      << "  size of location=" << sizeof(Real) << std::endl
      << "  size of variable=" << sizeof(float) << std::endl
      << "  number of variables=" << outvars.size() << std::endl
      << "  variables:  ";
  for (int n=0; n<outvars.size(); n++) {
    msg << outvars[n].label.c_str() << "  ";
  }
  msg << std::endl;
  if (global_variable::my_rank == 0) {
    binfile.Write_bytes(msg.str().c_str(),sizeof(char),msg.str().size());
  }
  header_offset += msg.str().size();}
  {std::stringstream msg;
  // prepare the input parameters
  std::stringstream ost;
  pin->ParameterDump(ost);
  std::string sbuf=ost.str();
  msg << "  header offset=" << sbuf.size()*sizeof(char)  << std::endl;
  if (global_variable::my_rank == 0) {
    binfile.Write_bytes(msg.str().c_str(),sizeof(char),msg.str().size());
    binfile.Write_bytes(sbuf.c_str(),sizeof(char),sbuf.size());
  }
  header_offset += sbuf.size()*sizeof(char);
  header_offset += msg.str().size();}

  //  5. Data.  An arbitrary number of scalars and vectors can be written (every node
  //  in the OutputData doubly linked lists), all in binary floats format

  int nout_vars = outvars.size();
  auto &indcs = pm->mb_indcs;
  int cells = indcs.nx1*indcs.nx2*indcs.nx3;

  // il1, il2, il3, level + x1min, x1max, x2min, x2max, x3min, x3max + data
  std::size_t data_size = 4*sizeof(int32_t) + 6*sizeof(Real)
                        + (cells*nout_vars)*sizeof(float);

  int nout_mbs = (outmbs.size());
  int ns_mbs = pm->gidslist[global_variable::my_rank];
  int nb_mbs = pm->nmblist[global_variable::my_rank];

  // allocate 1D vector of floats used to convert and output data
  char *data = new char[nb_mbs*data_size];
  float *single_data = new float[cells];

  // Loop over MeshBlocks
  for (int m=0; m<nout_mbs; ++m) {
    char *pdata=&(data[m*data_size]);
    LogicalLocation loc = pm->lloclist[outmbs[m].mb_gid];
    int &ois = outmbs[m].ois;
    int &oie = outmbs[m].oie;
    int &ojs = outmbs[m].ojs;
    int &oje = outmbs[m].oje;
    int &oks = outmbs[m].oks;
    int &oke = outmbs[m].oke;

    // logical location lx1, lx2, lx3
    int32_t nx = (int32_t)(loc.lx1);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx2);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx3);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // physical refinement level
    nx = (int32_t)(loc.level-pm->root_level);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // coordinate location
    Real xv = outmbs[m].x1min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x1max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);

    // output variables
    float tmp_data;
    for (int n=0; n<nout_vars; n++) {
      int cnt=0;
      for (int k=oks; k<=oke; k++) {
        for (int j=ojs; j<=oje; j++) {
          for (int i=ois; i<=oie; i++) {
            tmp_data = static_cast<float>(outarray(n,m,k-oks,j-ojs,i-ois));
            single_data[cnt] = tmp_data;
            cnt++;
          }
        }
      }
      memcpy(pdata,single_data,cells*sizeof(float));
      pdata+=cells*sizeof(float);
    }
  }

  // now write binary data in parallel
  std::size_t myoffset=header_offset+data_size*ns_mbs;
  binfile.Write_bytes_at_all(data,data_size,nb_mbs,myoffset);

  // close the output file and clean up ptrs to data
  binfile.Close();
  delete [] data;
  delete [] single_data;

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
