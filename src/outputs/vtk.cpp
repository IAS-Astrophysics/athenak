//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vtk.cpp
//  \brief writes output data in (legacy) vtk format.
//  Data is written in RECTILINEAR_GRID geometry, in BINARY format, and in FLOAT type

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls OutputType base class constructor

VTKOutput::VTKOutput(OutputParameters op, Mesh *pm)
  : OutputType(op, pm)
{
}

//----------------------------------------------------------------------------------------
// Functions to detect big endian machine, and to byte-swap 32-bit words.  The vtk
// legacy format requires data to be stored as big-endian.

namespace swap_functions {

int IsBigEndian()
{
  std::int32_t n = 1;
  char *ep = reinterpret_cast<char *>(&n);
  return (*ep == 0); // Returns 1 (true) on a big endian machine
}

inline void Swap4Bytes(void *vdat)
{
  char tmp, *dat = static_cast<char *>(vdat);
  tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
  tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
}

} // namespace swap_functions

//----------------------------------------------------------------------------------------
//! \fn void VTKOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes OutputData in (legacy) vtk format, one
//         MeshBlock per file

void VTKOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin)
{
  int big_end = swap_functions::IsBigEndian(); // =1 on big endian machine

  // create filename: "file_basename" + "." + "file_id" + "." + XXXXX + ".vtk"
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  fname.append(number);
  fname.append(".vtk");

  // open file for output
  FILE *pfile;
  std::stringstream msg;
  if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Output file '" << fname << "' could not be opened" << std::endl;
    exit(EXIT_FAILURE);
  }

  // There are five basic parts to the VTK "legacy" file format.
  //  1. Write file version and identifier
  std::fprintf(pfile, "# vtk DataFile Version 2.0\n");

  //  2. Header
  std::fprintf(pfile, "# Athena++ data at time=%e", pm->time);
  std::fprintf(pfile, "  cycle=%d", pm->ncycle);
  std::fprintf(pfile, "  variables=%s \n", out_params.variable.c_str());

  //  3. File format
  std::fprintf(pfile, "BINARY\n");

  //  4. Dataset structure
  // TODO numbers of cells in entire grid
  int ncoord1 = (nout1 > 1)? nout1+1 : nout1;
  int ncoord2 = (nout2 > 1)? nout2+1 : nout2;
  int ncoord3 = (nout3 > 1)? nout3+1 : nout3;

  // allocate 1D vector of floats used to convert and output data
  float *data;
  int ndata = std::max(std::max(ncoord1, ncoord2), ncoord3);
  data = new float[3*ndata];

  // Specify the type of data, dimensions, and coordinates.  If N>1, then write N+1
  // cell faces as binary floats.  If N=1, then write 1 cell center position.
  std::fprintf(pfile, "DATASET RECTILINEAR_GRID\n");
  std::fprintf(pfile, "DIMENSIONS %d %d %d\n", ncoord1, ncoord2, ncoord3);

  // TODO coordinates of entire grid for multiple MeshBlocks
  // write x1-coordinates as binary float in big endian order
  std::fprintf(pfile, "X_COORDINATES %d float\n", ncoord1);
  if (nout1 == 1) {
    data[0] = static_cast<float>(x1_cc_(0,ois));
  } else {
    for (int i=0; i<ncoord1; ++i) {
      data[i] = static_cast<float>(x1_fc_(0,i));
    }
  }
  if (!big_end) {for (int i=0; i<ncoord1; ++i) swap_functions::Swap4Bytes(&data[i]);}
  std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncoord1), pfile);

  // write x2-coordinates as binary float in big endian order
  std::fprintf(pfile, "\nY_COORDINATES %d float\n", ncoord2);
  if (nout2 == 1) {
    data[0] = static_cast<float>(x2_cc_(0,ojs));
  } else {
    for (int j=0; j<ncoord2; ++j) {
      data[j] = static_cast<float>(x2_fc_(0,j));
    }
  }
  if (!big_end) {for (int i=0; i<ncoord2; ++i) swap_functions::Swap4Bytes(&data[i]);}
  std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncoord2), pfile);

  // write x3-coordinates as binary float in big endian order
  std::fprintf(pfile, "\nZ_COORDINATES %d float\n", ncoord3);
  if (nout3 == 1) {
    data[0] = static_cast<float>(x3_cc_(0,oks));
  } else {
    for (int k=0; k<ncoord3; ++k) {
      data[k] = static_cast<float>(x3_fc_(0,k));
    }
  }
  if (!big_end) {for (int i=0; i<ncoord3; ++i) swap_functions::Swap4Bytes(&data[i]);}
  std::fwrite(data, sizeof(float), static_cast<std::size_t>(ncoord3), pfile);

  //  5. Data.  An arbitrary number of scalars and vectors can be written (every node
  //  in the OutputData doubly linked lists), all in binary floats format
  std::fprintf(pfile, "\nCELL_DATA %d", nout1*nout2*nout3);

  // the out_data_ vector stores each variable to be output over all cells and MeshBlocks.
  // So start iteration over elements of out_data_ vector (variables)

  for (auto it : out_data_) {
    // TODO get VECTORS working
    // write data type (SCALARS or VECTORS) and name
//    std::fprintf(pfile, "\n%s %s float\n", pdata->type.c_str(),  pdata->name.c_str());
//    int nvar = pdata->data.GetDim4();
//    if (nvar == 1) std::fprintf(pfile, "LOOKUP_TABLE default\n");
    std::fprintf(pfile, "\nSCALARS %s float\n", it.GetLabel().c_str());
    std::fprintf(pfile, "LOOKUP_TABLE default\n");

  // TODO write data properly for multiple MeshBlocks
    // Loop over MeshBlocks
    for (int n=0; n<pm->nmbthisrank; ++n) {
      for (int k=0; k<nout3; ++k) {
        for (int j=0; j<nout2; ++j) { 
          for (int i=0; i<nout1; ++i) {
            data[i] = static_cast<float>(it(n,k,j,i));
          }

          // write data in big endian order
          if (!big_end) {
            for (int i=0; i<nout1; ++i)
              swap_functions::Swap4Bytes(&data[i]);
          }
          std::fwrite(data, sizeof(float), static_cast<std::size_t>(nout1), pfile);
        }
      }
    }  // end loop over MeshBlocks
  }

  // don't forget to close the output file and clean up ptrs to data in OutputData
  std::fclose(pfile);
  delete [] data;

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
