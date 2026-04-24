//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vtk_mesh.cpp
//! \brief writes mesh data in (legacy) vtk format.
//! Data is written in STRUCTURED_POINTS geometry, in BINARY format, and in FLOAT type
//! Data over multiple MeshBlocks and MPI ranks is written to a single file using MPI-IO.

// TODO(@user): create new communicator for MPI-IO for slicing, including only those ranks
// that have MeshBlocks in slice.  Current design segfaults with slicing if there are
// ranks that do not write.

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <limits> // numeric_limits<>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor
// Checks compatibility options for VTK outputs

MeshVTKOutput::MeshVTKOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("vtk",0775);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshVTKOutput:::WriteOutputFile(Mesh *pm)
//! \brief Cycles over all MeshBlocks and writes output data in (legacy) vtk format.
//! With MPI, all MeshBlocks are written to the same file in proper order.
//!
//! There are five basic parts to the VTK "legacy" file format for structured meshes.
//!  1. File version and identifier
//!  2. Header (time, cycle, variables, etc.)
//!  3. File format
//!  4. Dataset structure, including type and dimensions of data, and coordinates.
//!  5. Data.  An arbitrary number of scalars and vectors can be written

void MeshVTKOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  int big_end = IsBigEndian(); // =1 on big endian machine
  // create filename: "vtk/file_basename"."file_id"."gid"."XXXXX".vtk
  // where XXXXX = 5-digit file_number, and gid only added if specified
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("vtk/");
  fname.append(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  if (out_params.gid >= 0) {
    fname.append(std::to_string(out_params.gid));
    fname.append(".");
  }
  fname.append(number);
  fname.append(".vtk");

  //  Select data ranges for outputs (entire grid unless gid is specified)
  int nout1=0,nout2=0,nout3=0;   // total number of cells output (across all MeshBlocks)
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout_mbs = (outmbs.size());
  if ((pm->nmb_total > 1) && (out_params.gid < 0)) {
    nout1 = (out_params.slice1)? 1 : (pm->nmb_rootx1*indcs.nx1);
    nout2 = (out_params.slice2)? 1 : (pm->nmb_rootx2*indcs.nx2);
    nout3 = (out_params.slice3)? 1 : (pm->nmb_rootx3*indcs.nx3);
  } else if (nout_mbs > 0) {
    nout1 = outmbs[0].oie - outmbs[0].ois + 1;
    nout2 = outmbs[0].oje - outmbs[0].ojs + 1;
    nout3 = outmbs[0].oke - outmbs[0].oks + 1;
  }
  int ncoord1 = (nout1 > 1)? nout1+1 : nout1;
  int ncoord2 = (nout2 > 1)? nout2+1 : nout2;
  int ncoord3 = (nout3 > 1)? nout3+1 : nout3;

  // Write parts 1-4: Create string with header text.
  std::stringstream msg;
  msg << "# vtk DataFile Version 2.0" << std::endl
      << "# Athena++ data at time= " << pm->time
      << "  level= 0"  // assuming uniform mesh
      << "  nranks= " << global_variable::nranks
      << "  cycle=" << pm->ncycle
      << "  variables=" << out_params.variable
      << std::endl << "BINARY" << std::endl
      << "DATASET STRUCTURED_POINTS" << std::endl
      << "DIMENSIONS " << ncoord1 << " " << ncoord2 << " " << ncoord3 << std::endl;

  // Specify uniform Cartesian mesh with grid minima and spacings
  // output physical dimensions of entire grid, unless gid is specified
  Real x1min,x2min,x3min,dx1,dx2,dx3;
  if (out_params.gid < 0) {
    x1min = pm->mesh_size.x1min;
    x2min = pm->mesh_size.x2min;
    x3min = pm->mesh_size.x3min;
    dx1 = pm->mesh_size.dx1;
    dx2 = pm->mesh_size.dx2;
    dx3 = pm->mesh_size.dx3;
  } else {
    x1min = pm->pmb_pack->pmb->mb_size.h_view(out_params.gid).x1min;
    x2min = pm->pmb_pack->pmb->mb_size.h_view(out_params.gid).x2min;
    x3min = pm->pmb_pack->pmb->mb_size.h_view(out_params.gid).x3min;
    dx1 = pm->pmb_pack->pmb->mb_size.h_view(out_params.gid).dx1;
    dx2 = pm->pmb_pack->pmb->mb_size.h_view(out_params.gid).dx2;
    dx3 = pm->pmb_pack->pmb->mb_size.h_view(out_params.gid).dx3;
  }
  if (out_params.include_gzs) {
    x1min -= (indcs.ng)*dx1;
    x2min -= (indcs.ng)*dx2;
    x3min -= (indcs.ng)*dx3;
  }
  msg.seekp(0, std::ios_base::end);
  msg << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1)
      << "ORIGIN " << x1min << " " << x2min << " " << x3min << " " <<  std::endl
      << "SPACING " << dx1  << " " << dx2   << " " << dx3   << " " <<  std::endl;

  // Write part 5: An arbitrary number of scalars and vectors can be written (every
  // element of the outvars vector), all in binary floats format
  msg.seekp(0, std::ios_base::end);
  msg << std::endl << "CELL_DATA " << nout1*nout2*nout3 << std::endl;

  bool parallel_write=false;
#if MPI_PARALLEL_ENABLED
  //----- WRITE IN PARALLEL WITH MPI: -----
  // For MPI runs, create derived data types for outarray and Cartesian grid of MBs.
  // MPI then takes care of writing to file in proper order.
  // If GID is specified in input file so only one MB written, then file is written
  // using serial mode below

  // open file and write file header
  if ((pm->nmb_total > 1) && (out_params.gid < 0)) {
    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
        exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      MPI_File_write(fh, msg.str().c_str(), msg.str().size(), MPI_BYTE,MPI_STATUS_IGNORE);
    }
    size_t header_size = msg.str().size();

    // allocate 1D vector of floats used to convert and output entire 3D data
    int nx1 = outmbs[0].oie - outmbs[0].ois + 1;
    int nx2 = outmbs[0].oje - outmbs[0].ojs + 1;
    int nx3 = outmbs[0].oke - outmbs[0].oks + 1;
    float *data = new float[nx1*nx2*nx3];

    // create new datatype representing array of cells in MeshBlocks
    MPI_Datatype block;
    int bsize[3] = {nx3, nx2, nx1};      // total number of cells in MB
    int bstrt[3] = {0, 0, 0};            // i/j/k starting index of this block
    MPI_Type_create_subarray(3,bsize,bsize,bstrt,MPI_ORDER_C,MPI_FLOAT,&block);
    MPI_Type_commit(&block);

    // create new datatype representing grid of MeshBlocks
    MPI_Datatype grid;
    int gridsize[3] = {nout3, nout2, nout1};   // total number of cells over all MBs
    int mbstrt[3] = {0, 0, 0};                 // i/j/k starting index of blocks
    int mbsize[3] = {nx3, nx2, nx1};           // number of cells in blocks
    MPI_Type_create_subarray(3,gridsize,mbsize,mbstrt,MPI_ORDER_C,MPI_FLOAT,&grid);
    MPI_Type_commit(&grid);

    // Loop over variables
    int nout_vars = outvars.size();
    for (int n=0; n<nout_vars; ++n) {
      // write header this variable (SCALARS or VECTORS, name, type, color table)
      std::stringstream data_msg;
      data_msg << std::endl << "SCALARS " << outvars[n].label.c_str()
               << " float" << std::endl
               << "LOOKUP_TABLE default" << std::endl;
      if (global_variable::my_rank == 0) {
        MPI_File_write(fh, data_msg.str().c_str(), data_msg.str().size(),
                          MPI_BYTE, MPI_STATUS_IGNORE);
      }
      header_size += data_msg.str().size();

      // Loop over max number of MeshBlocks to be written on any rank
      // This guarantees collective MPI functions are called by all ranks
      MPI_Datatype mygrid;
      for (int m=0; m<noutmbs_max; ++m) {
        // if there is a MB to be written, set location in 3D grid of MBs in output file.
        if (m < nout_mbs) {
          LogicalLocation lloc = pm->lloc_eachmb[outmbs[m].mb_gid];
          // calculate indices of this MeshBlock in 3D grid of MBs
          int imb = (out_params.slice1 || (out_params.gid >= 0))? 0 : lloc.lx1;
          int jmb = (out_params.slice2 || (out_params.gid >= 0))? 0 : lloc.lx2;
          int kmb = (out_params.slice3 || (out_params.gid >= 0))? 0 : lloc.lx3;

          // convert data to float and byte swap into big endian order
          for (int k=0; k<nx3; ++k) {
            for (int j=0; j<nx2; ++j) {
              for (int i=0; i<nx1; ++i) {
                int indx = i + j*indcs.nx1 + k*indcs.nx1*indcs.nx2;
                data[indx] = static_cast<float>(outarray(n,m,k,j,i));
              }
            }
          }
          if (!big_end) {
            for (int i=0; i<(nx1*nx2*nx3); ++i) {
              Swap4Bytes(&data[i]);
            }
          }
          // create new datatype representing this block in grid of MBs, and set file view
          int strt[3] = {kmb*nx3, jmb*nx2, imb*nx1};   // starting indices of this block
          MPI_Type_create_subarray(3,gridsize,mbsize,strt,MPI_ORDER_C,MPI_FLOAT,&mygrid);
          MPI_Type_commit(&mygrid);
          MPI_File_set_view(fh, header_size, MPI_FLOAT, mygrid, "native", MPI_INFO_NULL);
        } else {
          // if no data to be written, set file view to default
          // file view function is a collective operation, so must be called by all ranks
          MPI_File_set_view(fh, header_size, MPI_FLOAT, grid, "native", MPI_INFO_NULL);
        }

        // every rank has a MB to write, so write collectively
        if (m < noutmbs_min) {
          MPI_File_write_all(fh, &(data[0]), 1, block, MPI_STATUS_IGNORE);
        // some ranks are finished writing, so use non-collective write
        } else if (m < nout_mbs) {
          MPI_File_write(fh, &(data[0]), 1, block, MPI_STATUS_IGNORE);
        }
      }  // end loop over MeshBlocks
      MPI_Type_free(&mygrid);

      // reset view to stream of bytes in preparation for adding next data header
      header_size += nout1*nout2*nout3*sizeof(float);
      MPI_File_set_view(fh, header_size, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    }  // end loop over variables

    // close the output file and clean up
    MPI_Type_free(&block);
    MPI_Type_free(&grid);
    MPI_File_close(&fh);
    delete[] data;
    parallel_write=true;
  }

#endif
  if (!(parallel_write)) {
    //----- WRITE SERIAL FILES: -----
    // For serial (non-mpi) runs, use standard Unix-I/O functions.
    // open file and write header
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"w")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
        exit(EXIT_FAILURE);
    }
    std::fprintf(pfile,"%s",msg.str().c_str());

    // allocate 1D vector of floats used to convert and output entire 3D data
    float *data = new float[nout1*nout2*nout3];
    // Loop over variables
    int nout_vars = outvars.size();
    for (int n=0; n<nout_vars; ++n) {
      // write data type (SCALARS or VECTORS) and name
      std::stringstream data_msg;
      data_msg << std::endl << "SCALARS " << outvars[n].label.c_str()
               << " float" << std::endl
               << "LOOKUP_TABLE default" << std::endl;
      std::fprintf(pfile,"%s",data_msg.str().c_str());

      // Loop over MeshBlocks, insert variable into 3D array
      for (int m=0; m<nout_mbs; ++m) {
        LogicalLocation lloc = pm->lloc_eachmb[outmbs[m].mb_gid];
        // calculate indices of this MeshBlock in 3D grid of MBs
        int imb = (out_params.slice1 || (out_params.gid >= 0))? 0 : lloc.lx1;
        int jmb = (out_params.slice2 || (out_params.gid >= 0))? 0 : lloc.lx2;
        int kmb = (out_params.slice3 || (out_params.gid >= 0))? 0 : lloc.lx3;

        int &ois = outmbs[m].ois;
        int &oie = outmbs[m].oie;
        int &ojs = outmbs[m].ojs;
        int &oje = outmbs[m].oje;
        int &oks = outmbs[m].oks;
        int &oke = outmbs[m].oke;
        for (int k=oks; k<=oke; ++k) {
          for (int j=ojs; j<=oje; ++j) {
            for (int i=ois; i<=oie; ++i) {
              int indx = imb*indcs.nx1 + (i-ois) +
                        (jmb*indcs.nx2 + (j-ojs))*nout1 +
                        (kmb*indcs.nx3 + (k-oks))*nout1*nout2;
              data[indx] = static_cast<float>(outarray(n,m,k-oks,j-ojs,i-ois));
            }
          }
        }
      }  // end loop over MeshBlocks

      // swap data for this variable into big endian order
      if (!big_end) {
        for (int i=0; i<(nout1*nout2*nout3); ++i) { Swap4Bytes(&data[i]); }
      }
      // now write the data as unformatted binary
      std::fwrite(&(data[0]), sizeof(float), nout1*nout2*nout3, pfile);
    }
    // close the output file and clean up
    std::fclose(pfile);
    delete[] data;
  }

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
