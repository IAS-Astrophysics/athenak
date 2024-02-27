//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file formatted_table.cpp
//  \brief writes output data as a formatted (ASCI) table.  Since outputing data in this
//  format is very slow and creates large files, it cannot be used for anything other than
//  1D slices.  Code will issue error if this format is selected for 2D or 3D outputs.
//  Output is written to a single file even with multiple MeshBlocks and MPI ranks.

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

FormattedTableOutput::FormattedTableOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // check that 1D slice specified, otherwise issue warning and quit
  if (pm->multi_d) {
    if (!(out_params.slice1) && !(out_params.slice2)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Formatted table outputs can only contain 1D slices"
                << std::endl << "Please add additional slice planes" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  if (pm->three_d) {
    if ((!(out_params.slice2) && !(out_params.slice3)) ||
        (!(out_params.slice1) && !(out_params.slice3))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Formatted table outputs can only contain 1D slices"
                << std::endl << "Please add additional slice planes" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // create directories for outputs. Comments in binary.cpp constructor explain why
  mkdir("tab",0775);
}

//----------------------------------------------------------------------------------------
//! \fn void FormattedTableOutput:::WriteOutputFile(Mesh *pm)
//  \brief writes output_data_ to file in tabular format using C style std::fprintf

void FormattedTableOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // create filename: "tab/file_basename" + "." + "file_id" + "." + XXXXX + ".tab"
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("tab/");
  fname.append(out_params.file_basename);
  fname.append(".");
  fname.append(out_params.file_id);
  fname.append(".");
  fname.append(number);
  fname.append(".tab");

  // master rank creates file and writes header (even though it may not have any actual
  // data to write below)
  if (global_variable::my_rank == 0) {
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"w")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Output file '" << fname << "' could not be opened" << std::endl;
      exit(EXIT_FAILURE);
    }

    // print file header
    std::fprintf(pfile, "# Athena++ data at time=%e", pm->time);
    std::fprintf(pfile, "  cycle=%d \n", pm->ncycle);

    // write one of x1, x2, x3 column headers
    std::fprintf(pfile, "# gid  ");
    if (!(out_params.slice1)) std::fprintf(pfile, " i       x1v     ");
    if (!(out_params.slice2)) std::fprintf(pfile, " j       x2v     ");
    if (!(out_params.slice3)) std::fprintf(pfile, " k       x3v     ");

    // write data col headers from outvars vector
    for (auto it : outvars) {
      std::fprintf(pfile, "    %s     ", it.label.c_str());
    }
    std::fprintf(pfile, "\n"); // terminate line
    std::fclose(pfile);   // don't forget to close the output file
  }
#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif

  // now all ranks open file and append data
  FILE *pfile;
  if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Output file '" << fname << "' could not be opened" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int r=0; r<global_variable::nranks; ++r) {
    // MPI ranks append data one-at-a-time in order, due to MPI_Barrier at end of loop
    // This could be slow for very large numbers of ranks, however this is not a regime
    // where .tab files are expected to be used very much.
    if (r == global_variable::my_rank) {
      // loop over output MeshBlocks, output all data
      int nout_vars = outvars.size();
      int nout_mbs = (outmbs.size());
      for (int m=0; m<nout_mbs; ++m) {
        auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
        auto &size  = pm->pmb_pack->pmb->mb_size;
        MeshBlock* pmb = pm->pmb_pack->pmb;
        int idx = pm->FindMeshBlockIndex(outmbs[m].mb_gid);
        int &is = indcs.is;
        int &js = indcs.js;
        int &ks = indcs.ks;
        int &ois = outmbs[m].ois;
        int &oie = outmbs[m].oie;
        int &ojs = outmbs[m].ojs;
        int &oje = outmbs[m].oje;
        int &oks = outmbs[m].oks;
        int &oke = outmbs[m].oke;
        Real &x1min = size.h_view(idx).x1min;
        Real &x1max = size.h_view(idx).x1max;
        Real &x2min = size.h_view(idx).x2min;
        Real &x2max = size.h_view(idx).x2max;
        Real &x3min = size.h_view(idx).x3min;
        Real &x3max = size.h_view(idx).x3max;
        int &nx1 = indcs.nx1;
        int &nx2 = indcs.nx2;
        int &nx3 = indcs.nx3;
        for (int k=oks; k<=oke; ++k) {
          for (int j=ojs; j<=oje; ++j) {
            for (int i=ois; i<=oie; ++i) {
              std::fprintf(pfile, "%05d", pmb->mb_gid.h_view(idx));
              // write x1, x2, x3 indices and coordinates
              if (oie != ois) {
                std::fprintf(pfile, " %04d", i);  // note extra space for formatting
                Real x1cc = CellCenterX(i-is,nx1,x1min,x1max);
                std::fprintf(pfile, out_params.data_format.c_str(), x1cc);
              }
              if (oje != ojs) {
                std::fprintf(pfile, " %04d", j);  // note extra space for formatting
                Real x2cc = CellCenterX(j-js,nx2,x2min,x2max);
                std::fprintf(pfile, out_params.data_format.c_str(), x2cc);
              }
              if (oke != oks) {
                std::fprintf(pfile, " %04d", k);  // note extra space for formatting
                Real x3cc = CellCenterX(k-ks,nx3,x3min,x3max);
                std::fprintf(pfile, out_params.data_format.c_str(), x3cc);
              }

              // write each output variable on same line
              for (int n=0; n<nout_vars; ++n) {
                std::fprintf(pfile, out_params.data_format.c_str(),
                             outarray(n,m,k-oks,j-ojs,i-ois));
              }
              std::fprintf(pfile,"\n"); // terminate line
            }
          }
        }
      }  // end loop over MeshBlocks
    }
    std::fflush(pfile);
#if MPI_PARALLEL_ENABLED
    int ierr = MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  std::fclose(pfile);   // don't forget to close the output file

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  // store filenumber and time into ParameterInput for restarts
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  return;
}
