//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file formatted_table.cpp
//  \brief writes output data as a formatted table.  Should not be used to output large
//  3D data sets as this format is very slow and memory intensive.  Most useful for 1D
//  slices and/or sums.  Writes one file per Meshblock.

#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: calls OutputType base class constructor

FormattedTableOutput::FormattedTableOutput(OutputParameters op, std::unique_ptr<Mesh> &pm)
  : OutputType(op, pm) {
}

//----------------------------------------------------------------------------------------
//! \fn void FormattedTableOutput:::WriteOutputFile(Mesh *pm)
//  \brief writes OutputData to file in tabular format using C style std::fprintf
//         Writes one file per MeshBlock

void FormattedTableOutput::WriteOutputFile(std::unique_ptr<Mesh> &pm) {

  // create filename: "file_basename"+ "."+"blockid"+"."+"file_id"+"."+XXXXX+".tab",
  // where XXXXX = 5-digit file_number
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", output_params.file_number);

  fname.assign(output_params.file_basename);
  fname.append(".");
  fname.append(output_params.file_id);
  fname.append(".");
  fname.append(number);
  fname.append(".tab");

  // open file for output
  FILE *pfile;
  if ((pfile = std::fopen(fname.c_str(),"w")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << std::endl << "Output file '" << fname << "' could not be opened" << std::endl;
    exit(EXIT_FAILURE);
  }

  // print file header
//  std::fprintf(pfile, "# Athena++ data at time=%e", pm->time);
//  std::fprintf(pfile, "  cycle=%d", pmb->pmy_mesh->ncycle);
  std::fprintf(pfile, "# Athena++ data at time=");
  std::fprintf(pfile, "  cycle=");
  std::fprintf(pfile, "  variables=%s \n", output_params.variable.c_str());

  // write x1, x2, x3 column headers
  std::fprintf(pfile, "#");
  if (ois != oie) std::fprintf(pfile, " i       x1v     ");
  if (ojs != oje) std::fprintf(pfile, " j       x2v     ");
  if (oks != oke) std::fprintf(pfile, " k       x3v     ");
  // write data col headers from "name" stored in OutputData nodet
  for (auto it : data_list_) {
    if (it.type == "VECTORS") {
      for (int index = 1; index <= 3; ++index) {
        std::fprintf(pfile, "    %s%d     ", it.name.c_str(), index);
      }
    } else {
      std::fprintf(pfile, "    %s      ", it.name.c_str());
    }
  }
  std::fprintf(pfile, "\n"); // terminate line

    // loop over all cells in data arrays
  for (int n=0; n<pm->nmbthisrank; ++n) {
    for (int k=oks; k<=oke; ++k) {
      for (int j=ojs; j<=oje; ++j) {
        for (int i=ois; i<=oie; ++i) {
          // write x1, x2, x3 indices and coordinates on start of new line
          if (ois != oie) {
            std::fprintf(pfile, "%04d", i);
            std::fprintf(pfile, output_params.data_format.c_str(), x1posn);
          }
          if (ojs != oje) {
            std::fprintf(pfile, " %04d", j);  // note extra space for formatting
            std::fprintf(pfile, output_params.data_format.c_str(), x2posn);
          }
          if (oks != oke) {
            std::fprintf(pfile, " %04d", k);  // note extra space for formatting
            std::fprintf(pfile, output_params.data_format.c_str(), x3posn);
          }

          // step through doubly linked list of OutputData's and write each on same line
          for (auto it : data_list_) {
            std::fprintf(pfile, output_params.data_format.c_str(),
                         it.cc_data(n,(k-oks),(j-ojs),(i-ois)));
          }
          std::fprintf(pfile,"\n"); // terminate line
        }
      }
    }
  }  // end loop over MeshBlocks
    std::fclose(pfile);   // don't forget to close the output file

  // increment counters
  output_params.file_number++;
  output_params.last_time += output_params.dt;
  // store filenumber and time into ParameterInput for restarts
//  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
//  pin->SetReal(output_params.block_name, "last_time", output_params.last_time);

  return;
}
