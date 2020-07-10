//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outputs.cpp
//  \brief implements functions for Athena++ outputs
//
// The number and types of outputs are all controlled by the number and values of
// parameters specified in <output[n]> blocks in the input file.  Each output block must
// be labelled by a unique integer "n".  Following the convention of the parser
// implemented in the ParameterInput class, a second output block with the same integer
// "n" of an earlier block will silently overwrite the values read by the first block. The
// numbering of the output blocks does not need to be consecutive, and blocks may appear
// in any order in the input file.  Moreover, unlike the C version of Athena, the total
// number of <output[n]> blocks does not need to be specified -- in Athena++ a new output
// type will be created for each and every <output[n]> block in the input file.
//
// Required parameters that must be specified in an <output[n]> block are:
//   - variable     = cons,prim,D,d,E,e,m,m1,m2,m3,v,v1=vx,v2=vy,v3=vz,p,
//                    bcc,bcc1,bcc2,bcc3,b,b1,b2,b3,phi,uov
//   - file_type    = rst,tab,vtk,hst,hdf5
//   - dt           = problem time between outputs
//
// EXAMPLE of an <output[n]> block for a VTK dump:
//   <output3>
//   file_type   = tab       # Tabular data dump
//   variable    = prim      # variables to be output
//   data_format = %12.5e    # Optional data format string
//   dt          = 0.01      # time increment between outputs
//   x2_slice    = 0.0       # slice in x2
//   x3_slice    = 0.0       # slice in x3
//
// Each <output[n]> block will result in a new node being created in a linked list of
// OutputType stored in the Outputs class.  During a simulation, outputs are made when
// the simulation time satisfies the criteria implemented in the XXXX
//
// To implement a new output type, write a new OutputType derived class, and construct
// an object of this class in the Outputs constructor at the location indicated by the
// comment text: 'NEW_OUTPUT_TYPES'.
//========================================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>    // strcmp
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// OutputType constructor

OutputType::OutputType(OutputParameters opar) : output_params(opar) {
}

//----------------------------------------------------------------------------------------
// Outputs constructor

Outputs::Outputs(std::unique_ptr<Mesh> &pm, std::unique_ptr<ParameterInput> &pin) {

  // loop over input block names.  Find those that start with "output", read parameters,
  // and add to linked list of OutputTypes.

  int num_hst=0, num_rst=0; // count # of hst and rst outputs (should only be one each)
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 6, "output") == 0) {
      OutputParameters opar;  // define temporary OutputParameters struct

      // extract integer number of output block.  Save name and number
      std::string outn = it->block_name.substr(6); // 6 because counting starts at 0!
      opar.block_number = atoi(outn.c_str());
      opar.block_name.assign(it->block_name);

      // set time of last output, time between outputs
      opar.last_time = pin->GetOrAddReal(opar.block_name,"last_time", 0.0);
      opar.dt = pin->GetReal(opar.block_name,"dt");

      if (opar.dt <= 0.0) continue;  // only add output if dt>0

      // set file number, basename, id, and format
      opar.file_number = pin->GetOrAddInteger(opar.block_name,"file_number",0);
      opar.file_basename = pin->GetString("job","problem_id");
      char did[10];
      std::snprintf(did, sizeof(did), "out%d", opar.block_number);  // default id="outN"
      opar.file_id = pin->GetOrAddString(opar.block_name,"id",did);
      opar.file_type = pin->GetString(opar.block_name,"file_type");

      // read slicing options.  Check that slice is within mesh
      if (pin->DoesParameterExist(opar.block_name,"x1_slice")) {
        Real x1 = pin->GetReal(opar.block_name,"x1_slice");
        if (x1 >= pm->mesh_size.x1min && x1 < pm->mesh_size.x1max) {
          opar.x1_slice = x1;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x1=" << x1 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      // set islice negative if no slice in x1
      } else {
        opar.islice = -1;
      }

      if (pin->DoesParameterExist(opar.block_name,"x2_slice")) {
        Real x2 = pin->GetReal(opar.block_name,"x2_slice");
        if (x2 >= pm->mesh_size.x2min && x2 < pm->mesh_size.x2max) {
          opar.x2_slice = x2;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x2=" << x2 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      // set jslice negative if no slice in x2
      } else {
        opar.jslice = -1;
      }

      if (pin->DoesParameterExist(opar.block_name,"x3_slice")) {
        Real x3 = pin->GetReal(opar.block_name,"x3_slice");
        if (x3 >= pm->mesh_size.x3min && x3 < pm->mesh_size.x3max) {
          opar.x3_slice = x3;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x3=" << x3 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      // set kslice negative if no slice in x3
      } else {
        opar.kslice = -1;
      }

      // read ghost cell option
      opar.include_gzs = pin->GetOrAddBoolean(opar.block_name, "ghost_zones", false);

      // set output variable and optional data format string used in formatted writes
      if (opar.file_type.compare("hst") != 0 && opar.file_type.compare("rst") != 0) {
        opar.variable = pin->GetString(opar.block_name, "variable");
      }
      opar.data_format = pin->GetOrAddString(opar.block_name, "data_format", "%12.5e");
      opar.data_format.insert(0, " "); // prepend with blank to separate columns

      // Construct new OutputType according to file format
      // NEW_OUTPUT_TYPES: Add block to construct new types here
      if (opar.file_type.compare("tab") == 0) {
        FormattedTableOutput new_out(opar);
        output_list_.push_front(new_out);
//      } else if (opar.file_type.compare("hst") == 0) {
//        pnew_type = new HistoryOutput(opar);
//        num_hst++;
//      } else if (opar.file_type.compare("vtk") == 0) {
//        pnew_type = new VTKOutput(opar);
//      } else if (op.file_type.compare("rst") == 0) {
  // Move restarts to the tail end of the OutputType list, so file counters for other
  // output types are up-to-date in restart file
//        pnew_type = new RestartOutput(op);
//        num_rst++;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Unrecognized file format = '" << opar.file_type
            << "' in output block '" << opar.block_name << "'" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  // check there were no more than one history or restart files requested
  if (num_hst > 1 || num_rst > 1) {
   std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "More than one history or restart output block detected in input file"
        << std::endl;
   exit(EXIT_FAILURE);
  }

}

//----------------------------------------------------------------------------------------
// destructor - iterates through singly linked list of OutputTypes and deletes nodes

Outputs::~Outputs() {
}
