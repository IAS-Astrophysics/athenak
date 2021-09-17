//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outputs.cpp
//  \brief implements Outputs class constructor
//
// The number and types of outputs are all controlled by the number and values of
// parameters specified in <output[n]> blocks in the input file.  Each output block must
// be labelled by a unique integer "n".  Following the convention of the parser
// implemented in the ParameterInput class, a second output block with the same integer
// "n" of an earlier block will silently overwrite the values read by the first block. The
// numbering of the output blocks does not need to be consecutive, and blocks may appear
// in any order in the input file. A new output type will be created for each and every
// <output[n]> block in the input file.
//
// Required parameters that must be specified in an <output[n]> block are:
//   - variable     = cons,prim,D,d,E,e,m,m1,m2,m3,v,v1=vx,v2=vy,v3=vz,p,
//                    bcc,bcc1,bcc2,bcc3,b,b1,b2,b3,phi,uov
//   - file_type    = rst,tab,vtk,hst,hdf5
//   - dt           = problem time between outputs
//
// EXAMPLE of an <output[n]> block for a TAB dump:
//   <output3>
//   file_type   = tab       # Tabular data dump
//   variable    = prim      # variables to be output
//   data_format = %12.5e    # Optional data format string
//   dt          = 0.01      # time increment between outputs
//   slice_x2    = 0.0       # slice at x2 
//   slice_x3    = 0.0       # slice at x3
//
// Each <output[n]> block will result in a new node being created in a linked list of
// OutputType stored in the Outputs class.  During a simulation, outputs are made when
// the simulation time satisfies the criteria implemented in the Driver class.
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
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// Outputs constructor

Outputs::Outputs(ParameterInput *pin, Mesh *pm) {

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
      // when last_time < 0, then outputs will always be made
      opar.last_time = pin->GetOrAddReal(opar.block_name,"last_time", -1.0);
      opar.dt = pin->GetReal(opar.block_name,"dt");

      if (opar.dt <= 0.0) continue;  // only add output if dt>0

      // set file number, basename, and format
      opar.file_number = pin->GetOrAddInteger(opar.block_name,"file_number",0);
      opar.file_basename = pin->GetString("job","basename");
      opar.file_type = pin->GetString(opar.block_name,"file_type");

      // read slicing options.  Check that slice is within mesh
      if (pin->DoesParameterExist(opar.block_name,"slice_x1")) {
        Real x1 = pin->GetReal(opar.block_name,"slice_x1");
        if (x1 >= pm->mesh_size.x1min && x1 < pm->mesh_size.x1max) {
          opar.slice_x1 = x1;
          opar.slice1 = true;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x1=" << x1 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      } else {
        opar.slice1 = false;
      }

      if (pin->DoesParameterExist(opar.block_name,"slice_x2")) {
        Real x2 = pin->GetReal(opar.block_name,"slice_x2");
        if (x2 >= pm->mesh_size.x2min && x2 < pm->mesh_size.x2max) {
          opar.slice_x2 = x2;
          opar.slice2 = true;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x2=" << x2 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      } else {
        opar.slice2 = false;
      }

      if (pin->DoesParameterExist(opar.block_name,"slice_x3")) {
        Real x3 = pin->GetReal(opar.block_name,"slice_x3");
        if (x3 >= pm->mesh_size.x3min && x3 < pm->mesh_size.x3max) {
          opar.slice_x3 = x3;
          opar.slice3 = true;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x3=" << x3 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      } else {
        opar.slice3 = false;
      }

      // read ghost cell option
      opar.include_gzs = pin->GetOrAddBoolean(opar.block_name, "ghost_zones", false);

      // set output variable and optional file id (default is output variable name)
      if (opar.file_type.compare("hst") != 0 && opar.file_type.compare("rst") != 0) {
        opar.variable = GetOutputVariable(pin->GetString(opar.block_name, "variable"));
        opar.file_id = pin->GetOrAddString(
                       opar.block_name,"id",pin->GetString(opar.block_name, "variable"));
      }

      // set optional data format string used in formatted writes
      opar.data_format = pin->GetOrAddString(opar.block_name, "data_format", "%12.5e");
      opar.data_format.insert(0, " "); // prepend with blank to separate columns

      // Construct new OutputType according to file format
      // NEW_OUTPUT_TYPES: Add block to construct new types here
      OutputType *pnode;
      if (opar.file_type.compare("tab") == 0) {
        pnode = new FormattedTableOutput(opar,pm);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("hst") == 0) {
        pnode = new HistoryOutput(opar,pm);
        pout_list.insert(pout_list.begin(),pnode);
        num_hst++;
      } else if (opar.file_type.compare("vtk") == 0) {
        pnode = new VTKOutput(opar,pm);
        pout_list.insert(pout_list.begin(),pnode);
//      } else if (op.file_type.compare("rst") == 0) {
//    // Move restarts to the tail end of the OutputType list, so file counters for other
//    // output types are up-to-date in restart file
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

//----------------------------------------------------------------------------------------
//! \fn GetOutputVariable(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying output var.
//  Generally output utility function, not part of any outputs class

OutputVariable GetOutputVariable(const std::string& input_string)
{ 
  // hydro conserved variables
  if (input_string == "hydro_u_d") {
    return OutputVariable::hydro_u_d;
  } else if (input_string == "hydro_u_m1") {
    return OutputVariable::hydro_u_m1;
  } else if (input_string == "hydro_u_m2") {
    return OutputVariable::hydro_u_m2;
  } else if (input_string == "hydro_u_m3") {
    return OutputVariable::hydro_u_m3;
  } else if (input_string == "hydro_u_e") {
    return OutputVariable::hydro_u_e;
  } else if (input_string == "hydro_u") {
    return OutputVariable::hydro_u;

  // hydro primitive variables
  } else if (input_string == "hydro_w_d") {
    return OutputVariable::hydro_w_d;
  } else if (input_string == "hydro_w_vx") {
    return OutputVariable::hydro_w_vx;
  } else if (input_string == "hydro_w_vy") {
    return OutputVariable::hydro_w_vy;
  } else if (input_string == "hydro_w_vz") {
    return OutputVariable::hydro_w_vz;
  } else if (input_string == "hydro_w_p") {
    return OutputVariable::hydro_w_p;
  } else if (input_string == "hydro_w") {
    return OutputVariable::hydro_w;

  // mhd conserved variables
  } else if (input_string == "mhd_u_d") {
    return OutputVariable::mhd_u_d;
  } else if (input_string == "mhd_u_m1") {
    return OutputVariable::mhd_u_m1;
  } else if (input_string == "mhd_u_m2") {
    return OutputVariable::mhd_u_m2;
  } else if (input_string == "mhd_u_m3") {
    return OutputVariable::mhd_u_m3;
  } else if (input_string == "mhd_u_e") {
    return OutputVariable::mhd_u_e;
  } else if (input_string == "mhd_u") {
    return OutputVariable::mhd_u;

  // mhd primitive variables
  } else if (input_string == "mhd_w_d") {
    return OutputVariable::mhd_w_d;
  } else if (input_string == "mhd_w_vx") {
    return OutputVariable::mhd_w_vx;
  } else if (input_string == "mhd_w_vy") {
    return OutputVariable::mhd_w_vy;
  } else if (input_string == "mhd_w_vz") {
    return OutputVariable::mhd_w_vz;
  } else if (input_string == "mhd_w_p") {
    return OutputVariable::mhd_w_p;
  } else if (input_string == "mhd_w") {
    return OutputVariable::mhd_w;

  // cell-centered magnetic fields in mhd
  } else if (input_string == "mhd_bcc1") {
    return OutputVariable::mhd_bcc1;
  } else if (input_string == "mhd_bcc2") {
    return OutputVariable::mhd_bcc2;
  } else if (input_string == "mhd_bcc3") {
    return OutputVariable::mhd_bcc3;
  } else if (input_string == "mhd_bcc") {
    return OutputVariable::mhd_bcc;

  // mhd conserved variables and cell-centered magnetic fields
  } else if (input_string == "mhd_u_bcc") {
    return OutputVariable::mhd_u_bcc;

  // mhd primitive variables and cell-centered magnetic fields
  } else if (input_string == "mhd_w_bcc") {
    return OutputVariable::mhd_w_bcc;

  // cell-centered forcing added in turbulent driving problems
  } else if (input_string == "turb_force") {
    return OutputVariable::turb_force;

  // invalid variable requested
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Input string = '" << input_string << "' is not a valid output variable"
       << std::endl << "See the enum class OutputVariable defined in "
       << "src/outputs/outputs.hpp for a complete list" << std::endl
       << "of valid output variables" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn GetOutputVariableString(BoundaryFlag input_flag)
//  \brief Parses enumerated type OutputVariable internal integer representation to return
//  string describing the output variable. Typically used to format descriptive errors
//  or diagnostics. Inverse of GetOutputVariable().
//
//  Generally output utility function, not part of any outputs class

std::string GetOutputVariableString(OutputVariable input_flag)
{
  switch (input_flag) {

    // hydro conserved variables
    case OutputVariable::hydro_u_d:  // 0
      return "hydro_u_d";
    case OutputVariable::hydro_u_m1:
      return "hydro_u_m1";
    case OutputVariable::hydro_u_m2:
      return "hydro_u_m2";
    case OutputVariable::hydro_u_m3:
      return "hydro_u_m3";
    case OutputVariable::hydro_u_e:
      return "hydro_u_e";
    case OutputVariable::hydro_u:
      return "hydro_u";

    // hydro primitive variables
    case OutputVariable::hydro_w_d:
      return "hydro_w_d";
    case OutputVariable::hydro_w_vx:
      return "hydro_w_vx";
    case OutputVariable::hydro_w_vy:
      return "hydro_w_vy";
    case OutputVariable::hydro_w_vz:
      return "hydro_w_vz";
    case OutputVariable::hydro_w_p:
      return "hydro_w_p";
    case OutputVariable::hydro_w:
      return "hydro_w";

    // mhd conserved variables
    case OutputVariable::mhd_u_d:
      return "mhd_u_d";
    case OutputVariable::mhd_u_m1:
      return "mhd_u_m1";
    case OutputVariable::mhd_u_m2:
      return "mhd_u_m2";
    case OutputVariable::mhd_u_m3:
      return "mhd_u_m3";
    case OutputVariable::mhd_u_e:
      return "mhd_u_e";
    case OutputVariable::mhd_u:
      return "mhd_u";

    // mhd primitive variables
    case OutputVariable::mhd_w_d:
      return "mhd_w_d";
    case OutputVariable::mhd_w_vx:
      return "mhd_w_vx";
    case OutputVariable::mhd_w_vy:
      return "mhd_w_vy";
    case OutputVariable::mhd_w_vz:
      return "mhd_w_vz";
    case OutputVariable::mhd_w_p:
      return "mhd_w_p";
    case OutputVariable::mhd_w:
      return "mhd_w";

    // cell-centered magnetic fields in mhd
    case OutputVariable::mhd_bcc1:
      return "mhd_bcc1";
    case OutputVariable::mhd_bcc2:
      return "mhd_bcc2";
    case OutputVariable::mhd_bcc3:
      return "mhd_bcc3";
    case OutputVariable::mhd_bcc:
      return "mhd_bcc";

    // mhd conserved variables and cell-centered magnetic fields
    case OutputVariable::mhd_u_bcc:
      return "mhd_u_bcc";

    // mhd primitive variables and cell-centered magnetic fields
    case OutputVariable::mhd_w_bcc:
      return "mhd_w_bcc";

    // cell-centered forcing added in turbulent driving problems
    case OutputVariable::turb_force:
      return "turb_force";

    // undefined or unknown variable names
    case OutputVariable::undef:
      return "undef";
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Input enum OutputVariable=" << static_cast<int>(input_flag)
         << " is an invalid output variable" << std::endl;
      std::exit(EXIT_FAILURE);
      break;
  }
}
