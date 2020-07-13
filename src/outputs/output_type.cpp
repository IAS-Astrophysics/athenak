//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file output_type.cpp
//  \brief implements functions in OutputType class
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
#include "hydro/hydro.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// OutputType constructor

OutputType::OutputType(OutputParameters opar, std::unique_ptr<Mesh> &pm) :
   output_params(opar) {

    // figure out slicing


    // set size of output arrays, adjusted accordingly for slicing and ghost zones 
    auto it = pm->mblocks.begin();
    if (output_params.include_gzs) {
      nout1 = it->indx.ncells1;
      nout2 = it->indx.ncells2;
      nout3 = it->indx.ncells3;
    } else {
      nout1 = it->indx.nx1;
      nout2 = it->indx.nx2;
      nout3 = it->indx.nx3;
    }

    // set starting/ending indices of output arrays
    ois = it->indx.is; oie = it->indx.ie;
    ojs = it->indx.js; oje = it->indx.je;
    oks = it->indx.ks; oke = it->indx.ke;
    if (output_params.include_gzs) {
      if (nout1 > 1) ois -= it->indx.nghost; oie += it->indx.nghost;
      if (nout2 > 1) ojs -= it->indx.nghost; oje += it->indx.nghost;
      if (nout3 > 1) oks -= it->indx.nghost; oke += it->indx.nghost;
    }

}

//----------------------------------------------------------------------------------------
// OutputType::LoadOutputData()
// create linked list of OutputData objects containing data specified in <output> block
// for this output type

void OutputType::LoadOutputData(std::unique_ptr<Mesh> &pm) {

  data_list_.clear();  // start with a clean list

  // mass density
  if (output_params.variable.compare("d") == 0 ||
      output_params.variable.compare("density") == 0 ||
      output_params.variable.compare("prim") == 0 ||
      output_params.variable.compare("cons") == 0) {
    OutputData node;
    node.type = "SCALARS";
    node.name = "dens";

    node.cc_data.SetSize(pm->nmbthisrank, nout3, nout2, nout1);
    // deep copy one array for each MeshBlock on this rank
    for (int n=0; n<pm->nmbthisrank; ++n) {
      for (int k=oks; k<=oke; ++k) {
      for (int j=ojs; j<=oje; ++j) {
      for (int i=ois; i<=oie; ++i) {
        node.cc_data(n,k,j,i)  = pm->mblocks.begin()->phydro->u(hydro::IDN,k,j,i);
      }}}
    }
    data_list_.push_back(node);
  }
}
