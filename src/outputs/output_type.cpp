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
// OutputType base class constructor
// Sets parameters like size and indices of output arrays

OutputType::OutputType(OutputParameters opar, std::unique_ptr<Mesh> &pm) :
   output_params(opar) {

    // set size of output arrays, adjusted accordingly if ghost zones included 
    auto it = pm->mblocks.begin();
    if (output_params.include_gzs) {
      nout1 = it->mblock_cells.nx1 + 2*it->mblock_cells.nghost;
      nout2 = it->mblock_cells.nx2;
      nout3 = it->mblock_cells.nx3;
      if (nout2 > 1) nout2 += 2*it->mblock_cells.nghost;
      if (nout3 > 1) nout3 += 2*it->mblock_cells.nghost;
    } else {
      nout1 = it->mblock_cells.nx1;
      nout2 = it->mblock_cells.nx2;
      nout3 = it->mblock_cells.nx3;
    }

    // set starting indices of output arrays
    ois = it->mblock_cells.is;
    ojs = it->mblock_cells.js;
    oks = it->mblock_cells.ks;
    if (output_params.include_gzs) {
      if (nout1 > 1) ois = 0;
      if (nout2 > 1) ojs = 0;
      if (nout3 > 1) oks = 0;
    }

    // reset array dimensions and indices if data is being sliced
    if (output_params.slice1) { nout1 = 1; }
    if (output_params.slice2) { nout2 = 1; }
    if (output_params.slice3) { nout3 = 1; }

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
    auto pmb = pm->mblocks.begin();
    int islice=0, jslice=0, kslice=0;
    if (output_params.slice1) {
      islice = pm->CellCenterIndex(output_params.slice_x1, pmb->mblock_cells.nx1,
        pmb->mblock_size.x1min, pmb->mblock_size.x1max);
    }
    if (output_params.slice2) {
      jslice = pm->CellCenterIndex(output_params.slice_x2, pmb->mblock_cells.nx2,
        pmb->mblock_size.x2min, pmb->mblock_size.x2max);
    }
    if (output_params.slice3) {
      kslice = pm->CellCenterIndex(output_params.slice_x3, pmb->mblock_cells.nx3,
        pmb->mblock_size.x3min, pmb->mblock_size.x3max);
    }

    // note the complicated addressing of array indices.  The output array does not
    // include ghost zones (unless needed), so it is always addressed starting at 0.
    // When the array is sliced, only the value at (ijk)slice is stored.
    for (int n=0; n<pm->nmbthisrank; ++n) {
      for (int k=0; k<nout3; ++k) {
      for (int j=0; j<nout2; ++j) {
      for (int i=0; i<nout1; ++i) {
        node.cc_data(n,k,j,i) =
           pmb->phydro->u0(hydro::IDN,(k+oks+kslice),(j+ojs+jslice),(i+ois+islice));
      }}}
    }
    data_list_.push_back(node);
  }
}
