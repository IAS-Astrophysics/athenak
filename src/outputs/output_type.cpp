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

OutputType::OutputType(OutputParameters opar, Mesh *pm) :
   output_params(opar)
{
    // set size & starting indices of output arrays, adjusted accordingly if gz included 
    // Since all MeshBlocks the same, only need to compute values from first MB
    auto it = pm->mblocks.begin();
    if (output_params.include_gzs) {
      nout1 = it->mb_cells.nx1 + 2*(it->mb_cells.ng);
      nout2 = (it->mb_cells.nx2 > 1)? (it->mb_cells.nx2 + 2*(it->mb_cells.ng)) : 1;
      nout3 = (it->mb_cells.nx3 > 1)? (it->mb_cells.nx3 + 2*(it->mb_cells.ng)) : 1;
      ois = 0;
      ojs = 0;
      oks = 0;
    } else {
      nout1 = it->mb_cells.nx1;
      nout2 = it->mb_cells.nx2;
      nout3 = it->mb_cells.nx3;
      ois = it->mb_cells.is;
      ojs = it->mb_cells.js;
      oks = it->mb_cells.ks;
    }

    // reset array dimensions if data is being sliced
    if (output_params.slice1) { nout1 = 1; }
    if (output_params.slice2) { nout2 = 1; }
    if (output_params.slice3) { nout3 = 1; }

    // calculate spatial positions of output data
    output_x1posn_.SetSize(pm->nmbthisrank, nout1);
    output_x1posn_.SetLabel("x1v");
    output_x2posn_.SetSize(pm->nmbthisrank, nout2);
    output_x2posn_.SetLabel("x2v");
    output_x3posn_.SetSize(pm->nmbthisrank, nout3);
    output_x3posn_.SetLabel("x3v");

    // TODO get working with multuple meshblocks
    auto pmb = pm->mblocks.begin();
    for (int n=0; n<pm->nmbthisrank; ++n) {
      for (int i=0; i<nout1; ++i) {
        output_x1posn_(n,i) = pm->CellCenterX((i-(pmb->mb_cells.is - ois)),
           pmb->mb_cells.nx1, pmb->mb_size.x1min, pmb->mb_size.x1max);
      }
      for (int j=0; j<nout2; ++j) {
        output_x2posn_(n,j) = pm->CellCenterX((j-(pmb->mb_cells.js - ojs)),
           pmb->mb_cells.nx2, pmb->mb_size.x2min, pmb->mb_size.x2max);
      }
      for (int k=0; k<nout3; ++k) {
        output_x3posn_(n,k) = pm->CellCenterX((k-(pmb->mb_cells.ks - oks)),
           pmb->mb_cells.nx3, pmb->mb_size.x3min, pmb->mb_size.x3max);
      }
    }

}

//----------------------------------------------------------------------------------------
// OutputType::LoadOutputData()
// create std::vector of AthenaArrays containing data specified in <output> block for
// this output type

void OutputType::LoadOutputData(Mesh *pm)
{
  output_data_.clear();  // start with a clean list

  // mass density
  if (output_params.variable.compare("d") == 0 ||
      output_params.variable.compare("density") == 0 ||
      output_params.variable.compare("prim") == 0 ||
      output_params.variable.compare("cons") == 0) {
    AthenaArray<Real> new_data;

    new_data.SetSize(pm->nmbthisrank, nout3, nout2, nout1);
    new_data.SetLabel("dens");

    // deep copy one array for each MeshBlock on this rank
    // TODO add data from each MeshBlock
    // TODO add loop over all variables
    auto pmb = pm->mblocks.begin();
    int islice=0, jslice=0, kslice=0;
    if (output_params.slice1) {
      islice = pm->CellCenterIndex(output_params.slice_x1, pmb->mb_cells.nx1,
        pmb->mb_size.x1min, pmb->mb_size.x1max);
    }
    if (output_params.slice2) {
      jslice = pm->CellCenterIndex(output_params.slice_x2, pmb->mb_cells.nx2,
        pmb->mb_size.x2min, pmb->mb_size.x2max);
    }
    if (output_params.slice3) {
      kslice = pm->CellCenterIndex(output_params.slice_x3, pmb->mb_cells.nx3,
        pmb->mb_size.x3min, pmb->mb_size.x3max);
    }

    // note the complicated addressing of array indices.  The output array does not
    // include ghost zones (unless needed), so it is always addressed starting at 0.
    // When the array is sliced, only the value at (ijk)slice is stored.
    for (int n=0; n<pm->nmbthisrank; ++n) {
      for (int k=0; k<nout3; ++k) {
      for (int j=0; j<nout2; ++j) {
      for (int i=0; i<nout1; ++i) {
        new_data(n,k,j,i) =
           pmb->phydro->u0(hydro::IDN,(k+oks+kslice),(j+ojs+jslice),(i+ois+islice));
      }}}
    }
    output_data_.push_back(new_data);
  }
}
