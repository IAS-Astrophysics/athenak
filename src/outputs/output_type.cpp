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
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "utils/grid_locations.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// OutputType base class constructor
// Sets parameters like size and indices of output arrays

OutputType::OutputType(OutputParameters opar, Mesh *pm) :
   out_params(opar)
{
  // exit for history files
  if (out_params.file_type.compare("hst") == 0) {return;}

  // set size & starting indices of output arrays, adjusted accordingly if gz included 
  // Since all MeshBlocks the same, only need to compute values from first MB
  auto it = pm->mblocks.begin();
  if (out_params.include_gzs) {
    int nout1 = it->mb_cells.nx1 + 2*(it->mb_cells.ng);
    int nout2 = (it->mb_cells.nx2 > 1)? (it->mb_cells.nx2 + 2*(it->mb_cells.ng)) : 1;
    int nout3 = (it->mb_cells.nx3 > 1)? (it->mb_cells.nx3 + 2*(it->mb_cells.ng)) : 1;
    ois = 0; oie = nout1-1;
    ojs = 0; oje = nout2-1;
    oks = 0; oke = nout3-1;
  } else {
    ois = it->mb_cells.is; oie = it->mb_cells.ie;
    ojs = it->mb_cells.js; oje = it->mb_cells.je;
    oks = it->mb_cells.ks; oke = it->mb_cells.ke;
  }

  // parse list of variables for each physics and flag variables to be output
  nvar = 1;
  out_data_label_.push_back("dens");

  // check for valid output variable in <input> block
  if (false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output variable '" << out_params.variable << "' not implemented" << std::endl
       << "Allowed hydro variables: cons,D,E,mom,M1,M2,M3,prim,d,p,vel,vx,vy,vz"
       << std::endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// OutputType::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void OutputType::LoadOutputData(Mesh *pm)
{
  out_data_.clear();  // start with a clean list

  // out_data_ vector (indexed over # of output MBs) stores 4D array of variables
  // so start iteration over number of MeshBlocks

  // TODO: get this working for multiple physics, which may be either defined/undef

  // loop over all MeshBlocks
  for (auto &mb : pm->mblocks) {

    // check for slicing in each dimension
    if (out_params.slice1) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x1 <  mb.mb_size.x1min ||
          out_params.slice_x1 >= mb.mb_size.x1max) { continue; }
      // set index of slice
      Real &xmin = mb.mb_size.x1min;
      Real &xmax = mb.mb_size.x1max;
      ois = CellCenterIndex(out_params.slice_x1, mb.mb_cells.nx1, xmin, xmax);
      oie = ois;
    }

    if (out_params.slice2) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x2 <  mb.mb_size.x2min ||
          out_params.slice_x2 >= mb.mb_size.x2max) { continue; }
      // set index of slice
      Real &xmin = mb.mb_size.x2min;
      Real &xmax = mb.mb_size.x2max;
      ojs = CellCenterIndex(out_params.slice_x2, mb.mb_cells.nx2, xmin, xmax);
      oje = ojs;
    }

    if (out_params.slice3) {
      // skip if slice is out of range of this MB
      if (out_params.slice_x3 <  mb.mb_size.x3min ||
          out_params.slice_x3 >= mb.mb_size.x3max) { continue; }
      // set index of slice
      Real &xmin = mb.mb_size.x3min;
      Real &xmax = mb.mb_size.x3max;
      oks = CellCenterIndex(out_params.slice_x3, mb.mb_cells.nx3, xmin, xmax);
      oke = oks;
    }

    // load all the outpustr variables on this MeshBlock
    HostArray4D<Real> new_data("out",nvar,(oke-oks+1),(oje-ojs+1),(oie-ois+1));

    for (int n=0; n<nvar; ++n) {
/**** hardwired for IM2 to start ****/
      auto dev_data_slice = Kokkos::subview(mb.phydro->u0, 2, std::make_pair(oks,oke+1),
                            std::make_pair(ojs,oje+1), std::make_pair(ois,oie+1));
      auto hst_data_slice = Kokkos::create_mirror_view(dev_data_slice);
      auto new_data_slice = Kokkos::subview(new_data, n, Kokkos::ALL(), Kokkos::ALL(),
                      Kokkos::ALL());
      Kokkos::deep_copy(new_data_slice,hst_data_slice);
/****/
//      for (int k=oks; k<=oke; ++k) {
//      for (int j=ojs; j<=oje; ++j) {
//      for (int i=ois; i<=oie; ++i) {
//        new_data(n,k-oks,j-ojs,i-ois) = hst_data_slice(k-oks,j-ojs,i-ois);
//      }}}
//for (int i=ois; i<oie; ++i) {
//std::cout << new_data(0,0,0,i-ois) << std::endl;
//}
/****/
      
    }

    // append variables on this MeshBlock to end of out_data_ vector
    out_data_.push_back(new_data);
    out_data_gid_.push_back(mb.mb_gid);
  }
}
