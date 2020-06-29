//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in MeshBlock class

#include <cstdlib>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "meshblock.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, hydro, field
//                        and mesh refinement objects.

MeshBlock::MeshBlock(Mesh *pm, ParameterInput *pin, RegionSize input_block) :
    pmy_mesh(pm), block_size(input_block) {

  // initialize grid indices
  indx.is = block_size.nghost;
  indx.ie = indx.is + block_size.nx1 - 1;
  indx.nghost = block_size.nghost;
  indx.nx1 = block_size.nx1;
  indx.ncells1 = block_size.nx1 + 2*block_size.nghost;

  if (block_size.nx2 > 1) {
    indx.js = block_size.nghost;
    indx.je = indx.js + block_size.nx2 - 1;
    indx.nx2 = block_size.nx2;
    indx.ncells2 = block_size.nx2 + 2*block_size.nghost;
  } else {
    indx.js = 0;
    indx.je = 0;
    indx.nx1 = 1;
    indx.ncells2 = 1;
  }

  if (block_size.nx3 > 1) {
    indx.ks = block_size.nghost;
    indx.ke = indx.ks + block_size.nx3 - 1;
    indx.nx3 = block_size.nx3;
    indx.ncells3 = block_size.nx3 + 2*block_size.nghost;
  } else {
    indx.ks = 0;
    indx.ke = 0;
    indx.nx3 = 1;
    indx.ncells3 = 1;
  }

  // initialize coarse grid indices
  if (pm->multilevel) {
    cindx.nghost = (block_size.nghost + 1)/2 + 1;
    cindx.is = cindx.nghost;
    cindx.ie = cindx.is + block_size.nx1/2 - 1;
    cindx.nx1 = cindx.ie - cindx.is + 1;
    cindx.ncells1 = cindx.nx1 + 2*cindx.nghost;

    if (block_size.nx2 > 1) {
      cindx.js = cindx.nghost;
      cindx.je = cindx.js + block_size.nx2/2 - 1;
      cindx.nx2 = cindx.je - cindx.js + 1;
      cindx.ncells2 = cindx.nx2 + 2*cindx.nghost;
    } else {
      cindx.js = 0;
      cindx.je = 0;
      cindx.nx1 = 1;
      cindx.ncells2 = 1;
    }
  
    if (block_size.nx3 > 1) {
      cindx.ks = cindx.nghost;
      cindx.ke = cindx.ks + block_size.nx3/2 - 1;
      cindx.nx3 = cindx.ke - cindx.ks + 1;
      cindx.ncells3 = cindx.nx3 + 2*cindx.nghost;
    } else {
      cindx.ks = 0;
      cindx.ke = 0;
      cindx.nx3 = 1;
      cindx.ncells3 = 1;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts


//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock() {
}
