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
#include "bvals/bvals.hpp"
#include "mesh.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, hydro, field
//                        and mesh refinement objects.

MeshBlock::MeshBlock(Mesh *pm, std::unique_ptr<ParameterInput> &pin,
                     RegionSize input_block, int igid, BoundaryFlag *input_bcs) :
    pmy_mesh(pm), mblock_size(input_block), mblock_gid(igid) {

  // copy input boundary flags into MeshBlock 
  for (int i=0; i<6; ++i) {mblock_bcs[i] = input_bcs[i];}

  // initialize grid indices
  indx.is = mblock_size.nghost;
  indx.ie = indx.is + mblock_size.nx1 - 1;
  indx.nghost = mblock_size.nghost;
  indx.nx1 = mblock_size.nx1;
  indx.ncells1 = mblock_size.nx1 + 2*mblock_size.nghost;

  if (mblock_size.nx2 > 1) {
    indx.js = mblock_size.nghost;
    indx.je = indx.js + mblock_size.nx2 - 1;
    indx.nx2 = mblock_size.nx2;
    indx.ncells2 = mblock_size.nx2 + 2*mblock_size.nghost;
  } else {
    indx.js = 0;
    indx.je = 0;
    indx.nx2 = 1;
    indx.ncells2 = 1;
  }

  if (mblock_size.nx3 > 1) {
    indx.ks = mblock_size.nghost;
    indx.ke = indx.ks + mblock_size.nx3 - 1;
    indx.nx3 = mblock_size.nx3;
    indx.ncells3 = mblock_size.nx3 + 2*mblock_size.nghost;
  } else {
    indx.ks = 0;
    indx.ke = 0;
    indx.nx3 = 1;
    indx.ncells3 = 1;
  }

  // initialize coarse grid indices
  if (pm->multilevel) {
    cindx.nghost = (mblock_size.nghost + 1)/2 + 1;
    cindx.is = cindx.nghost;
    cindx.ie = cindx.is + mblock_size.nx1/2 - 1;
    cindx.nx1 = cindx.ie - cindx.is + 1;
    cindx.ncells1 = cindx.nx1 + 2*cindx.nghost;

    if (mblock_size.nx2 > 1) {
      cindx.js = cindx.nghost;
      cindx.je = cindx.js + mblock_size.nx2/2 - 1;
      cindx.nx2 = cindx.je - cindx.js + 1;
      cindx.ncells2 = cindx.nx2 + 2*cindx.nghost;
    } else {
      cindx.js = 0;
      cindx.je = 0;
      cindx.nx1 = 1;
      cindx.ncells2 = 1;
    }
  
    if (mblock_size.nx3 > 1) {
      cindx.ks = cindx.nghost;
      cindx.ke = cindx.ks + mblock_size.nx3/2 - 1;
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
