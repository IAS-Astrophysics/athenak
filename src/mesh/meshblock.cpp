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
               RegionSize isize, RegionCells icells, int igid, BoundaryFlag *input_bcs) :
    pmy_mesh(pm), mblock_size(isize), mblock_cells(icells), mblock_gid(igid) {

  // copy input boundary flags into MeshBlock 
  for (int i=0; i<6; ++i) {mblock_bcs[i] = input_bcs[i];}

  // initialize grid indices
  mblock_cells.is = mblock_cells.nghost;
  mblock_cells.ie = mblock_cells.is + mblock_cells.nx1 - 1;

  if (mblock_cells.nx2 > 1) {
    mblock_cells.js = mblock_cells.nghost;
    mblock_cells.je = mblock_cells.js + mblock_cells.nx2 - 1;
  } else {
    mblock_cells.js = 0;
    mblock_cells.je = 0;
  }

  if (mblock_cells.nx3 > 1) {
    mblock_cells.ks = mblock_cells.nghost;
    mblock_cells.ke = mblock_cells.ks + mblock_cells.nx3 - 1;
  } else {
    mblock_cells.ks = 0;
    mblock_cells.ke = 0;
  }

  // initialize coarse grid indices
  if (pm->multilevel) {
    cmb_cells.nghost = (mblock_cells.nghost + 1)/2 + 1;
    cmb_cells.is = cmb_cells.nghost;
    cmb_cells.ie = cmb_cells.is + mblock_cells.nx1/2 - 1;
    cmb_cells.nx1 = cmb_cells.ie - cmb_cells.is + 1;

    if (mblock_cells.nx2 > 1) {
      cmb_cells.js = cmb_cells.nghost;
      cmb_cells.je = cmb_cells.js + mblock_cells.nx2/2 - 1;
      cmb_cells.nx2 = cmb_cells.je - cmb_cells.js + 1;
    } else {
      cmb_cells.js = 0;
      cmb_cells.je = 0;
      cmb_cells.nx1 = 1;
    }
  
    if (mblock_cells.nx3 > 1) {
      cmb_cells.ks = cmb_cells.nghost;
      cmb_cells.ke = cmb_cells.ks + mblock_cells.nx3/2 - 1;
      cmb_cells.nx3 = cmb_cells.ke - cmb_cells.ks + 1;
    } else {
      cmb_cells.ks = 0;
      cmb_cells.ke = 0;
      cmb_cells.nx3 = 1;
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
