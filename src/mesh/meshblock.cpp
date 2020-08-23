//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in MeshBlock class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "bvals/bvals.hpp"
#include "mesh.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, hydro, field
//                        and mesh refinement objects.

MeshBlock::MeshBlock(Mesh *pm, std::unique_ptr<ParameterInput> &pin, int igid,
  RegionSize isize, RegionCells icells, BoundaryFlag *input_bcs) :
  pmesh_mb(pm), mb_gid(igid), mb_size(isize), mb_cells(icells) {

  // copy input boundary flags into MeshBlock 
//  for (int i=0; i<6; ++i) {mb_bcs[i] = input_bcs[i];}

  // initialize grid indices
  mb_cells.is = mb_cells.ng;
  mb_cells.ie = mb_cells.is + mb_cells.nx1 - 1;

  if (mb_cells.nx2 > 1) {
    mb_cells.js = mb_cells.ng;
    mb_cells.je = mb_cells.js + mb_cells.nx2 - 1;
  } else {
    mb_cells.js = 0;
    mb_cells.je = 0;
  }

  if (mb_cells.nx3 > 1) {
    mb_cells.ks = mb_cells.ng;
    mb_cells.ke = mb_cells.ks + mb_cells.nx3 - 1;
  } else {
    mb_cells.ks = 0;
    mb_cells.ke = 0;
  }

  // initialize coarse grid indices
  if (pm->multilevel) {
    cmb_cells.ng = (mb_cells.ng + 1)/2 + 1;
    cmb_cells.is = cmb_cells.ng;
    cmb_cells.ie = cmb_cells.is + mb_cells.nx1/2 - 1;
    cmb_cells.nx1 = cmb_cells.ie - cmb_cells.is + 1;

    if (mb_cells.nx2 > 1) {
      cmb_cells.js = cmb_cells.ng;
      cmb_cells.je = cmb_cells.js + mb_cells.nx2/2 - 1;
      cmb_cells.nx2 = cmb_cells.je - cmb_cells.js + 1;
    } else {
      cmb_cells.js = 0;
      cmb_cells.je = 0;
      cmb_cells.nx2 = 1;
    }
  
    if (mb_cells.nx3 > 1) {
      cmb_cells.ks = cmb_cells.ng;
      cmb_cells.ke = cmb_cells.ks + mb_cells.nx3/2 - 1;
      cmb_cells.nx3 = cmb_cells.ke - cmb_cells.ks + 1;
    } else {
      cmb_cells.ks = 0;
      cmb_cells.ke = 0;
      cmb_cells.nx3 = 1;
    }
  }

std::cout << "MBConstruct nx1=" << mb_cells.nx1 << " nx2=" << mb_cells.nx2 << " nx3=" << mb_cells.nx3 << std::endl;

  pbval = new BoundaryValues(this, pin, input_bcs);

  return;
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts


//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock() {
}

//----------------------------------------------------------------------------------------
// \!fn void MeshBlock::FindAndSetNeighbors()
// \brief Search and set all the neighbor blocks

void MeshBlock::SetNeighbors(MeshBlockTree *ptree, int *ranklist) {

  MeshBlockTree* neibt;
  LogicalLocation loc = pmesh_mb->loclist[mb_gid];

  int cnt=-1;
  // iterate over x3/x2/x1 faces and load all 26 neighbors on a uniform grid.
  // Neighbors are indexed sequentially starting from lower corner.
  for (int l=-1; l<=1; ++l) {
  for (int m=-1; m<=1; ++m) {
  for (int n=-1; n<=1; ++n) {
    if (n==0 && m==0 && l==0) {continue;}
    ++cnt;
    if (mb_cells.nx3 == 1 && l != 0) {continue;}  // skip if not 3D
    if (mb_cells.nx2 == 1 && m != 0) {continue;}  // skip if not 2D
    neibt = ptree->FindNeighbor(loc, n, m, l);
    if (neibt == nullptr) {continue;}
    pbval->neighbor[cnt].ngid   = neibt->gid_;
    pbval->neighbor[cnt].nlevel = neibt->loc_.level;
    pbval->neighbor[cnt].nrank  = ranklist[neibt->gid_];
  }}}

  return;
}
