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

MeshBlock::MeshBlock(Mesh *pm, ParameterInput *pin, int igid, RegionSize isize,
                     RegionCells icells, BoundaryFlag *ibcs) :
  mb_gid(igid), mb_size(isize), mb_cells(icells), exe_space(DevExecSpace()), pmesh_(pm)
{
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

  // construct boundary conditions object
  pbvals = new BoundaryValues(pmesh_, pin, mb_gid, ibcs);
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock()
{
}

//----------------------------------------------------------------------------------------
// \!fn void MeshBlock::FindAndSetNeighbors()
// \brief Search and set all the neighbor blocks

void MeshBlock::SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist)
{
  MeshBlockTree* neibt;
  LogicalLocation loc = pmesh_->loclist[mb_gid];

  // neighbors on x1face
  int cnt=0;
  for (int n=-1; n<=1; n+=2) {
    neibt = ptree->FindNeighbor(loc, n, 0, 0);
    if (neibt != nullptr) {
      pbvals->nghbr_x1face[cnt].gid   = neibt->gid_;
      pbvals->nghbr_x1face[cnt].level = neibt->loc_.level;
      pbvals->nghbr_x1face[cnt].rank  = ranklist[neibt->gid_];
    }
    ++cnt;
  }
  if (mb_cells.nx2 == 1) {return;}  // stop if 1D

  // neighbors on x2face and x1x2 edges
  cnt=0;
  for (int m=-1; m<=1; m+=2) {
    neibt = ptree->FindNeighbor(loc, 0, m, 0);
    if (neibt != nullptr) {
      pbvals->nghbr_x2face[cnt].gid   = neibt->gid_;
      pbvals->nghbr_x2face[cnt].level = neibt->loc_.level;
      pbvals->nghbr_x2face[cnt].rank  = ranklist[neibt->gid_];
    }
    ++cnt;
  }
  cnt=0;
  for (int m=-1; m<=1; m+=2) {
    for (int n=-1; n<=1; n+=2) {
      neibt = ptree->FindNeighbor(loc, n, m, 0);
      if (neibt != nullptr) {
        pbvals->nghbr_x1x2ed[cnt].gid   = neibt->gid_;
        pbvals->nghbr_x1x2ed[cnt].level = neibt->loc_.level;
        pbvals->nghbr_x1x2ed[cnt].rank  = ranklist[neibt->gid_];
      }
      ++cnt;
    }
  }
  if (mb_cells.nx3 == 1) {return;}  // stop if 2D

  // neighbors on x3face, x3x1 and x2x3 edges, and corners
  cnt=0;
  for (int l=-1; l<=1; l+=2) {
    neibt = ptree->FindNeighbor(loc, 0, 0, l);
    if (neibt != nullptr) {
      pbvals->nghbr_x3face[cnt].gid   = neibt->gid_;
      pbvals->nghbr_x3face[cnt].level = neibt->loc_.level;
      pbvals->nghbr_x3face[cnt].rank  = ranklist[neibt->gid_];
    }
    ++cnt;
  }
  cnt=0;
  for (int l=-1; l<=1; l+=2) {
    for (int n=-1; n<=1; n+=2) {
      neibt = ptree->FindNeighbor(loc, n, 0, l);
      if (neibt != nullptr) {
        pbvals->nghbr_x3x1ed[cnt].gid   = neibt->gid_;
        pbvals->nghbr_x3x1ed[cnt].level = neibt->loc_.level;
        pbvals->nghbr_x3x1ed[cnt].rank  = ranklist[neibt->gid_];
      }
      ++cnt;
    }
  }
  cnt=0;
  for (int l=-1; l<=1; l+=2) {
    for (int m=-1; m<=1; m+=2) {
      neibt = ptree->FindNeighbor(loc, 0, m, l);
      if (neibt != nullptr) {
        pbvals->nghbr_x2x3ed[cnt].gid   = neibt->gid_;
        pbvals->nghbr_x2x3ed[cnt].level = neibt->loc_.level;
        pbvals->nghbr_x2x3ed[cnt].rank  = ranklist[neibt->gid_];
      }
      ++cnt;
    }
  }
  cnt=0;
  for (int l=-1; l<=1; l+=2) {
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        neibt = ptree->FindNeighbor(loc, n, m, l);
        if (neibt != nullptr) {
          pbvals->nghbr_corner[cnt].gid   = neibt->gid_;
          pbvals->nghbr_corner[cnt].level = neibt->loc_.level;
          pbvals->nghbr_corner[cnt].rank  = ranklist[neibt->gid_];
        }
        ++cnt;
      }
    }
  }

  return;
}
