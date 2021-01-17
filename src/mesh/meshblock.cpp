//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock.cpp
//  \brief implementation of constructor and functions in MeshBlock class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "utils/grid_locations.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor:

MeshBlock::MeshBlock(MeshBlockPack* pp, int igid) : mb_gid(igid), pmy_pack(pp)
{
  Mesh* pm = pmy_pack->pmesh;
  std::int32_t &lx1 = pm->loclist[igid].lx1;
  std::int32_t &lev = pm->loclist[igid].level;
  std::int32_t nmbx1 = pm->nmb_rootx1 << (lev - pm->root_level);

  // calculate physical size of MeshBlock in x1
  if (lx1 == 0) {
    mb_size.x1min = pm->mesh_size.x1min;
    mb_bcs[BoundaryFace::inner_x1] = pm->mesh_bcs[BoundaryFace::inner_x1];
  } else {
    mb_size.x1min = LeftEdgeX(lx1, nmbx1, pm->mesh_size.x1min, pm->mesh_size.x1max);
    mb_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }

  if (lx1 == nmbx1 - 1) {
    mb_size.x1max = pm->mesh_size.x1max;
    mb_bcs[BoundaryFace::outer_x1] = pm->mesh_bcs[BoundaryFace::outer_x1];
  } else {
    mb_size.x1max = LeftEdgeX(lx1+1, nmbx1, pm->mesh_size.x1min, pm->mesh_size.x1max);
    mb_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical size of MeshBlock in x2
  if (pm->mesh_cells.nx2 == 1) {
    mb_size.x2min = pm->mesh_size.x2min;
    mb_size.x2max = pm->mesh_size.x2max;
    mb_bcs[BoundaryFace::inner_x2] = pm->mesh_bcs[BoundaryFace::inner_x2];
    mb_bcs[BoundaryFace::outer_x2] = pm->mesh_bcs[BoundaryFace::outer_x2];
  } else {

    std::int32_t &lx2 = pm->loclist[igid].lx2;
    std::int32_t nmbx2 = pm->nmb_rootx2 << (lev - pm->root_level);
    if (lx2 == 0) {
      mb_size.x2min = pm->mesh_size.x2min;
      mb_bcs[BoundaryFace::inner_x2] = pm->mesh_bcs[BoundaryFace::inner_x2];
    } else {
      mb_size.x2min = LeftEdgeX(lx2, nmbx2, pm->mesh_size.x2min, pm->mesh_size.x2max);
      mb_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }

    if (lx2 == (nmbx2) - 1) {
      mb_size.x2max = pm->mesh_size.x2max;
      mb_bcs[BoundaryFace::outer_x2] = pm->mesh_bcs[BoundaryFace::outer_x2];
    } else {
      mb_size.x2max = LeftEdgeX(lx2+1, nmbx2,pm->mesh_size.x2min,pm->mesh_size.x2max);
      mb_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }

  }

  // calculate physical size of MeshBlock in x3
  if (pm->mesh_cells.nx3 == 1) {
    mb_size.x3min = pm->mesh_size.x3min;
    mb_size.x3max = pm->mesh_size.x3max;
    mb_bcs[BoundaryFace::inner_x3] = pm->mesh_bcs[BoundaryFace::inner_x3];
    mb_bcs[BoundaryFace::outer_x3] = pm->mesh_bcs[BoundaryFace::outer_x3];
  } else {
    std::int32_t &lx3 = pm->loclist[igid].lx3;
    std::int32_t nmbx3 = pm->nmb_rootx3 << (lev - pm->root_level);
    if (lx3 == 0) {
      mb_size.x3min = pm->mesh_size.x3min;
      mb_bcs[BoundaryFace::inner_x3] = pm->mesh_bcs[BoundaryFace::inner_x3];
    } else {
      mb_size.x3min = LeftEdgeX(lx3, nmbx3, pm->mesh_size.x3min, pm->mesh_size.x3max);
      mb_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nmbx3) - 1) {
      mb_size.x3max = pm->mesh_size.x3max;
      mb_bcs[BoundaryFace::outer_x3] = pm->mesh_bcs[BoundaryFace::outer_x3];
    } else {
      mb_size.x3max = LeftEdgeX(lx3+1, nmbx3,pm->mesh_size.x3min,pm->mesh_size.x3max);
      mb_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }
  // grid spacing at this level.  Ensure all MeshBlocks at same level have same dx
  mb_size.dx1 = pm->mesh_size.dx1*static_cast<Real>(1<<(lev - pm->root_level));
  mb_size.dx2 = pm->mesh_size.dx2*static_cast<Real>(1<<(lev - pm->root_level));
  mb_size.dx3 = pm->mesh_size.dx3*static_cast<Real>(1<<(lev - pm->root_level));
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
  MeshBlockTree* nt;
  LogicalLocation loc = pmy_pack->pmesh->loclist[mb_gid];

  // neighbors on x1face
  for (int n=-1; n<=1; n+=2) {
    nt = ptree->FindNeighbor(loc, n, 0, 0);
    if (nt != nullptr) {
      nghbr.emplace_back(nt->gid_, nt->loc_.level, ranklist[nt->gid_], (1-n)/2);
    } else {
      nghbr.emplace_back(-1, -1, -1, -1);
    }
  }
  if (pmy_pack->mb_cells.nx2 == 1) {return;}  // stop if 1D

  // neighbors on x2face and x1x2 edges
  for (int m=-1; m<=1; m+=2) {
    nt = ptree->FindNeighbor(loc, 0, m, 0);
    if (nt != nullptr) {
      nghbr.emplace_back(nt->gid_, nt->loc_.level, ranklist[nt->gid_], 2+(1-m)/2);
    } else {
      nghbr.emplace_back(-1, -1, -1, -1);
    }
  }
  for (int m=-1; m<=1; m+=2) {
    for (int n=-1; n<=1; n+=2) {
      nt = ptree->FindNeighbor(loc, n, m, 0);
      if (nt != nullptr) {
        nghbr.emplace_back(nt->gid_, nt->loc_.level, ranklist[nt->gid_], 4+(1-m)+(1-n)/2);
      } else {
        nghbr.emplace_back(-1, -1, -1, -1);
      }
    }
  }
  if (pmy_pack->mb_cells.nx3 == 1) {return;}  // stop if 2D

  // neighbors on x3face, x3x1 and x2x3 edges, and corners
  for (int l=-1; l<=1; l+=2) {
    nt = ptree->FindNeighbor(loc, 0, 0, l);
    if (nt != nullptr) {
      nghbr.emplace_back(nt->gid_, nt->loc_.level, ranklist[nt->gid_], 8+(1-l)/2);
    } else {
      nghbr.emplace_back(-1, -1, -1, -1);
    }
  }
  for (int l=-1; l<=1; l+=2) {
    for (int n=-1; n<=1; n+=2) {
      nt = ptree->FindNeighbor(loc, n, 0, l);
      if (nt != nullptr) {
        nghbr.emplace_back(nt->gid_, nt->loc_.level, ranklist[nt->gid_], 10+(1-l)+(1-n)/2);
      } else {
        nghbr.emplace_back(-1, -1, -1, -1);
      }
    }
  }
  for (int l=-1; l<=1; l+=2) {
    for (int m=-1; m<=1; m+=2) {
      nt = ptree->FindNeighbor(loc, 0, m, l);
      if (nt != nullptr) {
        nghbr.emplace_back(nt->gid_, nt->loc_.level, ranklist[nt->gid_],14+(1-l)+(1-m)/2);
      } else {
        nghbr.emplace_back(-1, -1, -1, -1);
      }
    }
  }
  for (int l=-1; l<=1; l+=2) {
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        nt = ptree->FindNeighbor(loc, n, m, l);
        if (nt != nullptr) {
          nghbr.emplace_back(nt->gid_,nt->loc_.level,ranklist[nt->gid_],18+2*(1-l)+(1-m)+(1-n)/2);
        } else {
          nghbr.emplace_back(-1, -1, -1, -1);
        }
      }
    }
  }

  return;
}
