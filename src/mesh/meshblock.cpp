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
//
// Data for each MB are stored in Views of dimension [# of MBs]. Views can only store
// basic data types (int, float, double), not objects, so elements of arrays are:
//  mbsize(m,0) = x1min   mb_bcs(m,0) = inner_x1
//  mbsize(m,1) = x1max   mb_bcs(m,1) = outer_x1
//  mbsize(m,2) = x2min   mb_bcs(m,2) = inner_x2
//  mbsize(m,3) = x2max   mb_bcs(m,3) = outer_x2
//  mbsize(m,4) = x3min   mb_bcs(m,4) = inner_x3
//  mbsize(m,5) = x3max   mb_bcs(m,5) = outer_x3
//  mbsize(m,6) = dx1
//  mbsize(m,7) = dx2
//  mbsize(m,8) = dx3
// 

MeshBlock::MeshBlock(Mesh* pm, int igids, int igide) : 
  pmy_mesh(pm),
  h_mbgid("h_mbgid",1),
  h_mbsize("h_mbsize",1,1),
  h_mbnghbr("h_nghbr",1,1,1),

  d_mbgid("d_mbgid",1),
  d_mbsize("d_mbsize",1,1),
  d_mbnghbr("d_nghbr",1,1,1),

  mb_bcs("mbbcs",1,1),
  lb_cost("lbcost",1)
{
  // number of MeshBlocks in this object
  nmb = igide - igids + 1;

  // allocate memory for both host and device views
  Kokkos::realloc(h_mbgid, nmb);
  Kokkos::realloc(h_mbsize, nmb, 9);  // 9 data elements stored for each MB

  Kokkos::realloc(d_mbgid, nmb);
  Kokkos::realloc(d_mbsize, nmb, 9);  // 9 data elements stored for each MB

  Kokkos::realloc(mb_bcs, nmb, 6);   // 6 BoundaryFlags stored for each MB
  Kokkos::realloc(lb_cost, nmb);

  // initialize host arrays of gids, sizes, bcs over all MeshBlocks
  for (int m=0; m<nmb; ++m) {
    h_mbgid(m) = igids + m;

    std::int32_t &lx1 = pm->loclist[igids+m].lx1;
    std::int32_t &lev = pm->loclist[igids+m].level;
    std::int32_t nmbx1 = pm->nmb_rootx1 << (lev - pm->root_level);

    // calculate physical size of MeshBlock in x1.  Second index of mbsize represents:
    // x1min/max = 0,1;  x2min/max = 2,3;  x3min/max = 4,5;  dx1/2/3 = 6,7,8
    if (lx1 == 0) {
      h_mbsize(m,0) = pm->mesh_size.x1min;
      mb_bcs(m,0) = static_cast<int>(pm->mesh_bcs[BoundaryFace::inner_x1]);
    } else {
      h_mbsize(m,0) = LeftEdgeX(lx1, nmbx1, pm->mesh_size.x1min, pm->mesh_size.x1max);
      mb_bcs(m,0) = static_cast<int>(BoundaryFlag::block);
    }

    if (lx1 == nmbx1 - 1) {
      h_mbsize(m,1) = pm->mesh_size.x1max;
      mb_bcs(m,1) = static_cast<int>(pm->mesh_bcs[BoundaryFace::outer_x1]);
    } else {
      h_mbsize(m,1) = LeftEdgeX(lx1+1, nmbx1, pm->mesh_size.x1min, pm->mesh_size.x1max);
      mb_bcs(m,1) = static_cast<int>(BoundaryFlag::block);
    }

    // calculate physical size of MeshBlock in x2
    if (pm->mesh_cells.nx2 == 1) {
      h_mbsize(m,2) = pm->mesh_size.x2min;
      h_mbsize(m,3) = pm->mesh_size.x2max;
      mb_bcs(m,2) = static_cast<int>(pm->mesh_bcs[BoundaryFace::inner_x2]);
      mb_bcs(m,3) = static_cast<int>(pm->mesh_bcs[BoundaryFace::outer_x2]);
    } else {

      std::int32_t &lx2 = pm->loclist[igids+m].lx2;
      std::int32_t nmbx2 = pm->nmb_rootx2 << (lev - pm->root_level);
      if (lx2 == 0) {
        h_mbsize(m,2) = pm->mesh_size.x2min;
        mb_bcs(m,2) = static_cast<int>(pm->mesh_bcs[BoundaryFace::inner_x2]);
      } else {
        h_mbsize(m,2) = LeftEdgeX(lx2, nmbx2, pm->mesh_size.x2min, pm->mesh_size.x2max);
        mb_bcs(m,2) = static_cast<int>(BoundaryFlag::block);
      }

      if (lx2 == (nmbx2) - 1) {
        h_mbsize(m,3) = pm->mesh_size.x2max;
        mb_bcs(m,3) = static_cast<int>(pm->mesh_bcs[BoundaryFace::outer_x2]);
      } else {
        h_mbsize(m,3) = LeftEdgeX(lx2+1, nmbx2,pm->mesh_size.x2min,pm->mesh_size.x2max);
        mb_bcs(m,3) = static_cast<int>(BoundaryFlag::block);
      }

    }

    // calculate physical size of MeshBlock in x3
    if (pm->mesh_cells.nx3 == 1) {
      h_mbsize(m,4) = pm->mesh_size.x3min;
      h_mbsize(m,5) = pm->mesh_size.x3max;
      mb_bcs(m,4) = static_cast<int>(pm->mesh_bcs[BoundaryFace::inner_x3]);
      mb_bcs(m,5) = static_cast<int>(pm->mesh_bcs[BoundaryFace::outer_x3]);
    } else {
      std::int32_t &lx3 = pm->loclist[igids+m].lx3;
      std::int32_t nmbx3 = pm->nmb_rootx3 << (lev - pm->root_level);
      if (lx3 == 0) {
        h_mbsize(m,4) = pm->mesh_size.x3min;
        mb_bcs(m,4) = static_cast<int>(pm->mesh_bcs[BoundaryFace::inner_x3]);
      } else {
        h_mbsize(m,4) = LeftEdgeX(lx3, nmbx3, pm->mesh_size.x3min, pm->mesh_size.x3max);
        mb_bcs(m,4) = static_cast<int>(BoundaryFlag::block);
      }
      if (lx3 == (nmbx3) - 1) {
        h_mbsize(m,5) = pm->mesh_size.x3max;
        mb_bcs(m,5) = static_cast<int>(pm->mesh_bcs[BoundaryFace::outer_x3]);
      } else {
        h_mbsize(m,5) = LeftEdgeX(lx3+1, nmbx3,pm->mesh_size.x3min,pm->mesh_size.x3max);
        mb_bcs(m,5) = static_cast<int>(BoundaryFlag::block);
      }
    }
    // grid spacing at this level.  Ensure all MeshBlocks at same level have same dx
    h_mbsize(m,6) = pm->mesh_size.dx1*static_cast<Real>(1<<(lev - pm->root_level));
    h_mbsize(m,7) = pm->mesh_size.dx2*static_cast<Real>(1<<(lev - pm->root_level));
    h_mbsize(m,8) = pm->mesh_size.dx3*static_cast<Real>(1<<(lev - pm->root_level));
  }

  // copy host arrays to device arrays
  auto t_mbgid = Kokkos::create_mirror_view(d_mbgid);
  Kokkos::deep_copy(t_mbgid,h_mbgid);
  Kokkos::deep_copy(d_mbgid,t_mbgid);

  auto t_mbsize = Kokkos::create_mirror_view(d_mbsize);
  Kokkos::deep_copy(t_mbsize,h_mbsize);
  Kokkos::deep_copy(d_mbsize,t_mbsize);
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
// Information about Neighbors are stored in a 3D array, with the last index storing:
//   mbnghbr(m,n,0) = gid     global ID of neighbor
//   mbnghbr(m,n,1) = level   logical level " "
//   mbnghbr(m,n,2) = rank    MPI rank " "
//   mbnghbr(m,n,3) = destn   index of target recv buffer for comms

void MeshBlock::SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist)
{
  MeshBlockTree* nt;

  nnghbr = 2; // 1D problem
  if (pmy_mesh->nx2gt1) {nnghbr = 8;}   // 2D problem
  if (pmy_mesh->nx3gt1) {nnghbr = 26;}  // 3D problem

  Kokkos::realloc(d_mbnghbr, nmb, nnghbr, 4);
  Kokkos::realloc(h_mbnghbr, nmb, nnghbr, 4);

  // Initialize host array elements
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      for (int i=0; i<4; ++i) {
        h_mbnghbr(m,n,i) = -1;
      }
    }
  }

  // Search MeshBlock tree and find neighbors
  for (int b=0; b<nmb; ++b) {
    LogicalLocation loc = pmy_mesh->loclist[h_mbgid(b)];

    // neighbors on x1face
    for (int n=-1; n<=1; n+=2) {
      nt = ptree->FindNeighbor(loc, n, 0, 0);
      if (nt != nullptr) {
        h_mbnghbr(b,(1+n)/2,0) = nt->gid_;
        h_mbnghbr(b,(1+n)/2,1) = nt->loc_.level;
        h_mbnghbr(b,(1+n)/2,2) = ranklist[nt->gid_];
        h_mbnghbr(b,(1+n)/2,3) = (1-n)/2;
      }
    }

    // neighbors on x2face and x1x2 edges
    if (pmy_mesh->nx2gt1) {
      for (int m=-1; m<=1; m+=2) {
        nt = ptree->FindNeighbor(loc, 0, m, 0);
        if (nt != nullptr) {
          h_mbnghbr(b,2+(1+m)/2,0) = nt->gid_;
          h_mbnghbr(b,2+(1+m)/2,1) = nt->loc_.level;
          h_mbnghbr(b,2+(1+m)/2,2) = ranklist[nt->gid_];
          h_mbnghbr(b,2+(1+m)/2,3) = 2+(1-m)/2;
        }
      }
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          nt = ptree->FindNeighbor(loc, n, m, 0);
          if (nt != nullptr) {
            h_mbnghbr(b,4+(1+m)+(1+n)/2,0) = nt->gid_;
            h_mbnghbr(b,4+(1+m)+(1+n)/2,1) = nt->loc_.level;
            h_mbnghbr(b,4+(1+m)+(1+n)/2,2) = ranklist[nt->gid_];
            h_mbnghbr(b,4+(1+m)+(1+n)/2,3) = 4+(1-m)+(1-n)/2;
          }
        }
      }
    }

    // neighbors on x3face, x3x1 and x2x3 edges, and corners
    if (pmy_mesh->nx3gt1) {
      for (int l=-1; l<=1; l+=2) {
        nt = ptree->FindNeighbor(loc, 0, 0, l);
        if (nt != nullptr) {
          h_mbnghbr(b,8+(1+l)/2,0) = nt->gid_;
          h_mbnghbr(b,8+(1+l)/2,1) = nt->loc_.level;
          h_mbnghbr(b,8+(1+l)/2,2) = ranklist[nt->gid_];
          h_mbnghbr(b,8+(1+l)/2,3) = 8+(1-l)/2;
        }
      }
      for (int l=-1; l<=1; l+=2) {
        for (int n=-1; n<=1; n+=2) {
          nt = ptree->FindNeighbor(loc, n, 0, l);
          if (nt != nullptr) {
            h_mbnghbr(b,10+(1+l)+(1+n)/2,0) = nt->gid_;
            h_mbnghbr(b,10+(1+l)+(1+n)/2,1) = nt->loc_.level;
            h_mbnghbr(b,10+(1+l)+(1+n)/2,2) = ranklist[nt->gid_];
            h_mbnghbr(b,10+(1+l)+(1+n)/2,3) = 10+(1-l)+(1-n)/2;
          }
        }
      }
      for (int l=-1; l<=1; l+=2) {
        for (int m=-1; m<=1; m+=2) {
          nt = ptree->FindNeighbor(loc, 0, m, l);
          if (nt != nullptr) {
            h_mbnghbr(b,14+(1+l)+(1+m)/2,0) = nt->gid_;
            h_mbnghbr(b,14+(1+l)+(1+m)/2,1) = nt->loc_.level;
            h_mbnghbr(b,14+(1+l)+(1+m)/2,2) = ranklist[nt->gid_];
            h_mbnghbr(b,14+(1+l)+(1+m)/2,3) = 14+(1-l)+(1-m)/2;
          }
        }
      }
      for (int l=-1; l<=1; l+=2) {
        for (int m=-1; m<=1; m+=2) {
          for (int n=-1; n<=1; n+=2) {
            nt = ptree->FindNeighbor(loc, n, m, l);
            if (nt != nullptr) {
              h_mbnghbr(b,18+2*(1-l)+(1+m)+(1+n)/2,0) = nt->gid_;
              h_mbnghbr(b,18+2*(1-l)+(1+m)+(1+n)/2,1) = nt->loc_.level;
              h_mbnghbr(b,18+2*(1-l)+(1+m)+(1+n)/2,2) = ranklist[nt->gid_];
              h_mbnghbr(b,18+2*(1-l)+(1+m)+(1+n)/2,3) = 18+2*(1-l)+(1-m)+(1-n)/2;
            }
          }
        }
      }
    }
  }

  // copy host array to device array
  auto t_mbnghbr = Kokkos::create_mirror_view(d_mbnghbr);
  Kokkos::deep_copy(t_mbnghbr,h_mbnghbr);
  Kokkos::deep_copy(d_mbnghbr,t_mbnghbr);

  return;
}
