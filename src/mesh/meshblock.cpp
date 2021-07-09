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
#include "mesh_positions.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor:
//

MeshBlock::MeshBlock(Mesh* pm, int igids, int nmb) : 
  pmy_mesh(pm), nmb(nmb),
  mbgid("mbgid",nmb),
  mbsize(nmb),
  mb_bcs("mbbcs",nmb,6),
  lb_cost("lbcost",nmb)
{
  // initialize host arrays of gids, sizes, bcs over all MeshBlocks
  auto &msize = pm->mesh_size;
  for (int m=0; m<nmb; ++m) {
    mbgid.h_view(m) = igids + m;

    std::int32_t &lx1 = pm->loclist[igids+m].lx1;
    std::int32_t &lev = pm->loclist[igids+m].level;
    std::int32_t nmbx1 = pm->nmb_rootx1 << (lev - pm->root_level);

    // calculate physical size of MeshBlock in x1.  Second index of mbsize represents:
    // x1min/max = 0,1;  x2min/max = 2,3;  x3min/max = 4,5;  dx1/2/3 = 6,7,8
    if (lx1 == 0) {
      mbsize.x1min.h_view(m) = msize.x1min;
      mb_bcs(m,0) = pm->mesh_bcs[BoundaryFace::inner_x1];
    } else {
      mbsize.x1min.h_view(m) = LeftEdgeX(lx1, nmbx1, msize.x1min, msize.x1max);
      mb_bcs(m,0) = BoundaryFlag::block;
    }

    if (lx1 == nmbx1 - 1) {
      mbsize.x1max.h_view(m) = msize.x1max;
      mb_bcs(m,1) = pm->mesh_bcs[BoundaryFace::outer_x1];
    } else {
      mbsize.x1max.h_view(m) = LeftEdgeX(lx1+1, nmbx1, msize.x1min, msize.x1max);
      mb_bcs(m,1) = BoundaryFlag::block;
    }

    // calculate physical size of MeshBlock in x2
    if (pm->mesh_cells.nx2 == 1) {
      mbsize.x2min.h_view(m) = msize.x2min;
      mbsize.x2max.h_view(m) = msize.x2max;
      mb_bcs(m,2) = pm->mesh_bcs[BoundaryFace::inner_x2];
      mb_bcs(m,3) = pm->mesh_bcs[BoundaryFace::outer_x2];
    } else {

      std::int32_t &lx2 = pm->loclist[igids+m].lx2;
      std::int32_t nmbx2 = pm->nmb_rootx2 << (lev - pm->root_level);
      if (lx2 == 0) {
        mbsize.x2min.h_view(m) = msize.x2min;
        mb_bcs(m,2) = pm->mesh_bcs[BoundaryFace::inner_x2];
      } else {
        mbsize.x2min.h_view(m) = LeftEdgeX(lx2, nmbx2, msize.x2min, msize.x2max);
        mb_bcs(m,2) = BoundaryFlag::block;
      }

      if (lx2 == (nmbx2) - 1) {
        mbsize.x2max.h_view(m) = msize.x2max;
        mb_bcs(m,3) = pm->mesh_bcs[BoundaryFace::outer_x2];
      } else {
        mbsize.x2max.h_view(m) = LeftEdgeX(lx2+1, nmbx2, msize.x2min, msize.x2max);
        mb_bcs(m,3) = BoundaryFlag::block;
      }

    }

    // calculate physical size of MeshBlock in x3
    if (pm->mesh_cells.nx3 == 1) {
      mbsize.x3min.h_view(m) = msize.x3min;
      mbsize.x3max.h_view(m) = msize.x3max;
      mb_bcs(m,4) = pm->mesh_bcs[BoundaryFace::inner_x3];
      mb_bcs(m,5) = pm->mesh_bcs[BoundaryFace::outer_x3];
    } else {
      std::int32_t &lx3 = pm->loclist[igids+m].lx3;
      std::int32_t nmbx3 = pm->nmb_rootx3 << (lev - pm->root_level);
      if (lx3 == 0) {
        mbsize.x3min.h_view(m) = msize.x3min;
        mb_bcs(m,4) = pm->mesh_bcs[BoundaryFace::inner_x3];
      } else {
        mbsize.x3min.h_view(m) = LeftEdgeX(lx3, nmbx3, msize.x3min, msize.x3max);
        mb_bcs(m,4) = BoundaryFlag::block;
      }
      if (lx3 == (nmbx3) - 1) {
        mbsize.x3max.h_view(m) = msize.x3max;
        mb_bcs(m,5) = pm->mesh_bcs[BoundaryFace::outer_x3];
      } else {
        mbsize.x3max.h_view(m) = LeftEdgeX(lx3+1, nmbx3, msize.x3min, msize.x3max);
        mb_bcs(m,5) = BoundaryFlag::block;
      }
    }
    // grid spacing at this level.  Ensure all MeshBlocks at same level have same dx
    mbsize.dx1.h_view(m) = msize.dx1*static_cast<Real>(1<<(lev - pm->root_level));
    mbsize.dx2.h_view(m) = msize.dx2*static_cast<Real>(1<<(lev - pm->root_level));
    mbsize.dx3.h_view(m) = msize.dx3*static_cast<Real>(1<<(lev - pm->root_level));
  }

  // For each DualArray: mark host views as modified, and then sync to device array
  mbgid.template modify<HostMemSpace>();
  mbsize.x1min.template modify<HostMemSpace>();
  mbsize.x1max.template modify<HostMemSpace>();
  mbsize.x2min.template modify<HostMemSpace>();
  mbsize.x2max.template modify<HostMemSpace>();
  mbsize.x3min.template modify<HostMemSpace>();
  mbsize.x3max.template modify<HostMemSpace>();
  mbsize.dx1.template modify<HostMemSpace>();
  mbsize.dx2.template modify<HostMemSpace>();
  mbsize.dx3.template modify<HostMemSpace>();

  mbgid.template sync<DevExeSpace>();
  mbsize.x1min.template sync<DevExeSpace>();
  mbsize.x1max.template sync<DevExeSpace>();
  mbsize.x2min.template sync<DevExeSpace>();
  mbsize.x2max.template sync<DevExeSpace>();
  mbsize.x3min.template sync<DevExeSpace>();
  mbsize.x3max.template sync<DevExeSpace>();
  mbsize.dx1.template sync<DevExeSpace>();
  mbsize.dx2.template sync<DevExeSpace>();
  mbsize.dx3.template sync<DevExeSpace>();
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

  if (pmy_mesh->one_d) {nnghbr = 2;}
  if (pmy_mesh->two_d) {nnghbr = 8;}
  if (pmy_mesh->three_d) {nnghbr = 26;}

  // allocate size of DualArrays
  for (int n=0; n<nnghbr; ++n) {
    Kokkos::realloc(nghbr[n].gid, nmb);
    Kokkos::realloc(nghbr[n].lev, nmb);
    Kokkos::realloc(nghbr[n].rank, nmb);
    Kokkos::realloc(nghbr[n].destn, nmb);
  }

  // Initialize host view elements of DualViews
  for (int n=0; n<nnghbr; ++n) {
    for (int m=0; m<nmb; ++m) {
      nghbr[n].gid.h_view(m) = -1;
      nghbr[n].lev.h_view(m) = -1;
      nghbr[n].rank.h_view(m) = -1;
      nghbr[n].destn.h_view(m) = -1;
    }
  }

  // Search MeshBlock tree and find neighbors
  for (int b=0; b<nmb; ++b) {
    LogicalLocation loc = pmy_mesh->loclist[mbgid.h_view(b)];

    // neighbors on x1face
    for (int n=-1; n<=1; n+=2) {
      nt = ptree->FindNeighbor(loc, n, 0, 0);
      if (nt != nullptr) {
        nghbr[(1+n)/2].gid.h_view(b) = nt->gid_;
        nghbr[(1+n)/2].lev.h_view(b) = nt->loc_.level;
        nghbr[(1+n)/2].rank.h_view(b) = ranklist[nt->gid_];
        nghbr[(1+n)/2].destn.h_view(b) = (1-n)/2;
      }
    }

    // neighbors on x2face and x1x2 edges
    if (pmy_mesh->multi_d) {
      for (int m=-1; m<=1; m+=2) {
        nt = ptree->FindNeighbor(loc, 0, m, 0);
        if (nt != nullptr) {
          nghbr[2+(1+m)/2].gid.h_view(b) = nt->gid_;
          nghbr[2+(1+m)/2].lev.h_view(b) = nt->loc_.level;
          nghbr[2+(1+m)/2].rank.h_view(b) = ranklist[nt->gid_];
          nghbr[2+(1+m)/2].destn.h_view(b) = 2+(1-m)/2;
        }
      }
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          nt = ptree->FindNeighbor(loc, n, m, 0);
          if (nt != nullptr) {
            nghbr[4+(1+m)+(1+n)/2].gid.h_view(b) = nt->gid_;
            nghbr[4+(1+m)+(1+n)/2].lev.h_view(b) = nt->loc_.level;
            nghbr[4+(1+m)+(1+n)/2].rank.h_view(b) = ranklist[nt->gid_];
            nghbr[4+(1+m)+(1+n)/2].destn.h_view(b) = 4+(1-m)+(1-n)/2;
          }
        }
      }
    }

    // neighbors on x3face, x3x1 and x2x3 edges, and corners
    if (pmy_mesh->three_d) {
      for (int l=-1; l<=1; l+=2) {
        nt = ptree->FindNeighbor(loc, 0, 0, l);
        if (nt != nullptr) {
          nghbr[8+(1+l)/2].gid.h_view(b) = nt->gid_;
          nghbr[8+(1+l)/2].lev.h_view(b) = nt->loc_.level;
          nghbr[8+(1+l)/2].rank.h_view(b) = ranklist[nt->gid_];
          nghbr[8+(1+l)/2].destn.h_view(b) = 8+(1-l)/2;
        }
      }
      for (int l=-1; l<=1; l+=2) {
        for (int n=-1; n<=1; n+=2) {
          nt = ptree->FindNeighbor(loc, n, 0, l);
          if (nt != nullptr) {
            nghbr[10+(1+l)+(1+n)/2].gid.h_view(b) = nt->gid_;
            nghbr[10+(1+l)+(1+n)/2].lev.h_view(b) = nt->loc_.level;
            nghbr[10+(1+l)+(1+n)/2].rank.h_view(b) = ranklist[nt->gid_];
            nghbr[10+(1+l)+(1+n)/2].destn.h_view(b) = 10+(1-l)+(1-n)/2;
          }
        }
      }
      for (int l=-1; l<=1; l+=2) {
        for (int m=-1; m<=1; m+=2) {
          nt = ptree->FindNeighbor(loc, 0, m, l);
          if (nt != nullptr) {
            nghbr[14+(1+l)+(1+m)/2].gid.h_view(b) = nt->gid_;
            nghbr[14+(1+l)+(1+m)/2].lev.h_view(b) = nt->loc_.level;
            nghbr[14+(1+l)+(1+m)/2].rank.h_view(b) = ranklist[nt->gid_];
            nghbr[14+(1+l)+(1+m)/2].destn.h_view(b) = 14+(1-l)+(1-m)/2;
          }
        }
      }
      for (int l=-1; l<=1; l+=2) {
        for (int m=-1; m<=1; m+=2) {
          for (int n=-1; n<=1; n+=2) {
            nt = ptree->FindNeighbor(loc, n, m, l);
            if (nt != nullptr) {
              nghbr[18+2*(1+l)+(1+m)+(1+n)/2].gid.h_view(b) = nt->gid_;
              nghbr[18+2*(1+l)+(1+m)+(1+n)/2].lev.h_view(b) = nt->loc_.level;
              nghbr[18+2*(1+l)+(1+m)+(1+n)/2].rank.h_view(b) = ranklist[nt->gid_];
              nghbr[18+2*(1+l)+(1+m)+(1+n)/2].destn.h_view(b) = 18+2*(1-l)+(1-m)+(1-n)/2;
            }
          }
        }
      }
    }
  }

  // For each DualArray: mark host views as modified, and then sync to device array
  for (int n=0; n<nnghbr; ++n) {
    nghbr[n].gid.template modify<HostMemSpace>();
    nghbr[n].lev.template modify<HostMemSpace>();
    nghbr[n].rank.template modify<HostMemSpace>();
    nghbr[n].destn.template modify<HostMemSpace>();

    nghbr[n].gid.template sync<DevExeSpace>();
    nghbr[n].lev.template sync<DevExeSpace>();
    nghbr[n].rank.template sync<DevExeSpace>();
    nghbr[n].destn.template sync<DevExeSpace>();
  }

  return;
}
