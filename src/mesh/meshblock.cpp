//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock.cpp
//  \brief implementation of constructor and functions in MeshBlock class

#include <cstdlib>
#include <iostream>
#include <memory>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "nghbr_index.hpp"
#include "meshblock.hpp"

//----------------------------------------------------------------------------------------
// MeshBlock constructor:
// Initializes mb_gid, mb_lev, mb_size, mb_bcs arrays.  The nghbrs array is initialized
// by SetNeighbors function called by BuildTree***() functions.

MeshBlock::MeshBlock(MeshBlockPack* ppack, int igids, int nmb) :
  pmy_pack(ppack),
  mb_gid("mb_gid",nmb),
  mb_lev("mb_lev",nmb),
  mb_size("mbsize",nmb),
  mb_bcs("mbbcs",nmb,6) {
  Mesh* pm = pmy_pack->pmesh;
  auto &ms = pm->mesh_size;

  for (int m=0; m<nmb; ++m) {
    // initialize host array elements of gids, levels
    mb_gid.h_view(m) = igids + m;
    mb_lev.h_view(m) = pm->lloc_eachmb[igids+m].level;

    // calculate physical size and set BCs of each MeshBlock in x1
    std::int32_t &lx1 = pm->lloc_eachmb[igids+m].lx1;
    std::int32_t &lev = pm->lloc_eachmb[igids+m].level;
    std::int32_t nmbx1 = pm->nmb_rootx1 << (lev - pm->root_level);
    if (lx1 == 0) {
      mb_size.h_view(m).x1min = ms.x1min;
      mb_bcs.h_view(m,0) = pm->mesh_bcs[BoundaryFace::inner_x1];
    } else {
      mb_size.h_view(m).x1min = LeftEdgeX(lx1, nmbx1, ms.x1min, ms.x1max);
      mb_bcs.h_view(m,0) = BoundaryFlag::block;
    }

    if (lx1 == nmbx1 - 1) {
      mb_size.h_view(m).x1max = ms.x1max;
      mb_bcs.h_view(m,1) = pm->mesh_bcs[BoundaryFace::outer_x1];
    } else {
      mb_size.h_view(m).x1max = LeftEdgeX(lx1+1, nmbx1, ms.x1min, ms.x1max);
      mb_bcs.h_view(m,1) = BoundaryFlag::block;
    }

    // calculate physical size and set BCs of each MeshBlock in x2, dependng on whether
    // there are none (1D), one, or more MeshBlocks in this direction
    if (!(pm->multi_d)) {
      mb_size.h_view(m).x2min = ms.x2min;
      mb_size.h_view(m).x2max = ms.x2max;
      mb_bcs.h_view(m,2) = pm->mesh_bcs[BoundaryFace::inner_x2];
      mb_bcs.h_view(m,3) = pm->mesh_bcs[BoundaryFace::outer_x2];
    } else {
      std::int32_t &lx2 = pm->lloc_eachmb[igids+m].lx2;
      std::int32_t nmbx2 = pm->nmb_rootx2 << (lev - pm->root_level);
      if (lx2 == 0) {
        mb_size.h_view(m).x2min = ms.x2min;
        mb_bcs.h_view(m,2) = pm->mesh_bcs[BoundaryFace::inner_x2];
      } else {
        mb_size.h_view(m).x2min = LeftEdgeX(lx2, nmbx2, ms.x2min, ms.x2max);
        mb_bcs.h_view(m,2) = BoundaryFlag::block;
      }

      if (lx2 == (nmbx2) - 1) {
        mb_size.h_view(m).x2max = ms.x2max;
        mb_bcs.h_view(m,3) = pm->mesh_bcs[BoundaryFace::outer_x2];
      } else {
        mb_size.h_view(m).x2max = LeftEdgeX(lx2+1, nmbx2, ms.x2min, ms.x2max);
        mb_bcs.h_view(m,3) = BoundaryFlag::block;
      }
    }

    // calculate physical size and set BCs of each MeshBlock in x1, dependng on whether
    // there are none (1D/2D), one, or more MeshBlocks in this direction
    if (!(pm->three_d)) {
      mb_size.h_view(m).x3min = ms.x3min;
      mb_size.h_view(m).x3max = ms.x3max;
      mb_bcs.h_view(m,4) = pm->mesh_bcs[BoundaryFace::inner_x3];
      mb_bcs.h_view(m,5) = pm->mesh_bcs[BoundaryFace::outer_x3];
    } else {
      std::int32_t &lx3 = pm->lloc_eachmb[igids+m].lx3;
      std::int32_t nmbx3 = pm->nmb_rootx3 << (lev - pm->root_level);
      if (lx3 == 0) {
        mb_size.h_view(m).x3min = ms.x3min;
        mb_bcs.h_view(m,4) = pm->mesh_bcs[BoundaryFace::inner_x3];
      } else {
        mb_size.h_view(m).x3min = LeftEdgeX(lx3, nmbx3, ms.x3min, ms.x3max);
        mb_bcs.h_view(m,4) = BoundaryFlag::block;
      }
      if (lx3 == (nmbx3) - 1) {
        mb_size.h_view(m).x3max = ms.x3max;
        mb_bcs.h_view(m,5) = pm->mesh_bcs[BoundaryFace::outer_x3];
      } else {
        mb_size.h_view(m).x3max = LeftEdgeX(lx3+1, nmbx3, ms.x3min, ms.x3max);
        mb_bcs.h_view(m,5) = BoundaryFlag::block;
      }
    }

    // grid spacing at this level.
    mb_size.h_view(m).dx1 = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min)/
                            static_cast<Real>(pm->mb_indcs.nx1);
    mb_size.h_view(m).dx2 = (mb_size.h_view(m).x2max - mb_size.h_view(m).x2min)/
                            static_cast<Real>(pm->mb_indcs.nx2);
    mb_size.h_view(m).dx3 = (mb_size.h_view(m).x3max - mb_size.h_view(m).x3min)/
                            static_cast<Real>(pm->mb_indcs.nx3);
  }

  // For each DualArray: mark host views as modified, and then sync to device array
  mb_gid.template modify<HostMemSpace>();
  mb_lev.template modify<HostMemSpace>();
  mb_size.template modify<HostMemSpace>();
  mb_bcs.template modify<HostMemSpace>();

  mb_gid.template sync<DevExeSpace>();
  mb_lev.template sync<DevExeSpace>();
  mb_size.template sync<DevExeSpace>();
  mb_bcs.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
// \!fn void MeshBlock::SetNeighbors()
// \brief set information about all the neighboring MeshBlocks.  In 3D with SMR/AMR, that
// can be up to 56 neighbors, although sometimes edges and/or corners overlap with faces
// on the same neighbor, and so these are redundant and not needed.
// Information about Neighbors are stored in a 2D Dual view of NeighborBlock structs
// Indices of the view are (m,n) = (no. of MBs, no. of neighbors)
// Based on SearchAndSetNeighbors() function in /src/bvals/bvals_base.cpp in C++ version

void MeshBlock::SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist) {
  // min number of array elements needed to store MeshBlock neighbors withe SMR/AMR
  // Note not all buffers will be allocated for all nghbrs
  if (pmy_pack->pmesh->one_d) {nnghbr = 8;}
  if (pmy_pack->pmesh->two_d) {nnghbr = 24;}
  if (pmy_pack->pmesh->three_d) {nnghbr = 56;}

  // allocate size of DualArrays
  int nmb = pmy_pack->nmb_thispack;
  Kokkos::realloc(nghbr, nmb, nnghbr);

  // Initialize host view elements of DualViews
  for (int n=0; n<nnghbr; ++n) {
    for (int m=0; m<nmb; ++m) {
      nghbr.h_view(m,n).gid   = -1;
      nghbr.h_view(m,n).lev   = -1;
      nghbr.h_view(m,n).rank  = -1;
      nghbr.h_view(m,n).dest  = -1;
    }
  }

  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // Search MeshBlock tree and find neighbors
  for (int b=0; b<nmb; ++b) {
    LogicalLocation lloc = pmy_pack->pmesh->lloc_eachmb[mb_gid.h_view(b)];

    // find location of this MeshBlock relative to XXXX
    int myox1, myox2 = 0, myox3 = 0, myfx1, myfx2, myfx3;
    myfx1 = ((lloc.lx1 & 1) == 1);
    myfx2 = ((lloc.lx2 & 1) == 1);
    myfx3 = ((lloc.lx3 & 1) == 1);
    myox1 = ((lloc.lx1 & 1) == 1)*2 - 1;
    if (pmy_pack->pmesh->multi_d) myox2 = ((lloc.lx2 & 1) == 1)*2 - 1;
    if (pmy_pack->pmesh->three_d) myox3 = ((lloc.lx3 & 1) == 1)*2 - 1;

    // neighbors on x1face
    for (int n=-1; n<=1; n+=2) {
      MeshBlockTree* nt = ptree->FindNeighbor(lloc, n, 0, 0);
      if (nt != nullptr) {
        if (nt->pleaf_ != nullptr) {  // neighbor at finer level -- requires subblocks
          int ffx = 1 - (n + 1)/2; // 0 for BoundaryFace::outer_x1, 1 for inner_x1
          for (int fz=0; fz<nfz; fz++) {
            for (int fy = 0; fy<nfy; fy++) {
              MeshBlockTree* nf = nt->GetLeaf(ffx, fy, fz);
              int inghbr = NeighborIndex(n,0,0,fy,fz);
              nghbr.h_view(b,inghbr).gid = nf->gid_;
              nghbr.h_view(b,inghbr).lev = nf->lloc_.level;
              nghbr.h_view(b,inghbr).rank = ranklist[nf->gid_];
              nghbr.h_view(b,inghbr).dest = NeighborIndex(-n,0,0,fy,fz);
            }
          }
        } else {   // neighbor at same or coarser level
          int idest, inghbr;
          if (nt->lloc_.level == lloc.level) { // neighbor at same level -- no subblocks
            inghbr = NeighborIndex(n,0,0,0,0);
            idest = NeighborIndex(-n,0,0,0,0);
          } else { // neighbor at coarser level, set index/destn to appropriate subblock
            inghbr = NeighborIndex(n,0,0,myfx2,myfx3);
            idest = NeighborIndex(-n,0,0,myfx2,myfx3);
          }
          nghbr.h_view(b,inghbr).gid = nt->gid_;
          nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
          nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
          nghbr.h_view(b,inghbr).dest = idest;
        }
      }
    }

    // neighbors on x2face
    if (pmy_pack->pmesh->multi_d) {
      for (int m=-1; m<=1; m+=2) {
        MeshBlockTree* nt = ptree->FindNeighbor(lloc, 0, m, 0);
        if (nt != nullptr) {
          if (nt->pleaf_ != nullptr) {  // neighbor at finer level -- requires subblocks
            int ffy = 1 - (m + 1)/2; // 0 for BoundaryFace::outer_x2, 1 for inner_x2
            for (int fz=0; fz<nfz; fz++) {
              for (int fx = 0; fx<nfx; fx++) {
                MeshBlockTree* nf = nt->GetLeaf(fx, ffy, fz);
                int inghbr = NeighborIndex(0,m,0,fx,fz);
                nghbr.h_view(b,inghbr).gid = nf->gid_;
                nghbr.h_view(b,inghbr).lev = nf->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nf->gid_];
                nghbr.h_view(b,inghbr).dest = NeighborIndex(0,-m,0,fx,fz);
              }
            }
          } else {   // neighbor at same or coarser level
            int idest,inghbr;
            if (nt->lloc_.level == lloc.level) { // neighbor at same level -- no subblocks
              inghbr = NeighborIndex(0,m,0,0,0);
              idest = NeighborIndex(0,-m,0,0,0);
            } else { // neighbor at coarser level, set index/destn to appropriate subblock
              inghbr = NeighborIndex(0,m,0,myfx1,myfx3);
              idest = NeighborIndex(0,-m,0,myfx1,myfx3);
            }
            nghbr.h_view(b,inghbr).gid = nt->gid_;
            nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
            nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
            nghbr.h_view(b,inghbr).dest = idest;
          }
        }
      }

      // neighbors on x1x2 edges
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          MeshBlockTree* nt = ptree->FindNeighbor(lloc, n, m, 0);
          if (nt != nullptr) {
            if (nt->pleaf_ != nullptr) {  // neighbor at finer level -- requires subblocks
              int ffx = 1 - (n + 1)/2; // 0 for BoundaryFace::outer_x1, 1 for inner_x1
              int ffy = 1 - (m + 1)/2; // 0 for BoundaryFace::outer_x2, 1 for inner_x2
              for (int fz=0; fz<nfz; fz++) {
                MeshBlockTree* nf = nt->GetLeaf(ffx, ffy, fz);
                int inghbr = NeighborIndex(n,m,0,fz,0);
                nghbr.h_view(b,inghbr).gid = nf->gid_;
                nghbr.h_view(b,inghbr).lev = nf->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nf->gid_];
                nghbr.h_view(b,inghbr).dest = NeighborIndex(-n,-m,0,fz,0);
              }
            } else {   // neighbor at same or coarser level
              int idest,inghbr;
              if (nt->lloc_.level == lloc.level) { // same level -- no subblocks
                inghbr = NeighborIndex(n,m,0,0,0);
                idest = NeighborIndex(-n,-m,0,0,0);
              } else { // neighbor at coarser level, set indx/dest to appropriate subblock
                inghbr = NeighborIndex(n,m,0,myfx3,0);
                idest = NeighborIndex(-n,-m,0,myfx3,0);
              }
              // only set neighbor for exterior edges of coarser face
              if (nt->lloc_.level >= lloc.level || (myox1 == n && myox2 == m)) {
                nghbr.h_view(b,inghbr).gid = nt->gid_;
                nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
                nghbr.h_view(b,inghbr).dest = idest;
              }
            }
          }
        }
      }
    }

    // neighbors on x3face
    if (pmy_pack->pmesh->three_d) {
      for (int l=-1; l<=1; l+=2) {
        MeshBlockTree* nt = ptree->FindNeighbor(lloc, 0, 0, l);
        if (nt != nullptr) {
          if (nt->pleaf_ != nullptr) {  // neighbor at finer level -- requires subblocks
            int ffz = 1 - (l + 1)/2; // 0 for BoundaryFace::outer_x3, 1 for inner_x3
            for (int fy=0; fy<nfy; fy++) {
              for (int fx = 0; fx<nfx; fx++) {
                MeshBlockTree* nf = nt->GetLeaf(fx, fy, ffz);
                int inghbr = NeighborIndex(0,0,l,fx,fy);
                nghbr.h_view(b,inghbr).gid = nf->gid_;
                nghbr.h_view(b,inghbr).lev = nf->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nf->gid_];
                nghbr.h_view(b,inghbr).dest = NeighborIndex(0,0,-l,fx,fy);
              }
            }
          } else {   // neighbor at same or coarser level -- no subblocks
            int idest,inghbr;
            if (nt->lloc_.level == lloc.level) { // neighbor at same level
              inghbr = NeighborIndex(0,0,l,0,0);
              idest = NeighborIndex(0,0,-l,0,0);
            } else { // neighbor at coarser level, set index/destn to appropriate subblock
              inghbr = NeighborIndex(0,0,l,myfx1,myfx2);
              idest = NeighborIndex(0,0,-l,myfx1,myfx2);
            }
            nghbr.h_view(b,inghbr).gid = nt->gid_;
            nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
            nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
            nghbr.h_view(b,inghbr).dest = idest;
          }
        }
      }

      // neighbors on x3x1 edges
      for (int l=-1; l<=1; l+=2) {
        for (int n=-1; n<=1; n+=2) {
          MeshBlockTree* nt = ptree->FindNeighbor(lloc, n, 0, l);
          if (nt != nullptr) {
            if (nt->pleaf_ != nullptr) {  // neighbor at finer level -- requires subblocks
              int ffx = 1 - (n + 1)/2; // 0 for BoundaryFace::outer_x1, 1 for inner_x1
              int ffz = 1 - (l + 1)/2; // 0 for BoundaryFace::outer_x3, 1 for inner_x3
              for (int fy=0; fy<nfy; fy++) {
                MeshBlockTree* nf = nt->GetLeaf(ffx, fy, ffz);
                int inghbr = NeighborIndex(n,0,l,fy,0);
                nghbr.h_view(b,inghbr).gid = nf->gid_;
                nghbr.h_view(b,inghbr).lev = nf->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nf->gid_];
                nghbr.h_view(b,inghbr).dest = NeighborIndex(-n,0,-l,fy,0);
              }
            } else {   // neighbor at same or coarser level -- no subblocks
              int idest,inghbr;
              if (nt->lloc_.level == lloc.level) { // neighbor at same level
                inghbr = NeighborIndex(n,0,l,0,0);
                idest = NeighborIndex(-n,0,-l,0,0);
              } else { // neighbor at coarser level, set indx/dest to appropriate subblock
                inghbr = NeighborIndex(n,0,l,myfx2,0);
                idest = NeighborIndex(-n,0,-l,myfx2,0);
              }
              // only set neighbor for exterior edges of coarser face
              if (nt->lloc_.level >= lloc.level || (myox1 == n && myox3 == l)) {
                nghbr.h_view(b,inghbr).gid = nt->gid_;
                nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
                nghbr.h_view(b,inghbr).dest = idest;
              }
            }
          }
        }
      }

      // neighbors on x2x3 edges
      for (int l=-1; l<=1; l+=2) {
        for (int m=-1; m<=1; m+=2) {
          MeshBlockTree* nt = ptree->FindNeighbor(lloc, 0, m, l);
          if (nt != nullptr) {
            if (nt->pleaf_ != nullptr) {  // neighbor at finer level -- requires subblocks
              int ffy = 1 - (m + 1)/2; // 0 for BoundaryFace::outer_x2, 1 for inner_x2
              int ffz = 1 - (l + 1)/2; // 0 for BoundaryFace::outer_x3, 1 for inner_x3
              for (int fx=0; fx<nfy; fx++) {
                MeshBlockTree* nf = nt->GetLeaf(fx, ffy, ffz);
                int inghbr = NeighborIndex(0,m,l,fx,0);
                nghbr.h_view(b,inghbr).gid = nf->gid_;
                nghbr.h_view(b,inghbr).lev = nf->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nf->gid_];
                nghbr.h_view(b,inghbr).dest = NeighborIndex(0,-m,-l,fx,0);
              }
            } else {   // neighbor at same or coarser level -- no subblocks
              int idest,inghbr;
              if (nt->lloc_.level == lloc.level) { // neighbor at same level
                inghbr = NeighborIndex(0,m,l,0,0);
                idest = NeighborIndex(0,-m,-l,0,0);
              } else { // neighbor at coarser level, set indx/dest to appropriate subblock
                inghbr = NeighborIndex(0,m,l,myfx1,0);
                idest = NeighborIndex(0,-m,-l,myfx1,0);
              }
              // only set neighbor for exterior edges of coarser face
              if (nt->lloc_.level >= lloc.level || (myox2 == m && myox3 == l)) {
                nghbr.h_view(b,inghbr).gid = nt->gid_;
                nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
                nghbr.h_view(b,inghbr).dest = idest;
              }
            }
          }
        }
      }

      // neighbors on corners
      for (int l=-1; l<=1; l+=2) {
        for (int m=-1; m<=1; m+=2) {
          for (int n=-1; n<=1; n+=2) {
            MeshBlockTree* nt = ptree->FindNeighbor(lloc, n, m, l);
            if (nt != nullptr) {
              if (nt->pleaf_ != nullptr) {  // neighbor at finer level
                int ffx = 1 - (n + 1)/2; // 0 for BoundaryFace::outer_x1, 1 for inner_x1
                int ffy = 1 - (m + 1)/2; // 0 for BoundaryFace::outer_x2, 1 for inner_x2
                int ffz = 1 - (l + 1)/2; // 0 for BoundaryFace::outer_x3, 1 for inner_x3
                nt = nt->GetLeaf(ffx, ffy, ffz);
              }
              int nlevel = nt->lloc_.level;
              // only set neighbor for exterior corners of coarser face
              if (nlevel >= lloc.level || (myox1 == n && myox2 == m && myox3 == l)) {
                int inghbr = NeighborIndex(n,m,l,0,0);
                nghbr.h_view(b,inghbr).gid = nt->gid_;
                nghbr.h_view(b,inghbr).lev = nt->lloc_.level;
                nghbr.h_view(b,inghbr).rank = ranklist[nt->gid_];
                nghbr.h_view(b,inghbr).dest = NeighborIndex(-n,-m,-l,0,0);
              }
            }
          }
        }
      }
    }  // end loop over three_d
  }    // end loop over all MeshBlocks

  // For each DualArray: mark host views as modified, and then sync to device array
  nghbr.template modify<HostMemSpace>();
  nghbr.template sync<DevExeSpace>();

  return;
}
