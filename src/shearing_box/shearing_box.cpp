//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//! \brief constructor for ShearingBoxBoundary abstract base class, and utility functions

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! ShearingBoxBoundary base class constructor

ShearingBoxBoundary::ShearingBoxBoundary(MeshBlockPack *ppack, ParameterInput *pin) :
    nmb_x1bndry("nmbx1",2),
    x1bndry_mbgid("x1gid",1,1),
    pmy_pack(ppack) {
  // Create vector with GID of every MBs on this rank at ix1/ox1 shearing-box boundaries
  std::vector<int> tmp_ix1bndry_gid, tmp_ox1bndry_gid;
  auto &mbbcs = ppack->pmb->mb_bcs;
  for (int m=0; m<(ppack->nmb_thispack); ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      tmp_ix1bndry_gid.push_back(m + ppack->gids);
    }
    if (mbbcs.h_view(m,BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      tmp_ox1bndry_gid.push_back(m + ppack->gids);
    }
  }
  // number of MBs at ix1/ox1 boundaries is size of vectors
  nmb_x1bndry(0) = tmp_ix1bndry_gid.size();
  nmb_x1bndry(1) = tmp_ox1bndry_gid.size();

  // allocate mbgid array and initialize GIDs to -1
  int nmb = std::max(nmb_x1bndry(0),nmb_x1bndry(1));
  Kokkos::realloc(x1bndry_mbgid, 2, nmb);
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb; ++m) {
      x1bndry_mbgid.h_view(n,m) = -1;
    }
  }
  // load GIDs of meshblocks at x1 boundaries into DualArray
  for (int m=0; m<nmb_x1bndry(0); ++m) {
    x1bndry_mbgid.h_view(0,m) = tmp_ix1bndry_gid[m];
  }
  for (int m=0; m<nmb_x1bndry(1); ++m) {
    x1bndry_mbgid.h_view(1,m) = tmp_ox1bndry_gid[m];
  }
  // sync with device
  x1bndry_mbgid.template modify<HostMemSpace>();
  x1bndry_mbgid.template sync<DevExeSpace>();


#if MPI_PARALLEL_ENABLED
  // initialize vectors of MPI requests for ix1/ox1 boundaries in fixed length arrays
  // each MB on x1-face can communicate with up to 3 nghbrs
  for (int n=0; n<2; ++n) {
    if (nmb_x1bndry(n) > 0) {
      sendbuf[n].vars_req = new MPI_Request[3*nmb_x1bndry(n)];
      recvbuf[n].vars_req = new MPI_Request[3*nmb_x1bndry(n)];
      for (int m=0; m<nmb_x1bndry(0); ++m) {
        for (int l=0; l<3; ++l) {
          sendbuf[n].vars_req[3*m + l] = MPI_REQUEST_NULL;
          recvbuf[n].vars_req[3*m + l] = MPI_REQUEST_NULL;
        }
      }
    }
  }
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_sbox);
#endif
}

//----------------------------------------------------------------------------------------
// ShearingBoxBoundary base class destructor

ShearingBoxBoundary::~ShearingBoxBoundary() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    if (nmb_x1bndry(n) > 0) {
      delete [] sendbuf[n].vars_req;
      delete [] recvbuf[n].vars_req;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxBoundary::FindTargetMB()
//! \brief  function to find target MB offset by shear.  Returns GID and rank

void ShearingBoxBoundary::FindTargetMB(const int igid, const int jshift, int &gid,
                                       int &rank) {
  Mesh *pm = pmy_pack->pmesh;
  // find lloc of input MB
  LogicalLocation lloc = pm->lloc_eachmb[igid];
  // find number of MBs in x2 direction at this level
  std::int32_t nmbx2 = pm->nmb_rootx2 << (lloc.level - pm->root_level);
  // apply shift by input number of blocks
  lloc.lx2 = static_cast<std::int32_t>((lloc.lx2 + jshift) % nmbx2);
  // find target GID and rank
  gid = (pm->ptree->FindMeshBlock(lloc))->GetGID();
  rank = pm->rank_eachmb[gid];
  return;
}
