//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//! \brief implementation of ShearingBox class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! constructor, initializes data structures and parameters

ShearingBox::ShearingBox(MeshBlockPack *ppack, ParameterInput *pin, int nvar) :
    qshear(0.0),
    omega0(0.0),
    maxjshift(1),
    nmb_x1bndry(0),
    shearing_box_r_phi(false),
    pmy_pack(ppack) {
  // read shear parameters
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");

  // estimate maximum integer shift in x2-direction for orbital advection
  Real xmin = fabs(ppack->pmesh->mesh_size.x1min);
  Real xmax = fabs(ppack->pmesh->mesh_size.x1max);
  maxjshift = static_cast<int>((ppack->pmesh->cfl_no)*std::max(xmin,xmax)) + 1;

  // --- Step 1.  Initialize data for orbital advection
  {
    int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
    auto &indcs = ppack->pmesh->mb_indcs;
    int ncells3 = indcs.nx3;
    int ncells2 = indcs.ng + maxjshift;
    int ncells1 = indcs.nx1;
    // cell-centered data
    for (int n=0; n<2; ++n) {
      Kokkos::realloc(sendcc_orb[n].vars,nmb,(nvar*ncells3*ncells2*ncells1));
      Kokkos::realloc(recvcc_orb[n].vars,nmb,(nvar*ncells3*ncells2*ncells1));
    }
    // face-centered data
    for (int n=0; n<2; ++n) {
      Kokkos::realloc(sendfc_orb[n].vars,nmb,(2*(ncells3+1)*ncells2*(ncells1+1)));
      Kokkos::realloc(recvfc_orb[n].vars,nmb,(2*(ncells3+1)*ncells2*(ncells1+1)));
    }
  }
#if MPI_PARALLEL_ENABLED
  // For orbital advection, communication is only with x2-face neighbors
  // initialize vectors of MPI request in 2 elements of fixed length arrays
  for (int n=0; n<2; ++n) {
    int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
    sendcc_orb[n].vars_req = new MPI_Request[nmb];
    sendfc_orb[n].vars_req = new MPI_Request[nmb];
    recvcc_orb[n].vars_req = new MPI_Request[nmb];
    recvfc_orb[n].vars_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendcc_orb[n].vars_req[m] = MPI_REQUEST_NULL;
      sendfc_orb[n].vars_req[m] = MPI_REQUEST_NULL;
      recvcc_orb[n].vars_req[m] = MPI_REQUEST_NULL;
      recvfc_orb[n].vars_req[m] = MPI_REQUEST_NULL;
    }
  }
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_sbox);
#endif

  // --- Step 2.  Initialize data for shearing sheet BCs
  // Work out how many MBs on this rank are at x1 shearing-box boundaries
  std::vector<std::pair<int,int>> cnt_x1bndry_mbs;
  auto &mbbcs = ppack->pmb->mb_bcs;
  for (int m=0; m<(ppack->nmb_thispack); ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      cnt_x1bndry_mbs.push_back(std::make_pair(0,(m+ppack->gids)));
    }
    if (mbbcs.h_view(m,BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      cnt_x1bndry_mbs.push_back(std::make_pair(1,(m+ppack->gids)));
    }
  }
  nmb_x1bndry = cnt_x1bndry_mbs.size();

  // load GIDs of meshblocks at boundaries into DualArray
  Kokkos::realloc(x1bndry_mbs, nmb_x1bndry);
  for (int m=0; m<nmb_x1bndry; ++m) {
    x1bndry_mbs.h_view(m,0) = cnt_x1bndry_mbs[m].first;
    x1bndry_mbs.h_view(m,1) = cnt_x1bndry_mbs[m].second;
  }
  // sync device array
  x1bndry_mbs.template modify<HostMemSpace>();
  x1bndry_mbs.template sync<DevExeSpace>();

  // Now allocate send/recv buffers for shearing box BCs
  {
    auto &indcs = ppack->pmesh->mb_indcs;
    int ncells3 = indcs.nx3 + 2*indcs.ng;
    int ncells2 = indcs.nx2 + 2*indcs.ng;
    int ncells1 = indcs.ng;
    // cell-centered data
    Kokkos::realloc(sendcc_shr.vars,nmb_x1bndry,(nvar*ncells3*ncells2*ncells1));
    Kokkos::realloc(recvcc_shr.vars,nmb_x1bndry,(nvar*ncells3*ncells2*ncells1));
    // face-centered data
    Kokkos::realloc(sendfc_shr.vars,nmb_x1bndry,(3*(ncells3+1)*(ncells2+1)*ncells1));
    Kokkos::realloc(recvfc_shr.vars,nmb_x1bndry,(3*(ncells3+1)*(ncells2+1)*ncells1));
  }
}

//----------------------------------------------------------------------------------------
// destructor

ShearingBox::~ShearingBox() {
}
