//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//  \brief implementation of functions in BoundaryValues class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// BoundaryValues constructor:

BoundaryValues::BoundaryValues(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp)
{
} 
  
//----------------------------------------------------------------------------------------
// BoundaryValues destructor
  
BoundaryValues::~BoundaryValues()
{
}

//----------------------------------------------------------------------------------------
// \!fn void AllocateBuffersCC
// initialize array of send/recv BoundaryBuffers for cell-centered variables
// NOTE: order of array elements is crucial and cannot be changed.  It must match
// order of boundaries in nghbr vector

// TODO: extend for AMR

void BoundaryValues::AllocateBuffersCC(const int nvar)
{
  auto &ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int is = ncells.is, ie = ncells.ie;
  int js = ncells.js, je = ncells.je;
  int ks = ncells.ks, ke = ncells.ke;
  int ng1 = ng-1;
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  for (int n=0; n<nnghbr; ++n) {
  // allocate size of (some) Views
    Kokkos::realloc(send_buf[n].index, 6);
    Kokkos::realloc(recv_buf[n].index, 6);
    Kokkos::realloc(send_buf[n].bcomm_stat, nmb);
    Kokkos::realloc(recv_buf[n].bcomm_stat, nmb);
#if MPI_PARALLEL_ENABLED
    // cannot create Kokkos::View of type MPI_Request so construct STL vector instead
    for (int m=0; m<nmb; ++m) {
      MPI_Request send_req, recv_req;
      send_buf[n].comm_req.push_back(send_req);
      recv_buf[n].comm_req.push_back(recv_req);
    }
#endif
  }

  // initialize buffers for x1 faces
  send_buf[0].InitIndices(nmb, nvar, is,     is+ng1, js, je, ks, ke);
  send_buf[1].InitIndices(nmb, nvar, ie-ng1, ie,     js, je, ks, ke);

  recv_buf[0].InitIndices(nmb, nvar, is-ng, is-1,  js, je, ks, ke);
  recv_buf[1].InitIndices(nmb, nvar, ie+1,  ie+ng, js, je, ks, ke);

  // add more buffers in 2D
  if (nnghbr > 2) {
    // initialize buffers for x2 faces
    send_buf[2].InitIndices(nmb, nvar, is, ie, js,     js+ng1, ks, ke);
    send_buf[3].InitIndices(nmb, nvar, is, ie, je-ng1, je,     ks, ke);

    recv_buf[2].InitIndices(nmb, nvar, is, ie, js-ng, js-1,  ks, ke);
    recv_buf[3].InitIndices(nmb, nvar, is, ie, je+1,  je+ng, ks, ke);

    // initialize buffers for x1x2 edges
    send_buf[4].InitIndices(nmb, nvar, is,     is+ng1, js,     js+ng1, ks, ke);
    send_buf[5].InitIndices(nmb, nvar, ie-ng1, ie,     js,     js+ng1, ks, ke);
    send_buf[6].InitIndices(nmb, nvar, is,     is+ng1, je-ng1, je,     ks, ke);
    send_buf[7].InitIndices(nmb, nvar, ie-ng1, ie,     je-ng1, je,     ks, ke);

    recv_buf[4].InitIndices(nmb, nvar, is-ng, is-1,  js-ng, js-1,  ks, ke);
    recv_buf[5].InitIndices(nmb, nvar, ie+1,  ie+ng, js-ng, js-1,  ks, ke);
    recv_buf[6].InitIndices(nmb, nvar, is-ng, is-1,  je+1,  je+ng, ks, ke);
    recv_buf[7].InitIndices(nmb, nvar, ie+1,  ie+ng, je+1,  je+ng, ks, ke);

    // add more buffers in 3D
    if (nnghbr > 8) {

      // initialize buffers for x3 faces
      send_buf[8].InitIndices(nmb, nvar, is, ie, js, je, ks,     ks+ng1);
      send_buf[9].InitIndices(nmb, nvar, is, ie, js, je, ke-ng1, ke    );
    
      recv_buf[8].InitIndices(nmb, nvar, is, ie, js, je, ks-ng, ks-1 );
      recv_buf[9].InitIndices(nmb, nvar, is, ie, js, je, ke+1,  ke+ng);

      // initialize buffers for x3x1 edges
      send_buf[10].InitIndices(nmb, nvar, is,     is+ng1, js, je, ks,     ks+ng1);
      send_buf[11].InitIndices(nmb, nvar, ie-ng1, ie,     js, je, ks,     ks+ng1);
      send_buf[12].InitIndices(nmb, nvar, is,     is+ng1, js, je, ke-ng1, ke    );
      send_buf[13].InitIndices(nmb, nvar, ie-ng1, ie,     js, je, ke-ng1, ke    );
    
      recv_buf[10].InitIndices(nmb, nvar, is-ng, is-1,  js, je, ks-ng, ks-1 );
      recv_buf[11].InitIndices(nmb, nvar, ie+1,  ie+ng, js, je, ks-ng, ks-1 );
      recv_buf[12].InitIndices(nmb, nvar, is-ng, is-1,  js, je, ke+1,  ke+ng);
      recv_buf[13].InitIndices(nmb, nvar, ie+1,  ie+ng, js, je, ke+1,  ke+ng);

      // initialize buffers for x2x3 edges
      send_buf[14].InitIndices(nmb, nvar, is, ie, js,     js+ng1, ks,     ks+ng1);
      send_buf[15].InitIndices(nmb, nvar, is, ie, je-ng1, je,     ks,     ks+ng1);
      send_buf[16].InitIndices(nmb, nvar, is, ie, js,     js+ng1, ke-ng1, ke    );
      send_buf[17].InitIndices(nmb, nvar, is, ie, je-ng1, je,     ke-ng1, ke    );
  
      recv_buf[14].InitIndices(nmb, nvar, is, ie, js-ng, js-1,  ks-ng, ks-1 );
      recv_buf[15].InitIndices(nmb, nvar, is, ie, je+1,  je+ng, ks-ng, ks-1 );
      recv_buf[16].InitIndices(nmb, nvar, is, ie, js-ng, js-1,  ke+1,  ke+ng);
      recv_buf[17].InitIndices(nmb, nvar, is, ie, je+1,  je+ng, ke+1,  ke+ng);

      // initialize buffers for corners
      send_buf[18].InitIndices(nmb, nvar, is,     is+ng1, js,     js+ng1, ks,     ks+ng1);
      send_buf[19].InitIndices(nmb, nvar, ie-ng1, ie,     js,     js+ng1, ks,     ks+ng1);
      send_buf[20].InitIndices(nmb, nvar, is,     is+ng1, je-ng1, je,     ks,     ks+ng1);
      send_buf[21].InitIndices(nmb, nvar, ie-ng1, ie,     je-ng1, je,     ks,     ks+ng1);
      send_buf[22].InitIndices(nmb, nvar, is,     is+ng1, js,     js+ng1, ke-ng1, ke    );
      send_buf[23].InitIndices(nmb, nvar, ie-ng1, ie,     js,     js+ng1, ke-ng1, ke    );
      send_buf[24].InitIndices(nmb, nvar, is,     is+ng1, je-ng1, je,     ke-ng1, ke    );
      send_buf[25].InitIndices(nmb, nvar, ie-ng1, ie,     je-ng1, je,     ke-ng1, ke    );

      recv_buf[18].InitIndices(nmb, nvar, is-ng, is-1,  js-ng, js-1,  ks-ng, ks-1 );
      recv_buf[19].InitIndices(nmb, nvar, ie+1,  ie+ng, js-ng, js-1,  ks-ng, ks-1 );
      recv_buf[20].InitIndices(nmb, nvar, is-ng, is-1,  je+1,  je+ng, ks-ng, ks-1 );
      recv_buf[21].InitIndices(nmb, nvar, ie+1,  ie+ng, je+1,  je+ng, ks-ng, ks-1 );
      recv_buf[22].InitIndices(nmb, nvar, is-ng, is-1,  js-ng, js-1,  ke+1,  ke+ng);
      recv_buf[23].InitIndices(nmb, nvar, ie+1,  ie+ng, js-ng, js-1,  ke+1,  ke+ng);
      recv_buf[24].InitIndices(nmb, nvar, is-ng, is-1,  je+1,  je+ng, ke+1,  ke+ng);
      recv_buf[25].InitIndices(nmb, nvar, ie+1,  ie+ng, je+1,  je+ng, ke+1,  ke+ng);
    }
  }

  // for index DualArray, mark host views as modified, and then sync to device array
  for (int n=0; n<nnghbr; ++n) {
    send_buf[n].index.template modify<HostMemSpace>();
    recv_buf[n].index.template modify<HostMemSpace>();

    send_buf[n].index.template sync<DevExeSpace>();
    recv_buf[n].index.template sync<DevExeSpace>();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn int BoundaryValues::CreateMPITag(int lid, int bufid, int phys)
//  \brief calculate an MPI tag for boundary buffer communications
//  MPI tag = lid (remaining bits) + bufid (6 bits) + physics(4 bits)
//  Note the convention in Athena++ is lid and bufid are both for the *receiving* process

// WARNING: Generating unsigned integer bitfields from signed integer types and converting
// output to signed integer tags (required by MPI) may lead to unsafe conversions (and
// overflows from built-in types and MPI_TAG_UB).  Note, the MPI standard requires signed
// int tag, with MPI_TAG_UB>= 2^15-1 = 32,767 (inclusive)

int BoundaryValues::CreateMPITag(int lid, int bufid, int phys)
{
  return (lid<<10) | (bufid<<4) | phys;
}
