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
// \!fn void AllocateBuffersCC
// construct vector of BoundaryBuffers for cell-centered variables
// NOTE: order of vector elements is crucial and cannot be changed.  It must match
// order of boundaries in nghbr vector

// TODO: do not allocate memory for send buffers when target on same MPI rank
// TODO: with AMR, indices for buffers can be different on different MashBlocks

void AllocateBuffersCCVars(const int nvar, const RegionCells ncells,
  std::vector<BoundaryBuffer> &send_buf, std::vector<BoundaryBuffer> &recv_buf)
{
  const int &ng = ncells.ng;
  const int &is = ncells.is;
  const int &ie = ncells.ie;
  const int &js = ncells.js;
  const int &je = ncells.je;
  const int &ks = ncells.ks;
  const int &ke = ncells.ke;
  int ng1 = ng-1;

  // x1 faces
  send_buf.emplace_back(nvar, is,     is+ng1, js, je, ks, ke);
  send_buf.emplace_back(nvar, ie-ng1, ie,     js, je, ks, ke);

  recv_buf.emplace_back(nvar, is-ng, is-1,  js, je, ks, ke);
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, js, je, ks, ke);

  if (ncells.nx2 == 1) {return;}  // stop if 1D
   
  // x2 faces
  send_buf.emplace_back(nvar, is, ie, js,     js+ng1, ks, ke);
  send_buf.emplace_back(nvar, is, ie, je-ng1, je,     ks, ke);

  recv_buf.emplace_back(nvar, is, ie, js-ng, js-1,  ks, ke);
  recv_buf.emplace_back(nvar, is, ie, je+1,  je+ng, ks, ke);

  // x1x2 edges
  send_buf.emplace_back(nvar, is,     is+ng1, js,     js+ng1, ks, ke);
  send_buf.emplace_back(nvar, ie-ng1, ie,     js,     js+ng1, ks, ke);
  send_buf.emplace_back(nvar, is,     is+ng1, je-ng1, je,     ks, ke);
  send_buf.emplace_back(nvar, ie-ng1, ie,     je-ng1, je,     ks, ke);

  recv_buf.emplace_back(nvar, is-ng, is-1,  js-ng, js-1,  ks, ke);
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, js-ng, js-1,  ks, ke);
  recv_buf.emplace_back(nvar, is-ng, is-1,  je+1,  je+ng, ks, ke);
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, je+1,  je+ng, ks, ke);

  if (ncells.nx3 == 1) {return;}  // stop if 2D

  // x3 faces
  send_buf.emplace_back(nvar, is, ie, js, je, ks,     ks+ng1);
  send_buf.emplace_back(nvar, is, ie, js, je, ke-ng1, ke    );

  recv_buf.emplace_back(nvar, is, ie, js, je, ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, is, ie, js, je, ke+1,  ke+ng);

  // x3x1 edges
  send_buf.emplace_back(nvar, is,     is+ng1, js, je, ks,     ks+ng1);
  send_buf.emplace_back(nvar, ie-ng1, ie,     js, je, ks,     ks+ng1);
  send_buf.emplace_back(nvar, is,     is+ng1, js, je, ke-ng1, ke    );
  send_buf.emplace_back(nvar, ie-ng1, ie,     js, je, ke-ng1, ke    );

  recv_buf.emplace_back(nvar, is-ng, is-1,  js, je, ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, js, je, ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, is-ng, is-1,  js, je, ke+1,  ke+ng);
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, js, je, ke+1,  ke+ng);

  // x2x3 edges
  send_buf.emplace_back(nvar, is, ie, js,     js+ng1, ks,     ks+ng1);
  send_buf.emplace_back(nvar, is, ie, je-ng1, je,     ks,     ks+ng1);
  send_buf.emplace_back(nvar, is, ie, js,     js+ng1, ke-ng1, ke    );
  send_buf.emplace_back(nvar, is, ie, je-ng1, je,     ke-ng1, ke    );

  recv_buf.emplace_back(nvar, is, ie, js-ng, js-1,  ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, is, ie, je+1,  je+ng, ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, is, ie, js-ng, js-1,  ke+1,  ke+ng);
  recv_buf.emplace_back(nvar, is, ie, je+1,  je+ng, ke+1,  ke+ng);
  
  // corners
  send_buf.emplace_back(nvar, is,     is+ng1, js,     js+ng1, ks,     ks+ng1);
  send_buf.emplace_back(nvar, ie-ng1, ie,     js,     js+ng1, ks,     ks+ng1);
  send_buf.emplace_back(nvar, is,     is+ng1, je-ng1, je,     ks,     ks+ng1);
  send_buf.emplace_back(nvar, ie-ng1, ie,     je-ng1, je,     ks,     ks+ng1);
  send_buf.emplace_back(nvar, is,     is+ng1, js,     js+ng1, ke-ng1, ke    );
  send_buf.emplace_back(nvar, ie-ng1, ie,     js,     js+ng1, ke-ng1, ke    );
  send_buf.emplace_back(nvar, is,     is+ng1, je-ng1, je,     ke-ng1, ke    );
  send_buf.emplace_back(nvar, ie-ng1, ie,     je-ng1, je,     ke-ng1, ke    );

  recv_buf.emplace_back(nvar, is-ng, is-1,  js-ng, js-1,  ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, js-ng, js-1,  ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, is-ng, is-1,  je+1,  je+ng, ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, je+1,  je+ng, ks-ng, ks-1 );
  recv_buf.emplace_back(nvar, is-ng, is-1,  js-ng, js-1,  ke+1,  ke+ng);
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, js-ng, js-1,  ke+1,  ke+ng);
  recv_buf.emplace_back(nvar, is-ng, is-1,  je+1,  je+ng, ke+1,  ke+ng);
  recv_buf.emplace_back(nvar, ie+1,  ie+ng, je+1,  je+ng, ke+1,  ke+ng);

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

int CreateMPITag(int lid, int bufid, int phys)
{
  return (lid<<10) | (bufid<<4) | phys;
}
