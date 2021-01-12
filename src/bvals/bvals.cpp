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

BoundaryValues::BoundaryValues(Mesh *pm, ParameterInput *pin, int gid,
  BoundaryFlag *ibcs) : pmesh_(pm), my_mbgid_(gid)
{
  // inheret boundary flags from MeshBlock 
  for (int i=0; i<6; ++i) {bndry_flag[i] = ibcs[i];}
}

//----------------------------------------------------------------------------------------
// BoundaryValues destructor

BoundaryValues::~BoundaryValues()
{
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::AllocateBuffersCC

void BoundaryValues::AllocateBuffersCC(const RegionCells ncells, const int nvar,
  std::vector<BoundaryBuffer> &send_buf, std::vector<BoundaryBuffer> &recv_buf)
{
  // construct vector of BoundaryBuffers for cell-centered variables
  // NOTE: order of vector elements is crucial and cannot be changed.  It must match
  // order of boundaries in nghbr vector
  // TODO: do not allocate memory for send buffers when target on same MPI rank
  const int &ng = ncells.ng;
  const int &is = ncells.is;
  const int &ie = ncells.ie;
  const int &js = ncells.js;
  const int &je = ncells.je;
  const int &ks = ncells.ks;
  const int &ke = ncells.ke;
  int ng1 = ng-1;
  int nnghbr = nghbr.size();

  // x1 faces
  send_buf.emplace_back(nvar, is,    is+ng1, js, je, ks, ke);
  send_buf.emplace_back(nvar, ie-ng1, ie,    js, je, ks, ke);

  recv_buf.emplace_back(nvar, is-ng, is-1,   js, je, ks, ke);
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, js, je, ks, ke);

  if (ncells.nx2 == 1) {return;}  // stop if 1D
   
  // x2 faces
  send_buf.emplace_back(nvar, is, ie, js,    js+ng1, ks, ke);
  send_buf.emplace_back(nvar, is, ie, je-ng1, je,    ks, ke);

  recv_buf.emplace_back(nvar, is, ie, js-ng, js-1,   ks, ke);
  recv_buf.emplace_back(nvar, is, ie, je+1,   je+ng, ks, ke);

  // x1x2 edges
  send_buf.emplace_back(nvar, is,    is+ng1, js,    js+ng1, ks, ke);
  send_buf.emplace_back(nvar, ie-ng1, ie,    js,     js+ng1, ks, ke);
  send_buf.emplace_back(nvar, is,    is+ng1, je-ng1, je,    ks, ke);
  send_buf.emplace_back(nvar, ie-ng1, ie,     je-ng1, je,    ks, ke);

  recv_buf.emplace_back(nvar, is-ng, is-1,   js-ng, js-1,   ks, ke);
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, js-ng,  js-1,   ks, ke);
  recv_buf.emplace_back(nvar, is-ng, is-1,   je+1,   je+ng, ks, ke);
  recv_buf.emplace_back(nvar, ie+1,   ie+ng,  je+1,   je+ng, ks, ke);

  if (ncells.nx3 == 1) {return;}  // stop if 2D

  // x3 faces
  send_buf.emplace_back(nvar, is, ie, js, je, ks,    ks+ng1);
  send_buf.emplace_back(nvar, is, ie, js, je, ke-ng1, ke   );

  recv_buf.emplace_back(nvar, is, ie, js, je, ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, is, ie, js, je, ke+1,   ke+ng);

  // x3x1 edges
  send_buf.emplace_back(nvar, is,    is+ng1, js, je, ks,    ks+ng1);
  send_buf.emplace_back(nvar, ie-ng1, ie,    js, je, ks,    ks+ng1);
  send_buf.emplace_back(nvar, is,    is+ng1, js, je, ke-ng1, ke   );
  send_buf.emplace_back(nvar, ie-ng1, ie,    js, je, ke-ng1, ke   );

  recv_buf.emplace_back(nvar, is-ng, is-1,   js, je, ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, js, je, ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, is-ng, is-1,   js, je, ke+1,   ke+ng);
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, js, je, ke+1,   ke+ng);

  // x2x3 edges
  send_buf.emplace_back(nvar, is, ie, js,    js+ng1, ks,    ks+ng1);
  send_buf.emplace_back(nvar, is, ie, je-ng1, je,    ks,    ks+ng1);
  send_buf.emplace_back(nvar, is, ie, js,    js+ng1, ke-ng1, ke   );
  send_buf.emplace_back(nvar, is, ie, je-ng1, je,    ke-ng1, ke   );

  recv_buf.emplace_back(nvar, is, ie, js-ng, js-1,   ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, is, ie, je+1,   je+ng, ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, is, ie, js-ng, js-1,   ke+1,   ke+ng);
  recv_buf.emplace_back(nvar, is, ie, je+1,   je+ng, ke+1,   ke+ng);
  
  // corners
  send_buf.emplace_back(nvar, is,    is+ng1, js,    js+ng1, ks,    ks+ng1);
  send_buf.emplace_back(nvar, ie-ng1, ie,    js,    js+ng1, ks,    ks+ng1);
  send_buf.emplace_back(nvar, is,    is+ng1, je-ng1, je,    ks,    ks+ng1);
  send_buf.emplace_back(nvar, ie-ng1, ie,    je-ng1, je,    ks,    ks+ng1);
  send_buf.emplace_back(nvar, is,    is+ng1, js,    js+ng1, ke-ng1, ke   );
  send_buf.emplace_back(nvar, ie-ng1, ie,    js,    js+ng1, ke-ng1, ke   );
  send_buf.emplace_back(nvar, is,    is+ng1, je-ng1, je,    ke-ng1, ke   );
  send_buf.emplace_back(nvar, ie-ng1, ie,    je-ng1, je,    ke-ng1, ke   );

  recv_buf.emplace_back(nvar, is-ng, is-1,   js-ng, js-1,   ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, js-ng, js-1,   ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, is-ng, is-1,   je+1,   je+ng, ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, je+1,   je+ng, ks-ng, ks-1  );
  recv_buf.emplace_back(nvar, is-ng, is-1,   js-ng, js-1,   ke+1,   ke+ng);
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, js-ng, js-1,   ke+1,   ke+ng);
  recv_buf.emplace_back(nvar, is-ng, is-1,   je+1,   je+ng, ke+1,   ke+ng);
  recv_buf.emplace_back(nvar, ie+1,   ie+ng, je+1,   je+ng, ke+1,   ke+ng);

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::ApplyPhysicalBCs()
// \brief Apply physical boundary conditions to faces of MB when they are at the edge of
// the computational domain

TaskStatus BoundaryValues::ApplyPhysicalBCs(Driver* pdrive, int stage)
{
  // apply physical bounaries to inner_x1
  switch (bndry_flag[BoundaryFace::inner_x1]) {
    case BoundaryFlag::reflect:
      ReflectInnerX1();
      break;
    case BoundaryFlag::outflow:
      OutflowInnerX1();
      break;
    default:
      break;
  }

  // apply physical bounaries to outer_x1
  switch (bndry_flag[BoundaryFace::outer_x1]) {
    case BoundaryFlag::reflect:
      ReflectOuterX1();
      break;
    case BoundaryFlag::outflow:
      OutflowOuterX1();
      break;
    default:
      break;
  }
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  // apply physical bounaries to inner_x2
  switch (bndry_flag[BoundaryFace::inner_x2]) {
    case BoundaryFlag::reflect:
      ReflectInnerX2();
      break;
    case BoundaryFlag::outflow:
      OutflowInnerX2();
      break;
    default:
      break;
  }

  // apply physical bounaries to outer_x1
  switch (bndry_flag[BoundaryFace::outer_x2]) {
    case BoundaryFlag::reflect:
      ReflectOuterX2();
      break;
    case BoundaryFlag::outflow:
      OutflowOuterX2();
      break;
    default:
      break;
  }
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;

  // apply physical bounaries to inner_x3
  switch (bndry_flag[BoundaryFace::inner_x3]) {
    case BoundaryFlag::reflect:
      ReflectInnerX3();
      break;
    case BoundaryFlag::outflow:
      OutflowInnerX3();
      break;
    default:
      break;
  }

  // apply physical bounaries to outer_x3
  switch (bndry_flag[BoundaryFace::outer_x3]) {
    case BoundaryFlag::reflect:
      ReflectOuterX3();
      break;
    case BoundaryFlag::outflow:
      OutflowOuterX3();
      break;
    default:
      break;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn int BoundaryValues::CreateMPItag(int lid, int bufid, int phys)
//  \brief calculate an MPI tag for boundary buffer communications
//  MPI tag = lid (remaining bits) + bufid (6 bits) + physics(4 bits)
//  Note the convention in Athena++ is lid and bufid are both for the *receiving* process

// WARN: The below procedure of generating unsigned integer bitfields from signed integer
// types and converting output to signed integer tags (required by MPI) is tricky and may
// lead to unsafe conversions (and overflows from built-in types and MPI_TAG_UB).  Note,
// the MPI standard requires signed int tag, with MPI_TAG_UB>= 2^15-1 = 32,767 (inclusive)

int BoundaryValues::CreateMPItag(int lid, int bufid, int phys)
{
  return (lid<<10) | (bufid<<4) | phys;
}
