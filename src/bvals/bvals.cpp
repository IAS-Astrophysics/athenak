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
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// BoundaryValues constructor:

BoundaryValues::BoundaryValues(Mesh *pm, ParameterInput *pin, int gid,
  BoundaryFlag *ibcs) : pmesh_(pm), my_mbgid_(gid)
{
  // inheret boundary flags from MeshBlock 
  for (int i=0; i<6; ++i) {bndry_flags[i] = ibcs[i];}
}

//----------------------------------------------------------------------------------------
// BoundaryValues destructor

BoundaryValues::~BoundaryValues()
{
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::AllocateBuffers

void BoundaryValues::AllocateBuffers(int maxvar)
{
  // Allocate memory for send and receive boundary buffers for cell-centered variables.
  // These are stored in 7 different AthenaArrays corresponding to the faces, edges, and
  // corners of a 3D grid.
  // This implementation currently is specific to the 26 boundary buffers in a UNIFORM
  // grid with no adaptive refinement.

  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int nx1 = pmb->mb_cells.nx1;
  int nx2 = pmb->mb_cells.nx2;
  int nx3 = pmb->mb_cells.nx3;

  cc_send_x1face.SetSize(2,maxvar,nx3,nx2,ng);
  cc_recv_x1face.SetSize(2,maxvar,nx3,nx2,ng);

  if (pmesh_->nx2gt1) {
    cc_send_x2face.SetSize(2,maxvar,nx3,ng,nx1);
    cc_send_x1x2ed.SetSize(4,maxvar,nx3,ng,ng);
    cc_recv_x2face.SetSize(2,maxvar,nx3,ng,nx1);
    cc_recv_x1x2ed.SetSize(4,maxvar,nx3,ng,ng);
  }

  if (pmesh_->nx3gt1) {
    cc_send_x3face.SetSize(2,maxvar,ng,nx2,nx1);
    cc_send_x3x1ed.SetSize(4,maxvar,ng,nx2,ng);
    cc_send_x2x3ed.SetSize(4,maxvar,ng,ng,nx1);
    cc_send_corner.SetSize(8,maxvar,ng,ng,ng);
    cc_recv_x3face.SetSize(2,maxvar,ng,nx2,nx1);
    cc_recv_x3x1ed.SetSize(4,maxvar,ng,nx2,ng);
    cc_recv_x2x3ed.SetSize(4,maxvar,ng,ng,nx1);
    cc_recv_corner.SetSize(8,maxvar,ng,ng,ng);
  }
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::ApplyPhysicalBCs()
// \brief Apply physical boundary conditions to faces of MB when they are at the edge of
// the computational domain

TaskStatus BoundaryValues::ApplyPhysicalBCs(Driver* pdrive, int stage)
{
  // apply physical bounaries to inner_x1
  switch (bndry_flags[BoundaryFace::inner_x1]) {
    case BoundaryFlag::reflect:
      ReflectInnerX1();
      break;
    default:
      break;
  }

  // apply physical bounaries to outer_x1
  switch (bndry_flags[BoundaryFace::outer_x1]) {
    case BoundaryFlag::reflect:
      ReflectOuterX1();
      break;
    default:
      break;
  }
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  // apply physical bounaries to inner_x2
  switch (bndry_flags[BoundaryFace::inner_x2]) {
    case BoundaryFlag::reflect:
      ReflectInnerX2();
      break;
    default:
      break;
  }

  // apply physical bounaries to outer_x1
  switch (bndry_flags[BoundaryFace::outer_x2]) {
    case BoundaryFlag::reflect:
      ReflectOuterX2();
      break;
    default:
      break;
  }
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;

  // apply physical bounaries to inner_x3
  switch (bndry_flags[BoundaryFace::inner_x3]) {
    case BoundaryFlag::reflect:
      ReflectInnerX3();
      break;
    default:
      break;
  }

  // apply physical bounaries to outer_x3
  switch (bndry_flags[BoundaryFace::outer_x3]) {
    case BoundaryFlag::reflect:
      ReflectOuterX3();
      break;
    default:
      break;
  }

  return TaskStatus::complete;
}
