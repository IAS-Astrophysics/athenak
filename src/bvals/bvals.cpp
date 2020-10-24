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
#include "bvals.hpp"
#include "mesh/mesh.hpp"

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
// \!fn void BoundaryValues::AllocateBuffers

void BoundaryValues::AllocateBuffers(BBuffer &bbuf, const int maxvar)
{
  // Allocate memory for send and receive boundary buffers, and initialize 
  // BoundaryRecvStatus flags.
  // The buffers are stored in 7 different AthenaArrays corresponding to the faces, edges,
  // and corners of a 3D grid.  The flags are also stored in 7 arrays.
  // This implementation currently is specific to the 26 boundary buffers in a UNIFORM
  // grid with no adaptive refinement.

  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int nx1 = pmb->mb_cells.nx1;
  int nx2 = pmb->mb_cells.nx2;
  int nx3 = pmb->mb_cells.nx3;

  Kokkos::realloc(bbuf.send_x1face,2,maxvar,nx3,nx2,ng);
  Kokkos::realloc(bbuf.recv_x1face,2,maxvar,nx3,nx2,ng);

  if (pmesh_->nx2gt1) {
    Kokkos::realloc(bbuf.send_x2face,2,maxvar,nx3,ng,nx1);
    Kokkos::realloc(bbuf.send_x1x2ed,4,maxvar,nx3,ng,ng);
    Kokkos::realloc(bbuf.recv_x2face,2,maxvar,nx3,ng,nx1);
    Kokkos::realloc(bbuf.recv_x1x2ed,4,maxvar,nx3,ng,ng);
  }

  if (pmesh_->nx3gt1) {
    Kokkos::realloc(bbuf.send_x3face,2,maxvar,ng,nx2,nx1);
    Kokkos::realloc(bbuf.send_x3x1ed,4,maxvar,ng,nx2,ng);
    Kokkos::realloc(bbuf.send_x2x3ed,4,maxvar,ng,ng,nx1);
    Kokkos::realloc(bbuf.send_corner,8,maxvar,ng,ng,ng);
    Kokkos::realloc(bbuf.recv_x3face,2,maxvar,ng,nx2,nx1);
    Kokkos::realloc(bbuf.recv_x3x1ed,4,maxvar,ng,nx2,ng);
    Kokkos::realloc(bbuf.recv_x2x3ed,4,maxvar,ng,ng,nx1);
    Kokkos::realloc(bbuf.recv_corner,8,maxvar,ng,ng,ng);
  }

  // initialize all boundary status arrays to undef
  for (int i=0; i<2; ++i) {
    bbuf.bstat_x1face[i] = BoundaryRecvStatus::undef;
    bbuf.bstat_x2face[i] = BoundaryRecvStatus::undef;
    bbuf.bstat_x3face[i] = BoundaryRecvStatus::undef;
  }
  for (int i=0; i<4; ++i) {
    bbuf.bstat_x1x2ed[i] = BoundaryRecvStatus::undef;
    bbuf.bstat_x3x1ed[i] = BoundaryRecvStatus::undef;
    bbuf.bstat_x2x3ed[i] = BoundaryRecvStatus::undef;
  }
  for (int i=0; i<8; ++i) {
    bbuf.bstat_corner[i] = BoundaryRecvStatus::undef;
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
