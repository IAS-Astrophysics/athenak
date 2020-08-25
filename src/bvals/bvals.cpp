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
// MeshBlock constructor: constructs coordinate, boundary condition, hydro, field
//                        and mesh refinement objects.

BoundaryValues::BoundaryValues(Mesh *pm, std::unique_ptr<ParameterInput> &pin, int gid,
  BoundaryFlag *ibcs, int maxvar) : pmesh_(pm), my_mbgid_(gid)
{
  // inheret boundary flags from MeshBlock 
  for (int i=0; i<6; ++i) {bflags[i] = ibcs[i];}

  // calculate sizes and offsets for boundary buffers for cell-centered variables
  // This implementation currently is specific to the 26 boundary buffers in a UNIFORM
  // grid with no adaptive refinement.
  
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int nx1 = pmb->mb_cells.nx1;
  int nx2 = pmb->mb_cells.nx2;
  int nx3 = pmb->mb_cells.nx3;

  cc_bbuf_x1face.SetSize(2,maxvar,nx3,nx2,ng);

  if (pmesh_->nx2gt1) {
    cc_bbuf_x2face.SetSize(2,maxvar,nx3,ng,nx1);
    cc_bbuf_x1x2ed.SetSize(4,maxvar,nx3,ng,ng);
  }

  if (pmesh_->nx3gt1) {
    cc_bbuf_x3face.SetSize(2,maxvar,ng,nx2,nx1);
    cc_bbuf_x3x1ed.SetSize(4,maxvar,ng,nx2,ng);
    cc_bbuf_x2x3ed.SetSize(4,maxvar,ng,ng,nx1);
    cc_bbuf_corner.SetSize(8,maxvar,ng,ng,ng);
  }

std::cout << "Bvals nx1=" << pmb->mb_cells.nx1 << " nx2=" << pmb->mb_cells.nx2 << " nx3=" << pmb->mb_cells.nx3 << std::endl;

}



//----------------------------------------------------------------------------------------
// BoundaryValues destructor

BoundaryValues::~BoundaryValues()
{
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::SendCellCenteredVariables()
// \brief Pack boundary buffers for cell-centered variables, and send to neighbors

void BoundaryValues::SendCellCenteredVariables(AthenaArray<Real> &a, int nvar)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);

std::cout << "nvar=" << nvar << std::endl;

std::cout << "Bval Send nx1=" << pmb->mb_cells.nx1 << " nx2=" << pmb->mb_cells.nx2 << " nx3=" << pmb->mb_cells.nx3 << std::endl;

  // Now send boundary buffer to neighboring MeshBlocks using MPI
  // If neighbor is on same MPI rank, use memcpy()

/**
  for (int nb=0; nb<26; ++nb) {
    if (neighbor[nb].ngid != -1) {
      memcpy(&bbuf_send[offset[nb]], &bbuf_recv[offset[25-nb]], nvar*cc_bbuf_ncells[nb]);
    }
  }
**/

std::cout << "Done send" << std::endl;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::ReceiveCellCenteredVariables()
// \brief Unpack boundary buffers for cell-centered variables.

void BoundaryValues::ReceiveCellCenteredVariables(AthenaArray<Real> &a, int nvar)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);

std::cout << "nx1=" << pmb->mb_cells.nx1 << " nx2=" << pmb->mb_cells.nx2 << " nx3=" << pmb->mb_cells.nx3 << std::endl;

  int ng = pmb->mb_cells.ng;
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;
  int nx1 = pmb->mb_cells.nx1;
  int nx2 = pmb->mb_cells.nx2;
  int nx3 = pmb->mb_cells.nx3;

std::cout << "Bval Receive nx1=" << nx1 << "nx2=" << nx2 << "  nx3=" << nx3 << std::endl;


  // TO DO get pointer to appropriate boundary buffer based on key (input argument)
//  Real *pbbuf = bbuf_send[string];

  // Unpack cell-centered boundary buffer


std::cout << "Done receive" << std::endl;
}


