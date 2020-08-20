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

BoundaryValues::BoundaryValues(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin,
  BoundaryFlag *input_bcs) {
//  BoundaryFlag *input_bcs) : pmblock_bval_(pmb) {

  pmblock_bval_ = pmb;
  // copy input boundary flags into MeshBlock 
  for (int i=0; i<6; ++i) {mb_bcs[i] = input_bcs[i];}

  // calculate sizes and offsets for boundary buffers for cell-centered variables
  // This implementation currently is specific to the 26 boundary buffers in a UNIFORM
  // grid with no adaptive refinement.
  int ng = pmblock_bval_->mb_cells.ng;
  int nx1 = pmblock_bval_->mb_cells.nx1;
  int nx2 = pmblock_bval_->mb_cells.nx2;
  int nx3 = pmblock_bval_->mb_cells.nx3;

  if (pmblock_bval_->pmesh_mb->nx3gt1) {
    cc_bbuf_ncells[0]  = ng*ng*ng;
    cc_bbuf_ncells[1]  = ng*ng*nx1;
    cc_bbuf_ncells[2]  = ng*ng*ng;
    cc_bbuf_ncells[3]  = ng*ng*nx2;
    cc_bbuf_ncells[4]  = ng*nx1*nx2;
    cc_bbuf_ncells[5]  = ng*ng*nx2;
    cc_bbuf_ncells[6]  = ng*ng*ng;
    cc_bbuf_ncells[7]  = ng*ng*nx1;
    cc_bbuf_ncells[8]  = ng*ng*ng;
    cc_bbuf_ncells[17] = ng*ng*ng;
    cc_bbuf_ncells[18] = ng*ng*nx1;
    cc_bbuf_ncells[19] = ng*ng*ng;
    cc_bbuf_ncells[20] = ng*ng*nx2;
    cc_bbuf_ncells[21] = ng*nx1*nx2;
    cc_bbuf_ncells[22] = ng*ng*nx2;
    cc_bbuf_ncells[23] = ng*ng*ng;
    cc_bbuf_ncells[24] = ng*ng*nx1;
    cc_bbuf_ncells[25] = ng*ng*ng;
  } else {
    for (int n=0; n<=8; ++n) { cc_bbuf_ncells[n]=0; }
    for (int n=17; n<=25; ++n) { cc_bbuf_ncells[n]=0; }
  }
  
  if (pmblock_bval_->pmesh_mb->nx2gt1) {
    cc_bbuf_ncells[9]  = ng*ng*nx3;
    cc_bbuf_ncells[10] = ng*nx1*nx3;
    cc_bbuf_ncells[11] = ng*ng*nx3;
    cc_bbuf_ncells[14] = ng*ng*nx3;
    cc_bbuf_ncells[15] = ng*nx1*nx3;
    cc_bbuf_ncells[16] = ng*ng*nx3;
  } else {
    for (int n=9; n<=11; ++n) { cc_bbuf_ncells[n]=0; }
    for (int n=14; n<=16; ++n) { cc_bbuf_ncells[n]=0; }
  }

  cc_bbuf_ncells[12] = ng*nx2*nx3;
  cc_bbuf_ncells[13] = ng*nx2*nx3;

  cc_bbuf_ncells_offset[0] = 0;
  for (int n=1; n<=25; ++n) {
    cc_bbuf_ncells_offset[n] = cc_bbuf_ncells_offset[n-1] + cc_bbuf_ncells[n-1];
  }
  cc_bbuf_ncells_total = cc_bbuf_ncells_offset[25] + cc_bbuf_ncells[25];

std::cout << "ncells_total= " << cc_bbuf_ncells_total << std::endl;

  // Note: memory for boundary buffers allocated in Hydro class
std::cout << "nx1=" << pmblock_bval_->mb_cells.nx1 << " nx2=" << pmblock_bval_->mb_cells.nx2 << " nx3=" << pmblock_bval_->mb_cells.nx3 << std::endl;

}

//----------------------------------------------------------------------------------------
// BoundaryValues destructor

BoundaryValues::~BoundaryValues() {
}
