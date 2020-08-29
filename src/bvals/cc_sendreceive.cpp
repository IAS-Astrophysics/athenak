//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cc_sendreceive.cpp
//  \brief implementation of functions in BoundaryValues class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::SendCellCenteredVariables()
// \brief Pack boundary buffers for cell-centered variables, and send to neighbors

TaskStatus BoundaryValues::SendCellCenteredVariables(AthenaArray<Real> &a, int nvar)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int is = pmb->mb_cells.is, ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js, je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks, ke = pmb->mb_cells.ke;
  int nx1 = pmb->mb_cells.nx1;
  int nx2 = pmb->mb_cells.nx2;
  int nx3 = pmb->mb_cells.nx3;

  // load buffers, NO AMR

  for (int n=0; n<nvar; ++n) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      // 2D slice in bottom two cells in k-direction
      if (pmesh_->nx3gt1 && k<(ks+ng)) {
        if (pmesh_->nx2gt1 && j<(js+ng)) {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_corner(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x3x1ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1x2ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x2x3ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x3face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x2face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_corner(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x3x1ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1x2ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
          }
  
        } else if (pmesh_->nx2gt1 && j>=nx2) {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x3x1ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_corner(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1x2ed(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x3face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x2x3ed(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            cc_send_x2face(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x3x1ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_corner(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1x2ed(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
          }
  
        } else {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x3x1ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x3face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x3x1ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
          }
        }

      // 2D slice in top two cells in k-direction
      } else if (pmesh_->nx3gt1 && k>=nx3) {
        if (pmesh_->nx2gt1 && j<(js+ng)) {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x1x2ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_corner(4,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x3x1ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x2face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x2x3ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x3face(1,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x1x2ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_corner(5,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x3x1ed(3,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
          }

        } else if (pmesh_->nx2gt1 && j>=nx2) {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1x2ed(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            cc_send_x3x1ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            cc_send_corner(6,n,k-nx3,j-nx2,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x2face(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
            cc_send_x3face(1,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x2x3ed(3,n,k-nx3,j-nx2,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1x2ed(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
            cc_send_x3x1ed(3,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_corner(7,n,k-nx3,j-nx2,i-nx1) = a(n,k,j,i);
          }

        } else {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x3x1ed(2,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x3face(1,n,k-nx3,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x3x1ed(3,n,k-nx3,j-js ,i-nx1) = a(n,k,j,i);
          }
        }

      // 2D slice in middle of grid
      } else {
        if (pmesh_->nx2gt1 && j<(js+ng)) {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x1x2ed(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x2face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x1x2ed(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
          }

        } else if (pmesh_->nx2gt1 && j>=nx2) {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x1face(0,n,k-ks ,j-js ,i-is ) = a(n,k,j,i);
            cc_send_x1x2ed(2,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            cc_send_x2face(1,n,k-ks ,j-nx2,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x1face(1,n,k-ks ,j-js ,i-nx1) = a(n,k,j,i);
            cc_send_x1x2ed(3,n,k-ks ,j-nx2,i-nx1) = a(n,k,j,i);
          }

        } else {
          for (int i=is; i<(is+ng); ++i) {
            cc_send_x1face(0,n,k-ks ,j-js,i-is ) = a(n,k,j,i);
          }
          for (int i=nx1; i<=ie; ++i) {
            cc_send_x1face(1,n,k-ks ,j-js,i-nx1) = a(n,k,j,i);
          }
        }
      }
    }

  }}  // end loops over n,k

  // Now send boundary buffer to neighboring MeshBlocks using MPI
  // If neighbor is on same MPI rank, just copy data
  // TODO add MPI sends
  // TODO get working for multiple meshblocks

  // copy x1 faces
  int ndata = nvar*nx3*nx2*ng;
  for (int n=0; n<2; ++n) {
    Real *psrc = &(cc_send_x1face(1-n,0,0,0,0));
    Real *pdest = &(cc_recv_x1face(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }
  if (!(pmesh_->nx2gt1)) return TaskStatus::complete;

  // copy x2 faces and x1x2 edges
  ndata = nvar*nx3*ng*nx1;
  for (int n=0; n<2; ++n) {
    Real *psrc = &(cc_send_x2face(1-n,0,0,0,0));
    Real *pdest = &(cc_recv_x2face(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }
  ndata = nvar*nx3*ng*ng;
  for (int n=0; n<4; ++n) {
    Real *psrc = &(cc_send_x1x2ed(3-n,0,0,0,0));
    Real *pdest = &(cc_recv_x1x2ed(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }
  if (!(pmesh_->nx3gt1)) return TaskStatus::complete;
  
  // copy x3 faces, x3x1 and x2x3 edges, and corners
  ndata = nvar*ng*nx2*nx1;
  for (int n=0; n<2; ++n) {
    Real *psrc = &(cc_send_x3face(1-n,0,0,0,0));
    Real *pdest = &(cc_recv_x3face(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }
  ndata = nvar*ng*nx2*ng;
  for (int n=0; n<4; ++n) {
    Real *psrc = &(cc_send_x3x1ed(3-n,0,0,0,0));
    Real *pdest = &(cc_recv_x3x1ed(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }
  ndata = nvar*ng*ng*nx1;
  for (int n=0; n<4; ++n) {
    Real *psrc = &(cc_send_x2x3ed(3-n,0,0,0,0));
    Real *pdest = &(cc_recv_x2x3ed(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }
  ndata = nvar*ng*ng*ng;
  for (int n=0; n<8; ++n) {
    Real *psrc = &(cc_send_corner(7-n,0,0,0,0));
    Real *pdest = &(cc_recv_corner(n,0,0,0,0));
    memcpy(pdest, psrc, ndata*sizeof(Real));
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::ReceiveCellCenteredVariables()
// \brief Unpack boundary buffers for cell-centered variables.

TaskStatus BoundaryValues::ReceiveCellCenteredVariables(AthenaArray<Real> &a, int nvar)
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);

  int ng = pmb->mb_cells.ng;
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;

// unpack, NO AMR

  for (int n=0; n<nvar; ++n) {
  for (int k=0; k<ncells3; ++k) {
    for (int j=0; j<ncells2; ++j) {

      // 2D slice in bottom two cells in k-direction
      if (pmesh_->nx3gt1 && k<ks) {
        if (pmesh_->nx2gt1 && j<js) {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_corner(0,n,k,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x2x3ed(0,n,k,j,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_corner(1,n,k,j,i);
          }

        } else if (pmesh_->nx2gt1 && j>je) {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_corner(2,n,k,j-je-1,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x2x3ed(1,n,k,j-je-1,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_corner(3,n,k,j-je-1,i);
          }

        } else {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_x3x1ed(0,n,k,j-js,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x3face(0,n,k,j-js,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_x3x1ed(1,n,k,j-js,i);
          }
        }

      // 2D slice in top two cells in k-direction
      } else if (pmesh_->nx3gt1 && k>ke) {
        if (pmesh_->nx2gt1 && j<js) {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_corner(4,n,k-ke-1,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x2x3ed(2,n,k-ke-1,j,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_corner(5,n,k-ke-1,j,i);
          }

        } else if (pmesh_->nx2gt1 && j>je) {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_corner(6,n,k-ke-1,j-je-1,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x2x3ed(3,n,k-ke-1,j-je-1,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_corner(7,n,k-ke-1,j-je-1,i);
          }

        } else {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_x3x1ed(2,n,k-ke-1,j-js,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x3face(1,n,k-ke-1,j-js,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_x3x1ed(3,n,k-ke-1,j-js,i);
          }
        }

      // 2D slice in middle of grid
      } else {
        if (pmesh_->nx2gt1 && j<js) {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_x1x2ed(0,n,k-ks,j,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x2face(0,n,k-ks,j,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_x1x2ed(1,n,k,j,i);
          }

        } else if (pmesh_->nx2gt1 && j>je) {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_x1x2ed(2,n,k-ks,j-je-1,i);
          }
          for (int i=is; i<=ie; ++i) {
            a(n,k,j,i) = cc_recv_x2face(1,n,k-ks,j-je-1,i-is);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_x1x2ed(3,n,k-ks,j-je-1,i);
          }

        } else {
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i) = cc_recv_x1face(0,n,k-ks,j-js,i);
          }
          for (int i=0; i<ng; ++i) {
            a(n,k,j,i+ie+1) = cc_recv_x1face(1,n,k-ks,j-js,i);
          }
        }
      }
    }

  }}  // end loops over n,k

  return TaskStatus::complete;
}
