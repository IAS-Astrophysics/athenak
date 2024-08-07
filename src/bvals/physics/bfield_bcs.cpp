//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bfield_bcs.cpp
//  \brief

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \!fn void BoundaryValues::BFieldBCs()
//! \brief Apply physical boundary conditions for all field variables at faces of MB which
//! are at the edge of the computational domain

void MeshBoundaryValues::BFieldBCs(MeshBlockPack *ppack, DualArray2D<Real> b_in,
                               DvceFaceFld4D<Real> b0) {
  // loop over all MeshBlocks in this MeshBlockPack
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb = ppack->nmb_thispack;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is;
    int &ie = indcs.ie;
    par_for("bfield-bc_x1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      // apply physical boundaries to inner_x1
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            b0.x1f(m,k,j,is-i-1) = -b0.x1f(m,k,j,is+i+1);
            b0.x2f(m,k,j,is-i-1) =  b0.x2f(m,k,j,is+i);
            if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is+i);}
            b0.x3f(m,k,j,is-i-1) =  b0.x3f(m,k,j,is+i);
            if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is+i);}
          }
          break;
        case BoundaryFlag::outflow:
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) {
            b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
            b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
            if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
            b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
            if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
          }
          break;
        case BoundaryFlag::inflow:
          for (int i=0; i<ng; ++i) {
            b0.x1f(m,k,j,is-i-1) = b_in.d_view(IBX,BoundaryFace::inner_x1);
            b0.x2f(m,k,j,is-i-1) = b_in.d_view(IBY,BoundaryFace::inner_x1);
            if (j == n2-1) {
              b0.x2f(m,k,j+1,is-i-1) = b_in.d_view(IBY,BoundaryFace::inner_x1);
            }
            b0.x3f(m,k,j,is-i-1) = b_in.d_view(IBZ,BoundaryFace::inner_x1);
            if (k == n3-1) {
              b0.x3f(m,k+1,j,is-i-1) = b_in.d_view(IBZ,BoundaryFace::inner_x1);
            }
          }
          break;
        default:
          break;
      }

      // apply physical boundaries to outer_x1
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::reflect:
          for (int i=0; i<ng; ++i) {
            b0.x1f(m,k,j,ie+i+2) = -b0.x1f(m,k,j,ie-i);
            b0.x2f(m,k,j,ie+i+1) =  b0.x2f(m,k,j,ie-i);
            if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie-i);}
            b0.x3f(m,k,j,ie+i+1) =  b0.x3f(m,k,j,ie-i);
            if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie-i);}
          }
          break;
        case BoundaryFlag::outflow:
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
          for (int i=0; i<ng; ++i) {
            b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
            b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
            if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
            b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
            if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
          }
          break;
        case BoundaryFlag::inflow:
          for (int i=0; i<ng; ++i) {
            b0.x1f(m,k,j,ie+i+2) = b_in.d_view(IBX,BoundaryFace::outer_x1);
            b0.x2f(m,k,j,ie+i+1) = b_in.d_view(IBY,BoundaryFace::outer_x1);
            if (j == n2-1) {
              b0.x2f(m,k,j+1,ie+i+1) = b_in.d_view(IBY,BoundaryFace::outer_x1);
            }
            b0.x3f(m,k,j,ie+i+1) = b_in.d_view(IBZ,BoundaryFace::outer_x1);
            if (k == n3-1) {
              b0.x3f(m,k+1,j,ie+i+1) = b_in.d_view(IBZ,BoundaryFace::outer_x1);
            }
          }
          break;
        default:
          break;
      }
    });
  }
  if (pm->one_d) return;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    int &js = indcs.js;
    int &je = indcs.je;
    par_for("bfield-bc_x2", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      // apply physical boundaries to inner_x2
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            b0.x1f(m,k,js-j-1,i) =  b0.x1f(m,k,js+j,i);
            if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js+j,i+1);}
            b0.x2f(m,k,js-j-1,i) = -b0.x2f(m,k,js+j+1,i);
            b0.x3f(m,k,js-j-1,i) =  b0.x3f(m,k,js+j,i);
            if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js+j,i);}
          }
          break;
        case BoundaryFlag::outflow:
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) {
            b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
            if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
            b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
            b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
            if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
          }
          break;
        case BoundaryFlag::inflow:
          for (int j=0; j<ng; ++j) {
            b0.x1f(m,k,js-j-1,i) = b_in.d_view(IBX,BoundaryFace::inner_x2);
            if (i == n1-1) {
              b0.x1f(m,k,js-j-1,i+1) = b_in.d_view(IBX,BoundaryFace::inner_x2);
            }
            b0.x2f(m,k,js-j-1,i) = b_in.d_view(IBY,BoundaryFace::inner_x2);
            b0.x3f(m,k,js-j-1,i) = b_in.d_view(IBZ,BoundaryFace::inner_x2);
            if (k == n3-1) {
              b0.x3f(m,k+1,js-j-1,i) = b_in.d_view(IBZ,BoundaryFace::inner_x2);
            }
          }
          break;
        default:
          break;
      }

      // apply physical boundaries to outer_x2
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::reflect:
          for (int j=0; j<ng; ++j) {
            b0.x1f(m,k,je+j+1,i) =  b0.x1f(m,k,je-j,i);
            if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je-j,i+1);}
            b0.x2f(m,k,je+j+2,i) = -b0.x2f(m,k,je-j,i);
            b0.x3f(m,k,je+j+1,i) =  b0.x3f(m,k,je-j,i);
            if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je-j,i);}
          }
          break;
        case BoundaryFlag::outflow:
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
          for (int j=0; j<ng; ++j) {
            b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
            if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
            b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
            b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
            if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
          }
          break;
        case BoundaryFlag::inflow:
          for (int j=0; j<ng; ++j) {
            b0.x1f(m,k,je+j+1,i) = b_in.d_view(IBX,BoundaryFace::outer_x2);
            if (i == n1-1) {
              b0.x1f(m,k,je+j+1,i+1) = b_in.d_view(IBX,BoundaryFace::outer_x2);
            }
            b0.x2f(m,k,je+j+2,i) = b_in.d_view(IBY,BoundaryFace::outer_x2);
            b0.x3f(m,k,je+j+1,i) = b_in.d_view(IBZ,BoundaryFace::outer_x2);
            if (k == n3-1) {
              b0.x3f(m,k+1,je+j+1,i) = b_in.d_view(IBZ,BoundaryFace::outer_x2);
            }
          }
          break;
        default:
          break;
      }
    });
  }
  if (pm->two_d) return;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  par_for("bfield-bc_x3", DevExeSpace(), 0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    // apply physical boundaries to inner_x3
    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) =  b0.x1f(m,ks+k,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks+k,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) =  b0.x2f(m,ks+k,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks+k,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = -b0.x3f(m,ks+k+1,j,i);
        }
        break;
      case BoundaryFlag::outflow:
      case BoundaryFlag::diode:
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
        break;
      case BoundaryFlag::inflow:
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b_in.d_view(IBX,BoundaryFace::inner_x3);
          if (i == n1-1) {
            b0.x1f(m,ks-k-1,j,i+1) = b_in.d_view(IBX,BoundaryFace::inner_x3);
          }
          b0.x2f(m,ks-k-1,j,i) = b_in.d_view(IBY,BoundaryFace::inner_x3);
          if (j == n2-1) {
            b0.x2f(m,ks-k-1,j+1,i) = b_in.d_view(IBY,BoundaryFace::inner_x3);
          }
          b0.x3f(m,ks-k-1,j,i) = b_in.d_view(IBZ,BoundaryFace::inner_x3);
        }
        break;
      default:
        break;
    }

    // apply physical boundaries to outer_x3
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect:
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) =  b0.x1f(m,ke-k,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke-k,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) =  b0.x2f(m,ke-k,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke-k,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = -b0.x3f(m,ke-k,j,i);
        }
        break;
      case BoundaryFlag::outflow:
      case BoundaryFlag::diode:
      case BoundaryFlag::vacuum:
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
        break;
      case BoundaryFlag::inflow:
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b_in.d_view(IBX,BoundaryFace::outer_x3);
          if (i == n1-1) {
            b0.x1f(m,ke+k+1,j,i+1) = b_in.d_view(IBX,BoundaryFace::outer_x3);
          }
          b0.x2f(m,ke+k+1,j,i) = b_in.d_view(IBY,BoundaryFace::outer_x3);
          if (j == n2-1) {
            b0.x2f(m,ke+k+1,j+1,i) = b_in.d_view(IBY,BoundaryFace::outer_x3);
          }
          b0.x3f(m,ke+k+2,j,i) = b_in.d_view(IBZ,BoundaryFace::outer_x3);
        }
        break;
      default:
        break;
    }
  });

  return;
}
