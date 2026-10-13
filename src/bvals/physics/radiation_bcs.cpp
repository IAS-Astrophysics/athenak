//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file radiation_bcs.cpp
//  \brief

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \!fn void BoundaryValues::RadiationBCs()
//! \brief Apply physical boundary conditions for radiation at faces of MB which
//! are at the edge of the computational domain

void MeshBoundaryValues::RadiationBCs(MeshBlockPack *ppack, DualArray2D<Real> i_in,
                                      DvceArray5D<Real> i0) {
  // loop over all MeshBlocks in this MeshBlockPack
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = ppack->pmb->mb_bcs;

  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nvar = i0.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nmb = ppack->nmb_thispack;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is;
    int &ie = indcs.ie;
    par_for("radiationbc_x1", DevExeSpace(), 0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      // apply physical boundaries to inner_x1
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::outflow:
          for (int i=0; i<ng; ++i) {
            i0(m,n,k,j,is-i-1) = i0(m,n,k,j,is);
          }
          break;
        case BoundaryFlag::inflow:
          for (int i=0; i<ng; ++i) {
            i0(m,n,k,j,is-i-1) = i_in.d_view(n,BoundaryFace::inner_x1);
          }
          break;
        default:
          break;
      }

      // apply physical boundaries to outer_x1
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::outflow:
          for (int i=0; i<ng; ++i) {
            i0(m,n,k,j,ie+i+1) = i0(m,n,k,j,ie);
          }
          break;
        case BoundaryFlag::inflow:
          for (int i=0; i<ng; ++i) {
            i0(m,n,k,j,ie+i+1) = i_in.d_view(n,BoundaryFace::outer_x1);
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
    par_for("radiationbc_x2", DevExeSpace(), 0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      // apply physical boundaries to inner_x2
      switch (mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::outflow:
          for (int j=0; j<ng; ++j) {
            i0(m,n,k,js-j-1,i) = i0(m,n,k,js,i);
          }
          break;
        case BoundaryFlag::inflow:
          for (int j=0; j<ng; ++j) {
            i0(m,n,k,js-j-1,i) = i_in.d_view(n,BoundaryFace::inner_x2);
          }
          break;
        default:
          break;
      }

      // apply physical boundaries to outer_x2
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::outflow:
          for (int j=0; j<ng; ++j) {
            i0(m,n,k,je+j+1,i) = i0(m,n,k,je,i);
          }
          break;
        case BoundaryFlag::inflow:
          for (int j=0; j<ng; ++j) {
            i0(m,n,k,je+j+1,i) = i_in.d_view(n,BoundaryFace::outer_x2);
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
  par_for("radiationbc_x3", DevExeSpace(), 0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    // apply physical boundaries to inner_x3
    switch (mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::outflow:
        for (int k=0; k<ng; ++k) {
          i0(m,n,ks-k-1,j,i) = i0(m,n,ks,j,i);
        }
        break;
      case BoundaryFlag::inflow:
        for (int k=0; k<ng; ++k) {
          i0(m,n,ks-k-1,j,i) = i_in.d_view(n,BoundaryFace::inner_x3);
        }
        break;
      default:
        break;
    }

    // apply physical boundaries to outer_x3
    switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::outflow:
        for (int k=0; k<ng; ++k) {
          i0(m,n,ke+k+1,j,i) = i0(m,n,ke,j,i);
        }
        break;
      case BoundaryFlag::inflow:
        for (int k=0; k<ng; ++k) {
          i0(m,n,ke+k+1,j,i) = i_in.d_view(n,BoundaryFace::outer_x3);
        }
        break;
      default:
        break;
    }
  });

  return;
}
