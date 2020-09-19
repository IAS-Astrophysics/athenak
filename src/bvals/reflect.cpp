//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reflect.cpp
//  \brief implementation of reflecting BCs in each dimension

// Athena++ headers
#include "athena.hpp"
#include "bvals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn void HydroBoundaryVariable::ReflectInnerX1(
//  \brief REFLECTING boundary conditions, inner x1 boundary

void BoundaryValues::ReflectInnerX1()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int is = pmb->mb_cells.is;

  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(pmb->phydro->nhydro); ++n) {
    if (n == (hydro::IVX)) {  // reflect 1-velocity
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ng; ++i) {
            pmb->phydro->u0(hydro::IVX,k,j,is-i-1) = -pmb->phydro->u0(hydro::IVX,k,j,is+i);
          }
        }
      }
    } else {
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ng; ++i) {
            pmb->phydro->u0(n,k,j,is-i-1) = pmb->phydro->u0(n,k,j,is+i);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroBoundaryVariable::ReflectOuterX1(
//  \brief REFLECTING boundary conditions, outer x1 boundary

void BoundaryValues::ReflectOuterX1()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int ie = pmb->mb_cells.ie;

  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(pmb->phydro->nhydro); ++n) {
    if (n == (hydro::IVX)) {  // reflect 1-velocity
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ng; ++i) {
            pmb->phydro->u0(hydro::IVX,k,j,ie+i+1) = -pmb->phydro->u0(hydro::IVX,k,j,ie-i);
          }
        }
      }
    } else {
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ng; ++i) {
            pmb->phydro->u0(n,k,j,ie+i+1) = pmb->phydro->u0(n,k,j,ie-i);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroBoundaryVariable::ReflectInnerX2(
//  \brief REFLECTING boundary conditions, inner x2 boundary

void BoundaryValues::ReflectInnerX2()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int js = pmb->mb_cells.js;

  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(pmb->phydro->nhydro); ++n) {
    if (n == (hydro::IVY)) {  // reflect 2-velocity
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ng; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(hydro::IVY,k,js-j-1,i) = -pmb->phydro->u0(hydro::IVY,k,js+j,i);
          }
        }
      }
    } else {
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ng; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(n,k,js-j-1,i) = pmb->phydro->u0(n,k,js+j,i);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroBoundaryVariable::ReflectOuterX2(
//  \brief REFLECTING boundary conditions, outer x2 boundary

void BoundaryValues::ReflectOuterX2()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int je = pmb->mb_cells.je;

  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(pmb->phydro->nhydro); ++n) {
    if (n == (hydro::IVY)) {  // reflect 2-velocity
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ng; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(hydro::IVY,k,je+j+1,i) = -pmb->phydro->u0(hydro::IVY,k,je-j,i);
          }
        }
      }
    } else {
      for (int k=0; k<ncells3; ++k) {
        for (int j=0; j<ng; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(n,k,je+j+1,i) = pmb->phydro->u0(n,k,je-j,i);
          }
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void HydroBoundaryVariable::ReflectInnerX3(
//  \brief REFLECTING boundary conditions, inner x3 boundary

void BoundaryValues::ReflectInnerX3()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ks = pmb->mb_cells.ks;

  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(pmb->phydro->nhydro); ++n) {
    if (n == (hydro::IVZ)) {  // reflect 2-velocity
      for (int k=0; k<ng; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(hydro::IVZ,ks-k-1,j,i) = -pmb->phydro->u0(hydro::IVZ,ks+k,j,i);
          }
        }
      }
    } else {
      for (int k=0; k<ng; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(n,ks-k-1,j,i) = pmb->phydro->u0(n,ks+k,j,i);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroBoundaryVariable::ReflectOuterX3(
//  \brief REFLECTING boundary conditions, outer x3 boundary

void BoundaryValues::ReflectOuterX3()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ke = pmb->mb_cells.ke;

  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(pmb->phydro->nhydro); ++n) {
    if (n == (hydro::IVZ)) {  // reflect 2-velocity
      for (int k=0; k<ng; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(hydro::IVZ,ke+k+1,j,i) = -pmb->phydro->u0(hydro::IVZ,ke-k,j,i);
          }
        }
      }
    } else {
      for (int k=0; k<ng; ++k) {
        for (int j=0; j<ncells2; ++j) {
          for (int i=0; i<ncells1; ++i) {
            pmb->phydro->u0(n,ke+k+1,j,i) = pmb->phydro->u0(n,ke-k,j,i);
          }
        }
      }
    }
  }
  return;
}
