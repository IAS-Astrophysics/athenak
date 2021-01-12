//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reflect.cpp
//  \brief implementation of reflecting BCs in each dimension

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReflectInnerX1(
//  \brief REFLECTING boundary conditions, inner x1 boundary

void BoundaryValues::ReflectInnerX1()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &is = pmb->mb_cells.is;
  auto &u0_ = pmb->phydro->u0;

  // copy hydro variables into ghost zones, reflecting v1
  par_for("reflect_ix1", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (n2-1), 0, (ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      if (n == (hydro::IVX)) {  // reflect 1-velocity
        u0_(n,k,j,is-i-1) = -u0_(n,k,j,is+i);
      } else {
        u0_(n,k,j,is-i-1) =  u0_(n,k,j,is+i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReflectOuterX1(
//  \brief REFLECTING boundary conditions, outer x1 boundary

void BoundaryValues::ReflectOuterX1()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &ie = pmb->mb_cells.ie;
  auto &u0_ = pmb->phydro->u0;

  // copy hydro variables into ghost zones, reflecting v1
  par_for("reflect_ox1", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (n2-1), 0, (ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      if (n == (hydro::IVX)) {  // reflect 1-velocity
        u0_(n,k,j,ie+i+1) = -u0_(n,k,j,ie-i);
      } else {
        u0_(n,k,j,ie+i+1) =  u0_(n,k,j,ie-i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReflectInnerX2(
//  \brief REFLECTING boundary conditions, inner x2 boundary

void BoundaryValues::ReflectInnerX2()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &js = pmb->mb_cells.js;
  auto &u0_ = pmb->phydro->u0;

  // copy hydro variables into ghost zones, reflecting v2
  par_for("reflect_ix2", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (ng-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      if (n == (hydro::IVY)) {  // reflect 2-velocity
        u0_(n,k,js-j-1,i) = -u0_(n,k,js+j,i);
      } else {
        u0_(n,k,js-j-1,i) =  u0_(n,k,js+j,i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReflectOuterX2(
//  \brief REFLECTING boundary conditions, outer x2 boundary

void BoundaryValues::ReflectOuterX2()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &je = pmb->mb_cells.je;
  auto &u0_ = pmb->phydro->u0;

  // copy hydro variables into ghost zones, reflecting v2
  par_for("reflect_ox2", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (ng-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      if (n == (hydro::IVY)) {  // reflect 2-velocity
        u0_(n,k,je+j+1,i) = -u0_(n,k,je-j,i);
      } else {
        u0_(n,k,je+j+1,i) =  u0_(n,k,je-j,i);
      }   
    }   
  );

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReflectInnerX3(
//  \brief REFLECTING boundary conditions, inner x3 boundary

void BoundaryValues::ReflectInnerX3()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &ks = pmb->mb_cells.ks;
  auto &u0_ = pmb->phydro->u0;

  // copy hydro variables into ghost zones, reflecting v3
  par_for("reflect_ix3", pmb->exe_space, 0, (nvar-1), 0, (ng-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      if (n == (hydro::IVZ)) {  // reflect 3-velocity
        u0_(n,ks-k-1,j,i) = -u0_(n,ks+k,j,i);
      } else {
        u0_(n,ks-k-1,j,i) =  u0_(n,ks+k,j,i);
      }   
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ReflectOuterX3(
//  \brief REFLECTING boundary conditions, outer x3 boundary

void BoundaryValues::ReflectOuterX3()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &ke = pmb->mb_cells.ke;
  auto &u0_ = pmb->phydro->u0;

  // copy hydro variables into ghost zones, reflecting v3
  par_for("reflect_ox3", pmb->exe_space, 0, (nvar-1), 0, (ng-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {   
      if (n == (hydro::IVZ)) {  // reflect 3-velocity
        u0_(n,ke+k+1,j,i) = -u0_(n,ke-k,j,i);
      } else {
        u0_(n,ke+k+1,j,i) =  u0_(n,ke-k,j,i);
      }
    }
  );

  return;
}
