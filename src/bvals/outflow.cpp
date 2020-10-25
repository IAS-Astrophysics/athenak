//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outflow.cpp
//  \brief implementation of outflow BCs in each dimension

// Athena++ headers
#include "athena.hpp"
#include "bvals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::OutflowInnerX1(
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void BoundaryValues::OutflowInnerX1()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &is = pmb->mb_cells.is;
  auto &u0_ = pmb->phydro->u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ix1", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (n2-1), 0, (ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      u0_(n,k,j,is-i-1) = u0_(n,k,j,is);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::OutflowOuterX1(
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void BoundaryValues::OutflowOuterX1()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &ie = pmb->mb_cells.ie;
  auto &u0_ = pmb->phydro->u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ox1", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (n2-1), 0, (ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      u0_(n,k,j,ie+i+1) = u0_(n,k,j,ie);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::OutflowInnerX2(
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void BoundaryValues::OutflowInnerX2()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &js = pmb->mb_cells.js;
  auto &u0_ = pmb->phydro->u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ix2", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (ng-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      u0_(n,k,js-j-1,i) =  u0_(n,k,js,i);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::OutflowOuterX2(
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void BoundaryValues::OutflowOuterX2()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &je = pmb->mb_cells.je;
  auto &u0_ = pmb->phydro->u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ox2", pmb->exe_space, 0, (nvar-1), 0, (n3-1), 0, (ng-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      u0_(n,k,je+j+1,i) =  u0_(n,k,je,i);
    }   
  );

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::OutflowInnerX3(
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void BoundaryValues::OutflowInnerX3()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &ks = pmb->mb_cells.ks;
  auto &u0_ = pmb->phydro->u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ix3", pmb->exe_space, 0, (nvar-1), 0, (ng-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      u0_(n,ks-k-1,j,i) =  u0_(n,ks,j,i);
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::OutflowOuterX3(
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void BoundaryValues::OutflowOuterX3()
{
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int n1 = pmb->mb_cells.nx1 + 2*ng;
  int n2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int nvar = pmb->phydro->nhydro + pmb->phydro->nscalars;
  int &ke = pmb->mb_cells.ke;
  auto &u0_ = pmb->phydro->u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ox3", pmb->exe_space, 0, (nvar-1), 0, (ng-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {   
      u0_(n,ke+k+1,j,i) =  u0_(n,ke,j,i);
    }
  );

  return;
}
