//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outflow_hydro.cpp
//  \brief implementation of outflow BCs for Hydro conserved vars in each dimension
//   BCs applied to a single MeshBlock specified by input integer index to each function

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Hydro::OutflowInnerX1(
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void Hydro::OutflowInnerX1(int m)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int nvar = nhydro + nscalars;
  auto &u0_ = u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ix1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      u0_(m,n,k,j,is-i-1) = u0_(m,n,k,j,is);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::OutflowOuterX1(
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void Hydro::OutflowOuterX1(int m)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;
  int nvar = nhydro + nscalars;
  auto &u0_ = u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ox1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      u0_(m,n,k,j,ie+i+1) = u0_(m,n,k,j,ie);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::OutflowInnerX2(
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void Hydro::OutflowInnerX2(int m)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;
  int nvar = nhydro + nscalars;
  auto &u0_ = u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ix2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      u0_(m,n,k,js-j-1,i) =  u0_(m,n,k,js,i);
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::OutflowOuterX2(
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void Hydro::OutflowOuterX2(int m)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &je = indcs.je;
  int nvar = nhydro + nscalars;
  auto &u0_ = u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ox2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      u0_(m,n,k,je+j+1,i) =  u0_(m,n,k,je,i);
    }   
  );

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Hydro::OutflowInnerX3(
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void Hydro::OutflowInnerX3(int m)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ks = indcs.ks;
  int nvar = nhydro + nscalars;
  auto &u0_ = u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ix3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      u0_(m,n,ks-k-1,j,i) =  u0_(m,n,ks,j,i);
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::OutflowOuterX3(
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void Hydro::OutflowOuterX3(int m)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ke = indcs.ke;
  int nvar = nhydro + nscalars;
  auto &u0_ = u0;

  // project hydro variables in first active cell into ghost zones
  par_for("outflow_ox3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {   
      u0_(m,n,ke+k+1,j,i) =  u0_(m,n,ke,j,i);
    }
  );

  return;
}
} // namespace hydro
