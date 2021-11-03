//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outflow_mhd.cpp
//  \brief implementation of outflow BCs for MHD conserved vars in each dimension
//   BCs applied to a single MeshBlock specified by input integer index to each function

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void MHD::OutflowInnerX1(
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void MHD::OutflowInnerX1(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // project mhd variables in first active cell into ghost zones
  par_for("outflow_ix1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      // project x1-field
      if (n == nvar-3) {
        b0_.x1f(m,k,j,is-i-1) = b0_.x1f(m,k,j,is);

      // project extra row of x2-field
      } else if (n == nvar-2) {
        b0_.x2f(m,k,j,is-i-1) = b0_.x2f(m,k,j,is);
        if (j == n2-1) {b0_.x2f(m,k,j+1,is-i-1) = b0_.x2f(m,k,j+1,is);}

      // project extra row of x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,k,j,is-i-1) = b0_.x3f(m,k,j,is);
        if (k == n3-1) {b0_.x3f(m,k+1,j,is-i-1) = b0_.x3f(m,k+1,j,is);}

      // project everything else
      } else {
        u0_(m,n,k,j,is-i-1) = u0_(m,n,k,j,is);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::OutflowOuterX1(
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void MHD::OutflowOuterX1(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // project mhd variables in first active cell into ghost zones
  par_for("outflow_ox1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      // project x1-field
      if (n == nvar-3) {
        b0_.x1f(m,k,j,ie+i+2) = b0_.x1f(m,k,j,ie+1);
  
      // project extra row of x2-field
      } else if (n == nvar-2) {
        b0_.x2f(m,k,j,ie+i+1) = b0_.x2f(m,k,j,ie);
        if (j == n2-1) {b0_.x2f(m,k,j+1,ie+i+1) = b0_.x2f(m,k,j+1,ie);}

      // project extra row of x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,k,j,ie+i+1) = b0_.x3f(m,k,j,ie);
        if (k == n3-1) {b0_.x3f(m,k+1,j,ie+i+1) = b0_.x3f(m,k+1,j,ie);}

      // project everything else
      } else {
        u0_(m,n,k,j,ie+i+1) = u0_(m,n,k,j,ie);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::OutflowInnerX2(
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void MHD::OutflowInnerX2(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // project mhd variables in first active cell into ghost zones
  par_for("outflow_ix2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      // project extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,k,js-j-1,i) = b0_.x1f(m,k,js,i);
        if (i == n1-1) {b0_.x1f(m,k,js-j-1,i+1) = b0_.x1f(m,k,js,i+1);}

      // project x2-field
      } else if (n == nvar-2) {
        b0_.x2f(m,k,js-j-1,i) = b0_.x2f(m,k,js,i);

      // project extra row of x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,k,js-j-1,i) = b0_.x3f(m,k,js,i);
        if (k == n3-1) {b0_.x3f(m,k+1,js-j-1,i) = b0_.x3f(m,k+1,js,i);}

      // project everything else
      } else {
        u0_(m,n,k,js-j-1,i) =  u0_(m,n,k,js,i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::OutflowOuterX2(
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void MHD::OutflowOuterX2(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &je = indcs.je;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // project mhd variables in first active cell into ghost zones
  par_for("outflow_ox2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      // project extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,k,je+j+1,i) = b0_.x1f(m,k,je,i);
        if (i == n1-1) {b0_.x1f(m,k,je+j+1,i+1) = b0_.x1f(m,k,je,i+1);}
        
      // project x2-field
      } else if (n == nvar-2) {
        b0_.x2f(m,k,je+j+2,i) = b0_.x2f(m,k,je+1,i);
        
      // project extra row of x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,k,je+j+1,i) = b0_.x3f(m,k,je,i);
        if (k == n3-1) {b0_.x3f(m,k+1,je+j+1,i) = b0_.x3f(m,k+1,je,i);}
      
      // project everything else
      } else {
        u0_(m,n,k,je+j+1,i) =  u0_(m,n,k,je,i);
      }
    }   
  );

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MHD::OutflowInnerX3(
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void MHD::OutflowInnerX3(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ks = indcs.ks;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // project mhd variables in first active cell into ghost zones
  par_for("outflow_ix3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      // project extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,ks-k-1,j,i) = b0_.x1f(m,ks,j,i);
        if (i == n1-1) {b0_.x1f(m,ks-k-1,j,i+1) = b0_.x1f(m,ks,j,i+1);}

      // project extra row of x3-field
      } else if (n == nvar-2) {
        b0_.x2f(m,ks-k-1,j,i) = b0_.x2f(m,ks,j,i);
        if (j == n2-1) {b0_.x2f(m,ks-k-1,j+1,i) = b0_.x2f(m,ks,j+1,i);}

      // project x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,ks-k-1,j,i) = b0_.x3f(m,ks,j,i);

      // project everything else
      } else {
        u0_(m,n,ks-k-1,j,i) =  u0_(m,n,ks,j,i);
      }
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::OutflowOuterX3(
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void MHD::OutflowOuterX3(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ke = indcs.ke;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // project mhd variables in first active cell into ghost zones
  par_for("outflow_ox3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {   
      // project extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,ke+k+1,j,i) = b0_.x1f(m,ke,j,i);
        if (i == n1-1) {b0_.x1f(m,ke+k+1,j,i+1) = b0_.x1f(m,ke,j,i+1);}
  
      // project extra row of x3-field
      } else if (n == nvar-2) {
        b0_.x2f(m,ke+k+1,j,i) = b0_.x2f(m,ke,j,i);
        if (j == n2-1) {b0_.x2f(m,ke+k+1,j+1,i) = b0_.x2f(m,ke,j+1,i);}

      // project x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,ke+k+2,j,i) = b0_.x3f(m,ke+1,j,i);

      // project everything else
      } else { 
        u0_(m,n,ke+k+1,j,i) =  u0_(m,n,ke,j,i);
      }
    }
  );

  return;
}
} // namespace mhd
