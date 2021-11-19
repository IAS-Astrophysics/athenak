//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reflect_mhd.cpp
//  \brief implementation of reflecting BCs for MHD conserved vars in each dimension
//   BCs applied to a single MeshBlock specified by input integer index to each function

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void MHD::ReflectInnerX1(
//  \brief REFLECTING boundary conditions, inner x1 boundary

void MHD::ReflectInnerX1(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // copy MHD variables into ghost zones, reflecting v1 and b1
  par_for("reflect_ix1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      // reflect x1-field
      if (n == nvar-3) {   
        b0_.x1f(m,k,j,is-i-1) = -b0_.x1f(m,k,j,is+i+1);

      // copy extra row of x2-field
      } else if (n == nvar-2) {
        b0_.x2f(m,k,j,is-i-1) = b0_.x2f(m,k,j,is+i);
        if (j == n2-1) {b0_.x2f(m,k,j+1,is-i-1) = b0_.x2f(m,k,j+1,is+i);}

      // copy extra row of x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,k,j,is-i-1) = b0_.x3f(m,k,j,is+i);
        if (k == n3-1) {b0_.x3f(m,k+1,j,is-i-1) = b0_.x3f(m,k+1,j,is+i);}

      // reflect 1-velocity
      } else if (n == (IVX)) {
        u0_(m,n,k,j,is-i-1) = -u0_(m,n,k,j,is+i);

      // copy everything else
      } else {
        u0_(m,n,k,j,is-i-1) =  u0_(m,n,k,j,is+i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ReflectOuterX1(
//  \brief REFLECTING boundary conditions, outer x1 boundary

void MHD::ReflectOuterX1(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // copy MHD variables into ghost zones, reflecting v1 and b1
  par_for("reflect_ox1", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {
      // reflect x1-field
      if (n == nvar-3) {   
        b0_.x1f(m,k,j,ie+i+2) = -b0_.x1f(m,k,j,ie-i);

      // copy extra row of x2-field
      } else if (n == nvar-2) { 
        b0_.x2f(m,k,j,ie+i+1) = b0_.x2f(m,k,j,ie-i);
        if (j == n2-1) {b0_.x2f(m,k,j+1,ie+i+1) = b0_.x2f(m,k,j+1,ie-i);}

      // copy extra row of x3-field
      } else if (n == nvar-1) { 
        b0_.x3f(m,k,j,ie+i+1) = b0_.x3f(m,k,j,ie-i);
        if (k == n3-1) {b0_.x3f(m,k+1,j,ie+i+1) = b0_.x3f(m,k+1,j,ie-i);}

      // reflect 1-velocity
      } else if (n == (IVX)) {  
        u0_(m,n,k,j,ie+i+1) = -u0_(m,n,k,j,ie-i);

      // copy everything else
      } else {
        u0_(m,n,k,j,ie+i+1) =  u0_(m,n,k,j,ie-i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ReflectInnerX2(
//  \brief REFLECTING boundary conditions, inner x2 boundary

void MHD::ReflectInnerX2(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // copy MHD variables into ghost zones, reflecting v2 and b2
  par_for("reflect_ix2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      // copy extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,k,js-j-1,i) = b0_.x1f(m,k,js+j,i);
        if (i == n1-1) {b0_.x1f(m,k,js-j-1,i+1) = b0_.x1f(m,k,js+j,i+1);}
      
      // reflect x2-field
      } else if (n == nvar-2) { 
        b0_.x2f(m,k,js-j-1,i) = -b0_.x2f(m,k,js+j+1,i);
      
      // copy extra row of x3-field
      } else if (n == nvar-1) { 
        b0_.x3f(m,k,js-j-1,i) = b0_.x3f(m,k,js+j,i);
        if (k == n3-1) {b0_.x3f(m,k+1,js-j-1,i) = b0_.x3f(m,k+1,js+j,i);}

      // reflect 2-velocity
      } else if (n == (IVY)) {  
        u0_(m,n,k,js-j-1,i) = -u0_(m,n,k,js+j,i);

      // copy everything else
      } else {
        u0_(m,n,k,js-j-1,i) =  u0_(m,n,k,js+j,i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ReflectOuterX2(
//  \brief REFLECTING boundary conditions, outer x2 boundary

void MHD::ReflectOuterX2(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &je = indcs.je;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // copy MHD variables into ghost zones, reflecting v2 and b2
  par_for("reflect_ox2", DevExeSpace(),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      // copy extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,k,je+j+1,i) = b0_.x1f(m,k,je-j,i);
        if (i == n1-1) {b0_.x1f(m,k,je+j+1,i+1) = b0_.x1f(m,k,je-j,i+1);}

      // reflect x2-field
      } else if (n == nvar-2) {
        b0_.x2f(m,k,je+j+2,i) = -b0_.x2f(m,k,je-j,i);

      // copy extra row of x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,k,je+j+1,i) = b0_.x3f(m,k,je-j,i);
        if (k == n3-1) {b0_.x3f(m,k+1,je+j+1,i) = b0_.x3f(m,k+1,je-j,i);}

      // reflect 2-velocity
      } else if (n == (IVY)) {
        u0_(m,n,k,je+j+1,i) = -u0_(m,n,k,je-j,i);

      // copy everything else
      } else {
        u0_(m,n,k,je+j+1,i) =  u0_(m,n,k,je-j,i);
      }   
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ReflectInnerX3(
//  \brief REFLECTING boundary conditions, inner x3 boundary

void MHD::ReflectInnerX3(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ks = indcs.ks;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // copy MHD variables into ghost zones, reflecting v3 and b3
  par_for("reflect_ix3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    { 
      // copy extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,ks-k-1,j,i) = b0_.x1f(m,ks+k,j,i);
        if (i == n1-1) {b0_.x1f(m,ks-k-1,j,i+1) = b0_.x1f(m,ks+k,j,i+1);}
        
      // copy extra row of x3-field
      } else if (n == nvar-2) {
        b0_.x2f(m,ks-k-1,j,i) = b0_.x2f(m,ks+k,j,i);
        if (j == n2-1) {b0_.x2f(m,ks-k-1,j+1,i) = b0_.x2f(m,ks+k,j+1,i);}

      // reflect x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,ks-k-1,j,i) = -b0_.x3f(m,ks+k+1,j,i);

      // reflect 3-velocity
      } else if (n == (IVZ)) {
        u0_(m,n,ks-k-1,j,i) = -u0_(m,n,ks+k,j,i);

      // copy everything else
      } else {
        u0_(m,n,ks-k-1,j,i) =  u0_(m,n,ks+k,j,i);
      }   
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ReflectOuterX3(
//  \brief REFLECTING boundary conditions, outer x3 boundary

void MHD::ReflectOuterX3(int m)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ke = indcs.ke;
  int nvar = nmhd + nscalars + 3;
  auto &u0_ = u0;
  auto &b0_ = b0;

  // copy MHD variables into ghost zones, reflecting v3 and b3
  par_for("reflect_ox3", DevExeSpace(),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int n, int k, int j, int i)
    {   
      // copy extra row of x1-field
      if (n == nvar-3) {
        b0_.x1f(m,ke+k+1,j,i) = b0_.x1f(m,ke-k,j,i);
        if (i == n1-1) {b0_.x1f(m,ke+k+1,j,i+1) = b0_.x1f(m,ke-k,j,i+1);}

      // copy extra row of x3-field
      } else if (n == nvar-2) {
        b0_.x2f(m,ke+k+1,j,i) = b0_.x2f(m,ke-k,j,i);
        if (j == n2-1) {b0_.x2f(m,ke+k+1,j+1,i) = b0_.x2f(m,ke-k,j+1,i);}

      // reflect x3-field
      } else if (n == nvar-1) {
        b0_.x3f(m,ke+k+2,j,i) = -b0_.x3f(m,ke-k,j,i);

      // reflect 3-velocity
      } else if (n == (IVZ)) {
        u0_(m,n,ke+k+1,j,i) = -u0_(m,n,ke-k,j,i);

      // copy everything else
      } else {
        u0_(m,n,ke+k+1,j,i) =  u0_(m,n,ke-k,j,i);
      }
    }
  );

  return;
}
} // namespace mhd
