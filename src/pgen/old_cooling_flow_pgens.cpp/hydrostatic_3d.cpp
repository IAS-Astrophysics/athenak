//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mass_removal_test.cpp
//  \brief Problem generator for testing mass removal
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

void UserSource(Mesh* pm, const Real bdt);
void GravitySource(Mesh* pm, const Real bdt);
KOKKOS_INLINE_FUNCTION Real GravPot(Real r_scale, Real x1, Real x2, Real x3);
void Static(Mesh* pm);

Real r_scale;

//----------------------------------------------------------------------------------------
//  \brief Problem Generator for mass removal

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  // Enroll user functions 
  user_srcs_func = UserSource;
  user_bcs_func = Static;

  if (restart) return;

  // Read in constants
  r_scale = pin->GetOrAddReal("problem", "r_scale",1.0);

  // Capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Initialize Hydro variables -------------------------------
  auto &u0 = pmbp->phydro->u0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real r_s = r_scale;

  // Set initial conditions
  par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
    Real cs2 = 1.0;
    Real rho0 = 1.0;
    Real phi0 = log(11.0)/(10*r_s);

    Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
    Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;

    if (isnan(rho)){
	printf("%4.2f %4.2f %4.2f %4.2f \n",x1v,x2v,x3v,GravPot(r_s,x1v,x2v,x3v));
    }

    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = press/gm1 +
       0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
       SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
  });

  return;
}

//===========================================================================//
//                              Source Terms                                 //
//===========================================================================//

void UserSource(Mesh* pm, const Real bdt) {
  GravitySource(pm, bdt);
}

void GravitySource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  Real &r_s = r_scale;

  par_for("gravity_source", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1l =   LeftEdgeX(i-is, nx1, x1min, x1max);
    Real x1r = LeftEdgeX(i+1-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2l =   LeftEdgeX(j-js, nx2, x2min, x2max);
    Real x2r = LeftEdgeX(j+1-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3l =   LeftEdgeX(k-ks, nx3, x3min, x3max);
    Real x3r = LeftEdgeX(k+1-ks, nx3, x3min, x3max);

    Real phi1l = GravPot(r_s,x1l,x2v,x3v);
    Real phi1r = GravPot(r_s,x1r,x2v,x3v);
    Real f_x1 = -(phi1r-phi1l)/(x1r-x1l);

    Real phi2l = GravPot(r_s,x1v,x2l,x3v);
    Real phi2r = GravPot(r_s,x1v,x2r,x3v);
    Real f_x2 = -(phi2r-phi2l)/(x2r-x2l);

    Real phi3l = GravPot(r_s,x1v,x2v,x3l);
    Real phi3r = GravPot(r_s,x1v,x2v,x3r);
    Real f_x3 = -(phi3r-phi3l)/(x3r-x3l);

    Real src_x1 = bdt*w0(m,IDN,k,j,i)*f_x1;
    Real src_x2 = bdt*w0(m,IDN,k,j,i)*f_x2;
    Real src_x3 = bdt*w0(m,IDN,k,j,i)*f_x3;

    u0(m,IM1,k,j,i) += src_x1;	
    u0(m,IM2,k,j,i) += src_x2;
    u0(m,IM3,k,j,i) += src_x3;
    u0(m,IEN,k,j,i) += (src_x1*w0(m,IVX,k,j,i) 
                       +src_x2*w0(m,IVY,k,j,i) 
                       +src_x3*w0(m,IVZ,k,j,i));
  });

  return;
}

KOKKOS_INLINE_FUNCTION
Real GravPot(Real r_scale, Real x1, Real x2, Real x3) {
  Real r = sqrt(x1*x1 + x2*x2 + x3*x3);
  return -log(1 + r/r_scale)/r;
}

void Static(Mesh* pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;

  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real &r_s = r_scale;

  par_for("static_x1", DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real cs2 = 1.0;
    Real rho0 = 1.0;
    Real phi0 = log(11.0)/(10.0*r_s);

    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
      Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;

      u0(m,IDN,k,j,i) = rho;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = press/gm1;
    }

    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
      Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;

      u0(m,IDN,k,j,ie+i+1) = rho;
      u0(m,IM1,k,j,ie+i+1) = 0.0;
      u0(m,IM2,k,j,ie+i+1) = 0.0;
      u0(m,IM3,k,j,ie+i+1) = 0.0;
      u0(m,IEN,k,j,ie+i+1) = press/gm1;
    }

  });

  par_for("static_x2", DevExeSpace(),0,nmb1,0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    
    Real cs2 = 1.0;
    Real rho0 = 1.0;
    Real phi0 = log(11.0)/(10.0*r_s);

    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
      Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;
 
      u0(m,IDN,k,j,i) = rho;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = press/gm1;
    }

    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
      Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;

      u0(m,IDN,k,je+j+1,i) = rho;
      u0(m,IM1,k,je+j+1,i) = 0.0;
      u0(m,IM2,k,je+j+1,i) = 0.0;
      u0(m,IM3,k,je+j+1,i) = 0.0;
      u0(m,IEN,k,je+j+1,i) = press/gm1;
    }

  });

  par_for("static_x3", DevExeSpace(),0,nmb1,0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real cs2 = 1.0;
    Real rho0 = 1.0;
    Real phi0 = log(11.0)/(10.0*r_s);

    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
      Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;

      u0(m,IDN,k,j,i) = rho;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = press/gm1;
    }
    
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real rho = rho0*pow(1+gm1*(phi0-GravPot(r_s,x1v,x2v,x3v))/cs2,1/gm1);
      Real press = pow(rho,eos.gamma)*cs2*pow(rho0,-gm1)/eos.gamma;

      u0(m,IDN,ke+k+1,j,i) = rho;
      u0(m,IM1,ke+k+1,j,i) = 0.0;
      u0(m,IM2,ke+k+1,j,i) = 0.0;
      u0(m,IM3,ke+k+1,j,i) = 0.0;
      u0(m,IEN,ke+k+1,j,i) = press/gm1;
    }

  });
}
