//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cgm_static.cpp
//  \brief Problem generator for a hydrostatic CGM
#include <iostream>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "units/units.hpp"

KOKKOS_INLINE_FUNCTION 
void SetEquilibriumState(const DvceArray5D<Real> &u0,
    int m, int k, int j, int i, 
    Real x1v, Real x2v, Real x3v, Real G, Real r_s, 
    Real rho_s, Real m_g, Real a_g, Real r_m, Real rho_m,
    Real cs2, Real rho0, Real phi0, Real gm1, Real gamma);
void UserSource(Mesh* pm, const Real bdt);
void GravitySource(Mesh* pm, const Real bdt);
KOKKOS_INLINE_FUNCTION 
Real GravPot(Real x1, Real x2, Real x3,
    Real G, Real r_s, Real rho_s, 
    Real M_gal, Real a_gal, 
    Real R200, Real rho_mean);
void Static(Mesh* pm);

// Constants for gravity source
namespace {
  Real r_scale;
  Real rho_scale;
  Real m_gal;
  Real a_gal;
  Real r_200;
  Real rho_mean;

  Real r_vir = 100.0;  // Virial radius
  Real cs2_ref = 50.0; // Sound speed squared at r_vir
  Real rho_ref = 1e-4; // Density at r_vir
  Real phi_ref;        // Potential at r_vir
}

//===========================================================================//
//                               Initialize                                  //
//===========================================================================//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  // Enroll user functions 
  user_srcs_func = UserSource;
  user_bcs_func = Static;

  if (restart) return;

  // Read in constants
  r_scale   = pin->GetReal("potential", "r_scale");
  rho_scale = pin->GetReal("potential", "rho_scale");
  m_gal     = pin->GetReal("potential", "mass_gal");
  a_gal     = pin->GetReal("potential", "scale_gal");
  r_200     = pin->GetReal("potential", "r_200");
  rho_mean  = pin->GetReal("potential", "rho_mean");

  // Capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Initialize Hydro variables
  auto &u0 = pmbp->phydro->u0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  Real G = pmbp->punit->grav_constant();
  Real r_s = r_scale;
  Real rho_s = rho_scale;
  Real m_g = m_gal;
  Real a_g = a_gal;
  Real r_m = r_200;
  Real rho_m = rho_mean;

  Real cs2 = cs2_ref;
  Real rho0 = rho_ref;
  phi_ref = GravPot(r_vir,0,0,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
  Real phi0 = phi_ref;

  // Set initial conditions
  par_for("pgen_ic", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

    SetEquilibriumState(u0, m, k, j, i, x1v, x2v, x3v, G, r_s, rho_s, 
      m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
  });

  return;
}

KOKKOS_INLINE_FUNCTION
void SetEquilibriumState(const DvceArray5D<Real> &u0, int m, int k, int j, int i, 
                         Real x1v, Real x2v, Real x3v, Real G, Real r_s, 
                         Real rho_s, Real m_g, Real a_g, Real r_m, Real rho_m,
                         Real cs2, Real rho0, Real phi0, Real gm1, Real gamma) {
    Real phi = GravPot(x1v, x2v, x3v, G, r_s, rho_s, m_g, a_g, r_m, rho_m);
    Real rho = rho0 * pow(1 + gm1 * (phi0 - phi) / cs2, 1/gm1);
    Real press = pow(rho, gamma) * cs2 * pow(rho0, -gm1) / gamma;

    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    u0(m, IEN, k, j, i) = press/gm1 +
        0.5 * (SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i)) +
              SQR(u0(m, IM3, k, j, i))) / u0(m, IDN, k, j, i);
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
  
  Real G = pmbp->punit->grav_constant();
  Real r_s = r_scale;
  Real rho_s = rho_scale;
  Real m_g = m_gal;
  Real a_g = a_gal;
  Real r_m = r_200;
  Real rho_m = rho_mean;

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

    Real phi1l = GravPot(x1l,x2v,x3v,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
    Real phi1r = GravPot(x1r,x2v,x3v,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
    Real f_x1 = -(phi1r-phi1l)/(x1r-x1l);

    Real phi2l = GravPot(x1v,x2l,x3v,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
    Real phi2r = GravPot(x1v,x2r,x3v,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
    Real f_x2 = -(phi2r-phi2l)/(x2r-x2l);

    Real phi3l = GravPot(x1v,x2v,x3l,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
    Real phi3r = GravPot(x1v,x2v,x3r,G,r_s,rho_s,m_g,a_g,r_m,rho_m);
    Real f_x3 = -(phi3r-phi3l)/(x3r-x3l);

    Real density = w0(m, IDN, k, j, i);
    Real src_x1 = bdt*density*f_x1;
    Real src_x2 = bdt*density*f_x2;
    Real src_x3 = bdt*density*f_x3;

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
Real GravPot(Real x1, Real x2, Real x3,
             Real G, Real r_s, Real rho_s, 
             Real M_gal, Real a_gal, 
             Real R200, Real rho_mean) 
{
  Real r2 = x1*x1 + x2*x2 + x3*x3;
  Real r = sqrt(r2);

  // Avoid division by zero
  constexpr Real tiny = 1.0e-20;
  r = fmax(r, tiny);

  // NFW component
  Real x = r / r_s;
  Real phi_NFW = -4 * M_PI * G * rho_s * SQR(r_s) * log(1 + x) / x;
  
  // Plummer component
  Real phi_Plummer = -G * M_gal / sqrt(r2 + SQR(a_gal));
  
  // Outer component
  Real term1 = (4.0 / 3.0) * pow(5 * R200, 1.5) * sqrt(r);
  Real term2 = (1.0 / 6.0) * r2;
  Real phi_Outer = -4 * M_PI * G * rho_mean * (term1 + term2);
  
  // Total potential
  Real phi = phi_NFW + phi_Plummer + phi_Outer;
  
  return phi;
}

//===========================================================================//
//                             User Boundary                                 //
//===========================================================================//

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
  
  // Get constants needed for equilibrium state
  Real G = pmbp->punit->grav_constant();
  Real r_s = r_scale;
  Real rho_s = rho_scale;
  Real m_g = m_gal;
  Real a_g = a_gal;
  Real r_m = r_200;
  Real rho_m = rho_mean;
  Real cs2 = cs2_ref;
  Real rho0 = rho_ref;
  Real phi0 = phi_ref;
  
  // Handle X1 boundaries
  par_for("static_x1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    
    // Inner X1 boundary
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      
      SetEquilibriumState(u0, m, k, j, i, x1v, x2v, x3v, G, r_s, rho_s, 
                          m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
    }
  
    // Outer X1 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      int i_out = ie + i + 1;
      Real x1v = CellCenterX(i_out-is, indcs.nx1, x1min, x1max);
      
      SetEquilibriumState(u0, m, k, j, i_out, x1v, x2v, x3v, G, r_s, rho_s, 
                          m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
    }
  });
  
  // Handle X2 boundaries
  par_for("static_x2", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (ng-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    
    // Inner X2 boundary
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      
      SetEquilibriumState(u0, m, k, j, i, x1v, x2v, x3v, G, r_s, rho_s, 
                          m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
    }
  
    // Outer X2 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      int j_out = je + j + 1;
      Real x2v = CellCenterX(j_out-js, indcs.nx2, x2min, x2max);
      
      SetEquilibriumState(u0, m, k, j_out, i, x1v, x2v, x3v, G, r_s, rho_s, 
                          m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
    }
  });
  
  // Handle X3 boundaries
  par_for("static_x3", DevExeSpace(), 0, nmb1, 0, (ng-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    // Inner X3 boundary
    if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      
      SetEquilibriumState(u0, m, k, j, i, x1v, x2v, x3v, G, r_s, rho_s, 
                          m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
    }
    
    // Outer X3 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
      int k_out = ke + k + 1;
      Real x3v = CellCenterX(k_out-ks, indcs.nx3, x3min, x3max);
      
      SetEquilibriumState(u0, m, k_out, j, i, x1v, x2v, x3v, G, r_s, rho_s, 
                          m_g, a_g, r_m, rho_m, cs2, rho0, phi0, gm1, eos.gamma);
    }
  });
}