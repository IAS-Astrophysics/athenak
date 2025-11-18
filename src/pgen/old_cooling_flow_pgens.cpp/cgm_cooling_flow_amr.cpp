//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cgm_cooling_flow_amr.cpp
//  \brief Problem generator for a cooling flow CGM with AMR resolved disk

#include <iostream>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"
#include "units/units.hpp"
#include "utils/profile_reader.hpp"

//===========================================================================//
//                               Globals                                     //
//===========================================================================//

KOKKOS_INLINE_FUNCTION 
void SetEquilibriumState(const DvceArray5D<Real> &u0, 
  int m, int k, int j, int i, 
  Real x1v, Real x2v, Real x3v, 
  Real x1l, Real x1r, Real x2l, Real x2r, 
  Real G, Real r_s, Real rho_s, Real m_g, 
  Real a_g, Real z_g, Real r_m, Real rho_m,
  Real gm1, const ProfileReader &disk_profile);
    
KOKKOS_INLINE_FUNCTION
void SetCoolingFlowState(const DvceArray5D<Real> &u0, 
                         int m, int k, int j, int i, 
                         Real x1v, Real x2v, Real x3v, 
                         Real gm1, const ProfileReader &profile);

KOKKOS_INLINE_FUNCTION
void SetRotation(const DvceArray5D<Real> &u0, 
                 int m, int k, int j, int i, 
                 Real x1v, Real x2v, Real x3v, 
                 Real r_circ, Real v_circ);

KOKKOS_INLINE_FUNCTION 
Real GravPot(Real x1, Real x2, Real x3,
             Real G, Real r_s, Real rho_s, 
             Real M_gal, Real a_gal, Real z_gal,
             Real R200, Real rho_mean);

void UserSource(Mesh* pm, const Real bdt);
void GravitySource(Mesh* pm, const Real bdt);
void UserBoundary(Mesh* pm);
void FreeProfile(ParameterInput *pin, Mesh *pm);
void RefinementCondition(MeshBlockPack* pmbp);

namespace {
  // Constants for gravitational potential
  Real r_scale;
  Real rho_scale;
  Real m_gal;
  Real a_gal;
  Real z_gal;
  Real r_200;
  Real rho_mean;

  // Constants for rotation
  Real r_circ;
  Real v_circ;

  // Add profile readers on host and device
  ProfileReaderHost profile_reader_host;  // Host-side reader
  ProfileReader profile_reader;           // Device-side reader
  ProfileReaderHost disk_profile_reader_host;  // Host-side reader
  ProfileReader disk_profile_reader;           // Device-side reader
  
  // Refinment condition threshold
  Real ddens_threshold;
}

//===========================================================================//
//                               Initialize                                  //
//===========================================================================//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  // Enroll user functions 
  user_srcs_func  = UserSource;
  user_bcs_func   = UserBoundary;
  pgen_final_func = FreeProfile;
  user_ref_func  = RefinementCondition;

  if (restart) return;

  // Read in constants
  r_scale   = pin->GetReal("potential", "r_scale");
  rho_scale = pin->GetReal("potential", "rho_scale");
  m_gal     = pin->GetReal("potential", "mass_gal");
  a_gal     = pin->GetReal("potential", "scale_gal");
  z_gal     = pin->GetReal("potential", "z_gal");
  r_200     = pin->GetReal("potential", "r_200");
  rho_mean  = pin->GetReal("potential", "rho_mean");
  r_circ    = pin->GetReal("problem", "r_circ");
  v_circ    = pin->GetReal("problem", "v_circ");

  // Read the density gradient threshold for refinement
  ddens_threshold = pin->GetReal("problem", "ddens_max");

  // Read the CGM cooling flow profile file
  std::string profile_file = pin->GetString("problem", "profile_file");
  try {
    profile_reader_host.ReadProfiles(profile_file);
    // Create device-accessible reader
    profile_reader = profile_reader_host.CreateDeviceReader();
    std::cout << "Successfully loaded profiles from " << profile_file << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error loading profiles: " << e.what() << std::endl;
  }

  // Read the disk profile file
  std::string disk_profile_file = pin->GetString("problem", "disk_profile_file");
  try {
    disk_profile_reader_host.ReadProfiles(disk_profile_file);
    // Create device-accessible reader
    disk_profile_reader = disk_profile_reader_host.CreateDeviceReader();
    std::cout << "Successfully loaded disk profiles from " << disk_profile_file << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error loading disk profiles: " << e.what() << std::endl;
  }

  // Capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  auto &u0 = pmbp->phydro->u0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto &profile = profile_reader;
  auto &disk_profile = disk_profile_reader;

  Real G = pmbp->punit->grav_constant();
  Real r_s = r_scale;
  Real rho_s = rho_scale;
  Real m_g = m_gal;
  Real a_g = a_gal;
  Real z_g = z_gal;
  Real r_m = r_200;
  Real rho_m = rho_mean;

  Real r_c = r_circ;
  Real v_c = v_circ;

  // Use loaded profiles
  par_for("pgen_ic", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    SetCoolingFlowState(u0, m, k, j, i, x1v, x2v, x3v, gm1, profile);
    SetRotation(u0, m, k, j, i, x1v, x2v, x3v, r_c, v_c);
    SetEquilibriumState(u0, m, k, j, i, x1v, x2v, x3v, 
                        x1l, x1r, x2l, x2r, G, r_s, rho_s, 
                        m_g, a_g, z_g, r_m, rho_m, gm1, disk_profile);
  });

  return;
}

KOKKOS_INLINE_FUNCTION
void SetCoolingFlowState(const DvceArray5D<Real> &u0, 
                         int m, int k, int j, int i, 
                         Real x1v, Real x2v, Real x3v, 
                         Real gm1, const ProfileReader &profile) {
    // Calculate radius
    Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      
    // Get values from profiles via interpolation
    Real rho  = profile.GetDensity(r);
    Real temp = profile.GetTemperature(r);
    Real vr   = profile.GetVelocity(r);
    Real rmin = profile.GetRmin();

    // set vr to go to zero if r < rmin
    if (r < rmin) {
      vr *= (r / rmin);
    }
    
    // Calculate pressure from temperature
    Real press = rho * temp;
    
    // Set radial velocity components based on position
    Real v1 = 0.0, v2 = 0.0, v3 = 0.0;
    constexpr Real tiny = 1.0e-20;
    if (r > tiny) {  // Avoid division by zero
      // Negative sign accounts for inflowing vr
      v1 = -vr * x1v / r;
      v2 = -vr * x2v / r;
      v3 = -vr * x3v / r;
    }

    // Set state variables
    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = rho * v1;
    u0(m, IM2, k, j, i) = rho * v2;
    u0(m, IM3, k, j, i) = rho * v3;
    u0(m, IEN, k, j, i) = press/gm1 + 0.5*rho*(SQR(v1) + SQR(v2) + SQR(v3));
}

KOKKOS_INLINE_FUNCTION
void SetRotation(const DvceArray5D<Real> &u0, 
                 int m, int k, int j, int i, 
                 Real x1v, Real x2v, Real x3v, 
                 Real r_circ, Real v_circ) {
  // Calculate radius
  Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
  Real R = sqrt(x1v*x1v + x2v*x2v);
  
  // Calculate azimuthal velocity
  Real v1 = 0.0, v2 = 0.0, v3 = 0.0;
  constexpr Real tiny = 1.0e-20;
  if (r > tiny and R > tiny) {  // Avoid division by zero
    Real v_phi = 0.0;
    Real sin_theta = R / r;

    if (r < r_circ) {
      v_phi = v_circ * sin_theta;
    }
    if (r > r_circ) {
      v_phi = v_circ * sin_theta * r_circ / r;
    }
    
    // Calculate azimuthal velocity components
    v1 = -v_phi * x2v / R;
    v2 = v_phi * x1v / R;
    v3 = 0.0;
  }
  
  // Set state variables
  Real rho = u0(m, IDN, k, j, i);
  u0(m, IM1, k, j, i) += rho * v1;
  u0(m, IM2, k, j, i) += rho * v2;
  u0(m, IM3, k, j, i) += rho * v3;
  u0(m, IEN, k, j, i) += 0.5 * rho * (SQR(v1) + SQR(v2) + SQR(v3));
}

KOKKOS_INLINE_FUNCTION
void SetEquilibriumState(const DvceArray5D<Real> &u0, 
                         int m, int k, int j, int i, 
                         Real x1v, Real x2v, Real x3v, 
                         Real x1l, Real x1r, Real x2l, Real x2r, 
                         Real G, Real r_s, Real rho_s, Real m_g, 
                         Real a_g, Real z_g, Real r_m, Real rho_m,
                         Real gm1, const ProfileReader &disk_profile) {
    // Calculate radius
    Real R = sqrt(x1v * x1v + x2v * x2v);
    Real R1l = sqrt(x1l * x1l + x2v * x2v);
    Real R1r = sqrt(x1r * x1r + x2v * x2v);
    Real R2l = sqrt(x1v * x1v + x2l * x2l);
    Real R2r = sqrt(x1v * x1v + x2r * x2r);

    // Don't extrapolate past the last entry in the table
    if (R > disk_profile.GetRmax()) return;

    // Calculate Gravitational Potentials
    Real phi0    = GravPot(x1v, x2v, 0.0, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi0_1l = GravPot(x1l, x2v, 0.0, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi0_1r = GravPot(x1r, x2v, 0.0, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi0_2l = GravPot(x1v, x2l, 0.0, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi0_2r = GravPot(x1v, x2r, 0.0, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);

    Real phi   = GravPot(x1v, x2v, x3v, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi1r = GravPot(x1r, x2v, x3v, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi2l = GravPot(x1v, x2l, x3v, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi2r = GravPot(x1v, x2r, x3v, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);
    Real phi1l = GravPot(x1l, x2v, x3v, G, r_s, rho_s, m_g, a_g, z_g, r_m, rho_m);

    Real f_x1_ = -(phi1r - phi1l) / (x1r - x1l);
    Real f_x2_ = -(phi2r - phi2l) / (x2r - x2l);
    Real dPhi_dR = f_x1_ * x1v / R  + f_x2_ * x2v / R;

    // Compute sound speed squared
    Real temp = disk_profile.GetTemperature(R);
    Real cs2 = temp; // isothermal, no factor of gamma

    // Get densities from profiles via interpolation
    Real rho = disk_profile.GetDensity(R) * exp(-(phi - phi0) / cs2);
    Real rho_1l = disk_profile.GetDensity(R1l) * exp(-(phi1l - phi0_1l) / cs2);
    Real rho_1r = disk_profile.GetDensity(R1r) * exp(-(phi1r - phi0_1r) / cs2);
    Real rho_2l = disk_profile.GetDensity(R2l) * exp(-(phi2l - phi0_2l) / cs2);
    Real rho_2r = disk_profile.GetDensity(R2r) * exp(-(phi2r - phi0_2r) / cs2); 
    Real p_x1_ = temp*(rho_1r - rho_1l) / (x1r - x1l);
    Real p_x2_ = temp*(rho_2r - rho_2l) / (x2r - x2l);

    Real dP_dR_over_rho = 0.0;
    constexpr Real tiny = 1.0e-20;
    if (rho > tiny) {
      dP_dR_over_rho = (p_x1_ * x1v  + p_x2_ * x2v) / (R * rho);
    }
                      
    // Calculate circular velocity
    Real v_phi = sqrt(R*fmax(dP_dR_over_rho - dPhi_dR, 0.0));
    
    // Calculate azimuthal velocity
    Real v1 = 0.0, v2 = 0.0;
    if (R > tiny) {  // Avoid division by zero     
      // Calculate azimuthal velocity components
      v1 = -v_phi * x2v / R;
      v2 =  v_phi * x1v / R;
    }

    // Set state variables
    u0(m, IDN, k, j, i) += rho;
    u0(m, IM1, k, j, i) += rho * v1;
    u0(m, IM2, k, j, i) += rho * v2;
    u0(m, IM3, k, j, i) += 0.0;
    u0(m, IEN, k, j, i) += (rho * temp)/gm1 + 0.5 * rho * (SQR(v1) + SQR(v2));
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
  Real z_g = z_gal;
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

    Real phi1l = GravPot(x1l,x2v,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real phi1r = GravPot(x1r,x2v,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real f_x1_ = -(phi1r-phi1l)/(x1r-x1l);

    Real phi2l = GravPot(x1v,x2l,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real phi2r = GravPot(x1v,x2r,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real f_x2_ = -(phi2r-phi2l)/(x2r-x2l);

    Real phi3l = GravPot(x1v,x2v,x3l,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real phi3r = GravPot(x1v,x2v,x3r,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real f_x3_ = -(phi3r-phi3l)/(x3r-x3l);

    Real density = w0(m, IDN, k, j, i);
    Real src_x1 = bdt*density*f_x1_;
    Real src_x2 = bdt*density*f_x2_;
    Real src_x3 = bdt*density*f_x3_;

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
             Real M_gal, Real a_gal, Real z_gal,
             Real R200, Real rho_mean) {
  Real R = sqrt(x1*x1 + x2*x2);
  Real r2 = x1*x1 + x2*x2 + x3*x3;
  Real r = sqrt(r2);

  // Avoid division by zero
  constexpr Real tiny = 1.0e-20;
  r = fmax(r, tiny);

  // NFW component
  Real x = r / r_s;
  Real phi_NFW = -4 * M_PI * G * rho_s * SQR(r_s) * log(1 + x) / x;
  
  // Miyamoto-Nagai model
  Real phi_MN = -G * M_gal / sqrt(R*R + SQR(sqrt(x3*x3 + z_gal*z_gal) + a_gal));
  
  // Outer component
  Real term1 = (4.0 / 3.0) * pow(5 * R200, 1.5) * sqrt(r);
  Real term2 = (1.0 / 6.0) * r2;
  Real phi_Outer = 4 * M_PI * G * rho_mean * (term1 + term2);
  
  // Total potential
  Real phi = phi_NFW + phi_MN + phi_Outer;
  
  return phi;
}

//===========================================================================//
//                             User Boundary                                 //
//===========================================================================//

void UserBoundary(Mesh* pm) {
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
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  auto &u0 = pmbp->phydro->u0;
  auto &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto &profile = profile_reader;

  Real r_c = r_circ;
  Real v_c = v_circ;

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
      
      SetCoolingFlowState(u0, m, k, j, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j, i, x1v, x2v, x3v, r_c, v_c);
    }
  
    // Outer X1 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      int i_out = ie + i + 1;
      Real x1v = CellCenterX(i_out-is, indcs.nx1, x1min, x1max);
      
      SetCoolingFlowState(u0, m, k, j, i_out, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j, i_out, x1v, x2v, x3v,r_c, v_c);
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
      
      SetCoolingFlowState(u0, m, k, j, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j, i, x1v, x2v, x3v, r_c, v_c);
    }
  
    // Outer X2 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      int j_out = je + j + 1;
      Real x2v = CellCenterX(j_out-js, indcs.nx2, x2min, x2max);
      
      SetCoolingFlowState(u0, m, k, j_out, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j_out, i, x1v, x2v, x3v, r_c, v_c);
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
      
      SetCoolingFlowState(u0, m, k, j, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j, i, x1v, x2v, x3v, r_c, v_c);
    }
    
    // Outer X3 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
      int k_out = ke + k + 1;
      Real x3v = CellCenterX(k_out-ks, indcs.nx3, x3min, x3max);
      
      SetCoolingFlowState(u0, m, k_out, j, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k_out, j, i, x1v, x2v, x3v, r_c, v_c);
    }
  });
  
}

//===========================================================================//
//                              Refinement                                   //
//===========================================================================//

// Refine region based on density gradient threshold
void RefinementCondition(MeshBlockPack* pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &multi_d     = pmesh->multi_d;
  auto &three_d     = pmesh->three_d;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->phydro->u0;

  auto &ddens_thresh = ddens_threshold;

  par_for_outer("UserRefineCond",DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {

    Real team_ddmax;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, Real& ddmax) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/nx1;
      int i = (idx - k*nji - j*nx1) + is;
      j += js;
      k += ks;

      // Calculate density gradient
      Real d2 = SQR(u0(m,IDN,k,j,i+1) - u0(m,IDN,k,j,i-1));
      if (multi_d) {d2 += SQR(u0(m,IDN,k,j+1,i) - u0(m,IDN,k,j-1,i));}
      if (three_d) {d2 += SQR(u0(m,IDN,k+1,j,i) - u0(m,IDN,k-1,j,i));}
      ddmax = fmax((sqrt(d2)/u0(m,IDN,k,j,i)), ddmax);

    },Kokkos::Max<Real>(team_ddmax));

    if (team_ddmax > ddens_thresh) {refine_flag.d_view(m+mbs) = 1;}
    if (team_ddmax < 0.25*ddens_thresh) {refine_flag.d_view(m+mbs) = -1;}

  });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//===========================================================================//
//                            Post Main Loop                                 //
//===========================================================================//

void FreeProfile(ParameterInput *pin, Mesh *pm) {
  // Free Kokkos views before Kokkos::finalize is called
  profile_reader.~ProfileReader();
  disk_profile_reader.~ProfileReader();
}
