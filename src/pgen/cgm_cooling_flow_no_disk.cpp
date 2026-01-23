#include <iostream>
#include <cmath>
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
#include "utils/random.hpp"

//===========================================================================//
//                               Globals                                     //
//===========================================================================//

KOKKOS_INLINE_FUNCTION
void SetCoolingFlowState(const DvceArray5D<Real> &u0, 
                         int m, int k, int j, int i, 
                         Real x1v, Real x2v, Real x3v, 
                         Real gm1, const ProfileReader &profile);

KOKKOS_INLINE_FUNCTION 
Real GravPot(Real x1, Real x2, Real x3,
             Real G, Real r_s, Real rho_s, 
             Real M_gal, Real a_gal, Real z_gal,
             Real R200, Real rho_mean);

void UserSource(Mesh* pm, const Real bdt);
void GravitySource(Mesh* pm, const Real bdt);
void MassLossSource(Mesh* pm, const Real bdt);
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

  // Constants for mass removal
  Real mass_loss_rate;
  Real mass_loss_radius;

  // Constant for metallicity
  Real Z;

  // Profiles
  ProfileReader profile_reader;                
  
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
  user_ref_func   = RefinementCondition;

  if (global_variable::my_rank==0) {
    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Units                                         " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Unit Length         : " << 1.0/pmbp->punit->kpc() 
                                          << " kpc"    << std::endl;
    std::cout << "Unit Temperature    : " << 1.0/pmbp->punit->kelvin()
                                          << " K"     << std::endl;     
    std::cout << "Unit Number Density : " << std::pow(pmbp->punit->cm(),3) 
                                          << " cm^-3" << std::endl;        
    std::cout << "Unit Velocity       : " << 1.0/pmbp->punit->km_s() 
                                          << " km/s"  << std::endl;
    std::cout << "Unit Time           : " << 1.0/pmbp->punit->myr() 
                                          << " Myr"   << std::endl;
    std::cout << std::endl;
  }

  // Read in constants
  r_scale   = pin->GetReal("potential", "r_scale");
  rho_scale = pin->GetReal("potential", "rho_scale");
  m_gal     = pin->GetReal("potential", "mass_gal");
  a_gal     = pin->GetReal("potential", "scale_gal");
  z_gal     = pin->GetReal("potential", "z_gal");
  r_200     = pin->GetReal("potential", "r_200");
  rho_mean  = pin->GetReal("potential", "rho_mean");
  Z         = pin->GetOrAddReal("problem", "metallicity", 1.0/3);

  // Read the density gradient threshold for refinement
  ddens_threshold = pin->GetReal("problem", "ddens_max");

  mass_loss_rate = pin->GetReal("problem", "mass_loss_rate");
  mass_loss_radius = pin->GetReal("problem", "mass_loss_radius");
  
  // Output parameter information
  if (global_variable::my_rank == 0) {
    std::cout << "==============================================" << std::endl;
    std::cout << "Potential Parameters                          " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "r_scale             : " << r_scale    << std::endl;
    std::cout << "rho_scale           : " << rho_scale  << std::endl;
    std::cout << "m_gal               : " << m_gal      << std::endl;
    std::cout << "a_gal               : " << a_gal      << std::endl;
    std::cout << "z_gal               : " << z_gal      << std::endl;
    std::cout << "r_200               : " << r_200      << std::endl;
    std::cout << "rho_mean            : " << rho_mean   << std::endl;
    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Other Parameters                              " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "metallicity         : " << Z          << std::endl;
    std::cout << "ddens_threshold     : " << ddens_threshold << std::endl;
    std::cout << std::endl;
  }

  // Read the CGM cooling flow profile file
  std::string profile_file = pin->GetString("problem", "profile_file");
  try {
    ProfileReaderHost profile_reader_host;
    profile_reader_host.ReadProfiles(profile_file);
    // Create device-accessible reader
    profile_reader = profile_reader_host.CreateDeviceReader();
  } catch (const std::exception& e) {
    std::cerr << "Error loading profiles: " << e.what() << std::endl;
  }
  if (global_variable::my_rank==0) {
    std::cout << "Successfully loaded CGM profiles from " 
              << profile_file << std::endl;
  }

  if (restart) return;

  // Generate the initial turbulent field
  int nlow   = pin->GetOrAddInteger("problem", "cgm_turb_nlow", 1);
  int nhigh  = pin->GetOrAddInteger("problem", "cgm_turb_nhigh", 8);
  Real expo  = pin->GetOrAddReal("problem", "cgm_turb_expo", 5.0/3.0);
  Real v_rms = pin->GetOrAddReal("problem", "cgm_turb_rms", 0.1);
  Real cgm_turb_xscale = pin->GetOrAddReal("problem", "cgm_turb_xscale", 0.01);
  Real cgm_turb_yscale = pin->GetOrAddReal("problem", "cgm_turb_yscale", 0.01);
  Real cgm_turb_zscale = pin->GetOrAddReal("problem", "cgm_turb_zscale", 0.01);

  // Initialize random state
  RNG_State rstate;
  rstate.idum = -1;

  // Domain size
  Real lx = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real lz = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;
  Real dkx = 2.0*M_PI/lx;
  Real dky = 2.0*M_PI/ly;
  Real dkz = 2.0*M_PI/lz;

  // Count modes
  int nmodes = 0;
  for (int nkx = -nhigh; nkx <= nhigh; nkx++) {
    for (int nky = -nhigh; nky <= nhigh; nky++) {
      for (int nkz = -nhigh; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        int nsqr = nkx*nkx + nky*nky + nkz*nkz;
        if (nsqr >= nlow*nlow && nsqr <= nhigh*nhigh) {
          nmodes++;
        }
  }}}

  // Allocate arrays
  DualArray2D<Real> k_modes, aka, akb;
  Kokkos::realloc(k_modes, 3, nmodes);
  Kokkos::realloc(aka, 3, nmodes);
  Kokkos::realloc(akb, 3, nmodes);

  // Generate modes
  int nmode = 0;
  Real total_energy = 0.0;
  for (int nkx = -nhigh; nkx <= nhigh; nkx++) {
    for (int nky = -nhigh; nky <= nhigh; nky++) {
      for (int nkz = -nhigh; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        int nsqr = nkx*nkx + nky*nky + nkz*nkz;
        if (nsqr >= nlow*nlow && nsqr <= nhigh*nhigh) {
          Real kx = dkx*nkx, ky = dky*nky, kz = dkz*nkz;
          Real kiso = sqrt(kx*kx + ky*ky + kz*kz);

          k_modes.h_view(0, nmode) = kx;
          k_modes.h_view(1, nmode) = ky;
          k_modes.h_view(2, nmode) = kz;

          Real norm = 1.0/pow(kiso, (expo+2.0)/2.0);

          Real aval[3], bval[3];
          for (int dir = 0; dir < 3; dir++) {
            aval[dir] = norm * RanGaussianSt(&rstate);
            bval[dir] = norm * RanGaussianSt(&rstate);
          }

          Real k_dirs[3] = {kx, ky, kz};
          Real ka = kx*aval[0] + ky*aval[1] + kz*aval[2];
          Real kb = kx*bval[0] + ky*bval[1] + kz*bval[2];

          for (int dir = 0; dir < 3; dir++) {
            aval[dir] -= k_dirs[dir]*ka/(kiso*kiso);
            bval[dir] -= k_dirs[dir]*kb/(kiso*kiso);

            aka.h_view(dir,nmode) = aval[dir];
            akb.h_view(dir,nmode) = bval[dir];

            total_energy += 0.5*(aval[dir]*aval[dir] + bval[dir]*bval[dir]);
          }
          nmode++;
        }
      }
    }
  }

  Real v_norm = v_rms/sqrt(total_energy);

  k_modes.template modify<HostMemSpace>();
  k_modes.template sync<DevExeSpace>();
  aka.template modify<HostMemSpace>();
  aka.template sync<DevExeSpace>();
  akb.template modify<HostMemSpace>();
  akb.template sync<DevExeSpace>();

  // Capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nscalars = pmbp->phydro->nscalars;
  int nhydro = pmbp->phydro->nhydro;

  auto &u0 = pmbp->phydro->u0;
  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto &profile = profile_reader;

  Real G = pmbp->punit->grav_constant();
  Real r_s = r_scale;
  Real rho_s = rho_scale;
  Real m_g = m_gal;
  Real a_g = a_gal;
  Real z_g = z_gal;
  Real r_m = r_200;
  Real rho_m = rho_mean;

  Real Zsol = 0.02;
  Real Z_ = Z;

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
    
    // uniformly set metallicity
    u0(m, nhydro, k, j, i) = Z_ * Zsol * u0(m, IDN, k, j, i);

    // Compute turbulent velocities by summing Fourier modes
    Real vx = 0.0, vy = 0.0, vz = 0.0;
    for (int n = 0; n < nmodes; n++) {
      Real phase = k_modes.d_view(0,n)*x1v 
	         + k_modes.d_view(1,n)*x2v 
	         + k_modes.d_view(2,n)*x3v;
      Real cos_phase = cos(phase);
      Real sin_phase = sin(phase);

      vx += aka.d_view(0,n)*cos_phase - akb.d_view(0,n)*sin_phase;
      vy += aka.d_view(1,n)*cos_phase - akb.d_view(1,n)*sin_phase;
      vz += aka.d_view(2,n)*cos_phase - akb.d_view(2,n)*sin_phase;
    }

    // Attenuate in the center by 1 - Gaussian
    Real att = 1.0 - exp(-0.5 * ( SQR(x1v)/SQR(cgm_turb_xscale)
                                + SQR(x2v)/SQR(cgm_turb_yscale)
                                + SQR(x3v)/SQR(cgm_turb_zscale)));

    // Normalize to desired RMS velocity
    vx *= v_norm*att; vy *= v_norm*att; vz *= v_norm*att;

    // Add to conserved variables
    Real rho = u0(m,IDN,k,j,i);
    Real rho_v1 = u0(m,IM1,k,j,i);
    Real rho_v2 = u0(m,IM2,k,j,i);
    Real rho_v3 = u0(m,IM3,k,j,i);

    u0(m,IEN,k,j,i) += 0.5 * rho * (SQR(vx) + SQR(vy) + SQR(vz));
    u0(m,IEN,k,j,i) += rho_v1 * vx + rho_v2 * vy + rho_v3 * vz;
    u0(m,IM1,k,j,i) += rho * vx;
    u0(m,IM2,k,j,i) += rho * vy;
    u0(m,IM3,k,j,i) += rho * vz;
  });

  if (global_variable::my_rank==0) {
    std::cout << "Successfully initialized grid!" << std::endl;
  }

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

//===========================================================================//
//                              Source Terms                                 //
//===========================================================================//

void UserSource(Mesh* pm, const Real bdt) {
  GravitySource(pm, bdt);
  MassLossSource(pm, bdt);
  return;
}

void GravitySource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
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
    const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    const Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    const Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;

    const Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    const Real x1l = LeftEdgeX(i-is,   nx1, x1min, x1max);
    const Real x1r = LeftEdgeX(i+1-is, nx1, x1min, x1max);

    const Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    const Real x2l = LeftEdgeX(j-js,   nx2, x2min, x2max);
    const Real x2r = LeftEdgeX(j+1-js, nx2, x2min, x2max);

    const Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    const Real x3l = LeftEdgeX(k-ks,   nx3, x3min, x3max);
    const Real x3r = LeftEdgeX(k+1-ks, nx3, x3min, x3max);

    Real phi1l = GravPot(x1l,x2v,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real phi1r = GravPot(x1r,x2v,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);

    Real phi2l = GravPot(x1v,x2l,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real phi2r = GravPot(x1v,x2r,x3v,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);

    Real phi3l = GravPot(x1v,x2v,x3l,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);
    Real phi3r = GravPot(x1v,x2v,x3r,G,r_s,rho_s,m_g,a_g,z_g,r_m,rho_m);

    constexpr Real tiny = 1e-20;
    Real f_x1_ = -(phi1r - phi1l) / fmax(x1r - x1l, tiny);
    Real f_x2_ = -(phi2r - phi2l) / fmax(x2r - x2l, tiny);
    Real f_x3_ = -(phi3r - phi3l) / fmax(x3r - x3l, tiny);

    Real density = w0(m, IDN, k, j, i);
    Real src_x1 = bdt * density * f_x1_;
    Real src_x2 = bdt * density * f_x2_;
    Real src_x3 = bdt * density * f_x3_;

    u0(m,IM1,k,j,i) += src_x1;	
    u0(m,IM2,k,j,i) += src_x2;
    u0(m,IM3,k,j,i) += src_x3;
    u0(m,IEN,k,j,i) += (src_x1 * w0(m,IVX,k,j,i) +
                        src_x2 * w0(m,IVY,k,j,i) +
                        src_x3 * w0(m,IVZ,k,j,i));
    
  });

  return;
}

KOKKOS_INLINE_FUNCTION
Real GravPot(Real x1, Real x2, Real x3,
             Real G, Real r_s, Real rho_s, 
             Real M_gal, Real a_gal, Real z_gal,
             Real R200, Real rho_mean) {
  const Real R2 = fma(x1, x1 , x2*x2);
  const Real R  = sqrt(R2);
  const Real r2 = fma(x3 , x3 , R2);
  const Real r  = sqrt(fmax(r2, 1e-20));

  // NFW component
  Real x = r / r_s;
  Real phi_NFW = -4 * M_PI * G * rho_s * SQR(r_s) * log1p(x) / x;
  
  // Miyamoto-Nagai model
  Real phi_Plummer = -G * M_gal / sqrt(r2 + SQR(a_gal));
  //Real phi_MN = -G * M_gal / sqrt(R2 + SQR(sqrt(fma(x3 , x3 , z_gal*z_gal)) + a_gal));
  
  // Outer component
  Real c_outer = (4.0/3.0) * pow(5 * R200, 1.5);
  Real phi_Outer = 4 * M_PI * G * rho_mean * (c_outer * sqrt(r) + (1.0/6.0) * r2);
  
  // Total potential
  return phi_NFW + phi_Plummer + phi_Outer;
}

void MassLossSource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;

  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;

  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  auto &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &profile = profile_reader;

  Real ml_rate = mass_loss_rate;
  Real ml_radius = mass_loss_radius;

  par_for("user_source", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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
    if (r < 10.0) {
      SetCoolingFlowState(u0, m, k, j, i, x1v, x2v, x3v, gm1, profile);
    }

    // Real rho = w0(m,IDN,k,j,i);
    // Real temp = w0(m,IEN,k,j,i)/rho*gm1;

    // if (r < ml_radius) {
    //   Real mdot = ml_rate/(4./3.*M_PI*ml_radius*ml_radius*ml_radius);
    //   Real dm = min(bdt*mdot, 0.99*rho);
    //   u0(m,IDN,k,j,i) -= dm;
    //   u0(m,IM1,k,j,i) -= dm*w0(m,IVX,k,j,i);
    //   u0(m,IM2,k,j,i) -= dm*w0(m,IVY,k,j,i);
    //   u0(m,IM3,k,j,i) -= dm*w0(m,IVZ,k,j,i);
    //   u0(m,IEN,k,j,i) -= dm*(temp/gm1 + 0.5*(SQR(w0(m,IVX,k,j,i)) +
    //                                          SQR(w0(m,IVY,k,j,i)) +
    //                                          SQR(w0(m,IVZ,k,j,i))));

    //   // add some cooling to counteract excessive shock heating at the center
    //   if (temp>profile.GetTemperature(0.0)) {
    //     u0(m,IEN,k,j,i) += (profile.GetTemperature(0.0)-temp)*rho/gm1;
    //   }
    // }
  });

  return;
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

  int nscalars = pmbp->phydro->nscalars;
  int nhydro = pmbp->phydro->nhydro;

  auto &u0 = pmbp->phydro->u0;
  auto &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto &profile = profile_reader;

  Real Zsol = 0.02;
  Real Z_ = Z;

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
      u0(m, nhydro, k, j, i) = Z_ * Zsol * u0(m, IDN, k, j, i);
    }
  
    // Outer X1 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      int i_out = ie + i + 1;
      Real x1v = CellCenterX(i_out-is, indcs.nx1, x1min, x1max);
      
      SetCoolingFlowState(u0, m, k, j, i_out, x1v, x2v, x3v, gm1, profile);
      u0(m, nhydro, k, j, i_out) = Z_ * Zsol * u0(m, IDN, k, j, i_out);
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
      u0(m, nhydro, k, j, i) = Z_ * Zsol * u0(m, IDN, k, j, i);
    }
  
    // Outer X2 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      int j_out = je + j + 1;
      Real x2v = CellCenterX(j_out-js, indcs.nx2, x2min, x2max);
      
      SetCoolingFlowState(u0, m, k, j_out, i, x1v, x2v, x3v, gm1, profile);
      u0(m, nhydro, k, j_out, i) = Z_ * Zsol * u0(m, IDN, k, j_out, i);
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
      u0(m, nhydro, k, j, i) = Z_ * Zsol * u0(m, IDN, k, j, i);
    }
    
    // Outer X3 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
      int k_out = ke + k + 1;
      Real x3v = CellCenterX(k_out-ks, indcs.nx3, x3min, x3max);
      
      SetCoolingFlowState(u0, m, k_out, j, i, x1v, x2v, x3v, gm1, profile);
      u0(m, nhydro, k_out, j, i) = Z_ * Zsol * u0(m, IDN, k_out, j, i);
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
  auto &w0       = pmbp->phydro->w0;

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
      Real d2 = (SQR(u0(m,IDN,k,j,i+1) - u0(m,IDN,k,j,i-1))
               + SQR(u0(m,IDN,k,j+1,i) - u0(m,IDN,k,j-1,i))
               + SQR(u0(m,IDN,k+1,j,i) - u0(m,IDN,k-1,j,i)));
      ddmax = fmax((sqrt(d2)/u0(m,IDN,k,j,i)), ddmax);

      // Calculate pressure gradient
      Real p2 = (SQR(w0(m,IEN,k,j,i+1) - w0(m,IEN,k,j,i-1))
               + SQR(w0(m,IEN,k,j+1,i) - w0(m,IEN,k,j-1,i))
	       + SQR(w0(m,IEN,k+1,j,i) - w0(m,IEN,k-1,j,i)));
      ddmax = fmax((sqrt(p2)/w0(m,IEN,k,j,i)), ddmax);
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
}
