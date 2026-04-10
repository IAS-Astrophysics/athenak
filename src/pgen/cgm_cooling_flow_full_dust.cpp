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
#include "utils/sn_scheduler.hpp"
#include "utils/random.hpp"
#include "particles/particles.hpp"

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
             Real c_outer, Real rho_mean);

void UserSource(Mesh* pm, const Real bdt);
void GravitySource(Mesh* pm, const Real bdt);
void SNSource(Mesh* pm, const Real bdt);
void DustSource(Mesh* pm, const Real bdt);
void DensityCeilingSource(Mesh* pm, const Real bdt);
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

  // Constant for metallicity
  Real Z;
  Real zfloor;

  // Constants for SN
  Real r_inj;
  Real e_sn;
  Real m_ej;

  // Constants for dust
  Real d_z_init;
  Real d_z_sn;
  Real rho_gr;
  Real a_gr_s;
  Real a_gr_l;
  Real Z_solar = 0.02;

  // Profiles
  ProfileReader profile_reader;                
  ProfileReader disk_profile_reader;           
  
  // Refinment condition threshold
  Real ddens_threshold;

  // SN injection persistent buffer
  DvceArray2D<Real> sn_centers_buffer;
  Kokkos::View<int> sn_counter;

  // SN injection flags
  int last_sn_detect_cycle = -1;
  int num_sn_this_cycle;
}

//===========================================================================//
//                               Initialize                                  //
//===========================================================================//

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->phydro->nscalars != 3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Problem generator 'cgm_cooling_flow_full_dust' requires "
              << "<hydro>/nscalars = 3, but got " << pmbp->phydro->nscalars
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

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
  r_circ    = pin->GetReal("problem", "r_circ");
  v_circ    = pin->GetReal("problem", "v_circ");
  Z         = pin->GetOrAddReal("problem", "metallicity", 1.0/3);
  zfloor    = pin->GetOrAddReal("problem", "zfloor", 1.0e-20);

  // Read in SN injection radius and compute energy and mass injection densities
  r_inj = pin->GetReal("SN","r_inj"); // Input in code unis
  const Real sphere_vol = (4.0/3.0)*M_PI*std::pow(r_inj,3);
  const Real E_def = 1e51; // Default 10^51 ergs
  const Real M_def = 8.4;  // Default 8.4 solar masses
  e_sn  = pin->GetOrAddReal("SN","E_sn",E_def)*pmbp->punit->erg()/sphere_vol;
  m_ej  = pin->GetOrAddReal("SN","M_ej",M_def)*pmbp->punit->msun()/sphere_vol;

  // Read in dust model parameters
  d_z_init = pin->GetReal("dust","d_z_init"); 
  d_z_sn   = pin->GetReal("dust","d_z_sn"); 
  rho_gr   = pin->GetOrAddReal("dust", "rho_gr", 3.3);
  a_gr_s   = pin->GetReal("dust", "a_gr_s");
  a_gr_l   = pin->GetReal("dust", "a_gr_l");

  // Read the density gradient threshold for refinement
  ddens_threshold = pin->GetReal("problem", "ddens_max");
  
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
    std::cout << "r_circ              : " << r_circ     << std::endl;
    std::cout << "v_circ              : " << v_circ     << std::endl;
    std::cout << "metallicity         : " << Z          << std::endl;
    std::cout << "ddens_threshold     : " << ddens_threshold << std::endl;
    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Supernova Parameters                          " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "r_inj               : " << r_inj      << std::endl;
    std::cout << "e_sn                : " << e_sn       << std::endl;
    std::cout << "m_ej                : " << m_ej       << std::endl;
    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Dust Parameters                          " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "d_z_init            : " << d_z_init      << std::endl;
    std::cout << "d_z_sn              : " << d_z_sn        << std::endl;
    std::cout << std::endl;
  }

  // Read the CGM cooling flow profile file
  std::string profile_file = pin->GetString("problem", "profile_file");
  ProfileReaderHost profile_reader_host;
  profile_reader_host.ReadProfiles(profile_file);
  profile_reader = profile_reader_host.CreateDeviceReader();
  if (global_variable::my_rank==0) {
    std::cout << "Successfully loaded CGM profiles from " 
              << profile_file << std::endl;
  }

  // Read in the disk profile file
  std::string disk_profile_file = pin->GetString("problem", "disk_profile_file");
  ProfileReaderHost disk_profile_reader_host;
  disk_profile_reader_host.ReadProfiles(disk_profile_file);
  disk_profile_reader = disk_profile_reader_host.CreateDeviceReader();
  if (global_variable::my_rank==0) {
    std::cout << "Successfully loaded disk profiles from " 
	      << disk_profile_file << std::endl;
  }

  // Count total particles and initialize SN centers buffer
  pmy_mesh_->CountParticles();
  sn_centers_buffer = DvceArray2D<Real>("sn_centers_buffer", 3, pmy_mesh_->nprtcl_total);
  sn_counter = Kokkos::View<int>("sn_counter");
  if (global_variable::my_rank==0) {
    std::cout << "Successfully initialized " << pmy_mesh_->nprtcl_total
              << " particles!" << std::endl;
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
  int nhydro = pmbp->phydro->nhydro;

  int IZS = nhydro; // index of passive scalar for metallicity
  int IDS = nhydro+1; // index of passive scalar for small dust
  int IDL = nhydro+2; // index of passive scalar for large dust

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

  Real Zsol = Z_solar;
  Real Z_ = Z;
  Real dz_init = d_z_init;

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
    
    // uniformly set metallicity
    u0(m, IZS, k, j, i) = (1.-dz_init) * Z_  * Zsol * u0(m, IDN, k, j, i);
    // uniformly set dust, assuming equal masses in small and large grains
    u0(m, IDS, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
    u0(m, IDL, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);

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

KOKKOS_INLINE_FUNCTION
void SetRotation(const DvceArray5D<Real> &u0, 
                 int m, int k, int j, int i, 
                 Real x1v, Real x2v, Real x3v, 
                 Real r_circ, Real v_circ) {
  // Calculate radius
  Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
  Real R = sqrt(x1v*x1v + x2v*x2v);
  
  // Calculate azimuthal velocity
  Real vx = 0.0, vy = 0.0, vz = 0.0;
  constexpr Real tiny = 1.0e-20;
  if (r > tiny && R > tiny) {  // Avoid division by zero
    Real v_phi = 0.0;
    Real sin_theta = R / r;

    if (r < r_circ) {
      v_phi = v_circ * sin_theta;
    }
    if (r > r_circ) {
      v_phi = v_circ * sin_theta * r_circ / r;
    }
    
    // Calculate azimuthal velocity components
    vx = -v_phi * x2v / R;
    vy = v_phi * x1v / R;
  }
  
  // Set state variables
  Real rho = u0(m,IDN,k,j,i);
  Real rho_v1 = u0(m,IM1,k,j,i);
  Real rho_v2 = u0(m,IM2,k,j,i);
  Real rho_v3 = u0(m,IM3,k,j,i);

  u0(m,IEN,k,j,i) += 0.5 * rho * (SQR(vx) + SQR(vy) + SQR(vz));
  u0(m,IEN,k,j,i) += rho_v1 * vx + rho_v2 * vy + rho_v3 * vz;
  u0(m,IM1,k,j,i) += rho * vx;
  u0(m,IM2,k,j,i) += rho * vy;
  u0(m,IM3,k,j,i) += rho * vz;
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

    // Calculate Gravitational Potentialsi
    Real c_out = (4.0/3.0) * pow(5 * r_m, 1.5);
    
    Real phi0    = GravPot(x1v, x2v, 0.0, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi0_1l = GravPot(x1l, x2v, 0.0, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi0_1r = GravPot(x1r, x2v, 0.0, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi0_2l = GravPot(x1v, x2l, 0.0, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi0_2r = GravPot(x1v, x2r, 0.0, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);

    Real phi   = GravPot(x1v, x2v, x3v, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi1r = GravPot(x1r, x2v, x3v, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi2l = GravPot(x1v, x2l, x3v, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi2r = GravPot(x1v, x2r, x3v, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);
    Real phi1l = GravPot(x1l, x2v, x3v, G, r_s, rho_s, m_g, a_g, z_g, c_out, rho_m);

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

    // Combine cgm and disk material consistently
    Real rho_cgm  = u0(m, IDN, k, j, i);
    Real mom1_cgm = u0(m, IM1, k, j, i);
    Real mom2_cgm = u0(m, IM2, k, j, i);
    Real mom3_cgm = u0(m, IM3, k, j, i);
    Real E_cgm    = u0(m, IEN, k, j, i);

    Real KE_cgm  = 0.5*(SQR(mom1_cgm) + SQR(mom2_cgm) + SQR(mom3_cgm))/rho_cgm;
    Real Eth_cgm = E_cgm - KE_cgm;

    Real rho_tot  = rho_cgm + rho;
    Real mom1_tot = mom1_cgm + rho * v1;
    Real mom2_tot = mom2_cgm + rho * v2;
    Real mom3_tot = mom3_cgm;

    Real KE_tot = 0.5*(SQR(mom1_tot)+SQR(mom2_tot)+SQR(mom3_tot))/rho_tot;

    // Set state variables
    u0(m, IDN, k, j, i) = rho_tot;
    u0(m, IM1, k, j, i) = mom1_tot;
    u0(m, IM2, k, j, i) = mom2_tot;
    u0(m, IM3, k, j, i) = mom3_tot;
    u0(m, IEN, k, j, i) = Eth_cgm + (rho * temp)/gm1 + KE_tot;
}

//===========================================================================//
//                              Source Terms                                 //
//===========================================================================//

void UserSource(Mesh* pm, const Real bdt) {
  SNSource(pm, bdt);
  GravitySource(pm, bdt);
  DustSource(pm, bdt);
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
  Real rho_m = rho_mean;
  Real c_out = (4.0/3.0) * pow(5 * r_200, 1.5);

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

    Real phi1l = GravPot(x1l,x2v,x3v,G,r_s,rho_s,m_g,a_g,z_g,c_out,rho_m);
    Real phi1r = GravPot(x1r,x2v,x3v,G,r_s,rho_s,m_g,a_g,z_g,c_out,rho_m);

    Real phi2l = GravPot(x1v,x2l,x3v,G,r_s,rho_s,m_g,a_g,z_g,c_out,rho_m);
    Real phi2r = GravPot(x1v,x2r,x3v,G,r_s,rho_s,m_g,a_g,z_g,c_out,rho_m);

    Real phi3l = GravPot(x1v,x2v,x3l,G,r_s,rho_s,m_g,a_g,z_g,c_out,rho_m);
    Real phi3r = GravPot(x1v,x2v,x3r,G,r_s,rho_s,m_g,a_g,z_g,c_out,rho_m);

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
             Real c_outer, Real rho_mean) {
  const Real R2 = fma(x1, x1 , x2*x2);
  const Real R  = sqrt(R2);
  const Real r2 = fma(x3 , x3 , R2);
  const Real r  = sqrt(fmax(r2, 1e-20));

  // NFW component
  Real x = r / r_s;
  Real phi_NFW = -4 * M_PI * G * rho_s * SQR(r_s) * log1p(x) / x;
  
  // Miyamoto-Nagai model
  Real phi_MN = -G * M_gal / sqrt(R2 + SQR(sqrt(fma(x3 , x3 , z_gal*z_gal)) + a_gal));
  
  // Outer component
  Real phi_Outer = 4 * M_PI * G * rho_mean * (c_outer * sqrt(r) + (1.0/6.0) * r2);
  
  // Total potential
  return phi_NFW + phi_MN + phi_Outer;
}

void SNSource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;
  int nhydro = pmbp->phydro->nhydro;

  int IZS = nhydro; // index of passive scalar for metallicity
  int IDS = nhydro+1; // index of passive scalar for small dust
  int IDL = nhydro+2; // index of passive scalar for large dust

  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;

  Real dr = r_inj;
  Real dz_sn = d_z_sn;

  if (pm->ncycle != last_sn_detect_cycle) {
    last_sn_detect_cycle = pm->ncycle;

    auto &pr = pmbp->ppart->prtcl_rdata;
    auto &pi = pmbp->ppart->prtcl_idata;
    int npart = pmbp->ppart->nprtcl_thispack;

    auto gids = pmbp->gids;

    Real time = pm->time;
    int nrdata = pmbp->ppart->nrdata;
    Real unit_time = pmbp->punit->time_cgs();

    // Array of positions where SNs go off at this timestep
    auto &sn_centers = sn_centers_buffer;
    Kokkos::deep_copy(sn_counter, 0);
    auto d_counter = sn_counter;

    par_for("sn_source", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    
      Real next_sn_time = pr(nrdata-1, p);
    
      if (time > next_sn_time) {
        // Update particle sn tracking
        pi(2, p) += 1;
        int sn_idx = pi(2, p);
        Real par_t_create = pr(nrdata-3, p);
        Real cluster_mass = pr(nrdata-2, p);
        pr(nrdata-1, p) = GetNthSNTime(cluster_mass, par_t_create, unit_time, sn_idx);

        // Register SN center location and adjust for boundaries
        // Warning : if r_inj is large this will lead to unexpected behavior at the start
        int idx = Kokkos::atomic_fetch_add(&d_counter(), 1);
        int m = pi(PGID, p) - gids;
      
        Real x1min = size.d_view(m).x1min;
        Real x1max = size.d_view(m).x1max;
        Real x2min = size.d_view(m).x2min;
        Real x2max = size.d_view(m).x2max;
        Real x3min = size.d_view(m).x3min;
        Real x3max = size.d_view(m).x3max;

        sn_centers(0, idx) = min(max(pr(IPX,p), x1min+dr), x1max-dr);
        sn_centers(1, idx) = min(max(pr(IPY,p), x2min+dr), x2max-dr);
        sn_centers(2, idx) = min(max(pr(IPZ,p), x3min+dr), x3max-dr);
      }
    });

    DevExeSpace().fence();

    // Get number of SNe on host
    Kokkos::deep_copy(num_sn_this_cycle, d_counter);
  }

  if (num_sn_this_cycle > 0) {
    Real beta = bdt / pm->dt;
    Real e_sn_ = e_sn * beta;
    Real m_ej_ = m_ej * beta;
    Real Z_ej_ = 0.1;

    auto &sn_centers = sn_centers_buffer;
    int num_sn = num_sn_this_cycle;

    par_for("sn_injection", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x1min = size.d_view(m).x1min;
      Real x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real x2min = size.d_view(m).x2min;
      Real x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real x3min = size.d_view(m).x3min;
      Real x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      for (int sn = 0; sn < num_sn; ++sn) {
        Real sn_x = sn_centers(0, sn);
        Real sn_y = sn_centers(1, sn);
        Real sn_z = sn_centers(2, sn);
              
        // Calculate distance from SN center
        Real dx = x1v - sn_x;
        Real dy = x2v - sn_y;
        Real dz = x3v - sn_z;
        Real r = sqrt(dx*dx + dy*dy + dz*dz);
              
        // Inject energy if within injection radius
        if (r <= dr) {
          u0(m,IDN,k,j,i) += m_ej_;
          u0(m,IEN,k,j,i) += e_sn_;
          u0(m, IZS, k, j, i) += (1.-dz_sn) * Z_ej_ * m_ej_;
          u0(m, IDS, k, j, i) += 0.5 * dz_sn * Z_ej_ * m_ej_;
          u0(m, IDL, k, j, i) += 0.5 * dz_sn * Z_ej_ * m_ej_;
        }
      }

    });
  }
  
  return;
}

void DustSource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;
  int nhydro = pmbp->phydro->nhydro;

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gamma = eos.gamma;
  Real gm1 = gamma - 1.0;

  int IZS = nhydro; // index of passive scalar for metallicity
  int IDS = nhydro+1; // index of passive scalar for small dust
  int IDL = nhydro+2; // index of passive scalar for large dust

  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0; 
  
  // Get code unit variables for temperature conversions
  Real KELVIN = 1.0;
  Real KM_S = 1.0;
  if (pmbp->punit != nullptr) {
    KELVIN = pmbp->punit->temperature_cgs();
    KM_S = pmbp->punit->km_s();
  } else if (global_variable::my_rank == 0) {
    std::cout << "ERROR: <units> block missing..." << std::endl;
    std::exit(1);
  }

  Real rhog = rho_gr;
  Real a_s = a_gr_s;
  Real a_l = a_gr_l;

  Real Zsol = Z_solar;
  Real F_coag = 0.5;
  Real v_co = 0.1; // km/s, relative velocity for coagulation timescale

  // To prevent NaNs
  Real Z_tol = zfloor; 
  Real clamp_dust_tot = 1.0; // tmp var used for clamping later

  Real cs_unit = sqrt(gamma) / KM_S;

  par_for("gravity_source", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

    Real dens = w0(m, IDN, k, j, i);
    Real temp = w0(m, IEN, k, j, i) * gm1 / dens;
    Real cs_g = sqrt(temp) * cs_unit; // in km/s

    temp *= KELVIN; // convert to K


    // current metallicity
    // Real Zloc = Kokkos::max(Z_tol, w0(m,IZS,k,j,i));
    Real Zloc = w0(m,IZS,k,j,i);

    // current dust amount in the bin
    // Real D_s = Kokkos::max(0.5 * Z_tol, w0(m,IDS,k,j,i));
    // Real D_l = Kokkos::max(0.5 * Z_tol, w0(m,IDL,k,j,i));
    Real D_s = w0(m,IDS,k,j,i);
    Real D_l = w0(m,IDL,k,j,i);

    Real D_tot = D_s + D_l;
    Real d_s = D_s * dens;
    Real d_l = D_l * dens;
    
    Real rate_s = 0.0, rate_l = 0.0, rate_z = 0.0;
    Real tmp_rate_s = 0.0, tmp_rate_l = 0.0;
      
    //* Thermal sputtering
    // Tsai & Mathews 1995, McKinnon et al 2017
    Real t_sp = 102.0; // in Myr
    t_sp *= (1.e-3/dens);
    t_sp *= 1. + pow((temp/2.0e6), -2.5);

    if (D_s>0.5*Z_tol) tmp_rate_s = -3. * d_s / (t_sp * a_s);
    if (D_l>0.5*Z_tol) tmp_rate_l = -3. * d_l / (t_sp * a_l);

    rate_z += -tmp_rate_s - tmp_rate_l;

    rate_s += tmp_rate_s;
    rate_l += tmp_rate_l;

    tmp_rate_s = 0.0; 
    tmp_rate_l = 0.0;

    //* Accretion
    // Popping 2017
    Real t_ac = 75.0; // in Myr
    t_ac *= (20.0/dens);
    t_ac *= pow((50.0/temp), 0.5);
    t_ac *= Zsol/Zloc; // in Z_sol 

    t_ac /= 1.-(D_tot/Zloc);

    if (D_s>0.5*Z_tol) tmp_rate_s = (Zloc>Z_tol) ? d_s / (t_ac * a_s): 0.0;
    if (D_l>0.5*Z_tol) tmp_rate_l = (Zloc>Z_tol) ? d_l / (t_ac * a_l): 0.0;

    rate_z += -tmp_rate_s - tmp_rate_l;
    rate_s += tmp_rate_s;
    rate_l += tmp_rate_l;


    // Convert rate_z to metal mass
    rate_z *= 1.0;  //! Assuming Z_dust ~ 1

    tmp_rate_s = 0.0; 
    tmp_rate_l = 0.0;
    //* Shattering large grains, into small ones
    // From Dubois 2024
    Real t_shatt = 54.0; // in Myr
    t_shatt *= (1.0/dens); // in cm^-3
    t_shatt *= (rhog/3.0);  // in cm^-3
    t_shatt *= (0.01/D_l); 
    t_shatt *= (10/cs_g);  // in km/s

    if (D_l>0.5*Z_tol) tmp_rate_s = d_l / (t_shatt * a_l);
    rate_l += -tmp_rate_s;
    rate_s += tmp_rate_s;

    tmp_rate_s = 0.0; 
    tmp_rate_l = 0.0;
    // * Coagulation of small grains, into large ones
    // From Dubois 2024
    Real t_coag = 0.27; // in Myr
    t_coag *= (rhog/3.0);  // in g/cc
    t_coag *= (1.0e3/dens); // in cm^-3
    t_coag *= (0.01/D_s); 
    t_coag *= (0.1/v_co);  // in km/s
    t_coag *= F_coag;

    if (D_s>0.5*Z_tol) tmp_rate_l = d_s / (t_coag * a_s / 0.05);
    rate_s += -tmp_rate_l;
    rate_l += tmp_rate_l;

    //* Update metallicity and dust scalars with source terms
    //! These are w0 * dens
    u0(m, IZS, k, j, i) += bdt * rate_z;
    u0(m, IDS, k, j, i) += bdt * rate_s;
    u0(m, IDL, k, j, i) += bdt * rate_l;

    Real d_mass = u0(m, IDS, k, j, i) + u0(m, IDL, k, j, i); // for later

    // We check that scalars are guaranteed to be in [0,1] after all source terms are added.
    // Clamp metallicity 
    u0(m, IZS, k, j, i) = Kokkos::clamp(u0(m, IZS, k, j, i), 0.0, dens);
    // Clamp total dust-to-gas ratio to [0,1]
    // Dust to metal ratio is not clamped, as all of gas metals can be locked in dust
    // Also dust might have been created in a metal-rich area and moved...
    Real clamp_d_mass = (d_mass>0.0) ? Kokkos::clamp(d_mass, 0.0, dens) / d_mass : 0.0;

    // Scaled to sum to clamp_dust_tot 
    u0(m, IDS, k, j, i) *= clamp_d_mass;
    u0(m, IDL, k, j, i) *= clamp_d_mass;
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
  int nhydro = pmbp->phydro->nhydro;

  int IZS = nhydro; // index of passive scalar for metallicity
  int IDS = nhydro+1; // index of passive scalar for small dust
  int IDL = nhydro+2; // index of passive scalar for large dust


  auto &u0 = pmbp->phydro->u0;
  auto &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  auto &profile = profile_reader;

  Real r_c = r_circ;
  Real v_c = v_circ;

  Real Zsol = 0.02;
  Real Z_ = Z;
  Real dz_init = d_z_init;

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
      u0(m, IZS, k, j, i) = (1. - dz_init) * Z_ * Zsol * u0(m, IDN, k, j, i);
      u0(m, IDS, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
      u0(m, IDL, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
    }
  
    // Outer X1 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      int i_out = ie + i + 1;
      Real x1v = CellCenterX(i_out-is, indcs.nx1, x1min, x1max);
      
      SetCoolingFlowState(u0, m, k, j, i_out, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j, i_out, x1v, x2v, x3v, r_c, v_c);
      u0(m, IZS, k, j, i_out) = (1. - dz_init) * Z_ * Zsol * u0(m, IDN, k, j, i_out);
      u0(m, IDS, k, j, i_out) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i_out);
      u0(m, IDL, k, j, i_out) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i_out);
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
      u0(m, IZS, k, j, i) = (1. - dz_init) * Z_ * Zsol * u0(m, IDN, k, j, i);
      u0(m, IDS, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
      u0(m, IDL, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
    }
  
    // Outer X2 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      int j_out = je + j + 1;
      Real x2v = CellCenterX(j_out-js, indcs.nx2, x2min, x2max);
      
      SetCoolingFlowState(u0, m, k, j_out, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k, j_out, i, x1v, x2v, x3v, r_c, v_c);
      u0(m, IZS, k, j_out, i) = (1. - dz_init) * Z_ * Zsol * u0(m, IDN, k, j_out, i);
      u0(m, IDS, k, j_out, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j_out, i);
      u0(m, IDL, k, j_out, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j_out, i);
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
      u0(m, IZS, k, j, i) = (1. - dz_init) * Z_ * Zsol * u0(m, IDN, k, j, i);
      u0(m, IDS, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
      u0(m, IDL, k, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k, j, i);
    }
    
    // Outer X3 boundary
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
      int k_out = ke + k + 1;
      Real x3v = CellCenterX(k_out-ks, indcs.nx3, x3min, x3max);
      
      SetCoolingFlowState(u0, m, k_out, j, i, x1v, x2v, x3v, gm1, profile);
      SetRotation(u0, m, k_out, j, i, x1v, x2v, x3v, r_c, v_c);
      u0(m, IZS, k_out, j, i) = (1. - dz_init) * Z_ * Zsol * u0(m, IDN, k_out, j, i);
      u0(m, IDS, k_out, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k_out, j, i);
      u0(m, IDL, k_out, j, i) = 0.5 * dz_init * Z_ * Zsol * u0(m, IDN, k_out, j, i);
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
  profile_reader = ProfileReader();
  disk_profile_reader = ProfileReader();
  sn_centers_buffer = DvceArray2D<Real>();
  sn_counter = Kokkos::View<int>();
}
