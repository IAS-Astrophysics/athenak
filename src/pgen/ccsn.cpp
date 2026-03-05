//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ccsn.cpp
//  \brief Problem generator for CCSN simulations. Only works when ADM is enabled.
//         Stellar profiles from Woosley:2002zz

#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \class StellarProfile
//  \brief class to read and interpolate 1d stellar profiles
class StellarProfile {
  public:
    enum vars
    {
      alp = 0,
      gxx = 1,
      mass = 2,
      vel = 3,
      rho = 4,
      temp = 5,
      ye = 6,
      press = 7,
      phi = 8,
      nvars = 9,
    };

  public:
    StellarProfile(ParameterInput *pin);
    ~StellarProfile();

    Real Eval(int var, Real r) const;

  private:
    int siz;
    Real *pr;
    Real *pvars[nvars];
    Real *Phi;
};

// Rotation laws
KOKKOS_INLINE_FUNCTION
Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A);

KOKKOS_INLINE_FUNCTION
Real OmegaGRB(Real rad, Real Omega_0, Real Omega_A,
	      Real rfe,Real drtrans, Real dropfac);
  
// Vector potential 
// https://arxiv.org/abs/1004.2896
// https://arxiv.org/abs/1403.1230
KOKKOS_INLINE_FUNCTION
Real VectorPotential(Real xp, Real yp, Real zp,
                     Real B0_amp, Real B0_rad,
                     int component);
    
//----------------------------------------------------------------------------------------
//! \class Deleptonization
//  \brief class for deleptonization scheme by astro-ph/0504072
//         Uses updated fits from 1701.02752
  
// Conversion mass-density units from geom.solar to CGS 
#define UDENS (6.1762691458861632e+17)

class Deleptonization {
  public:
  Deleptonization(ParameterInput* pin) {
    // Deleptonization scheme parameters
    E_nu_avg = pin->GetOrAddReal("deleptonization", "E_nu_avg_MeV", 10.0);
    rho_trap = pin->GetOrAddReal("deleptonization", "rho_trap_cgs", 1e12) / UDENS;

    // Fit parameters. Default values are SFHo from 1701.02752
    log10_rho1 = pin->GetOrAddReal("deleptonization", "log10_rho1", 7.795);
    log10_rho2 = pin->GetOrAddReal("deleptonization", "log10_rho2", 12.816);
    Ye_2 = pin->GetOrAddReal("deleptonization", "Ye_2", 0.308);
    Ye_c = pin->GetOrAddReal("deleptonization", "Ye_c", 0.0412);
    Ye_H = pin->GetOrAddReal("deleptonization", "Ye_H", 0.257);
  };

  Real E_nu_avg;
  Real rho_trap;
  
  Real Ye_of_rho(Real rho) const {
    Real const Ye_1 = 0.5;
    Real const log10_rhoH = 15;

    Real const log10_rho = log10(rho * UDENS);
    Real const x = max(-1.0, min(1.0, (2 * log10_rho - log10_rho2 - log10_rho1) /
                                        (log10_rho2 - log10_rho1)));
    Real const m = (Ye_H - Ye_2) / (log10_rhoH - log10_rho2);

    if (log10_rho > log10_rho2) {
      return Ye_2 + m * (log10_rho - log10_rho2);
    } else {
      return 0.5 * (Ye_2 + Ye_1) + 0.5 * x * (Ye_2 - Ye_1) +
        Ye_c * (1 - abs(x) + 4 * abs(x) * (abs(x) - 0.5) * (abs(x) - 1));
    }
  };

  ~Deleptonization() {};
  
  private:
    Real log10_rho1;
    Real log10_rho2;
    Real Ye_2;
    Real Ye_c;
    Real Ye_H;
}

}// namespace

// Mass threshold for refinement condition
Real amr_delta_min_m;
Real amr_delta_max_m;

void CCSNHistory(HistoryData *pdata, Mesh *pm);
void CCSNRefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for massive star progenitor
//  Compile with '-D PROBLEM=ccsn' to enroll as user-specific problem generator
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CCSN problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_ref_func  = &CCSNRefinementCondition;
  user_hist_func = &CCSNHistory;
  
  if (restart) return;
  
  auto& w0_ = pmbp->pmhd->w0;
  int& nvars_ = pmbp->pmhd->nmhd;
  int& nscal_ = pmbp->pmhd->nscalars;
  
  // Parameters
  bool interpolate_Phi = pin->GetOrAddBoolean("problem", "interpolate_Phi", false);
  
  const Real Omega_0 = pin->GetOrAddReal("problem", "Omega_0", 0.0); // all in code units
  const Real Omega_A = pin->GetOrAddReal("problem", "Omega_A", 0.0);  
  const Real B0_amp = pin->GetOrAddReal("problem", "B0_amp", 0.0);
  const Real B0_rad = pin->GetOrAddReal("problem", "B0_rad", 0.0);
  const Real rho_cut = pin->GetOrAddReal("problem", "rho_cut", 1e-5); 

  // Mesh refinement parameters
  const Real INF = std::numeric_limits<Real>::infinity();
  
  amr_delta_min_m = pin->GetOrAddReal("problem", "amr_delta_min_m", -INF);
  amr_delta_max_m = pin->GetOrAddReal("problem", "amr_delta_max_m", INF);

  // Others
  const Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);
  
  // Read stellar profile
  StellarProfile *pstar = new StellarProfile(pin);

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ie = indcs.ie;
  int &je = indcs.je;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  auto &eos_ = eos;
  par_for("pgen_ccsn", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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
    
    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    Real s = sqrt(SQR(x1v) + SQR(x2v));
    
    // Weak-field metric
    Real alp, A;
    if (interpolate_Phi) {
      const Real Phi = pstar->Eval(StellarProfile::phi, r);
      A = 1.0 - 2.0 * Phi;
      alp = sqrt(1 + 2.0 * Phi);
    }
    else {
      A = pstar->Eval(StellarProfile::gxx, r);
      alp = pstar->Eval(StellarProfile::alp, r);
    }
    
    // Matter fields
    Real rho = pstar->Eval(StellarProfile::rho, r);
    Real press = pstar->Eval(StellarProfile::press, r);
    Real ye = pstar->Eval(StellarProfile::ye, r);
    
    //TODO overwrite some of these with EOS value?
    
    Real vr = pstar->Eval(StellarProfile::vel, r);
    Real vx = vr * x1v / r;
    Real vy = vr * x2v / r;
    Real vz = vr * x3v / r;
    
    // Add rotation
    if (Omega_0 > 0.0 and Omega_A > 0.0) {
      Real const Omega = OmegaLaw(s, Omega_0, Omega_A); 
      vx -= Omega * x2v; // s * sinphi;
      vy += Omega * x1v; // s * cosphi;
    }
    
    // Cut profile below given density
    if (rho_cut > 0.0 and rho < rho_cut) {
      rho = 0.0;
      press = 0.0;
      vx = 0.0;
      vy = 0.0;
      vz = 0.0;
    }
    
    //TODO Set ATM floors: rho, press, ye (TAB)
    //Real ye = ye_atmo; 
    //auto &use_ye_ = use_ye; //FIXME
    
    // Set hydrodynamic quantities
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IPR,k,j,i) = press;
    //w0(m, IEN, k, j, i) = egas; //FIXME

    Real W = 1.0 / sqrt(1.0 - A * (vx * vx + vy * vy + vz * vz));
        
    w0_(m,IVX,k,j,i) = W * vx;
    w0_(m,IVY,k,j,i) = W * vy;
    w0_(m,IVZ,k,j,i) = W * vz;
    
    if (use_ye && nscal >= 1) {
      auto &nvars = nvars_;
      auto &nscal = nscal_;
      w0_(m,nvars,k,j,i) = ye;
    }
    
    // Set ADM variables
    adm.alpha(m,k,j,i) = alp;
    adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = A;
    adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
    
    Real det = adm::SpatialDet(
              adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm.psi4(m,k,j,i) = pow(det, 1./3.);

    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });

  delete pstar;
  
  // Add magnetic field
    
  // Compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pmbp->nmb_thispack;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);
  
  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
     Real &x1min = size.d_view(m).x1min;
     Real &x1max = size.d_view(m).x1max;
     int nx1 = indcs.nx1;
     Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
     Real x1f = LeftEdgeX(i-is,nx1,x1min,x1max);
     
     Real &x2min = size.d_view(m).x2min;
     Real &x2max = size.d_view(m).x2max;
     int nx2 = indcs.nx2;
     Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
     Real x2f = LeftEdgeX(j-js,nx2,x2min,x2max);
     
     Real &x3min = size.d_view(m).x3min;
     Real &x3max = size.d_view(m).x3max;
     int nx3 = indcs.nx3;
     Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
     Real x3f = LeftEdgeX(k-ks,nx3,x3min,x3max);
     
     Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
     Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
     Real x3fp1 = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
     Real dx1 = size.d_view(m).dx1;
     Real dx2 = size.d_view(m).dx2;
     Real dx3 = size.d_view(m).dx3;
     
     a1(m,k,j,i) = VectorPotential(x1v, x2f, x3f, B0_amp, B0_rad, 1);
     a2(m,k,j,i) = VectorPotential(x1f, x2v, x3f, B0_amp, B0_rad, 2);
     a3(m,k,j,i) = VectorPotential(x1f, x2f, x3v, B0_amp, B0_rad, 3);
     
     // When neighboring MeshBock is at finer level, compute vector potential as sum of
     // values at fine grid resolution.  This guarantees flux on shared fine/coarse
     // faces is identical.
     
     // Correct A1 at x2-faces, x3-faces, and x2x3-edges
     if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
	 (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
	 (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
	 (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
	 (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
	 (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
	 (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
	 (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
       Real xl = x1v + 0.25*dx1;
       Real xr = x1v - 0.25*dx1;
       a1(m,k,j,i) = 0.5*(
			  VectorPotential(xl, x2f, x3f, B0_amp, B0_rad, 1) +
			  VectorPotential(xr, x2f, x3f, B0_amp, B0_rad, 1)
			  );
     }

     // Correct A2 at x1-faces, x3-faces, and x1x3-edges
     if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
	 (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
	 (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
	 (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
	 (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
	 (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
	 (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
	 (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
	 (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
	 (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
       Real xl = x2v + 0.25*dx2;
       Real xr = x2v - 0.25*dx2;
       a2(m,k,j,i) = 0.5*(
			  VectorPotential(x1f, xl, x3f, B0_amp, B0_rad, 2) +
			  VectorPotential(x1f, xr, x3f, B0_amp, B0_rad, 2)
			  );
     }

     // Correct A3 at x1-faces, x2-faces, and x1x2-edges
     if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
	 (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
	 (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
	 (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
	 (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
	 (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
	 (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
	 (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
	 (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
	 (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
	 (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
	 (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
       Real xl = x3v + 0.25*dx3;
       Real xr = x3v - 0.25*dx3;
       a3(m,k,j,i) = 0.5*(
			  VectorPotential(x1f, x2f, xl, B0_amp, B0_rad, 3) +
			  VectorPotential(x1f, x2f, xr, B0_amp, B0_rad, 3)
			  );
     }
  });
    
  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    
    b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
		       (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
		       (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
		       (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);
    
    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
			   (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
			   (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
			   (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
 });
  
  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_Bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });
  
  // Mesh block info for loop limits 
  // auto &indcs = pmy_mesh_->mb_indcs;
  // int &ng = indcs.ng;
  // int n1 = indcs.nx1 + 2*ng;
  // int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  // int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  // Convert primitives to conserved
  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
  
  // Z4c
  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
    }
  }
  
  return;
}

//----------------------------------------------------------------------------------------
// \brief Deleptonization scheme
//

//TODO Here below is the relevant GR-Athena++ code, 
//     triggered during: GRMHD_Z4c::UserWork
//     Shall be adapted for the K: embedded in a class/scheduled after the tstep etc etc

/*
void Mesh::UserWorkInLoop()
{
  bool active = pin->GetOrAddBool("deleptonization", "active", false);
  if (not active) return;

  Deleptonization *pdelept = new Deleptonization(pin);

  const Real E_nu_avg = delept_E_nu_avg;
  const Real rho_trap = delept_rho_trap;

  MeshBlock * pmb = pblock;
  Coordinates * pco;
  EquationOfState * peos = pblock->peos;
  Hydro *ph;
  PassiveScalars *ps;
  Field *pf;
  Z4c * pz4c;

  const Real mb = peos->GetEOS().GetBaryonMass();

  while (pmb != nullptr)
  {
      peos = pmb->peos;
      ph = pmb->phydro;
      ps = pmb->pscalars;
      pf = pmb->pfield;
      pco = pmb->pcoord;

      AA temperature;
      temperature.InitWithShallowSlice(ph->derived_ms, IX_T, 1);

      CC_GLOOP3(k, j, i)
      {
        const Real rho = ph->w(IDN,k,j,i);
        Real & Y_e = ps->r(IYE,k,j,i);
        Real Y[MAX_SPECIES]  { 0 };
        Y[IYE] = Y_e;

        // 
        // Electron fraction update
        //
        
        const Real Y_e_bar = pdelept->Ye_of_rho(rho);

        const Real delta_Y_e = std::min( 0.0, (Y_e_bar - Y_e) );

        if (delta_Y_e < 0.0)
        {
          Y_e += delta_Y_e;  // N.B: Update directly to state-vector

          //
          // Entropy update 
          //
          
          const Real n = rho / mb;
          const Real p = ph->w(IPR,k,j,i);
          const Real T = peos->GetEOS().GetTemperatureFromP(n, p, Y);
          Real s = peos->GetEOS().GetEntropyPerBaryon(n, T, Y);

          // Prepare chemical potentials (BD: TODO- check)
          // mu_nu := mu_e - mu_n + mu_p
          
          //const Real mu_b = peos->GetEOS().GetBaryonChemicalPotential(n, T, Y);
          //const Real mu_q = peos->GetEOS().GetChargeChemicalPotential(n, T, Y);
          //const Real mu_l = peos->GetEOS().GetElectronLeptonChemicalPotential(n, T, Y);
          //const Real mu_n = mu_b;
          //const Real mu_p = mu_b + mu_q;
          //const Real mu_e = mu_l - mu_q;
          //const Real mu_nu = mu_e - mu_n + mu_p;

          const Real mu_nu = peos->GetEOS().GetElectronLeptonChemicalPotential(n, T, Y);

          if ((mu_nu > pdelept.E_nu_avg) &&  // MeV units
              (rho < pdelept.rho_trap))      // code units
          {
            s -= delta_Y_e * (mu_nu - E_nu_avg) / T;
          }

          //
          // Y_e and s has been updated
          // Recompute Temperature and pressure
          //

          Y[IYE] = Y_e;   
          const Real Tnew = peos->GetEOS().GetTemperatureFromEntropy(n, s, Y);
          //const Real e = peos->GetEOS().GetEnergy(n, Tnew, Y);
          const Real pnew = peos->GetEOS().GetPressure(n, Tnew, Y);

          // push back to prim state vector
          ph->w(IPR,k,j,i) = pnew;
          temperature(k,j,i) = Tnew;

          //
          // Update conservatives
          //
          
          peos->PrimitiveToConserved(
            ph->w,
            ps->r,
            pf->bcc,
            ph->u,
            ps->s,
            pco,
            i, i,
            j, j,
            k, k
          );
          
        } // if (delta_Y_e < 0.0)
        
      } // CC_GLOOP3(k, j, i)

      pmb = pmb->next;
      
    } // while (pmb != nullptr)

    delete pdelept;
    
};

*/


//----------------------------------------------------------------------------------------
// \brief History function
//

//TODO add central, bounce/shock quantities

void CCSNHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    // coompute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need MPI_MAX
  // and MPI_MIN operations instead. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

//----------------------------------------------------------------------------------------
// \brief Refinement condition based on min/max mass per mesh block
//

void CCSNRefinementCondition(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  
  par_for_outer(
    "CCSN_AMR::MassMB", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_mloc;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &mloc) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;

	  Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
          mloc = vol * u0(m,IDN,k,j,i); //CHECK this assumes D is densitized
        },
        Kokkos::Sum<Real>(team_mloc));

      if (team_mloc > amr_delta_max_m) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmax < amr_delta_min_m) {
        refine_flag.d_view(m + mbs) = -1;
      }
    });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}


namespace {
 
StellarProfile::StellarProfile(ParameterInput *pin) : siz(0) {
  string fname = pin->GetString("problem", "initial_data_file");

  // count number of lines in file
  std::ifstream in;
  in.open(fname.c_str());
  std::string line;
  if (!in.is_open())
  {
    stringstream msg;
    msg << "### FATAL ERROR problem/progenitor: " << string(fname) << " "
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }
  else
  {
    while (std::getline(in, line))
    {
      if (line.find("#") < line.length())
      {
        // line with comment (ignored)
      }
      else
      {
        ++siz;
      }
    }
    in.close();
  }

  // allocate & parse file 
  pr = new Real[siz];
  for (int vi = 0; vi < nvars; ++vi)
  {
    pvars[vi] = new Real[siz];
  }

  // parse
  in.open(fname.c_str());
  int el = 0;
  while (std::getline(in, line))
  {
    if (line.find("#") < line.length())
    {
      // comment; pass line
    }
    else if (line.find(" ") < line.length())
    {
      // process elements#
      std::vector<std::string> vs;
      tokenize(line, ' ', vs);

      pr[el] = std::stod(vs[0]);

      for (int ix=mass; ix<nvars; ++ix)
      {
        pvars[ix][el] = std::stod(vs[ix-mass+1]);
      }
      el++;
    }
  }
  in.close();

  // Compute metric functions

  // mass at face 1 (=0 at face 0)
  Real massi = 4.0/3.0 * PI * pow(pr[1],3) * pvars[rho][0];
  
  Phi = new Real[siz];
  Phi[0] = 0.0;
  Phi[1] = massi/pr[1];
  for (int i = 2; i < siz; ++i) {
    Real dr = pr[i] - pr[i-1];
    massi += ( 4.0 * PI * pow(pr[i-1], 2) * pvars[rho][i-1] )*dr;
    Phi[i] = Phi[i-1] + ( massi /( pow(pr[i], 2) ) ) * dr;
  }
  Real const dPhi = Phi[siz-1] - massi/pr[siz-1];
  for (int i = 0; i < siz; ++i) {
    Phi[i] -= dPhi; 
    pvars[alp][i] = sqrt(1. + 2. * Phi[i]);
    pvars[gxx][i] = 1. - 2. * Phi[i];
   }
}

StellarProfile::~StellarProfile() {
  delete[] pr;
  for (int vi = 0; vi < nvars; ++vi)  {
    delete[] pvars[vi];
  }
  delete[] Phi;
}

Real StellarProfile::Eval(int vi, Real rad) const {
  assert(vi >= 0 && vi < nvars);
  if (rad <= pr[0]) {
    return pvars[vi][0];
  }
  if (rad >= pr[siz - 1]) {
    return pvars[vi][siz - 1];
  }
  int offset = lower_bound(pr, pr + siz, rad) - pr - 1;
  Real lam = (rad - pr[offset]) / (pr[offset + 1] - pr[offset]);
  return pvars[vi][offset] * (1 - lam) + pvars[vi][offset + 1] * lam;
}

// Standard rotation law
KOKKOS_INLINE_FUNCTION
Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A)
{
  return Omega_0/(1.0 + SQR(rad/Omega_A));
}

// Rotation law motivated by GRB progenitor
// https://arxiv.org/abs/astro-ph/0508175
// https://arxiv.org/abs/1012.1853
// This implementation is taken from Zelmani
Real OmegaGRB_lam(Real rad, Real drtrans, Real rfe)
{
  return 0.5*(1.0 + tanh((rad - rfe)/drtrans));
}

KOKKOS_INLINE_FUNCTION
Real OmegaGRB(Real rad, Real Omega_0, Real Omega_A,
	      Real rfe,Real drtrans, Real dropfac)
{ 
  Real const lam = OmegaGRB_lam(rad, drtrans, rfe);
  Real const Omega_r = OmegaLaw(rad, Omega_0, Omega_A);
  Real const Omega_rfe = OmegaLaw(rfe, Omega_0, Omega_A);
  Real const fac = dropfac/(1 + pow(abs(rad-rfe)/Omega_A, 1.0/3.0));  
  return ( (1.0-lam) * Omega_r + lam* Omega_rfe/fac );
}

// Vector potential A_i model used in e.g.
// https://arxiv.org/abs/1004.2896
// https://arxiv.org/abs/1403.1230
KOKKOS_INLINE_FUNCTION
Real VectorPotential(Real xp, Real yp, Real zp,
                     Real B0_amp, Real B0_rad,
                     //Real &Ax, Real &Ay, Real &Az,
                     int component) {
    const Real rad_cyl = sqrt(SQR(xp) + SQR(yp));
    const Real oo_rad_cyl = 1.0/rad_cyl;

    const Real rad = sqrt(SQR(xp) + SQR(yp) + SQR(zp));
    const Real oo_rad = 1.0/rad;

    const Real drdx = xp * oo_rad;
    const Real drdy = yp * oo_rad;
    const Real drdz = zp * oo_rad;

    const Real dthdx = (xp * zp) * SQR(oo_rad) * oo_rad_cyl;
    const Real dthdy = (yp * zp) * SQR(oo_rad) * oo_rad_cyl;
    const Real dthdz = - rad_cyl * SQR(oo_rad);

    const Real dphdx = - yp * SQR(oo_rad_cyl);
    const Real dphdy = xp * SQR(oo_rad_cyl);
    const Real dphdz = 0.0;

    Real Ar = 0.0;
    Real Ath = 0.0;
    Real Aph = 0.0;
    
    if (B0_amp > 0.0 && B0_rad > 0.0) {
      Aph = B0_amp * rad_cyl /( pow(rad,3) + pow(B0_rad,3) );
    }

    Ax = drdx * Ar + dthdx * Ath + dphdx * Aph;
    Ay = drdy * Ar + dthdy * Ath + dphdy * Aph;
    Az = drdz * Ar + dthdz * Ath + dphdz * Aph;

    if (component == 1) {
      return Ax;
    }
    if (component == 2) {
      return Ay;
    }
    if (component == 3) {
      return Az;
    }
    assert(false);
}
  
} // namespace
