//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_torus.cpp
//  \brief Problem generator for Fishbone-Moncrief torus, specialized for cartesian
//  Kerr-Schild coordinates.  Based on gr_torus.cpp in Athena++, with edits by CJW and SR.
//  Simplified and implemented in Kokkos by JMS.

#include <Kokkos_Random.hpp>

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <cmath>      // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <cfloat>
#include <stdio.h>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

// function prototypes

KOKKOS_INLINE_FUNCTION
static Real CalculateLFromRPeak(Real r);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

// Global variables
static Real mass, spin;                            // black hole parameters
static Real l_peak;                                // fixed torus parameters
static Real psi, sin_psi, cos_psi;                 // tilt parameters
static Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
static Real potential_cutoff;                      // sets region of torus to magnetize
static Real potential_r_pow, potential_rho_pow;    // set how vector potential scales
static Real beta_min;                              // min ratio of gas to mag pressure
static int sample_n_r, sample_n_theta;             // number of cells in 2D sample grid
static int sample_n_phi;                           // number of cells in 3D sample grid
static Real sample_r_rat;                          // sample grid geometric spacing ratio
static Real sample_cutoff;                         // density cutoff for sample grid
static Real x1_min, x1_max, x2_min, x2_max;        // 2D limits in chosen coordinates
static Real x3_min, x3_max;                        // 3D limits in chosen coordinates
static Real r_min, r_max, theta_min, theta_max;    // limits in r,theta for 2D samples
static Real phi_min, phi_max;                      // limits in phi for 3D samples
static Real dfloor,pfloor;                         // density and pressure floors


//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for Fishbone-Moncrief torus in GR
//  Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator 
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//              Fishbone 1977, ApJ 215 323 (F)
//   assumes x3 is axisymmetric direction

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // Read problem-specific parameters from input file
  // global parameters
  rho_min = pin->GetReal("problem", "rho_min");
  rho_pow = pin->GetReal("problem", "rho_pow");
  pgas_min = pin->GetReal("problem", "pgas_min");
  pgas_pow = pin->GetReal("problem", "pgas_pow");
  psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * (M_PI/180.0);
  sin_psi = std::sin(psi);
  cos_psi = std::cos(psi);


  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));
  
  // local parameters
  Real rho_max = pin->GetReal("problem", "rho_max");
  Real k_adi = pin->GetReal("problem", "k_adi");
  Real r_edge = pin->GetReal("problem", "r_edge");
  Real r_peak = pin->GetReal("problem", "r_peak");
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  

  // capture variables for kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  // initialize Hydro primitive variables ------------------------------------------------
  if (pmbp->phydro != nullptr) {
    auto w0_ = pmbp->phydro->w0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &coord = pmbp->coord.coord_data;
    Real gm1 = eos.gamma - 1.0;

    // Get mass and spin of black hole
    mass = coord.bh_mass;
    spin = coord.bh_spin;

    // compute angular momentum give radius of pressure maximum
    l_peak = CalculateLFromRPeak(r_peak);

    // Prepare constants describing primitives
    Real log_h_edge = LogHAux(r_edge, 1.0);
    Real log_h_peak = LogHAux(r_peak, 1.0) - log_h_edge;
    Real pgas_over_rho_peak = gm1/eos.gamma * (exp(log_h_peak)-1.0);
    Real rho_peak = pow(pgas_over_rho_peak/k_adi, 1.0/gm1) / rho_max;

    Kokkos::Random_XorShift64_Pool<> rand_pool64(5374857);
    par_for("pgen_torus1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real &x1min = coord.mb_size.d_view(m).x1min;
        Real &x1max = coord.mb_size.d_view(m).x1max;
        int nx1 = coord.mb_indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = coord.mb_size.d_view(m).x2min;
        Real &x2max = coord.mb_size.d_view(m).x2max;
        int nx2 = coord.mb_indcs.nx2;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = coord.mb_size.d_view(m).x3min;
        Real &x3max = coord.mb_size.d_view(m).x3max;
        int nx3 = coord.mb_indcs.nx3;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

        // Calculate Boyer-Lindquist coordinates of cell
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(x1v, x2v, x3v, &r, &theta, &phi);
        Real sin_theta = sin(theta);
        Real cos_theta = cos(theta);
        Real sin_phi = sin(phi);
        Real cos_phi = cos(phi);

        Real sin_vartheta = abs(sin_theta);
        Real cos_vartheta = cos_theta;
        Real varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
        Real sin_varphi = sin(varphi);
        Real cos_varphi = cos(varphi);

        // Determine if we are in the torus
        Real log_h;
        bool in_torus = false;
        if (r >= r_edge) {
          log_h = LogHAux(r, sin_vartheta) - log_h_edge;  // (FM 3.6)
          if (log_h >= 0.0) {
            in_torus = true;
          }
        }

        // Calculate background primitives
        Real rho = rho_min * pow(r, rho_pow);
        Real pgas = pgas_min * pow(r, pgas_pow);
        Real uu1 = 0.0;
        Real uu2 = 0.0;
        Real uu3 = 0.0;

        Real perturbation = 0.0;
        // Overwrite primitives inside torus
        if (in_torus) {

          auto rand_gen = rand_pool64.get_state(); // get random number state this thread
          perturbation = pert_amp*(rand_gen.frand() - 0.5);

          // Calculate thermodynamic variables
          Real pgas_over_rho = gm1/eos.gamma * (exp(log_h) - 1.0);
          rho = std::pow(pgas_over_rho/k_adi, 1.0/gm1) / rho_peak;
          pgas = pgas_over_rho * rho;

          // Calculate velocities in Boyer-Lindquist coordinates
          Real u0_bl, u1_bl, u2_bl, u3_bl;
          CalculateVelocityInTiltedTorus(r, theta, phi, &u0_bl, &u1_bl, &u2_bl, &u3_bl);

          // Transform to preferred coordinates
          Real u0, u1, u2, u3;
          TransformVector(u0_bl, 0.0, u2_bl, u3_bl, x1v, x2v, x3v, &u0, &u1, &u2, &u3);

          Real g_[NMETRIC], gi_[NMETRIC];
          ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, true,
                                  coord.bh_spin, g_, gi_);
          uu1 = u1 - gi_[I01]/gi_[I00] * u0;
          uu2 = u2 - gi_[I02]/gi_[I00] * u0;
          uu3 = u3 - gi_[I03]/gi_[I00] * u0;
        }

        // Set primitive values, including random perturbations to pressure
        w0_(m,IDN,k,j,i) = rho;
        w0_(m,IPR,k,j,i) = pgas * (1.0 + perturbation);
        w0_(m,IVX,k,j,i) = uu1;
        w0_(m,IVY,k,j,i) = uu2;
        w0_(m,IVZ,k,j,i) = uu3;
      }
    );

    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0_, u0);

  } // end hydro initialization

  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating angular momentum variable l
// Inputs:
//   r: desired radius of pressure maximum
// Outputs:
//   returned value: l = u^t u_\phi such that pressure maximum occurs at r_peak
// Notes:
//   beware many different definitions of l abound; this is *not* -u_phi/u_t
//   Harm has a similar function: lfish_calc() in init.c
//     Harm's function assumes M = 1 and that corotation is desired
//     it is equivalent to this, though seeing this requires much manipulation
//   implements (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation

KOKKOS_INLINE_FUNCTION
static Real CalculateLFromRPeak(Real r)
{
  Real num = SQR(r*r) + SQR(spin*r) - 2.0*mass*SQR(spin)*r
           - spin*(r*r - spin*spin)*sqrt(mass*r);
  Real denom = SQR(r) - 3.0*mass*r + 2.0*spin*sqrt(mass*r);
  return 1.0/r * sqrt(mass/r) * num/denom;
}

//----------------------------------------------------------------------------------------
// Function to calculate enthalpy
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)

KOKKOS_INLINE_FUNCTION
static Real LogHAux(Real r, Real sin_theta)
{
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*mass*r + SQR(spin);                    // \Delta
  Real sigma = SQR(r) + SQR(spin)*cos_sq_theta;                    // \Sigma
  Real aa = SQR(SQR(r)+SQR(spin)) - delta*SQR(spin)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // \exp(-2\chi) (cf. FM 2.15)
  Real omega = 2.0*mass*spin*r/aa;                              // \omega (FM 3.5)
  Real var_a = sqrt(1.0 + 4.0*SQR(l_peak)*exp_neg2chi);
  Real var_b = 0.5 * log((1.0+var_a) / (sigma*delta/aa));
  Real var_c = -0.5 * var_a;
  Real var_d = -l_peak * omega;
  return var_b + var_c + var_d;                              // (FM 3.4)
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) 
{
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                   + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2( (r*x2-spin*x1)/(SQR(r)+SQR(spin)), (spin*x2+r*x1)/(SQR(r)+SQR(spin)) );
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside tilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   theta,phi: Boyer-Lindquist theta and phi in BH-aligned coordinates
// Outputs:
//   pu0,pu1,pu2,pu3: u^\mu set (Boyer-Lindquist coordinates)
// Notes:
//   first finds corresponding location in untilted torus
//   next calculates velocity at that point in untilted case
//   finally transforms that velocity into coordinates in which torus is tilted

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3)
{
  // Calculate corresponding location
  Real sin_theta = std::sin(theta);
  Real cos_theta = std::cos(theta);
  Real sin_phi = std::sin(phi);
  Real cos_phi = std::cos(phi);
  Real sin_vartheta, cos_vartheta, varphi;
  if (psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = cos_psi * x - sin_psi * z;
    Real vary = y;
    Real varz = sin_psi * x + cos_psi * z;
    sin_vartheta = std::sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = std::atan2(vary, varx);
  } else {
    sin_vartheta = std::abs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
  }
  Real sin_varphi = std::sin(varphi);
  Real cos_varphi = std::cos(varphi);

  // Calculate untilted velocity
  Real u0_tilt, u3_tilt;
  CalculateVelocityInTorus(r, sin_vartheta, &u0_tilt, &u3_tilt);
  Real u1_tilt = 0.0;
  Real u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (psi != 0.0) {
    Real dtheta_dvartheta =
        (cos_psi * sin_vartheta + sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    Real dtheta_dvarphi = -sin_psi * sin_vartheta * sin_varphi / sin_theta;
    Real dphi_dvartheta = sin_psi * sin_varphi / SQR(sin_theta);
    Real dphi_dvarphi = sin_vartheta / SQR(sin_theta)
        * (cos_psi * sin_vartheta + sin_psi * cos_vartheta * cos_varphi);
    *pu2 = dtheta_dvartheta * u2_tilt + dtheta_dvarphi * u3_tilt;
    *pu3 = dphi_dvartheta * u2_tilt + dphi_dvarphi * u3_tilt;
  } else {
    *pu2 = u2_tilt;
    *pu3 = u3_tilt;
  }
  if (sin_theta < 0.0) {
    *pu2 *= -1.0;
    *pu3 *= -1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside untilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   sin_theta: sine of Boyer-Lindquist theta
// Outputs:
//   pu0: u^t set (Boyer-Lindquist coordinates)
//   pu3: u^\phi set (Boyer-Lindquist coordinates)
// Notes:
//   The formula for u^3 as a function of u_{(\phi)} is tedious to derive, but this
//       matches the formula used in Harm (init.c).

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3)
{
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*mass*r + SQR(spin);                    // \Delta
  Real sigma = SQR(r) + SQR(spin)*cos_sq_theta;                    // \Sigma
  Real aa = SQR(SQR(r)+SQR(spin)) - delta*SQR(spin)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // \exp(-2\chi) (cf. FM 2.15)
  Real u_phi_proj_a = 1.0 + 4.0*SQR(l_peak)*exp_neg2chi;
  Real u_phi_proj_b = -1.0 + std::sqrt(u_phi_proj_a);
  Real u_phi_proj = std::sqrt(0.5 * u_phi_proj_b);           // (FM 3.3)
  Real u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
  Real u3_b = 2.0*mass*spin*r * sqrt(u3_a);
  Real u3_c = sqrt(sigma/aa) / sin_theta;
  Real u3 = u3_b + u3_c * u_phi_proj;
  Real g_00 = -(1.0 - 2.0*mass*r/sigma);
  Real g_03 = -2.0*mass*spin*r/sigma * sin_sq_theta;
  Real g_33 = (sigma + (1.0 + 2.0*mass*r/sigma)*SQR(spin) * sin_sq_theta) * sin_sq_theta;
  Real u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
  Real u0_b = sqrt(u0_a - g_00);
  Real u0 = -1.0/g_00 * (g_03*u3 + u0_b);
  *pu0 = u0;
  *pu3 = u3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) 
{
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real rad = sqrt( SQR(x) + SQR(y) + SQR(z) );
  Real r = sqrt( SQR(rad) - SQR(spin) + sqrt( SQR(SQR(rad) - SQR(spin))
               + 4.0*SQR(spin)*SQR(z) ) )/ sqrt(2.0);
  Real delta = SQR(r) - 2.0*mass*r + SQR(spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x+spin*y)/(SQR(r) + SQR(spin)) - y*spin/delta) + 
         a2_bl * x*z/r * sqrt((SQR(r) + SQR(spin))/(SQR(x) + SQR(y))) - 
         a3_bl * y; 
  *pa2 = a1_bl * ( (r*y-spin*x)/(SQR(r) + SQR(spin)) + x*spin/delta) + 
         a2_bl * y*z/r * sqrt((SQR(r) + SQR(spin))/(SQR(x) + SQR(y))) + 
         a3_bl * x;
  *pa3 = a1_bl * z/r - 
         a2_bl * r * sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(spin)));
  return;
}
