//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_torus.cpp
//  \brief Problem generator for Fishbone-Moncrief torus, specialized for cartesian
//  Kerr-Schild coordinates.  Based on gr_torus.cpp in Athena++, with edits by CJW and SR.
//  Simplified and implemented in Kokkos by JMS.

#include <stdio.h>

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <cmath>      // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <cfloat>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

#include <Kokkos_Random.hpp>

// prototypes for functions used internally to this pgen
namespace {
KOKKOS_INLINE_FUNCTION
static Real CalculateLFromRPeak(struct torus_pgen pgen, Real r);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(struct torus_pgen pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct torus_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct torus_pgen pgen,
                                           Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct torus_pgen pgen,
                                     Real r, Real sin_theta, Real *pu0, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct torus_pgen pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
Real A1(struct torus_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct torus_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct torus_pgen pgen, Real x1, Real x2, Real x3);

// useful container for physical parameters of torus
struct torus_pgen {
  Real mass, spin;                            // black hole parameters
  Real dexcise, pexcise;                      // excision parameters
  Real gamma_adi, k_adi;                      // EOS parameters
  Real r_edge, r_peak, l, rho_max;            // fixed torus parameters
  Real l_peak;                                // fixed torus parameters
  Real log_h_edge, log_h_peak;                // calculated torus parameters
  Real pgas_over_rho_peak, rho_peak;          // more calculated torus parameters
  Real psi, sin_psi, cos_psi;                 // tilt parameters
  Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
  Real potential_cutoff;                      // sets region of torus to magnetize
  Real potential_r_pow, potential_rho_pow;    // set how vector potential scales
  Real b_norm;                                // min ratio of gas to mag pressure
  Real dfloor,pfloor;                         // density and pressure floors
};

  torus_pgen torus;

} // namespace

// prototypes for user-defined BCs
void NoInflowTorus(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for Fishbone-Moncrief torus in GR
//  Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//              Fishbone 1977, ApJ 215 323 (F)
//   assumes x3 is axisymmetric direction

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR torus problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_bcs_func = NoInflowTorus;

  // Read problem-specific parameters from input file
  // global parameters
  torus.rho_min = pin->GetReal("problem", "rho_min");
  torus.rho_pow = pin->GetReal("problem", "rho_pow");
  torus.pgas_min = pin->GetReal("problem", "pgas_min");
  torus.pgas_pow = pin->GetReal("problem", "pgas_pow");
  torus.psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * (M_PI/180.0);
  torus.sin_psi = sin(torus.psi);
  torus.cos_psi = cos(torus.psi);

  torus.rho_max = pin->GetReal("problem", "rho_max");
  torus.k_adi = pin->GetReal("problem", "k_adi");
  torus.r_edge = pin->GetReal("problem", "r_edge");
  torus.r_peak = pin->GetReal("problem", "r_peak");

  torus.b_norm = pin->GetReal("problem", "b_norm");

  // local parameters
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &coord = pmbp->pcoord->coord_data;

  // Get ideal gas EOS data
  Real gm1;
  if (pmbp->phydro != nullptr) {
    torus.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
    torus.dfloor = pmbp->phydro->peos->eos_data.dfloor;
    torus.pfloor = pmbp->phydro->peos->eos_data.pfloor;
  } else if (pmbp->pmhd != nullptr) {
    torus.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
    torus.dfloor = pmbp->pmhd->peos->eos_data.dfloor;
    torus.pfloor = pmbp->pmhd->peos->eos_data.pfloor;
  }
  gm1 = torus.gamma_adi - 1.0;

  // Get mass and spin of black hole
  torus.mass = coord.bh_mass;
  torus.spin = coord.bh_spin;

  // Get excision parameters
  torus.dexcise = coord.dexcise;
  torus.pexcise = coord.pexcise;

  // compute angular momentum give radius of pressure maximum
  torus.l_peak = CalculateLFromRPeak(torus, torus.r_peak);

  // Prepare constants describing primitives
  torus.log_h_edge = LogHAux(torus, torus.r_edge, 1.0);
  torus.log_h_peak = LogHAux(torus, torus.r_peak, 1.0) - torus.log_h_edge;
  torus.pgas_over_rho_peak = gm1/torus.gamma_adi * (exp(torus.log_h_peak)-1.0);
  torus.rho_peak = pow(torus.pgas_over_rho_peak/torus.k_adi, 1.0/gm1) / torus.rho_max;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // initialize primitive variables for restart ---------------------------------------

  if (restart) {
    auto trs = torus;
    auto &size = pmbp->pmb->mb_size;
    Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
    par_for("pgen_torus0", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
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

      // Calculate Boyer-Lindquist coordinates of cell
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);

      // Calculate background primitives
      Real rho_bg, pgas_bg;
      if (r > 1.0) {
        rho_bg = trs.rho_min * pow(r, trs.rho_pow);
        pgas_bg = trs.pgas_min * pow(r, trs.pgas_pow);
      } else {
        rho_bg = trs.dexcise;
        pgas_bg = trs.pexcise;
      }

      // Set primitive values on restart, to get solution in excise region
      w0_(m,IDN,k,j,i) = rho_bg;
      w0_(m,IEN,k,j,i) = pgas_bg / gm1;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVY,k,j,i) = 0.0;
      w0_(m,IVZ,k,j,i) = 0.0;
    });
    return;
  }


  // initialize primitive variables for new run ---------------------------------------

  auto trs = torus;
  auto &size = pmbp->pmb->mb_size;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_torus1", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
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

    // Calculate Boyer-Lindquist coordinates of cell
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);
    Real sin_theta = sin(theta);
    Real cos_theta = cos(theta);
    Real sin_phi = sin(phi);
    Real cos_phi = cos(phi);

    Real sin_vartheta = abs(sin_theta);
    Real varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
    Real sin_varphi = sin(varphi);
    Real cos_varphi = cos(varphi);

    // Determine if we are in the torus
    Real log_h;
    bool in_torus = false;
    if (r >= trs.r_edge) {
      log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
      if (log_h >= 0.0) {
        in_torus = true;
      }
    }

    // Calculate background primitives
    Real rho_bg, pgas_bg;
    if (r > 1.0) {
      rho_bg = trs.rho_min * pow(r, trs.rho_pow);
      pgas_bg = trs.pgas_min * pow(r, trs.pgas_pow);
    } else {
      rho_bg = trs.dexcise;
      pgas_bg = trs.pexcise;
    }

    Real rho = rho_bg;
    Real pgas = pgas_bg;
    Real uu1 = 0.0;
    Real uu2 = 0.0;
    Real uu3 = 0.0;
    Real perturbation = 0.0;
    // Overwrite primitives inside torus
    if (in_torus) {
      // Calculate perturbation
      auto rand_gen = rand_pool64.get_state(); // get random number state this thread
      perturbation = pert_amp*(rand_gen.frand() - 0.5);
      rand_pool64.free_state(rand_gen);  // free state for use by other threads

      // Calculate thermodynamic variables
      Real pgas_over_rho = gm1/trs.gamma_adi * (exp(log_h) - 1.0);
      rho = pow(pgas_over_rho/trs.k_adi, 1.0/gm1) / trs.rho_peak;
      pgas = pgas_over_rho * rho;

      // Calculate velocities in Boyer-Lindquist coordinates
      Real u0_bl, u1_bl, u2_bl, u3_bl;
      CalculateVelocityInTiltedTorus(trs, r, theta, phi,
                                     &u0_bl, &u1_bl, &u2_bl, &u3_bl);

      // Transform to preferred coordinates
      Real u0, u1, u2, u3;
      TransformVector(trs, u0_bl, 0.0, u2_bl, u3_bl,
                      x1v, x2v, x3v, &u0, &u1, &u2, &u3);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin, g_, gi_);
      uu1 = u1 - gi_[I01]/gi_[I00] * u0;
      uu2 = u2 - gi_[I02]/gi_[I00] * u0;
      uu3 = u3 - gi_[I03]/gi_[I00] * u0;
    }

    // Set primitive values, including random perturbations to pressure
    w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
    w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg) * (1.0 + perturbation) / gm1;
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;
  });

  // initialize magnetic fields ---------------------------------------

  if (pmbp->pmhd != nullptr) {
    // parse some more parameters from input
    torus.potential_cutoff = pin->GetReal("problem", "potential_cutoff");
    torus.potential_r_pow = pin->GetReal("problem", "potential_r_pow");
    torus.potential_rho_pow = pin->GetReal("problem", "potential_rho_pow");

    auto &b0 = pmbp->pmhd->b0;
    auto trs = torus;
    par_for("pgen_torus2", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
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

      // Compute face-centered fields from curl(A).
      Real x1f   = LeftEdgeX(i  -is, indcs.nx1, x1min, x1max);
      Real x1fp1 = LeftEdgeX(i+1-is, indcs.nx1, x1min, x1max);
      Real x2f   = LeftEdgeX(j  -js, indcs.nx2, x2min, x2max);
      Real x2fp1 = LeftEdgeX(j+1-js, indcs.nx2, x2min, x2max);
      Real x3f   = LeftEdgeX(k  -ks, indcs.nx3, x3min, x3max);
      Real x3fp1 = LeftEdgeX(k+1-ks, indcs.nx3, x3min, x3max);
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = trs.b_norm *
                        ((A3(trs,x1f,  x2fp1,x3v  ) - A3(trs,x1f,x2f,x3v))/dx2 -
                         (A2(trs,x1f,  x2v,  x3fp1) - A2(trs,x1f,x2v,x3f))/dx3);
      b0.x2f(m,k,j,i) = trs.b_norm *
                        ((A1(trs,x1v,  x2f,  x3fp1) - A1(trs,x1v,x2f,x3f))/dx3 -
                         (A3(trs,x1fp1,x2f,  x3v  ) - A3(trs,x1f,x2f,x3v))/dx1);
      b0.x3f(m,k,j,i) = trs.b_norm *
                        ((A2(trs,x1fp1,x2v,  x3f  ) - A2(trs,x1f,x2v,x3f))/dx1 -
                         (A1(trs,x1v,  x2fp1,x3f  ) - A1(trs,x1v,x2f,x3f))/dx2);


      // Include extra face-component at edge of block in each direction
      if (i==(n1-1)) {
        b0.x1f(m,k,j,i+1) = trs.b_norm *
                            ((A3(trs,x1fp1,x2fp1,x3v  ) - A3(trs,x1fp1,x2f,x3v))/dx2 -
                             (A2(trs,x1fp1,x2v,  x3fp1) - A2(trs,x1fp1,x2v,x3f))/dx3);
      }
      if (j==(n2-1)) {
        b0.x2f(m,k,j+1,i) = trs.b_norm *
                            ((A1(trs,x1v,  x2fp1,x3fp1) - A1(trs,x1v,x2fp1,x3f))/dx3 -
                             (A3(trs,x1fp1,x2fp1,x3v  ) - A3(trs,x1f,x2fp1,x3v))/dx1);
      }
      if (k==(n3-1)) {
        b0.x3f(m,k+1,j,i) = trs.b_norm *
                            ((A2(trs,x1fp1,x2v,  x3fp1) - A2(trs,x1f,x2v,x3fp1))/dx1 -
                             (A1(trs,x1v,  x2fp1,x3fp1) - A1(trs,x1v,x2f,x3fp1))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_torus2", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  return;
}

namespace {

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
static Real CalculateLFromRPeak(struct torus_pgen pgen, Real r) {
  Real num = SQR(r*r) + SQR(pgen.spin*r) - 2.0*pgen.mass*SQR(pgen.spin)*r
           - pgen.spin*(r*r - pgen.spin*pgen.spin)*sqrt(pgen.mass*r);
  Real denom = SQR(r) - 3.0*pgen.mass*r + 2.0*pgen.spin*sqrt(pgen.mass*r);
  return 1.0/r * sqrt(pgen.mass/r) * num/denom;
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
static Real LogHAux(struct torus_pgen pgen, Real r, Real sin_theta) {
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*pgen.mass*r + SQR(pgen.spin);  // \Delta
  Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;       // \Sigma
  Real aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                       // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;               // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                   // \exp(-2\chi) (cf. FM 2.15)
  Real omega = 2.0*pgen.mass*pgen.spin*r/aa;               // \omega (FM 3.5)
  Real var_a = sqrt(1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi);
  Real var_b = 0.5 * log((1.0+var_a) / (sigma*delta/aa));
  Real var_c = -0.5 * var_a;
  Real var_d = -pgen.l_peak * omega;
  return var_b + var_c + var_d;                              // (FM 3.4)
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct torus_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                        + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2( (r*x2-pgen.spin*x1)/(SQR(r)+SQR(pgen.spin)),
                   (pgen.spin*x2+r*x1)/(SQR(r)+SQR(pgen.spin)) );
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
static void CalculateVelocityInTiltedTorus(struct torus_pgen pgen,
                                           Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3) {
  // Calculate corresponding location
  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real sin_phi = sin(phi);
  Real cos_phi = cos(phi);
  Real sin_vartheta, cos_vartheta, varphi;
  if (pgen.psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = pgen.cos_psi * x - pgen.sin_psi * z;
    Real vary = y;
    Real varz = pgen.sin_psi * x + pgen.cos_psi * z;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = atan2(vary, varx);
  } else {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
  }
  Real sin_varphi = sin(varphi);
  Real cos_varphi = cos(varphi);

  // Calculate untilted velocity
  Real u0_tilt, u3_tilt;
  CalculateVelocityInTorus(pgen, r, sin_vartheta, &u0_tilt, &u3_tilt);
  Real u1_tilt = 0.0;
  Real u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (pgen.psi != 0.0) {
    Real dtheta_dvartheta =
        (pgen.cos_psi * sin_vartheta
         + pgen.sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    Real dtheta_dvarphi = -pgen.sin_psi * sin_vartheta * sin_varphi / sin_theta;
    Real dphi_dvartheta = pgen.sin_psi * sin_varphi / SQR(sin_theta);
    Real dphi_dvarphi = sin_vartheta / SQR(sin_theta)
        * (pgen.cos_psi * sin_vartheta + pgen.sin_psi * cos_vartheta * cos_varphi);
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
static void CalculateVelocityInTorus(struct torus_pgen pgen,
                                     Real r, Real sin_theta, Real *pu0, Real *pu3) {
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*pgen.mass*r + SQR(pgen.spin);                    // \Delta
  Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;                    // \Sigma
  Real aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // \exp(-2\chi) (cf. FM 2.15)
  Real u_phi_proj_a = 1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi;
  Real u_phi_proj_b = -1.0 + sqrt(u_phi_proj_a);
  Real u_phi_proj = sqrt(0.5 * u_phi_proj_b);           // (FM 3.3)
  Real u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
  Real u3_b = 2.0*pgen.mass*pgen.spin*r * sqrt(u3_a);
  Real u3_c = sqrt(sigma/aa) / sin_theta;
  Real u3 = u3_b + u3_c * u_phi_proj;
  Real g_00 = -(1.0 - 2.0*pgen.mass*r/sigma);
  Real g_03 = -2.0*pgen.mass*pgen.spin*r/sigma * sin_sq_theta;
  Real g_33 = (sigma + (1.0 + 2.0*pgen.mass*r/sigma) *
               SQR(pgen.spin) * sin_sq_theta) * sin_sq_theta;
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
static void TransformVector(struct torus_pgen pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real rad = sqrt( SQR(x) + SQR(y) + SQR(z) );
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*pgen.mass*r + SQR(pgen.spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x+pgen.spin*y)/(SQR(r) + SQR(pgen.spin)) - y*pgen.spin/delta) +
         a2_bl * x*z/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x) + SQR(y))) -
         a3_bl * y;
  *pa2 = a1_bl * ( (r*y-pgen.spin*x)/(SQR(r) + SQR(pgen.spin)) + x*pgen.spin/delta) +
         a2_bl * y*z/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x) + SQR(y))) +
         a3_bl * x;
  *pa3 = a1_bl * z/r -
         a2_bl * r * sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(pgen.spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function to compute 1-component of vector potential.  First computes phi-componenent
// in BL coordinates, then transforms to Cartesian KS, assuming A_r = A_theta = 0
// A_\mu (cks) = A_nu (ks)  dx^nu (ks)/dx^\mu (cks) = A_phi (ks) dphi (ks)/dx^\mu
// phi_ks = arctan((r*y + a*x)/(r*x - a*y) )
//

KOKKOS_INLINE_FUNCTION
Real A1(struct torus_pgen trs, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  Real aphi = 0.0;
  GetBoyerLindquistCoordinates(trs, x1, x2, x3, &r, &theta, &phi);
  if (r >= trs.r_edge) {
    Real sin_vartheta = abs(sin(theta));
    Real log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
    if (log_h >= 0.0) {
      Real pgas_over_rho = (trs.gamma_adi-1.0)/trs.gamma_adi * (exp(log_h)-1.0);
      Real rho = pow(pgas_over_rho/trs.k_adi, 1.0/(trs.gamma_adi-1.0)) / trs.rho_peak;
      Real rho_cutoff = fmax(rho - trs.potential_cutoff, static_cast<Real>(0.0));
      aphi = pow(r, trs.potential_r_pow) * pow(rho_cutoff, trs.potential_rho_pow);
    }
  }

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(trs.spin);

  //dphi/dx =  partial phi/partial x + partial phi/partial r partial r/partial x
  return aphi*(-x2/(SQR(x1)+SQR(x2)) + trs.spin*x1*r/((SQR(trs.spin)+SQR(r))*sqrt_term));
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(struct torus_pgen trs, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  Real aphi = 0.0;
  GetBoyerLindquistCoordinates(trs, x1, x2, x3, &r, &theta, &phi);
  if (r >= trs.r_edge) {
    Real sin_vartheta = abs(sin(theta));
    Real log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
    if (log_h >= 0.0) {
      Real pgas_over_rho = (trs.gamma_adi-1.0)/trs.gamma_adi * (exp(log_h)-1.0);
      Real rho = pow(pgas_over_rho/trs.k_adi, 1.0/(trs.gamma_adi-1.0)) / trs.rho_peak;
      Real rho_cutoff = fmax(rho - trs.potential_cutoff, static_cast<Real>(0.0));
      aphi = pow(r, trs.potential_r_pow) * pow(rho_cutoff, trs.potential_rho_pow);
    }
  }

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(trs.spin);

  //dphi/dx =  partial phi/partial y + partial phi/partial r partial r/partial y
  return aphi*( x1/(SQR(x1)+SQR(x2)) + trs.spin*x2*r/((SQR(trs.spin)+SQR(r))*sqrt_term) );
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(struct torus_pgen trs, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  Real aphi = 0.0;
  GetBoyerLindquistCoordinates(trs, x1, x2, x3, &r, &theta, &phi);
  if (r >= trs.r_edge) {
    Real sin_vartheta = abs(sin(theta));
    Real log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
    if (log_h >= 0.0) {
      Real pgas_over_rho = (trs.gamma_adi-1.0)/trs.gamma_adi * (exp(log_h)-1.0);
      Real rho = pow(pgas_over_rho/trs.k_adi, 1.0/(trs.gamma_adi-1.0)) / trs.rho_peak;
      Real rho_cutoff = fmax(rho - trs.potential_cutoff, static_cast<Real>(0.0));
      aphi = pow(r, trs.potential_r_pow) * pow(rho_cutoff, trs.potential_rho_pow);
    }
  }

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(trs.spin);

  //dphi/dx =   partial phi/partial r partial r/partial z
  return aphi * ( trs.spin*x3/(r*sqrt_term) );
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn NoInflowTorus
//  \brief Sets boundary condition on surfaces of computational domain

void NoInflowTorus(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // X1-Boundary
  // ConsToPrim over all x1 ghost zones *and* at the innermost/outermost x1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,ie,ie+ng,0,(n2-1),0,(n3-1));
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,is-i-1) = fmin(0.0,w0_(m,n,k,j,is));
        } else {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,ie+i+1) = fmax(0.0,w0_(m,n,k,j,ie));
        } else {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
        }
      }
    }
  });
  // Set X1-BCs on b0 and bcc0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc_ = pm->pmb_pack->pmhd->bcc0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          bcc_(m,IBX,k,j,is-i-1) = 0.5*(b0.x1f(m,k,j,is-i-1) + b0.x1f(m,k,  j,  is-i  ));
          bcc_(m,IBY,k,j,is-i-1) = 0.5*(b0.x2f(m,k,j,is-i-1) + b0.x2f(m,k,  j+1,is-i-1));
          bcc_(m,IBZ,k,j,is-i-1) = 0.5*(b0.x3f(m,k,j,is-i-1) + b0.x3f(m,k+1,j  ,is-i-1));
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          bcc_(m,IBX,k,j,ie+i+1) = 0.5*(b0.x1f(m,k,j,ie+i+1) + b0.x1f(m,k  ,j  ,ie+i+2));
          bcc_(m,IBY,k,j,ie+i+1) = 0.5*(b0.x2f(m,k,j,ie+i+1) + b0.x2f(m,k  ,j+1,ie+i+1));
          bcc_(m,IBZ,k,j,ie+i+1) = 0.5*(b0.x3f(m,k,j,ie+i+1) + b0.x3f(m,k+1,j  ,ie+i+1));
        }
      }
    });
  }
  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  }

  // X2-Boundary
  // ConsToPrim over all x2 ghost zones *and* at the innermost/outermost x2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,0,(n1-1),je,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,0,(n1-1),je,je+ng,0,(n3-1));
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,js-j-1,i) = fmin(0.0,w0_(m,n,k,js,i));
        } else {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,je+j+1,i) = fmax(0.0,w0_(m,n,k,je,i));
        } else {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    }
  });
  // Set X2-BCs on b0 and bcc0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc_ = pm->pmb_pack->pmhd->bcc0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          bcc_(m,IBX,k,js-j-1,i) = 0.5*(b0.x1f(m,k,js-j-1,i) + b0.x1f(m,k  ,js-j-1,i+1));
          bcc_(m,IBY,k,js-j-1,i) = 0.5*(b0.x2f(m,k,js-j-1,i) + b0.x2f(m,k  ,js-j  ,i  ));
          bcc_(m,IBZ,k,js-j-1,i) = 0.5*(b0.x3f(m,k,js-j-1,i) + b0.x3f(m,k+1,js-j-1,i  ));
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          bcc_(m,IBX,k,je+j+1,i) = 0.5*(b0.x1f(m,k,je+j+1,i) + b0.x1f(m,k  ,je+j+1,i+1));
          bcc_(m,IBY,k,je+j+1,i) = 0.5*(b0.x2f(m,k,je+j+1,i) + b0.x2f(m,k  ,je+j+2,i  ));
          bcc_(m,IBZ,k,je+j+1,i) = 0.5*(b0.x3f(m,k,je+j+1,i) + b0.x3f(m,k+1,je+j+1,i  ));
        }
      }
    });
  }
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  }

  // x3-Boundary
  // ConsToPrim over all x3 ghost zones *and* at the innermost/outermost x3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,0,(n1-1),0,(n2-1),ke,ke+ng);
  }
  // Set x3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = fmin(0.0,w0_(m,n,ks,j,i));
        } else {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ke+k+1,j,i) = fmax(0.0,w0_(m,n,ke,j,i));
        } else {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    }
  });
  // Set x3-BCs on b0 and bcc0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc_ = pm->pmb_pack->pmhd->bcc0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          bcc_(m,IBX,ks-k-1,j,i) = 0.5*(b0.x1f(m,ks-k-1,j,i) + b0.x1f(m,ks-k-1,j  ,i+1));
          bcc_(m,IBY,ks-k-1,j,i) = 0.5*(b0.x2f(m,ks-k-1,j,i) + b0.x2f(m,ks-k-1,j+1,i  ));
          bcc_(m,IBZ,ks-k-1,j,i) = 0.5*(b0.x3f(m,ks-k-1,j,i) + b0.x3f(m,ks-k  ,j  ,i  ));
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          bcc_(m,IBX,ke+k+1,j,i) = 0.5*(b0.x1f(m,ke+k+1,j,i) + b0.x1f(m,ke+k+1,j  ,i+1));
          bcc_(m,IBY,ke+k+1,j,i) = 0.5*(b0.x2f(m,ke+k+1,j,i) + b0.x2f(m,ke+k+1,j+1,i  ));
          bcc_(m,IBZ,ke+k+1,j,i) = 0.5*(b0.x3f(m,ke+k+1,j,i) + b0.x3f(m,ke+k+2,j  ,i  ));
        }
      }
    });
  }
  // PrimToCons on x3 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  }

  return;
}
