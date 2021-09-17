//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()

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
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, 
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, 
                            Real x1, Real x2, Real x3, 
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
void CalculatePrimitives(Real r, Real temp_min, Real temp_max, Real *prho,
                         Real *ppgas, Real *put, Real *pur);

KOKKOS_INLINE_FUNCTION
Real TemperatureMin(Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
Real TemperatureBisect(Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
Real TemperatureResidual(Real t, Real r);

// Global variables
Real mass, spin;        // black hole mass and spin
Real n_adi, k_adi, gm;  // hydro EOS parameters
Real r_crit;            // sonic point radius
Real c1, c2;            // useful constants

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=gr_bondi' to enroll as user-specific problem generator 
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // Read problem-specific parameters from input file
  // global parameters
  k_adi = pin->GetReal("problem", "k_adi");
  r_crit = pin->GetReal("problem", "r_crit");
  gm = pin->GetReal("eos", "gamma");

  // Parameters
  const Real temp_min = 1.0e-2;  // lesser temperature root must be greater than this
  const Real temp_max = 1.0e1;   // greater temperature root must be less than this

  // Get mass and spin of black hole
  mass = pmbp->coord.coord_data.bh_mass;
  spin = pmbp->coord.coord_data.bh_spin;

  // Get ratio of specific heats
  n_adi = 1.0/(gm - 1.0);

  // Prepare various constants for determining primitives
  Real u_crit_sq = mass/(2.0*r_crit);                                       // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = n_adi/(n_adi+1.0) * u_crit_sq/(1.0-(n_adi+3.0)*u_crit_sq);  // (HSW 74)
  c1 = pow(t_crit, n_adi) * u_crit * SQR(r_crit);                           // (HSW 68)
  c2 = SQR(1.0 + (n_adi+1.0) * t_crit) * (1.0 - 3.0*mass/(2.0*r_crit));     // (HSW 69)

  // capture variables for the kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto coord = pmbp->coord.coord_data;
  auto w0_ = pmbp->phydro->w0;

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      // Compute primitive in BL coordinates, transform to Cartesian KS
      Real rho, pgas, ut, ur;
      CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
      Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
      TransformVector(ut, ur, 0.0, 0.0, x1v, x2v, x3v, &u0, &u1, &u2, &u3);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, true,
                              coord.bh_spin, g_, gi_);
      Real uu1 = u1 - gi_[I01]/gi_[I00] * u0;
      Real uu2 = u2 - gi_[I02]/gi_[I00] * u0;
      Real uu3 = u3 - gi_[I03]/gi_[I00] * u0;
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IPR,k,j,i) = pgas;
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }
  );

  // Convert primitives to conserved
  auto &u0_ = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0_, u0_);

  return;
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
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = sqrt( SQR(rad) - SQR(spin) + sqrt( SQR(SQR(rad) - SQR(spin))
               + 4.0*SQR(spin)*SQR(x3) ) )/ sqrt(2.0);
  Real delta = SQR(r) - 2.0*mass*r + SQR(spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1+spin*x2)/(SQR(r) + SQR(spin)) - x2*spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-spin*x1)/(SQR(r) + SQR(spin)) + x1*spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

KOKKOS_INLINE_FUNCTION
void CalculatePrimitives(Real r, Real temp_min, Real temp_max,
                         Real *prho, Real *ppgas, Real *put, Real *pur)
{
  // Calculate solution to (HSW 76)
  Real temp_neg_res = TemperatureMin(r, temp_min, temp_max);
  Real temp;
  if (r <= r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(r, temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(r, temp_neg_res, temp_max);
  }

  // Calculate primitives
  Real rho = pow(temp/k_adi, n_adi);             // not same K as HSW
  Real pgas = temp * rho;
  Real ur = c1 / (SQR(r) * pow(temp, n_adi));    // (HSW 75)
  Real ut = sqrt(1.0/SQR(1.0 - 2.0*mass/r) * SQR(ur) + 1.0/(1.0 - 2.0*mass/r));

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *put = ut;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

KOKKOS_INLINE_FUNCTION
Real TemperatureMin(Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 30;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

KOKKOS_INLINE_FUNCTION
Real TemperatureBisect(Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 20;
  const Real tol_residual = 1.0e-6;
  const Real tol_temperature = 1.0e-6;

  // Find initial residuals
  Real res_min = TemperatureResidual(t_min, r);
  Real res_max = TemperatureResidual(t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 && res_max < 0.0) || (res_min > 0.0 && res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 && res_min < 0.0) || (res_mid > 0.0 && res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

KOKKOS_INLINE_FUNCTION
Real TemperatureResidual(Real t, Real r) {
  return SQR(1.0 + (n_adi+1.0) * t)
      * (1.0 - 2.0*mass/r + SQR(c1) / (SQR(SQR(r)) * pow(t, 2.0*n_adi))) - c2;
}
