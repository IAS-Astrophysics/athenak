#ifndef COORDINATES_CARTESIAN_KS_HPP_
#define COORDINATES_CARTESIAN_KS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cartesian_ks.hpp
//! \brief implements functions for Cartesian Kerr-Schild coordinates in GR.  This
//! includes inline functions to compute the metric and derivatives of the metric.  Based
//! on functions in 'gr_user.cpp' file in Athena++, as well as CartesianGR.cpp function
//! from CJW and SR.

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"

// #define SMALL_NUMBER 1.0e-5

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricAndInverse
//! \brief computes 10 covariant and contravariant components of metric in
//!  Cartesian Kerr-Schild coordinates

KOKKOS_INLINE_FUNCTION
void ComputeMetricAndInverse(Real x, Real y, Real z, bool minkowski, Real a,
                             Real glower[][4], Real gupper[][4]) {
  // NOTE(@pdmullen): The following commented out floor on z dealt with the metric
  // singularity encountered for small z near the horizon (e.g., see g_00). However, this
  // floor was operating on z even for r_ks > 1.0, where (I believe) the metric should be
  // well-behaved for all z.  We floor r_ks to 1.0, therefore, the floor on z is
  // seemingly not necessary, however, if something goes awry for z ~ 0 in future
  // applications (even after flooring r_ks = 1.0), consider revisiting this z floor...
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  r = fmax(r, 1.0);  // floor r_ks to 0.5*(r_inner + r_outer)

  // Set covariant components
  // null vector l
  Real l_lower[4];
  l_lower[0] = 1.0;
  l_lower[1] = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  l_lower[2] = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  l_lower[3] = z/r;

  // g_nm = f*l_n*l_m + eta_nm, where eta_nm is Minkowski metric
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {f=0.0;}
  glower[0][0] = f * l_lower[0]*l_lower[0] - 1.0;
  glower[0][1] = f * l_lower[0]*l_lower[1];
  glower[0][2] = f * l_lower[0]*l_lower[2];
  glower[0][3] = f * l_lower[0]*l_lower[3];
  glower[1][0] = glower[0][1];
  glower[1][1] = f * l_lower[1]*l_lower[1] + 1.0;
  glower[1][2] = f * l_lower[1]*l_lower[2];
  glower[1][3] = f * l_lower[1]*l_lower[3];
  glower[2][0] = glower[0][2];
  glower[2][1] = glower[1][2];
  glower[2][2] = f * l_lower[2]*l_lower[2] + 1.0;
  glower[2][3] = f * l_lower[2]*l_lower[3];
  glower[3][0] = glower[0][3];
  glower[3][1] = glower[1][3];
  glower[3][2] = glower[2][3];
  glower[3][3] = f * l_lower[3]*l_lower[3] + 1.0;

  // Set contravariant components
  // null vector l
  Real l_upper[4];
  l_upper[0] = -1.0;
  l_upper[1] = l_lower[1];
  l_upper[2] = l_lower[2];
  l_upper[3] = l_lower[3];

  // g^nm = -f*l^n*l^m + eta^nm, where eta^nm is Minkowski metric
  gupper[0][0] = -f * l_upper[0]*l_upper[0] - 1.0;
  gupper[0][1] = -f * l_upper[0]*l_upper[1];
  gupper[0][2] = -f * l_upper[0]*l_upper[2];
  gupper[0][3] = -f * l_upper[0]*l_upper[3];
  gupper[1][0] = gupper[0][1];
  gupper[1][1] = -f * l_upper[1]*l_upper[1] + 1.0;
  gupper[1][2] = -f * l_upper[1]*l_upper[2];
  gupper[1][3] = -f * l_upper[1]*l_upper[3];
  gupper[2][0] = gupper[0][2];
  gupper[2][1] = gupper[1][2];
  gupper[2][2] = -f * l_upper[2]*l_upper[2] + 1.0;
  gupper[2][3] = -f * l_upper[2]*l_upper[3];
  gupper[3][0] = gupper[0][3];
  gupper[3][1] = gupper[1][3];
  gupper[3][2] = gupper[2][3];
  gupper[3][3] = -f * l_upper[3]*l_upper[3] + 1.0;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricDerivatives
//! \brief computes derivates of metric in Cartesian Kerr-Schild coordinates, which are
//!  used to compute the coordinate source terms in the equations of motion.

KOKKOS_INLINE_FUNCTION
void ComputeMetricDerivatives(Real x, Real y, Real z, bool minkowski, Real a,
                              Real dg_dx1[][4], Real dg_dx2[][4], Real dg_dx3[][4]) {
  // NOTE(@pdmullen): See comment in ComputeMetricAndInverse
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  r = fmax(r, 1.0);  // floor r_ks to 0.5*(r_inner + r_outer)

  Real llower[4];
  llower[0] = 1.0;
  llower[1] = (r*x + a * y)/( SQR(r) + SQR(a) );
  llower[2] = (r*y - a * x)/( SQR(r) + SQR(a) );
  llower[3] = z/r;

  Real qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  Real qb = SQR(r) + SQR(a);
  Real qc = 3.0*SQR(a * z)-SQR(r)*SQR(r);
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));

  Real df_dx1 = SQR(f)*x/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx2 = SQR(f)*y/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx3 = SQR(f)*z/(2.0*pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r));
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + r/( qb );
  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + a/( qb );
  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (qb) * qa );
  Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) - a/( qb );
  Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) + r/( qb );
  Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (qb) * qa );
  Real dl3_dx1 = - x*z/(r*qa);
  Real dl3_dx2 = - y*z/(r*qa);
  Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( qb )/( qa ) + 1.0/r;

  Real dl0_dx1 = 0.0;
  Real dl0_dx2 = 0.0;
  Real dl0_dx3 = 0.0;

  if (minkowski) {
    f = 0.0;
    df_dx1 = 0.0;
    df_dx2 = 0.0;
    df_dx3 = 0.0;
  }

  // Set x-derivatives of covariant components
  dg_dx1[0][0] = df_dx1*llower[0]*llower[0] + f*dl0_dx1*llower[0] + f*llower[0]*dl0_dx1;
  dg_dx1[0][1] = df_dx1*llower[0]*llower[1] + f*dl0_dx1*llower[1] + f*llower[0]*dl1_dx1;
  dg_dx1[0][2] = df_dx1*llower[0]*llower[2] + f*dl0_dx1*llower[2] + f*llower[0]*dl2_dx1;
  dg_dx1[0][3] = df_dx1*llower[0]*llower[3] + f*dl0_dx1*llower[3] + f*llower[0]*dl3_dx1;
  dg_dx1[1][0] = dg_dx1[0][1];
  dg_dx1[1][1] = df_dx1*llower[1]*llower[1] + f*dl1_dx1*llower[1] + f*llower[1]*dl1_dx1;
  dg_dx1[1][2] = df_dx1*llower[1]*llower[2] + f*dl1_dx1*llower[2] + f*llower[1]*dl2_dx1;
  dg_dx1[1][3] = df_dx1*llower[1]*llower[3] + f*dl1_dx1*llower[3] + f*llower[1]*dl3_dx1;
  dg_dx1[2][0] = dg_dx1[0][2];
  dg_dx1[2][1] = dg_dx1[1][2];
  dg_dx1[2][2] = df_dx1*llower[2]*llower[2] + f*dl2_dx1*llower[2] + f*llower[2]*dl2_dx1;
  dg_dx1[2][3] = df_dx1*llower[2]*llower[3] + f*dl2_dx1*llower[3] + f*llower[2]*dl3_dx1;
  dg_dx1[3][0] = dg_dx1[0][3];
  dg_dx1[3][1] = dg_dx1[1][3];
  dg_dx1[3][2] = dg_dx1[2][3];
  dg_dx1[3][3] = df_dx1*llower[3]*llower[3] + f*dl3_dx1*llower[3] + f*llower[3]*dl3_dx1;

  // Set y-derivatives of covariant components
  dg_dx2[0][0] = df_dx2*llower[0]*llower[0] + f*dl0_dx2*llower[0] + f*llower[0]*dl0_dx2;
  dg_dx2[0][1] = df_dx2*llower[0]*llower[1] + f*dl0_dx2*llower[1] + f*llower[0]*dl1_dx2;
  dg_dx2[0][2] = df_dx2*llower[0]*llower[2] + f*dl0_dx2*llower[2] + f*llower[0]*dl2_dx2;
  dg_dx2[0][3] = df_dx2*llower[0]*llower[3] + f*dl0_dx2*llower[3] + f*llower[0]*dl3_dx2;
  dg_dx2[1][0] = dg_dx2[0][1];
  dg_dx2[1][1] = df_dx2*llower[1]*llower[1] + f*dl1_dx2*llower[1] + f*llower[1]*dl1_dx2;
  dg_dx2[1][2] = df_dx2*llower[1]*llower[2] + f*dl1_dx2*llower[2] + f*llower[1]*dl2_dx2;
  dg_dx2[1][3] = df_dx2*llower[1]*llower[3] + f*dl1_dx2*llower[3] + f*llower[1]*dl3_dx2;
  dg_dx2[2][0] = dg_dx2[0][2];
  dg_dx2[2][1] = dg_dx2[1][2];
  dg_dx2[2][2] = df_dx2*llower[2]*llower[2] + f*dl2_dx2*llower[2] + f*llower[2]*dl2_dx2;
  dg_dx2[2][3] = df_dx2*llower[2]*llower[3] + f*dl2_dx2*llower[3] + f*llower[2]*dl3_dx2;
  dg_dx2[3][0] = dg_dx2[0][3];
  dg_dx2[3][1] = dg_dx2[1][3];
  dg_dx2[3][2] = dg_dx2[2][3];
  dg_dx2[3][3] = df_dx2*llower[3]*llower[3] + f*dl3_dx2*llower[3] + f*llower[3]*dl3_dx2;

  // Set phi-derivatives of covariant components
  dg_dx3[0][0] = df_dx3*llower[0]*llower[0] + f*dl0_dx3*llower[0] + f*llower[0]*dl0_dx3;
  dg_dx3[0][1] = df_dx3*llower[0]*llower[1] + f*dl0_dx3*llower[1] + f*llower[0]*dl1_dx3;
  dg_dx3[0][2] = df_dx3*llower[0]*llower[2] + f*dl0_dx3*llower[2] + f*llower[0]*dl2_dx3;
  dg_dx3[0][3] = df_dx3*llower[0]*llower[3] + f*dl0_dx3*llower[3] + f*llower[0]*dl3_dx3;
  dg_dx3[1][0] = dg_dx3[0][1];
  dg_dx3[1][1] = df_dx3*llower[1]*llower[1] + f*dl1_dx3*llower[1] + f*llower[1]*dl1_dx3;
  dg_dx3[1][2] = df_dx3*llower[1]*llower[2] + f*dl1_dx3*llower[2] + f*llower[1]*dl2_dx3;
  dg_dx3[1][3] = df_dx3*llower[1]*llower[3] + f*dl1_dx3*llower[3] + f*llower[1]*dl3_dx3;
  dg_dx3[2][0] = dg_dx3[0][2];
  dg_dx3[2][1] = dg_dx3[1][2];
  dg_dx3[2][2] = df_dx3*llower[2]*llower[2] + f*dl2_dx3*llower[2] + f*llower[2]*dl2_dx3;
  dg_dx3[2][3] = df_dx3*llower[2]*llower[3] + f*dl2_dx3*llower[3] + f*llower[2]*dl3_dx3;
  dg_dx3[3][0] = dg_dx3[0][3];
  dg_dx3[3][1] = dg_dx3[1][3];
  dg_dx3[3][2] = dg_dx3[2][3];
  dg_dx3[3][3] = df_dx3*llower[3]*llower[3] + f*dl3_dx3*llower[3] + f*llower[3]*dl3_dx3;

  return;
}

#endif // COORDINATES_CARTESIAN_KS_HPP_
