//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cartesian_gr.hpp
//! \brief implements functions for Cartesian Kerr-Schild coordinates in GR.  This
//! includes inline functions to compute metric, derivatives of the metric, and function
//! to compute "cordinate source terms".  Based on functions in 'gr_user.cpp' file in
//! Athena++, as well as CartesianGR.cpp function from CJW and SR.

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
  
#define SMALL_NUMBER 1.0e-5

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricAndInverse
//! \brief computes 10 covariant and contravariant components of metric in
//!  Cartesian Kerr-Schild coordinates

KOKKOS_INLINE_FUNCTION
void ComputeMetricAndInverse(Real x, Real y, Real z, bool minkowski, bool ic,
                             Real a, Real g[], Real ginv[])
{
  if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real rad = fmax(sqrt(SQR(x) + SQR(y) + SQR(z)),1.0);  // avoid singularity for rad<1
  if (ic) {rad=sqrt(SQR(x) + SQR(y) + SQR(z));}  // unless you are trying to set ic
  Real r = SQR(rad)-SQR(a) + sqrt( SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z) );
  r = sqrt(r/2.0);
  
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
  g[I00] = f * l_lower[0]*l_lower[0] - 1.0;
  g[I01] = f * l_lower[0]*l_lower[1];
  g[I02] = f * l_lower[0]*l_lower[2];
  g[I03] = f * l_lower[0]*l_lower[3];
  g[I11] = f * l_lower[1]*l_lower[1] + 1.0;
  g[I12] = f * l_lower[1]*l_lower[2];
  g[I13] = f * l_lower[1]*l_lower[3];
  g[I22] = f * l_lower[2]*l_lower[2] + 1.0;
  g[I23] = f * l_lower[2]*l_lower[3];
  g[I33] = f * l_lower[3]*l_lower[3] + 1.0;

  // Set contravariant components
  // null vector l
  Real l_upper[4];
  l_upper[0] = -1.0;
  l_upper[1] = l_lower[1];
  l_upper[2] = l_lower[2];
  l_upper[3] = l_lower[3];

  // g^nm = -f*l^n*l^m + eta^nm, where eta^nm is Minkowski metric
  ginv[I00] = -f * l_upper[0]*l_upper[0] - 1.0;
  ginv[I01] = -f * l_upper[0]*l_upper[1];
  ginv[I02] = -f * l_upper[0]*l_upper[2];
  ginv[I03] = -f * l_upper[0]*l_upper[3];
  ginv[I11] = -f * l_upper[1]*l_upper[1] + 1.0;
  ginv[I12] = -f * l_upper[1]*l_upper[2];
  ginv[I13] = -f * l_upper[1]*l_upper[3];
  ginv[I22] = -f * l_upper[2]*l_upper[2] + 1.0;
  ginv[I23] = -f * l_upper[2]*l_upper[3];
  ginv[I33] = -f * l_upper[3]*l_upper[3] + 1.0;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricDerivatives
//! \brief computes derivates of metric in Cartesian Kerr-Schild coordinates, which are
//!  used to compute the coordinate source terms in the equations of motion.

KOKKOS_INLINE_FUNCTION
void ComputeMetricDerivatives(Real x, Real y, Real z, bool minkowski,
                              Real a, Real dg_dx1[], Real dg_dx2[], Real dg_dx3[])
{
  if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real rad = fmax(sqrt(SQR(x) + SQR(y) + SQR(z)),1.0);  // avoid singularity for rad<1
  Real r = SQR(rad)-SQR(a) + sqrt( SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z) );
  r = sqrt(r/2.0);

  Real llower[4];
  llower[0] = 1.0;
  llower[1] = (r*x + a * y)/( SQR(r) + SQR(a) );
  llower[2] = (r*y - a * x)/( SQR(r) + SQR(a) );
  llower[3] = z/r;

  Real qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  Real qb = SQR(r) + SQR(a);
  Real qc = 3.0*SQR(a * z)-SQR(r)*SQR(r);
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));

  Real df_dx1 = SQR(f)*x/(2.0*std::pow(r,3)) * ( ( qc ) )/ qa ;
  //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
  Real df_dx2 = SQR(f)*y/(2.0*std::pow(r,3)) * ( ( qc ) )/ qa ;
  Real df_dx3 = SQR(f)*z/(2.0*std::pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r)) ;
  //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + r/( qb );
  // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa )+ a/( qb );
  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (qb) * qa ) ;
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
  dg_dx1[I00] = df_dx1*llower[0]*llower[0] + f*dl0_dx1*llower[0] + f*llower[0]*dl0_dx1;
  dg_dx1[I01] = df_dx1*llower[0]*llower[1] + f*dl0_dx1*llower[1] + f*llower[0]*dl1_dx1;
  dg_dx1[I02] = df_dx1*llower[0]*llower[2] + f*dl0_dx1*llower[2] + f*llower[0]*dl2_dx1;
  dg_dx1[I03] = df_dx1*llower[0]*llower[3] + f*dl0_dx1*llower[3] + f*llower[0]*dl3_dx1;
  dg_dx1[I11] = df_dx1*llower[1]*llower[1] + f*dl1_dx1*llower[1] + f*llower[1]*dl1_dx1;
  dg_dx1[I12] = df_dx1*llower[1]*llower[2] + f*dl1_dx1*llower[2] + f*llower[1]*dl2_dx1;
  dg_dx1[I13] = df_dx1*llower[1]*llower[3] + f*dl1_dx1*llower[3] + f*llower[1]*dl3_dx1;
  dg_dx1[I22] = df_dx1*llower[2]*llower[2] + f*dl2_dx1*llower[2] + f*llower[2]*dl2_dx1;
  dg_dx1[I23] = df_dx1*llower[2]*llower[3] + f*dl2_dx1*llower[3] + f*llower[2]*dl3_dx1;
  dg_dx1[I33] = df_dx1*llower[3]*llower[3] + f*dl3_dx1*llower[3] + f*llower[3]*dl3_dx1;

  // Set y-derivatives of covariant components
  dg_dx2[I00] = df_dx2*llower[0]*llower[0] + f*dl0_dx2*llower[0] + f*llower[0]*dl0_dx2;
  dg_dx2[I01] = df_dx2*llower[0]*llower[1] + f*dl0_dx2*llower[1] + f*llower[0]*dl1_dx2;
  dg_dx2[I02] = df_dx2*llower[0]*llower[2] + f*dl0_dx2*llower[2] + f*llower[0]*dl2_dx2;
  dg_dx2[I03] = df_dx2*llower[0]*llower[3] + f*dl0_dx2*llower[3] + f*llower[0]*dl3_dx2;
  dg_dx2[I11] = df_dx2*llower[1]*llower[1] + f*dl1_dx2*llower[1] + f*llower[1]*dl1_dx2;
  dg_dx2[I12] = df_dx2*llower[1]*llower[2] + f*dl1_dx2*llower[2] + f*llower[1]*dl2_dx2;
  dg_dx2[I13] = df_dx2*llower[1]*llower[3] + f*dl1_dx2*llower[3] + f*llower[1]*dl3_dx2;
  dg_dx2[I22] = df_dx2*llower[2]*llower[2] + f*dl2_dx2*llower[2] + f*llower[2]*dl2_dx2;
  dg_dx2[I23] = df_dx2*llower[2]*llower[3] + f*dl2_dx2*llower[3] + f*llower[2]*dl3_dx2;
  dg_dx2[I33] = df_dx2*llower[3]*llower[3] + f*dl3_dx2*llower[3] + f*llower[3]*dl3_dx2;

  // Set phi-derivatives of covariant components
  dg_dx3[I00] = df_dx3*llower[0]*llower[0] + f*dl0_dx3*llower[0] + f*llower[0]*dl0_dx3;
  dg_dx3[I01] = df_dx3*llower[0]*llower[1] + f*dl0_dx3*llower[1] + f*llower[0]*dl1_dx3;
  dg_dx3[I02] = df_dx3*llower[0]*llower[2] + f*dl0_dx3*llower[2] + f*llower[0]*dl2_dx3;
  dg_dx3[I03] = df_dx3*llower[0]*llower[3] + f*dl0_dx3*llower[3] + f*llower[0]*dl3_dx3;
  dg_dx3[I11] = df_dx3*llower[1]*llower[1] + f*dl1_dx3*llower[1] + f*llower[1]*dl1_dx3;
  dg_dx3[I12] = df_dx3*llower[1]*llower[2] + f*dl1_dx3*llower[2] + f*llower[1]*dl2_dx3;
  dg_dx3[I13] = df_dx3*llower[1]*llower[3] + f*dl1_dx3*llower[3] + f*llower[1]*dl3_dx3;
  dg_dx3[I22] = df_dx3*llower[2]*llower[2] + f*dl2_dx3*llower[2] + f*llower[2]*dl2_dx3;
  dg_dx3[I23] = df_dx3*llower[2]*llower[3] + f*dl2_dx3*llower[3] + f*llower[2]*dl3_dx3;
  dg_dx3[I33] = df_dx3*llower[3]*llower[3] + f*dl3_dx3*llower[3] + f*llower[3]*dl3_dx3;

  return;
}
