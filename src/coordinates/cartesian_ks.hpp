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
#include "mesh/mesh.hpp"
  
#define SMALL_NUMBER 1.0e-5

//----------------------------------------------------------------------------------------
//! \fn void metric
//! \brief computes 9 components of metric in Cartesian Kerr-Schild coordinates
//

KOKKOS_INLINE_FUNCTION
void ComputeMetricAndInverse(Real x, Real y, Real z, Real g[], Real ginv[])
{
  using namespace global_variable;
  if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real R = fmax(sqrt(SQR(x) + SQR(y) + SQR(z)),1.0); // avoid singularity for R<1
  Real r = SQR(R)-SQR(bh_spin) + sqrt( SQR(SQR(R)-SQR(bh_spin))+4.0*SQR(bh_spin)*SQR(z) );
  r = sqrt(r/2.0);
  
  // Set covariant components
  // null vector l
  Real l_lower[4];
  l_lower[0] = 1.0;
  l_lower[1] = (r*x + (bh_spin)*y)/( SQR(r) + SQR(bh_spin) );
  l_lower[2] = (r*y - (bh_spin)*x)/( SQR(r) + SQR(bh_spin) );
  l_lower[3] = z/r;
  
  // g_nm = f*l_n*l_m + eta_nm, where eta_nm is Minkowski metric
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(bh_spin)*SQR(z));
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
