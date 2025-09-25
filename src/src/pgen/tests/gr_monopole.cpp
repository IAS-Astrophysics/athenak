//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_monopole.cpp
//  \brief Problem generator for monopole problem (following split monopole described in
//  Blandford-Znajek 1977 and implemented in, e.g., harmpi). Based on gr_torus pgen that
//  is included in this repository, with edits by GNW.

// C/C++ headers
#include <stdio.h>
#include <sys/stat.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio> // fclose

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

// user-defined BCs
void ReflectingMonopole(Mesh *pm);

// user-defined analysis called at end of run
void MonopoleDiagnostic(ParameterInput *pin, Mesh *pm);

// prototypes for functions used internally to this pgen
namespace {

KOKKOS_INLINE_FUNCTION
static void GetKerrSchildCoordinates(Real spin,
                                     Real x1, Real x2, Real x3,
                                     Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
Real A1(Real a_norm, Real spin, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(Real a_norm, Real spin, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(Real a_norm, Real spin, Real x1, Real x2, Real x3);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Monopole()
//  \brief Sets initial conditions for GR Monopole test
//   assumes x3 is axisymmetric direction

void ProblemGenerator::Monopole(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (!(pmbp->pcoord->is_general_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR monopole problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR monopole problem can only be run when MHD enabled via <mhd> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // set monopole diagnostics function
  pgen_final_func = MonopoleDiagnostic;

  // User boundary function
  user_bcs_func = ReflectingMonopole;

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  auto &size = pmbp->pmb->mb_size;

  // Extract BH parameters
  Real &spin = coord.bh_spin;

  // Extract conserved and primitive arrays
  auto &u0_ = pmbp->pmhd->u0;
  auto &w0_ = pmbp->pmhd->w0;

  // Get ideal gas EOS data
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  // Extract problem parameters
  Real sigma_max = pin->GetOrAddReal("problem", "sigma_max", 1.e2);
  Real rhomin = pin->GetOrAddReal("problem", "rhomin", 1.e-6);
  Real umin = pin->GetOrAddReal("problem", "umin", 1.e-8);
  Real a_norm = pin->GetOrAddReal("problem", "a_norm", 1.0);
  Real rh = 1.0 + sqrt(1.-SQR(spin));
  Real rc = 10.0*rh;
  Real &dexcise = coord.dexcise;
  Real &pexcise = coord.pexcise;

  // initialize primitive variables for new run ------------------------------------------

  par_for("pgen_monopole1", DevExeSpace(), 0,nmb-1,ks,ke,js,je,js,je,
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

    // Calculate Kerr-Schild r
    Real r, theta, phi;
    GetKerrSchildCoordinates(spin, x1v, x2v, x3v, &r, &theta, &phi);

    // Calculate background primitives
    Real rho_bg, pgas_bg;
    if (r > 1.0) {
      rho_bg  =     (rhomin + (r/rc)/pow(r,4.)/sigma_max);
      pgas_bg = gm1*(umin   + (r/rc)/pow(r,4.)/sigma_max);
    } else {
      rho_bg  = dexcise;
      pgas_bg = pexcise;
    }

    // Set primitives
    w0_(m,IDN,k,j,i) = rho_bg;
    w0_(m,IEN,k,j,i) = pgas_bg / gm1;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;
  });

  // initialize magnetic fields ----------------------------------------------------------

  // compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_monopole2", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1(a_norm, spin, x1v, x2f, x3f);
    a2(m,k,j,i) = A2(a_norm, spin, x1f, x2v, x3f);
    a3(m,k,j,i) = A3(a_norm, spin, x1f, x2f, x3v);

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
      a1(m,k,j,i) = 0.5*(A1(a_norm,spin,xl,x2f,x3f) + A1(a_norm,spin,xr,x2f,x3f));
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
      a2(m,k,j,i) = 0.5*(A2(a_norm,spin,x1f,xl,x3f) + A2(a_norm,spin,x1f,xr,x3f));
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
      a3(m,k,j,i) = 0.5*(A3(a_norm,spin,x1f,x2f,xl) + A3(a_norm,spin,x1f,x2f,xr));
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_torus2", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
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
  par_for("pgen_torus2", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  // Convert primitives to conserved
  pmbp->pmhd->peos->PrimToCons(w0_, bcc_, u0_, is, ie, js, je, ks, ke);

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Kerr-Schild coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Kerr-Schild coordinates

KOKKOS_INLINE_FUNCTION
static void GetKerrSchildCoordinates(Real spin, Real x1, Real x2, Real x3,
                                     Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  x3 = (rad < 1.0 && fabs(x3) < 1.0e-5) ? 1.0e-5 : x3;
  rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0);
  *pr = r;
  *ptheta = acos(((fabs(x3/r) > 1.0) ? copysign(1.0, x3/r) : x3/r));
  *pphi = atan2( (r*x2-spin*x1), (spin*x2+r*x1) );
  return;
}

//----------------------------------------------------------------------------------------
// Function to compute 1-component of vector potential.  First computes phi-componenent
// in KS coordinates, then transforms to Cartesian KS, assuming A_r = A_theta = 0
// A_\mu (cks) = A_nu (ks)  dx^nu (ks)/dx^\mu (cks) = A_phi (ks) dphi (ks)/dx^\mu
// phi_ks = arctan((r*y + a*x)/(r*x - a*y) )

KOKKOS_INLINE_FUNCTION
Real A1(Real a_norm, Real spin, Real x1, Real x2, Real x3) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  x3 = (rad < 1.0 && fabs(x3) < 1.0e-5) ? 1.0e-5 : x3;
  Real r, theta, phi;
  GetKerrSchildCoordinates(spin, x1, x2, x3, &r, &theta, &phi);

  Real aphi =  1.-cos(theta);
  aphi *= a_norm;

  Real sqrt_term =  2.0*SQR(r) - SQR(rad) + SQR(spin);

  Real a1_val = aphi*(-x2/(SQR(x1)+SQR(x2)) + spin*x1*r/((SQR(spin)+SQR(r))*sqrt_term));

  // multiply by ramp function that goes to zero at r = 0
  if (r < 1.0) {
    a1_val *= sin(0.5*M_PI*SQR(r));
  }

  //dphi/dx =  partial phi/partial x + partial phi/partial r partial r/partial x
  return a1_val;
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(Real a_norm, Real spin, Real x1, Real x2, Real x3) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  x3 = (rad < 1.0 && fabs(x3) < 1.0e-5) ? 1.0e-5 : x3;
  Real r, theta, phi;
  GetKerrSchildCoordinates(spin, x1, x2, x3, &r, &theta, &phi);

  Real aphi =  1.-cos(theta);
  aphi *= a_norm;

  Real sqrt_term =  2.0*SQR(r) - SQR(rad) + SQR(spin);

  Real a2_val = aphi*( x1/(SQR(x1)+SQR(x2)) + spin*x2*r/((SQR(spin)+SQR(r))*sqrt_term) );

  // multiply by ramp function that goes to zero at r = 0
  if (r < 1.0) {
    a2_val *= sin(0.5*M_PI*SQR(r));
  }

  //dphi/dx =  partial phi/partial y + partial phi/partial r partial r/partial y
  return a2_val;
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(Real a_norm, Real spin, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetKerrSchildCoordinates(spin, x1, x2, x3, &r, &theta, &phi);
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  x3 = (rad < 1.0 && fabs(x3) < 1.0e-5) ? 1.0e-5 : x3;

  Real aphi =  1.-cos(theta);
  aphi *= a_norm;

  Real sqrt_term =  2.0*SQR(r) - SQR(rad) + SQR(spin);

  Real a3_val = aphi*(spin*x3/(r*sqrt_term));

  // multiply by ramp function that goes to zero at r = 0
  if (r < 1.0) {
    a3_val *= sin(0.5*M_PI*SQR(r));
  }

  //dphi/dx =   partial phi/partial r partial r/partial z
  return a3_val;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ReflectingMonopole
//  \brief Sets boundary condition on surfaces of computational domain

void ReflectingMonopole(Mesh *pm) {
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
  auto &u0_ = pm->pmb_pack->pmhd->u0;
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &b0 = pm->pmb_pack->pmhd->b0;
  auto &bcc_ = pm->pmb_pack->pmhd->bcc0;
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // X1-Boundary
  // ConsToPrim over all x1 ghost zones *and* at the innermost/outermost x1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,is-ng,is,0,(n2-1),0,(n3-1));
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
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

  // PrimToCons on X1 ghost zones
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

  // X2-Boundary
  // ConsToPrim over all x2 ghost zones *and* at the innermost/outermost x2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),js-ng,js,0,(n3-1));
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),je,je+ng,0,(n3-1));
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

  // PrimToCons on X2 ghost zones
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

  // x3-Boundary
  // ConsToPrim over all x3 ghost zones *and* at the innermost/outermost x3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
  pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  // Set x3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = -w0_(m,n,ks,j,i);
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
  par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
        if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
        b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
        if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
        b0.x3f(m,ks-k-1,j,i) = -b0.x3f(m,ks,j,i);
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
  // PrimToCons on x3 ghost zones
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing monopole diagnostic at constant spherical KS radius

void MonopoleDiagnostic(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;
  Real rh = 1.0 + sqrt(1.-SQR(spin));

  // construct spherical grid
  int nlevel = pin->GetOrAddInteger("problem", "nlevel", 10);
  SphericalGrid *psph = new SphericalGrid(pmbp, nlevel, rh);

  // capture variables
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &bcc_ = pmbp->pmhd->bcc0;
  int nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;

  // interpolate cell-centered magnetic fields and store
  // NOTE(@pdmullen): We later reuse the interp_vals array to interpolate primitives.
  // Therefore, we must stow interpolated field components.
  psph->InterpolateToSphere(3, bcc_);
  DualArray2D<Real> interpolated_bcc;
  Kokkos::realloc(interpolated_bcc, psph->nangles, 3);
  Kokkos::deep_copy(interpolated_bcc, psph->interp_vals);
  interpolated_bcc.template modify<DevExeSpace>();
  interpolated_bcc.template sync<HostMemSpace>();

  // interpolate primitives
  psph->InterpolateToSphere(nvars, w0_);

  // calculate diagnostics
  std::vector<Real> monopole_diag(psph->nangles);
  for (int n=0; n<psph->nangles; ++n) {
    // extract coordinate data at this angle
    Real r = psph->radius;
    Real theta = psph->polar_pos.h_view(n,0);
    Real x1 = psph->interp_coord.h_view(n,0);
    Real x2 = psph->interp_coord.h_view(n,1);
    Real x3 = psph->interp_coord.h_view(n,2);
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

    // extract interpolated primitives
    Real &int_vx = psph->interp_vals.h_view(n,IVX);
    Real &int_vy = psph->interp_vals.h_view(n,IVY);
    Real &int_vz = psph->interp_vals.h_view(n,IVZ);

    // extract interpolated field components
    Real &int_bx = interpolated_bcc.h_view(n,IBX);
    Real &int_by = interpolated_bcc.h_view(n,IBY);
    Real &int_bz = interpolated_bcc.h_view(n,IBZ);

    // Compute interpolated u^\mu in CKS
    Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
             2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
             2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;
    Real u1 = int_vx - alpha * gamma * gupper[0][1];
    Real u2 = int_vy - alpha * gamma * gupper[0][2];
    Real u3 = int_vz - alpha * gamma * gupper[0][3];

    // Lower vector indices
    Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
    Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
    Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

    // Calculate 4-magnetic field
    Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
    Real b1 = (int_bx + b0 * u1) / u0;
    Real b2 = (int_by + b0 * u2) / u0;
    Real b3 = (int_bz + b0 * u3) / u0;

    // Transform CKS 4-velocity and 4-magnetic field to spherical KS
    Real a2 = SQR(spin);
    Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
    Real r2 = SQR(r);
    Real sth = sin(theta);
    Real drdx = r*x1/(2.0*r2 - rad2 + a2);
    Real drdy = r*x2/(2.0*r2 - rad2 + a2);
    Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
    Real dphdx = (-x2/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdx);
    Real dphdy = ( x1/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdy);
    Real dphdz = (spin/(r2 + a2)*drdz);
    // r,phi component of 4-velocity in spherical KS
    Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
    Real uph = dphdx*u1 + dphdy*u2 + dphdz*u3;
    // r,phi component of 4-magnetic field in spherical KS
    Real br  = drdx *b1 + drdy *b2 + drdz *b3;
    Real bph = dphdx*b1 + dphdy*b2 + dphdz*b3;

    // Compute field rotation rate (in units of rotation rate of horizon)
    // Should give value ~1/2
    Real omega = 0.0;
    if (x3 > 0) {omega = ((uph*br - ur*bph)/fmax(u0*br - ur*b0, 1.0e-12))/(0.5*spin/rh);}

    // store field rotation rate at theta, phi locations
    monopole_diag[n] = omega;
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, monopole_diag.data(), psph->nangles,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // root process opens output file and writes out diagnostics
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-diag.dat");
    FILE *pfile;

    if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Error output file could not be opened" <<std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(pfile, "# theta  phi  omega");
    std::fprintf(pfile, "\n");

    // write diagnostics
    for (int n=0; n<psph->nangles; ++n) {
      if (psph->interp_coord.h_view(n,2) > 0.0) {
        std::fprintf(pfile, "%12.5e ", psph->polar_pos.h_view(n,0));
        std::fprintf(pfile, "%12.5e ", psph->polar_pos.h_view(n,1));
        std::fprintf(pfile, "%12.5e ", monopole_diag[n]);
        std::fprintf(pfile, "\n");
      }
    }
    std::fclose(pfile);
  }

  // delete SphericalGrid object
  delete psph;

  return;
}
