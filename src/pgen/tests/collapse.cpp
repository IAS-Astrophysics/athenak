//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file collapse.cpp
//! \brief Problem generator for Oppenheimer-Snyder spherical dust collapse in full GR.
//! Can be done in full GR with DynGRMHD and Z4c, or can optionally be done with an
//! analytic metric for DynGRMHD alone.

#include <math.h>

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "pgen/pgen.hpp"

// Useful container for physical parameters of the collapse
struct collapse_pgen {
  Real mass;
  Real R0;
} collapse;

void AnalyticCollapse(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::SphericalCollapse_()
//! \brief Problem Generator for the spherical collapse tests
void ProblemGenerator::SphericalCollapse(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Spherical collapse problem can only be run with DynGRMHD.\n"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (restart) return;
  // parse input parameters
  Real mass = pin->GetReal("problem", "mass");
  Real R0   = pin->GetReal("problem", "R0");

  collapse.mass = mass;
  collapse.R0   = R0;

  // Calculate the base density
  Real rho0 = mass/(4./3.*M_PI*R0*R0*R0);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int ng = indcs.ng;
  int nmb1 = pmbp->nmb_thispack - 1;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  // Initialize MHD and ADM variables
  auto &w0 = pmbp->pmhd->w0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  auto &adm = pmbp->padm->adm;
  par_for("pgen_init", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real y = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real z = CellCenterX(k-ks, nx3, x3min, x3max);

    Real r = sqrt(x*x + y*y + z*z);

    Real M = mass;

    // MHD variables
    if (r < R0) {
      M = 4./3.*M_PI*r*r*r*rho0;
      w0(m, IDN, k, j, i) = rho0;
    } else {
      w0(m, IDN, k, j, i) = 0.0;
    }
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IPR, k, j, i) = 0.0;
    b0.x1f(m, k, j, i) = 0.0;
    b0.x2f(m, k, j, i) = 0.0;
    b0.x3f(m, k, j, i) = 0.0;
    if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
    if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
    if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
    bcc0(m, IBX, k, j, i) = 0.0;
    bcc0(m, IBY, k, j, i) = 0.0;
    bcc0(m, IBZ, k, j, i) = 0.0;

    Real a, b;
    if (r < 1e-15) {
      a = 1.;
      b = 0.;
    }
    a = 1. - 2*M/(r + 1.e-15);
    b = (1.0/a - 1.0)/SQR(r + 1.e-15);

    // ADM variables
    adm.alpha(m, k, j, i) = sqrt(a);
    adm.beta_u(m, 0, k, j, i) = 0.0;
    adm.beta_u(m, 1, k, j, i) = 0.0;
    adm.beta_u(m, 2, k, j, i) = 0.0;

    adm.psi4(m, k, j, i) = 1.0/sqrt(adm.alpha(m, k, j, i));
    adm.g_dd(m, 0, 0, k, j, i) = 1. + x*x*b;
    adm.g_dd(m, 0, 1, k, j, i) = 2.*x*y*b;
    adm.g_dd(m, 0, 2, k, j, i) = 2.*x*z*b;
    adm.g_dd(m, 1, 1, k, j, i) = 1 + y*y*b;
    adm.g_dd(m, 1, 2, k, j, i) = 2.*y*z*b;
    adm.g_dd(m, 2, 2, k, j, i) = 1 + z*z*b;

    adm.vK_dd(m, 0, 0, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 2, 2, k, j, i) = 0.0;
  });

  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              break;
    }
  }
}

void AnalyticCollapse(MeshBlockPack* pmbp) {
}
