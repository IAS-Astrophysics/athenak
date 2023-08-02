//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyngr/dyngr.hpp"
#include "adm/adm.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  Real rout = pin->GetReal("problem", "outer_radius");
  Real rin  = rout - pin->GetReal("problem", "inner_radius");
  Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real bamb = pin->GetOrAddReal("problem", "bamb", 0.1);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD
  Real gm1;
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
    gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
    gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  }
  if (pmbp->pcoord->is_dynamical_relativistic) {
    gm1 = 1.0; // DynGR uses pressure, not energy.
  }

  par_for("pgen_blast1",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    Real den = da;
    Real pres = pa;
    if (rad < rout) {
      if (rad < rin) {
        den = drat*da;
        pres = prat*pa;
      } else {   // add smooth ramp in density
        Real f = (rad-rin) / (rout-rin);
        Real log_den = (1.0-f) * log(drat*da) + f * log(da);
        den = exp(log_den);
        Real log_pres = (1.0-f) * log(prat*pa) + f * log(pa);
        pres = exp(log_pres);
      }
    }

    w0_(m,IDN,k,j,i) = den;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;
    w0_(m,IEN,k,j,i) = pres/gm1;
  });

  // initialize magnetic fields ---------------------------------------

  if (pmbp->pmhd != nullptr) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_blast2",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = bamb;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;

      // Include extra face-component at edge of block in each direction
      if (i==ie) {b0.x1f(m,k,j,i+1) = bamb;}
      if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_blast3",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  }

  // Initialize ADM variables -----------------------------------------
  if (pmbp->padm != nullptr) {
    // Assume Minkowski space
    auto &adm = pmbp->padm->adm;
    int nmb1 = pmbp->nmb_thispack - 1;
    int ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
    par_for("pgen_adm_vars", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Set ADM to flat space
      adm.alpha(m, k, j, i) = 1.0;
      adm.beta_u(m, 0, k, j, i) = 0.0;
      adm.beta_u(m, 1, k, j, i) = 0.0;
      adm.beta_u(m, 2, k, j, i) = 0.0;

      adm.psi4(m, k, j, i) = 1.0;

      adm.g_dd(m, 0, 0, k, j, i) = 1.0;
      adm.g_dd(m, 0, 1, k, j, i) = 0.0;
      adm.g_dd(m, 0, 2, k, j, i) = 0.0;
      adm.g_dd(m, 1, 1, k, j, i) = 1.0;
      adm.g_dd(m, 1, 2, k, j, i) = 0.0;
      adm.g_dd(m, 2, 2, k, j, i) = 1.0;

      adm.K_dd(m, 0, 0, k, j, i) = 0.0;
      adm.K_dd(m, 0, 1, k, j, i) = 0.0;
      adm.K_dd(m, 0, 2, k, j, i) = 0.0;
      adm.K_dd(m, 1, 1, k, j, i) = 0.0;
      adm.K_dd(m, 1, 2, k, j, i) = 0.0;
      adm.K_dd(m, 2, 2, k, j, i) = 0.0;
    });

    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}
