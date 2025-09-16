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

#include <math.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

namespace {
  Real R_max0;     // Maximum radius at t=t0.
  Real v_max;      // Maximum speed.
  Real t0;
  Real fac;
  void SetADMVariablesToFLRW(MeshBlockPack *pmbp);
}

KOKKOS_INLINE_FUNCTION
Real GetCartesianFromSnake(Real w, Real y, Real A, Real k) {
  return w + A*sin(k*M_PI*y);
}

KOKKOS_INLINE_FUNCTION
void GetCartesianFromRipple(Real &x, Real &y, Real w, Real v, Real A, Real k) {
  // We do a 2D Newton-Raphson solve for the Cartesian coordinates. Since it follows that
  // w \in [x-A, x+A], we know that x \in [w-2A, w+2A]. The same holds for y. Thus we can
  // use these as bounds to keep the solver from diverging.
  Real xlb = w - 2.0*A;
  Real xub = w + 2.0*A;
  Real ylb = v - 2.0*A;
  Real yub = v + 2.0*A;
  Real tol = 1e-15;

  x = w;
  y = v;

  Real fx = x - w - A*sin(k*M_PI*y);
  Real fy = y - v - A*sin(k*M_PI*x);
  int its = 0;
  int max_its = 30;
  while ((fabs(fx) > tol || fabs(fy) > tol) && its < max_its) {
    // Auxiliary quantities needed for the root solve.
    Real delx = A*k*M_PI*cos(k*M_PI*x);
    Real dely = A*k*M_PI*cos(k*M_PI*y);
    Real idet = 1.0/(1.0 - delx*dely);

    // Estimate the updated roots (J^-1 f)
    x = x - (fx + dely*fy)*idet;
    y = y - (fy + delx*fx)*idet;
    // Limit the roots
    x = (x < xlb) ? xlb : x;
    x = (x > xub) ? xub : x;
    y = (y < ylb) ? ylb : y;
    y = (y > yub) ? yub : y;

    // Update the function values
    fx = x - w - A*sin(k*M_PI*y);
    fy = y - v - A*sin(k*M_PI*x);
  }
}

KOKKOS_INLINE_FUNCTION
Real GetCartesianFromScrewball(Real u, Real a) {
  // We define u = x*exp(-x^2/2a^2), so the transformation from u to x is not analytic.
  // However, we restrict u s.t. u \in [0, a/sqrt(e)], so we also know that x \in [u, a].
  // We thus define x as the solution to x - u*exp(x^2/2a^2) = 0.

  // In the event that u = 0, we can return the exact solution.
  if (u == 0.0 || u == -0.0) {
    return 0.0;
  }

  // Flip the sign of u if necessary. We can do this because u is an odd function.
  Real sign = 1.0;
  if (u < 0) {
    sign = -1;
    u = -u;
  }
  Real lb = u;
  Real ub = a;
  Real x = 0.5*(lb + ub);
  Real gauss = exp(x*x/(2.0*a*a));
  Real f = x - u*gauss;
  Real tol = 1e-15;
  int its = 0;
  int max_its = 30;
  while (fabs(f) > tol && its < max_its) {
    // Newton iteration.
    Real df = 1.0 - u*x/(a*a)*gauss;
    x = x - f/df;

    // Use bisection if we would travel outside the bracket.
    if (x < lb || x > ub) {
      x = 0.5*(lb + ub);
    }
    // Check how well x satisfies f.
    gauss = exp(x*x/(2.0*a*a));
    f = x - u*gauss;
    // Update the bracket.
    if (f < 0) {
      lb = x;
    } else {
      ub = x;
    }
    its++;
  }

  return sign*x;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_expanding = pin->GetOrAddBoolean("problem", "flrw", false);
  if (is_expanding) {
    R_max0 = pin->GetOrAddReal("problem", "R_max0", 1.0);
    v_max = pin->GetOrAddReal("problem", "v_max0", 1.0);
    t0 = pin->GetOrAddReal("problem", "t0", 0.0);
    fac = v_max / R_max0;
    pmbp->padm->SetADMVariables = &SetADMVariablesToFLRW;
  }

  if (restart) return;

  Real rout = pin->GetReal("problem", "outer_radius");
  Real rin  = rout - pin->GetReal("problem", "inner_radius");
  // values for neutrals (hydro fluid)
  Real pn_amb   = pin->GetOrAddReal("problem", "pn_amb", 1.0);
  Real dn_amb   = pin->GetOrAddReal("problem", "dn_amb", 1.0);
  // values for ions (hydro fluid)
  Real pi_amb   = pin->GetOrAddReal("problem", "pi_amb", 1.0);
  Real di_amb   = pin->GetOrAddReal("problem", "di_amb", 1.0);
  // ratios in blast (same for both ions and neutrals)
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real b_amb = pin->GetOrAddReal("problem", "b_amb", 0.1);
  std::string coords = pin->GetOrAddString("problem", "coordinates", "cartesian");
  bool warp = false;
  bool snake = false;
  bool ripple = false;
  if (coords.compare("cartesian") != 0 && pmy_mesh_->pmb_pack->padm == nullptr) {
    std::cout << "Alternate coordinates are only supported for DynGRMHD.\n"
              << "Defaulting to Cartesian.\n";
  } else {
    if (coords.compare("warp") == 0) {
      warp = true;
    } else if (coords.compare("snake") == 0) {
      snake = true;
    } else if (coords.compare("ripple") == 0) {
      ripple = true;
    } else if (coords.compare("cartesian") != 0) {
      std::cout << "Unknown coordinates '" << coords << "' requested.\n"
                << "Defaulting to Cartesian.\n";
    }
  }
  Real a_warp = pin->GetOrAddReal("problem", "a_warp", 6.0);
  Real A_snake = pin->GetOrAddReal("problem", "A_snake", 0.1);
  Real k_snake = pin->GetOrAddReal("problem", "k_snake", 2.0);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    auto &w0_ = pmbp->phydro->w0;
    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
    if (pmbp->pcoord->is_dynamical_relativistic) {
      gm1 = 1.0; // DynGRMHD uses pressure, not energy.
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

      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

      Real den = dn_amb;
      Real pres = pn_amb;
      if (rad < rout) {
        if (rad < rin) {
          den *= drat;
          pres *= prat;
        } else {   // add smooth ramp in density
          Real f = (rad-rin) / (rout-rin);
          Real log_den = (1.0-f) * log(drat*dn_amb) + f * log(dn_amb);
          den = exp(log_den);
          Real log_pres = (1.0-f) * log(prat*pn_amb) + f * log(pn_amb);
          pres = exp(log_pres);
        }
      }
      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVY,k,j,i) = 0.0;
      w0_(m,IVZ,k,j,i) = 0.0;
      w0_(m,IEN,k,j,i) = pres/gm1;
    });

    // Convert primitives to conserved
    if (!pmbp->pcoord->is_dynamical_relativistic) {
      pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);
    }
  }  // End initialization Hydro variables

  // initialize MHD variables ------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &w0_ = pmbp->pmhd->w0;
    Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
    if (pmbp->pcoord->is_dynamical_relativistic) {
      gm1 = 1.0; // DynGRMHD uses pressure, not energy.
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

      Real rad;
      if (warp) {
        Real x = GetCartesianFromScrewball(x1v, a_warp);
        Real y = GetCartesianFromScrewball(x2v, a_warp);
        rad = sqrt(SQR(x) + SQR(y) + SQR(x3v));
      } else if (snake) {
        Real x = GetCartesianFromSnake(x1v, x2v, A_snake, k_snake);
        rad = sqrt(SQR(x) + SQR(x2v) + SQR(x3v));
      } else if (ripple) {
        Real x, y;
        GetCartesianFromRipple(x, y, x1v, x2v, A_snake, k_snake);
        rad = sqrt(SQR(x) + SQR(y) + SQR(x3v));
      } else {
        rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      }

      Real den = di_amb;
      Real pres = pi_amb;
      if (rad < rout) {
        if (rad < rin) {
          den *= drat;
          pres *= prat;
        } else {   // add smooth ramp in density
          Real f = (rad-rin) / (rout-rin);
          Real log_den = (1.0-f) * log(drat*di_amb) + f * log(di_amb);
          den = exp(log_den);
          Real log_pres = (1.0-f) * log(prat*pi_amb) + f * log(pi_amb);
          pres = exp(log_pres);
        }
      }
      w0_(m,IDN,k,j,i) = den;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVY,k,j,i) = 0.0;
      w0_(m,IVZ,k,j,i) = 0.0;
      w0_(m,IEN,k,j,i) = pres/gm1;
    });

    // initialize magnetic fields
    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = pmbp->nmb_thispack;
    DvceArray4D<Real> a3;
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    int ku = ke;
    if (ncells3 > 1) {
      ku = ke + 1;
    }

    par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ku,js,je+1,is,ie+1,
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

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      Real y = x2f;

      if (warp) {
        y = GetCartesianFromScrewball(x2f, a_warp);
      } else if (ripple) {
        Real x;
        GetCartesianFromRipple(x, y, x1f, x2f, A_snake, k_snake);
      }

      a3(m,k,j,i) = b_amb*y;
    });


    // initialize magnetic fields
    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_blast2",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;

      b0.x1f(m,k,j,i) = (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2;
      b0.x2f(m,k,j,i) = -(a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2;
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = -(a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
      }
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

    // Convert primitives to conserved
    if (!pmbp->pcoord->is_dynamical_relativistic) {
      pmbp->pmhd->peos->PrimToCons(w0_, bcc_, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
    }
  }  // End initialization MHD variables

  // Initialize ADM variables -----------------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
void SetADMVariablesToFLRW(MeshBlockPack *pmbp) {
  const Real t = pmbp->pmesh->time;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  Real a = 1.0 + fac*(t-t0);
  Real a2 = a*a;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
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

    adm.g_dd(m,0,0,k,j,i) = a2;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,1,k,j,i) = a2;
    adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.g_dd(m,2,2,k,j,i) = a2;

    adm.vK_dd(m,0,0,k,j,i) = -a*fac;
    adm.vK_dd(m,0,1,k,j,i) = 0.0;
    adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = -a*fac;
    adm.vK_dd(m,1,2,k,j,i) = 0.0;
    adm.vK_dd(m,2,2,k,j,i) = -a*fac;

    adm.alpha(m,k,j,i) = 1.0;
    adm.beta_u(m,0,k,j,i) = 0.0;
    adm.beta_u(m,1,k,j,i) = 0.0;
    adm.beta_u(m,2,k,j,i) = 0.0;
  });
}

} // namespace
