//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_tov.cpp
//  \brief Problem generator for TOV star. Only works when ADM is enabled.

#include <math.h>     // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"

#include <Kokkos_Random.hpp>

// Prototypes for vector potential
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A1(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3);
template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A2(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3);

// Prototypes for user-defined BCs and history
void TOVHistory(HistoryData *pdata, Mesh *pm);

template<class TOVEOS>
void SetupTOV(ParameterInput *pin, Mesh* pmy_mesh_) {
  Real v_pert = pin->GetOrAddReal("problem", "v_pert", 0.0);
  Real p_pert = pin->GetOrAddReal("problem", "p_pert", 0.0);
  bool isotropic = pin->GetOrAddReal("problem", "isotropic", false);

  bool minkowski = pin->GetOrAddBoolean("problem", "minkowski", false);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Use the TOV solver with the specified EOS.
  TOVEOS eos{pin};
  auto my_tov = tov::TOVStar::ConstructTOV(pin, eos);


  constexpr bool use_ye = tov::UsesYe<TOVEOS>;
  Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);

  //auto& u0_ = pmbp->pmhd->u0;
  auto& w0_ = pmbp->pmhd->w0;
  int& nvars_ = pmbp->pmhd->nmhd;

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &ie = indcs.ie;
  int &je = indcs.je;
  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;
  auto &tov_ = my_tov;
  auto &eos_ = eos;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_tov1", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

    // Calculate the rest-mass density, pressure, and mass for a specific isotropic
    // radial coordinate.
    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    Real s = sqrt(SQR(x1v) + SQR(x2v));
    Real rho, p, mass, alp, r_schw;
    Real vr = 0.;
    Real p_pert = 0.;
    Real ye = ye_atmo;
    auto &use_ye_ = use_ye;
    if (!isotropic) {
      tov_.GetPrimitivesAtPoint(eos_, r, rho, p, mass, alp);
      if (r <= tov_.R_edge) {
        Real x = r/tov_.R_edge;
        vr = 0.5*v_pert*(3.0*x - x*x*x);
        auto rand_gen = rand_pool64.get_state();
        p_pert = 2.0*p_pert*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);
        if constexpr (use_ye) {
          ye = eos_.template GetYeFromRho<tov::LocationTag::Device>(rho);
        }
      }
    } else {
      tov_.GetPrimitivesAtIsoPoint(eos_, r, rho, p, mass, alp);
      r_schw = tov_.FindSchwarzschildR(r, mass);
      if (r_schw <= tov_.R_edge) {
        Real x = r_schw/tov_.R_edge;
        vr = 0.5*v_pert*(3.0*x - x*x*x);
        auto rand_gen = rand_pool64.get_state();
        p_pert = 2.0*p_pert*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);
        if constexpr (use_ye) {
          ye = eos_.template GetYeFromRho<tov::LocationTag::Device>(rho);
        }
      }
    }

    // Set hydrodynamic quantities
    //w0_(m,IDN,k,j,i) = fmax(rho, tov_.dfloor);
    //w0_(m,IPR,k,j,i) = fmax(p*(1. + p_pert), tov_.pfloor);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IPR,k,j,i) = p*(1. + p_pert);
    w0_(m,IVX,k,j,i) = vr*x1v/r;
    w0_(m,IVY,k,j,i) = vr*x2v/r;
    w0_(m,IVZ,k,j,i) = vr*x3v/r;
    auto &nvars = nvars_;
    if constexpr (use_ye) {
      w0_(m,nvars,k,j,i) = ye;
    }

    // Set ADM variables
    adm.alpha(m,k,j,i) = alp;
    if (minkowski) {
      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = 1.0;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.alpha(m,k,j,i) = 1.0;
    } else if (!isotropic) {
      // Auxiliary metric quantities
      Real fmet = 0.0;
      if (r > 0) {
        fmet = (1./(1. - 2*mass/r) - 1.)/(r*r);
      }

      adm.g_dd(m,0,0,k,j,i) = x1v*x1v*fmet + 1.0;
      adm.g_dd(m,0,1,k,j,i) = x1v*x2v*fmet;
      adm.g_dd(m,0,2,k,j,i) = x1v*x3v*fmet;
      adm.g_dd(m,1,1,k,j,i) = x2v*x2v*fmet + 1.0;
      adm.g_dd(m,1,2,k,j,i) = x2v*x3v*fmet;
      adm.g_dd(m,2,2,k,j,i) = x3v*x3v*fmet + 1.0;
      Real det = adm::SpatialDet(
              adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      adm.psi4(m,k,j,i) = pow(det, 1./3.);
    } else {
      Real fmet = 1.;
      if (r > 0) {
        fmet = r_schw/r;
      }
      Real psi4 = fmet*fmet;

      adm.g_dd(m,0,0,k,j,i) = adm.g_dd(m,1,1,k,j,i) = adm.g_dd(m,2,2,k,j,i) = psi4;
      adm.g_dd(m,0,1,k,j,i) = adm.g_dd(m,0,2,k,j,i) = adm.g_dd(m,1,2,k,j,i) = 0.0;
      adm.psi4(m,k,j,i) = psi4;
    }
    adm.beta_u(m,0,k,j,i) = adm.beta_u(m,1,k,j,i) = adm.beta_u(m,2,k,j,i) = 0.0;
    adm.vK_dd(m,0,0,k,j,i) = adm.vK_dd(m,0,1,k,j,i) = adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = adm.vK_dd(m,1,2,k,j,i) = adm.vK_dd(m,2,2,k,j,i) = 0.0;
  });

  // parse some parameters
  Real b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
  Real pcut = pin->GetOrAddReal("problem", "pcut", 1e-6);
  Real magindex = pin->GetOrAddReal("problem", "magindex", 2);

  // If use_pcut_rel = true, we take pcut to be a percentage of pmax rather than
  // an absolute cutoff
  if (pin->GetOrAddBoolean("problem", "use_pcut_rel", false)) {
    Real pmax = eos_.template GetPFromRho<tov::LocationTag::Device>(tov_.rhoc);
    pcut = pcut * pmax;
  }

  // compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pmbp->nmb_thispack;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
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

    Real x1fp1 = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x2fp1 = LeftEdgeX(j+1-js, nx2, x2min, x2max);
    Real x3fp1 = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1(tov_, eos_, isotropic, pcut, magindex, x1v, x2f, x3f);
    a2(m,k,j,i) = A2(tov_, eos_, isotropic, pcut, magindex, x1f, x2v, x3f);
    a3(m,k,j,i) = 0.0;

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
      a1(m,k,j,i) = 0.5*(A1(tov_, eos_, isotropic, pcut, magindex, xl,x2f,x3f) +
                         A1(tov_, eos_, isotropic, pcut, magindex, xr,x2f,x3f));
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
      a2(m,k,j,i) = 0.5*(A2(tov_, eos_, isotropic, pcut, magindex, x1f,xl,x3f) +
                         A2(tov_, eos_, isotropic, pcut, magindex, x1f,xr,x3f));
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = b_norm*((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                       (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = b_norm*((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                       (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = b_norm*((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                       (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = b_norm*((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                           (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = b_norm*((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                           (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = b_norm*((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                           (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_Bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
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

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Sets initial conditions for TOV star in DynGRMHD
//  Compile with '-D PROBLEM=dyngr_tov' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "TOV star problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &TOVHistory;

  // initialize primitive variables for restart
  if (restart) {
    return;
  }

  // Select the right TOV template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupTOV<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupTOV<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for TOV star problem" << std::endl
              << "Defaulting to fixed polytropic EOS" << std::endl;
    SetupTOV<tov::PolytropeEOS>(pin, pmy_mesh_);
  }

  // Mesh block info for loop limits
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  // Convert primitives to conserved
  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<2>(pmbp);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<3>(pmbp);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              pmbp->pz4c->ADMConstraints<4>(pmbp);
              break;
    }
  }

  return;
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPrimitivesAtPoint(const tov_pgen& tov, const TOVEOS& eos, Real r,
                                 Real &rho, Real &p, Real &m, Real &alp) {
  // Check if we're past the edge of the star.
  // If so, we just return atmosphere with Schwarzschild.
  if (r >= tov.R_edge) {
    rho = 0.0;
    p = 0.0;
    m = tov.M_edge;
    alp = sqrt(1.0 - 2.0*m/r);
    return;
  }
  // Get the lower index for where our point must be located.
  int idx = static_cast<int>(r/tov.dr);
  const auto &R = tov.R.d_view;
  const auto &Ps = tov.P.d_view;
  const auto &alps = tov.alp.d_view;
  const auto &Ms = tov.M.d_view;
  // Interpolate to get the primitive.
  p = Interpolate(r, R(idx), R(idx+1), Ps(idx), Ps(idx+1));
  m = Interpolate(r, R(idx), R(idx+1), Ms(idx), Ms(idx+1));
  alp = Interpolate(r, R(idx), R(idx+1), alps(idx), alps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

KOKKOS_INLINE_FUNCTION
static int FindIsotropicIndex(const tov_pgen& tov, Real r_iso) {
  // Perform a bisection search to find the closest index to the requested isotropic
  // point.
  const auto &R_iso = tov.R_iso.d_view;
  int lb = 0;
  int ub = tov.n_r;
  int idx = lb;
  while (R_iso(lb+1) < r_iso) {
    idx = (lb + ub)/2;
    if (R_iso(idx) < r_iso) {
      lb = idx;
    } else {
      ub = idx;
    }
  }
  return lb;
}

KOKKOS_INLINE_FUNCTION
static Real FindSchwarzschildR(const tov_pgen& tov, Real r_iso, Real mass) {
  if (r_iso > tov.R_edge_iso) {
    Real psi = 1.0 + mass/(2.*r_iso);
    return r_iso*psi*psi;
  }

  int idx = FindIsotropicIndex(tov, r_iso);
  const auto &R_iso = tov.R_iso.d_view;
  const auto &R = tov.R.d_view;
  return Interpolate(r_iso, R_iso(idx), R_iso(idx+1), R(idx), R(idx+1));
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPrimitivesAtIsoPoint(const tov_pgen& tov, const TOVEOS& eos, Real r_iso,
                                    Real &rho, Real &p, Real &m, Real &alp) {
  // Check if we're past the edge of the star.
  // If so, we just return atmosphere with Schwarzschild.
  if (r_iso >= tov.R_edge_iso) {
    rho = 0.0;
    p = 0.0;
    m = tov.M_edge;
    alp = (1. - m/(2.*r_iso))/(1. + m/(2.*r_iso));
    return;
  }
  // Because the isotropic coordinates are not evenly spaced, we need to search to find
  // the right index. We can set a lower bound because r_iso <= r, and then we choose the
  // edge of the star as an upper bound.
  const auto &R_iso = tov.R_iso.d_view;
  int idx = FindIsotropicIndex(tov, r_iso);
  const auto &Ps = tov.P.d_view;
  const auto &alps = tov.alp.d_view;
  const auto &Ms = tov.M.d_view;
  if (idx >= tov.npoints || idx < 0) {
    Kokkos::printf("There's a problem with the index!\n" // NOLINT
           " idx = %d\n"
           " r_iso = %g\n"
           " dr = %g\n",idx,r_iso,tov.dr);
  }
  // Interpolate to get the primitive.
  p = Interpolate(r_iso, R_iso(idx), R_iso(idx+1), Ps(idx), Ps(idx+1));
  m = Interpolate(r_iso, R_iso(idx), R_iso(idx+1), Ms(idx), Ms(idx+1));
  alp = Interpolate(r_iso, R_iso(idx), R_iso(idx+1), alps(idx), alps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(fmax(p, tov.pfloor));
  if (!isfinite(p)) {
    Kokkos::printf("There's a problem with p!\n"); // NOLINT
    assert(false);
  }
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPandRho(const tov_pgen& tov, const TOVEOS& eos,
                       Real r, Real &rho, Real &p) {
  if (r >= tov.R_edge) {
    rho = 0.;
    p   = 0.;
    return;
  }
  // Get the lower index for where our point must be located.
  int idx = static_cast<int>(r/tov.dr);
  const auto &R = tov.R.d_view;
  const auto &Ps = tov.P.d_view;
  // Interpolate to get the pressure
  p = Interpolate(r, R(idx), R(idx+1), Ps(idx), Ps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static void GetPandRhoIso(const tov_pgen& tov, const TOVEOS& eos,
                          Real r, Real &rho, Real &p) {
  if (r >= tov.R_edge_iso) {
    rho = 0.;
    p   = 0.;
    return;
  }
  // We need to search to find the right index because isotropic coordinates aren't
  // evenly spaced.
  int idx = FindIsotropicIndex(tov, r);
  const auto R_iso = tov.R_iso.d_view;
  const auto &Ps = tov.P.d_view;
  p = Interpolate(r, R_iso(idx), R_iso(idx+1), Ps(idx), Ps(idx+1));
  // FIXME: Assumes ideal gas!
  //rho = pow(p/tov.kappa, 1.0/tov.gamma);
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A1(const tov_pgen& tov, const TOVEOS& eos, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real p, rho;
  if (!isotropic) {
    tov_.GetPandRho(eos, r, rho, p);
  } else {
    tov_.GetPandRhoIso(eos, r, rho, p);
  }
  return -x2*fmax(p - pcut, 0.0)*pow(1.0 - rho/tov_.rhoc,magindex);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
static Real A2(const tov::TOVStar& tov_, const TOVEOS& eos, bool isotropic, Real pcut,
               Real magindex, Real x1, Real x2, Real x3) {
  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real p, rho;
  if (!isotropic) {
    tov_.GetPandRho(eos, r, rho, p);
  } else {
    tov_.GetPandRhoIso(eos, r, rho, p);
  }
  return x1*fmax(p - pcut, 0.0)*pow(1.0 - rho/tov_.rhoc,magindex);
}

// History function
void TOVHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    // coompute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need MPI_MAX
  // and MPI_MIN operations instead. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}
