//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file diffusion.cpp
//! \brief problem generator for tests of the diffusion modules (viscosity, resistivity,
//! thermal conduction).  Sets up a Gaussian pulse that spreads under pure diffusion.
//! The pulse can vary along one, two, or three coordinate axes (selected with the
//! spread_x1/x2/x3 flags), giving an isotropic n-dimensional Gaussian whose amplitude
//! decays as 1/(1+4*D*t)^(n/2).  For thermal conduction the pulse is in the gas
//! temperature/pressure; for viscosity the pulse is in a single (transverse) velocity
//! component selected with vel_comp; for resistivity the pulse is in a single
//! (transverse) magnetic-field component (also selected with vel_comp: 1->Bx, 2->By,
//! 3->Bz, stored on the corresponding cell face and uniform along its own axis so that
//! div(B)=0).  Must be run in kinematic mode.  Errors in the final solution are computed
//! from the analytic profile at final time in DiffusionErrors().  Resistivity runs use
//! the <mhd> block; conduction/viscosity use the <hydro> block.

// C headers

// C++ headers
#include <cmath>      // sqrt(), pow(), exp()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/resistivity.hpp"

// Prototype for function to compute errors in solution at end of run
void DiffusionErrors(ParameterInput *pin, Mesh *pm);
// Prototype for user-defined BCs
void GaussianProfileBCs(Mesh *pm);

// Anonymous namespace used to prevent name collisions outside of this file
namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;
// input parameters passed to the initialization kernel and user-defined BC function
struct DiffusionVariables {
  bool conduction_test, viscosity_test, resistivity_test;
  bool spread_x1, spread_x2, spread_x3;
  int vel_comp;                 // 1/2/3 -> velocity (viscosity) or B (resistivity) compt
  Real amp, x10, x20, x30;      // amplitude and Gaussian centers
};

DiffusionVariables diffvars;

//----------------------------------------------------------------------------------------
//! \fn DiffusionGaussian
//! \brief Device-callable helper returning the scalar amplitude of the isotropic n-D
//! radial Gaussian pulse at (x1,x2,x3) and time, spreading with diffusivity "coef".

KOKKOS_INLINE_FUNCTION
Real DiffusionGaussian(const DiffusionVariables dv, const Real coef, const Real time,
                       const Real x1, const Real x2, const Real x3) {
  Real ndim = (dv.spread_x1 ? 1.0 : 0.0)
            + (dv.spread_x2 ? 1.0 : 0.0)
            + (dv.spread_x3 ? 1.0 : 0.0);
  Real spread = 1.0 + 4.0*coef*time;
  Real r2 = 0.0;
  if (dv.spread_x1) r2 += SQR(x1 - dv.x10);
  if (dv.spread_x2) r2 += SQR(x2 - dv.x20);
  if (dv.spread_x3) r2 += SQR(x3 - dv.x30);
  return (dv.amp/pow(spread, 0.5*ndim))*exp(-r2/spread);
}

//----------------------------------------------------------------------------------------
//! \fn DiffusionConsState
//! \brief Device-callable helper that returns the analytic conserved state of the
//! Gaussian-pulse diffusion solution at position (x1,x2,x3) and time. "coef" is the
//! relevant diffusivity (nu_iso for viscosity, alpha_iso for conduction), "gamma" the
//! adiabatic index.  Results are written into cons[IDN..IEN].

KOKKOS_INLINE_FUNCTION
void DiffusionConsState(const DiffusionVariables dv, const Real coef, const Real gamma,
                        const Real time, const Real x1, const Real x2, const Real x3,
                        Real cons[5]) {
  Real g = DiffusionGaussian(dv, coef, time, x1, x2, x3);

  Real gm1 = gamma - 1.0;
  Real rho = 1.0;
  Real m1 = 0.0, m2 = 0.0, m3 = 0.0;
  Real p0 = 1.0/gamma;       // background pressure (used by viscosity test)
  if (dv.conduction_test) {
    p0 = g;                  // Gaussian temperature/pressure pulse
  }
  if (dv.viscosity_test) {
    if (dv.vel_comp == 1) {
      m1 = rho*g;
    } else if (dv.vel_comp == 2) {
      m2 = rho*g;
    } else {
      m3 = rho*g;
    }
  }
  cons[IDN] = rho;
  cons[IM1] = m1;
  cons[IM2] = m2;
  cons[IM3] = m3;
  cons[IEN] = p0/gm1 + 0.5*(SQR(m1) + SQR(m2) + SQR(m3))/rho;
}

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Diffusion()
//! \brief Sets initial conditions for diffusion tests

void ProblemGenerator::Diffusion(ParameterInput *pin, const bool restart) {
  std::string evolution_t = pin->GetString("time","evolution");
  if (evolution_t.compare("kinematic") != 0) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "Diffusion tests must be run in kinematic mode" << std::endl;
    exit(EXIT_FAILURE);
  }
  // set diffusion errors function
  pgen_final_func = DiffusionErrors;
  // user-defined BC
  user_bcs_func = GaussianProfileBCs;
  if (restart) return;

  // Read problem parameters
  diffvars.amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  diffvars.x10 = pin->GetOrAddReal("problem", "x10", 0.0);
  diffvars.x20 = pin->GetOrAddReal("problem", "x20", 0.0);
  diffvars.x30 = pin->GetOrAddReal("problem", "x30", 0.0);
  diffvars.conduction_test = pin->GetBoolean("problem", "conduction_test");
  diffvars.viscosity_test = pin->GetBoolean("problem", "viscosity_test");
  diffvars.resistivity_test = pin->GetBoolean("problem", "resistivity_test");
  diffvars.spread_x1 = pin->GetOrAddBoolean("problem", "spread_x1", true);
  diffvars.spread_x2 = pin->GetOrAddBoolean("problem", "spread_x2", false);
  diffvars.spread_x3 = pin->GetOrAddBoolean("problem", "spread_x3", false);
  diffvars.vel_comp = pin->GetOrAddInteger("problem", "vel_comp", 2);
  int ntests = (diffvars.conduction_test ? 1 : 0)
             + (diffvars.viscosity_test ? 1 : 0)
             + (diffvars.resistivity_test ? 1 : 0);
  if (ntests != 1) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "Exactly one of conduction_test/viscosity_test/resistivity_test must be "
            << "set true (got " << ntests << ")" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  Real time = pmbp->pmesh->time;
  auto dv = diffvars;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    if (dv.conduction_test && pmbp->phydro->pcond == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Conduction not defined in Hydro input block" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (dv.viscosity_test && pmbp->phydro->pvisc == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Viscosity not defined in Hydro input block" << std::endl;
      exit(EXIT_FAILURE);
    }

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    if (!(eos.is_ideal)) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Diffusion test requires ideal EOS in Hydro block" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real gamma = eos.gamma;
    // Effective diffusivity of the pulse for the active test.  For viscosity the
    // transverse momentum diffuses with the kinematic viscosity nu_iso.  For conduction
    // the heat flux uses T = (gamma-1)*eint/rho = p/rho, giving an energy/pressure pulse
    // that (with rho=1) diffuses with D = (gamma-1) * alpha_iso.
    Real coef = 0.0;
    if (dv.conduction_test) coef = (gamma-1.0)*pmbp->phydro->pcond->alpha_iso;
    if (dv.viscosity_test) coef = pmbp->phydro->pvisc->nu_iso;

    // compute solution in u1 register. For initial conditions, set u1 -> u0.
    auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;

    par_for("pgen_diff", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real cons[5];
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u1(m,IDN,k,j,i) = cons[IDN];
      u1(m,IM1,k,j,i) = cons[IM1];
      u1(m,IM2,k,j,i) = cons[IM2];
      u1(m,IM3,k,j,i) = cons[IM3];
      u1(m,IEN,k,j,i) = cons[IEN];
    });
  } // End initialization of Hydro variables

  // Initialize MHD variables (resistivity test) --------------
  if (pmbp->pmhd != nullptr) {
    if (!dv.resistivity_test) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "MHD diffusion test only supports the resistivity test" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (pmbp->pmhd->presist == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Resistivity (mhd/ohmic_resistivity) not defined in MHD input block"
              << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    if (!(eos.is_ideal)) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Diffusion test requires ideal EOS in MHD block" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real gamma = eos.gamma;
    Real gm1 = gamma - 1.0;
    // The transverse magnetic field diffuses with the Ohmic diffusivity eta_ohm.
    Real coef = pmbp->pmhd->presist->eta_ohm;
    int bcomp = dv.vel_comp;          // 1->Bx, 2->By, 3->Bz
    Real p0 = 1.0/gamma;              // uniform background gas pressure

    // face- and cell-centered fields go to b0/b1 depending on whether these are ICs
    auto &b = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;
    auto &bcc0 = pmbp->pmhd->bcc0;
    auto &w0 = pmbp->pmhd->w0;

    // The pulse lives in a single B component that is uniform along its own coordinate
    // axis (which is never a spread axis), so the staggered face value equals the
    // Gaussian evaluated at the cell-centered spread coordinates and div(B)=0.
    par_for("pgen_resist_b", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real g = DiffusionGaussian(dv, coef, time, x1v, x2v, x3v);
      b.x1f(m,k,j,i) = (bcomp == 1) ? g : 0.0;
      b.x2f(m,k,j,i) = (bcomp == 2) ? g : 0.0;
      b.x3f(m,k,j,i) = (bcomp == 3) ? g : 0.0;
      if (i == ie) {
        b.x1f(m,k,j,i+1) = (bcomp == 1) ? g : 0.0;
      }
      if (j == je) {
        b.x2f(m,k,j+1,i) = (bcomp == 2) ? g : 0.0;
      }
      if (k == ke) {
        b.x3f(m,k+1,j,i) = (bcomp == 3) ? g : 0.0;
      }

      // cell-centered field and primitive state (rho=1, v=0, uniform pressure)
      bcc0(m,IBX,k,j,i) = (bcomp == 1) ? g : 0.0;
      bcc0(m,IBY,k,j,i) = (bcomp == 2) ? g : 0.0;
      bcc0(m,IBZ,k,j,i) = (bcomp == 3) ? g : 0.0;
      w0(m,IDN,k,j,i) = 1.0;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = p0/gm1;
    });

    // convert primitives + cell-centered B to conserved variables in u0/u1
    auto &u = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
    pmbp->pmhd->peos->PrimToCons(w0, bcc0, u, is, ie, js, je, ks, ke);
  } // End initialization of MHD variables
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DiffusionErrors_()
//! \brief Computes errors in diffusion solution by calling initialization function
//! again to compute initial condictions, and subtracting current solution from ICs, and
//! outputs errors to file.

void DiffusionErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->Diffusion(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn GaussianProfileBCs
//! \brief Sets boundary conditions to the time-dependent analytic Gaussian profile on
//! all (user) boundary faces of the computational domain.

void GaussianProfileBCs(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;

  // The analytic-profile user BCs are only used by the hydro (conduction/viscosity)
  // tests.  Resistivity (MHD) runs use outflow BCs, since the field is negligible at the
  // (wide) domain boundaries, so nothing to do here when hydro is absent.
  auto *phydro = pm->pmb_pack->phydro;
  if (phydro == nullptr) return;
  EOS_Data &eos = phydro->peos->eos_data;
  Real gamma = eos.gamma;
  auto &u0 = phydro->u0;

  // capture variables for the kernel
  auto dv = diffvars;
  Real time = pm->time;
  Real coef = 0.0;
  if (dv.conduction_test && phydro->pcond != nullptr) {
    coef = (gamma-1.0)*phydro->pcond->alpha_iso;
  }
  if (dv.viscosity_test && phydro->pvisc != nullptr) coef = phydro->pvisc->nu_iso;

  // X1 boundaries
  par_for("diffbc_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real cons[5];

    // inner x1
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,i) = cons[IDN];
      u0(m,IM1,k,j,i) = cons[IM1];
      u0(m,IM2,k,j,i) = cons[IM2];
      u0(m,IM3,k,j,i) = cons[IM3];
      u0(m,IEN,k,j,i) = cons[IEN];
    }
    // outer x1
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      Real x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,(ie+i+1)) = cons[IDN];
      u0(m,IM1,k,j,(ie+i+1)) = cons[IM1];
      u0(m,IM2,k,j,(ie+i+1)) = cons[IM2];
      u0(m,IM3,k,j,(ie+i+1)) = cons[IM3];
      u0(m,IEN,k,j,(ie+i+1)) = cons[IEN];
    }
  });
  if (pm->one_d) return;

  // X2 boundaries
  par_for("diffbc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real cons[5];

    // inner x2
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,i) = cons[IDN];
      u0(m,IM1,k,j,i) = cons[IM1];
      u0(m,IM2,k,j,i) = cons[IM2];
      u0(m,IM3,k,j,i) = cons[IM3];
      u0(m,IEN,k,j,i) = cons[IEN];
    }
    // outer x2
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      Real x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,(je+j+1),i) = cons[IDN];
      u0(m,IM1,k,(je+j+1),i) = cons[IM1];
      u0(m,IM2,k,(je+j+1),i) = cons[IM2];
      u0(m,IM3,k,(je+j+1),i) = cons[IM3];
      u0(m,IEN,k,(je+j+1),i) = cons[IEN];
    }
  });
  if (pm->two_d) return;

  // X3 boundaries
  par_for("diffbc_x3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real cons[5];

    // inner x3
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,i) = cons[IDN];
      u0(m,IM1,k,j,i) = cons[IM1];
      u0(m,IM2,k,j,i) = cons[IM2];
      u0(m,IM3,k,j,i) = cons[IM3];
      u0(m,IEN,k,j,i) = cons[IEN];
    }
    // outer x3
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      Real x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,(ke+k+1),j,i) = cons[IDN];
      u0(m,IM1,(ke+k+1),j,i) = cons[IM1];
      u0(m,IM2,(ke+k+1),j,i) = cons[IM2];
      u0(m,IM3,(ke+k+1),j,i) = cons[IM3];
      u0(m,IEN,(ke+k+1),j,i) = cons[IEN];
    }
  });
  return;
}
