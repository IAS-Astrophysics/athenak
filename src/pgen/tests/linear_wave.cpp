//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_wave.c
//! \brief Linear wave problem generator for 1D/2D/3D problems. Initializes both hydro and
//! MHD problems for non-relativistic and SR/GR relativistic flows, and for GRMHD in
//! dynamical spacetimes (dynGR)..
//!
//! Direction of the wavevector is set to be along the x? axis by using the
//! along_x? input flags, else it is automatically set along the grid diagonal in 2D/3D
//! See comments in Athena4.2 linear wave problem generator for more details.
//!
//! Errors in solution after an integer number of wave periods are automatically output
//! at end of calculation.

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \struct LinWaveVariables
//! \brief container for variables shared with vector potential and perturbation functions

struct LinWaveVariables {
  Real d0, p0, vx_0, vy_0, vz_0, bx_0, by_0, bz_0, dby, dbz, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real wgas, cs_sq, gamma_adi;
  int wave_flag;
};

// function to compute errors in solution at end of run
void LinearWaveErrors(ParameterInput *pin, Mesh *pm);

// functions to compute eigenvectors of linearized eqns in PRIMITIVE variables
void HydroEigensystemPrim(const Real d, const Real v1, const Real v2, const Real v3,
                          const Real p, const EOS_Data &eos,
                          Real eigenvalues[5], Real right_eigenmatrix[5][5]);
void MHDEigensystemPrim(const Real d, const Real v1, const Real v2, const Real v3,
                        const Real p, const Real b1, const Real b2, const Real b3,
                        const Real x, const Real y, const EOS_Data &eos,
                        Real eigenvalues[7], Real right_eigenmatrix[7][7]);
// functions to compute linear wave perturbations in relativistic hydro and MHD
void RelHydroPerturbations(LinWaveVariables lwv, Real u[4],
                       Real &lambda, Real &delta_rho, Real &delta_pgas, Real delta_v[4]);
void RelMHDPerturbations(LinWaveVariables lwv, Real u[4], Real b[4],
     Real &lambda, Real &delta_rho, Real &delta_pgas, Real delta_v[4], Real delta_b[4]);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//! \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//! Az are functions of x and y alone.

KOKKOS_INLINE_FUNCTION
Real A1(const Real x1, const Real x2, const Real x3, const LinWaveVariables lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Ay =  lw.bz_0*x - (lw.dbz/lw.k_par)*std::cos(lw.k_par*(x));
  Real Az = -lw.by_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.bx_0*y;

  return -Ay*lw.sin_a3 - Az*lw.sin_a2*lw.cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//! \brief A2: 2-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A2(const Real x1, const Real x2, const Real x3, const LinWaveVariables lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Ay =  lw.bz_0*x - (lw.dbz/lw.k_par)*std::cos(lw.k_par*(x));
  Real Az = -lw.by_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.bx_0*y;

  return Ay*lw.cos_a3 - Az*lw.sin_a2*lw.sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//! \brief A3: 3-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A3(const Real x1, const Real x2, const Real x3, const LinWaveVariables lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Az = -lw.by_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.bx_0*y;

  return Az*lw.cos_a2;
}

//----------------------------------------------------------------------------------------
// Function for finding root of monic quadratic equation
// Inputs:
//   a1: linear coefficient
//   a0: constant coefficient
//   greater_root: flag indicating that larger root is to be returned
//     "larger" does not mean absolute value
// Outputs:
//   returned value: desired root
// Notes:
//   solves x^2 + a_1 x + a_0 = 0 for x
//   returns abscissa of vertex if there are no real roots
//   follows advice in Numerical Recipes, 3rd ed. (5.6) for avoiding large cancellations

Real QuadraticRoot(Real a1, Real a0, bool greater_root) {
  if (a1*a1 < 4.0*a0) {  // no real roots
    return -a1/2.0;
  }
  if (greater_root) {
    if (a1 >= 0.0) {
      return -2.0*a0 / (a1 + std::sqrt(a1*a1 - 4.0*a0));
    } else {
      return (-a1 + std::sqrt(a1*a1 - 4.0*a0)) / 2.0;
    }
  } else {
    if (a1 >= 0.0) {
      return (-a1 - std::sqrt(a1*a1 - 4.0*a0)) / 2.0;
    } else {
      return -2.0*a0 / (a1 - std::sqrt(a1*a1 - 4.0*a0));
    }
  }
}

//----------------------------------------------------------------------------------------
// Function for finding real root of monic cubic equation
// Inputs:
//   a2: quadratic coefficient
//   a1: linear coefficient
//   a0: constant coefficient
// Outputs:
//   returned value: a real root
// Notes:
//   solves x^3 + a_2 x^2 + a_1 x + a_0 = 0 for x
//   references Numerical Recipes, 3rd ed. (NR)

Real CubicRootReal(Real a2, Real a1, Real a0) {
  Real q = (a2*a2 - 3.0*a1) / 9.0;                       // (NR 5.6.10)
  Real r = (2.0*a2*a2*a2 - 9.0*a1*a2 + 27.0*a0) / 54.0;  // (NR 5.6.10)
  if (r*r - q*q*q < 0.0) {
    Real theta = std::acos(r/std::sqrt(q*q*q));                 // (NR 5.6.11)
    return -2.0 * std::sqrt(q) * std::cos(theta/3.0) - a2/3.0;  // (NR 5.6.12)
  } else {
    Real a = -copysign(1.0, r)
             * std::cbrt(std::abs(r) + std::sqrt(r*r - q*q*q));  // (NR 5.6.15)
    Real b = (a != 0.0) ? q/a : 0.0;                   // (NR 5.6.16)
    return a + b - a2/3.0;
  }
}

//----------------------------------------------------------------------------------------
// Function for finding extremal real roots of monic quartic equation
// Inputs:
//   a3: cubic coefficient
//   a2: quadratic coefficient
//   a1: linear coefficient
//   a0: constant coefficient
// Outputs:
//   px1: value set to least real root
//   px2: value set to second least real root
//   px3: value set to second greatest real root
//   px4: value set to greatest real root
// Notes:
//   solves x^4 + a3 x^3 + a2 x^2 + a1 x + a0 = 0 for x
//   uses following procedure:
//     1) eliminate cubic term y^4 + b2 y^2 + b1 y + b0
//     2) construct resolvent cubic z^3 + c2 z^2 + c1 z + c0
//     3) find real root z0 of cubic
//     4) construct quadratics:
//          y^2 + d1 y + d0
//          y^2 + e1 y + e0
//     5) find roots of quadratics

void QuarticRoots(Real a3, Real a2, Real a1, Real a0, Real *px1, Real *px2,
                  Real *px3, Real *px4) {
  // Step 1: Find reduced quartic coefficients
  Real b2 = a2 - 3.0/8.0*SQR(a3);
  Real b1 = a1 - 1.0/2.0*a2*a3 + 1.0/8.0*a3*SQR(a3);
  Real b0 = a0 - 1.0/4.0*a1*a3 + 1.0/16.0*a2*SQR(a3) - 3.0/256.0*SQR(SQR(a3));

  // Step 2: Find resolvent cubic coefficients
  Real c2 = -b2;
  Real c1 = -4.0*b0;
  Real c0 = 4.0*b0*b2 - SQR(b1);

  // Step 3: Solve cubic
  Real z0 = CubicRootReal(c2, c1, c0);

  // Step 4: Find quadratic coefficients
  Real d1 = (z0 - b2 > 0.0) ? std::sqrt(z0 - b2) : 0.0;
  Real e1 = -d1;
  Real d0, e0;
  if (b1 < 0) {
    d0 = z0/2.0 + std::sqrt(SQR(z0)/4.0 - b0);
    e0 = z0/2.0 - std::sqrt(SQR(z0)/4.0 - b0);
  } else {
    d0 = z0/2.0 - std::sqrt(SQR(z0)/4.0 - b0);
    e0 = z0/2.0 + std::sqrt(SQR(z0)/4.0 - b0);
  }

  // Step 5: Solve quadratics
  Real y1 = QuadraticRoot(d1, d0, false);
  Real y2 = QuadraticRoot(d1, d0, true);
  Real y3 = QuadraticRoot(e1, e0, false);
  Real y4 = QuadraticRoot(e1, e0, true);

  // Step 6: Set original quartic roots
  *px1 = std::min(y1, y3) - a3/4.0;
  Real mid_1 = std::max(y1, y3) - a3/4.0;
  *px4 = std::max(y2, y4) - a3/4.0;
  Real mid_2 = std::min(y2, y4) - a3/4.0;
  *px2 = std::min(mid_1, mid_2);
  *px3 = std::max(mid_1, mid_2);
  return;
}
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWave()
//! \brief Sets initial conditions for linear wave tests

void ProblemGenerator::LinearWave(ParameterInput *pin, const bool restart) {
  // set linear wave errors function
  pgen_final_func = LinearWaveErrors;
  if (restart) return;

  // Read and/or calculate direction of wavevector
  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Can only specify one of along_x1/2/3 to be true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((along_x2 || along_x3) && pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x2 or x3 axis in 1D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (along_x3 && pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x3 axis in 2D" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // start with wavevector along x1 axis
  LinWaveVariables lwv;
  lwv.cos_a3 = 1.0;
  lwv.sin_a3 = 0.0;
  lwv.cos_a2 = 1.0;
  lwv.sin_a2 = 0.0;
  if (pmy_mesh_->multi_d && !(along_x1)) {
    Real ang_3 = std::atan(x1size/x2size);
    lwv.sin_a3 = std::sin(ang_3);
    lwv.cos_a3 = std::cos(ang_3);
  }
  if (pmy_mesh_->three_d && !(along_x1)) {
    Real ang_2 = std::atan(0.5*(x1size*lwv.cos_a3 + x2size*lwv.sin_a3)/x3size);
    lwv.sin_a2 = std::sin(ang_2);
    lwv.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 1.0;
    lwv.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 0.0;
    lwv.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lx = std::numeric_limits<float>::max();
  if (lwv.cos_a2*lwv.cos_a3 > 0.0) {
    lx = std::min(lx, x1size*lwv.cos_a2*lwv.cos_a3);
  }
  if (lwv.cos_a2*lwv.sin_a3 > 0.0) {
    lx = std::min(lx, x2size*lwv.cos_a2*lwv.sin_a3);
  }
  if (lwv.sin_a2 > 0.0) lx = std::min(lx, x3size*lwv.sin_a2);

  // Initialize k_parallel
  lwv.k_par = 2.0*(M_PI)/lx;

  // read global parameters and background state
  // vx_0 is parallel to wavevector.
  // bx_0 is parallel to wavevector, by_0/bz_0 are perpendicular
  lwv.wave_flag = pin->GetInteger("problem", "wave_flag");
  Real amp      = pin->GetReal("problem", "amp");
  lwv.d0   = pin->GetReal("problem", "dens");
  lwv.p0   = pin->GetReal("problem", "pgas");
  lwv.vx_0 = pin->GetOrAddReal("problem", "vx0", 0.0);
  lwv.vy_0 = pin->GetOrAddReal("problem", "vy0", 0.0);
  lwv.vz_0 = pin->GetOrAddReal("problem", "vz0", 0.0);
  lwv.bx_0 = pin->GetOrAddReal("problem", "bx0", 0.0);
  lwv.by_0 = pin->GetOrAddReal("problem", "by0", 0.0);
  lwv.bz_0 = pin->GetOrAddReal("problem", "bz0", 0.0);
// Legacy values that used to be hardwired into code
//  lwv.b1_0 = 1.0;
//  lwv.b2_0 = std::sqrt(2.0);
//  lwv.b3_0 = 0.5;

  // capture variables for kernels
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  bool relativistic = false;
  if (pmbp->pcoord->is_general_relativistic ||
      pmbp->pcoord->is_special_relativistic ||
      pmbp->pcoord->is_dynamical_relativistic) {
    relativistic = true;
  }
  bool &dynamical_relativistic = pmbp->pcoord->is_dynamical_relativistic;

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    lwv.gamma_adi = eos.gamma;
    Real gm1 = eos.gamma - 1.0;
    Real gamma_adi_red = eos.gamma / (eos.gamma - 1.0);

    // Calculate linear wave perturbations in hydro
    Real rem[5][5], ev[5];
    Real lambda, delta_rho, delta_pgas, delta_v[4];
    if (relativistic) {
      // Calculate background 4-vectors
      Real u[4];
      Real v_sq = SQR(lwv.vx_0) + SQR(lwv.vy_0) + SQR(lwv.vz_0);
      u[0] = 1.0 / std::sqrt(1.0 - v_sq);
      u[1] = u[0]*lwv.vx_0;
      u[2] = u[0]*lwv.vy_0;
      u[3] = u[0]*lwv.vz_0;
      // compute some useful quantities, store in LinWaveVariables container
      lwv.wgas = lwv.d0 + gamma_adi_red * lwv.p0;
      lwv.cs_sq = eos.gamma * lwv.p0 / lwv.wgas;
      // calculate perturbations in primitives in relativistic dynamics
      RelHydroPerturbations(lwv,u,lambda,delta_rho,delta_pgas,delta_v);
    } else {
      // calculate eigenvectors in primitive variables in non-relativistic dynamics
      HydroEigensystemPrim(lwv.d0, lwv.vx_0, 0.0, 0.0, lwv.p0, eos, ev, rem);
    }

    // set new time limit in ParameterInput (to be read by Driver constructor) based on
    // wave speed of selected mode.
    // input tlim is interpreted as number of wave periods for evolution
    if (set_initial_conditions) {
      Real tlim = pin->GetReal("time", "tlim");
      if (relativistic) {
        pin->SetReal("time", "tlim", tlim*(std::abs(lx/lambda)));
      } else {
        pin->SetReal("time", "tlim", tlim*(std::abs(lx/ev[lwv.wave_flag])));
      }
    }


    // Calculate cell-centered primitive variables
    auto &w0 = pmbp->phydro->w0;
    par_for("pgen_linwave1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

      Real x = lwv.cos_a2*(x1v*lwv.cos_a3 + x2v*lwv.sin_a3) + x3v*lwv.sin_a2;
      Real sn = std::sin(lwv.k_par*x);
      Real rho, vx, vy, vz, egas;

      if (relativistic) {
        rho = lwv.d0 + amp*sn * delta_rho;
        Real vx_mink = lwv.vx_0 + amp*sn * delta_v[1];
        Real vy_mink = lwv.vy_0 + amp*sn * delta_v[2];
        Real vz_mink = lwv.vz_0 + amp*sn * delta_v[3];
        Real lor = 1.0 / std::sqrt(1.0 - SQR(vx_mink) - SQR(vy_mink) - SQR(vz_mink));
        vx = lor * vx_mink;
        vy = lor * vy_mink;
        vz = lor * vz_mink;
        if (dynamical_relativistic) {
          egas = lwv.p0 + amp*sn * delta_pgas;        // set pressure
        } else {
          egas = (lwv.p0 + amp*sn * delta_pgas)/gm1;  // set internal energy
        }
      } else {
        rho  = lwv.d0   + amp*sn*rem[0][lwv.wave_flag];
        vx   = lwv.vx_0 + amp*sn*rem[1][lwv.wave_flag];
        vy   = lwv.vy_0 + amp*sn*rem[2][lwv.wave_flag];
        vz   = lwv.vz_0 + amp*sn*rem[3][lwv.wave_flag];
        egas = (lwv.p0 + amp*sn*rem[4][lwv.wave_flag])/gm1;
      }
      // set cell-centered conserved variables
      w0(m,IDN,k,j,i)=rho;
      w0(m,IVX,k,j,i)=vx*lwv.cos_a2*lwv.cos_a3 -vy*lwv.sin_a3 -vz*lwv.sin_a2*lwv.cos_a3;
      w0(m,IVY,k,j,i)=vx*lwv.cos_a2*lwv.sin_a3 +vy*lwv.cos_a3 -vz*lwv.sin_a2*lwv.sin_a3;
      w0(m,IVZ,k,j,i)=vx*lwv.sin_a2                           +vz*lwv.cos_a2;
      if (eos.is_ideal) {
        w0(m,IEN,k,j,i) = egas;
      }
    });

    // Convert primitive to conserved
    if (pmbp->padm != nullptr) {
      // If we're using the ADM variables, then we've got dynamic GR enabled.
      // Because we need the metric, we can't initialize the conserved variables
      // until we've filled out the ADM variables.
      pmbp->padm->SetADMVariables(pmbp);
      if (set_initial_conditions) {
        pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
      } else {
        // extremely ugly pointer cast to access PrimToCons function used in dynGR EOS
        auto& eos = static_cast<dyngr::DynGRMHDPS
                     <Primitive::IdealGas, Primitive::ResetFloor>*>(pmbp->pdyngr)->eos;
        eos.PrimToCons(w0, pmbp->pmhd->bcc0, pmbp->pmhd->u1, is, ie, js, je, ks, ke);
      }
    } else {
      // "regular" GRHydro in stationary spacetimes
      if (set_initial_conditions) {
        pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u0, is, ie, js, je, ks, ke);
      } else {
        pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u1, is, ie, js, je, ks, ke);
      }
    }
  }  // End initialization Hydro variables

  // initialize MHD variables ------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    lwv.gamma_adi = eos.gamma;
    Real gm1 = eos.gamma - 1.0;
    Real gamma_adi_red = eos.gamma / (eos.gamma - 1.0);
    int nmb = pmbp->nmb_thispack;
    int nmhd_ = pmbp->pmhd->nmhd;

    // Calculate linear wave perturbations in MHD
    Real rem[7][7], ev[7];
    Real lambda, delta_rho, delta_pgas,u[4],delta_u[4],delta_b[4];
    if (relativistic) {
      // Calculate background 4-vectors
      Real v_sq = SQR(lwv.vx_0) + SQR(lwv.vy_0) + SQR(lwv.vz_0);
      Real b[4];              // contravariant quantities
      u[0] = 1.0 / std::sqrt(1.0 - v_sq);
      u[1] = u[0]*lwv.vx_0;
      u[2] = u[0]*lwv.vy_0;
      u[3] = u[0]*lwv.vz_0;
      b[0] = lwv.bx_0*u[1] + lwv.by_0*u[2] + lwv.bz_0*u[3];
      b[1] = 1.0/u[0] * (lwv.bx_0 + b[0]*u[1]);
      b[2] = 1.0/u[0] * (lwv.by_0 + b[0]*u[2]);
      b[3] = 1.0/u[0] * (lwv.bz_0 + b[0]*u[3]);
      // compute some useful quantities, store in LinWaveVariables container
      lwv.wgas = lwv.d0 + gamma_adi_red * lwv.p0;
      lwv.cs_sq = eos.gamma * lwv.p0 / lwv.wgas;
      RelMHDPerturbations(lwv,u,b,lambda,delta_rho,delta_pgas,delta_u,delta_b);
      // Modify relativistic transverse magnetic fields for linear wave perturbations
      lwv.by_0 = b[2]*u[0] - b[0]*u[2];
      lwv.bz_0 = b[3]*u[0] - b[0]*u[3];
      lwv.dby = amp*((b[2]*delta_u[0] - b[0]*delta_u[2])
                   + (delta_b[2]*u[0] - delta_b[0]*u[2]));
      lwv.dbz = amp*((b[3]*delta_u[0] - b[0]*delta_u[3])
                   + (delta_b[3]*u[0] - delta_b[0]*u[3]));
    } else {
      Real xfact = 0.0;
      Real yfact = 1.0;
      MHDEigensystemPrim(lwv.d0, lwv.vx_0, 0.0, 0.0, lwv.p0, lwv.bx_0, lwv.by_0, lwv.bz_0,
                         xfact, yfact, eos, ev, rem);
      // Set linear wave magnetic field perturbations
      lwv.dby = amp*rem[nmhd_  ][lwv.wave_flag];
      lwv.dbz = amp*rem[nmhd_+1][lwv.wave_flag];
    }

    // set new time limit in ParameterInput (to be read by Driver constructor) based on
    // wave speed of selected mode.
    // input tlim is interpreted as number of wave periods for evolution
    if (set_initial_conditions) {
      Real tlim = pin->GetReal("time", "tlim");
      if (relativistic) {
        pin->SetReal("time", "tlim", tlim*(std::abs(lx/lambda)));
      } else {
        pin->SetReal("time", "tlim", tlim*(std::abs(lx/ev[lwv.wave_flag])));
      }
    }

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 2;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 2;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    par_for("pgen_linwave2", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
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

      a1(m,k,j,i) = A1(x1v, x2f, x3f, lwv);
      a2(m,k,j,i) = A2(x1f, x2v, x3f, lwv);
      a3(m,k,j,i) = A3(x1f, x2f, x3v, lwv);

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
        a1(m,k,j,i) = 0.5*(A1(xl,x2f,x3f,lwv) + A1(xr,x2f,x3f,lwv));
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
        a2(m,k,j,i) = 0.5*(A2(x1f,xl,x3f,lwv) + A2(x1f,xr,x3f,lwv));
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
        a3(m,k,j,i) = 0.5*(A3(x1f,x2f,xl,lwv) + A3(x1f,x2f,xr,lwv));
      }
    });

    // compute solution in b0/b1 registers depending on whether this is ICs
    auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;

    // now compute primitive quantities, as well as face- and cell-centered fields
    auto &w0 = pmbp->pmhd->w0;
    par_for("pgen_linwave3", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

      Real x = lwv.cos_a2*(x1v*lwv.cos_a3 + x2v*lwv.sin_a3) + x3v*lwv.sin_a2;
      Real sn = std::sin(lwv.k_par*x);
      Real rho, vx, vy, vz, egas;
      if (relativistic) {
        rho = lwv.d0 + amp*sn * delta_rho;
        vx  = u[1] + amp*sn * delta_u[1];
        vy  = u[2] + amp*sn * delta_u[2];
        vz  = u[3] + amp*sn * delta_u[3];
        if (dynamical_relativistic) {
          egas = lwv.p0 + amp*sn * delta_pgas;        // set pressure
        } else {
          egas = (lwv.p0 + amp*sn * delta_pgas)/gm1;  // set internal energy
        }
      } else {
        rho  = lwv.d0   + amp*sn*rem[0][lwv.wave_flag];
        vx   = lwv.vx_0 + amp*sn*rem[1][lwv.wave_flag];
        vy   = lwv.vy_0 + amp*sn*rem[2][lwv.wave_flag];
        vz   = lwv.vz_0 + amp*sn*rem[3][lwv.wave_flag];
        egas = (lwv.p0 + amp*sn*rem[4][lwv.wave_flag])/gm1;
      }
      // compute cell-centered primitive variables
      w0(m,IDN,k,j,i)=rho;
      w0(m,IVX,k,j,i)=vx*lwv.cos_a2*lwv.cos_a3 - vy*lwv.sin_a3 -vz*lwv.sin_a2*lwv.cos_a3;
      w0(m,IVY,k,j,i)=vx*lwv.cos_a2*lwv.sin_a3 + vy*lwv.cos_a3 -vz*lwv.sin_a2*lwv.sin_a3;
      w0(m,IVZ,k,j,i)=vx*lwv.sin_a2                            +vz*lwv.cos_a2;
      if (eos.is_ideal) {
        w0(m,IEN,k,j,i) = egas;
      }

      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b1.x1f(m,k,j,i) = (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                        (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3;
      b1.x2f(m,k,j,i) = (a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                        (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
      b1.x3f(m,k,j,i) = (a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                        (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2;

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b1.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                            (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3;
      }
      if (j==je) {
        b1.x2f(m,k,j+1,i) = (a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                            (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
      }
      if (k==ke) {
        b1.x3f(m,k+1,j,i) = (a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                            (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2;
      }
    });

    // Compute cell-centered fields
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc0(m, IBX, k, j, i) = 0.5*(b1.x1f(m, k, j, i) + b1.x1f(m, k, j, i+1));
      bcc0(m, IBY, k, j, i) = 0.5*(b1.x2f(m, k, j, i) + b1.x2f(m, k, j+1, i));
      bcc0(m, IBZ, k, j, i) = 0.5*(b1.x3f(m, k, j, i) + b1.x3f(m, k+1, j, i));
    });

    if (pmbp->padm != nullptr) {
      // If we're using the ADM variables, then we've got dynamic GR enabled.
      // Because we need the metric, we can't initialize the conserved variables
      // until we've filled out the ADM variables.
      pmbp->padm->SetADMVariables(pmbp);
      if (set_initial_conditions) {
        pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
      } else {
        // extremely ugly pointer cast to access PrimToCons function used in dynGR EOS
        auto& eos = static_cast<dyngr::DynGRMHDPS
                     <Primitive::IdealGas, Primitive::ResetFloor>*>(pmbp->pdyngr)->eos;
        eos.PrimToCons(w0, bcc0, pmbp->pmhd->u1, is, ie, js, je, ks, ke);
      }
    } else {
      // "regular" GRMHD in stationary spacetimes
      if (set_initial_conditions) {
        pmbp->pmhd->peos->PrimToCons(w0, bcc0, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
      } else {
        pmbp->pmhd->peos->PrimToCons(w0, bcc0, pmbp->pmhd->u1, is, ie, js, je, ks, ke);
      }
    }
  }  // End initialization MHD variables

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroEigensystemPrim()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal hydrodynamics
//! for the linearized system in the PRIMITIVE variables, i.e. W,t = AW,x, where
//! W=(d,vx,vy,vz,[P]).
//! Taken from esys_prim functions in Athena4.2


void HydroEigensystemPrim(const Real d, const Real v1, const Real v2, const Real v3,
                          const Real p, const EOS_Data &eos,
                          Real eigenvalues[5], Real right_eigenmatrix[5][5]) {
  //--- Ideal Gas Hydrodynamics ---
  if (eos.is_ideal) {
    Real vsq = v1*v1 + v2*v2 + v3*v3;
    Real h = (p/(eos.gamma - 1.0) + 0.5*d*vsq + p)/d;
    Real a = std::sqrt(eos.gamma*p/d);

    // Compute eigenvalues (eq. B2)
    eigenvalues[0] = v1 - a;
    eigenvalues[1] = v1;
    eigenvalues[2] = v1;
    eigenvalues[3] = v1;
    eigenvalues[4] = v1 + a;

    // Right-eigenvectors, stored as COLUMNS (eq. B3)
    right_eigenmatrix[0][0] = 1.0;
    right_eigenmatrix[1][0] = -a/d;
    right_eigenmatrix[2][0] = 0.0;
    right_eigenmatrix[3][0] = 0.0;
    right_eigenmatrix[4][0] = a*a;

    right_eigenmatrix[0][1] = 1.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = 0.0;
    right_eigenmatrix[3][1] = 0.0;
    right_eigenmatrix[4][1] = 0.0;

    right_eigenmatrix[0][2] = 0.0;
    right_eigenmatrix[1][2] = 0.0;
    right_eigenmatrix[2][2] = 1.0;
    right_eigenmatrix[3][2] = 0.0;
    right_eigenmatrix[4][2] = 0.0;

    right_eigenmatrix[0][3] = 0.0;
    right_eigenmatrix[1][3] = 0.0;
    right_eigenmatrix[2][3] = 0.0;
    right_eigenmatrix[3][3] = 1.0;
    right_eigenmatrix[4][3] = 0.0;

    right_eigenmatrix[0][4] = 1.0;
    right_eigenmatrix[1][4] = a/d;
    right_eigenmatrix[2][4] = 0.0;
    right_eigenmatrix[3][4] = 0.0;
    right_eigenmatrix[4][4] = a*a;

  //--- Isothermal Hydrodynamics ---
  } else {
    // Compute eigenvalues (eq. B6)
    eigenvalues[0] = v1 - eos.iso_cs;
    eigenvalues[1] = v1;
    eigenvalues[2] = v1;
    eigenvalues[3] = v1 + eos.iso_cs;

    // Right-eigenvectors, stored as COLUMNS (eq. B3)
    right_eigenmatrix[0][0] = 1.0;
    right_eigenmatrix[1][0] = - eos.iso_cs/d;
    right_eigenmatrix[2][0] = 0.0;
    right_eigenmatrix[3][0] = 0.0;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = 1.0;
    right_eigenmatrix[3][1] = 0.0;

    right_eigenmatrix[0][2] = 0.0;
    right_eigenmatrix[1][2] = 0.0;
    right_eigenmatrix[2][2] = 0.0;
    right_eigenmatrix[3][2] = 1.0;

    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[1][3] = eos.iso_cs/d;
    right_eigenmatrix[2][3] = 0.0;
    right_eigenmatrix[3][3] = 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHDEigensystemPrim()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal MHD
//! for the linearized system in the PRIMITIVE variables, i.e. W,t = AW,x, where
//! W=(d,vx,vy,vz,[P],By,Bz).
//! Taken from esys_prim functions in Athena4.2

void MHDEigensystemPrim(const Real d, const Real v1, const Real v2, const Real v3,
                        const Real p, const Real b1, const Real b2, const Real b3,
                        const Real x, const Real y, const EOS_Data &eos,
                        Real eigenvalues[7], Real right_eigenmatrix[7][7]) {
  // common factors for both ideal gas and isothermal eigenvectors
  Real btsq = b2*b2 + b3*b3;
  Real bt = std::sqrt(btsq);
  Real asq = (eos.gamma*p/d);
  // beta's (eqs. A17, B28, B40)
  Real bet2,bet3;
  if (bt == 0.0) {
    bet2 = 1.0;
    bet3 = 0.0;
  } else {
    bet2 = b2/bt;
    bet3 = b3/bt;
  }

  //--- Ideal Gas MHD ---
  if (eos.is_ideal) {
    Real vsq = v1*v1 + v2*v2 + v3*v3;
    Real gm1 = eos.gamma - 1.0;
    Real h = (p/gm1 + 0.5*d*vsq + p + b1*b1 + btsq)/d;
    Real bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
    Real vaxsq = b1*b1/d;
    Real hp = h - (vaxsq + btsq/d);

    // Compute fast- and slow-magnetosonic speeds (eq. A10)
    Real ct2 = bt_starsq/d;
    Real tsum = vaxsq + ct2 + asq;
    Real tdif = vaxsq + ct2 - asq;
    Real cf2_cs2 = std::sqrt(tdif*tdif + 4.0*asq*ct2);

    Real cfsq = 0.5*(tsum + cf2_cs2);
    Real cf = std::sqrt(cfsq);

    Real cssq = asq*vaxsq/cfsq;
    Real cs = std::sqrt(cssq);

    // Compute beta(s) (eqs. A17)
    Real bt_star = std::sqrt(bt_starsq);
    Real bet2_star = bet2/std::sqrt(gm1 - (gm1-1.0)*y);
    Real bet3_star = bet3/std::sqrt(gm1 - (gm1-1.0)*y);
    Real bet_starsq = bet2_star*bet2_star + bet3_star*bet3_star;
    Real vbet = v2*bet2_star + v3*bet3_star;

    // Compute alpha(s) (eq. A16)
    Real alpha_f,alpha_s;
    if ((cfsq - cssq) == 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else if ( (asq - cssq) <= 0.0) {
      alpha_f = 0.0;
      alpha_s = 1.0;
    } else if ( (cfsq - asq) <= 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else {
      alpha_f = std::sqrt((asq - cssq)/(cfsq - cssq));
      alpha_s = std::sqrt((cfsq - asq)/(cfsq - cssq));
    }

    // Compute Q(s) and A(s) (eq. A14-15), etc.
    Real sqrtd = std::sqrt(d);
    Real s = SIGN(b1);
    Real a = std::sqrt(asq);
    Real qf = cf*alpha_f*s;
    Real qs = cs*alpha_s*s;
    Real af = a*alpha_f*sqrtd;
    Real as = a*alpha_s*sqrtd;

    // Compute eigenvalues (eq. A9)
    Real vax = std::sqrt(vaxsq);
    eigenvalues[0] = v1 - cf;
    eigenvalues[1] = v1 - vax;
    eigenvalues[2] = v1 - cs;
    eigenvalues[3] = v1;
    eigenvalues[4] = v1 + cs;
    eigenvalues[5] = v1 + vax;
    eigenvalues[6] = v1 + cf;

    // Right-eigenvectors, stored as COLUMNS (eq. A12) */
    right_eigenmatrix[0][0] = d*alpha_f;
    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[0][2] = d*alpha_s;
    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[0][4] = d*alpha_s;
    right_eigenmatrix[0][5] = 0.0;
    right_eigenmatrix[0][6] = d*alpha_f;

    right_eigenmatrix[1][0] = -cf*alpha_f;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[1][2] = -cs*alpha_s;
    right_eigenmatrix[1][3] = 0.0;
    right_eigenmatrix[1][4] = cs*alpha_s;
    right_eigenmatrix[1][5] = 0.0;
    right_eigenmatrix[1][6] = cf*alpha_f;

    Real qa = alpha_f*v2;
    Real qb = alpha_s*v2;
    Real qc = qs*bet2_star;
    Real qd = qf*bet2_star;
    right_eigenmatrix[2][0] = qs*bet2;
    right_eigenmatrix[2][1] = -bet3;
    right_eigenmatrix[2][2] = -qf*bet2;
    right_eigenmatrix[2][3] = 0.0;
    right_eigenmatrix[2][4] = qf*bet2;
    right_eigenmatrix[2][5] = bet3;
    right_eigenmatrix[2][6] = -qs*bet2;

    qa = alpha_f*v3;
    qb = alpha_s*v3;
    qc = qs*bet3_star;
    qd = qf*bet3_star;
    right_eigenmatrix[3][0] = qs*bet3;
    right_eigenmatrix[3][1] = bet2;
    right_eigenmatrix[3][2] = -qf*bet3;
    right_eigenmatrix[3][3] = 0.0;
    right_eigenmatrix[3][4] = qf*bet3;
    right_eigenmatrix[3][5] = -bet2;
    right_eigenmatrix[3][6] = -qs*bet3;

    right_eigenmatrix[4][0] = d*asq*alpha_f;
    right_eigenmatrix[4][1] = 0.0;
    right_eigenmatrix[4][2] = d*asq*alpha_s;
    right_eigenmatrix[4][3] = 0.0;
    right_eigenmatrix[4][4] = d*asq*alpha_s;
    right_eigenmatrix[4][5] = 0.0;
    right_eigenmatrix[4][6] = d*asq*alpha_f;

    right_eigenmatrix[5][0] = as*bet2;
    right_eigenmatrix[5][1] = -bet3*s*sqrtd;
    right_eigenmatrix[5][2] = -af*bet2;
    right_eigenmatrix[5][3] = 0.0;
    right_eigenmatrix[5][4] = right_eigenmatrix[5][2];
    right_eigenmatrix[5][5] = right_eigenmatrix[5][1];
    right_eigenmatrix[5][6] = right_eigenmatrix[5][0];

    right_eigenmatrix[6][0] = as*bet3;
    right_eigenmatrix[6][1] = bet2*s*sqrtd;
    right_eigenmatrix[6][2] = -af*bet3;
    right_eigenmatrix[6][3] = 0.0;
    right_eigenmatrix[6][4] = right_eigenmatrix[6][2];
    right_eigenmatrix[6][5] = right_eigenmatrix[6][1];
    right_eigenmatrix[6][6] = right_eigenmatrix[6][0];
  //--- Isothermal MHD ---
  } else {
    Real bt_starsq = btsq*y;
    Real vaxsq = b1*b1/d;
    Real iso_cs2 = (eos.iso_cs*eos.iso_cs);

    // Compute fast- and slow-magnetosonic speeds (eq. A10)
    Real ct2 = bt_starsq/d;
    Real tsum = vaxsq + ct2 + iso_cs2;
    Real tdif = vaxsq + ct2 - iso_cs2;
    Real cf2_cs2 = std::sqrt(tdif*tdif + 4.0*iso_cs2*ct2);

    Real cfsq = 0.5*(tsum + cf2_cs2);
    Real cf = std::sqrt(cfsq);

    Real cssq = iso_cs2*vaxsq/cfsq;
    Real cs = std::sqrt(cssq);

    // Compute alpha's (eq. A16)
    Real alpha_f,alpha_s;
    if ((cfsq-cssq) == 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else if ((iso_cs2 - cssq) <= 0.0) {
      alpha_f = 0.0;
      alpha_s = 1.0;
    } else if ((cfsq - iso_cs2) <= 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else {
      alpha_f = std::sqrt((iso_cs2 - cssq)/(cfsq - cssq));
      alpha_s = std::sqrt((cfsq - iso_cs2)/(cfsq - cssq));
    }

    // Compute Q's (eq. A14-15), etc.
    Real sqrtd = std::sqrt(d);
    Real s = SIGN(b1);
    Real qf = cf*alpha_f*s;
    Real qs = cs*alpha_s*s;
    Real af = (eos.iso_cs)*alpha_f*sqrtd;
    Real as = (eos.iso_cs)*alpha_s*sqrtd;

    // Compute eigenvalues (eq. A21)
    Real vax  = std::sqrt(vaxsq);
    eigenvalues[0] = v1 - cf;
    eigenvalues[1] = v1 - vax;
    eigenvalues[2] = v1 - cs;
    eigenvalues[3] = v1 + cs;
    eigenvalues[4] = v1 + vax;
    eigenvalues[5] = v1 + cf;

    // Right-eigenvectors, stored as COLUMNS (eq. A12)
    right_eigenmatrix[0][0] = d*alpha_f;
    right_eigenmatrix[1][0] = -cf*alpha_f;
    right_eigenmatrix[2][0] = qs*bet2;
    right_eigenmatrix[3][0] = qs*bet3;
    right_eigenmatrix[4][0] = as*bet2;
    right_eigenmatrix[5][0] = as*bet3;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = -bet3;
    right_eigenmatrix[3][1] = bet2;
    right_eigenmatrix[4][1] = -bet3*s*sqrtd;
    right_eigenmatrix[5][1] = bet2*s*sqrtd;

    right_eigenmatrix[0][2] = d*alpha_s;
    right_eigenmatrix[1][2] = -cs*alpha_s;
    right_eigenmatrix[2][2] = -qf*bet2;
    right_eigenmatrix[3][2] = -qf*bet3;
    right_eigenmatrix[4][2] = -af*bet2;
    right_eigenmatrix[5][2] = -af*bet3;

    right_eigenmatrix[0][3] = d*alpha_s;
    right_eigenmatrix[1][3] = cs*alpha_s;
    right_eigenmatrix[2][3] = qf*bet2;
    right_eigenmatrix[3][3] = qf*bet3;
    right_eigenmatrix[4][3] = right_eigenmatrix[4][2];
    right_eigenmatrix[5][3] = right_eigenmatrix[5][2];

    right_eigenmatrix[0][4] = 0.0;
    right_eigenmatrix[1][4] = 0.0;
    right_eigenmatrix[2][4] = bet3;
    right_eigenmatrix[3][4] = -bet2;
    right_eigenmatrix[4][4] = right_eigenmatrix[4][1];
    right_eigenmatrix[5][4] = right_eigenmatrix[5][1];

    right_eigenmatrix[0][5] = d*alpha_f;
    right_eigenmatrix[1][5] = cf*alpha_f;
    right_eigenmatrix[2][5] = -qs*bet2;
    right_eigenmatrix[3][5] = -qs*bet3;
    right_eigenmatrix[4][5] = right_eigenmatrix[4][0];
    right_eigenmatrix[5][5] = right_eigenmatrix[5][0];
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RelHydroPerturbations()
//! Calculates perturbation amplitudes for relativistic hydrodynamics

void RelHydroPerturbations(LinWaveVariables lwv, Real u[4],
                      Real &lambda, Real &delta_rho, Real &delta_pgas, Real delta_v[4]) {
  Real delta_u[4];
  // Calculate perturbation in 4-velocity components (Q of FK)
  switch (lwv.wave_flag) {
    case 1:  // entropy 1/3
      lambda = lwv.vx_0;
      delta_rho = 1.0;
      delta_pgas = 0.0;
      delta_u[1] = delta_u[2] = delta_u[3] = 0.0;
      break;
    case 2:  // entropy 2/3
      lambda = lwv.vx_0;
      delta_rho = 0.0;
      delta_pgas = 0.0;
      delta_u[1] = lwv.vx_0 * lwv.vy_0 / (1.0 - SQR(lwv.vx_0));
      delta_u[2] = 1.0;
      delta_u[3] = 0.0;
      break;
    case 3:  // entropy 3/3
      lambda = lwv.vx_0;
      delta_rho = 0.0;
      delta_pgas = 0.0;
      delta_u[1] = lwv.vx_0 * lwv.vz_0 / (1.0 - SQR(lwv.vx_0));
      delta_u[2] = 0.0;
      delta_u[3] = 1.0;
      break;
    default:  // sound
      Real delta = SQR(u[0]) * (1.0-lwv.cs_sq) + lwv.cs_sq;
      Real v_minus_lambda_a = lwv.vx_0 * lwv.cs_sq;
      Real v_minus_lambda_b = std::sqrt(lwv.cs_sq *
                         (SQR(u[0]) * (1.0-lwv.cs_sq) * (1.0-SQR(lwv.vx_0)) + lwv.cs_sq));
      Real v_minus_lambda;
      if (lwv.wave_flag == 0) {  // leftgoing
        v_minus_lambda = (v_minus_lambda_a + v_minus_lambda_b) / delta;  // (FK A1)
      } else {  // rightgoing
        v_minus_lambda = (v_minus_lambda_a - v_minus_lambda_b) / delta;  // (FK A1)
      }
      lambda = lwv.vx_0 - v_minus_lambda;
      delta_rho = lwv.d0;
      delta_pgas = lwv.wgas * lwv.cs_sq;
      delta_u[1] = -lwv.cs_sq * u[1] - lwv.cs_sq / u[0] / v_minus_lambda;
      delta_u[2] = -lwv.cs_sq * u[2];
      delta_u[3] = -lwv.cs_sq * u[3];
  }

  // Calculate perturbation in 3-velocity components (P of FK)
  delta_v[1] = (1.0-SQR(lwv.vx_0)) * delta_u[1] -
                 lwv.vx_0*lwv.vy_0 * delta_u[2] - lwv.vx_0*lwv.vz_0 * delta_u[3];
  delta_v[2] = -lwv.vx_0*lwv.vy_0 * delta_u[1] +
                 (1.0-SQR(lwv.vy_0)) * delta_u[2] - lwv.vy_0*lwv.vz_0 * delta_u[3];
  delta_v[3] = -lwv.vx_0*lwv.vz_0 * delta_u[1] -
                  lwv.vy_0*lwv.vz_0 * delta_u[2] + (1.0-SQR(lwv.vz_0)) * delta_u[3];
  for (int i = 1; i < 4; ++i) {
    delta_v[i] /= u[0];
  }

  // Renormalize perturbation to unit L^2 norm
  Real perturbation_size = SQR(delta_rho) + SQR(delta_pgas);
  for (int i = 1; i < 4; ++i) {
    perturbation_size += SQR(delta_v[i]);
  }
  perturbation_size = std::sqrt(perturbation_size);
  delta_rho /= perturbation_size;
  delta_pgas /= perturbation_size;
  for (int i = 1; i < 4; ++i) {
    delta_v[i] /= perturbation_size;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RelMHDPerturbations()
//! Calculates perturbation amplitudes for relativistic MHD

void RelMHDPerturbations(LinWaveVariables lwv, Real u[4], Real b[4],
     Real &lambda, Real &delta_rho, Real &delta_pgas, Real delta_u[4], Real delta_b[4]) {
  Real b_sq = -SQR(b[0]) + SQR(b[1]) + SQR(b[2]) + SQR(b[3]);
  Real wtot = lwv.wgas + b_sq;
  Real cs = std::sqrt(lwv.cs_sq);
  switch (lwv.wave_flag) {
    case 3: {  // entropy (A 46)
      lambda = lwv.vx_0;
      delta_rho = 1.0;
      delta_pgas = 0.0;
      for (int mu = 0; mu < 4; ++mu) {
        delta_u[mu] = 0.0;
        delta_b[mu] = 0.0;
      }
      break;
    }
    case 1: case 5: {  // Alfven (A 65)
      // Calculate wavespeed
      Real lambda_ap = (b[1] + std::sqrt(wtot) * u[1])
                       / (b[0] + std::sqrt(wtot) * u[0]);            // (A 38)
      Real lambda_am = (b[1] - std::sqrt(wtot) * u[1])
                       / (b[0] - std::sqrt(wtot) * u[0]);            // (A 38)
      Real sign = 1.0;
      if (lambda_ap > lambda_am) {  // \lambda_{a,\pm} = \lambda_a^\pm
        if (lwv.wave_flag == 1) {  // leftgoing
          sign = -1.0;
        }
      } else {  // lambda_{a,\pm} = \lambda_a^\mp
        if (lwv.wave_flag == 5) {  // rightgoing
          sign = -1.0;
        }
      }
      if (sign > 0) {  // want \lambda_{a,+}
        lambda = lambda_ap;
      } else {  // want \lambda_{a,-} instead
        lambda = lambda_am;
      }

      // Prepare auxiliary quantities
      Real alpha_1[4], alpha_2[4];
      alpha_1[0] = u[3];                                              // (A 58)
      alpha_1[1] = lambda * u[3];                                     // (A 58)
      alpha_1[2] = 0.0;                                               // (A 58)
      alpha_1[3] = u[0] - lambda * u[1];                              // (A 58)
      alpha_2[0] = -u[2];                                             // (A 59)
      alpha_2[1] = -lambda * u[2];                                    // (A 59)
      alpha_2[2] = lambda * u[1] - u[0];                              // (A 59)
      alpha_2[3] = 0.0;                                               // (A 59)
      // A60 and A61
      Real g_1 = 1.0/u[0]*(lwv.by_0 + lambda*lwv.vy_0/(1.0-lambda*lwv.vx_0) * lwv.bx_0);
      Real g_2 = 1.0/u[0]*(lwv.bz_0 + lambda*lwv.vz_0/(1.0-lambda*lwv.vx_0) * lwv.bx_0);
      Real f_1, f_2;
      if (g_1 == 0.0 && g_2 == 0.0) {
        f_1 = f_2 = 1.0/sqrt(2.0);  // (A 67)
      } else {
        f_1 = g_1 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
        f_2 = g_2 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
      }

      // Set perturbation
      delta_rho = 0.0;
      delta_pgas = 0.0;
      for (int mu = 0; mu < 4; ++mu) {
        delta_u[mu] = f_1 * alpha_1[mu] + f_2 * alpha_2[mu];
        delta_b[mu] = -sign * std::sqrt(wtot) * delta_u[mu];
      }
      break;
    }
    default: {  // magnetosonic (A 71)
      // Calculate wavespeed
      Real factor_a = lwv.wgas * (1.0/lwv.cs_sq - 1.0);
      Real factor_b = -(lwv.wgas + b_sq/lwv.cs_sq);
      Real gamma_2 = SQR(u[0]);
      Real gamma_4 = SQR(gamma_2);
      Real coeff_4 = factor_a * gamma_4
                     - factor_b * gamma_2
                     - SQR(b[0]);
      Real coeff_3 = -factor_a * 4.0 * gamma_4 * lwv.vx_0
                     + factor_b * 2.0 * gamma_2 * lwv.vx_0
                     + 2.0 * b[0] * b[1];
      Real coeff_2 = factor_a * 6.0 * gamma_4 * SQR(lwv.vx_0)
                     + factor_b * gamma_2 * (1.0-SQR(lwv.vx_0))
                     + SQR(b[0]) - SQR(b[1]);
      Real coeff_1 = -factor_a * 4.0 * gamma_4 * lwv.vx_0*SQR(lwv.vx_0)
                     - factor_b * 2.0 * gamma_2 * lwv.vx_0
                     - 2.0 * b[0] * b[1];
      Real coeff_0 = factor_a * gamma_4 * SQR(SQR(lwv.vx_0))
                     + factor_b * gamma_2 * SQR(lwv.vx_0)
                     + SQR(b[1]);
      Real lambda_fl, lambda_sl, lambda_sr, lambda_fr;
      QuarticRoots(coeff_3/coeff_4, coeff_2/coeff_4, coeff_1/coeff_4, coeff_0/coeff_4,
                   &lambda_fl, &lambda_sl, &lambda_sr, &lambda_fr);
      Real lambda_other_ms;
      if (lwv.wave_flag == 0) {
        lambda = lambda_fl;
        lambda_other_ms = lambda_sl;
      }
      if (lwv.wave_flag == 2) {
        lambda = lambda_sl;
        lambda_other_ms = lambda_fl;
      }
      if (lwv.wave_flag == 4) {
        lambda = lambda_sr;
        lambda_other_ms = lambda_fr;
      }
      if (lwv.wave_flag == 6) {
        lambda = lambda_fr;
        lambda_other_ms = lambda_sr;
      }

      // Determine which sign to use
      Real lambda_ap = (b[1] + std::sqrt(wtot) * u[1])
                       / (b[0] + std::sqrt(wtot) * u[0]);            // (A 38)
      Real lambda_am = (b[1] - std::sqrt(wtot) * u[1])
                       / (b[0] - std::sqrt(wtot) * u[0]);            // (A 38)
      Real lambda_a = lambda_ap;
      Real sign = 1.0;
      if (lambda_ap > lambda_am) {  // \lambda_{a,\pm} = \lambda_a^\pm
        if (lwv.wave_flag < 3) {  // leftgoing
          lambda_a = lambda_am;
          sign = -1.0;
        }
      } else {  // lambda_{a,\pm} = \lambda_a^\mp
        if (lwv.wave_flag > 3) {  // rightgoing
          lambda_a = lambda_am;
          sign = -1.0;
        }
      }

      // Prepare auxiliary quantities
      Real a = u[0] * (lwv.vx_0 - lambda);                                       // (A 39)
      Real g = 1.0 - SQR(lambda);                                          // (A 41)
      Real b_over_a = -sign * std::sqrt(-factor_b - factor_a * SQR(a)/g);  // (A 68)
      Real alpha_1[4], alpha_2[4];
      alpha_1[0] = u[3];                                                   // (A 58)
      alpha_1[1] = lambda * u[3];                                          // (A 58)
      alpha_1[2] = 0.0;                                                    // (A 58)
      alpha_1[3] = u[0] - lambda * u[1];                                   // (A 58)
      alpha_2[0] = -u[2];                                                  // (A 59)
      alpha_2[1] = -lambda * u[2];                                         // (A 59)
      alpha_2[2] = lambda * u[1] - u[0];                                   // (A 59)
      alpha_2[3] = 0.0;                                                    // (A 59)
      Real alpha_11 = -SQR(alpha_1[0]);
      Real alpha_12 = -alpha_1[0] * alpha_2[0];
      Real alpha_22 = -SQR(alpha_2[0]);
      for (int i = 1; i < 4; ++i) {
        alpha_11 += SQR(alpha_1[i]);
        alpha_12 += alpha_1[i] * alpha_2[i];
        alpha_22 += SQR(alpha_2[i]);
      }
      // A60 and A61
      Real g_1 = 1.0/u[0]*(lwv.by_0 + lambda*lwv.vy_0 / (1.0-lambda*lwv.vx_0) * lwv.bx_0);
      Real g_2 = 1.0/u[0]*(lwv.bz_0 + lambda*lwv.vz_0 / (1.0-lambda*lwv.vx_0) * lwv.bx_0);
      Real c_1 = (g_1*alpha_12 + g_2*alpha_22)
                 / (alpha_11*alpha_22 - SQR(alpha_12))
                 * u[0] * (1.0-lambda*lwv.vx_0);                                 // (A 63)
      Real c_2 = -(g_1*alpha_11 + g_2*alpha_12)
                 / (alpha_11*alpha_22 - SQR(alpha_12))
                 * u[0] * (1.0-lambda*lwv.vx_0);                                 // (A 63)
      Real b_t[4];
      for (int mu = 0; mu < 4; ++mu) {
        b_t[mu] = c_1 * alpha_1[mu] + c_2 * alpha_2[mu];  // (A 62)
      }
      Real f_1, f_2;
      if (g_1 == 0.0 && g_2 == 0.0) {
        f_1 = f_2 = 1.0/sqrt(2.0);  // (A 67)
      } else {
        f_1 = g_1 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
        f_2 = g_2 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
      }
      Real phi_plus_a_u[4];
      for (int mu = 0; mu < 4; ++mu) {
        phi_plus_a_u[mu] = a * u[mu];
      }
      phi_plus_a_u[0] += lambda;
      phi_plus_a_u[1] += 1.0;

      // Set perturbation
      if (std::abs(lambda-lambda_a)                 // using closer magnetosonic wave...
          <= std::abs(lambda_other_ms-lambda_a)) {  // ...to the associated Alfven wave
        Real b_t_normalized[4];
        Real denom = std::sqrt((alpha_11*alpha_22 - SQR(alpha_12))
                               * (SQR(f_1)*alpha_11 +
                                  2.0*f_1*f_2*alpha_12 +
                                  SQR(f_2)*alpha_22));
        for (int mu = 0; mu < 4; ++mu) {
          b_t_normalized[mu] =
              ((f_1*alpha_12+f_2*alpha_22) * alpha_1[mu]
               - (f_1*alpha_11+f_2*alpha_12) * alpha_2[mu]) / denom;        // (A 75)
        }
        Real b_t_norm = -SQR(b_t[0]);
        for (int i = 1; i < 4; ++i) {
          b_t_norm += SQR(b_t[i]);
        }
        b_t_norm = std::sqrt(b_t_norm);
        denom = SQR(a) - (g+SQR(a)) * lwv.cs_sq;
        if (denom == 0.0) {
          delta_pgas = 0.0;
        } else {
          delta_pgas = -(g+SQR(a)) * lwv.cs_sq / denom * b_t_norm;              // (A 74)
        }
        delta_rho = lwv.d0 / (lwv.gamma_adi*lwv.p0) * delta_pgas;
        for (int mu = 0; mu < 4; ++mu) {
          delta_u[mu] =
              -a*delta_pgas / (lwv.wgas*lwv.cs_sq*(g+SQR(a))) * phi_plus_a_u[mu]
              - b_over_a / lwv.wgas * b_t_normalized[mu];                       // (A 72)
          delta_b[mu] = -b_over_a * delta_pgas/lwv.wgas * u[mu]
                        - (1.0+SQR(a)/g) * b_t_normalized[mu];              // (A 73)
        }
      } else {  // using more distant magnetosonic wave
        delta_pgas = -1.0;                                                  // (A 78)
        delta_rho = lwv.d0 / (lwv.gamma_adi*lwv.p0) * delta_pgas;
        Real b_t_reduced[4] = {0.0};                                        // (A 79)
        Real denom = lwv.wgas * SQR(a) - b_sq * g;
        if (denom != 0.0) {
          for (int mu = 0; mu < 4; ++mu) {
            b_t_reduced[mu] = b_t[mu] / denom;
          }
        }
        for (int mu = 0; mu < 4; ++mu) {
          delta_u[mu] = a / (lwv.wgas*lwv.cs_sq*(g+SQR(a))) * phi_plus_a_u[mu]
                        - b_over_a * g/lwv.wgas * b_t_reduced[mu];              // (A 76)
          delta_b[mu] = b_over_a / lwv.wgas * u[mu]
                        - (1.0+SQR(a)/g) * g * b_t_reduced[mu];             // (A 77)
        }
      }
    }
  }
  // Renormalize perturbation to unit L^2 norm
  Real perturbation_size = SQR(delta_rho) + SQR(delta_pgas);
  for (int mu = 0; mu < 4; ++mu) {
    perturbation_size += SQR(delta_u[mu]) + SQR(delta_b[mu]);
  }
  perturbation_size = std::sqrt(perturbation_size);
  delta_rho /= perturbation_size;
  delta_pgas /= perturbation_size;
  for (int mu = 0; mu < 4; ++mu) {
    delta_u[mu] /= perturbation_size;
    delta_b[mu] /= perturbation_size;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void LinearWaveErrors_()
//! \brief Computes errors in linear wave solution by calling initialization function
//! again to compute initial condictions, and then calling generic error output function
//! that subtracts current solution from ICs, and outputs errors to file. Problem must be
//! run for an integer number of wave periods.

void LinearWaveErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->LinearWave(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}
