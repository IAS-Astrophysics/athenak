//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file linear_wave.c
//! \brief Linear wave problem generator for 1D/2D/3D problems. Initializes both hydro and
//! MHD problems. Direction of the wavevector is set to be along the x? axis by using the
//! along_x? input flags, else it is automatically set along the grid diagonal in 2D/3D
//! This file also contains a function to compute L1 errors in solution, called in
//! Driver::Finalize().

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
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void LinearWaveErrors(ParameterInput *pin, Mesh *pm);

// function to compute eigenvectors of linear waves in hydrodynamics
void HydroEigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                      const Real p, const EOS_Data &eos,
                      Real eigenvalues[5], Real right_eigenmatrix[5][5]);
// function to compute eigenvectors of linear waves in mhd
void MHDEigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                    const Real p, const Real b1, const Real b2, const Real b3,
                    const Real x, const Real y, const EOS_Data &eos,
                    Real eigenvalues[7], Real right_eigenmatrix[7][7]);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

//----------------------------------------------------------------------------------------
//! \struct LinWaveVariables
//! \brief container for variables shared with vector potential and error functions

struct LinWaveVariables {
  Real d0, p0, v1_0, b1_0, b2_0, b3_0, dby, dbz, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
};

//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//! \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//! Az are functions of x and y alone.

KOKKOS_INLINE_FUNCTION
Real A1(const Real x1, const Real x2, const Real x3, const LinWaveVariables lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Ay =  lw.b3_0*x - (lw.dbz/lw.k_par)*std::cos(lw.k_par*(x));
  Real Az = -lw.b2_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.b1_0*y;

  return -Ay*lw.sin_a3 - Az*lw.sin_a2*lw.cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//! \brief A2: 2-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A2(const Real x1, const Real x2, const Real x3, const LinWaveVariables lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Ay =  lw.b3_0*x - (lw.dbz/lw.k_par)*std::cos(lw.k_par*(x));
  Real Az = -lw.b2_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.b1_0*y;

  return Ay*lw.cos_a3 - Az*lw.sin_a2*lw.sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//! \brief A3: 3-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A3(const Real x1, const Real x2, const Real x3, const LinWaveVariables lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Az = -lw.b2_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.b1_0*y;

  return Az*lw.cos_a2;
}
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWave_()
//! \brief Sets initial conditions for linear wave tests

void ProblemGenerator::LinearWave(ParameterInput *pin, const bool restart) {
  // set linear wave errors function
  pgen_final_func = LinearWaveErrors;
  if (restart) return;

  // read global parameters
  int wave_flag = pin->GetInteger("problem", "wave_flag");
  Real amp = pin->GetReal("problem", "amp");
  Real vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
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
  Real lambda = std::numeric_limits<float>::max();
  if (lwv.cos_a2*lwv.cos_a3 > 0.0) {
    lambda = std::min(lambda, x1size*lwv.cos_a2*lwv.cos_a3);
  }
  if (lwv.cos_a2*lwv.sin_a3 > 0.0) {
    lambda = std::min(lambda, x2size*lwv.cos_a2*lwv.sin_a3);
  }
  if (lwv.sin_a2 > 0.0) lambda = std::min(lambda, x3size*lwv.sin_a2);

  // Initialize k_parallel
  lwv.k_par = 2.0*(M_PI)/lambda;

  // Set background state: v1_0 is parallel to wavevector.
  // Similarly, for MHD:   b1_0 is parallel to wavevector, b2_0/b3_0 are perpendicular
  lwv.d0 = 1.0;
  lwv.v1_0 = vflow;
  lwv.b1_0 = 1.0;
  lwv.b2_0 = std::sqrt(2.0);
  lwv.b3_0 = 0.5;
  Real xfact = 0.0;
  Real yfact = 1.0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // initialize Hydro variables ----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;

    // Compute eigenvectors in hydrodynamics
    Real rem[5][5];
    Real ev[5];
    HydroEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, p0, eos, ev, rem);

    // set new time limit in ParameterInput (to be read by Driver constructor) based on
    // wave speed of selected mode.
    // input tlim is interpreted asnumber of wave periods for evolution
    if (set_initial_conditions) {
      Real tlim = pin->GetReal("time", "tlim");
      pin->SetReal("time", "tlim", tlim*(std::abs(lambda/ev[wave_flag])));
    }

    // compute solution in u1 register. For initial conditions, set u1 -> u0.
    auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;

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
      Real mx = lwv.d0*vflow + amp*sn*rem[1][wave_flag];
      Real my = amp*sn*rem[2][wave_flag];
      Real mz = amp*sn*rem[3][wave_flag];

      // compute cell-centered conserved variables
      u1(m,IDN,k,j,i)=lwv.d0 + amp*sn*rem[0][wave_flag];
      u1(m,IM1,k,j,i)=mx*lwv.cos_a2*lwv.cos_a3 -my*lwv.sin_a3 -mz*lwv.sin_a2*lwv.cos_a3;
      u1(m,IM2,k,j,i)=mx*lwv.cos_a2*lwv.sin_a3 +my*lwv.cos_a3 -mz*lwv.sin_a2*lwv.sin_a3;
      u1(m,IM3,k,j,i)=mx*lwv.sin_a2                           +mz*lwv.cos_a2;

      if (eos.is_ideal) {
        u1(m,IEN,k,j,i) = p0/gm1 + 0.5*lwv.d0*(lwv.v1_0)*(lwv.v1_0) +
                        amp*sn*rem[4][wave_flag];
      }
    });
  }  // End initialization Hydro variables

  // initialize MHD variables ------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    int nmb = pmbp->nmb_thispack;
    int nmhd_ = pmbp->pmhd->nmhd;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;

    // Compute eigenvectors in mhd
    Real rem[7][7];
    Real ev[7];
    MHDEigensystem(lwv.d0, lwv.v1_0, 0.0, 0.0, p0, lwv.b1_0, lwv.b2_0, lwv.b3_0,
                   xfact, yfact, eos, ev, rem);
    lwv.dby = amp*rem[nmhd_  ][wave_flag];
    lwv.dbz = amp*rem[nmhd_+1][wave_flag];

    // set new time limit in ParameterInput (to be read by Driver constructor) based on
    // wave speed of selected mode.
    // input tlim should be interpreted as number of wave periods for evolution
    if (set_initial_conditions) {
      Real tlim = pin->GetReal("time", "tlim");
      pin->SetReal("time", "tlim", tlim*(std::abs(lambda/ev[wave_flag])));
    }

    // compute solution in u1/b1 registers. For initial conditions, set u1/b1 -> u0/b0.
    auto &u1 = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
    auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;

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

    // now compute conserved quantities, as well as face-centered fields
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
      Real mx = lwv.d0*vflow + amp*sn*rem[1][wave_flag];
      Real my = amp*sn*rem[2][wave_flag];
      Real mz = amp*sn*rem[3][wave_flag];

      // compute cell-centered conserved variables
      u1(m,IDN,k,j,i)=lwv.d0 + amp*sn*rem[0][wave_flag];
      u1(m,IM1,k,j,i)=mx*lwv.cos_a2*lwv.cos_a3 -my*lwv.sin_a3 -mz*lwv.sin_a2*lwv.cos_a3;
      u1(m,IM2,k,j,i)=mx*lwv.cos_a2*lwv.sin_a3 +my*lwv.cos_a3 -mz*lwv.sin_a2*lwv.sin_a3;
      u1(m,IM3,k,j,i)=mx*lwv.sin_a2                           +mz*lwv.cos_a2;

      if (eos.is_ideal) {
        u1(m,IEN,k,j,i) = p0/gm1 + 0.5*lwv.d0*SQR(lwv.v1_0) +
         amp*sn*rem[4][wave_flag] + 0.5*(SQR(lwv.b1_0) + SQR(lwv.b2_0) + SQR(lwv.b3_0));
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
  }  // End initialization MHD variables

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroEigensystem()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal hydrodynamics

void HydroEigensystem(const Real d, const Real v1, const Real v2, const Real v3,
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
    right_eigenmatrix[1][0] = v1 - a;
    right_eigenmatrix[2][0] = v2;
    right_eigenmatrix[3][0] = v3;
    right_eigenmatrix[4][0] = h - v1*a;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = 1.0;
    right_eigenmatrix[3][1] = 0.0;
    right_eigenmatrix[4][1] = v2;

    right_eigenmatrix[0][2] = 0.0;
    right_eigenmatrix[1][2] = 0.0;
    right_eigenmatrix[2][2] = 0.0;
    right_eigenmatrix[3][2] = 1.0;
    right_eigenmatrix[4][2] = v3;

    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[1][3] = v1;
    right_eigenmatrix[2][3] = v2;
    right_eigenmatrix[3][3] = v3;
    right_eigenmatrix[4][3] = 0.5*vsq;

    right_eigenmatrix[0][4] = 1.0;
    right_eigenmatrix[1][4] = v1 + a;
    right_eigenmatrix[2][4] = v2;
    right_eigenmatrix[3][4] = v3;
    right_eigenmatrix[4][4] = h + v1*a;

  //--- Isothermal Hydrodynamics ---
  } else {
    // Compute eigenvalues (eq. B6)
    eigenvalues[0] = v1 - eos.iso_cs;
    eigenvalues[1] = v1;
    eigenvalues[2] = v1;
    eigenvalues[3] = v1 + eos.iso_cs;

    // Right-eigenvectors, stored as COLUMNS (eq. B3)
    right_eigenmatrix[0][0] = 1.0;
    right_eigenmatrix[1][0] = v1 - eos.iso_cs;
    right_eigenmatrix[2][0] = v2;
    right_eigenmatrix[3][0] = v3;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = 1.0;
    right_eigenmatrix[3][1] = 0.0;

    right_eigenmatrix[0][2] = 0.0;
    right_eigenmatrix[1][2] = 0.0;
    right_eigenmatrix[2][2] = 0.0;
    right_eigenmatrix[3][2] = 1.0;

    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[1][3] = v1 + eos.iso_cs;
    right_eigenmatrix[2][3] = v2;
    right_eigenmatrix[3][3] = v3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHDEigensystem()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal mhd

void MHDEigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                    const Real p, const Real b1, const Real b2, const Real b3,
                    const Real x, const Real y, const EOS_Data &eos,
                    Real eigenvalues[7], Real right_eigenmatrix[7][7]) {
  // common factors for both ideal gas and isothermal eigenvectors
  Real btsq = b2*b2 + b3*b3;
  Real bt = std::sqrt(btsq);
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
    Real twid_asq = std::max( (gm1*(hp-0.5*vsq) - (gm1-1.0)*x),
                              static_cast<Real>(std::numeric_limits<float>::min()) );

    // Compute fast- and slow-magnetosonic speeds (eq. B18)
    Real ct2 = bt_starsq/d;
    Real tsum = vaxsq + ct2 + twid_asq;
    Real tdif = vaxsq + ct2 - twid_asq;
    Real cf2_cs2 = std::sqrt(tdif*tdif + 4.0*twid_asq*ct2);

    Real cfsq = 0.5*(tsum + cf2_cs2);
    Real cf = std::sqrt(cfsq);

    Real cssq = twid_asq*vaxsq/cfsq;
    Real cs = std::sqrt(cssq);

    // Compute beta(s) (eqs. A17, B20, B28)
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
    } else if ( (twid_asq - cssq) <= 0.0) {
      alpha_f = 0.0;
      alpha_s = 1.0;
    } else if ( (cfsq - twid_asq) <= 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else {
      alpha_f = std::sqrt((twid_asq - cssq)/(cfsq - cssq));
      alpha_s = std::sqrt((cfsq - twid_asq)/(cfsq - cssq));
    }

    // Compute Q(s) and A(s) (eq. A14-15), etc.
    Real sqrtd = std::sqrt(d);
    Real s = SIGN(b1);
    Real twid_a = std::sqrt(twid_asq);
    Real qf = cf*alpha_f*s;
    Real qs = cs*alpha_s*s;
    Real af_prime = twid_a*alpha_f/sqrtd;
    Real as_prime = twid_a*alpha_s/sqrtd;
    Real afpbb = af_prime*bt_star*bet_starsq;
    Real aspbb = as_prime*bt_star*bet_starsq;

    // Compute eigenvalues (eq. B17)
    Real vax = std::sqrt(vaxsq);
    eigenvalues[0] = v1 - cf;
    eigenvalues[1] = v1 - vax;
    eigenvalues[2] = v1 - cs;
    eigenvalues[3] = v1;
    eigenvalues[4] = v1 + cs;
    eigenvalues[5] = v1 + vax;
    eigenvalues[6] = v1 + cf;

    // Right-eigenvectors, stored as COLUMNS (eq. B21) */
    right_eigenmatrix[0][0] = alpha_f;
    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[0][2] = alpha_s;
    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[0][4] = alpha_s;
    right_eigenmatrix[0][5] = 0.0;
    right_eigenmatrix[0][6] = alpha_f;

    right_eigenmatrix[1][0] = alpha_f*(v1 - cf);
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[1][2] = alpha_s*(v1 - cs);
    right_eigenmatrix[1][3] = v1;
    right_eigenmatrix[1][4] = alpha_s*(v1 + cs);
    right_eigenmatrix[1][5] = 0.0;
    right_eigenmatrix[1][6] = alpha_f*(v1 + cf);

    Real qa = alpha_f*v2;
    Real qb = alpha_s*v2;
    Real qc = qs*bet2_star;
    Real qd = qf*bet2_star;
    right_eigenmatrix[2][0] = qa + qc;
    right_eigenmatrix[2][1] = -bet3;
    right_eigenmatrix[2][2] = qb - qd;
    right_eigenmatrix[2][3] = v2;
    right_eigenmatrix[2][4] = qb + qd;
    right_eigenmatrix[2][5] = bet3;
    right_eigenmatrix[2][6] = qa - qc;

    qa = alpha_f*v3;
    qb = alpha_s*v3;
    qc = qs*bet3_star;
    qd = qf*bet3_star;
    right_eigenmatrix[3][0] = qa + qc;
    right_eigenmatrix[3][1] = bet2;
    right_eigenmatrix[3][2] = qb - qd;
    right_eigenmatrix[3][3] = v3;
    right_eigenmatrix[3][4] = qb + qd;
    right_eigenmatrix[3][5] = -bet2;
    right_eigenmatrix[3][6] = qa - qc;

    right_eigenmatrix[4][0] = alpha_f*(hp - v1*cf) + qs*vbet + aspbb;
    right_eigenmatrix[4][1] = -(v2*bet3 - v3*bet2);
    right_eigenmatrix[4][2] = alpha_s*(hp - v1*cs) - qf*vbet - afpbb;
    right_eigenmatrix[4][3] = 0.5*vsq + (gm1-1.0)*x/gm1;
    right_eigenmatrix[4][4] = alpha_s*(hp + v1*cs) + qf*vbet - afpbb;
    right_eigenmatrix[4][5] = -right_eigenmatrix[4][1];
    right_eigenmatrix[4][6] = alpha_f*(hp + v1*cf) - qs*vbet + aspbb;

    right_eigenmatrix[5][0] = as_prime*bet2_star;
    right_eigenmatrix[5][1] = -bet3*s/sqrtd;
    right_eigenmatrix[5][2] = -af_prime*bet2_star;
    right_eigenmatrix[5][3] = 0.0;
    right_eigenmatrix[5][4] = right_eigenmatrix[5][2];
    right_eigenmatrix[5][5] = right_eigenmatrix[5][1];
    right_eigenmatrix[5][6] = right_eigenmatrix[5][0];

    right_eigenmatrix[6][0] = as_prime*bet3_star;
    right_eigenmatrix[6][1] = bet2*s/sqrtd;
    right_eigenmatrix[6][2] = -af_prime*bet3_star;
    right_eigenmatrix[6][3] = 0.0;
    right_eigenmatrix[6][4] = right_eigenmatrix[6][2];
    right_eigenmatrix[6][5] = right_eigenmatrix[6][1];
    right_eigenmatrix[6][6] = right_eigenmatrix[6][0];
  //--- Isothermal MHD ---
  } else {
    Real bt_starsq = btsq*y;
    Real vaxsq = b1*b1/d;
    Real twid_csq = (eos.iso_cs*eos.iso_cs) + x;

    // Compute fast- and slow-magnetosonic speeds (eq. B39)
    Real ct2 = bt_starsq/d;
    Real tsum = vaxsq + ct2 + twid_csq;
    Real tdif = vaxsq + ct2 - twid_csq;
    Real cf2_cs2 = std::sqrt(tdif*tdif + 4.0*twid_csq*ct2);

    Real cfsq = 0.5*(tsum + cf2_cs2);
    Real cf = std::sqrt(cfsq);

    Real cssq = twid_csq*vaxsq/cfsq;
    Real cs = std::sqrt(cssq);

    Real bet2_star = bet2/std::sqrt(y);
    Real bet3_star = bet3/std::sqrt(y);

    // Compute alpha's (eq. A16)
    Real alpha_f,alpha_s;
    if ((cfsq-cssq) == 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else if ((twid_csq - cssq) <= 0.0) {
      alpha_f = 0.0;
      alpha_s = 1.0;
    } else if ((cfsq - twid_csq) <= 0.0) {
      alpha_f = 1.0;
      alpha_s = 0.0;
    } else {
      alpha_f = std::sqrt((twid_csq - cssq)/(cfsq - cssq));
      alpha_s = std::sqrt((cfsq - twid_csq)/(cfsq - cssq));
    }

    // Compute Q's (eq. A14-15), etc.
    Real sqrtd = std::sqrt(d);
    Real s = SIGN(b1);
    Real twid_c = std::sqrt(twid_csq);
    Real qf = cf*alpha_f*s;
    Real qs = cs*alpha_s*s;
    Real af_prime = twid_c*alpha_f/sqrtd;
    Real as_prime = twid_c*alpha_s/sqrtd;

    // Compute eigenvalues (eq. B38)
    Real vax  = std::sqrt(vaxsq);
    eigenvalues[0] = v1 - cf;
    eigenvalues[1] = v1 - vax;
    eigenvalues[2] = v1 - cs;
    eigenvalues[3] = v1 + cs;
    eigenvalues[4] = v1 + vax;
    eigenvalues[5] = v1 + cf;

    // Right-eigenvectors, stored as COLUMNS (eq. B21)
    right_eigenmatrix[0][0] = alpha_f;
    right_eigenmatrix[1][0] = alpha_f*(v1 - cf);
    right_eigenmatrix[2][0] = alpha_f*v2 + qs*bet2_star;
    right_eigenmatrix[3][0] = alpha_f*v3 + qs*bet3_star;
    right_eigenmatrix[4][0] = as_prime*bet2_star;
    right_eigenmatrix[5][0] = as_prime*bet3_star;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = -bet3;
    right_eigenmatrix[3][1] = bet2;
    right_eigenmatrix[4][1] = -bet3*s/sqrtd;
    right_eigenmatrix[5][1] = bet2*s/sqrtd;

    right_eigenmatrix[0][2] = alpha_s;
    right_eigenmatrix[1][2] = alpha_s*(v1 - cs);
    right_eigenmatrix[2][2] = alpha_s*v2 - qf*bet2_star;
    right_eigenmatrix[3][2] = alpha_s*v3 - qf*bet3_star;
    right_eigenmatrix[4][2] = -af_prime*bet2_star;
    right_eigenmatrix[5][2] = -af_prime*bet3_star;

    right_eigenmatrix[0][3] = alpha_s;
    right_eigenmatrix[1][3] = alpha_s*(v1 + cs);
    right_eigenmatrix[2][3] = alpha_s*v2 + qf*bet2_star;
    right_eigenmatrix[3][3] = alpha_s*v3 + qf*bet3_star;
    right_eigenmatrix[4][3] = right_eigenmatrix[4][2];
    right_eigenmatrix[5][3] = right_eigenmatrix[5][2];

    right_eigenmatrix[0][4] = 0.0;
    right_eigenmatrix[1][4] = 0.0;
    right_eigenmatrix[2][4] = bet3;
    right_eigenmatrix[3][4] = -bet2;
    right_eigenmatrix[4][4] = right_eigenmatrix[4][1];
    right_eigenmatrix[5][4] = right_eigenmatrix[5][1];

    right_eigenmatrix[0][5] = alpha_f;
    right_eigenmatrix[1][5] = alpha_f*(v1 + cf);
    right_eigenmatrix[2][5] = alpha_f*v2 - qs*bet2_star;
    right_eigenmatrix[3][5] = alpha_f*v3 - qs*bet3_star;
    right_eigenmatrix[4][5] = right_eigenmatrix[4][0];
    right_eigenmatrix[5][5] = right_eigenmatrix[5][0];
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void LinearWaveErrors_()
//! \brief Computes errors in linear wave solution by calling initialization function
//! again to compute initial condictions, and subtracting current solution from ICs, and
//! outputs errors to file. Problem must be run for an integer number of wave periods.

void LinearWaveErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->LinearWave(pin, false);

  Real l1_err[8];
  Real linfty_err=0.0;
  int nvars=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("LW-err",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

  // compute errors for MHD  -------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + 3;  // include 3-compts of cell-centered B in errors

    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    auto &u0_ = pmbp->pmhd->u0;
    auto &u1_ = pmbp->pmhd->u1;
    auto &b0_ = pmbp->pmhd->b0;
    auto &b1_ = pmbp->pmhd->b1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("LW-err-Sums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // cell-centered B
      Real bcc0 = 0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1));
      Real bcc1 = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      evars.the_array[IEN+1] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+1]);

      bcc0 = 0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
      bcc1 = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      evars.the_array[IEN+2] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+2]);

      bcc0 = 0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i));
      bcc1 = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));
      evars.the_array[IEN+3] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+3]);

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nvars, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] = l1_err[i]/vol;
  linfty_err /= vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1    L-infty       ");
      std::fprintf(pfile,"d_L1         M1_L1         M2_L1         M3_L1         E_L1");
      if (pmbp->pmhd != nullptr) {
        std::fprintf(pfile,"          B1_L1         B2_L1         B3_L1");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}
