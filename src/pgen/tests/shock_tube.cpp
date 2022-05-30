//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shock_tube.cpp
//! \brief Problem generator for shock tube (1-D Riemann) problems in both hydro and MHD.
//! Works for both non-relativistic and relativistic dynamics in flat (Minkowski)
//! spacetimes.  Can be used to test GR, but metric must be Minkowski.
//!
//! Works by initializing plane-parallel shock along x1 (in 1D, 2D, 3D), along x2
//! (in 2D, 3D), and along x3 (in 3D).  Shock must be along a coordinate directions,
//! i.e. shocks propagating along an angle inclined to grid are not implemented.

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ShockTube_()
//! \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::ShockTube(ParameterInput *pin, const bool restart) {
  if (restart) return;
  // parse shock direction: {1,2,3} -> {x1,x2,x3}
  int shk_dir = pin->GetInteger("problem","shock_dir");
  if (shk_dir < 1 || shk_dir > 3) {
    // Invaild input value for shk_dir
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "shock_dir=" <<shk_dir<< " must be either 1,2, or 3" << std::endl;
    exit(EXIT_FAILURE);
  }
  // set indices of parallel and perpendicular velocities
  int ivx = shk_dir;
  int ivy = IVX + ((ivx - IVX) + 1)%3;
  int ivz = IVX + ((ivx - IVX) + 2)%3;

  // parse shock location (must be inside grid; set L/R states equal to each other to
  // initialize uniform initial conditions))
  Real xshock = pin->GetReal("problem","xshock");
  if (shk_dir == 1 && (xshock < pmy_mesh_->mesh_size.x1min ||
                       xshock > pmy_mesh_->mesh_size.x1max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x1 domain" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (shk_dir == 2 && (xshock < pmy_mesh_->mesh_size.x2min ||
                       xshock > pmy_mesh_->mesh_size.x2max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x2 domain" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (shk_dir == 3 && (xshock < pmy_mesh_->mesh_size.x3min ||
                       xshock > pmy_mesh_->mesh_size.x3max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x3 domain" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &eos = pmbp->phydro->peos->eos_data;
    // Parse left state read from input file: d,vx,vy,vz,[P]
    HydPrim1D wl,wr;
    wl.d  = pin->GetReal("problem","dl");
    wl.vx = pin->GetReal("problem","ul");
    wl.vy = pin->GetReal("problem","vl");
    wl.vz = pin->GetReal("problem","wl");
    wl.e  = (pin->GetReal("problem","pl"))/(eos.gamma - 1.0);
    // compute Lorentz factor (needed for SR/GR)
    Real u0l = 1.0;
    if (pmbp->pcoord->is_special_relativistic || pmbp->pcoord->is_general_relativistic) {
      u0l = 1.0/sqrt( 1.0 - (SQR(wl.vx) + SQR(wl.vy) + SQR(wl.vz)) );
    }

    // Parse right state read from input file: d,vx,vy,vz,[P]
    wr.d  = pin->GetReal("problem","dr");
    wr.vx = pin->GetReal("problem","ur");
    wr.vy = pin->GetReal("problem","vr");
    wr.vz = pin->GetReal("problem","wr");
    wr.e  = (pin->GetReal("problem","pr"))/(eos.gamma - 1.0);
    // compute Lorentz factor (needed for SR/GR)
    Real u0r = 1.0;
    if (pmbp->pcoord->is_special_relativistic || pmbp->pcoord->is_general_relativistic) {
      u0r = 1.0/sqrt( 1.0 - (SQR(wr.vx) + SQR(wr.vy) + SQR(wr.vz)) );
    }

    auto &w0 = pmbp->phydro->w0;
    par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real x;
      if (shk_dir == 1) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        x = CellCenterX(i-is, nx1, x1min, x1max);
      } else if (shk_dir == 2) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        x = CellCenterX(j-js, nx2, x2min, x2max);
      } else {
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        x = CellCenterX(k-ks, nx3, x3min, x3max);
      }

      // in SR/GR, primitive variables use spatial components of 4-vel u^i = gamma * v^i
      if (x < xshock) {
        w0(m,IDN,k,j,i) = wl.d;
        w0(m,ivx,k,j,i) = wl.vx*u0l;
        w0(m,ivy,k,j,i) = wl.vy*u0l;
        w0(m,ivz,k,j,i) = wl.vz*u0l;
        w0(m,IEN,k,j,i) = wl.e;
      } else {
        w0(m,IDN,k,j,i) = wr.d;
        w0(m,ivx,k,j,i) = wr.vx*u0r;
        w0(m,ivy,k,j,i) = wr.vy*u0r;
        w0(m,ivz,k,j,i) = wr.vz*u0r;
        w0(m,IEN,k,j,i) = wr.e;
      }
    });

    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
  } // End initialization of Hydro variables

  // Initialize MHD variables -------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &eos = pmbp->pmhd->peos->eos_data;
    // Parse left state read from input file: d,vx,vy,vz,[P]
    MHDPrim1D wl,wr;
    wl.d  = pin->GetReal("problem","dl");
    wl.vx = pin->GetReal("problem","ul");
    wl.vy = pin->GetReal("problem","vl");
    wl.vz = pin->GetReal("problem","wl");
    wl.e  = (pin->GetReal("problem","pl"))/(eos.gamma - 1.0);
    wl.by = pin->GetReal("problem","byl");
    wl.bz = pin->GetReal("problem","bzl");
    Real bx_l = pin->GetReal("problem","bxl");
    // compute Lorentz factor (needed for SR/GR)
    Real u0l = 1.0;
    if (pmbp->pcoord->is_special_relativistic || pmbp->pcoord->is_general_relativistic) {
      u0l = 1.0/sqrt( 1.0 - (SQR(wl.vx) + SQR(wl.vy) + SQR(wl.vz)) );
    }

    // Parse right state read from input file: d,vx,vy,vz,[P]
    wr.d  = pin->GetReal("problem","dr");
    wr.vx = pin->GetReal("problem","ur");
    wr.vy = pin->GetReal("problem","vr");
    wr.vz = pin->GetReal("problem","wr");
    wr.e  = (pin->GetReal("problem","pr"))/(eos.gamma - 1.0);
    wr.by = pin->GetReal("problem","byr");
    wr.bz = pin->GetReal("problem","bzr");
    Real bx_r = pin->GetReal("problem","bxr");
    // compute Lorentz factor (needed for SR/GR)
    Real u0r = 1.0;
    if (pmbp->pcoord->is_special_relativistic || pmbp->pcoord->is_general_relativistic) {
      u0r = 1.0/sqrt( 1.0 - (SQR(wr.vx) + SQR(wr.vy) + SQR(wr.vz)) );
    }

    auto &w0 = pmbp->pmhd->w0;
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real x,bxl,byl,bzl,bxr,byr,bzr;
      if (shk_dir == 1) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        x = CellCenterX(i-is, nx1, x1min, x1max);
        bxl = bx_l; byl = wl.by; bzl = wl.bz;
        bxr = bx_r; byr = wr.by; bzr = wr.bz;
      } else if (shk_dir == 2) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        x = CellCenterX(j-js, nx2, x2min, x2max);
        bxl = wl.bz; byl = bx_l; bzl = wl.by;
        bxr = wr.bz; byr = bx_r; bzr = wr.by;
      } else {
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        x = CellCenterX(k-ks, nx3, x3min, x3max);
        bxl = wl.by; byl = wl.bz; bzl = bx_l;
        bxr = wr.by; byr = wr.bz; bzr = bx_r;
      }

      // in SR/GR, primitive variables use spatial components of 4-vel u^i = gamma * v^i
      if (x < xshock) {
        w0(m,IDN,k,j,i) = wl.d;
        w0(m,ivx,k,j,i) = wl.vx*u0l;
        w0(m,ivy,k,j,i) = wl.vy*u0l;
        w0(m,ivz,k,j,i) = wl.vz*u0l;
        w0(m,IEN,k,j,i) = wl.e;
        b0.x1f(m,k,j,i) = bxl;
        b0.x2f(m,k,j,i) = byl;
        b0.x3f(m,k,j,i) = bzl;
        if (i==ie) {b0.x1f(m,k,j,i+1) = bxl;}
        if (j==je) {b0.x2f(m,k,j+1,i) = byl;}
        if (k==ke) {b0.x3f(m,k+1,j,i) = bzl;}
        bcc0(m,IBX,k,j,i) = bxl;
        bcc0(m,IBY,k,j,i) = byl;
        bcc0(m,IBZ,k,j,i) = bzl;
      } else {
        w0(m,IDN,k,j,i) = wr.d;
        w0(m,ivx,k,j,i) = wr.vx*u0r;
        w0(m,ivy,k,j,i) = wr.vy*u0r;
        w0(m,ivz,k,j,i) = wr.vz*u0r;
        w0(m,IEN,k,j,i) = wr.e;
        b0.x1f(m,k,j,i) = bxr;
        b0.x2f(m,k,j,i) = byr;
        b0.x3f(m,k,j,i) = bzr;
        if (i==ie) {b0.x1f(m,k,j,i+1) = bxr;}
        if (j==je) {b0.x2f(m,k,j+1,i) = byr;}
        if (k==ke) {b0.x3f(m,k+1,j,i) = bzr;}
        bcc0(m,IBX,k,j,i) = bxr;
        bcc0(m,IBY,k,j,i) = byr;
        bcc0(m,IBZ,k,j,i) = bzr;
      }
    });
    // Convert primitives to conserved
    auto &u0 = pmbp->pmhd->u0;
    pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
  } // End initialization of MHD variables

  return;
}
