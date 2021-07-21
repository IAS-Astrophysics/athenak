//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shock_tube.cpp
//  \brief Problem generator for shock tube (1-D Riemann) problems in both hydro and MHD.
//
// Initializes plane-parallel shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D),
// and along x3 (in 3D).

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

// Define some containers to hold Hydro conserved/primitive variables
struct Cons1D {
  Real d, m1, m2, m3, e;
};
struct Prim1D {
  Real d, vx, vy, vz, p;
};

// Define a function to compute prim->cons for either Newtonian or SR dynamics
// TODO: Currently this pgen is the only function that needs Prim->Cons transorms. If
// such a functions is needed in more places (e.g. AMR prolongation) then it should
// be moved to EOS class.

// TODO: need to add MHD PrimToCons when add SR MHD
void PrimToConsHydro(Prim1D &w, Cons1D &u, Real gam_eos, bool is_sr)
{
  if (is_sr) {
    Real v_sq = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
    Real gamma_sq = 1.0/(1.0 - v_sq);
    Real gamma = sqrt(gamma_sq);
    //FIXME: Only ideal fluid for now
    Real wgas = w.d + (gam_eos/(gam_eos - 1.))*w.p;

    u.d  = gamma * w.d;
    u.m1 = wgas * gamma_sq * w.vx;
    u.m2 = wgas * gamma_sq * w.vy;
    u.m3 = wgas * gamma_sq * w.vz;
    u.e  = wgas * gamma_sq - w.p - gamma * w.d; 
  } else {
    u.d  = w.d;
    u.m1 = w.vx*w.d;
    u.m2 = w.vy*w.d;
    u.m3 = w.vz*w.d;
    u.e  = w.p/(gam_eos - 1.0) + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)); 
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ShockTube_()
//  \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::ShockTube_(MeshBlockPack *pmbp, ParameterInput *pin)
{
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
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto coord = pmbp->coord.coord_data;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {

    // Parse left state read from input file: d,vx,vy,vz,[P]
    Prim1D wl,wr;
    Cons1D ul,ur;
    wl.d  = pin->GetReal("problem","dn_l");
    wl.vx = pin->GetReal("problem","un_l");
    wl.vy = pin->GetReal("problem","vn_l");
    wl.vz = pin->GetReal("problem","wn_l");
    wl.p  = pin->GetReal("problem","pn_l");
    Real gam = pmbp->phydro->peos->eos_data.gamma;
    PrimToConsHydro(wl,ul,gam,pmbp->phydro->is_special_relativistic);
  
    // Parse right state read from input file: d,vx,vy,vz,[P]
    wr.d  = pin->GetReal("problem","dn_r");
    wr.vx = pin->GetReal("problem","un_r");
    wr.vy = pin->GetReal("problem","vn_r");
    wr.vz = pin->GetReal("problem","wn_r");
    wr.p  = pin->GetReal("problem","pn_r");
    PrimToConsHydro(wr,ur,gam,pmbp->phydro->is_special_relativistic);

    auto &u0 = pmbp->phydro->u0;
    par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k, int j, int i)
      {
        Real x;
        if (shk_dir == 1) {
          Real &x1min = coord.mb_size.d_view(m).x1min;
          Real &x1max = coord.mb_size.d_view(m).x1max;
          int nx1 = coord.mb_indcs.nx1;
          x = CellCenterX(i-is, nx1, x1min, x1max);
        } else if (shk_dir == 2) {
          Real &x2min = coord.mb_size.d_view(m).x2min;
          Real &x2max = coord.mb_size.d_view(m).x2max;
          int nx2 = coord.mb_indcs.nx2;
          x = CellCenterX(j-js, nx2, x2min, x2max);
        } else {
          Real &x3min = coord.mb_size.d_view(m).x3min;
          Real &x3max = coord.mb_size.d_view(m).x3max;
          int nx3 = coord.mb_indcs.nx3;
          x = CellCenterX(k-ks, nx3, x3min, x3max);
        }

        if (x < xshock) {
          u0(m,IDN,k,j,i) = ul.d;
          u0(m,ivx,k,j,i) = ul.m1;
          u0(m,ivy,k,j,i) = ul.m2;
          u0(m,ivz,k,j,i) = ul.m3;
          u0(m,IEN,k,j,i) = ul.e;
        } else {
          u0(m,IDN,k,j,i) = ur.d;
          u0(m,ivx,k,j,i) = ur.m1;
          u0(m,ivy,k,j,i) = ur.m2;
          u0(m,ivz,k,j,i) = ur.m3;
          u0(m,IEN,k,j,i) = ur.e;
        }
      }
    );
  } // End initialization of Hydro variables

  // Initialize MHD variables -------------------------------
  if (pmbp->pmhd != nullptr) {
  
    // Parse left state read from input file: d,vx,vy,vz,[P]
    Real wl[5];
    wl[IDN] = pin->GetReal("problem","di_l");
    wl[IVX] = pin->GetReal("problem","ui_l");
    wl[IVY] = pin->GetReal("problem","vi_l");
    wl[IVZ] = pin->GetReal("problem","wi_l");
    wl[IPR] = pin->GetReal("problem","pi_l");
    Real wl_bx = pin->GetReal("problem","bxl");
    Real wl_by = pin->GetReal("problem","byl");
    Real wl_bz = pin->GetReal("problem","bzl");
    
    // Parse right state read from input file: d,vx,vy,vz,[P]
    Real wr[5];
    wr[IDN] = pin->GetReal("problem","di_r");
    wr[IVX] = pin->GetReal("problem","ui_r");
    wr[IVY] = pin->GetReal("problem","vi_r");
    wr[IVZ] = pin->GetReal("problem","wi_r");
    wr[IPR] = pin->GetReal("problem","pi_r");
    Real wr_bx = pin->GetReal("problem","bxr");
    Real wr_by = pin->GetReal("problem","byr");
    Real wr_bz = pin->GetReal("problem","bzr");
    
    Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k, int j, int i)
      {
        Real x,bxl,byl,bzl,bxr,byr,bzr;
        if (shk_dir == 1) {
          Real &x1min = coord.mb_size.d_view(m).x1min;
          Real &x1max = coord.mb_size.d_view(m).x1max;
          int nx1 = coord.mb_indcs.nx1;
          x = CellCenterX(i-is, nx1, x1min, x1max);
          bxl = wl_bx; byl = wl_by; bzl = wl_bz;
          bxr = wr_bx; byr = wr_by; bzr = wr_bz;
        } else if (shk_dir == 2) {
          Real &x2min = coord.mb_size.d_view(m).x2min;
          Real &x2max = coord.mb_size.d_view(m).x2max;
          int nx2 = coord.mb_indcs.nx2;
          x = CellCenterX(j-js, nx2, x2min, x2max);
          bxl = wl_bz; byl = wl_bx; bzl = wl_by;
          bxr = wr_bz; byr = wr_bx; bzr = wr_by;
        } else {
          Real &x3min = coord.mb_size.d_view(m).x3min;
          Real &x3max = coord.mb_size.d_view(m).x3max;
          int nx3 = coord.mb_indcs.nx3;
          x = CellCenterX(k-ks, nx3, x3min, x3max);
          bxl = wl_by; byl = wl_bz; bzl = wl_bx;
          bxr = wr_by; byr = wr_bz; bzr = wr_bx;
        } 
          
        if (x < xshock) {
          u0(m,IDN,k,j,i) = wl[IDN]; 
          u0(m,ivx,k,j,i) = wl[IVX]*wl[IDN];
          u0(m,ivy,k,j,i) = wl[IVY]*wl[IDN];
          u0(m,ivz,k,j,i) = wl[IVZ]*wl[IDN];
          u0(m,IEN,k,j,i) = wl[IPR]/gm1 +
             0.5*wl[IDN]*(SQR(wl[IVX]) + SQR(wl[IVY]) + SQR(wl[IVZ])) +
             0.5*(SQR(bxl) + SQR(byl) + SQR(bzl));
          b0.x1f(m,k,j,i) = bxl;
          b0.x2f(m,k,j,i) = byl;
          b0.x3f(m,k,j,i) = bzl;
          if (i==ie) {b0.x1f(m,k,j,i+1) = bxl;}
          if (j==je) {b0.x2f(m,k,j+1,i) = byl;}
          if (k==ke) {b0.x3f(m,k+1,j,i) = bzl;}
        } else {
          u0(m,IDN,k,j,i) = wr[IDN];
          u0(m,ivx,k,j,i) = wr[IVX]*wr[IDN];
          u0(m,ivy,k,j,i) = wr[IVY]*wr[IDN];
          u0(m,ivz,k,j,i) = wr[IVZ]*wr[IDN];
          u0(m,IEN,k,j,i) = wr[IPR]/gm1 +
             0.5*wr[IDN]*(SQR(wr[IVX]) + SQR(wr[IVY]) + SQR(wr[IVZ])) +
             0.5*(SQR(bxr) + SQR(byr) + SQR(bzr));
          b0.x1f(m,k,j,i) = bxr;
          b0.x2f(m,k,j,i) = byr;
          b0.x3f(m,k,j,i) = bzr;
          if (i==ie) {b0.x1f(m,k,j,i+1) = bxr;}
          if (j==je) {b0.x2f(m,k,j+1,i) = byr;}
          if (k==ke) {b0.x3f(m,k+1,j,i) = bzr;}
        }
      }
    );
  } // End initialization of MHD variables

  return;
}
