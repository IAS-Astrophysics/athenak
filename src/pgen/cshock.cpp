//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cshock.cpp
//! \brief problem generator for C-shock test of two-fluid MHD. Solves ODE on host to
//! compute C-shock profile, then initializes this on the grid. Can then test whether
//! cold holds this profile stably.

// C headers

// C++ headers
#include <cmath>      // sqrt()
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
#include "ion-neutral/ion-neutral.hpp"

struct UpstreamICs {
  Real di0, dn0;
  Real vix0, vnx0;
  Real viy0, vny0;
  Real by0;
};

void RHS(UpstreamICs ics, Real alpha, Real cis, Real cns, Real v[2], Real dvdx[2]) {
  Real di = ics.di0*ics.vix0/v[0];
  Real dn = ics.dn0*ics.vnx0/v[1];
  Real by = ics.by0*di/ics.di0;
  dvdx[0] = -alpha*dn*v[0]*(v[0]-v[1])/(SQR(v[0]) - SQR(by)/di - SQR(cis));
  dvdx[1] =  alpha*di*v[1]*(v[0]-v[1])/(SQR(v[1])              - SQR(cns));
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pmhd == nullptr) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "C-shock problem requires both Hydro and MHD" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->phydro->peos->eos_data.is_ideal) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "C-shock problem requires isothermal EOS for Hydro" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->peos->eos_data.is_ideal) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "C-shock problem requires isothermal EOS for MHD" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read problem parameters
  UpstreamICs ics;
  ics.di0 = pin->GetReal("problem", "di0");
  ics.dn0 = pin->GetReal("problem", "dn0");
  ics.vix0 = pin->GetReal("problem", "vix0");
  ics.vnx0 = pin->GetReal("problem", "vnx0");
  ics.viy0 = pin->GetReal("problem", "viy0");
  ics.vny0 = pin->GetReal("problem", "vny0");
  ics.by0 = pin->GetReal("problem", "by0");
  Real pert = pin->GetOrAddReal("problem", "pert", 1.0e-4);

  // Fluid properties
  Real alpha = pmbp->pionn->drag_coeff;
  Real cns   = pmbp->phydro->peos->eos_data.iso_cs;
  Real cis   = pmbp->pmhd->peos->eos_data.iso_cs;

  // Use RK4 to integrate C-shock profile on host using 10x finer mesh
  // v[0] = ion velocity
  // v[1] = neutral velocity
  Real v[2], dvdx[2];
  int nxshk = 10*pmy_mesh_->mesh_indcs.nx1;
  Real dxshk = (pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min)/
                static_cast<Real>(nxshk);
  HostArray1D<Real> xshk("xshk", nxshk);
  HostArray2D<Real> vshk("vshk", 2, nxshk);
  xshk(0) = pmy_mesh_->mesh_size.x1min + (0.5*dxshk);
  vshk(0,0) = ics.vix0 - pert;
  vshk(1,0) = ics.vnx0;
  for (int n=0; n<(nxshk-1); ++n) {
    Real v[2], dvdx1[2], dvdx2[2], dvdx3[2], dvdx4[2];
    v[0] = vshk(0,n);
    v[1] = vshk(1,n);
    RHS(ics, alpha, cis, cns, v, dvdx1);

    v[0] = vshk(0,n) + dvdx1[0]*dxshk/2.0;
    v[1] = vshk(1,n) + dvdx1[1]*dxshk/2.0;
    RHS(ics, alpha, cis, cns, v, dvdx2);

    v[0] = vshk(0,n) +  dvdx2[0]*dxshk/2.0;
    v[1] = vshk(1,n) +  dvdx2[1]*dxshk/2.0;
    RHS(ics, alpha, cis, cns, v, dvdx3);

    v[0] = vshk(0,n) +  dvdx3[0]*dxshk;
    v[1] = vshk(1,n) +  dvdx3[1]*dxshk;
    RHS(ics, alpha, cis, cns, v, dvdx4);

    xshk(n+1) = xshk(n) + dxshk;
    vshk(0,n+1) = vshk(0,n) + dxshk*(dvdx1[0] + 2.*dvdx2[0] + 2.*dvdx3[0] + dvdx4[0])/6.0;
    vshk(1,n+1) = vshk(1,n) + dxshk*(dvdx1[1] + 2.*dvdx2[1] + 2.*dvdx3[1] + dvdx4[1])/6.0;
  }

  // bin solution into DualArray at resolution of grid, sync to device
  // shksol indices refer to:  0=di, 1=dn, 2=vix, 3=vnx, 4=viy, 5=vny, 6=by
  DualArray2D<Real> shksol("shksol",7,pmy_mesh_->mesh_indcs.nx1);
  for (int n=0; n<(pmy_mesh_->mesh_indcs.nx1); ++n) {
    for (int m=0; m<7; ++m) {
      shksol.h_view(m,n) = 0.0;
    }
    for (int m=0; m<10; ++m) {
      shksol.h_view(0,n) += ics.di0*ics.vix0/vshk(0,(m + 10*n));
      shksol.h_view(1,n) += ics.dn0*ics.vnx0/vshk(1,(m + 10*n));
      shksol.h_view(2,n) += vshk(0,(m + 10*n));
      shksol.h_view(3,n) += vshk(1,(m + 10*n));
      shksol.h_view(4,n) += 0.0;
      shksol.h_view(5,n) += 0.0;
      shksol.h_view(6,n) += ics.by0*ics.vix0/vshk(0,(m + 10*n));
    }
    for (int m=0; m<7; ++m) {
      shksol.h_view(m,n) *= 0.1;
    }
  }
  shksol.template modify<HostMemSpace>();
  shksol.template sync<DevExeSpace>();

  // set inflow state in BoundaryValues, sync to device
  auto un_in = pmbp->phydro->pbval_u->u_in;
  auto ui_in = pmbp->pmhd->pbval_u->u_in;
  auto bi_in = pmbp->pmhd->pbval_b->b_in;
  un_in.h_view(IDN,BoundaryFace::inner_x1) = ics.dn0;
  ui_in.h_view(IDN,BoundaryFace::inner_x1) = ics.di0;
  un_in.h_view(IM1,BoundaryFace::inner_x1) = ics.dn0*ics.vnx0;
  ui_in.h_view(IM1,BoundaryFace::inner_x1) = ics.di0*ics.vix0;
  un_in.h_view(IM2,BoundaryFace::inner_x1) = ics.dn0*ics.vny0;
  ui_in.h_view(IM2,BoundaryFace::inner_x1) = ics.di0*ics.viy0;
  bi_in.h_view(IBX,BoundaryFace::inner_x1) = 0.0;
  bi_in.h_view(IBY,BoundaryFace::inner_x1) = ics.by0;
  bi_in.h_view(IBZ,BoundaryFace::inner_x1) = 0.0;
  un_in.template modify<HostMemSpace>();
  un_in.template sync<DevExeSpace>();
  ui_in.template modify<HostMemSpace>();
  ui_in.template sync<DevExeSpace>();
  bi_in.template modify<HostMemSpace>();
  bi_in.template sync<DevExeSpace>();

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &u0_hyd = pmbp->phydro->u0;
  auto &u0_mhd = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  // Initialize Hydro and MHD variables on device using precomputed C-shock solution
  // shksol indices refer to:  0=di, 1=dn, 2=vix, 3=vnx, 4=viy, 5=vny, 6=by
  par_for("pgen_cshock", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    u0_mhd(m,IDN,k,j,i) = shksol.d_view(0,i-is);
    u0_hyd(m,IDN,k,j,i) = shksol.d_view(1,i-is);

    u0_mhd(m,IM1,k,j,i) = shksol.d_view(0,i-is)*shksol.d_view(2,i-is);
    u0_hyd(m,IM1,k,j,i) = shksol.d_view(1,i-is)*shksol.d_view(3,i-is);

    u0_mhd(m,IM2,k,j,i) = shksol.d_view(0,i-is)*shksol.d_view(4,i-is);
    u0_hyd(m,IM2,k,j,i) = shksol.d_view(1,i-is)*shksol.d_view(5,i-is);

    u0_hyd(m,IM3,k,j,i) = 0.0;
    u0_mhd(m,IM3,k,j,i) = 0.0;

    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = shksol.d_view(6,i-is);
    b0.x3f(m,k,j,i) = 0.0;

    if (i==ie) b0.x1f(m,k,j,i+1) = 0.0;
    if (j==je) b0.x2f(m,k,j+1,i) = shksol.d_view(6,i-is);
    if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
  });

  return;
}
