//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cshock.cpp
//! \brief problem generator for C-shock test of two-fluid MHD. Solves ODE on host to
//! compute C-shock profile, then initializes this on the grid. Error function calculates
//! difference between final and initial solution to test whether code holds profile.
//! Shocks may be initialized propagating in x1-, x2-, or x3-directions and on grids with
//! multiple MeshBlocks, but does not work with SMR or AMR.
//! Works for both perpendicular and oblique C-shocks

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

struct TwoFluidVars {
  Real di, dn;
  Real vix, vnx;
  Real viy, vny;
  Real bx, by;
};

// Function to compute RHS of ODEs
// v[0]=vix, v[1]=vnx, v[2]=viy, v[3]=vny, derivative use same indexing
void RHS(TwoFluidVars init, Real v[4], Real alpha, Real cis, Real cns, Real dvdx[4]) {
  Real di = init.di*init.vix/v[0];
  Real dn = init.dn*init.vnx/v[1];
  Real bx = init.bx;
  Real by;
  if (bx==0.0) {  // perpendicular shock
    by = init.by*di/init.di;
  } else {        // oblique shock
    by = init.bx*v[2]/v[0];
  }
// equations derived in S4 of ZEUS-2F workbook
// See also Toth, ApJ 425, 171 (1994), eqs 4.2
  dvdx[0] = -alpha*dn*v[0]*(v[0]-v[1])/(SQR(v[0]) - SQR(cis) - SQR(by)/di);
  dvdx[1] =  alpha*di*v[1]*(v[0]-v[1])/(SQR(v[1]) - SQR(cns));
  dvdx[2] = (alpha*dn*v[0]*(v[2]-v[3]) + (bx*by/di)*dvdx[0])/(SQR(v[0]) - SQR(bx)/di);
  dvdx[3] = alpha*di*(v[2]-v[3])/v[1];
  return;
}

// function to compute errors in solution at end of run
void CShockErrors(ParameterInput *pin, Mesh *pm);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CShock()
//! \brief Problem Generator for steady C-shocks

void ProblemGenerator::CShock(ParameterInput *pin, const bool restart) {
  // set error function
  pgen_final_func = CShockErrors;
  if (restart) return;

  // Check physics is set properly
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

  // parse shock direction: {1,2,3} -> {x1,x2,x3}
  int shk_dir = pin->GetInteger("problem","shock_dir");
  if (shk_dir < 1 || shk_dir > 3) {
    // Invaild input value for shk_dir
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "shock_dir=" <<shk_dir<< " must be either 1,2, or 3" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read problem parameters
  TwoFluidVars init;
  init.di = pin->GetReal("problem", "di0");
  init.dn = pin->GetReal("problem", "dn0");
  init.vix = pin->GetReal("problem", "vix0");
  init.vnx = pin->GetReal("problem", "vnx0");
  init.viy = pin->GetReal("problem", "viy0");
  init.vny = pin->GetReal("problem", "vny0");
  init.bx = pin->GetReal("problem", "bx0");
  init.by = pin->GetReal("problem", "by0");
  Real pert = pin->GetOrAddReal("problem", "pert", 1.0e-4);

  // Fluid properties
  Real alpha = pmbp->pionn->drag_coeff;
  Real cns   = pmbp->phydro->peos->eos_data.iso_cs;
  Real cis   = pmbp->pmhd->peos->eos_data.iso_cs;

  // Use RK4 to integrate C-shock profile on host using (NFACT)x finer mesh
  // v[0] = ion x-velocity
  // v[1] = neutral x-velocity
  // v[2] = ion y-velocity
  // v[3] = neutral y-velocity
  const int NFACT=10;
  Mesh *pm  = pmy_mesh_;
  Real xmin,dxshk;
  int npts;
  if (shk_dir == 1) {
    npts = pm->mesh_indcs.nx1;
    xmin = pm->mesh_size.x1min;
    dxshk = (pm->mesh_size.x1max - pm->mesh_size.x1min) / static_cast<Real>((NFACT)*npts);
  } else if (shk_dir == 2) {
    npts = pm->mesh_indcs.nx2;
    xmin = pm->mesh_size.x2min;
    dxshk = (pm->mesh_size.x2max - pm->mesh_size.x2min) / static_cast<Real>((NFACT)*npts);
  } else {
    npts = pm->mesh_indcs.nx3;
    xmin = pm->mesh_size.x3min;
    dxshk = (pm->mesh_size.x3max - pm->mesh_size.x3min) / static_cast<Real>((NFACT)*npts);
  }
  HostArray1D<Real> xshk("xshk", (NFACT)*npts);
  HostArray1D<TwoFluidVars> soln("soln", (NFACT)*npts);
  xshk(0) = xmin + (0.5*dxshk);
  soln(0).vix = init.vix - pert;
  soln(0).vnx = init.vnx;
  soln(0).viy = init.viy;
  soln(0).vny = init.vny;
  for (int n=0; n<((NFACT)*npts - 1); ++n) {
    Real v[4],dvdx1[4],dvdx2[4],dvdx3[4],dvdx4[4];
    v[0] = soln(n).vix;
    v[1] = soln(n).vnx;
    v[2] = soln(n).viy;
    v[3] = soln(n).vny;
    RHS(init,v,alpha,cis,cns,dvdx1);

    v[0] = soln(n).vix + dvdx1[0]*dxshk/2.0;
    v[1] = soln(n).vnx + dvdx1[1]*dxshk/2.0;
    v[2] = soln(n).viy + dvdx1[2]*dxshk/2.0;
    v[3] = soln(n).vny + dvdx1[3]*dxshk/2.0;
    RHS(init,v,alpha,cis,cns,dvdx2);

    v[0] = soln(n).vix + dvdx2[0]*dxshk/2.0;
    v[1] = soln(n).vnx + dvdx2[1]*dxshk/2.0;
    v[2] = soln(n).viy + dvdx2[2]*dxshk/2.0;
    v[3] = soln(n).vny + dvdx2[3]*dxshk/2.0;
    RHS(init,v,alpha,cis,cns,dvdx3);

    v[0] = soln(n).vix + dvdx3[0]*dxshk;
    v[1] = soln(n).vnx + dvdx3[1]*dxshk;
    v[2] = soln(n).viy + dvdx3[2]*dxshk;
    v[3] = soln(n).vny + dvdx3[3]*dxshk;
    RHS(init,v,alpha,cis,cns,dvdx4);

    xshk(n+1) = xshk(n) + dxshk;
    soln(n+1).vix = soln(n).vix + dxshk*(dvdx1[0] +2.0*(dvdx2[0]+dvdx3[0]) +dvdx4[0])/6.0;
    soln(n+1).vnx = soln(n).vnx + dxshk*(dvdx1[1] +2.0*(dvdx2[1]+dvdx3[1]) +dvdx4[1])/6.0;
    soln(n+1).viy = soln(n).viy + dxshk*(dvdx1[2] +2.0*(dvdx2[2]+dvdx3[2]) +dvdx4[2])/6.0;
    soln(n+1).vny = soln(n).vny + dxshk*(dvdx1[3] +2.0*(dvdx2[3]+dvdx3[3]) +dvdx4[3])/6.0;
  }

  // bin solution into DualArray at resolution of grid, sync to device
  // shksol indices refer to:  0=di, 1=dn, 2=vix, 3=vnx, 4=viy, 5=vny, 6=by
  DualArray1D<TwoFluidVars> shksol("shksol",npts);
  for (int n=0; n<npts; ++n) {
    shksol.h_view(n).di  = 0.0;
    shksol.h_view(n).dn  = 0.0;
    shksol.h_view(n).vix = 0.0;
    shksol.h_view(n).vnx = 0.0;
    shksol.h_view(n).viy = 0.0;
    shksol.h_view(n).vny = 0.0;
    shksol.h_view(n).bx  = 0.0;
    shksol.h_view(n).by  = 0.0;
    for (int m=0; m<(NFACT); ++m) {
      shksol.h_view(n).di  += init.di*init.vix/soln(m + (NFACT)*n).vix;
      shksol.h_view(n).dn  += init.dn*init.vnx/soln(m + (NFACT)*n).vnx;
      shksol.h_view(n).vix += soln(m + (NFACT)*n).vix;
      shksol.h_view(n).vnx += soln(m + (NFACT)*n).vnx;
      shksol.h_view(n).viy += soln(m + (NFACT)*n).viy;
      shksol.h_view(n).vny += soln(m + (NFACT)*n).vny;
      shksol.h_view(n).bx  += init.bx;
      if (init.bx==0.0) {  // perpendicular shock
        shksol.h_view(n).by  += init.by*init.vix/soln(m + (NFACT)*n).vix;
      } else {             // oblique shock
        shksol.h_view(n).by  += init.bx*soln(m + (NFACT)*n).viy/soln(m + (NFACT)*n).vix;
      }
    }
    shksol.h_view(n).di  /= static_cast<Real> (NFACT);
    shksol.h_view(n).dn  /= static_cast<Real> (NFACT);
    shksol.h_view(n).vix /= static_cast<Real> (NFACT);
    shksol.h_view(n).vnx /= static_cast<Real> (NFACT);
    shksol.h_view(n).viy /= static_cast<Real> (NFACT);
    shksol.h_view(n).vny /= static_cast<Real> (NFACT);
    shksol.h_view(n).bx  /= static_cast<Real> (NFACT);
    shksol.h_view(n).by  /= static_cast<Real> (NFACT);
  }
  shksol.template modify<HostMemSpace>();
  shksol.template sync<DevExeSpace>();

  // set inflow state in BoundaryValues depending on shock direction, sync to device
  auto un_in = pmbp->phydro->pbval_u->u_in;
  auto ui_in = pmbp->pmhd->pbval_u->u_in;
  auto bi_in = pmbp->pmhd->pbval_b->b_in;
  if (shk_dir == 1) {
    un_in.h_view(IDN,BoundaryFace::inner_x1) = init.dn;
    ui_in.h_view(IDN,BoundaryFace::inner_x1) = init.di;
    un_in.h_view(IM1,BoundaryFace::inner_x1) = init.dn*init.vnx;
    ui_in.h_view(IM1,BoundaryFace::inner_x1) = init.di*init.vix;
    un_in.h_view(IM2,BoundaryFace::inner_x1) = init.dn*init.vny;
    ui_in.h_view(IM2,BoundaryFace::inner_x1) = init.di*init.viy;
    bi_in.h_view(IBX,BoundaryFace::inner_x1) = init.bx;
    bi_in.h_view(IBY,BoundaryFace::inner_x1) = init.by;
    bi_in.h_view(IBZ,BoundaryFace::inner_x1) = 0.0;
  } else if (shk_dir == 2) {
    un_in.h_view(IDN,BoundaryFace::inner_x2) = init.dn;
    ui_in.h_view(IDN,BoundaryFace::inner_x2) = init.di;
    un_in.h_view(IM2,BoundaryFace::inner_x2) = init.dn*init.vnx;
    ui_in.h_view(IM2,BoundaryFace::inner_x2) = init.di*init.vix;
    un_in.h_view(IM3,BoundaryFace::inner_x2) = init.dn*init.vny;
    ui_in.h_view(IM3,BoundaryFace::inner_x2) = init.di*init.viy;
    bi_in.h_view(IBY,BoundaryFace::inner_x2) = init.bx;
    bi_in.h_view(IBZ,BoundaryFace::inner_x2) = init.by;
    bi_in.h_view(IBX,BoundaryFace::inner_x2) = 0.0;
  } else {
    un_in.h_view(IDN,BoundaryFace::inner_x3) = init.dn;
    ui_in.h_view(IDN,BoundaryFace::inner_x3) = init.di;
    un_in.h_view(IM3,BoundaryFace::inner_x3) = init.dn*init.vnx;
    ui_in.h_view(IM3,BoundaryFace::inner_x3) = init.di*init.vix;
    un_in.h_view(IM1,BoundaryFace::inner_x3) = init.dn*init.vny;
    ui_in.h_view(IM1,BoundaryFace::inner_x3) = init.di*init.viy;
    bi_in.h_view(IBZ,BoundaryFace::inner_x3) = init.bx;
    bi_in.h_view(IBX,BoundaryFace::inner_x3) = init.by;
    bi_in.h_view(IBY,BoundaryFace::inner_x3) = 0.0;
  }
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

  // compute solution in u0 register when computing initial conditions
  // compute solution in u1 register when computing errors
  auto &u0_hyd = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;
  auto &u0_mhd = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
  auto &b0 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;

  // Initialize Hydro and MHD variables on device using precomputed C-shock solution
  // shksol indices refer to:  0=di, 1=dn, 2=vix, 3=vnx, 4=viy, 5=vny, 6=by
  if (shk_dir==1) {
    // Calculate index-offset of MeshBlocks in shksol array
    DualArray1D<int> ioff("offset",pmbp->nmb_thispack);
    for (int m=0; m<(pmbp->nmb_thispack); m++) {
      int igid = pmbp->gids + m;
      LogicalLocation lloc=pm->lloc_eachmb[igid];
      ioff.h_view(m) = lloc.lx1*(pm->mb_indcs.nx1);
    }
    // sync with device
    ioff.template modify<HostMemSpace>();
    ioff.template sync<DevExeSpace>();

    par_for("pgen_cshock", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      int io = ioff.d_view(m);
      u0_mhd(m,IDN,k,j,i) = shksol.d_view(io+i-is).di;
      u0_hyd(m,IDN,k,j,i) = shksol.d_view(io+i-is).dn;
      u0_mhd(m,IM1,k,j,i) = shksol.d_view(io+i-is).di*shksol.d_view(io+i-is).vix;
      u0_hyd(m,IM1,k,j,i) = shksol.d_view(io+i-is).dn*shksol.d_view(io+i-is).vnx;
      u0_mhd(m,IM2,k,j,i) = shksol.d_view(io+i-is).di*shksol.d_view(io+i-is).viy;
      u0_hyd(m,IM2,k,j,i) = shksol.d_view(io+i-is).dn*shksol.d_view(io+i-is).vny;
      u0_hyd(m,IM3,k,j,i) = 0.0;
      u0_mhd(m,IM3,k,j,i) = 0.0;
      b0.x1f(m,k,j,i) = init.bx;
      b0.x2f(m,k,j,i) = shksol.d_view(io+i-is).by;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = init.bx;
      if (j==je) b0.x2f(m,k,j+1,i) = shksol.d_view(io+i-is).by;
      if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
    });
  } else if (shk_dir==2) {
    // Calculate index-offset of MeshBlocks in shksol array
    DualArray1D<int> joff("offset",pmbp->nmb_thispack);
    for (int m=0; m<(pmbp->nmb_thispack); m++) {
      int igid = pmbp->gids + m;
      LogicalLocation lloc=pm->lloc_eachmb[igid];
      joff.h_view(m) = lloc.lx2*(pm->mb_indcs.nx2);
    }
    // sync with device
    joff.template modify<HostMemSpace>();
    joff.template sync<DevExeSpace>();

    par_for("pgen_cshock", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      int jo = joff.d_view(m);
      u0_mhd(m,IDN,k,j,i) = shksol.d_view(jo+j-js).di;
      u0_hyd(m,IDN,k,j,i) = shksol.d_view(jo+j-js).dn;
      u0_mhd(m,IM2,k,j,i) = shksol.d_view(jo+j-js).di*shksol.d_view(jo+j-js).vix;
      u0_hyd(m,IM2,k,j,i) = shksol.d_view(jo+j-js).dn*shksol.d_view(jo+j-js).vnx;
      u0_mhd(m,IM3,k,j,i) = shksol.d_view(jo+j-js).di*shksol.d_view(jo+j-js).viy;
      u0_hyd(m,IM3,k,j,i) = shksol.d_view(jo+j-js).dn*shksol.d_view(jo+j-js).vny;
      u0_hyd(m,IM1,k,j,i) = 0.0;
      u0_mhd(m,IM1,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = init.bx;
      b0.x3f(m,k,j,i) = shksol.d_view(jo+j-js).by;
      b0.x1f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = 0.0;
      if (j==je) b0.x2f(m,k,j+1,i) = init.bx;
      if (k==ke) b0.x3f(m,k+1,j,i) = shksol.d_view(jo+j-js).by;
    });
  } else {
    // Calculate index-offset of MeshBlocks in shksol array
    DualArray1D<int> koff("offset",pmbp->nmb_thispack);
    for (int m=0; m<(pmbp->nmb_thispack); m++) {
      int igid = pmbp->gids + m;
      LogicalLocation lloc=pm->lloc_eachmb[igid];
      koff.h_view(m) = lloc.lx3*(pm->mb_indcs.nx3);
    }
    // sync with device
    koff.template modify<HostMemSpace>();
    koff.template sync<DevExeSpace>();

    par_for("pgen_cshock", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      int ko = koff.d_view(m);
      u0_mhd(m,IDN,k,j,i) = shksol.d_view(ko+k-ks).di;
      u0_hyd(m,IDN,k,j,i) = shksol.d_view(ko+k-ks).dn;
      u0_mhd(m,IM3,k,j,i) = shksol.d_view(ko+k-ks).di*shksol.d_view(ko+k-ks).vix;
      u0_hyd(m,IM3,k,j,i) = shksol.d_view(ko+k-ks).dn*shksol.d_view(ko+k-ks).vnx;
      u0_mhd(m,IM1,k,j,i) = shksol.d_view(ko+k-ks).di*shksol.d_view(ko+k-ks).viy;
      u0_hyd(m,IM1,k,j,i) = shksol.d_view(ko+k-ks).dn*shksol.d_view(ko+k-ks).vny;
      u0_hyd(m,IM2,k,j,i) = 0.0;
      u0_mhd(m,IM2,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = init.bx;
      b0.x1f(m,k,j,i) = shksol.d_view(ko+k-ks).by;
      b0.x2f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = shksol.d_view(ko+k-ks).by;
      if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
      if (k==ke) b0.x3f(m,k+1,j,i) = init.bx;
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CShockErrors_()
//! \brief Computes errors in cshock solution by calling initialization function
//! again to compute initial condictions, and then calling generic error output function
//! that subtracts current solution from ICs, and outputs errors to file. Error will be
//! small only if shock remains steady

void CShockErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->CShock(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}
