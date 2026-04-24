//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lpaw_paniso.cpp
//  \brief Problem generator for linearly polarized anisotropic-pressure Alfven wave

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for LPAW test with pressure anisotropy

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  // read problem parameters from input file
  Real amp   = pin->GetReal("problem","amp");
  Real pp0    = pin->GetOrAddReal("problem","pp0",1.0);         //background pressure
  Real pr0  = pin->GetOrAddReal("problem","pr0",1.0);         //background pressure anisotropy
  std::string eq_state = pin->GetString("mhd","eos");

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD
  Real gm1;
  int nfluid, nscalars;
  if (pmbp->phydro != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "This test requires MHD" << std::endl;
    exit(EXIT_FAILURE);
  } else if (pmbp->pmhd != nullptr) {
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->pmhd->nmhd;
    nscalars = pmbp->pmhd->nscalars;
  }
  if (pmbp->padm != nullptr) {
    gm1 = 1.0;
  }
  auto &w0_ = (pmbp->phydro != nullptr)? pmbp->phydro->w0 : pmbp->pmhd->w0;

  // initialize primitive variables
  par_for("pgen_prim", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    
    // Lorentz factor (needed to initializve 4-velocity in SR)
    Real u00 = 1.0;

    Real dens,vx,vy,vz,scal;

    //set up primitives
    dens = 1.0;
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;

    // set primitives in newtonian MHD
    w0_(m,IDN,k,j,i) = dens;
    w0_(m,IVX,k,j,i) = u00*vx;
    w0_(m,IVY,k,j,i) = u00*vy;
    w0_(m,IVZ,k,j,i) = u00*vz;
    if (eq_state.compare("cgl") == 0) {
      w0_(m,IPP,k,j,i) = pp0;
      w0_(m,IPR,k,j,i) = pr0*(1+amp*sin(2.*M_PI*x1v));
    }  else {
      w0_(m,IPR,k,j,i) = pr0*(1+amp*sin(2.*M_PI*x1v));
    }
  });

  // initialize magnetic fields
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_b0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1f    = LeftEdgeX  (i-is, nx1, x1min, x1max);
    Real x1v    = CellCenterX(i-is, nx1, x1min, x1max);
      
    b0.x1f(m,k,j,i) = 1.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = 0.0;
    if (i==ie) b0.x1f(m,k,j,i+1) = 1.0;
    if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
    if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
    bcc0(m,IBX,k,j,i) = 1.0;
    bcc0(m,IBY,k,j,i) = 0.0;
    bcc0(m,IBZ,k,j,i) = 0.0;
  });
  
  
  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    if (pmbp->phydro != nullptr) {
      auto &u0_ = pmbp->phydro->u0;
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &u0_ = pmbp->pmhd->u0;
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  }

  return;
}
