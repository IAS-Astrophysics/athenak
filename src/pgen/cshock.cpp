//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb.cpp
//  \brief Problem generator for turbulence

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::Turb_()
//  \brief Problem Generator for turbulence

void ProblemGenerator::UserProblem(ParameterInput *pin,const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Turbulence problem generator can only be run with Hydro and/or MHD, but no "
       << "<hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  Real gm_hyd = pmbp->phydro->peos->eos_data.gamma;
  Real gm1_hyd = gm_hyd - 1.0;
  Real gm_mhd = pmbp->pmhd->peos->eos_data.gamma;
  Real gm1_mhd = gm_mhd - 1.0;

  // read initial condition from input file
  Real B0_x = pin->GetOrAddReal("problem","B0_x",1.414213562373095);
  Real x0sh = pin->GetOrAddReal("problem","x0sh",0.0);

  // left state
  Real Bl_y = pin->GetOrAddReal("problem","Bl_y",1.414213562373095);
  Real dln = pin->GetOrAddReal("problem","dln",0.5);
  Real dli = pin->GetOrAddReal("problem","dli",0.01*dln);
  Real uln_x = pin->GetOrAddReal("problem","uln_x",5.0);
  Real uli_x = pin->GetOrAddReal("problem","uli_x",5.0);
  Real uln_y = pin->GetOrAddReal("problem","uln_y",0.0);
  Real uli_y = pin->GetOrAddReal("problem","uli_y",0.0);

  // set inflow state in BoundaryValues
  auto &u_in_hyd = pmbp->phydro->pbval_u->u_in;
  u_in_hyd.h_view(IDN,BoundaryFace::inner_x1) = dln;
  u_in_hyd.h_view(IM1,BoundaryFace::inner_x1) = dln*uln_x;
  u_in_hyd.h_view(IM2,BoundaryFace::inner_x1) = dln*uln_y;
  u_in_hyd.h_view(IEN,BoundaryFace::inner_x1) = dln/gm1_hyd + 0.5*dln*(uln_x*uln_x+uln_y*uln_y);
  auto &u_in_mhd = pmbp->pmhd->pbval_u->u_in;
  u_in_mhd.h_view(IDN,BoundaryFace::inner_x1) = dli;
  u_in_mhd.h_view(IM1,BoundaryFace::inner_x1) = dli*uli_x;
  u_in_mhd.h_view(IM2,BoundaryFace::inner_x1) = dli*uli_y;
  u_in_mhd.h_view(IEN,BoundaryFace::inner_x1) = dli/gm1_mhd + 0.5*dli*(uli_x*uli_x+uli_y*uli_y)
                                                     + 0.5*(B0_x*B0_x+Bl_y*Bl_y);
  auto &b_in_mhd = pmbp->pmhd->pbval_b->b_in;
  b_in_mhd.h_view(0,BoundaryFace::inner_x1) = B0_x;
  b_in_mhd.h_view(1,BoundaryFace::inner_x1) = Bl_y;
  b_in_mhd.h_view(2,BoundaryFace::inner_x1) = 0.0;

  // right state
  Real Br_y = pin->GetOrAddReal("problem","Br_y",3.4327);
  Real drn = pin->GetOrAddReal("problem","drn",0.9880);
  Real dri = pin->GetOrAddReal("problem","dri",0.01*drn);
  Real urn_x = pin->GetOrAddReal("problem","urn_x",2.5303);
  Real uri_x = pin->GetOrAddReal("problem","uri_x",1.1415);
  Real urn_y = pin->GetOrAddReal("problem","urn_y",2.5303);
  Real uri_y = pin->GetOrAddReal("problem","uri_y",1.1415);

  // set inflow state in BoundaryValues
  u_in_hyd.h_view(IDN,BoundaryFace::outer_x1) = drn;
  u_in_hyd.h_view(IM1,BoundaryFace::outer_x1) = drn*urn_x;
  u_in_hyd.h_view(IM2,BoundaryFace::outer_x1) = drn*urn_y;
  u_in_hyd.h_view(IEN,BoundaryFace::outer_x1) = drn/gm1_hyd + 0.5*drn*(urn_x*urn_x+urn_y*urn_y);
  u_in_mhd.h_view(IDN,BoundaryFace::outer_x1) = dri;
  u_in_mhd.h_view(IM1,BoundaryFace::outer_x1) = dri*uri_x;
  u_in_mhd.h_view(IM2,BoundaryFace::outer_x1) = dri*uri_y;
  u_in_mhd.h_view(IEN,BoundaryFace::outer_x1) = dri/gm1_mhd + 0.5*dri*(uri_x*uri_x+uri_y*uri_y)
                                                     + 0.5*(B0_x*B0_x+Br_y*Br_y);
  b_in_mhd.h_view(0,BoundaryFace::outer_x1) = B0_x;
  b_in_mhd.h_view(1,BoundaryFace::outer_x1) = Br_y;
  b_in_mhd.h_view(2,BoundaryFace::outer_x1) = 0.0;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;

    // Set initial conditions
    par_for("pgen_cshock", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      if (x1v <= x0sh) {
        u0(m,IDN,k,j,i) = dln;
        u0(m,IM1,k,j,i) = dln*uln_x;
        u0(m,IM2,k,j,i) = dln*uln_y;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = dln*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i);
        }
      } else {
        u0(m,IDN,k,j,i) = drn;
        u0(m,IM1,k,j,i) = drn*urn_x;
        u0(m,IM2,k,j,i) = drn*urn_y;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = dln*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i);
        }
      }
    });
  }

  // Initialize MHD variables ---------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;

    // Set initial conditions
    par_for("pgen_cshock", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      // initialize B
      b0.x1f(m,k,j,i) = B0_x;
      if (i==ie) {b0.x1f(m,k,j,i+1) = B0_x;}
      b0.x3f(m,k,j,i) = 0.0;
      if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}

      if (x1v <= x0sh) {
        u0(m,IDN,k,j,i) = dli;
        u0(m,IM1,k,j,i) = dli*uli_x;
        u0(m,IM2,k,j,i) = dli*uli_y;
        u0(m,IM3,k,j,i) = 0.0;

        b0.x2f(m,k,j,i) = Bl_y;
        if (j==je) {b0.x2f(m,k,j+1,i) = Bl_y;}
        Real bsq = B0_x*B0_x + Bl_y*Bl_y;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = dli*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i)+0.5*bsq;
        }
      } else {
        u0(m,IDN,k,j,i) = dri;
        u0(m,IM1,k,j,i) = dri*uri_x;
        u0(m,IM2,k,j,i) = dri*uri_y;
        u0(m,IM3,k,j,i) = 0.0;

        b0.x2f(m,k,j,i) = Br_y;
        if (j==je) {b0.x2f(m,k,j+1,i) = Br_y;}
        Real bsq = B0_x*B0_x + Br_y*Br_y;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = dri*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i)+0.5*bsq;
        }
      }
    });
  }

  // Initialize ion-neutral variables -------------------------
  if (pmbp->pionn != nullptr) {
    // MHD
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma; // TODO(@user): multiply by ionized density

    // Set initial conditions
    par_for("pgen_cshock_mhd", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      // initialize B 
      b0.x1f(m,k,j,i) = B0_x;
      if (i==ie) {b0.x1f(m,k,j,i+1) = B0_x;}
      b0.x3f(m,k,j,i) = 0.0;
      if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
      
      if (x1v <= x0sh) {
        u0(m,IDN,k,j,i) = dli;
        u0(m,IM1,k,j,i) = dli*uli_x;
        u0(m,IM2,k,j,i) = dli*uli_y;
        u0(m,IM3,k,j,i) = 0.0;
        
        b0.x2f(m,k,j,i) = Bl_y;
        if (j==je) {b0.x2f(m,k,j+1,i) = Bl_y;}
        Real bsq = B0_x*B0_x + Bl_y*Bl_y;
        if (eos.is_ideal) { 
          u0(m,IEN,k,j,i) = dli*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i)+0.5*bsq;
        }
      } else {
        u0(m,IDN,k,j,i) = dri;
        u0(m,IM1,k,j,i) = dri*uri_x;
        u0(m,IM2,k,j,i) = dri*uri_y;
        u0(m,IM3,k,j,i) = 0.0;
        
        b0.x2f(m,k,j,i) = Br_y;
        if (j==je) {b0.x2f(m,k,j+1,i) = Br_y;}
        Real bsq = B0_x*B0_x + Br_y*Br_y;
        if (eos.is_ideal) { 
          u0(m,IEN,k,j,i) = dri*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i)+0.5*bsq;
        }                   
      }
    });
    // Hydro
    auto &u0_ = pmbp->phydro->u0;
    EOS_Data &eos_ = pmbp->phydro->peos->eos_data;
    Real gm1_ = eos_.gamma - 1.0;
    Real p0_ = 1.0/eos_.gamma; // TODO(@user): multiply by neutral density

    // Set initial conditions
    par_for("pgen_cshock_hydro", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      if (x1v <= x0sh) {
        u0(m,IDN,k,j,i) = dln;
        u0(m,IM1,k,j,i) = dln*uln_x;
        u0(m,IM2,k,j,i) = dln*uln_y;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = dln*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i);
        }
      } else {
        u0(m,IDN,k,j,i) = drn;
        u0(m,IM1,k,j,i) = drn*urn_x;
        u0(m,IM2,k,j,i) = drn*urn_y;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = drn*p0/gm1 + 0.5*(u0(m,IM1,k,j,i)*u0(m,IM1,k,j,i)+
                            u0(m,IM2,k,j,i)*u0(m,IM2,k,j,i)+
                            u0(m,IM3,k,j,i)*u0(m,IM3,k,j,i))/u0(m,IDN,k,j,i);
        }
      }
    });
  }

  return;
}
