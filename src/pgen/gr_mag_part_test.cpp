//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_random.cpp
//! \brief Problem generator that initializes random particle positions and velocities.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "globals.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for random particle positions/velocities

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  //user_hist_func = LarmorMotionErrors;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Magnetized particles test requires <particles> block in input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Magnetized particles test requires <mhd> block in input file"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for the kernel
  auto &mbsize = pmy_mesh_->pmb_pack->pmb->mb_size;
  auto &pr = pmy_mesh_->pmb_pack->ppart->prtcl_rdata;
  auto &pi = pmy_mesh_->pmb_pack->ppart->prtcl_idata;
  auto &npart = pmy_mesh_->pmb_pack->ppart->nprtcl_thispack;
  auto gids = pmy_mesh_->pmb_pack->gids;
  auto gide = pmy_mesh_->pmb_pack->gide;

  Real max_init_vel = pin->GetOrAddReal("problem", "max_init_vel", 1.0);
  Real prtcl_mass = pin->GetOrAddReal("particles", "mass", 1.0E-10);
  Real prtcl_charge = pin->GetOrAddReal("particles", "charge", 1.0);
  Real mesh_x1_min = fmin(fabs(pin->GetReal("mesh", "x1max")), fabs(pin->GetReal("mesh", "x1min")));
  Real mesh_x2_min = fmin(fabs(pin->GetReal("mesh", "x2max")), fabs(pin->GetReal("mesh", "x2min")));
  Real mesh_x3_min = fmin(fabs(pin->GetReal("mesh", "x3max")), fabs(pin->GetReal("mesh", "x3min")));
  // initialize particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("part_init",DevExeSpace(),0,(npart-1),
  KOKKOS_LAMBDA(const int p){
    Real r,th,phi;
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    pi(PGID,p) = gids + m;

    Real x1min = mbsize.d_view(m).x1min;
    Real x2min = mbsize.d_view(m).x2min;
    Real x3min = mbsize.d_view(m).x3min;
    Real x1max = mbsize.d_view(m).x1max;
    Real x2max = mbsize.d_view(m).x2max;
    Real x3max = mbsize.d_view(m).x3max;
    Real R_min_plane2 = SQR( fmin(fabs(x1min), fabs(x1max)) ) + SQR( fmin(fabs(x2min), fabs(x2max)) );
    Real R_max_plane2 = SQR( fmax(fabs(x1min), fabs(x1max)) ) + SQR( fmax(fabs(x2min), fabs(x2max)) );
    Real R_min = sqrt( R_min_plane2 + SQR( fmin(fabs(x3min), fabs(x3max)) ) );
    Real R_max = sqrt( R_max_plane2 + SQR( fmax(fabs(x3min), fabs(x3max)) ) );

    Real den = fabs(x3min) > fabs(x3max) ? x3min : x3max;
    Real th_min = atan2( sqrt(R_min_plane2) , den );
    den = fabs(x3min) < fabs(x3max) ? x3min : x3max;
    Real th_max = atan2( sqrt(R_max_plane2) , den );
    Real num = fabs(x2min) < fabs(x2max) ? x2min : x2max;
    den = fabs(x1min) > fabs(x1max) ? x1min : x1max;
    Real phi_min = atan2( num , den );
    num = fabs(x2min) > fabs(x2max) ? x2min : x2max;
    den = fabs(x1min) < fabs(x1max) ? x1min : x1max;
    Real phi_max = atan2( num , den );
    // Avoid generating particles too close to the BH
    R_min = fmax(2.0,R_min);
    // R_max might exceed x_i_max because of diagonals
    R_max = fmin( R_max, mesh_x1_min );
    R_max = fmin( R_max, mesh_x2_min );
    R_max = fmin( R_max, mesh_x3_min );
    r = R_min + (R_max - R_min)*rand_gen.frand();
    th = th_min + (th_max - th_min)*rand_gen.frand();
    phi = phi_min + (phi_max - phi_min)*rand_gen.frand();

    pr(IPVX,p) = 2.0*max_init_vel*(rand_gen.frand()-0.5);
    pr(IPVY,p) = 2.0*max_init_vel*(rand_gen.frand()-0.5);
    pr(IPVZ,p) = 2.0*max_init_vel*(rand_gen.frand()-0.5);
    rand_pool64.free_state(rand_gen);  // free state for use by other threads
    
    pr(IPX,p) = r*sin(th)*cos(phi);
    pr(IPY,p) = r*sin(th)*sin(phi);
    pr(IPZ,p) = r*cos(th);

    pr(IPM,p) = prtcl_mass;
    pr(IPC,p) = prtcl_charge;
  });
  
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmy_mesh_->pmb_pack->nmb_thispack;
  auto &b0_ = pmy_mesh_->pmb_pack->pmhd->b0;
  auto &e0_ = pmy_mesh_->pmb_pack->pmhd->efld;
  auto &bcc0_ = pmy_mesh_->pmb_pack->pmhd->bcc0;
  auto &w0_ = pmy_mesh_->pmb_pack->pmhd->w0;

  Real B_strength = pin->GetOrAddReal("problem", "b0_strength", 1.0E-8);
  // Init vertical field
  par_for("b0_init",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    // int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    w0_(m,IDN,k,j,i) = 1.0E+10; // Ensure magnetosonic wave won't limit time step
    w0_(m,IEN,k,j,i) = 0.0;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;
    // Magnetic field is vertical in local frame
    b0_.x1f(m,k,j,i) = 0.0;
    b0_.x2f(m,k,j,i) = 0.0;
    b0_.x3f(m,k,j,i) = B_strength;
    e0_.x1e(m,k,j,i) = 0.0;
    e0_.x2e(m,k,j,i) = 0.0;
    e0_.x3e(m,k,j,i) = 0.0;
    bcc0_(m,IBX,k,j,i) = 0.0;
    bcc0_(m,IBY,k,j,i) = 0.0;
    bcc0_(m,IBZ,k,j,i) = B_strength;
    if (i==ie) {
   	 b0_.x1f(m,k,j,i+1) = 0.0;
   	 e0_.x1e(m,k,j,i+1) = 0.0;
    }
    if (j==je) {
	    b0_.x2f(m,k,j+1,i) = 0.0;
	    e0_.x2e(m,k,j+1,i) = 0.0;
    }
    if (k==ke) {
	    b0_.x3f(m,k+1,j,i) = B_strength;
	    e0_.x3e(m,k+1,j,i) = 0.0;
    }
  });
  // Need to initialize all MHD properties to ensure pmhd->newdt is computed as expected
  auto &u0_ = pmy_mesh_->pmb_pack->pmhd->u0;
  pmy_mesh_->pmb_pack->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke); 
  // set timestep (which will remain constant for entire run
  // Assumes uniform mesh (no SMR or AMR)
  // Assumes velocities normalized to one, so dt=min(dx)
  Real &dtnew_ = pmy_mesh_->pmb_pack->ppart->dtnew;
  dtnew_ = std::min(mbsize.h_view(0).dx1, mbsize.h_view(0).dx2);
  dtnew_ = std::min(dtnew_, mbsize.h_view(0).dx3);
  dtnew_ *= pin->GetOrAddReal("time", "cfl_number", 0.8);

  return;
}
