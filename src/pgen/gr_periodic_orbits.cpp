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
              << "GR particles test requires <particles> block in input file"
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

  Real max_init_vel = pin->GetOrAddReal("problem", "max_init_vel", 0.9);
  Real L = pin->GetOrAddReal("problem", "angular_momentum", 0.9);
  bool equator = pin->GetOrAddBoolean("problem", "equatorial", false);
  Real prtcl_mass = pin->GetOrAddReal("particles", "mass", 1.0);
  Real prtcl_charge = pin->GetOrAddReal("particles", "charge", 0.0);
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
    
    Real E = max_init_vel*(1.0 + 0.015*rand_gen.frand());

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
    R_min = fmax(5.0,R_min);
    // R_max might exceed x_i_max because of diagonals
    R_max = 15.0;
    r = L/sqrt(2.0*E); //R_min + (R_max - R_min)*rand_gen.frand();
    th = equator ? M_PI/2.0 : th_min + (th_max - th_min)*rand_gen.frand();
    // All particles are initialized at y = 0 
    phi = 0.0;//phi_min + (phi_max - phi_min)*rand_gen.frand();

    // All particles are initialized with v_x = 0 
    pr(IPVX,p) = 0.0;
    pr(IPVY,p) = sqrt(2.0*E);
    pr(IPVZ,p) = equator ? 0.0 : 2.0*max_init_vel*(rand_gen.frand()-0.5);
    rand_pool64.free_state(rand_gen);  // free state for use by other threads
    // Check particles are not faster than light
    Real v_mod = sqrt(SQR(pr(IPVX,p)) + SQR(pr(IPVY,p)) + SQR(pr(IPVZ,p)));
    if (v_mod >= 1){
    	pr(IPVX,p) /= (v_mod*1.1);
        pr(IPVY,p) /= (v_mod*1.1);
        pr(IPVZ,p) /= (v_mod*1.1);
    }
    pr(IPX,p) = r*sin(th)*cos(phi);
    pr(IPY,p) = r*sin(th)*sin(phi);
    pr(IPZ,p) = equator ? 0.0 : r*cos(th);

    pr(IPM,p) = prtcl_mass;
    pr(IPC,p) = prtcl_charge;
  });
  // set timestep (which will remain constant for entire run
  // Assumes uniform mesh (no SMR or AMR)
  // Assumes velocities normalized to one, so dt=min(dx)
  Real &dtnew_ = pmy_mesh_->pmb_pack->ppart->dtnew;
  dtnew_ = std::min(mbsize.h_view(0).dx1, mbsize.h_view(0).dx2);
  dtnew_ = std::min(dtnew_, mbsize.h_view(0).dx3);
  dtnew_ *= pin->GetOrAddReal("time", "cfl_number", 0.8);

  return;
}
