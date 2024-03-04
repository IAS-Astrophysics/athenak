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

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for random particle positions/velocities

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
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
  // initialize particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("part_init",DevExeSpace(),0,(npart-1),
  KOKKOS_LAMBDA(const int p) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    pi(PGID,p) = gids + m;
    pr(IPM,p) = prtcl_mass;
    pr(IPC,p) = prtcl_charge;

    Real rand = rand_gen.frand();
    pr(IPX,p) = (1. - rand)*mbsize.d_view(m).x1min + rand*mbsize.d_view(m).x1max;
    pr(IPX,p) = fmin(pr(IPX,p),mbsize.d_view(m).x1max);
    pr(IPX,p) = fmax(pr(IPX,p),mbsize.d_view(m).x1min);

    rand = rand_gen.frand();
    pr(IPY,p) = (1. - rand)*mbsize.d_view(m).x2min + rand*mbsize.d_view(m).x2max;
    pr(IPY,p) = fmin(pr(IPY,p),mbsize.d_view(m).x2max);
    pr(IPY,p) = fmax(pr(IPY,p),mbsize.d_view(m).x2min);

    rand = rand_gen.frand();
    pr(IPZ,p) = (1. - rand)*mbsize.d_view(m).x3min + rand*mbsize.d_view(m).x3max;
    pr(IPZ,p) = fmin(pr(IPZ,p),mbsize.d_view(m).x3max);
    pr(IPZ,p) = fmax(pr(IPZ,p),mbsize.d_view(m).x3min);

    pr(IPVX,p) = max_init_vel*2.0*(rand_gen.frand() - 0.5);
    pr(IPVY,p) = max_init_vel*2.0*(rand_gen.frand() - 0.5);
    pr(IPVZ,p) = max_init_vel*2.0*(rand_gen.frand() - 0.5);

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmy_mesh_->pmb_pack->nmb_thispack;
  auto &b0_ = pmy_mesh_->pmb_pack->pmhd->b0;
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
    b0_.x1f(m,k,j,i) = 0.0;
    b0_.x2f(m,k,j,i) = 0.0;
    b0_.x3f(m,k,j,i) = B_strength;
    bcc0_(m,IBX,k,j,i) = 0.0;
    bcc0_(m,IBY,k,j,i) = 0.0;
    bcc0_(m,IBZ,k,j,i) = B_strength;
    if (i==ie) { b0_.x1f(m,k,j,i+1) = 0.0; }
    if (j==je) { b0_.x2f(m,k,j+1,i) = 0.0; }
    if (k==ke) { b0_.x3f(m,k+1,j,i) = B_strength; }
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

  return;
}
