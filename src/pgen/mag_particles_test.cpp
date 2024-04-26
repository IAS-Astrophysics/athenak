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

void LarmorMotionErrors(HistoryData *hdata, Mesh *pm);
Real * x_init;
Real * v_init;
Real * x_aux;
Real * v_aux;
Real * all_diffs_;
ParameterInput * aux_pin;

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = LarmorMotionErrors;
  aux_pin = pin;
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
  int n_sqrt = int(sqrt(npart));
  x_init = (Real *) malloc( 3*npart*sizeof(Real) );
  v_init = (Real *) malloc( 3*npart*sizeof(Real) );
  x_aux = (Real *) malloc( 3*npart*sizeof(Real) );
  v_aux = (Real *) malloc( 3*npart*sizeof(Real) );
  all_diffs_ = (Real *) malloc( npart*sizeof(Real) );
  Real * x = x_init;
  Real * v = v_init;
  // initialize particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("part_init",DevExeSpace(),0,(npart-1),
  KOKKOS_LAMBDA(const int p) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    // choose parent MeshBlock randomly
    int m = static_cast<int>(rand_gen.frand()*(gide - gids + 1.0));
    pi(PGID,p) = gids + m;
    pr(IPVZ,p) = (rand_gen.frand()-0.5);
    rand_pool64.free_state(rand_gen);  // free state for use by other threads
    pr(IPM,p) = prtcl_mass;
    pr(IPC,p) = prtcl_charge;

    // Basically avoid having to deal with periodic boundary conditions
    Real x1min_actual = (mbsize.d_view(m).x1min <= 0) ? 0.75*mbsize.d_view(m).x1min : 1.25*mbsize.d_view(m).x1min;
    Real x2min_actual = (mbsize.d_view(m).x2min <= 0) ? 0.75*mbsize.d_view(m).x2min : 1.25*mbsize.d_view(m).x2min;
    Real x3min_actual = (mbsize.d_view(m).x3min <= 0) ? 0.75*mbsize.d_view(m).x3min : 1.25*mbsize.d_view(m).x3min;
    Real x1max_actual = (mbsize.d_view(m).x1max <= 0) ? 1.25*mbsize.d_view(m).x1max : 0.75*mbsize.d_view(m).x1max;
    Real x2max_actual = (mbsize.d_view(m).x2max <= 0) ? 1.25*mbsize.d_view(m).x2max : 0.75*mbsize.d_view(m).x2max;
    Real x3max_actual = (mbsize.d_view(m).x3max <= 0) ? 1.25*mbsize.d_view(m).x3max : 0.75*mbsize.d_view(m).x3max;
    pr(IPX,p) = (x1max_actual + x1min_actual)/2.0;
    pr(IPY,p) = (x2max_actual + x2min_actual)/2.0;
    pr(IPZ,p) = (x3max_actual + x3min_actual)/2.0;
    x[3*p] = pr(IPX,p);
    x[3*p+1] = pr(IPY,p);
    x[3*p+2] = pr(IPZ,p);

    pr(IPVX,p) = max_init_vel*(p+0.01)/npart*(x1max_actual - x1min_actual);
    pr(IPVY,p) = max_init_vel*(p+0.01)/npart*(x2max_actual - x2min_actual);
    pr(IPVZ,p) *= 2.5E-3;
    v[3*p] = pr(IPVX,p);
    v[3*p+1] = pr(IPVY,p);
    v[3*p+2] = pr(IPVZ,p);
  });

  /***
  std::cout << pr(IPX,25) << " " << pr(IPY,25) << " " << pr(IPZ,25) << " " << x[75] << " " << x[76] << " " << x[77] << " " << std::endl;
  ***/
  
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

void LarmorMotionErrors(HistoryData *pdata, Mesh *pm){
	// This function computes the expected position for each particle
	// at t=tlim using the analytical expression for helical/Larmor motion.
	// This is used as a metric for the accuracy of the Boris pusher.
	
	  pdata->nhist = 3;
	  pdata->label[0] = "rms_diff"; //rms difference between the analytically computed position and the numerical one
	  pdata->label[1] = "max_diff"; //sqrt of the maximum of the squared differences in position
	  pdata->label[2] = "avg_rel_dEdt"; //rms difference in the kinetic energy between the initial condition and the current time
					//Only perpendicular motion

	  // Re-compute the initial positions
	  auto &mbsize = pm->pmb_pack->pmb->mb_size;
	  auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
	  auto &pi = pm->pmb_pack->ppart->prtcl_idata;
	  auto &npart = pm->pmb_pack->ppart->nprtcl_thispack;
	  auto gids = pm->pmb_pack->gids;
	  auto gide = pm->pmb_pack->gide;
	  int n_sqrt = int(sqrt(npart));

	  Real * x = x_init;
	  Real * v = v_init;
	  Real * x1 = x_aux;
	  Real * v1 = v_aux;
	  Real * all_diffs = all_diffs_;
	  Real B_strength = aux_pin->GetOrAddReal("problem", "b0_strength", 1.0E-8);
	  Real curr_time = pm->time;

	  Real sfac = (curr_time == 0.0) ? 0.0 : 1.0; //First diagnostic should not be evolved

	  Real rms_diff = 0.0;
	  Real rel_dEdt = 0.0;
	  Kokkos::parallel_reduce("part_compare",Kokkos::RangePolicy<>(DevExeSpace(),0,(npart-1)),
			  KOKKOS_LAMBDA(const int p, Real &rms_pos, Real &rel_dE) {

	    //int p = pi(PTAG,p);
	    Real prtcl_mass = pr(IPM,p);
	    Real prtcl_charge = pr(IPC,p);
	    // In idealized case Lorentz gamma should be constant throughout simulation
	    // and there's no acceleration, thus compute with initial velocities
	    Real g_Lor = 1.0/sqrt(1.0 - SQR(v[3*p]) - SQR(v[3*p+1]) - SQR(v[3*p+2]));
	    Real v_mod = sqrt( SQR(v[3*p]) + SQR(v[3*p+1]) +SQR(v[3*p+2]) );
	    Real ang_to_b = atan2( sqrt( SQR(v[3*p]) + SQR(v[3*p+1]) ) , v[3*p+2] );
	    Real v_perp = ( v_mod*sin(ang_to_b) );
	    Real v_par = v[3*p+2];
	    Real init_phase = - atan2(v[3*p+1],v[3*p]);
	    Real rho = (prtcl_mass*v_perp*g_Lor)/(prtcl_charge*B_strength); //Larmor radius
	    Real omega = prtcl_charge*B_strength/(prtcl_mass); //Larmor frequency
	    Real aux = 0.0;

	    x1[3*p] = x[3*p] + sfac*rho*sin( init_phase + omega*curr_time );
	    x1[3*p+1] = x[3*p+1] + sfac*rho*cos( init_phase + omega*curr_time );
	    x1[3*p+2] = x[3*p+2] + sfac*v_par*curr_time;
	    
	    v1[3*p] = v_perp*cos( init_phase + omega*curr_time );
	    v1[3*p+1] = - v_perp*sin( init_phase + omega*curr_time );
	    v1[3*p+2] = v_par;// + sfac*v_par*curr_time;

	    // Initial kinetic energy vs. current kinetic energy (pure magnetic field shouldn't do any work)
	    aux = ( SQR(v1[3*p]) + SQR(v1[3*p+1]) + SQR(v1[3*p+2]) ) - ( SQR(v[3*p]) + SQR(v[3*p+1]) + SQR(v[3*p+2]) );
	    aux /= ( SQR(v[3*p]) + SQR(v[3*p+1]) + SQR(v[3*p+2]) );
	    rel_dE += aux;
	    rms_pos += ( SQR(pr(IPX,p) - x1[3*p]) + SQR(pr(IPY,p) - x1[3*p+1]) + SQR(pr(IPZ,p) - x1[3*p+2]) )/SQR(rho);
	    all_diffs[p] = ( SQR(pr(IPX,p) - x1[3*p]) + SQR(pr(IPY,p) - x1[3*p+1]) + SQR(pr(IPZ,p) - x1[3*p+2]) )/SQR(rho);
	  }, Kokkos::Sum<Real>(rms_diff), Kokkos::Sum<Real>(rel_dEdt));

	  rms_diff = sqrt( rms_diff/(npart-1) );
	  rel_dEdt = rel_dEdt/(npart) ;
	  Real ismax = -1.0;
	  for (int ip = 0; ip<npart; ++ip){
	  	if (all_diffs[ip] > ismax){
			ismax = all_diffs[ip];
		}
	  }
	  ismax = sqrt(ismax);
	  /***
	  std::cout << "On rank " << global_variable::my_rank << " at time " << curr_time << " rel_dEdt: " << rel_dEdt << std::endl;
	  std::cout << "On rank " << global_variable::my_rank << " at time " << curr_time << " rms_diff: " << rms_diff << std::endl;
	  std::cout << "On rank " << global_variable::my_rank << " at time " << curr_time << " max_diff: " << ismax << std::endl;
	  std::cout << pr(IPVX,25) << " " << pr(IPVY,25) << " " << pr(IPVZ,25) << " " << pr(IPVX,65) << " " << pr(IPVY,65) << " " << pr(IPVZ,65) << " " << std::endl;
	  std::cout << v1[75] << " " << v1[76] << " " << v1[77] << " " << v1[195] << " " << v1[196] << " " << v1[197] << " " << std::endl;
	  std::cout << "=====================================" << std::endl;
	  std::cout << pr(IPX,25) << " " << pr(IPY,25) << " " << pr(IPZ,25) << " " << pr(IPX,65) << " " << pr(IPY,65) << " " << pr(IPZ,65) << " " << std::endl;
	  std::cout << x1[75] << " " << x1[76] << " " << x1[77] << " " << x1[195] << " " << x1[196] << " " << x1[197] << " " << std::endl;
	  ***/
	  pdata->hdata[0] = rms_diff;
	  pdata->hdata[1] = ismax;
	  pdata->hdata[2] = rel_dEdt;
	  if (curr_time == aux_pin->GetOrAddReal("time", "tlim", 1.0)){
		std::cout << "Freeing auxiliary mallocs for particle test problem." << std::endl;
	  	free(x_init);
	  	free(v_init);
	  	free(x_aux);
	  	free(v_aux);
	  	free(all_diffs_);
	  }
	  return;
	  
}
