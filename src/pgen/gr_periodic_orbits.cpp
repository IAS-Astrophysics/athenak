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
void EnergyConservationTest(HistoryData *hdata, Mesh *pm);
Real * E_init;
Real * all_diffs_;
ParameterInput * aux_pin;

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = EnergyConservationTest;
  aux_pin = pin;
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

  E_init = (Real *) malloc( npart*sizeof(Real) );
  all_diffs_ = (Real *) malloc( npart*sizeof(Real) );

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

    // Following is needed to ensure particle is being created
    // within the randomly chosen meshblock
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
    // Avoid generating particles that become either unbound or that fall into the black hole
    if (L < 4.0){
    	R_min = SQR(L)/2.0 - L/2.0*sqrt(SQR(L) - 12.0);
    } else {
	R_min = SQR(L)/4.0 + sqrt(SQR(L) - 16.0)/4.0;
    }
    R_max = SQR(L)/2.0 + L/2.0*sqrt(SQR(L) - 12.0);
    r = R_min + (R_max - R_min)*p/npart;
    th = equator ? M_PI/2.0 : th_min + (th_max - th_min)*rand_gen.frand();
    // All particles are initialized at y = 0 
    // Such that v_y is toroidal velocity

    // All particles are initialized with v_x = 0 (i.e. only "toroidal" velocity)
    pr(IPVX,p) = 0.0;
    pr(IPVY,p) = L/r; // For choice of setup v^y = v_y, though this is not true in general
    pr(IPVZ,p) = equator ? 0.0 : 2.0*max_init_vel*(rand_gen.frand()-0.5);
    rand_pool64.free_state(rand_gen);  // free state for use by other threads

    E_init[p] = 1.0 - 2.0/r + SQR(L)/SQR(r) - SQR(L)/SQR(r)/r + SQR(pr(IPVZ,p)); //Initial motion has no radial velocity

    pr(IPX,p) = r;
    pr(IPY,p) = 0.0;//r*sin(th)*sin(phi);
    pr(IPZ,p) = 0.0;//equator ? 0.0 : r*cos(th);

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

void EnergyConservationTest(HistoryData *pdata, Mesh *pm){
	// This function computes the energy at time "curr_time" for each particle
	// and compares to the energy at the beginning of the simulation.
	// This is used as a metric for the accuracy of the gr integration.
	
	  pdata->nhist = 2;
	  pdata->label[0] = "avg_rel_dEdt"; //rms difference in the energy between the initial condition and the current time
	  pdata->label[1] = "max_diff"; //sqrt of the maximum of the squared differences in energy

	  // Re-compute the initial positions
	  auto &mbsize = pm->pmb_pack->pmb->mb_size;
	  auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
	  auto &pi = pm->pmb_pack->ppart->prtcl_idata;
	  auto &npart = pm->pmb_pack->ppart->nprtcl_thispack;
	  auto gids = pm->pmb_pack->gids;
	  auto gide = pm->pmb_pack->gide;
	  int n_sqrt = int(sqrt(npart));

	  Real * E_i = E_init;
	  Real * all_diffs = all_diffs_;
	  Real curr_time = pm->time;

	  Real rel_dEdt = 0.0;
	  for (int ip = 0; ip<npart; ++ip){
		all_diffs[ip] = 0.0;
	  }
	  Kokkos::parallel_reduce("part_compare",Kokkos::RangePolicy<>(DevExeSpace(),0,(npart-1)),
			  KOKKOS_LAMBDA(const int p, Real &rel_dE) {

	    Real r = sqrt( SQR(pr(IPX,p)) + SQR(pr(IPY,p)) + SQR(pr(IPZ,p)) );
	    Real E = SQR(pr(IPVX,p)) + SQR(pr(IPVY,p)) + SQR(pr(IPVZ,p)) + 1.0 - 1.0/r;

	    // Initial energy vs. current energy
	    rel_dE = ( SQR(E) - SQR(E_i[p]) )/SQR(E_i[p]);
	    all_diffs[p] = ( SQR(E) - SQR(E_i[p]) )/SQR(E_i[p]);
	  }, Kokkos::Sum<Real>(rel_dEdt));

	  rel_dEdt = rel_dEdt/(npart) ;
	  Real ismax = -1.0;
	  for (int ip = 0; ip<npart; ++ip){
	  	if (all_diffs[ip] > ismax){
			ismax = all_diffs[ip];
		}
	  }
	  ismax = sqrt(ismax);
	  pdata->hdata[0] = rel_dEdt;
	  pdata->hdata[1] = ismax;
	  if (curr_time == aux_pin->GetOrAddReal("time", "tlim", 1.0)){
		std::cout << "Freeing auxiliary mallocs for particle test problem." << std::endl;
	  	free(E_init);
	  	free(all_diffs_);
	  }
	  return;
	  
}
