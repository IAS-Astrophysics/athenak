//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gca_gr.cpp
//  \brief
//  Largely based on work in  Bacchini et al. 2020 (https://doi.org/10.3847/1538-4365/abb604)

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "hamiltonian_gr.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include <Kokkos_Random.hpp> // For the gyrophase

//----------------------------------------------------------------------------------------
//! \fn  void GCAComputeDrifts
//!  \brief
//! Compute drift velocities
KOKKOS_INLINE_FUNCTION
void GCAComputeDrifts( const Real gupper[4][4], const Real ADM_upper[3][3], const Real * E, const Real * B, const Real B_norm, Real * H ){
	
	Real adm_det; 
	ComputeDeterminant3( ADM_upper, adm_det );
	//Compute ExB drift 
  // First need to convert electric field to normal frame
	// Vector product results in covariant vector
	Real E_beta[3] = {
		gupper[0][2]/gupper[0][0]*B[2] - B[1]*gupper[0][3]/gupper[0][0],
		gupper[0][3]/gupper[0][0]*B[0] - B[2]*gupper[0][1]/gupper[0][0],
		gupper[0][1]/gupper[0][0]*B[1] - B[0]*gupper[0][2]/gupper[0][0]
	};
	for (int i = 0; i < 3; ++i ){ E_beta[i] /= adm_det; }
	Real E_aux[3] = {0.0};
	E_aux[0] = ADM_upper[0][0]*E_beta[0] + ADM_upper[0][1]*E_beta[1] + ADM_upper[0][2]*E_beta[2];
	E_aux[1] = ADM_upper[1][0]*E_beta[0] + ADM_upper[1][1]*E_beta[1] + ADM_upper[1][2]*E_beta[2];
	E_aux[2] = ADM_upper[2][0]*E_beta[0] + ADM_upper[2][1]*E_beta[1] + ADM_upper[2][2]*E_beta[2];
	for (int i = 0; i<3; ++i) { E_aux[i] = E[i] - E_aux[i]; }

	//Compute vector product ExB
  // B is provided already normalized
	Real ExB[3] = {
		E_aux[1]*B[2] - B[1]*E_aux[2],
		E_aux[2]*B[0] - B[2]*E_aux[0],
		E_aux[0]*B[1] - B[0]*E_aux[1]
	};
	for (int i = 0; i < 3; ++i ){ ExB[i] /= adm_det; }
	for (int i = 0; i < 3; ++i ){ H[i] = ExB[i]/SQR(B_norm); }
	
	//Compute curvature drift

	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void GCAEquation_Position
//  \brief
// Update position
KOKKOS_INLINE_FUNCTION
void GCAEquation_Position( const Real gupper[4][4],  const Real v_par, const Real * v_drift,
				const Real * b, const Real Gamma, Real * H ){
	
	Real ADM[3][3];
	GetUpperAdmMetric( gupper, ADM );
	// Raise indeces on v_drift
  Real vup_drift[3] = {0.0};
	vup_drift[0] = ADM[0][0]*v_drift[0] + ADM[0][1]*v_drift[1] + ADM[0][2]*v_drift[2];
	vup_drift[1] = ADM[1][0]*v_drift[0] + ADM[1][1]*v_drift[1] + ADM[1][2]*v_drift[2];
	vup_drift[2] = ADM[2][0]*v_drift[0] + ADM[2][1]*v_drift[1] + ADM[2][2]*v_drift[2];

	// Push gyro-center position (hence no u_perp)
	for (int i=0; i<3; ++i) {
		H[i] = sqrt(-1.0/gupper[0][0])*(v_par*b[i]/Gamma + vup_drift[i]) + gupper[0][i+1]/gupper[0][0];
	}

	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void GCAElectricPush
//  \brief
// Electric field push in GCA.
KOKKOS_INLINE_FUNCTION
void GCAElectricPush(const Real * x_0, const Real * E, const Real * B, const Real B_norm, const Real spin, const Real q_o_m, Real * H){

	// Get metric components at new location x1,x2,x3
	Real glower[4][4], gupper[4][4], ADM[3][3]; // Metric 
								 // (remember: sqrt(-1/gupper[0][0]) = alpha)
	// Get metric components at new location x
	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], false, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM );
	Real adm_det; 
	ComputeDeterminant3( ADM, adm_det );

  // Project electric field along magnetic field
  // First need to convert electric field to normal frame
	// Vector product results in covariant vector
	Real E_beta[3] = {
		gupper[0][2]/gupper[0][0]*B[2] - B[1]*gupper[0][3]/gupper[0][0],
		gupper[0][3]/gupper[0][0]*B[0] - B[2]*gupper[0][1]/gupper[0][0],
		gupper[0][1]/gupper[0][0]*B[1] - B[0]*gupper[0][2]/gupper[0][0]
	};
	for (int i = 0; i < 3; ++i ){ E_beta[i] /= adm_det; }
	Real E_low[3] = {0.0};
	E_low[0] = glower[1][1]*E[0] + glower[1][2]*E[1] + glower[1][3]*E[2];
	E_low[1] = glower[2][1]*E[0] + glower[2][2]*E[1] + glower[2][3]*E[2];
	E_low[2] = glower[3][1]*E[0] + glower[3][2]*E[1] + glower[3][3]*E[2];
	for (int i = 0; i<3; ++i) { E_low[i] -= E_beta[i]; }
	// Compute scalar product of magnetic field and electric field in normal frame
	Real E_par = 0.0;
	for (int i =0; i<3; ++i) { E_par += E_low[i]*B[i]/B_norm; }
	// Need to update covariant covector
	*H = q_o_m*E_par;

	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void :GCAEquation_Velocity
//!  \brief
//! Handles geodesic terms. Returns a 3D covariant velocity, to be projected on B afterwards 
//! Kept separate in case in the future one wants to include the electromagnetic terms in the iterations
KOKKOS_INLINE_FUNCTION
void GCAEquation_Velocity( const Real * x_init, const Real * x_eval, const Real * v_init, const Real * v_eval, const Real x_step,
			const Real spin, const Real it_tol, Real * H ){

	// HamiltonEquation_Velocity sets H[i] = 0 internally, no need to do this before hand
	HamiltonEquation_Velocity( x_init, x_eval, v_init, v_eval, x_step, spin, it_tol, H );

	return;
}

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::GCAIterations
//  \brief
// Main loop for the gryo-center approximation algorithm
void Particles::GCAIterations( const Real dt ){
	auto &pr = prtcl_rdata;
	auto &pi = prtcl_idata;
	auto &b0_ = pmy_pack->pmhd->b0;
	auto &e0_ = pmy_pack->pmhd->efld;
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real it_tol = iter_tolerance;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const int it_max = max_iter;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;
	const Real &q_over_m = charge_over_mass;
	auto gids = pmy_pack->gids;
	auto &indcs = pmy_pack->pmesh->mb_indcs;
	auto &mbsize = pmy_pack->pmb->mb_size;
	const Real x_step = 1.0E-07;
	const Real v_step = 1.0E-07;
	Real avg_iter = 0.0;
	Real prtcl_mass = 1.0;

	Kokkos::Random_XorShift64_Pool<> prtcl_rand(gids);

	Kokkos::parallel_reduce("part_gca",Kokkos::RangePolicy<>(DevExeSpace(),0,(nprtcl_thispack-1)),
		KOKKOS_LAMBDA(const int p, Real &aux_n_iter) {
		//par_for("part_fullgr",DevExeSpace(),0,(nprtcl_thispack-1),
		//KOKKOS_LAMBDA(const int p) {

		// Note that procedure is very similar to full_gr pusher, except here the velocity is easier to handle
		// Iterate per particle such that those that converge quicker don't go through as many iterations
		// Initialize iteration variables
		Real x_init[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
		Real x_prev[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
		Real v_drift[3] = {0.0};
		Real v_init[3] = {0.0};
		Real v_grad[3] = {0.0};
		const Real v_par_init = pr(IPVX,p);
		const Real mag_mom = pr(IPVY,p);
		Real v_par_eval = pr(IPVX,p);
		Real v_par_grad = pr(IPVX,p);
		Real v_par_prev = pr(IPVX,p);
		//u is always contravariant. Iteration variables
		Real RHS_eval_vpar, RHS_eval_v[3], RHS_eval_x[3]; // parallel velocity is now scalar quantity
		Real Jacob[3][3], inv_Jacob[3][3];
		Real x_grad[3];
		Real RHS_grad_1[3], RHS_grad_2[3];
		int n_iter = 0;

		//Construct a ``full velocity vector'' starting from the parallel velocity, drift and the magnetic momentum
		Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric 
									 // (remember: sqrt(-1/gupper[0][0]) = alpha)

		// Get metric components at location x
		ComputeMetricAndInverse(x_init[0],x_init[1],x_init[2], false, spin, glower, gupper); 
		GetUpperAdmMetric( gupper, ADM_upper );

		int m = pi(PGID,p) - gids;
		Real E[3], B[3], b[3], b_low[3];
		InterpolateFields( x_init, b0_, e0_, mbsize, indcs, m, E, B );
		//Compute B norm
		Real B_norm = 0.0;
		for (int i1 = 0; i1<3; ++i1) {
			for (int i2 = 0; i2<3; ++i2) {
				B_norm += glower[i1+1][i2+1]*B[i1]*B[i2];
			}
		}	
		B_norm = sqrt(B_norm);
		// Get drifts at this position
		GCAComputeDrifts( gupper, ADM_upper, E, B, B_norm, v_drift );
		Real drift_g = 0.0;
		for (int i1 = 0; i1<3; ++i1) {
			for (int i2 = 0; i2<3; ++i2) {
				drift_g += ADM_upper[i1][i2]*v_drift[i2]*v_drift[i1];
			}	
		}	
		drift_g = 1/sqrt(1-drift_g);

		// Get perpendicular velocity
		Real v_perp_mod = sqrt( 2.0*B_norm*drift_g*mag_mom/prtcl_mass );
		Real Gamma = drift_g*sqrt( 1 + SQR(v_par_init) + SQR(v_perp_mod) );
		// Get perpendicular vector
    // phase kept constant throughout iteration
		Real perp_low[3], perp_up[3];
		auto prtcl_phase_rnd = prtcl_rand.get_state();
		Real gphase = prtcl_phase_rnd.frand();
		prtcl_rand.free_state(prtcl_phase_rnd);
		for (int i = 0; i < 3; ++i ){ b[i] = B[i]/B_norm; }
		perp_low[0] = cos(gphase);
		perp_low[1] = sin(gphase);
		perp_low[2] = - ( (b[0]*perp_low[0] + b[1]*perp_low[1])/b[2] );
		Real perp_norm = 0.0;
		for (int i = 0; i<3; ++i){
			for (int j = 0; j<3; ++j){ perp_norm += ADM_upper[i][j]*perp_low[i]*perp_low[j]; }
		}
		for (int i = 0; i<3; ++i){ perp_low[i] /= sqrt( perp_norm ); }
		b_low[0] = glower[1][1]*b[0] + glower[1][2]*b[1] + glower[1][3]*b[2];
		b_low[1] = glower[2][1]*b[0] + glower[2][2]*b[1] + glower[2][3]*b[2];
		b_low[2] = glower[3][1]*b[0] + glower[3][2]*b[1] + glower[3][3]*b[2];
		
		for (int i = 0; i < 3; ++i ){ v_init[i] = v_par_init*b_low[i] + v_perp_mod*perp_low[i] + v_drift[i]*Gamma; }
		// Use v_drift in order to no define another array
		v_drift[0] = ADM_upper[0][0]*v_init[0] + ADM_upper[0][1]*v_init[1] + ADM_upper[0][2]*v_init[2];
		v_drift[1] = ADM_upper[1][0]*v_init[0] + ADM_upper[1][1]*v_init[1] + ADM_upper[1][2]*v_init[2];
		v_drift[2] = ADM_upper[2][0]*v_init[0] + ADM_upper[2][1]*v_init[1] + ADM_upper[2][2]*v_init[2];
		Gamma = Gamma/sqrt(-1.0/gupper[0][0]);
		Real x_eval[3] = {pr(IPX,p)+v_init[0]/Gamma*dt, pr(IPY,p)+v_init[1]/Gamma*dt, pr(IPZ,p)+v_init[2]/Gamma*dt};

		// First half-push with the electric field
		GCAElectricPush( x_init, E, B, B_norm, spin, q_over_m, &RHS_eval_vpar );
		v_par_eval += RHS_eval_vpar*dt/2.0;
		
		// Start iterating
		// Using Newton method, thus computing the Jacobian at each iteration
		do{
			
		++n_iter;
		                                    
		for (int i = 0; i<3; ++i){ x_grad[i] = (x_eval[i] + x_init[i])/2.0; }
		v_par_grad = (v_par_eval + v_par_init)/2.0;
		
		//Construct a ``full velocity vector'' starting from the parallel velocity and the magnetic momentum
		ComputeMetricAndInverse(x_grad[0],x_grad[1],x_grad[2], false, spin, glower, gupper); 
		InterpolateFields( x_grad, b0_, e0_, mbsize, indcs, m, E, B );
		//Compute B norm
		B_norm = 0.0;
		for (int i1 = 0; i1<3; ++i1) {
			for (int i2 = 0; i2<3; ++i2) {
				B_norm += glower[i1+1][i2+1]*B[i1]*B[i2];
			}
		}	
		B_norm = sqrt(B_norm);
		// Get drifts at this position
		GetUpperAdmMetric( gupper, ADM_upper );
		// Drifts are returned as covariant
		GCAComputeDrifts( gupper, ADM_upper, E, B, B_norm, v_drift );
		drift_g = 0.0;
		for (int i1 = 0; i1<3; ++i1) {
			for (int i2 = 0; i2<3; ++i2) {
				drift_g += ADM_upper[i1][i2]*v_drift[i2]*v_drift[i1];
			}	
		}	
		drift_g = 1/sqrt(1-drift_g);
		v_perp_mod = sqrt( 2.0*B_norm*drift_g*mag_mom/prtcl_mass );
		Gamma = drift_g*sqrt( 1 + SQR(v_par_grad) + SQR(v_perp_mod) );
		// Get perpendicular velocity at this position
		for (int i = 0; i < 3; ++i ){ b[i] = B[i]/B_norm; }
		perp_low[2] = - ( (b[0]*perp_low[0] + b[1]*perp_low[1])/b[2] ); //Keep the random phase constant throughout iterations
		perp_norm = 0.0;
		for (int i = 0; i<3; ++i){
			for (int j = 0; j<3; ++j){ perp_norm += ADM_upper[i][j]*perp_low[i]*perp_low[j]; }
		}
		for (int i = 0; i<3; ++i){ perp_low[i] /= sqrt( perp_norm ); }
		b_low[0] = glower[1][1]*b[0] + glower[1][2]*b[1] + glower[1][3]*b[2];
		b_low[1] = glower[2][1]*b[0] + glower[2][2]*b[1] + glower[2][3]*b[2];
		b_low[2] = glower[3][1]*b[0] + glower[3][2]*b[1] + glower[3][3]*b[2];

		GCAEquation_Position(gupper, v_par_grad, v_drift, b, Gamma, RHS_eval_x);
		for (int i = 0; i<3; ++i){ v_grad[i] = v_par_grad*b_low[i] + v_perp_mod*perp_low[i] + v_drift[i]*Gamma; }
		GCAEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, spin, it_tol, RHS_eval_v);
    // Project 3D co-variant RHS for geodesic push onto the magnetic field
    RHS_eval_vpar = 0.0;
    for (int i = 0; i < 3; ++i ) { RHS_eval_vpar += b[i]*RHS_eval_v[i]; } 

		// Store for next iteration
		for (int i=0; i<3; ++i) { x_prev[i] = x_eval[i]; }
		v_par_prev = v_par_eval;
		for (int i=0; i<3; ++i) { x_eval[i] = x_init[i] + RHS_eval_x[i]*dt; }
		v_par_eval = v_par_init + RHS_eval_vpar*dt;

		}while(
			n_iter < it_max
			&& ( sqrt(SQR(x_eval[0] - x_prev[0]) + SQR(x_eval[1] - x_prev[1]) + SQR(x_eval[2] - x_prev[2])) > it_tol
			|| fabs(v_par_eval - v_par_prev) > it_tol )
				 );

		InterpolateFields( x_eval, b0_, e0_, mbsize, indcs, m, E, B );
		//Compute B norm
		B_norm = 0.0;
		for (int i1 = 0; i1<3; ++i1) {
			for (int i2 = 0; i2<3; ++i2) {
				B_norm += glower[i1+1][i2+1]*B[i1]*B[i2];
			}
		}	
		B_norm = sqrt(B_norm);
		// Second half-push with the electric field
		GCAElectricPush( x_eval, E, B, B_norm, spin, q_over_m, &RHS_eval_vpar );
		v_par_eval += RHS_eval_vpar*dt/2.0;

		// Done with iterations, update ``true'' values
		pr(IPVX,p) = v_par_eval;
		pr(IPX,p) = x_eval[0];
		if (multi_d) { pr(IPY,p) = x_eval[1]; }
		if (three_d) { pr(IPZ,p) = x_eval[2]; }
		aux_n_iter += n_iter;
	}, Kokkos::Sum<Real>(avg_iter));
	average_iteration_number += avg_iter / nprtcl_thispack;
	return;
}

} // namespace particles
