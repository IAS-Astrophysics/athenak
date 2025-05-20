//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boris_gr.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "hamiltonian_gr.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::BorisStep
//  \brief
//Provide dt and only_v as input parameter in order to be able to use this function
//also for half-steps in the full_gr pusher
//Largely implemented following Ripperda et al. 2018 (https://doi.org/10.3847/1538-4365/aab114)
void Particles::BorisStep( const Real dt, const bool only_v ){
	
	auto &npart = nprtcl_thispack;
	auto &pi = prtcl_idata;
	auto &pr = prtcl_rdata;
	auto &b0_ = pmy_pack->pmhd->b0;
	auto &e0_ = pmy_pack->pmhd->efld;
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;
	const Real &q_over_m = charge_over_mass;
	auto gids = pmy_pack->gids;
	auto &indcs = pmy_pack->pmesh->mb_indcs;
	auto &mbsize = pmy_pack->pmb->mb_size;

	// First half-step in space
	par_for("part_boris",DevExeSpace(),0,(npart-1),
	KOKKOS_LAMBDA(const int p) {
      
		// Contravariant and co-variant 4-velocities in the normal frame
		// To have compatibility with GR the velocity stored should be the covariant one
		Real u_cov[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
		Real u_con[3];
		Real x[3] =  {pr(IPX,p), pr(IPY,p), pr(IPZ,p)}; // Position.
		Real g_Lor;
		// Get metric components at new location x1,x2,x3
		Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric 
									 // (remember: sqrt(-1/gupper[0][0]) = alpha)
		ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
		// Compute 3x3 ADM spatial metric from metric 
		GetUpperAdmMetric( gupper, ADM_upper );
		// Raise indeces of u_cov
		for (int i1 = 0; i1 < 3; ++i1 ){ 
			u_con[i1] = 0.0;
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			u_con[i1] += ADM_upper[i1][i2]*u_cov[i2];
			}
		}
		//Use definition of the Lorentz factor in ADM formalism
		g_Lor = ADM_upper[0][0]*SQR(u_cov[0]) + ADM_upper[1][1]*SQR(u_cov[1]) + ADM_upper[2][2]*SQR(u_cov[2])
			+ 2.0*ADM_upper[0][1]*u_cov[0]*u_cov[1] + 2.0*ADM_upper[0][2]*u_cov[0]*u_cov[2] + 2.0*ADM_upper[1][2]*u_cov[1]*u_cov[2];
		// In principle the 1.0 should be replaced by a 0 if
		// the particle is massless, but I don't know of 
		// any massless particle that can interact with an 
		// electromagnetic field (unless one goes into quantum mechanics)
		g_Lor = sqrt(1.0 + g_Lor)*sqrt(-gupper[0][0]);

		x[0] = pr(IPX,p) + dt/(2.0)*(u_con[0]/g_Lor + gupper[0][1]/gupper[0][0]) ;
		if (multi_d) { x[1] = pr(IPY,p) + dt/(2.0)*(u_con[1]/g_Lor + gupper[0][2]/gupper[0][0]) ; }
		if (three_d) { x[2] = pr(IPZ,p) + dt/(2.0)*(u_con[2]/g_Lor + gupper[0][3]/gupper[0][0]) ; }

		int m = pi(PGID,p) - gids;
		Real uE[3]; //Evolution of the velocity due to the electric field (first half).
		Real uB[3]; //Evolution of the velocity due to the magnetic field.
		Real E[3], B[3];
		InterpolateFields( x, b0_, e0_, mbsize, indcs, m, E, B );

		// Get metric components at new location x
		ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
		GetUpperAdmMetric( gupper, ADM_upper );
		// Determinant of metric needed for vector products
		Real adm_det; 
		ComputeDeterminant3( ADM_upper, adm_det );
		// Determinant need is that of covariant metric
		adm_det = 1.0/adm_det;

		// Electric field is stored in coordinate frame, need to combine with B to operate on velocity in normal frame
		// Vector product results in covariant vector
		Real E_beta[3] = {
			- gupper[0][2]/gupper[0][0]*B[2] + B[1]*gupper[0][3]/gupper[0][0],
			- gupper[0][3]/gupper[0][0]*B[0] + B[2]*gupper[0][1]/gupper[0][0],
			- gupper[0][1]/gupper[0][0]*B[1] + B[0]*gupper[0][2]/gupper[0][0]
		};
		for (int i = 0; i < 3; ++i ){ E_beta[i] *= adm_det; }
		Real vec_ut[3] = {0.0};
		//Raise indeces to contravariant
		for (int i1 = 0; i1 < 3; ++i1 ){ 
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			vec_ut[i1] += ADM_upper[i1][i2]*E_beta[i2];
			}
		}
		for (int i = 0; i < 3; ++i ){ E[i] -= vec_ut[i]; }

		// Push 4-velocity with electric field in normal frame
		uE[0] = u_con[0] + dt*q_over_m/(2.0)*E[0];
					if (multi_d) { uE[1] = u_con[1] + dt*q_over_m/(2.0)*E[1]; }
					if (three_d) { uE[2] = u_con[2] + dt*q_over_m/(2.0)*E[2]; }

		//Intermediate Lorentz gamma factor in normal frame
		g_Lor = glower[1][1]*SQR(uE[0]) + glower[2][2]*SQR(uE[1]) + glower[3][3]*SQR(uE[2])
			+ 2.0*glower[1][2]*uE[0]*uE[1] + 2.0*glower[1][3]*uE[0]*uE[2] + 2.0*glower[2][3]*uE[1]*uE[2];
		g_Lor = sqrt(1.0 + g_Lor);

		// Rotation of velocity due to magnetic field done in 2 steps
		// i.e. Boris algorithm
		Real mod_t_sqr = 0.0;
		Real t[3];
		for (int i1 = 0; i1 < 3; ++i1 ){ t[i1] = B[i1]*q_over_m/(2.0*g_Lor)*dt; }
		for (int i1 = 0; i1 < 3; ++i1 ){ 
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			mod_t_sqr += glower[i1+1][i2+1]*t[i1]*t[i2];
			}
		}

		// Save the vector product of u and t 
		// Vector product results in covariant vector
		Real vec_ut_cov[3] = {
		uE[1]*t[2] - uE[2]*t[1],
		uE[2]*t[0] - uE[0]*t[2],
		uE[0]*t[1] - uE[1]*t[0]
		};
		//Raise indeces to contravariant
		for (int i1 = 0; i1 < 3; ++i1 ){ 
			vec_ut[i1] = 0.0;
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			vec_ut[i1] += ADM_upper[i1][i2]*vec_ut_cov[i2];
			}
		}
		// Used a vector product, correct for volume
		for (int i = 0; i < 3; ++i ){ vec_ut[i] *= adm_det; }
		// Re-use arrays
		vec_ut_cov[0] = (uE[1] + vec_ut[1])*t[2] - (uE[2] + vec_ut[2])*t[1];
		vec_ut_cov[1] = (uE[2] + vec_ut[2])*t[0] - (uE[0] + vec_ut[0])*t[2];
		vec_ut_cov[2] = (uE[0] + vec_ut[0])*t[1] - (uE[1] + vec_ut[1])*t[0];
		for (int i1 = 0; i1 < 3; ++i1 ){ 
			vec_ut[i1] = 0.0;
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			vec_ut[i1] += ADM_upper[i1][i2]*vec_ut_cov[i2];
			}
		}
		for (int i = 0; i < 3; ++i ){ vec_ut[i] *= adm_det; }

		// Finalize roation
		uB[0] = uE[0] + 2.0/(1.0+mod_t_sqr)*( vec_ut[0] );
		if (multi_d) { uB[1] = uE[1] + 2.0/(1.0+mod_t_sqr)*( vec_ut[1] ); }
		if (three_d) { uB[2] = uE[2] + 2.0/(1.0+mod_t_sqr)*( vec_ut[2] ); }

		//Second half-step with shifted electric field
		uE[0] = uB[0] + dt*q_over_m/(2.0)*E[0];
		if (multi_d) { uE[1] = uB[1] + dt*q_over_m/(2.0)*E[1]; }
		if (three_d) { uE[2] = uB[2] + dt*q_over_m/(2.0)*E[2]; }

		for (int i1 = 0; i1 < 3; ++i1 ){ 
			u_cov[i1] = 0.0;
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			u_cov[i1] += glower[i1+1][i2+1]*uE[i2];
			}
		}
		// Finally update velocity in normal frame
		pr(IPVX,p) = u_cov[0];
		pr(IPVY,p) = u_cov[1];
		pr(IPVZ,p) = u_cov[2];

		if (!only_v){
		//Final Lorentz gamma factor
		g_Lor = ADM_upper[0][0]*SQR(uE[0]) + ADM_upper[1][1]*SQR(uE[1]) + ADM_upper[2][2]*SQR(uE[2])
			+ 2.0*ADM_upper[0][1]*uE[0]*uE[1] + 2.0*ADM_upper[0][2]*uE[0]*uE[2] + 2.0*ADM_upper[1][2]*uE[1]*uE[2];
		g_Lor = sqrt(1.0 + g_Lor)*sqrt(-gupper[0][0]);
		// Raise indeces of u_cov to update position
		for (int i1 = 0; i1 < 3; ++i1 ){ 
			u_con[i1] = 0.0;
			for (int i2 = 0; i2 < 3; ++i2 ){ 
			u_con[i1] += ADM_upper[i1][i2]*u_cov[i2];
			}
		}
		pr(IPX,p) = x[0] + dt/(2.0)*(u_con[0]/g_Lor + gupper[0][1]/gupper[0][0]) ;
					if (multi_d) { pr(IPY,p) = x[1] + dt/(2.0)*(u_con[1]/g_Lor + gupper[0][2]/gupper[0][0]) ; }
					if (three_d) { pr(IPZ,p) = x[2] + dt/(2.0)*(u_con[2]/g_Lor + gupper[0][3]/gupper[0][0]) ; }
		}
	});
	return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Particles::GeodesicIterations
//  \brief
//Provide dt as input parameter in order to be able to use this function
//also for half-steps
//Largely implemented following Bacchini et al. 2020 (https://doi.org/10.3847/1538-4365/abb604)
void Particles::GeodesicIterations( const Real dt ){
	auto &pr = prtcl_rdata;
	const Real it_tol = iter_tolerance;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const int it_max = max_iter;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;
	const Real x_step = 1.0E-07;
	const Real v_step = 1.0E-07;
	Real avg_iter = 0.0;

	Kokkos::parallel_reduce("part_fullgr",Kokkos::RangePolicy<>(DevExeSpace(),0,(nprtcl_thispack-1)),
		KOKKOS_LAMBDA(const int p, Real &aux_n_iter) {
	//par_for("part_fullgr",DevExeSpace(),0,(nprtcl_thispack-1),
	//KOKKOS_LAMBDA(const int p) {

		// Iterate per particle such that those that converge quicker don't go through as many iterations
		// Initialize iteration variables
		Real x_init[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
		Real v_init[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
		Real x_eval[3] = {pr(IPX,p)+pr(IPVX,p)*dt, pr(IPY,p)+pr(IPVY,p)*dt, pr(IPZ,p)+pr(IPVZ,p)*dt};
		Real v_eval[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
		Real x_prev[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
		Real v_prev[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
		//u is always contravariant. Iteration variables
		Real RHS_eval_v[3], RHS_eval_x[3]; 
		Real Jacob[3][3], inv_Jacob[3][3];
		Real x_grad[3], v_grad[3];
		Real RHS_grad_1[3], RHS_grad_2[3];
		int n_iter = 0;
		Real step_fac = 1.0;

		// Start iterating
		// Using Newton method, thus computing the Jacobian at each iteration
		do{
			
		++n_iter;
		if (n_iter > 5){ step_fac = 1E+3; }

		HamiltonEquation_Position(x_init, x_eval, v_init, v_eval, spin, RHS_eval_x);
		HamiltonEquation_Velocity(x_init, x_eval, v_init, v_eval, x_step, spin, it_tol, RHS_eval_v);

		// First Jacobian for position
		// Variation along x
		x_grad[0] = x_eval[0] + x_step/step_fac;
		x_grad[1] = x_eval[1]; x_grad[2] = x_eval[2];
		HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_1);
		x_grad[0] = x_eval[0] - x_step/step_fac;
		HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_2);
		for (int i=0; i<3; ++i) { Jacob[0][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step/step_fac); }
		Jacob[0][0] += 1.0; // Diagonal terms
		// Variation along y
		x_grad[1] = x_eval[1] + x_step/step_fac;
		x_grad[0] = x_eval[0]; x_grad[2] = x_eval[2];
		HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_1);
		x_grad[1] = x_eval[1] - x_step/step_fac;
		HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_2);
		for (int i=0; i<3; ++i) { Jacob[1][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step/step_fac); }
		Jacob[1][1] += 1.0; // Diagonal terms
		// Variation along z
		x_grad[2] = x_eval[2] + x_step/step_fac;
		x_grad[0] = x_eval[0]; x_grad[1] = x_eval[1];
		HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_1);
		x_grad[2] = x_eval[2] - x_step/step_fac;
		HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, spin, RHS_grad_2);
		for (int i=0; i<3; ++i) { Jacob[2][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step/step_fac); }
		Jacob[2][2] += 1.0; // Diagonal terms
		ComputeInverseMatrix3( Jacob, inv_Jacob );

		// Store values for use with velocity Jacobian
		for (int i=0; i<3; ++i) { x_grad[i] = x_eval[i]; }

		for (int i=0; i<3; ++i){
			for (int j=0; j<3; ++j){ x_eval[i] -= inv_Jacob[j][i]*(x_grad[j] - x_init[j] - RHS_eval_x[j]*dt); }
		}

		// Then Jacobian for velocity
		// Variation along x
		// Not that the velocity here is covariant, thus derivatives along 
		// a given velocity direction result in "upper" indeces
		// and the lower indeces are provided by the rest function itself
		v_grad[0] = v_eval[0] + v_step/step_fac;
		v_grad[1] = v_eval[1]; v_grad[2] = v_eval[2];
		HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step/step_fac, spin, it_tol, RHS_grad_1);
		v_grad[0] = v_eval[0] - v_step/step_fac;
		HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step/step_fac, spin, it_tol, RHS_grad_2);
		for (int i=0; i<3; ++i) { Jacob[i][0] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step/step_fac); }
		Jacob[0][0] += 1.0; // Diagonal terms
		// Variation along y
		v_grad[1] = v_eval[1] + v_step/step_fac;
		v_grad[0] = v_eval[0]; v_grad[2] = v_eval[2];
		HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step/step_fac, spin, it_tol, RHS_grad_1);
		v_grad[1] = v_eval[1] - v_step/step_fac;
		HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step/step_fac, spin, it_tol, RHS_grad_2);
		for (int i=0; i<3; ++i) { Jacob[i][1] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step/step_fac); }
		Jacob[1][1] += 1.0; // Diagonal terms
		// Variation along z
		v_grad[2] = v_eval[2] + v_step/step_fac;
		v_grad[0] = v_eval[0]; v_grad[1] = v_eval[1];
		HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step/step_fac, spin, it_tol, RHS_grad_1);
		v_grad[2] = v_eval[2] - v_step/step_fac;
		HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step/step_fac, spin, it_tol, RHS_grad_2);
		for (int i=0; i<3; ++i) { Jacob[i][2] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step/step_fac); }
		Jacob[2][2] += 1.0; // Diagonal terms
		ComputeInverseMatrix3( Jacob, inv_Jacob );
		
		for (int i=0; i<3; ++i) { v_grad[i] = v_eval[i]; }

		for (int i=0; i<3; ++i){
			for (int j=0; j<3; ++j){ v_eval[i] -= inv_Jacob[j][i]*(v_grad[j] - v_init[j] - RHS_eval_v[j]*dt); }
		}

		// Store for next iteration
		for (int i=0; i<3; ++i) { x_prev[i] = x_grad[i]; }
		for (int i=0; i<3; ++i) { v_prev[i] = v_grad[i]; }

		}while(
			n_iter < it_max
			&& ( sqrt(SQR(x_eval[0] - x_prev[0]) + SQR(x_eval[1] - x_prev[1]) + SQR(x_eval[2] - x_prev[2])) > it_tol
			|| sqrt(SQR(v_eval[0] - v_prev[0]) + SQR(v_eval[1] - v_prev[1]) + SQR(v_eval[2] - v_prev[2])) > it_tol )
				 );

		// Done with iterations, update ``true'' values
		pr(IPVX,p) = v_eval[0];
					if (multi_d) { pr(IPVY,p) = v_eval[1]; }
		if (three_d) { pr(IPVZ,p) = v_eval[2]; }
		pr(IPX,p) = x_eval[0];
					if (multi_d) { pr(IPY,p) = x_eval[1]; }
					if (three_d) { pr(IPZ,p) = x_eval[2]; }
		aux_n_iter += n_iter;
	}, Kokkos::Sum<Real>(avg_iter));
	average_iteration_number += avg_iter / nprtcl_thispack;
	return;
}

} // namespace particles
