//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = (pmy_pack->pmesh->dt);
  auto gids = pmy_pack->gids;

  switch (pusher) {
    case ParticlesPusher::drift:

      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        int m = pi(PGID,p) - gids;
        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        pr(IPX,p) += 0.5*dt_*pr(IPVX,p);

        if (multi_d) {
          int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
          pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
        }

        if (three_d) {
          int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
          pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
        }
      });

    break;
    
    case ParticlesPusher::boris:
      BorisStep(dt_, false);
    break;
    
    case ParticlesPusher::full_gr:
      //BorisStep(dt_/2.0, true);
      GeodesicIterations(dt_);
      //BorisStep(dt_/2.0, true);
    break;

    default:
    break;
  }

  return TaskStatus::complete;
}

//Provide dt and only_v as input parameter in order to be able to use this function
//also for half-steps in the full_gr pusher
//Largely implemented following Ripperda et al. 2018 (https://doi.org/10.3847/1538-4365/aab114)
KOKKOS_INLINE_FUNCTION
void Particles::BorisStep( const Real dt, const bool only_v ){
	
	auto &npart = nprtcl_thispack;
	auto &pi = prtcl_idata;
	auto &pr = prtcl_rdata;
	auto &b0_ = pmy_pack->pmhd->b0;
	auto &e0_ = pmy_pack->pmhd->efld;
	auto &indcs = pmy_pack->pmesh->mb_indcs;
	const int is = indcs.is;
	const int js = indcs.js;
	const int ks = indcs.ks;
	auto &mbsize = pmy_pack->pmb->mb_size;
	auto gids = pmy_pack->gids;
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;

      // First half-step in space
      par_for("part_boris",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
      
        // Contravariant and co-variant 4-velocities
	Real u_con[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	//Real u_cov[3];
	Real x[3]; //Half-step increment.
	Real g_Lor;
	x[0] = pr(IPX,p) + dt/(2.0)*u_con[0];
        if (multi_d) { x[1] = pr(IPY,p) + dt/(2.0)*u_con[1]; }
        if (three_d) { x[2] = pr(IPZ,p) + dt/(2.0)*u_con[2]; }
	// Get metric components at new location x1,x2,x3
	Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric 
					       // (remember: sqrt(-1/gupper[0][0]) = alpha, glower[0][i] = beta[i])
	ComputeMetricAndInverse(pr(IPX,p),pr(IPY,p),pr(IPZ,p), is_minkowski, spin, glower, gupper); 
	// Compute 3x3 ADM spatial metric from covariant metric 
	GetUpperAdmMetric( gupper, ADM_upper );
	// ADM_lower = glower for i,j = 1,2,3
	// Lower indeces of u_con (g_Lor is applied afterwards)
	// for (int i1 = 0; i1 < 3; ++i1 ){ 
	// 	u_cov[i1] = 0.0;
	// 	for (int i2 = 0; i2 < 3; ++i2 ){ 
	// 	u_cov[i1] += glower[i1+1][i2+1]*u_con[i2];
	// 	}
	// }
	//Use definition of the Lorentz factor in ADM formalism
	g_Lor = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_con[i1]*u_con[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
	//Boost velocities
	// for ( int i=0; i<3; ++i){
	// 	u_con[i] *= g_Lor;
	// }

	Real uE[3]; //Evolution of the velocity due to the electric field (first half). Index 1... stands for dimension (0 is time).
	Real uB[3]; //Evolution of the velocity due to the magnetic field. Index 1... stands for dimension (0 is time).

        int m = pi(PGID,p) - gids;
        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
	int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
	int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;

	Real &x1min = mbsize.d_view(m).x1min;
	Real &x2min = mbsize.d_view(m).x2min;
	Real &x3min = mbsize.d_view(m).x3min;
	Real &x1max = mbsize.d_view(m).x1max;
	Real &x2max = mbsize.d_view(m).x2max;
	Real &x3max = mbsize.d_view(m).x3max;
	Real x1v,x2v,x3v;
	x1v = LeftEdgeX(ip, indcs.nx1, x1min, x1max);
	x2v = LeftEdgeX(jp, indcs.nx2, x2min, x2max);
	x3v = LeftEdgeX(kp, indcs.nx3, x3min, x3max);
	Real Dx,Dy,Dz;
	Dx = (x1max - x1min)/indcs.nx1;
	Dy = (x2max - x2min)/indcs.nx2;
	Dz = (x3max - x3min)/indcs.nx3;
        // Interpolate Electric Field at new particle location x1, x2, x3
	// Store it in an array for convenience 
	Real E[3] = {0.0, 0.0, 0.0};
	E[0] = e0_.x1e(m, kp, jp, ip) + (x[0] - x1v)*(e0_.x1e(m, kp, jp, ip+1) - e0_.x1e(m, kp, jp, ip))/Dx;
	E[1] = e0_.x2e(m, kp, jp, ip) + (x[1] - x2v)*(e0_.x2e(m, kp, jp+1, ip) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[2] = e0_.x3e(m, kp, jp, ip) + (x[2] - x3v)*(e0_.x3e(m, kp+1, jp, ip) - e0_.x3e(m, kp, jp, ip))/Dz;

	uE[0] = u_con[0] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[0];
        if (multi_d) { uE[1] = u_con[1] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[1]; }
        if (three_d) { uE[2] = u_con[2] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[2]; }

	// Get metric components at new location x1,x2,x3
	ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );
	g_Lor = 0.0; //Intermediate Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_con[i1]*u_con[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
        // Interpolate Magnetic Field at new particle location x1, x2, x3
	// Store it in an array for convenience 
	Real B[3] = {0.0, 0.0, 0.0};
	B[0] = b0_.x1f(m, kp, jp, ip) + (x[0] - x1v)*(b0_.x1f(m, kp, jp, ip+1) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[1] = b0_.x2f(m, kp, jp, ip) + (x[1] - x2v)*(b0_.x2f(m, kp, jp+1, ip) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[2] = b0_.x3f(m, kp, jp, ip) + (x[2] - x3v)*(b0_.x3f(m, kp+1, jp, ip) - b0_.x3f(m, kp, jp, ip))/Dz;

	Real mod_t_sqr = 0.0;
	Real t[3];
	for (int i1 = 0; i1 < 3; ++i1 ){ 
	t[i1] = B[i1]*pr(IPC,p)*dt/(2.0*pr(IPM,p)*g_Lor);
	mod_t_sqr += SQR(t[i1]);
	}

	// Save the vector product of u and t 
	Real vec_ut[3] = {
	uE[1]*t[2] - uE[2]*t[1],
	uE[2]*t[0] - uE[0]*t[2],
	uE[0]*t[1] - uE[1]*t[0],
	};

	uB[0] = uE[0] + 2.0/(1.0+mod_t_sqr)*( (uE[1] + vec_ut[1])*t[2] - (uE[2] + vec_ut[2])*t[1] );
        if (multi_d) { uB[1] = uE[1] + 2.0/(1.0+mod_t_sqr)*( (uE[2] + vec_ut[2])*t[0] - (uE[0] + vec_ut[0])*t[2] ); }
        if (three_d) { uB[2] = uE[2] + 2.0/(1.0+mod_t_sqr)*( (uE[0] + vec_ut[0])*t[1] - (uE[1] + vec_ut[1])*t[0] ); }

	g_Lor = 0.0; //Intermediate Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_con[i1]*u_con[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
	// Finally update velocity in local space
	pr(IPVX,p) = (uB[0] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[0])/g_Lor ;
	pr(IPVY,p) = (uB[1] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[1])/g_Lor ;
	pr(IPVZ,p) = (uB[2] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[2])/g_Lor ;

	if (!only_v){
	pr(IPX,p) = x[0] + dt/(2.0)*pr(IPVX,p);
        if (multi_d) { pr(IPY,p) = x[1] + dt/(2.0)*pr(IPVY,p); }
        if (three_d) { pr(IPZ,p) = x[2] + dt/(2.0)*pr(IPVZ,p); }
	}
      });
      return;
}

//Provide dt as input parameter in order to be able to use this function
//also for half-steps
//Largely implemented following Bacchini et al. 2020 (https://doi.org/10.3847/1538-4365/abb604)
KOKKOS_INLINE_FUNCTION
void Particles::GeodesicIterations( const Real dt ){
	auto &pi = prtcl_idata;
	auto &pr = prtcl_rdata;
	const Real it_tol = iter_tolerance;
	const int it_max = max_iter;
	const bool &multi_d = pmy_pack->pmesh->multi_d;
	const bool &three_d = pmy_pack->pmesh->three_d;
	const Real x_step = 1.0E-10;
	const Real v_step = 1.0E-10;

      // First attempt: not iterative, approximate
      par_for("part_fullgr",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {

	std::cout << "Particle: " << p << " " << pr(IPX,p) << " " << pr(IPY,p) << " " << pr(IPZ,p) << " " << pr(IPVX,p) << " " << pr(IPVY,p) << " " << pr(IPVZ,p) << std::endl;
        // Iterate per particle such that those that converge quicker don't go through as many iterations
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	// Initialize iteration variables
	int n_iter = 0;
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

	// Start iterating
	// Using Newton method, thus computing the Jacobian at each iteration
	do{
		
	++n_iter;

	HamiltonEquation_Position(x_init, x_eval, v_init, v_eval, RHS_eval_x);
	HamiltonEquation_Velocity(x_init, x_eval, v_init, v_eval, x_step, RHS_eval_v);
	std::cout << "RHS_eval_x: " << RHS_eval_x[0] << " " << RHS_eval_x[1] << " " << RHS_eval_x[2] << std::endl;
	std::cout << "RHS_eval_v: " << RHS_eval_v[0] << " " << RHS_eval_v[1] << " " << RHS_eval_v[2] << std::endl;

	// First Jacobian for position
	// Variation along x
	x_grad[0] = x_eval[0] + x_step;
	x_grad[1] = x_eval[1]; x_grad[2] = x_eval[2];
	HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, RHS_grad_1);
	x_grad[0] = x_eval[0] - x_step;
	HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, RHS_grad_2);
	for (int i=0; i<3; ++i) { Jacob[0][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step); }
	Jacob[0][0] += 1.0; // Diagonal terms
	// Variation along y
	x_grad[1] = x_eval[1] + x_step;
	x_grad[0] = x_eval[0]; x_grad[2] = x_eval[2];
	HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, RHS_grad_1);
	x_grad[1] = x_eval[1] - x_step;
	HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, RHS_grad_2);
	for (int i=0; i<3; ++i) { Jacob[1][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step); }
	Jacob[1][1] += 1.0; // Diagonal terms
	// Variation along z
	x_grad[2] = x_eval[2] + x_step;
	x_grad[0] = x_eval[0]; x_grad[1] = x_eval[1];
	HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, RHS_grad_1);
	x_grad[2] = x_eval[2] - x_step;
	HamiltonEquation_Position(x_init, x_grad, v_init, v_eval, RHS_grad_2);
	for (int i=0; i<3; ++i) { Jacob[2][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*x_step); }
	Jacob[2][2] += 1.0; // Diagonal terms
	ComputeInverseMatrix3( Jacob, inv_Jacob );

	// Store values for use with velocity Jacobian
	for (int i=0; i<3; ++i) { x_grad[i] = x_eval[i]; }

	for (int i=0; i<3; ++i){
		for (int j=0; j<3; ++j){ x_eval[i] -= inv_Jacob[j][i]*(x_grad[j] - x_init[j] - RHS_eval_x[j]*dt); }
	}

	// Then Jacobian for velocity
	// Variation along x
	v_grad[0] = v_eval[0] + v_step;
	v_grad[1] = v_eval[1]; v_grad[2] = v_eval[2];
	HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, RHS_grad_1);
	v_grad[0] = v_eval[0] - v_step;
	HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, RHS_grad_2);
	for (int i=0; i<3; ++i) { Jacob[0][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step); }
	Jacob[0][0] += 1.0; // Diagonal terms
	// Variation along y
	v_grad[1] = v_eval[1] + v_step;
	v_grad[0] = v_eval[0]; v_grad[2] = v_eval[2];
	HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, RHS_grad_1);
	v_grad[1] = v_eval[1] - v_step;
	HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, RHS_grad_2);
	for (int i=0; i<3; ++i) { Jacob[1][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step); }
	Jacob[1][1] += 1.0; // Diagonal terms
	// Variation along z
	v_grad[2] = v_eval[2] + v_step;
	v_grad[0] = v_eval[0]; v_grad[1] = v_eval[1];
	HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, RHS_grad_1);
	v_grad[2] = v_eval[2] - v_step;
	HamiltonEquation_Velocity(x_init, x_grad, v_init, v_grad, x_step, RHS_grad_2);
	for (int i=0; i<3; ++i) { Jacob[2][i] = - (RHS_grad_1[i] - RHS_grad_2[i])*dt/(2.0*v_step); }
	Jacob[2][2] += 1.0; // Diagonal terms
	ComputeInverseMatrix3( Jacob, inv_Jacob );
	
	for (int i=0; i<3; ++i) { v_grad[i] = v_eval[i]; }

	for (int i=0; i<3; ++i){
		for (int j=0; j<3; ++j){ v_eval[i] -= inv_Jacob[j][i]*(v_grad[j] - v_init[j] - RHS_eval_v[j]*dt); }
	}

	// Store for next iteration
	for (int i=0; i<3; ++i) { x_prev[i] = x_grad[i]; }
	for (int i=0; i<3; ++i) { v_prev[i] = v_grad[i]; }

        std::cout << "x_diff " << sqrt(SQR(x_eval[0] - x_prev[0]) + SQR(x_eval[1] - x_prev[1]) + SQR(x_eval[2] - x_prev[2])) << std::endl;
        std::cout << "u_diff " << sqrt(SQR(v_eval[0] - v_prev[0]) + SQR(v_eval[1] - v_prev[1]) + SQR(v_eval[2] - v_prev[2])) << std::endl;
	}while(
		n_iter < it_max
		&& ( sqrt(SQR(x_eval[0] - x_prev[0]) + SQR(x_eval[1] - x_prev[1]) + SQR(x_eval[2] - x_prev[2])) > it_tol
		|| sqrt(SQR(v_eval[0] - v_prev[0]) + SQR(v_eval[1] - v_prev[1]) + SQR(v_eval[2] - v_prev[2])) > it_tol )
	     );

	if (n_iter == it_max) { std::cout << "Limit of iterations reached on particle " << pi(PTAG,p) << " on rank " << global_variable::my_rank << std::endl; }

	std::cout << "Particle " << p << ": " << n_iter << " iteration(s)." << std::endl;
	// Done with iterations, update ``true'' values
	pr(IPVX,p) = v_eval[0];
        if (multi_d) { pr(IPVY,p) = v_eval[1]; }
	if (three_d) { pr(IPVZ,p) = v_eval[2]; }
	pr(IPX,p) = x_eval[0];
        if (multi_d) { pr(IPY,p) = x_eval[1]; }
        if (three_d) { pr(IPZ,p) = x_eval[2]; }
	std::cout << "Particle: " << p << " " << pr(IPX,p) << " " << pr(IPY,p) << " " << pr(IPZ,p) << " " << pr(IPVX,p) << " " << pr(IPVY,p) << " " << pr(IPVZ,p) << std::endl;
      });
      return;
}

// Following function largely implented based on the appendix in Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca
// It's an expression of the derivative of the ADM Hamiltonian for geodesics with respect to the velocity
// That prevents singularities for small increments and is used to compute the time derivative of the position
// Would be nice to come up with a way to compute these terms algorithmically, rather than by hard coding, but thus far I wasn't able to
KOKKOS_INLINE_FUNCTION
void Particles::HamiltonEquation_Position(const Real * x_0, const Real * x_1, const Real * u_0, const Real * u_1, Real * H){

	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric components
	Real massive = 1.0; //TODO for photons/massless particles this needs to be 0: condition on ptype
	Real U_1, U_0, perp;

	for (int i=0; i<3; ++i){ H[i] = 0.0; }

	//Metric with all old positions
	//Common to all terms
	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[1][1]*SQR(u_0[1]) + ADM_upper[2][2]*SQR(u_0[2]) + ADM_upper[1][2]*u_0[1]*u_0[2];
	U_1 = ADM_upper[0][0]*SQR(u_1[0]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_0[2];
	U_1 += perp;	
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[0][0]*SQR(u_0[0]) + 2.0*ADM_upper[0][1]*u_0[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_0[2];
	U_0 += perp;	
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[0] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[0][0]*(u_1[0]+u_0[0]) + 2.0*ADM_upper[0][1]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[0] += gupper[0][1]/gupper[0][0];
	
	perp =  ADM_upper[0][0]*SQR(u_0[0]) + ADM_upper[2][2]*SQR(u_0[2]) + ADM_upper[0][2]*u_0[0]*u_0[2];
	U_1 = ADM_upper[1][1]*SQR(u_1[1]) + 2.0*ADM_upper[0][1]*u_1[1]*u_0[0] + 2.0*ADM_upper[1][2]*u_1[1]*u_0[2];
	U_1 += perp;	
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[1][1]*SQR(u_0[1]) + 2.0*ADM_upper[0][1]*u_0[0]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_0[2];
	U_0 += perp;	
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[1] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[1][1]*(u_1[1]+u_0[1]) + 2.0*ADM_upper[0][1]*u_0[0] + 2.0*ADM_upper[1][2]*u_0[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[1] += gupper[0][2]/gupper[0][0];

	perp = ADM_upper[0][0]*SQR(u_0[0]) + ADM_upper[1][1]*SQR(u_0[1]) + ADM_upper[1][0]*u_0[1]*u_0[0];
	U_1 = ADM_upper[2][2]*SQR(u_1[2]) + 2.0*ADM_upper[0][2]*u_1[0]*u_0[2] + 2.0*ADM_upper[1][2]*u_1[1]*u_0[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[2][2]*SQR(u_0[2]) + 2.0*ADM_upper[2][1]*u_0[2]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_0[2];
	U_0 += perp;	
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[2] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[2][2]*(u_1[2]+u_0[2]) + 2.0*ADM_upper[0][2]*u_0[0] + 2.0*ADM_upper[1][2]*u_0[1]
	       	)/(U_0 + U_1);
	// Beta term
	H[2] += gupper[0][3]/gupper[0][0];
	
	//
	//Metric with all new positions
	//Common to all terms
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[1][1]*SQR(u_1[1]) + ADM_upper[2][2]*SQR(u_1[2]) + ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 = ADM_upper[0][0]*SQR(u_1[0]) + 2.0*ADM_upper[0][1]*u_1[0]*u_1[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[0][0]*SQR(u_0[0]) + 2.0*ADM_upper[0][1]*u_0[0]*u_1[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[0] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[0][0]*(u_1[0]+u_0[0]) + 2.0*ADM_upper[0][1]*u_1[1] + 2.0*ADM_upper[0][2]*u_1[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[0] += gupper[0][1]/gupper[0][0];
	
	perp = ADM_upper[0][0]*SQR(u_1[0]) + ADM_upper[2][2]*SQR(u_1[2]) + ADM_upper[0][2]*u_1[0]*u_1[2];
	U_1 = ADM_upper[1][1]*SQR(u_1[1]) + 2.0*ADM_upper[0][1]*u_1[1]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[1][1]*SQR(u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_1[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[1] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[1][1]*(u_1[1]+u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[1] += gupper[0][2]/gupper[0][0];

	perp = ADM_upper[0][0]*SQR(u_1[0]) + ADM_upper[1][1]*SQR(u_1[1]) + ADM_upper[1][0]*u_1[1]*u_1[0];
	U_1 = ADM_upper[2][2]*SQR(u_1[2]) + 2.0*ADM_upper[0][2]*u_1[0]*u_1[2] + 2.0*ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[2][2]*SQR(u_0[2]) + 2.0*ADM_upper[2][1]*u_0[2]*u_1[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[2] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[2][2]*(u_1[2]+u_0[2]) + 2.0*ADM_upper[0][2]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[1]
	       	)/(U_0 + U_1);
	// Beta term
	H[2] += gupper[0][3]/gupper[0][0];

	//
	//Metric with x_0[0], x_1[1], x_1[2]
	//Common to terms 0 and 2
	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[1][1]*SQR(u_1[1]) + ADM_upper[2][2]*SQR(u_1[2]) + ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 = ADM_upper[0][0]*SQR(u_1[0]) + 2.0*ADM_upper[0][1]*u_1[0]*u_1[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[0][0]*SQR(u_0[0]) + 2.0*ADM_upper[0][1]*u_0[0]*u_1[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[0] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[0][0]*(u_1[0]+u_0[0]) + 2.0*ADM_upper[0][1]*u_1[1] + 2.0*ADM_upper[0][2]*u_1[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[0] += gupper[0][1]/gupper[0][0];

	perp = ADM_upper[0][0]*SQR(u_0[0]) + ADM_upper[1][1]*SQR(u_1[1]) + ADM_upper[1][0]*u_1[1]*u_0[0];
	U_1 = ADM_upper[2][2]*SQR(u_1[2]) + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2] + 2.0*ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[2][2]*SQR(u_0[2]) + 2.0*ADM_upper[2][1]*u_0[2]*u_1[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[2] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[2][2]*(u_1[2]+u_0[2]) + 2.0*ADM_upper[0][2]*u_0[0] + 2.0*ADM_upper[1][2]*u_1[1]
	       	)/(U_0 + U_1);
	// Beta term
	H[2] += gupper[0][3]/gupper[0][0];


	//
	//Metric with x_1[0], x_0[1], x_0[2]
	//Common to terms 0 and 1
	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[1][1]*SQR(u_0[1]) + ADM_upper[2][2]*SQR(u_0[2]) + ADM_upper[1][2]*u_0[1]*u_0[2];
	U_1 = ADM_upper[0][0]*SQR(u_1[0]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_0[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[0][0]*SQR(u_0[0]) + 2.0*ADM_upper[0][1]*u_0[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[0] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[0][0]*(u_1[0]+u_0[0]) + 2.0*ADM_upper[0][1]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[0] += gupper[0][1]/gupper[0][0];
	
	perp = ADM_upper[0][0]*SQR(u_1[0]) + ADM_upper[2][2]*SQR(u_0[2]) + ADM_upper[0][2]*u_1[0]*u_0[2];
	U_1 = ADM_upper[1][1]*SQR(u_1[1]) + 2.0*ADM_upper[0][1]*u_1[1]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[1]*u_0[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[1][1]*SQR(u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[1] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[1][1]*(u_1[1]+u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0] + 2.0*ADM_upper[1][2]*u_0[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[1] += gupper[0][2]/gupper[0][0];


	//
	//Metric with x_1[0], x_0[1], x_1[2]
	//Common to terms 0 and 1
	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[1][1]*SQR(u_0[1]) + ADM_upper[2][2]*SQR(u_1[2]) + ADM_upper[1][2]*u_0[1]*u_1[2];
	U_1 = ADM_upper[0][0]*SQR(u_1[0]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[0][0]*SQR(u_0[0]) + 2.0*ADM_upper[0][1]*u_0[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[0] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[0][0]*(u_1[0]+u_0[0]) + 2.0*ADM_upper[0][1]*u_0[1] + 2.0*ADM_upper[0][2]*u_1[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[0] += gupper[0][1]/gupper[0][0];
	
	perp = ADM_upper[0][0]*SQR(u_1[0]) + ADM_upper[2][2]*SQR(u_1[2]) + ADM_upper[0][2]*u_1[0]*u_1[2];
	U_1 = ADM_upper[1][1]*SQR(u_1[1]) + 2.0*ADM_upper[0][1]*u_1[1]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[1][1]*SQR(u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_1[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[1] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[1][1]*(u_1[1]+u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[1] += gupper[0][2]/gupper[0][0];


	//
	//Metric with x_0[0], x_0[1], x_1[2]
	//Common to terms 0 and 2
	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[1][1]*SQR(u_0[1]) + ADM_upper[2][2]*SQR(u_1[2]) + ADM_upper[1][2]*u_0[1]*u_1[2];
	U_1 = ADM_upper[0][0]*SQR(u_1[0]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[0][0]*SQR(u_0[0]) + 2.0*ADM_upper[0][1]*u_0[0]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[0] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[0][0]*(u_1[0]+u_0[0]) + 2.0*ADM_upper[0][1]*u_0[1] + 2.0*ADM_upper[0][2]*u_1[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[0] += gupper[0][1]/gupper[0][0];

	perp = ADM_upper[0][0]*SQR(u_0[0]) + ADM_upper[1][1]*SQR(u_0[1]) + ADM_upper[1][0]*u_0[1]*u_0[0];
	U_1 = ADM_upper[2][2]*SQR(u_1[2]) + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2] + 2.0*ADM_upper[1][2]*u_0[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[2][2]*SQR(u_0[2]) + 2.0*ADM_upper[2][1]*u_0[2]*u_0[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[2] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[2][2]*(u_1[2]+u_0[2]) + 2.0*ADM_upper[0][2]*u_0[0] + 2.0*ADM_upper[1][2]*u_0[1]
	       	)/(U_0 + U_1);
	// Beta term
	H[2] += gupper[0][3]/gupper[0][0];


	//
	//Metric with x_1[0], x_1[1], x_0[2]
	//Common to terms 1 and 2
	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[0][0]*SQR(u_1[0]) + ADM_upper[2][2]*SQR(u_0[2]) + ADM_upper[0][2]*u_1[0]*u_0[2];
	U_1 = ADM_upper[1][1]*SQR(u_1[1]) + 2.0*ADM_upper[0][1]*u_1[1]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[1]*u_0[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[1][1]*SQR(u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[1] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[1][1]*(u_1[1]+u_0[1]) + 2.0*ADM_upper[0][1]*u_1[0] + 2.0*ADM_upper[1][2]*u_0[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[1] += gupper[0][2]/gupper[0][0];

	perp = ADM_upper[0][0]*SQR(u_1[0]) + ADM_upper[1][1]*SQR(u_1[1]) + ADM_upper[1][0]*u_1[1]*u_1[0];
	U_1 = ADM_upper[2][2]*SQR(u_1[2]) + 2.0*ADM_upper[0][2]*u_1[0]*u_1[2] + 2.0*ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[2][2]*SQR(u_0[2]) + 2.0*ADM_upper[2][1]*u_0[2]*u_1[1] + 2.0*ADM_upper[0][2]*u_1[0]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[2] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[2][2]*(u_1[2]+u_0[2]) + 2.0*ADM_upper[0][2]*u_1[0] + 2.0*ADM_upper[1][2]*u_1[1]
	       	)/(U_0 + U_1);
	// Beta term
	H[2] += gupper[0][3]/gupper[0][0];


	//
	//Metric with x_0[0], x_1[1], x_0[2]
	//Common to terms 1 and 2
	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );

	perp = ADM_upper[0][0]*SQR(u_0[0]) + ADM_upper[2][2]*SQR(u_0[2]) + ADM_upper[0][2]*u_0[0]*u_0[2];
	U_1 = ADM_upper[1][1]*SQR(u_1[1]) + 2.0*ADM_upper[0][1]*u_1[1]*u_0[0] + 2.0*ADM_upper[1][2]*u_1[1]*u_0[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[1][1]*SQR(u_0[1]) + 2.0*ADM_upper[0][1]*u_0[0]*u_0[1] + 2.0*ADM_upper[1][2]*u_0[1]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[1] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[1][1]*(u_1[1]+u_0[1]) + 2.0*ADM_upper[0][1]*u_0[0] + 2.0*ADM_upper[1][2]*u_0[2]
	       	)/(U_0 + U_1);
	// Beta term
	H[1] += gupper[0][2]/gupper[0][0];

	perp = ADM_upper[0][0]*SQR(u_0[0]) + ADM_upper[1][1]*SQR(u_1[1]) + ADM_upper[1][0]*u_1[1]*u_0[0];
	U_1 = ADM_upper[2][2]*SQR(u_1[2]) + 2.0*ADM_upper[0][2]*u_0[0]*u_1[2] + 2.0*ADM_upper[1][2]*u_1[1]*u_1[2];
	U_1 += perp;
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM_upper[2][2]*SQR(u_0[2]) + 2.0*ADM_upper[2][1]*u_0[2]*u_1[1] + 2.0*ADM_upper[0][2]*u_0[0]*u_0[2];
	U_0 += perp;
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	H[2] += sqrt(-1.0/gupper[0][0])*(
		ADM_upper[2][2]*(u_1[2]+u_0[2]) + 2.0*ADM_upper[0][2]*u_0[0] + 2.0*ADM_upper[1][2]*u_1[1]
	       	)/(U_0 + U_1);
	// Beta term
	H[2] += gupper[0][3]/gupper[0][0];
	
	for (int i=0; i<3; ++i){ H[i] /= 6.0; }
}

// Following function largely implented based on the appendix in Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca
// It's an expression of the derivative of the ADM Hamiltonian for geodesics with respect to the position
// That prevents singularities for small increments and is used to compute the time derivative of the velocity
// Would be nice to come up with a way to compute these terms algorithmically, rather than by hard coding, but thus far I wasn't able to
KOKKOS_INLINE_FUNCTION
void Particles::HamiltonEquation_Velocity(const Real * x_0, const Real * x_1, const Real * u_0, const Real * u_1, const Real x_step, Real * H){

	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	const Real it_tol = iter_tolerance;
	Real gl_0[4][4], gu_0[4][4], gl_1[4][4], gu_1[4][4]; // Metric components
	Real aux_gl_0[4][4], aux_gu_0[4][4], aux_gl_1[4][4], aux_gu_1[4][4]; // Metric components
	Real dx, dy, dz, u[3];
       	// Difference between old and new points might be too small
	// Use drivative instead
	bool use_derivative_x, use_derivative_y, use_derivative_z;

	for (int i=0; i<3; ++i){ H[i] = 0.0; }
	dx = x_1[0] - x_0[0];
	use_derivative_x = (fabs(dx) < sqrt(it_tol)) ? true : false;
	dy = x_1[1] - x_0[1];
	use_derivative_y = (fabs(dy) < sqrt(it_tol)) ? true : false;
	dz = x_1[2] - x_0[2];
	use_derivative_z = (fabs(dz) < sqrt(it_tol)) ? true : false;

	//Terms with all old velocities
	//Common to all directions
	u[0] = u_0[0]; 
	u[1] = u_0[1]; 
	u[2] = u_0[2]; 

	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 

	if (use_derivative_x){
		ComputeMetricAndInverse(x_0[0]-x_step,x_0[1],x_0[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step,x_0[1],x_0[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}
	
	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_x){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[0]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[0]);
	}
	
	// Change only _1 quantities as old quantities are unchanged
	if (use_derivative_y){
		ComputeMetricAndInverse(x_0[0],x_0[1]-x_step,x_0[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0],x_0[1]+x_step,x_0[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_y){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[1]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[1]);
	}

	if (use_derivative_z){
		ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2]-x_step, is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2]+x_step, is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_z){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[2]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[2]);
	}

	//Terms with all new velocities
	//Common to all directions
	u[0] = u_1[0]; 
	u[1] = u_1[1]; 
	u[2] = u_1[2]; 

	// Notice inversion of gu_1 and gu_0 as constant
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 

	if (use_derivative_x){
		ComputeMetricAndInverse(x_1[0]-x_step,x_1[1],x_1[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0]+x_step,x_1[1],x_1[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}
	
	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_x){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[0]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[0]);
	}

	if (use_derivative_y){
		ComputeMetricAndInverse(x_1[0],x_1[1]-x_step,x_1[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0],x_1[1]+x_step,x_1[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_y){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[1]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[1]);
	}

	if (use_derivative_z){
		ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2]-x_step, is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2]+x_step, is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_z){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[2]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[2]);
	}
	
	//Terms with new velocities for y and z
	//Common to terms 0 and 2
	u[0] = u_0[0]; 
	u[1] = u_1[1]; 
	u[2] = u_1[2]; 

	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_x){
		ComputeMetricAndInverse(x_0[0]-x_step,x_1[1],x_1[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step,x_1[1],x_1[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}
	
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_x){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[0]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[0]);
	}
	
	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_z){
		ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2]-x_step, is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2]+x_step, is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_z){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[2]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[2]);
	}
	
	//Terms with new velocities for x and y
	//Common to terms 1 and 2
	u[0] = u_1[0]; 
	u[1] = u_1[1]; 
	u[2] = u_0[2]; 

	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_y){
		ComputeMetricAndInverse(x_1[0],x_0[1]-x_step,x_0[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0],x_0[1]+x_step,x_0[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_y){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[1]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[1]);
	}
	
	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_z){
		ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2]-x_step, is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2]+x_step, is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_z){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[2]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[2]);
	}

	//Terms with new velocities for x and z
	//Common to terms 0 and 1
	u[0] = u_1[0]; 
	u[1] = u_0[1]; 
	u[2] = u_1[2]; 

	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_x){
		ComputeMetricAndInverse(x_0[0]-x_step,x_0[1],x_1[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step,x_0[1],x_1[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}
	
	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_x){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[0]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[0]);
	}
	
	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_y){
		ComputeMetricAndInverse(x_1[0],x_0[1]-x_step,x_1[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0],x_0[1]+x_step,x_1[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_y){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[1]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[1]);
	}

	//Terms with new velocities for y only
	//Common to terms 1 and 2
	u[0] = u_0[0]; 
	u[1] = u_1[1]; 
	u[2] = u_0[2]; 

	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_y){
		ComputeMetricAndInverse(x_0[0],x_0[1]-x_step,x_0[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0],x_0[1]+x_step,x_0[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_y){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[1]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[1]);
	}
	
	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_z){
		ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2]-x_step, is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2]+x_step, is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_z){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[2]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[2]);
	}

	//Terms with new velocities for z only
	//Common to terms 0 and 2
	u[0] = u_0[0]; 
	u[1] = u_0[1]; 
	u[2] = u_1[2]; 

	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_x){
		ComputeMetricAndInverse(x_0[0]-x_step,x_0[1],x_1[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step,x_0[1],x_1[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}
	
	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_x){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[0]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[0]);
	}
	
	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_z){
		ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2]-x_step, is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2]+x_step, is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_z){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[2]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[2]);
	}

	//Terms with new velocities for x only
	//Common to terms 0 and 1
	u[0] = u_1[0]; 
	u[1] = u_0[1]; 
	u[2] = u_0[2]; 

	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_x){
		ComputeMetricAndInverse(x_0[0]-x_step,x_0[1],x_0[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step,x_0[1],x_0[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}
	
	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_x){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[0]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[0]);
	}

	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, gl_0, gu_0); 
	if (use_derivative_y){
		ComputeMetricAndInverse(x_1[0],x_0[1]-x_step,x_0[2], is_minkowski, spin, aux_gl_1, aux_gu_1); 
		ComputeMetricAndInverse(x_1[0],x_0[1]+x_step,x_0[2], is_minkowski, spin, aux_gl_0, aux_gu_0);
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
	}

	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, gl_1, gu_1); 
	if (use_derivative_y){
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, &H[1]);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, &H[1]);
	}

	for (int i=0; i<3; ++i){ H[i] /= 6.0; }
	if (!use_derivative_x){	H[0] /= dx; }
	if (!use_derivative_y){	H[1] /= dy; }
	if (!use_derivative_z){	H[2] /= dz; }
	//std::cout << "H: " << H[0] << " " << H[1] << " " << H[2] << std::endl;
}

KOKKOS_INLINE_FUNCTION
void Particles::ComputeAndAddSingleTerm_Velocity(const Real gu_0[4][4], const Real gu_1[4][4], const Real * u, Real * H){
	Real U_0, U_1, aux, beta_t;
	Real ADM_0[3][3], ADM_1[3][3];
	Real massive = 1.0; //TODO for photons/massless particles this needs to be 0: condition on ptype

	GetUpperAdmMetric( gu_0, ADM_0 );
	GetUpperAdmMetric( gu_1, ADM_1 );

	U_0 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		U_0 += ADM_0[i1][i2]*u[i1]*u[i2];
		}
	}
	U_0 = sqrt(U_0 + massive); 
	U_1 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		U_1 += ADM_1[i1][i2]*u[i1]*u[i2];
		}
	}
	U_1 = sqrt(U_1 + massive); 
	aux = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		aux += (ADM_1[i1][i2] - ADM_0[i1][i2])*u[i1]*u[i2];
		}
	}
	beta_t = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		beta_t += ( -gu_1[0][i1]/gu_1[0][0] + gu_0[0][i1]/gu_0[0][0])*u[i1];
	}

	*H -= 0.5*(U_0 + U_1)*( sqrt(-1.0/gu_1[0][0]) - sqrt(-1.0/gu_0[0][0]) ) ;
	*H -= 0.5*aux*( sqrt(-1.0/gu_1[0][0]) + sqrt(-1.0/gu_0[0][0]) )/(U_1 + U_0) ;
	*H += beta_t ;
}

// Overload previous function to use the derivative of the metric if needed
KOKKOS_INLINE_FUNCTION
void Particles::ComputeAndAddSingleTerm_Velocity(const Real gu_0[4][4], const Real gu_1[4][4], const Real g_der[4][4], const Real * u, Real * H){
	Real U_0, U_1, aux, beta_t;
	Real ADM_0[3][3], ADM_1[3][3], ADM_der[3][3];
	Real massive = 1.0; //TODO for photons/massless particles this needs to be 0: condition on ptype

	GetUpperAdmMetric( gu_0, ADM_0 );
	GetUpperAdmMetric( gu_1, ADM_1 );
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_der[i1][i2] = g_der[i1+1][i2+1]
		       	+ g_der[0][0]*gu_0[0][i1+1]*gu_0[0][i2+1]/SQR(gu_0[0][0])
			- 2.0*g_der[0][i1+1]*gu_0[0][i2+1]/gu_0[0][0];
		}
	}
	U_0 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		U_0 += ADM_0[i1][i2]*u[i1]*u[i2];
		}
	}
	U_0 = sqrt(U_0 + massive); 
	U_1 = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		U_1 += ADM_1[i1][i2]*u[i1]*u[i2];
		}
	}
	U_1 = sqrt(U_1 + massive); 
	aux = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		aux += ADM_der[i1][i2]*u[i1]*u[i2];
		}
	}
	beta_t = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		beta_t -= (g_der[0][i1] - gu_0[0][i1]*g_der[0][0]/gu_0[0][0])/gu_0[0][0]*u[i1];
	}

	// Notice plus sign from derivative of alpha
	*H += 0.5*(U_0 + U_1)*( 0.5*sqrt(-1.0/gu_0[0][0])*g_der[0][0]/fabs(gu_0[0][0]) ) ;
	*H -= 0.5*aux*( sqrt(-1.0/gu_1[0][0]) + sqrt(-1.0/gu_0[0][0]) )/(U_1 + U_0) ;
	*H += beta_t ;
}

KOKKOS_INLINE_FUNCTION
void Particles::ComputeInverseMatrix3( const Real inputMat[][3], Real outputMat[][3] ){

	Real determinant = 0.0;
	determinant = inputMat[0][0]*inputMat[1][1]*inputMat[2][2] + inputMat[0][1]*inputMat[1][2]*inputMat[2][0]
		+ inputMat[1][0]*inputMat[2][1]*inputMat[0][2]
		- inputMat[2][0]*inputMat[1][1]*inputMat[0][2] - inputMat[1][0]*inputMat[0][1]*inputMat[2][2]
		- inputMat[2][1]*inputMat[1][2]*inputMat[1][1];

	// Transposition of Jacobian is skipped and indeces are inverted instead
	outputMat[0][0] = (inputMat[1][1]*inputMat[2][2] - inputMat[2][1]*inputMat[1][2])/determinant;
	outputMat[0][1] = -(inputMat[1][0]*inputMat[2][2] - inputMat[0][2]*inputMat[2][1])/determinant;
	outputMat[0][2] = (inputMat[1][1]*inputMat[1][2] - inputMat[0][2]*inputMat[1][1])/determinant;
	outputMat[1][0] = -(inputMat[1][0]*inputMat[2][2] - inputMat[1][2]*inputMat[2][0])/determinant;
	outputMat[1][1] = (inputMat[0][0]*inputMat[2][2] - inputMat[0][2]*inputMat[2][0])/determinant;
	outputMat[1][2] = -(inputMat[0][0]*inputMat[1][2] - inputMat[0][2]*inputMat[1][0])/determinant;
	outputMat[2][0] = (inputMat[1][0]*inputMat[2][1] - inputMat[1][1]*inputMat[2][0])/determinant;
	outputMat[2][1] = -(inputMat[0][0]*inputMat[2][1] - inputMat[0][1]*inputMat[2][0])/determinant;
	outputMat[2][2] = (inputMat[0][0]*inputMat[1][1] - inputMat[0][1]*inputMat[1][0])/determinant;

}

KOKKOS_INLINE_FUNCTION
void Particles::GetUpperAdmMetric( const Real inputMat[][4], Real outputMat[][3] ){
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		outputMat[i1][i2] = inputMat[i1+1][i2+1] - inputMat[0][i2+1]*inputMat[i1+1][0]/inputMat[0][0];
		}
	}
}

} // namespace particles
