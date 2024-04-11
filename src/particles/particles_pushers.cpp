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
	Real u_cov[3];
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
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_upper[i1][i2] = gupper[i1+1][i2+1] + gupper[0][i2+1]*gupper[i1+1][0];
		}
	}
	// ADM_lower = glower for i,j = 1,2,3
	// Lower indeces of u_con (g_Lor is applied afterwards)
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		u_cov[i1] = 0.0;
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		u_cov[i1] += glower[i1+1][i2+1]*u_con[i2];
		}
	}
	//Use definition of the Lorentz factor in ADM formalism
	g_Lor = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_cov[i1]*u_cov[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
	//Boost velocities
	for ( int i=0; i<3; ++i){
		u_con[i] *= g_Lor;
	}

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
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_upper[i1][i2] = gupper[i1+1][i2+1] + gupper[0][i2+1]*gupper[i1+1][0];
		}
	}
	// Lower indeces of u_con
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		u_cov[i1] = 0.0;
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		u_cov[i1] += glower[i1+1][i2+1]*uE[i2];
		}
	}
	g_Lor = 0.0; //Intermediate Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_cov[i1]*u_cov[i2];
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

	// Lower indeces of u_con
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		u_cov[i1] = 0.0;
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		u_cov[i1] += glower[i1+1][i2+1]*uB[i2];
		}
	}
	g_Lor = 0.0; //Intermediate Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_cov[i1]*u_cov[i2];
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

      // First attempt: not iterative, approximate
      par_for("part_fullgr",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {

	std::cout << "Particle: " << p << " " << pr(IPX,p) << " " << pr(IPY,p) << " " << pr(IPZ,p) << " " << pr(IPVX,p) << " " << pr(IPVY,p) << " " << pr(IPVZ,p) << std::endl;
        // Iterate per particle such that those that converge quicker don't go through as many iterations
	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	Real glower[4][4], gupper[4][4]; // Metric components
	// Initialize iteration variables
	int n_iter = 0;
	Real rest_u[4] = {1.0,1.0,1.0,1.0};
	Real rest_x[3] = {1.0,1.0,1.0};
	Real x_init[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
	Real v_init[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	Real a_init[4] = {0.0, 0.0, 0.0, 0.0};
	Real x_eval[3] = {pr(IPX,p) + dt*pr(IPVX,p), pr(IPY,p) + dt*pr(IPVY,p), pr(IPZ,p) + dt*pr(IPVZ,p)};
	Real v_eval[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	Real x_prev[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
	Real v_prev[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	//u is always contravariant. Iteration variables
	Real u_eval[4], a_eval[4]; 
	Real Jacob[3][3], inv_Jacob[3][3], JT[4][4];
	Real aux, dx_g_Lor[3], du_g_Lor[3], dtau, dtau_init;
	Real g_Lor, u_init[4];

	// To keep for evaluating rest function on velocity
	ComputeLorentzFactorGR(x_init, v_init, u_init, &g_Lor);
	std::cout << "g_Lor: " << g_Lor << " p: " << p << std::endl;
	ComputeGeodesicTerms(x_init, u_init, a_init);
	dtau_init = dt/g_Lor;
	for (int i=0; i<4; ++i) { u_eval[i] = u_init[i] + dtau_init*a_init[i]; }
	//ComputeLorentzFactorGR(x_init, u_eval, &g_Lor);
	g_Lor = u_eval[0];
	dtau = dt/g_Lor;
	ComputeMetricAndInverse(x_init[0],x_init[1],x_init[2], is_minkowski, spin, glower, gupper); 
	for (int i=0; i<3; ++i) { v_eval[i] = u_eval[i+1]*sqrt(glower[i+1][i+1])/g_Lor; }
	for (int i=0; i<3; ++i){ rest_x[i] = x_eval[i] - x_init[i] - (v_init[i] + v_eval[i])*dt/2.0; }
	for (int i=0; i<3; ++i) { x_eval[i] -= rest_x[i]; }
	// Note that definition of a_eval requires sign inversion
	for (int i=0; i<4; ++i){ rest_u[i] = u_eval[i] - u_init[i] - (dtau_init*a_init[i] + dtau*a_eval[i])/2.0; }
	ComputeJacobianTerms(x_eval, u_eval, JT, dx_g_Lor, du_g_Lor);
	// u_eval used for du_g_Lor is boosted, need to divide by g_Lor on top of the SQR(g_Lor) already in the derivative
	// Technically this is missing term with second derivatives of the metric
	// this would require 6 4x4 matrices to be computed
	// // Then do V
	for (int i = 0; i<3; ++i){
		for (int j = 0; j<3; ++j){
			aux = (i == j) ? 1.0 : 0.0;
			Jacob[i][j] = aux +  dt/2.0*( 2.0*JT[i+1][j+1]/g_Lor - du_g_Lor[i]*a_eval[j+1]/SQR(g_Lor)/g_Lor );
		}
	}
	ComputeInverseMatrix3(Jacob, inv_Jacob);
	ComputeMetricAndInverse(x_eval[0],x_eval[1],x_eval[2], is_minkowski, spin, glower, gupper); 
	for (int j = 0; j<3; ++j){ v_eval[0] = (u_eval[1] - rest_u[j+1]*inv_Jacob[0][j])*sqrt(glower[1][1])/g_Lor; }
	for (int j = 0; j<3; ++j){ v_eval[1] = (u_eval[2] - rest_u[j+1]*inv_Jacob[1][j])*sqrt(glower[2][2])/g_Lor; }
	for (int j = 0; j<3; ++j){ v_eval[2] = (u_eval[3] - rest_u[j+1]*inv_Jacob[2][j])*sqrt(glower[3][3])/g_Lor; }
	
        std::cout << "x_diff " << sqrt(SQR(x_eval[0] - x_prev[0]) + SQR(x_eval[1] - x_prev[1]) + SQR(x_eval[2] - x_prev[2])) << std::endl;
        std::cout << "u_diff " << sqrt(SQR(v_eval[0] - v_prev[0]) + SQR(v_eval[1] - v_prev[1]) + SQR(v_eval[2] - v_prev[2])) << std::endl;
	
	// Start iterating
	do{
		
	++n_iter;
	for (int i=0; i<3; ++i) { x_prev[i] = x_eval[i]; v_prev[i] = v_eval[i]; }

	ComputeLorentzFactorGR(x_eval, v_eval, u_eval, &g_Lor);
	//u_eval is not boosted
	std::cout << "g_Lor: " << g_Lor << " p: " << p << std::endl;
	dtau = dt/g_Lor;
	ComputeGeodesicTerms(x_eval, u_eval, a_eval);
        //std::cout << "r_next " << sqrt(SQR(x_next[0]) + SQR(x_next[1]) + SQR(x_next[2])) << " r_curr " << sqrt(SQR(x_curr[0]) + SQR(x_curr[1]) + SQR(x_curr[2])) << std::endl;
	// Check for convergence
	
	//
	//Compute next values with gradient descent

	// // Update X
	for (int i=0; i<3; ++i){ rest_x[i] = x_eval[i] - x_init[i] - (v_init[i] + v_eval[i])*dt/2.0; }
	for (int i=0; i<3; ++i) { x_eval[i] -= rest_x[i]; }
	for (int i=0; i<4; ++i){ rest_u[i] = u_eval[i] - u_init[i] - (dtau_init*a_init[i] + dtau*a_eval[i])/2.0; }
	// Note that definition of a_eval requires sign inversion
	ComputeJacobianTerms(x_eval, u_eval, JT, dx_g_Lor, du_g_Lor);
	// u_eval used for du_g_Lor is boosted, need to divide by g_Lor on top of the SQR(g_Lor) already in the derivative
	// Technically this is missing term with second derivatives of the metric
	// this would require 6 4x4 matrices to be computed
	// // Then do V
	for (int i = 0; i<3; ++i){
		for (int j = 0; j<3; ++j){
			aux = (i == j) ? 1.0 : 0.0;
			Jacob[i][j] = aux +  dt/2.0*( 2.0*JT[i+1][j+1]/g_Lor - du_g_Lor[i]*a_eval[j+1]/SQR(g_Lor)/g_Lor );
		}
	}
	ComputeInverseMatrix3(Jacob, inv_Jacob);
	ComputeMetricAndInverse(x_eval[0],x_eval[1],x_eval[2], is_minkowski, spin, glower, gupper); 
	for (int j = 0; j<3; ++j){ v_eval[0] = (u_eval[1] - rest_u[j+1]*inv_Jacob[0][j])*sqrt(glower[1][1])/g_Lor; }
	for (int j = 0; j<3; ++j){ v_eval[1] = (u_eval[2] - rest_u[j+1]*inv_Jacob[1][j])*sqrt(glower[2][2])/g_Lor; }
	for (int j = 0; j<3; ++j){ v_eval[2] = (u_eval[3] - rest_u[j+1]*inv_Jacob[2][j])*sqrt(glower[3][3])/g_Lor; }
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

KOKKOS_INLINE_FUNCTION
void Particles::ComputeGeodesicTerms(const Real * x, const Real * u, Real * acc){

	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	Real glower[4][4], gupper[4][4]; // Metric components
	Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4]; // Metric derivatives
	Real T0, T1, T2, T3, T4_1, T4_2, T4_3; //Temporary variables
					       //
	Real half = 0.5;
	ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
	ComputeMetricDerivatives(x[0],x[1],x[2], is_minkowski, spin, dg_dx1, dg_dx2, dg_dx3); 
	// Evolve covariant velocity but convert to contravariant at the end
	T0, T1, T2, T3 = 0.0;
	    	       // Following expression technically has a factor 2 before each term
		       // that is not on the diagonal, but this gets multiplied by 1/2 in the end,
		       // so just skip it and divide by 2 values on diagonal of metric.
		     
	// Terms with 0 index in metric
	T0 += half*dg_dx1[0][0]*u[0]*u[1] + half*dg_dx2[0][0]*u[0]*u[2] + half*dg_dx3[0][0]*u[3]*u[0];
	// diagonal terms
	T0 += dg_dx1[0][1]*u[1]*u[1] + dg_dx2[0][2]*u[2]*u[2] + dg_dx3[0][3]*u[3]*u[3];
	// off-diagonal terms
	T0 += (dg_dx1[0][2] + dg_dx2[1][0])*u[1]*u[2] 
	    + (dg_dx1[0][3] + dg_dx3[0][1])*u[1]*u[3] + (dg_dx2[0][3] + dg_dx3[0][2])*u[2]*u[3];
		
	T1 += dg_dx1[0][1]*u[0]*u[1] + dg_dx2[0][1]*u[0]*u[2] + dg_dx3[0][1]*u[3]*u[0];
	T1 += half*dg_dx1[1][1]*u[1]*u[1] + dg_dx2[2][1]*u[2]*u[2] + dg_dx3[3][1]*u[3]*u[3];
	T1 += (dg_dx1[2][1] + half*dg_dx2[1][1])*u[1]*u[2] 
	     + (dg_dx1[3][1] + half*dg_dx3[1][1])*u[1]*u[3] + (dg_dx2[3][1] + dg_dx3[2][1])*u[2]*u[3];

	T2 += dg_dx1[0][2]*u[0]*u[1] + dg_dx2[0][2]*u[0]*u[2] + dg_dx3[0][2]*u[3]*u[0];
	T2 += dg_dx1[1][2]*u[1]*u[1] + half*dg_dx2[2][2]*u[2]*u[2] + dg_dx3[3][2]*u[3]*u[3];
	T2 += (half*dg_dx1[2][2] + dg_dx2[1][2])*u[1]*u[2] 
	    + (dg_dx1[3][2] + dg_dx3[1][2])*u[1]*u[3] + (dg_dx2[3][2] + half*dg_dx3[2][2])*u[2]*u[3];

	T3 += dg_dx1[0][3]*u[0]*u[1] + dg_dx2[0][3]*u[0]*u[2] + dg_dx3[0][3]*u[3]*u[0];
	T3 += dg_dx1[1][3]*u[1]*u[1] + dg_dx2[2][3]*u[2]*u[2] + half*dg_dx3[3][3]*u[3]*u[3];
	T3 += (dg_dx1[2][3] + dg_dx2[1][3])*u[1]*u[2] 
	   + (half*dg_dx1[3][3] + dg_dx3[1][3])*u[1]*u[3] + (half*dg_dx2[3][3] + dg_dx3[2][3])*u[2]*u[3];

	// Static metric has no T4_0
	// Terms where the derivative has the same index as the resulting component
	T4_1, T4_2, T4_3 = 0.0;
	for ( int i = 0; i<4; ++i){
	    for (int j = 0; j<4; ++j){
		half = (i == j) ? 0.5 : 1.0;
	    	T4_1 += half*dg_dx1[i][j]*u[i]*u[j];
	    	T4_2 += half*dg_dx2[i][j]*u[i]*u[j];
	    	T4_3 += half*dg_dx3[i][j]*u[i]*u[j];
	    }
	}
	        
	//Convert to contravariant for ease of passage to local variables
	acc[0] = - T0*gupper[0][0]
		 + (T4_1 - T1)*gupper[0][1] 
	         + (T4_2 - T2)*gupper[0][2] 
	         + (T4_3 - T3)*gupper[0][3];
	acc[1] = - T0*gupper[1][0]
		 + (T4_1 - T1)*gupper[1][1] 
	         + (T4_2 - T2)*gupper[1][2] 
	         + (T4_3 - T3)*gupper[1][3];
	acc[2] = - T0*gupper[2][0]
		 + (T4_1 - T1)*gupper[2][1] 
	         + (T4_2 - T2)*gupper[2][2] 
	         + (T4_3 - T3)*gupper[2][3];
	acc[3] = - T0*gupper[3][0]
		 + (T4_1 - T1)*gupper[3][1] 
		 + (T4_2 - T2)*gupper[3][2] 
		 + (T4_3 - T3)*gupper[3][3];
}

KOKKOS_INLINE_FUNCTION
void Particles::ComputeLorentzFactorGR( const Real * x, const Real * v, Real * u, Real * g_Lor ){

	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	Real glower[4][4], gupper[4][4]; // Metric components
	Real v0, a;				       
	
	ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 

	u[0] = 1.0;
	u[1] = v[0]/sqrt(glower[1][1]);
	u[2] = v[1]/sqrt(glower[2][2]);
	u[3] = v[2]/sqrt(glower[3][3]);

	a = 0.0;
	for (int j=0; j<4; ++j){
		for (int i=0; i<4; ++i){
		       	a += glower[j][i]*u[i]*u[j] ;
	       	}
	}
	std::cout << "a: " << a << " glower[0][0]: " << glower[0][0] << std::endl;
	v0 = sqrt( 1.0/a );
	u[0] = v0;
	u[1] *= v0;
	u[2] *= v0;
	u[3] *= v0;

	*g_Lor = v0;
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
void Particles::ComputeJacobianTerms(const Real * x, const Real * u, Real JT[][4], Real * dx_g_Lor, Real * du_g_Lor){

	const bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	const Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	Real glower[4][4], gupper[4][4]; // Metric components
	Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4]; // Metric derivatives
	Real auxM[4][4], auxN[4][4];	
	//
	Real half = 0.5;
	ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
	ComputeMetricDerivatives(x[0],x[1],x[2], is_minkowski, spin, dg_dx1, dg_dx2, dg_dx3); 
		     
	for (int j=0; j<4; ++j){
		auxN[j][0] = dg_dx1[j][0]*u[1] + dg_dx2[j][0]*u[2] + dg_dx3[j][0]*u[3];
		auxN[j][1] = dg_dx1[j][1]*u[1] + dg_dx2[j][1]*u[2] + dg_dx3[j][1]*u[3];
		for (int i=0; i<4; ++i) { auxN[j][1] += dg_dx1[j][i]*u[i] ; }
		auxN[j][2] = dg_dx1[j][2]*u[1] + dg_dx2[j][2]*u[2] + dg_dx3[j][2]*u[3];
		for (int i=0; i<4; ++i) { auxN[j][2] += dg_dx2[j][i]*u[i] ; }
		auxN[j][3] = dg_dx1[j][3]*u[1] + dg_dx2[j][3]*u[2] + dg_dx3[j][3]*u[3];
		for (int i=0; i<4; ++i) { auxN[j][3] += dg_dx3[j][i]*u[i] ; }
	}

	for (int j = 0; j<4; ++j) {
		auxM[1][j] = auxN[1][j];
        	for (int i = 0; i<4; ++i) {
		        half = (i == j+1) ? 0.5 : 1.0;
			auxM[1][j] -= half*dg_dx2[j][i]*u[i];
	       	}
	}
	for (int j = 0; j<4; ++j) {
		auxM[2][j] = auxN[2][j];
        	for (int i = 0; i<4; ++i) {
		        half = (i == j+1) ? 0.5 : 1.0;
			auxM[2][j] -= half*dg_dx3[j][i]*u[i];
	       	}
	}
	for (int j = 0; j<4; ++j) {
		auxM[3][j] = auxN[3][j];
        	for (int i = 0; i<4; ++i) {
		        half = (i == j+1) ? 0.5 : 1.0;
			auxM[3][j] -= half*dg_dx3[j][i]*u[i];
	       	}
	}
	
	//Raise indices
	for (int i=0; i<4; ++i){
		for (int j=0; j<4; ++j){
			JT[i][j] = 0.0;
			for (int k=0; k<4; ++k){
				JT[i][j] += gupper[i][k]*auxM[k][j];
			}
		}
	}
	        
	dx_g_Lor[0] = 0.0;
	du_g_Lor[0] = 0.0;
	dx_g_Lor[1] = 0.0;
	du_g_Lor[1] = 0.0;
	dx_g_Lor[2] = 0.0;
	du_g_Lor[2] = 0.0;
	for (int i1=0; i1<4; ++i1){
		for (int i2=0; i2<4; ++i2){
			dx_g_Lor[0] += dg_dx1[i1][i2]*u[i1]*u[i1];
			dx_g_Lor[1] += dg_dx2[i1][i2]*u[i1]*u[i1];
			dx_g_Lor[2] += dg_dx3[i1][i2]*u[i1]*u[i1];
		}
		du_g_Lor[0] += 2.0*glower[1][i1]*u[i1];
		du_g_Lor[1] += 2.0*glower[2][i1]*u[i1];
		du_g_Lor[2] += 2.0*glower[3][i1]*u[i1];
	}
}
} // namespace particles
