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
  auto gide = pmy_pack->gide;

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
      BorisStep(dt_, multi_d, three_d);
    break;
    
    case ParticlesPusher::full_gr:
      BorisStep(dt_/2.0, multi_d, three_d);
      GeodesicIterations(dt_, multi_d, three_d);
      BorisStep(dt_/2.0, multi_d, three_d);
    break;

    default:
    break;
  }

  return TaskStatus::complete;
}

//Provide dt as input parameter in order to be able to use this function
//also for half-steps
//Largely implemented following Ripperda et al. 2018 (https://doi.org/10.3847/1538-4365/aab114)
KOKKOS_INLINE_FUNCTION
void Particles::BorisStep( const Real dt, const bool multi_d, const bool three_d ){
	
	auto &npart = nprtcl_thispack;
	auto &pi = prtcl_idata;
	auto &pr = prtcl_rdata;
	auto &b0_ = pmy_pack->pmhd->b0;
	auto &e0_ = pmy_pack->pmhd->efld;
	auto &indcs = pmy_pack->pmesh->mb_indcs;
	int is = indcs.is;
	int js = indcs.js;
	int ks = indcs.ks;
	int ie = indcs.ie;
	int je = indcs.je;
	int ke = indcs.ke;
	auto &mbsize = pmy_pack->pmb->mb_size;
	auto gids = pmy_pack->gids;
	bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	Real spin = pmy_pack->pcoord->coord_data.bh_spin;
	int nmb1 = pmy_pack -> nmb_thispack-1;

      // First half-step in space
      par_for("part_boris",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
      
	Real up[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	Real x1,x2,x3; //Half-step increment. Index 1... stands for dimension (0 is time).
	// Push in local frame
	x1 = pr(IPX,p) + dt/(2.0)*up[0];
        if (multi_d) { x2 = pr(IPY,p) + dt/(2.0)*up[1]; }
        if (three_d) { x3 = pr(IPZ,p) + dt/(2.0)*up[2]; }
	// Get metric components at new location x1,x2,x3
	Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric 
					       // (remember: sqrt(-1/gupper[0][0]) = alpha, glower[0][i] = beta[i])
	ComputeMetricAndInverse(pr(IPX,p),pr(IPY,p),pr(IPZ,p), is_minkowski, spin, glower, gupper); 
	//Boost the local velocity to the covariant one using the "usual" definition of Lorentz factor
	Real g_Lor = 0.0;
	for (int i1 = 0; i1<3; ++i1){ g_Lor += SQR(up[i1]); }
	g_Lor = 1.0/sqrt(1.0 - g_Lor);
	for (int i1 = 0; i1<3; ++i1){ up[i1] *= g_Lor; }
	// Compute 3x3 ADM spatial metric from covariant metric 
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_upper[i1][i2] = gupper[i1+1][i2+1] + gupper[0][i2+1]*gupper[i1+1][0];
		}
	}
	//Use deinition of the Lorentz factor in ADM formalism
	g_Lor = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*up[i1]*up[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
        //std::cout << "g_Lor " << p << " " << g_Lor << std::endl;

        //std::cout << "x123 " << global_variable::my_rank << " " << pi(PTAG,p) << " " << x1 << " " << x2 << " " << x3 << std::endl;
	Real uE[3]; //Evolution of the velocity due to the electric field (first half). Index 1... stands for dimension (0 is time).
	Real uB[3]; //Evolution of the velocity due to the magnetic field. Index 1... stands for dimension (0 is time).

        int m = pi(PGID,p) - gids;
        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
	int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
	int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
	/***
	if ( (ip*jp*kp < 0.0) || (ip > ie) || (jp > je) || (kp > ke) || (m > nmb1)){
	std::cout << global_variable::my_rank <<std::endl;
	std::cout << "gids " << gids << " " << pi(PGID,p) << std::endl; 
	std::cout << "mbsize " << m << " " << mbsize.d_view(m).x2min << " " << mbsize.d_view(m).dx2 << std::endl; 
	std::cout << "pr " << pi(PTAG,p) << " " << pr(IPX,p) << " " << pr(IPY,p) << " " << pr(IPZ,p) << std::endl; 
	std::cout << "Idx i "  << pi(PTAG,p) << " "<< is << " " << ip << " " << ie << std::endl; 
	std::cout << "Idx j "  << pi(PTAG,p) << " "<< js << " " << jp << " " << je << std::endl; 
	std::cout << "Idx k "  << pi(PTAG,p) << " "<< ks << " " << kp << " " << ke << std::endl; 
	}
	***/
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
	E[0] = e0_.x1e(m, kp, jp, ip) + (x1 - x1v)*(e0_.x1e(m, kp, jp, ip+1) - e0_.x1e(m, kp, jp, ip))/Dx;
	E[1] = e0_.x2e(m, kp, jp, ip) + (x2 - x2v)*(e0_.x2e(m, kp, jp+1, ip) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[2] = e0_.x3e(m, kp, jp, ip) + (x3 - x3v)*(e0_.x3e(m, kp+1, jp, ip) - e0_.x3e(m, kp, jp, ip))/Dz;

	uE[0] = up[0] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[0];
        if (multi_d) { uE[1] = up[1] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[1]; }
        if (three_d) { uE[2] = up[2] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[2]; }

	// Get metric components at new location x1,x2,x3
	ComputeMetricAndInverse(x1,x2,x3, is_minkowski, spin, glower, gupper); 
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_upper[i1][i2] = gupper[i1+1][i2+1] + gupper[0][i2+1]*gupper[i1+1][0];
		}
	}
	g_Lor = 0.0; //Intermediate Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*uE[i1]*uE[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
        // Interpolate Magnetic Field at new particle location x1, x2, x3
	// Store it in an array for convenience 
	Real B[3] = {0.0, 0.0, 0.0};
	B[0] = b0_.x1f(m, kp, jp, ip) + (x1 - x1v)*(b0_.x1f(m, kp, jp, ip+1) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[1] = b0_.x2f(m, kp, jp, ip) + (x2 - x2v)*(b0_.x2f(m, kp, jp+1, ip) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[2] = b0_.x3f(m, kp, jp, ip) + (x3 - x3v)*(b0_.x3f(m, kp+1, jp, ip) - b0_.x3f(m, kp, jp, ip))/Dz;

	// if (p == 53){
        // std::cout << "B " << p << " " << B[0] << " " << B[1] << " " << B[2]<<std::endl;
	// }
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

        //std::cout << "vec_ut " << p << " " << vec_ut[0] << " " << vec_ut[1] << " " << vec_ut[2]<<//std::endl;
	uB[0] = uE[0] + 2.0/(1.0+mod_t_sqr)*( (uE[1] + vec_ut[1])*t[2] - (uE[2] + vec_ut[2])*t[1] );
        if (multi_d) { uB[1] = uE[1] + 2.0/(1.0+mod_t_sqr)*( (uE[2] + vec_ut[2])*t[0] - (uE[0] + vec_ut[0])*t[2] ); }
        if (three_d) { uB[2] = uE[2] + 2.0/(1.0+mod_t_sqr)*( (uE[0] + vec_ut[0])*t[1] - (uE[1] + vec_ut[1])*t[0] ); }

	g_Lor = 0.0; //Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*uB[i1]*uB[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
	// Finally update velocity in local space
	pr(IPVX,p) = (uB[0] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[0])/g_Lor ;
	pr(IPVY,p) = (uB[1] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[1])/g_Lor ;
	pr(IPVZ,p) = (uB[2] + dt*pr(IPC,p)/(2.0*pr(IPM,p))*E[2])/g_Lor ;

        //std::cout << "pr1 " << p << " " << pr(IPVX,p)<<//std::endl;

	pr(IPX,p) = x1 + dt/(2.0)*pr(IPVX,p);
        if (multi_d) { pr(IPY,p) = x2 + dt/(2.0)*pr(IPVY,p); }
        if (three_d) { pr(IPZ,p) = x3 + dt/(2.0)*pr(IPVZ,p); }
        //std::cout << "pp1 " << p << " " << pr(IPX,p)<< " " << pr(IPY,p)<< " " << pr(IPZ,p)<<//std::endl;
      });
      return;
}

//Provide dt as input parameter in order to be able to use this function
//also for half-steps
//Largely implemented following Bacchini et al. 2020 (https://doi.org/10.3847/1538-4365/abb604)
KOKKOS_INLINE_FUNCTION
void Particles::GeodesicIterations( const Real dt, const bool multi_d, const bool three_d ){
	auto &pi = prtcl_idata;
	auto &pr = prtcl_rdata;
	bool is_minkowski = pmy_pack->pcoord->coord_data.is_minkowski;
	Real spin = pmy_pack->pcoord->coord_data.bh_spin;

      // First attempt: not iterative, approximate
      par_for("part_fullgr",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {

        // Iterate per particle such that those that converge quicker don't go through as many iterations
	int n_iter = 0;
	Real rest[3] = {1.0,1.0,1.0};
	Real x_curr[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)}; 
	// u0 depends on nature of the particles (massive/massless)
	Real u_curr[4] = {1.0, pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)}; //c = 1??
	Real glower[4][4], gupper[4][4]; // Metric components
	Real dg_dx1[4][4], dg_dx2[4][4], dg_dx3[4][4]; // Metric derivatives
	Real T1, T2, T3, T4_1, T4_2, T4_3; // Auxiliary variables
	Real accel[4] = {0.0, 0.0, 0.0, 0.0}; //c = 1??
	Real alpha;

//	while( n_iter < max_iter ){

	// Get metric components at new location x1,x2,x3
	ComputeMetricAndInverse(x_curr[0],x_curr[1],x_curr[2], is_minkowski, spin, glower, gupper); 
	ComputeMetricDerivatives(x_curr[0],x_curr[1],x_curr[2], is_minkowski, spin, dg_dx1, dg_dx2, dg_dx3); 
	alpha = sqrt(-1.0/gupper[0][0]);
	for (int i1 = 0; i1 < 4; ++i1){
		u_curr[i1] *= alpha;
	}
	T1, T2, T3 = 0.0;
	    	       // Following ex_currression technically has a factor 2 before each term,
	    	       // but this gets multiplied by 1/2 in the end, so just skip it.
	    	       // Also, with static metric terms with g_0i don't contribute.
	T1 += dg_dx1[0][1]*u_curr[0]*u_curr[1] + dg_dx2[0][1]*u_curr[0]*u_curr[2] + dg_dx3[0][1]*u_curr[3]*u_curr[0];
	T1 += dg_dx1[1][1]*u_curr[1]*u_curr[1] + dg_dx2[2][1]*u_curr[2]*u_curr[2] + dg_dx3[3][1]*u_curr[3]*u_curr[3];
	T1 += (dg_dx1[2][1] + dg_dx2[1][1])*u_curr[1]*u_curr[2] 
	     + (dg_dx1[3][1] + dg_dx3[1][1])*u_curr[1]*u_curr[3] + (dg_dx2[3][1] + dg_dx3[2][1])*u_curr[2]*u_curr[3];

	T2 += dg_dx1[0][2]*u_curr[0]*u_curr[1] + dg_dx2[0][2]*u_curr[0]*u_curr[2] + dg_dx3[0][2]*u_curr[3]*u_curr[0];
	T2 += dg_dx1[1][2]*u_curr[1]*u_curr[1] + dg_dx2[2][2]*u_curr[2]*u_curr[2] + dg_dx3[3][2]*u_curr[3]*u_curr[3];
	T2 += (dg_dx1[2][2] + dg_dx2[1][2])*u_curr[1]*u_curr[2] 
	    + (dg_dx1[3][2] + dg_dx3[1][2])*u_curr[1]*u_curr[3] + (dg_dx2[3][2] + dg_dx3[2][2])*u_curr[2]*u_curr[3];

	T3 += dg_dx1[0][3]*u_curr[0]*u_curr[1] + dg_dx2[0][3]*u_curr[0]*u_curr[2] + dg_dx3[0][3]*u_curr[3]*u_curr[0];
	T3 += dg_dx1[1][3]*u_curr[1]*u_curr[1] + dg_dx2[2][3]*u_curr[2]*u_curr[2] + dg_dx3[3][3]*u_curr[3]*u_curr[3];
	T3 += (dg_dx1[2][3] + dg_dx2[1][3])*u_curr[1]*u_curr[2] 
	   + (dg_dx1[3][3] + dg_dx3[1][3])*u_curr[1]*u_curr[3] + (dg_dx2[3][3] + dg_dx3[2][3])*u_curr[2]*u_curr[3];

	T4_1, T4_2, T4_3 = 0.0;
	for ( int i = 0; i<4; ++i){
	    for (int j = 0; j<4; ++j){
	    	T4_1 += dg_dx1[i][j]*u_curr[i]*u_curr[j];
	    	T4_2 += dg_dx2[i][j]*u_curr[i]*u_curr[j];
	    	T4_3 += dg_dx3[i][j]*u_curr[i]*u_curr[j];
	    }
	}
	        
	/***/
	accel[0] = (T4_1 - T1)*gupper[0][1] 
	        + (T4_2 - T2)*gupper[0][2] 
	        + (T4_3 - T3)*gupper[0][3];
	/***/
	accel[1] = (T4_1 - T1)*gupper[1][1] 
	        + (T4_2 - T2)*gupper[1][2] 
	        + (T4_3 - T3)*gupper[1][3];
	accel[2] = (T4_1 - T1)*gupper[2][1] 
	        + (T4_2 - T2)*gupper[2][2] 
	        + (T4_3 - T3)*gupper[2][3];
	accel[3] = (T4_1 - T1)*gupper[3][1] 
		+ (T4_2 - T2)*gupper[3][2] 
		+ (T4_3 - T3)*gupper[3][3];
	
	u_curr[1] += dt*accel[1]; 
        if (multi_d) { u_curr[2] += dt*accel[2]; }
        if (three_d) { u_curr[3] += dt*accel[3]; }
	x_curr[0] += dt/2.0*u_curr[1];
        if (multi_d) { x_curr[1] += dt/2.0*u_curr[2]; }
        if (three_d) { x_curr[2] += dt/2.0*u_curr[3]; }
	
        //std::cout << "pp1 " << p << " " << pr(IPX,p)<< " " << pr(IPY,p)<< " " << pr(IPZ,p)<<//std::endl;
	rest[0] = x_curr[0] - pr(IPX, p) - dt/2.0*u_curr[1];
	rest[1] = x_curr[1] - pr(IPY, p) - dt/2.0*u_curr[2];
	rest[2] = x_curr[2] - pr(IPZ, p) - dt/2.0*u_curr[3];
	++n_iter;
//	if ( sqrt(SQR(rest[0]) + SQR(rest[1]) + SQR(rest[2])) > iter_tolerance ){
		//Compute values for next iteration with gradient descent

//	} else {
		//Stop iterations
//		break;
//	}
//	}
//	if (n_iter == (max_iter - 1 )) { std::cout << "Limit of implicit iterations reached on particle " << pi(PTAG,p) << " on rank " << global_variable::my_rank << std::endl; }

	// Done with iterations, update ``true'' values
	pr(IPVX,p) = u_curr[1]/alpha;
        if (multi_d) { pr(IPVY,p) = u_curr[2]/alpha; }
	if (three_d) { pr(IPVZ,p) = u_curr[3]/alpha; }
	pr(IPX,p) += dt*pr(IPVX,p);
        if (multi_d) { pr(IPY,p) += dt*pr(IPVY,p); }
        if (three_d) { pr(IPZ,p) += dt*pr(IPVZ,p); }
      });
      return;
}
} // namespace particles
