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
      BorisStep(dt_, pr, pi, multi_d, three_d);
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
void Particles::BorisStep( const Real dt, DvceArray2D<Real> &pr, const DvceArray2D<int> &pi, const bool multi_d, const bool three_d ){
	
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
      par_for("part_boris",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {

	Real up[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	Real x1,x2,x3; //Half-step increment. Index 1... stands for dimension (0 is time).
	// Get metric components at new location x1,x2,x3
	Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric components in ADM formalism
					       // (remember: sqrt(-1/gupper[0][0]) = alpha, glower[0][i] = beta[i])
	ComputeMetricAndInverse(pr(IPX,p),pr(IPY,p),pr(IPZ,p), is_minkowski, spin, glower, gupper); 
	// Compute 3x3 ADM spatial metric from covariant metric 
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_upper[i1][i2] = gupper[i1+1][i2+1] + gupper[0][i2+1]*gupper[i1+1][0];
		}
	}
	Real alpha = sqrt(-1.0/gupper[0][0]);
	Real g_Lor = 0.0; //Lorentz gamma factor (though it's not the usual expression it plays a similar role in these equations)
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*up[i1]*up[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);
        //std::cout << "g_Lor " << p << " " << g_Lor << std::endl;

	x1 = pr(IPX,p) + dt/(2.0*g_Lor)*up[0];
        if (multi_d) { x2 = pr(IPY,p) + dt/(2.0*g_Lor)*up[1]; }
        if (three_d) { x3 = pr(IPZ,p) + dt/(2.0*g_Lor)*up[2]; }

        //std::cout << "x123 " << global_variable::my_rank << " " << pi(PTAG,p) << " " << x1 << " " << x2 << " " << x3 << std::endl;
	Real uE[3]; //Evolution of the velocity due to the electric field (first half). Index 1... stands for dimension (0 is time).
	Real uB[3]; //Evolution of the velocity due to the magnetic field. Index 1... stands for dimension (0 is time).

        int m = pi(PGID,p) - gids;
        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
	int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
	int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
	/***
	if ( (ip*jp*kp < 0.0) || (ip > 33) || (jp > 33) || (kp > 33) || (m > nmb1)){
	std::cout << global_variable::my_rank <<std::endl;
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

	uE[0] = up[0] + dt*pr(IPC,p)*alpha/(2.0*pr(IPM,p))*E[0];
        if (multi_d) { uE[1] = up[1] + dt*pr(IPC,p)*alpha/(2.0*pr(IPM,p))*E[1]; }
        if (three_d) { uE[2] = up[2] + dt*pr(IPC,p)*alpha/(2.0*pr(IPM,p))*E[2]; }

	// Get metric components at new location x1,x2,x3
	ComputeMetricAndInverse(x1,x2,x3, is_minkowski, spin, glower, gupper); 
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		ADM_upper[i1][i2] = gupper[i1+1][i2+1] + gupper[0][i2+1]*gupper[i1+1][0];
		}
	}
	alpha = sqrt(-1.0/gupper[0][0]);
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

	// Finally update velocity
	pr(IPVX,p) = uB[0] + dt*pr(IPC,p)*alpha/(2.0*pr(IPM,p))*E[0] ;
	pr(IPVY,p) = uB[1] + dt*pr(IPC,p)*alpha/(2.0*pr(IPM,p))*E[1] ;
	pr(IPVZ,p) = uB[2] + dt*pr(IPC,p)*alpha/(2.0*pr(IPM,p))*E[2] ;

        //std::cout << "pr1 " << p << " " << pr(IPVX,p)<<//std::endl;
      // Second half-step in space
	g_Lor = 0.0; //Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*pr(2*i1+1,p)*pr(2*i2+1,p);
		}
	}
	g_Lor = sqrt(1.0 + g_Lor);

	pr(IPX,p) = x1 + dt/(2.0*g_Lor)*pr(IPVX,p);
        if (multi_d) { pr(IPY,p) = x2 + dt/(2.0*g_Lor)*pr(IPVY,p); }
        if (three_d) { pr(IPZ,p) = x3 + dt/(2.0*g_Lor)*pr(IPVZ,p); }
        //std::cout << "pp1 " << p << " " << pr(IPX,p)<< " " << pr(IPY,p)<< " " << pr(IPZ,p)<<//std::endl;
      });
      return;
}
} // namespace particles
