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
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

//----------------------------------------------------------------------------------------
//! \fn  void GetUpperAdmMetric
//  \brief
KOKKOS_INLINE_FUNCTION
void GetUpperAdmMetric( const Real inputMat[][4], Real outputMat[][3] ){
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		outputMat[i1][i2] = inputMat[i1+1][i2+1] - inputMat[0][i2+1]*inputMat[i1+1][0]/inputMat[0][0];
		}
	}
}

//----------------------------------------------------------------------------------------
//! \fn  void SingleTermHelper_Position
//! \brief
//! Helper function to reduce amount of repetition in HamiltonEquation_Position
KOKKOS_INLINE_FUNCTION
void SingleTermHelper_Position(const Real * u_0, const Real * u_1, const Real alpha, const Real beta, const Real ADM[][3], const int dir, Real * H){

	Real massive = 1.0; //todo for photons/massless particles this needs to be 0: condition on ptype
	Real U_1, U_0, perp;
	int i1, i2;
	if (dir == 0) { i1 = 1; i2 = 2; }
	else if (dir == 1) { i1 = 0; i2 = 2; }
	else if (dir == 2) { i1 = 0; i2 = 1; }

	perp = ADM[i1][i1]*SQR(u_0[i1]) + ADM[i2][i2]*SQR(u_0[i2]) + 2.0*ADM[i1][i2]*u_0[i1]*u_0[i2];
	U_1 = ADM[dir][dir]*SQR(u_1[dir]) + 2.0*ADM[dir][i1]*u_1[dir]*u_1[i1] + 2.0*ADM[dir][i2]*u_1[dir]*u_1[i2];
	U_1 += perp;	
	U_1 = sqrt(U_1 + massive); 
	U_0 = ADM[dir][dir]*SQR(u_0[dir]) + 2.0*ADM[dir][i1]*u_0[dir]*u_0[i1] + 2.0*ADM[dir][i2]*u_0[dir]*u_0[i2];
	U_0 += perp;	
	U_0 = sqrt(U_0 + massive); 
	// Alpha term
	*H += alpha*(
		ADM[dir][dir]*(u_1[dir]+u_0[dir]) + 2.0*ADM[dir][i1]*u_1[i1] + 2.0*ADM[dir][i2]*u_1[i2]
	       	)/(U_0 + U_1);
	// Beta term
	*H += beta;
}

//----------------------------------------------------------------------------------------
//! \fn  void HamiltonEquation_Position
//  \brief
// Following function largely implented based on the appendix in Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca
// It's an expression of the derivative of the ADM Hamiltonian for geodesics with respect to the velocity
// That prevents singularities for small increments and is used to compute the time derivative of the position
// Would be nice to come up with a way to compute these terms algorithmically, rather than by hard coding, but thus far I wasn't able to
KOKKOS_INLINE_FUNCTION
void HamiltonEquation_Position(const Real * x_0, const Real * x_1, const Real * u_0, const Real * u_1, const Real spin, Real * H){

	const bool is_minkowski = false; //Since this function is only for the GR pusher, this can be kept as a ``constant''
	Real glower[4][4], gupper[4][4], ADM_g[3][3]; // Metric components
	Real aux_u0[3], aux_u1[3]; //Send these to helper function on a 'per-case' basis
				   // Assembling these requires referring to the paper for the correct combinations

	for (int i=0; i<3; ++i){ H[i] = 0.0; }

	//Metric with all old positions
	//Common to all terms
	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );
	
        aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

        aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

        aux_u1[0] = u_0[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);
	
	//Metric with all new positions
	//Common to all terms
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_1[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

        aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

        aux_u0[0] = u_1[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_0[0], x_1[1], x_1[2]
	//Common to terms 0 and 2
	ComputeMetricAndInverse(x_0[0],x_1[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

        aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_1[0], x_0[1], x_0[2]
	//Common to terms 0 and 1
	ComputeMetricAndInverse(x_1[0],x_0[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);
	
        aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	//
	//Metric with x_1[0], x_0[1], x_1[2]
	//Common to terms 0 and 1
	ComputeMetricAndInverse(x_1[0],x_0[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);
	
        aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

	//
	//Metric with x_0[0], x_0[1], x_1[2]
	//Common to terms 0 and 2
	ComputeMetricAndInverse(x_0[0],x_0[1],x_1[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_1[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][1]/gupper[0][0], ADM_g, 0, &H[0]);

        aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_0[0]; aux_u1[1] = u_0[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_1[0], x_1[1], x_0[2]
	//Common to terms 1 and 2
	ComputeMetricAndInverse(x_1[0],x_1[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_1[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

        aux_u0[0] = u_1[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_1[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);

	//
	//Metric with x_0[0], x_1[1], x_0[2]
	//Common to terms 1 and 2
	ComputeMetricAndInverse(x_0[0],x_1[1],x_0[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_g );

        aux_u0[0] = u_0[0]; aux_u0[1] = u_0[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_0[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][2]/gupper[0][0], ADM_g, 1, &H[1]);

        aux_u0[0] = u_0[0]; aux_u0[1] = u_1[1]; aux_u0[2] = u_0[2];
        aux_u1[0] = u_0[0]; aux_u1[1] = u_1[1]; aux_u1[2] = u_1[2];
        SingleTermHelper_Position(aux_u0, aux_u1, sqrt(-1.0/gupper[0][0]), gupper[0][3]/gupper[0][0], ADM_g, 2, &H[2]);
	
	for (int i=0; i<3; ++i){ H[i] /= 6.0; }
}
//----------------------------------------------------------------------------------------
//! \fn  void ComputeAndAddSingleTerm_Velocity
//  \brief
KOKKOS_INLINE_FUNCTION
void ComputeAndAddSingleTerm_Velocity(const Real gu_0[4][4], const Real gu_1[4][4], const Real * u, Real * H){
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
                beta_t += ( -gu_1[0][i1+1]/gu_1[0][0] + gu_0[0][i1+1]/gu_0[0][0])*u[i1];
        }

        *H -= 0.5*(U_0 + U_1)*( sqrt(-1.0/gu_1[0][0]) - sqrt(-1.0/gu_0[0][0]) ) ;
        *H -= 0.5*aux*( sqrt(-1.0/gu_1[0][0]) + sqrt(-1.0/gu_0[0][0]) )/(U_1 + U_0) ;
        *H += beta_t ;
}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeAndAddSingleTerm_Velocity
//  \brief      
// Overload previous function to use the derivative of the metric if needed
KOKKOS_INLINE_FUNCTION
void ComputeAndAddSingleTerm_Velocity(const Real gu_0[4][4], const Real gu_1[4][4], const Real g_der[4][4], const Real * u, Real * H){
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
                beta_t -= (g_der[0][i1+1] - gu_0[0][i1+1]*g_der[0][0]/gu_0[0][0])/gu_0[0][0]*u[i1];
        }
        
        *H -= 0.5*(U_0 + U_1)*( 0.5*sqrt(-1.0/gu_0[0][0])*g_der[0][0]/fabs(gu_0[0][0]) ) ;
        *H -= 0.5*aux*( sqrt(-1.0/gu_1[0][0]) + sqrt(-1.0/gu_0[0][0]) )/(U_1 + U_0) ;
        *H += beta_t ;
}       

//----------------------------------------------------------------------------------------
//! \fn  void SingleTermHelper_Velocity
//! \brief
//! Helper function to reduce amount of repetition in HamiltonEquation_Velocity
KOKKOS_INLINE_FUNCTION
void SingleTermHelper_Velocity(const Real * x_0, const Real * x_1, const Real * u, const bool use_der, const int dir, const Real x_step, const Real spin, Real * H){

	const bool is_minkowski = false; //Since this function is only for the GR pusher, this can be kept as a ``constant''
	Real gl[4][4], gu_0[4][4], gu_1[4][4]; // Metric components
	Real aux_gu_0[4][4], aux_gu_1[4][4]; // Metric components
	int dir_fac[3] = {0,0,0}; 
	dir_fac[dir] = 1; // Use this to perform step only in direction of interest

	ComputeMetricAndInverse(x_0[0],x_0[1],x_0[2], is_minkowski, spin, gl, gu_0); 
	ComputeMetricAndInverse(x_1[0],x_1[1],x_1[2], is_minkowski, spin, gl, gu_1); 

	if (use_der){
		ComputeMetricAndInverse(x_0[0]-x_step*dir_fac[0], x_0[1]-x_step*dir_fac[1], x_0[2]-x_step*dir_fac[2],
			       	is_minkowski, spin, gl, aux_gu_1); 
		ComputeMetricAndInverse(x_0[0]+x_step*dir_fac[0], x_0[1]+x_step*dir_fac[1], x_0[2]+x_step*dir_fac[2],
			       	is_minkowski, spin, gl, aux_gu_0); 
	        for(int i=0; i<4; ++i){
			for(int j=0; j<4; ++j){
				aux_gu_0[i][j] -= aux_gu_1[i][j];
				aux_gu_0[i][j] /= (2.0*x_step);
			}
		}	
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, aux_gu_0, u, H);
	}else{
		ComputeAndAddSingleTerm_Velocity(gu_0, gu_1, u, H);
	}
									     //
}

//----------------------------------------------------------------------------------------
//! \fn  void HamiltonEquation_Velocity
//!  \brief
//! Following function largely implented based on the appendix in Bacchini et al. 2018 (doi.org/10.3847/1538-4365/aac9ca
//! It's an expression of the derivative of the ADM Hamiltonian for geodesics with respect to the position
//! That prevents singularities for small increments and is used to compute the time derivative of the velocity
//! Would be nice to come up with a way to compute these terms algorithmically, rather than by hard coding, but thus far I wasn't able to
KOKKOS_INLINE_FUNCTION
void HamiltonEquation_Velocity(const Real * x_0, const Real * x_1, const Real * u_0, const Real * u_1, const Real x_step, const Real spin, const Real it_tol, Real * H){

	Real incr[3], u[3];
       	// Difference between old and new points might be too small
	// Use drivative instead
	bool use_derivative[3]; // array to determine whether or not the derivative is needed
	Real aux_x0[3], aux_x1[3]; // Send these to helper function, but assemble them case by case following energy conserving scheme

	for (int i=0; i<3; ++i){ H[i] = 0.0; }
	incr[0] = x_1[0] - x_0[0];
	use_derivative[0] = (fabs(incr[0]) < sqrt(it_tol)) ? true : false;
	incr[1] = x_1[1] - x_0[1];
	use_derivative[1] = (fabs(incr[1]) < sqrt(it_tol)) ? true : false;
	incr[2] = x_1[2] - x_0[2];
	use_derivative[2] = (fabs(incr[2]) < sqrt(it_tol)) ? true : false;

	//Terms with all old velocities
	//Common to all directions
	u[0] = u_0[0]; u[1] = u_0[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);
	
	aux_x1[0] = x_0[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with all new velocities
	//Common to all directions
	u[0] = u_1[0]; u[1] = u_1[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);

	aux_x0[0] = x_1[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for y and z
	//Common to terms 0 and 2
	u[0] = u_0[0]; u[1] = u_1[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);
	
	//Terms with new velocities for x and y
	//Common to terms 1 and 2
	u[0] = u_1[0]; u[1] = u_1[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);
	
	aux_x0[0] = x_1[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for x and z
	//Common to terms 0 and 1
	u[0] = u_1[0]; u[1] = u_0[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);
	
	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);

	//Terms with new velocities for y only
	//Common to terms 1 and 2
	u[0] = u_0[0]; u[1] = u_1[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);
	
	aux_x0[0] = x_0[0]; aux_x0[1] = x_1[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for z only
	//Common to terms 0 and 2
	u[0] = u_0[0]; u[1] = u_0[1]; u[2] = u_1[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_1[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);
	
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_0[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_1[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[2], 2, x_step, spin, &H[2]);

	//Terms with new velocities for x only
	//Common to terms 0 and 1
	u[0] = u_1[0]; u[1] = u_0[1]; u[2] = u_0[2]; 
	aux_x0[0] = x_0[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_0[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[0], 0, x_step, spin, &H[0]);

	aux_x0[0] = x_1[0]; aux_x0[1] = x_0[1]; aux_x0[2] = x_0[2];
	aux_x1[0] = x_1[0]; aux_x1[1] = x_1[1]; aux_x1[2] = x_0[2];
	SingleTermHelper_Velocity(aux_x0, aux_x1, u, use_derivative[1], 1, x_step, spin, &H[1]);

	for (int i=0; i<3; ++i){
	       	H[i] /= 6.0;
		// When using the derivative the division by the increment is done to compute the derivatives
		if (!use_derivative[i]) { H[i] /= incr[i]; }

       	}
}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeDeterminant3
//  \brief Compute the determinant of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
void ComputeDeterminant3( const Real inputMat[][3], Real &determinant ){

	determinant = inputMat[0][0]*inputMat[1][1]*inputMat[2][2] + inputMat[0][1]*inputMat[1][2]*inputMat[2][0]
		+ inputMat[1][0]*inputMat[2][1]*inputMat[0][2]
		- inputMat[2][0]*inputMat[1][1]*inputMat[0][2] - inputMat[1][0]*inputMat[0][1]*inputMat[2][2]
		- inputMat[2][1]*inputMat[1][2]*inputMat[1][1];

}

//----------------------------------------------------------------------------------------
//! \fn  void ComputeInverseMatrix3
//  \brief Compute the inverse of a 3x3 matrix
KOKKOS_INLINE_FUNCTION
void ComputeInverseMatrix3( const Real inputMat[][3], Real outputMat[][3] ){

	Real determinant = 0.0;
	ComputeDeterminant3( inputMat, determinant );

	// Transposition of Jacobian is skipped and indeces are inverted instead
	outputMat[0][0] = (inputMat[1][1]*inputMat[2][2] - inputMat[2][1]*inputMat[1][2])/determinant;
	outputMat[1][0] = -(inputMat[1][0]*inputMat[2][2] - inputMat[0][2]*inputMat[2][1])/determinant;
	outputMat[2][0] = (inputMat[1][1]*inputMat[1][2] - inputMat[0][2]*inputMat[1][1])/determinant;
	outputMat[0][1] = -(inputMat[1][0]*inputMat[2][2] - inputMat[1][2]*inputMat[2][0])/determinant;
	outputMat[1][1] = (inputMat[0][0]*inputMat[2][2] - inputMat[0][2]*inputMat[2][0])/determinant;
	outputMat[2][1] = -(inputMat[0][0]*inputMat[1][2] - inputMat[0][2]*inputMat[1][0])/determinant;
	outputMat[0][2] = (inputMat[1][0]*inputMat[2][1] - inputMat[1][1]*inputMat[2][0])/determinant;
	outputMat[1][2] = -(inputMat[0][0]*inputMat[2][1] - inputMat[0][1]*inputMat[2][0])/determinant;
	outputMat[2][2] = (inputMat[0][0]*inputMat[1][1] - inputMat[0][1]*inputMat[1][0])/determinant;

}

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
	const Real &q_over_m = charge_over_mass;

      // First half-step in space
      par_for("part_boris",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
      
        // Contravariant and co-variant 4-velocities
	// To have compatibility with GR the velocity stored should be the covariant one
	Real u_cov[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
	Real u_con[3];
	Real x[3]; //Half-step increment.
	Real g_Lor;
	// Get metric components at new location x1,x2,x3
	Real glower[4][4], gupper[4][4], ADM_upper[3][3]; // Metric 
					       // (remember: sqrt(-1/gupper[0][0]) = alpha)
        Real x1p = pr(IPX,p);
        Real x2p = pr(IPY,p);
        Real x3p = pr(IPZ,p);
	ComputeMetricAndInverse(x1p,x2p,x3p, is_minkowski, spin, glower, gupper); 
	// Compute 3x3 ADM spatial metric from covariant metric 
	GetUpperAdmMetric( gupper, ADM_upper );
	// Raise indeces of u_cov
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		u_con[i1] = 0.0;
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		u_con[i1] += ADM_upper[i1][i2]*u_cov[i2];
		}
	}
	//Use definition of the Lorentz factor in ADM formalism
	//Sometimes called u0 instead
	g_Lor = 0.0;
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_cov[i1]*u_cov[i2];
		}
	}
	// In principle the 1.0 should be replaced by a 0 if
	// the particle is massless, but I don't know of 
	// any massless particle that can interact with an 
	// electromagnetic field (unless one goes into quantum mechanics)
	g_Lor = sqrt(1.0 + g_Lor)/sqrt(-gupper[0][0]);

	x[0] = pr(IPX,p) + dt/(2.0)*(u_con[0]/g_Lor + gupper[0][1]/gupper[0][0]) ;
        if (multi_d) { x[1] = pr(IPY,p) + dt/(2.0)*(u_con[1]/g_Lor + gupper[0][2]/gupper[0][0]) ; }
        if (three_d) { x[2] = pr(IPZ,p) + dt/(2.0)*(u_con[2]/g_Lor + gupper[0][3]/gupper[0][0]) ; }

	Real uE[3]; //Evolution of the velocity due to the electric field (first half).
	Real uB[3]; //Evolution of the velocity due to the magnetic field.

	int m = pi(PGID,p) - gids;
	int ip = (x[0] - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
	int jp = (x[1] - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
	int kp = (x[2] - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;

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
	E[0] += e0_.x1e(m, kp, jp, ip) + (x[1] - x2v)*(e0_.x1e(m, kp, jp+1, ip) - e0_.x1e(m, kp, jp, ip))/Dx;
	E[0] += e0_.x1e(m, kp, jp, ip) + (x[2] - x3v)*(e0_.x1e(m, kp+1, jp, ip) - e0_.x1e(m, kp, jp, ip))/Dx;
	E[0] /= 3.0;
	E[1] = e0_.x2e(m, kp, jp, ip) + (x[0] - x1v)*(e0_.x2e(m, kp, jp, ip+1) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[1] += e0_.x2e(m, kp, jp, ip) + (x[1] - x2v)*(e0_.x2e(m, kp, jp+1, ip) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[1] += e0_.x2e(m, kp, jp, ip) + (x[2] - x3v)*(e0_.x2e(m, kp+1, jp, ip) - e0_.x2e(m, kp, jp, ip))/Dy;
	E[1] /= 3.0;
	E[2] = e0_.x3e(m, kp, jp, ip) + (x[0] - x1v)*(e0_.x3e(m, kp, jp, ip+1) - e0_.x3e(m, kp, jp, ip))/Dz;
	E[2] += e0_.x3e(m, kp, jp, ip) + (x[1] - x2v)*(e0_.x3e(m, kp, jp+1, ip) - e0_.x3e(m, kp, jp, ip))/Dz;
	E[2] += e0_.x3e(m, kp, jp, ip) + (x[2] - x3v)*(e0_.x3e(m, kp+1, jp, ip) - e0_.x3e(m, kp, jp, ip))/Dz;
	E[2] /= 3.0;

	uE[0] = u_con[0] + dt*q_over_m/(2.0)*E[0];
        if (multi_d) { uE[1] = u_con[1] + dt*q_over_m/(2.0)*E[1]; }
        if (three_d) { uE[2] = u_con[2] + dt*q_over_m/(2.0)*E[2]; }

	// Get metric components at new location x1,x2,x3
	ComputeMetricAndInverse(x[0],x[1],x[2], is_minkowski, spin, glower, gupper); 
	GetUpperAdmMetric( gupper, ADM_upper );
	// Lower indeces of u_con
	// This and the following computation of g_Lor
	// can probably be compressed into a single expression
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
        // Interpolate Magnetic Field at new particle location x1, x2, x3
	// Store it in an array for convenience 
	Real B[3] = {0.0, 0.0, 0.0};
	B[0] = b0_.x1f(m, kp, jp, ip) + (x[0] - x1v)*(b0_.x1f(m, kp, jp, ip+1) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[0] += b0_.x1f(m, kp, jp, ip) + (x[1] - x2v)*(b0_.x1f(m, kp, jp+1, ip) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[0] += b0_.x1f(m, kp, jp, ip) + (x[2] - x3v)*(b0_.x1f(m, kp+1, jp, ip) - b0_.x1f(m, kp, jp, ip))/Dx;
	B[0] /= 3.0;
	B[1] = b0_.x2f(m, kp, jp, ip) + (x[0] - x1v)*(b0_.x2f(m, kp, jp, ip+1) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[1] += b0_.x2f(m, kp, jp, ip) + (x[1] - x2v)*(b0_.x2f(m, kp, jp+1, ip) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[1] += b0_.x2f(m, kp, jp, ip) + (x[2] - x3v)*(b0_.x2f(m, kp+1, jp, ip) - b0_.x2f(m, kp, jp, ip))/Dy;
	B[1] /= 3.0;
	B[2] = b0_.x3f(m, kp, jp, ip) + (x[0] - x1v)*(b0_.x3f(m, kp, jp, ip+1) - b0_.x3f(m, kp, jp, ip))/Dz;
	B[2] += b0_.x3f(m, kp, jp, ip) + (x[1] - x2v)*(b0_.x3f(m, kp, jp+1, ip) - b0_.x3f(m, kp, jp, ip))/Dz;
	B[2] += b0_.x3f(m, kp, jp, ip) + (x[2] - x3v)*(b0_.x3f(m, kp+1, jp, ip) - b0_.x3f(m, kp, jp, ip))/Dz;
	B[2] /= 3.0;

	// When operating with magnetic field in normal coordinate the velocity must be combined with the metric beta
	Real uE_beta[3] = {
		uE[0]/g_Lor + gupper[0][1]/gupper[0][0],
		uE[1]/g_Lor + gupper[0][2]/gupper[0][0],
		uE[2]/g_Lor + gupper[0][3]/gupper[0][0]
	};

	// g_Lor = sqrt(1.0 + g_Lor)/sqrt(-gupper[0][0]);
	Real mod_t_sqr = 0.0;
	Real t[3];
	for (int i1 = 0; i1 < 3; ++i1 ){ 
	t[i1] = B[i1]*q_over_m/(2.0*g_Lor)*dt;
	}
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		mod_t_sqr += glower[i1+1][i2+1]*t[i1]*t[i2];
		}
	}

	Real adm_det; 
	ComputeDeterminant3( ADM_upper, adm_det );
	// Save the vector product of u and t 
	// Vector product results in covariant vector
	Real vec_ut_cov[3] = {
	uE_beta[1]*t[2] - uE_beta[2]*t[1],
	uE_beta[2]*t[0] - uE_beta[0]*t[2],
	uE_beta[0]*t[1] - uE_beta[1]*t[0]
	};
	//Raise indeces to contravariant
	Real vec_ut[3] = {0.0};
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		vec_ut[i1] += ADM_upper[i1][i2]*vec_ut_cov[i2];
		}
	}
	// Used a vector product, correct for volume
	for (int i = 0; i < 3; ++i ){ 
		vec_ut[i] = vec_ut[i] * adm_det;
	}
	// Re-use arrays
	vec_ut_cov[0] = (uE_beta[1] + vec_ut[1])*t[2] - (uE_beta[2] + vec_ut[2])*t[1];
	vec_ut_cov[1] = (uE_beta[2] + vec_ut[2])*t[0] - (uE_beta[0] + vec_ut[0])*t[2];
	vec_ut_cov[2] = (uE_beta[0] + vec_ut[0])*t[1] - (uE_beta[1] + vec_ut[1])*t[0];
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		vec_ut[i1] = 0.0;
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		vec_ut[i1] += ADM_upper[i1][i2]*vec_ut_cov[i2];
		}
	}
	for (int i = 0; i < 3; ++i ){ 
		vec_ut[i] = vec_ut[i] * adm_det;
	}
	

	uB[0] = uE[0] + 2.0/(1.0+mod_t_sqr)*( vec_ut[0] );
        if (multi_d) { uB[1] = uE[1] + 2.0/(1.0+mod_t_sqr)*( vec_ut[1] ); }
        if (three_d) { uB[2] = uE[2] + 2.0/(1.0+mod_t_sqr)*( vec_ut[2] ); }
	
	//Second half-step with electric field
	uE[0] = uB[0] + dt*q_over_m/(2.0)*E[0];
        if (multi_d) { uE[1] = uB[1] + dt*q_over_m/(2.0)*E[1]; }
        if (three_d) { uE[2] = uB[2] + dt*q_over_m/(2.0)*E[2]; }

	for (int i1 = 0; i1 < 3; ++i1 ){ 
		u_cov[i1] = 0.0;
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		u_cov[i1] += glower[i1+1][i2+1]*uE[i2];
		}
	}
	// Finally update velocity in local space
	pr(IPVX,p) = u_cov[0];
	pr(IPVY,p) = u_cov[1];
	pr(IPVZ,p) = u_cov[2];

	if (!only_v){
	g_Lor = 0.0; //Intermediate Lorentz gamma factor
	for (int i1 = 0; i1 < 3; ++i1 ){ 
		for (int i2 = 0; i2 < 3; ++i2 ){ 
		g_Lor += ADM_upper[i1][i2]*u_cov[i1]*u_cov[i2];
		}
	}
	g_Lor = sqrt(1.0 + g_Lor)/sqrt(-gupper[0][0]);
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
	if (n_iter > 5){
		step_fac = 1E+3;
	}

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
