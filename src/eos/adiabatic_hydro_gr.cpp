//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adiabatic_hydro_gr.cpp
//  \brief implements EOS functions in derived class for general relativistic ad. hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_positions.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor
    
AdiabaticHydroGR::AdiabaticHydroGR(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{      
  eos_data.is_adiabatic = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;
}  

//----------------------------------------------------------------------------------------
// \!fn Real EquationC22()
// \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
// The ConsToPrim algorithms finds the root of this function f(z)=0

KOKKOS_INLINE_FUNCTION
Real EquationC22(Real z, Real &u_d, Real q, Real r, Real gm1, Real pfloor)
{
  Real const w = sqrt(1.0 + z*z);         // (C15)
  Real const wd = u_d/w;                  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)

  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(pfloor/(wd*gm1), eps);                          // (C18)
  Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (C1) & (C21)

  return (z - r/h); // (C22)
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables.
// Operates over entire MeshBlock, including ghost cells.  

void AdiabaticHydroGR::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim)
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto size = pmy_pack->pmb->mbsize;
  Real gm1 = eos_data.gamma - 1.0; 
  Real pfloor_ = eos_data.pressure_floor;
  Real &dfloor_ = eos_data.density_floor;

  // Parameters
  int const max_iterations = 25;
  Real const tol = 1.0e-12;
  Real const v_sq_max = 1.0 - tol;

  Real const spin= pmy_pack->coord.coord_data.bh_spin;

  par_for("hyd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real u_d  = cons(m, IDN,k,j,i);
      Real u_e  = cons(m, IEN,k,j,i);
      Real u_m1 = cons(m, IM1,k,j,i);
      Real u_m2 = cons(m, IM2,k,j,i);
      Real u_m3 = cons(m, IM3,k,j,i);

      Real& w_d  = prim(m, IDN,k,j,i);
      Real& w_p  = prim(m, IPR,k,j,i);
      Real& w_vx = prim(m, IVX,k,j,i);
      Real& w_vy = prim(m, IVY,k,j,i);
      Real& w_vz = prim(m, IVZ,k,j,i);

      Real x1 = CellCenterX(i, n1, size.x1min.d_view(m), size.x1max.d_view(m));
      Real x2 = CellCenterX(j, n2, size.x2min.d_view(m), size.x2max.d_view(m));
      Real x3 = CellCenterX(k, n3, size.x3min.d_view(m), size.x3max.d_view(m));

      // Extract components of metric
      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1, x2, x3, spin, g_, gi_);

      const Real
	&g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
	&g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
	&g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
	&g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];
      const Real
	&g00 = gi_[I00], &g01 = gi_[I01], &g02 = gi_[I02], &g03 = gi_[I03],
	&g10 = gi_[I01], &g11 = gi_[I11], &g12 = gi_[I12], &g13 = gi_[I13],
	&g20 = gi_[I02], &g21 = gi_[I12], &g22 = gi_[I22], &g23 = gi_[I23],
	&g30 = gi_[I03], &g31 = gi_[I13], &g32 = gi_[I23], &g33 = gi_[I33];
      Real alpha = std::sqrt(-1.0/g00);


      // Need to convert the conservatives
      

      // We are evolving T^t_t, but the SR C2P algorithm is only consistent
      // with alpha^2 T^{tt}
      // compute T^{tt} = g^0\mu T^t_\mu
      
      u_e = g00 * u_e + g01 * u_m1 + g02 * u_m2 + g03 * u_m3;

      // This is only true if sqrt{-g}=1!
      u_e *= (-1./g00); // Multiply by alpha^2

      // Need to multiply the conserved density by alpha, so that it
      // contains a lorentz factor
      
      u_d *= alpha;

      // Subtract density for consistency with the rest of the algorithm
      u_e -= u_d;            


      // Need to treat the conserved momenta. Also they lack an alpha
      // This is only true if sqrt{-g}=1!
      
      u_m1 *= alpha;
      u_m2 *= alpha;
      u_m3 *= alpha;


      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;

      // apply energy floor
//      Real ee_min = pfloor_/gm1;
//      u_e = (u_e > ee_min) ?  u_e : ee_min;


      // Recast all variables (eq C2)
      // Variables q and r defined in anonymous namspace: global this file
      Real q = u_e/u_d;
//      Real r = sqrt(SQR(u_m1) + SQR(u_m2) + SQR(u_m3))/u_d;

      Real r   = g_11*SQR(u_m1) + 2.0*g_12*u_m1*u_m2 + 2.0*g_13*u_m1*u_m3
               + g_22*SQR(u_m1) + 2.0*g_23*u_m1*u_m3
               + g_33*SQR(u_m1);

      r = sqrt(r)/u_d;

      Real kk = r/(1.+q);

      // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
      // the value of p will guarantee "some" result of the inversion
      kk = fmin(2.* sqrt(v_sq_max)/(1.0 + v_sq_max), kk);

      // Compute bracket (C23)
      auto zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
      auto zp = kk/sqrt(1.0 - kk*kk);

      // Evaluate master function (eq C22) at bracket values
      Real fm = EquationC22(zm, u_d, q, r, gm1, pfloor_);
      Real fp = EquationC22(zp, u_d, q, r, gm1, pfloor_);

      // For simplicity on the GPU, find roots using the false position method
      int iterations = max_iterations;
      // If bracket within tolerances, don't bother doing any iterations
      if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
        iterations = -1;
      }
      Real z = 0.5*(zm + zp);

      for (int ii=0; ii < iterations; ++ii) {
	z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = EquationC22(z, u_d, q, r, gm1, pfloor_);

        // Quit if convergence reached
	// NOTE: both z and f are of order unity
	if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )){
/**
std::cout << "|zm-zp|=" <<fabs(zm-zp)<<" |f|="<< fabs(f) << "for i=" <<  ii << std::endl;
**/
	    break;
	}

        // assign zm-->zp if root bracketed by [z,zp]
	if (f * fp < 0.0) {
	   zm = zp;
	   fm = fp;
	   zp = z;
	   fp = f;

        // assign zp-->z if root bracketed by [zm,z]
	} else {
	   fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
	   zp = z;
	   fp = f;
	}
      }

      // iterations ended, compute primitives from resulting value of z
      Real const w = sqrt(1.0 + z*z); // (C15)
      w_d = u_d/w;                    // (C15)

      //NOTE: The following generalizes to ANY equation of state
      Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)
      eps = fmax(pfloor_/w_d/gm1, eps);                 // (C18)
      Real h = (1. + eps) * (1.0 + (gm1*eps)/(1.+eps)); // (C1) & (C21)
      w_p = w_d*gm1*eps;

      Real const conv = 1.0/(h*u_d); // (C26)


      // Need to raise indices on u_m1, which transforms using the spatial
      // 3-metric.
      // This is slightly more involved
      //
      // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
      // 		   g^0i = beta^i/alpha^2
      // 		   g^00 = -1/ alpha^2
      // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
      
      w_vx = conv *((g11 - g01*g01/g00) * u_m1 + 
		    (g12 - g01*g02/g00) * u_m2 + 
		    (g13 - g01*g03/g00) * u_m3);           // (C26)

      w_vy = conv *((g12 - g01*g02/g00) * u_m1 + 
		    (g22 - g02*g02/g00) * u_m2 + 
		    (g23 - g02*g03/g00) * u_m3);           // (C26)

      w_vz = conv *((g13 - g01*g03/g00) * u_m1 + 
		    (g23 - g02*g03/g00) * u_m2 + 
		    (g33 - g03*g03/g00) * u_m3);           // (C26)


      // These are the covariant velocities: W v_i
      // Need to raise them using the three metric.
      


      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }

      // TODO error handling
      // The expressions below are not correct for GR
//
//      if (false)
//      {
//	Real gamma_adi = gm1+1.;
//	Real rho_eps = w_p / gm1;
//	//FIXME ERM: Only ideal fluid for now
//        Real wgas = w_d + gamma_adi / gm1 *w_p;
//	
//	auto gamma = sqrt(1. +z*z);
//        cons(m,IDN,k,j,i) = w_d * gamma;
//        cons(m,IEN,k,j,i) = wgas*gamma*gamma - w_p - w_d * gamma; 
//        cons(m,IM1,k,j,i) = wgas * gamma * w_vx;
//        cons(m,IM2,k,j,i) = wgas * gamma * w_vy;
//        cons(m,IM3,k,j,i) = wgas * gamma * w_vz;
//      }

    }
  );

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCons()
// \brief Converts primitive into conserved variables.  Operates only over active cells.

void AdiabaticHydroGR::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons)
{
  auto ncells = pmy_pack->mb_cells;
  int &nx1 = ncells.nx1;
  int &nx2 = ncells.nx2;
  int &nx3 = ncells.nx3;
  int is = pmy_pack->mb_cells.is;
  int js = pmy_pack->mb_cells.js;
  int ks = pmy_pack->mb_cells.ks;
  auto size = pmy_pack->pmb->mbsize;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);
  Real &spin = pmy_pack->coord.coord_data.bh_spin;

  par_for("hyd_prim2cons", DevExeSpace(), 0, (nmb-1), 0, (nx3-1), 0, (nx2-1), 0, (nx1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      // Extract components of metric
      Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
      Real x2v = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
      Real x3v = CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m));
      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, spin, g_, gi_);
      const Real
        &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
        &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
        &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
        &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];

      const Real& w_d  = prim(m,IDN,k,j,i);
      const Real& w_p  = prim(m,IPR,k,j,i);
      const Real& w_vx = prim(m,IVX,k,j,i);
      const Real& w_vy = prim(m,IVY,k,j,i);
      const Real& w_vz = prim(m,IVZ,k,j,i);

      // Calculate 4-velocity
      Real alpha = sqrt(-1.0/gi_[I00]);
      Real tmp = g_[I11]*w_vx*w_vx + 2.0*g_[I12]*w_vx*w_vy + 2.0*g_[I13]*w_vx*w_vz
               + g_[I22]*w_vy*w_vy + 2.0*g_[I23]*w_vy*w_vz
               + g_[I33]*w_vz*w_vz;
      Real gamma = sqrt(1.0 + tmp);
      Real u0 = gamma/alpha;
      Real u1 = w_vx - alpha * gamma * gi_[I01];
      Real u2 = w_vy - alpha * gamma * gi_[I02];
      Real u3 = w_vz - alpha * gamma * gi_[I03];
      Real u_0 = g_00*u0 + g_01*u1 + g_02*u2 + g_03*u3;
      Real u_1 = g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3;
      Real u_2 = g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3;
      Real u_3 = g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3;

      // Set conserved quantities
      Real& u_d  = cons(m,IDN,k,j,i);
      Real& u_e  = cons(m,IEN,k,j,i);
      Real& u_m1 = cons(m,IM1,k,j,i);
      Real& u_m2 = cons(m,IM2,k,j,i);
      Real& u_m3 = cons(m,IM3,k,j,i);

      Real wgas_u0 = (w_d + gamma_prime * w_p) * u0;
      u_d  = w_d * u0;
      u_e  = wgas_u0 * u_0 + w_p;
      u_m1 = wgas_u0 * u_1;
      u_m2 = wgas_u0 * u_2;
      u_m3 = wgas_u0 * u_3;
    }
  );

  return;
}
