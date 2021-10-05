//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements EOS functions in derived class for nonrelativistic adiabatic hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor
    
IdealSRMHD::IdealSRMHD(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{      
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;
}  

//----------------------------------------------------------------------------------------
// \!fn Real Equation49()
// \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
// The root fa(mu)==0 of this function corresponds to the upper bracket for
// solving Equation44

KOKKOS_INLINE_FUNCTION
Real Equation49(Real mu, Real b2, Real rpar, Real r, Real q)
{
  Real const x = 1./(1.+mu*b2);           // (26)

  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar); // (38)

  return mu*sqrt(1.+rbar) - 1.;
}

//----------------------------------------------------------------------------------------
// \!fn Real Equation44()
// \brief Inline function to compute function f(mu) defined in eq. 44 of Kastaun et al.
// The ConsToPRim algorithms finds the root of this function f(mu)=0

KOKKOS_INLINE_FUNCTION
Real Equation44(Real mu, Real b2, Real rpar, Real r, Real q, Real ud, Real pfloor, Real gm1)
{
  Real const x = 1./(1.+mu*b2);           // (26)

  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar); // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
//  rbar = sqrt(rbar);


  Real z2 = (mu*mu*rbar/(abs(1.- SQR(mu)*rbar))); // (32)
  Real w = sqrt(1.+z2);

  Real const wd = ud/w;                  // (34)
  Real eps = w*(qbar - mu*rbar)+  z2/(w+1.);


  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(pfloor/(wd*gm1), eps);                          // 
  Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (43)

  return mu - 1./(h/w + rbar*mu); // (45)
}

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief No-Op version of MHD cons to prim functions.  Never used in MHD.

void IdealSRMHD::ConsToPrim(DvceArray5D<Real> &cons,
         const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc)
{

  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_adi = eos_data.gamma;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;
  Real ee_min = pfloor_/gm1;

  Real mm_sq_ee_sq_max = 1.0 - 1.0e-12;  // max. of squared momentum over energy

    // Parameters
    int const max_iterations = 15;
    Real const tol = 1.0e-12;
    Real const pgas_uniform_min = 1.0e-12;
    Real const a_min = 1.0e-12;
    Real const v_sq_max = 1.0 - 1.0e-12;
    Real const rr_max = 1.0 - 1.0e-12;


  par_for("mhd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m, IDN,k,j,i);
      Real& u_m1 = cons(m, IM1,k,j,i);
      Real& u_m2 = cons(m, IM2,k,j,i);
      Real& u_m3 = cons(m, IM3,k,j,i);
      Real& u_e  = cons(m, IEN,k,j,i);

      Real& w_d  = prim(m, IDN,k,j,i);
      Real& w_vx = prim(m, IVX,k,j,i);
      Real& w_vy = prim(m, IVY,k,j,i);
      Real& w_vz = prim(m, IVZ,k,j,i);
      Real& w_p  = prim(m, IPR,k,j,i);

      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc(m,IBX,k,j,i);
      Real& w_by = bcc(m,IBY,k,j,i);
      Real& w_bz = bcc(m,IBZ,k,j,i);
      w_bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));  
      w_by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      w_bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;

      // apply energy floor
//      Real ee_min = pfloor_/gm1;
//      u_e = (u_e > ee_min) ?  u_e : ee_min;


      // Recast all variables (eq 22-24)
      // Variables q and r defined in anonymous namspace: global this file
      Real q = u_e/u_d;
      Real r = sqrt(SQR(u_m1) + SQR(u_m2) + SQR(u_m3))/u_d;

      Real sqrtd = sqrt(u_d);
      Real bx = w_bx/sqrtd;
      Real by = w_by/sqrtd;
      Real bz = w_bz/sqrtd;
      Real b2 = (bx*bx+by*by+bz*bz);

      Real rpar = (bx*u_m1 +  by*u_m2 +  bz*u_m3)/u_d;


      // Need to find initial bracket. Requires separate solve

      Real zm=0.;
      Real zp=1.; // This is the lowest specific enthalpy admitted by the EOS

      // Evaluate master function (eq 49) at bracket values
      Real fm = Equation49(zm, b2, rpar, r, q);
      Real fp = Equation49(zp, b2, rpar, r, q);
      
      
      // For simplicity on the GPU, find roots using the false position method
      int iterations = max_iterations;
      // If bracket within tolerances, don't bother doing any iterations
      if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
        iterations = -1;
      }
      Real z = 0.5*(zm + zp);

      for (int ii=0; ii < iterations; ++ii) {
	z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
      	Real f = Equation49(z, b2, rpar, r, q);

        // Quit if convergence reached
	// NOTE: both z and f are of order unity
	if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )){
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


      zm= 0.;
      zp= z;

      // Evaluate master function (eq 44) at bracket values
      fm = Equation44(zm, b2, rpar, r, q, u_d, pfloor_, gm1);
      fp = Equation44(zp, b2, rpar, r, q, u_d, pfloor_, gm1);

      // For simplicity on the GPU, find roots using the false position method
      iterations = max_iterations;
      // If bracket within tolerances, don't bother doing any iterations
      if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
        iterations = -1;
      }
      z = 0.5*(zm + zp);

      for (int ii=0; ii < iterations; ++ii) {
	z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = Equation44(z, b2, rpar, r, q, u_d, pfloor_, gm1);

        // Quit if convergence reached
	// NOTE: both z and f are of order unity
	if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )){
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

      Real &mu = z;

      Real const x = 1./(1.+mu*b2);           // (26)

      Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar); // (38)
      Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
    //  rbar = sqrt(rbar);


      Real z2 = (mu*mu*rbar/(abs(1.- SQR(mu)*rbar))); // (32)
      Real w = sqrt(1.+z2);

      Real const wd = u_d/w;                  // (34)
      Real eps = w*(qbar - mu*rbar)+  z2/(w+1.);

    //NOTE: The following generalizes to ANY equation of state
      eps = fmax(pfloor_/(wd*gm1), eps);                          // 
      Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (43)
      w_p = w_d*gm1*eps;

      Real const conv = w/(h*w + b2); // (C26)
      w_vx = conv * ( u_m1/u_d + bx * rpar/(h*w));           // (C26)
      w_vy = conv * ( u_m2/u_d + by * rpar/(h*w));           // (C26)
      w_vz = conv * ( u_m3/u_d + bz * rpar/(h*w));           // (C26)

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }

      }


    // TODO error handling

//     if (false)
//     {
//       Real &gamma =w;
//       Real gamma_adi = gm1+1.;
//       Real rho_eps = w_p / gm1;
//       //FIXME ERM: Only ideal fluid for now
//       Real wgas = w_d + gamma_adi / gm1 *w_p;
//
//       Real b0 = w_bx * w_vx + w_by * w_vy + w_bz * w_vz;
//       Real b1 = (w_bx + b0 * w_vx) / gamma;
//       Real b2 = (w_by + b0 * w_vy) / gamma;
//       Real b3 = (w_bz + b0 * w_vz) / gamma;
//       Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);
//
//       wgas += b_sq;
//       
//       cons(m,IDN,k,j,i) = w_d * gamma;
//       cons(m,IEN,k,j,i) = wgas*gamma*gamma - (w_p + 0.5*b_sq) - w_d * gamma - b0*b0; //rho_eps * gamma_sq + (w_p + cons(IDN,k,j,i)/(gamma+1.))*(v_sq*gamma_sq);
//       cons(m,IM1,k,j,i) = wgas * gamma * w_vx - b0*b1;
//       cons(m,IM2,k,j,i) = wgas * gamma * w_vy - b0*b2;
//       cons(m,IM3,k,j,i) = wgas * gamma * w_vz - b0*b3;
//     }
//
//     // convert scalars (if any)
//     for (int n=nmhd; n<(nmhd+nscal); ++n) {
//       prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
//     }
//   }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for SR magnetohydrodynamics. Operates
//! only over active cells.
//! Recall in SR hydrodynamics the conserved variables are: (D, E-D, m^i, bcc),
//!                        and the primitive variables are: (\rho, P_gas, u^i).

void IdealSRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc, 
   			    DvceArray5D<Real> &cons)
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto coord = pmy_pack->coord.coord_data;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_adi = eos_data.gamma;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0); 

  // FIXME Remove when finishing the function
  std::cout << "PrimToCons not implemented for SRMHD" << std::endl;
  std::exit(EXIT_FAILURE);

  par_for("hyd_prim2cons", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m, IDN,k,j,i);
      Real& u_e  = cons(m, IEN,k,j,i);
      Real& u_m1 = cons(m, IM1,k,j,i);
      Real& u_m2 = cons(m, IM2,k,j,i);
      Real& u_m3 = cons(m, IM3,k,j,i);

      const Real& w_d  = prim(m, IDN,k,j,i);
      const Real& w_p  = prim(m, IPR,k,j,i);
      const Real& w_ux = prim(m, IVX,k,j,i);
      const Real& w_uy = prim(m, IVY,k,j,i);
      const Real& w_uz = prim(m, IVZ,k,j,i);

      // Calculate Lorentz factor
      Real u0 = sqrt(1.0 + SQR(w_ux) + SQR(w_uy) + SQR(w_uz));
      Real wgas_u0 = (w_d + gamma_prime * w_p) * u0;

      // TODO NEED TO ADD MHD

      // Set conserved quantities
      u_d  = w_d * u0;
      u_e  = wgas_u0 * u0 - w_p - u_d;  // In SR, evolve E - D
      u_m1 = wgas_u0 * w_ux;            // In SR, w_ux/y/z are 4-velocity
      u_m2 = wgas_u0 * w_uy;
      u_m3 = wgas_u0 * w_uz;

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
      }

    }
  );

  return;
}

