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

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor
    
IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin)
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

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons,
         const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc)
{

  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_adi = eos_data.gamma;

  auto coord = pmy_pack->coord.coord_data;
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

      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

      bool floor_hit = false;

      // Extract components of metric
      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                              coord.bh_spin, g_, gi_);

      // Only execute cons2prim if outside excised region
      if (rad > coord.bh_rmin) {

        // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
        // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
        // We are also evolving (E-D) as conserved variable, so must convert to E 
        Real ue_tmp = gi_[I00]*(u_e+u_d) + gi_[I01]*u_m1 + gi_[I02]*u_m2 + gi_[I03]*u_m3;

        // This is only true if sqrt{-g}=1!
        ue_tmp *= (-1./gi_[I00]);  // Multiply by alpha^2

        // Need to multiply the conserved density by alpha, so that it
        // contains a lorentz factor
        Real alpha = sqrt(-1.0/gi_[I00]);
        Real ud_tmp = u_d*alpha;

        // Subtract density for consistency with the rest of the algorithm
        ue_tmp -= ud_tmp;

        // Need to treat the conserved momenta. Also they lack an alpha
        // This is only true if sqrt{-g}=1!
        Real um1_tmp = u_m1*alpha;
        Real um2_tmp = u_m2*alpha;
        Real um3_tmp = u_m3*alpha;

        // apply density floor, without changing momentum or energy
        if (ud_tmp < dfloor_) {
          ud_tmp = dfloor_;
          floor_hit = true;
        }

        // apply energy floor
        if (ue_tmp < pfloor_/gm1) {
          ue_tmp = pfloor_/gm1;
          floor_hit = true;
        }

        // Recast all variables 
        // Need to raise indices on u_m1, which transforms using the spatial 3-metric.
        // This is slightly more involved
        //
        // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
        //       g^0i = beta^i/alpha^2
        //       g^00 = -1/ alpha^2
        // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
        Real m1u = ((gi_[I11] - gi_[I01]*gi_[I01]/gi_[I00])*um1_tmp +
                    (gi_[I12] - gi_[I01]*gi_[I02]/gi_[I00])*um2_tmp +
                    (gi_[I13] - gi_[I01]*gi_[I03]/gi_[I00])*um3_tmp);  // (C26)

        Real m2u = ((gi_[I12] - gi_[I01]*gi_[I02]/gi_[I00])*um1_tmp +
                    (gi_[I22] - gi_[I02]*gi_[I02]/gi_[I00])*um2_tmp +
                    (gi_[I23] - gi_[I02]*gi_[I03]/gi_[I00])*um3_tmp);  // (C26)

        Real m3u = ((gi_[I13] - gi_[I01]*gi_[I03]/gi_[I00])*um1_tmp +
                    (gi_[I23] - gi_[I02]*gi_[I03]/gi_[I00])*um2_tmp +
                    (gi_[I33] - gi_[I03]*gi_[I03]/gi_[I00])*um3_tmp);  // (C26)


	// Recast all variables (eq 22-24)
	// Variables q and r defined in anonymous namspace: global this file
        Real q = ue_tmp/ud_tmp;
        Real r = sqrt(um1_tmp*m1u + um2_tmp*m2u + um3_tmp*m3u)/ud_tmp;

	Real sqrtd = sqrt(u_d);
	Real bx = w_bx/sqrtd;
	Real by = w_by/sqrtd;
	Real bz = w_bz/sqrtd;

        // Need to treat the magnetic fields. Also they lack an alpha
        // This is only true if sqrt{-g}=1!
        bx *= alpha;
        by *= alpha;
        bz *= alpha;


	Real b2 =     g_[I11] * bx * bx + g_[I22] * by * by + g_[I33] * bz * bz
	        +2.*( g_[I12] * bx * by + g_[I13] * bx * bz + g_[I23] * by * bz);

	Real rpar = (bx*um1_tmp +  by*um2_tmp +  bz*um3_tmp)/u_d;


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
	w_vx = conv * ( m1u/u_d + bx * rpar/(h*w));           // (C26)
	w_vy = conv * ( m2u/u_d + by * rpar/(h*w));           // (C26)
	w_vz = conv * ( m3u/u_d + bz * rpar/(h*w));           // (C26)

	// convert scalars (if any)
	for (int n=nmhd; n<(nmhd+nscal); ++n) {
	  prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
	}
    }

      // FIXME ADD GRMHD PrimToCons

//     // reset conserved variables inside excised regions or if floor is hit
//     if (rad <= coord.bh_rmin || floor_hit) {
//       Real ud, ue, um1, um2, um3;
//       eos.PrimToConsSingleGR(g_, gi_, w_d, w_p, w_ux, w_uy, w_uz,
//                              ud, ue, um1, um2, um3);
//       cons(m,IDN,k,j,i) = ud;
//       cons(m,IEN,k,j,i) = ue;
//       cons(m,IM1,k,j,i) = um1;
//       cons(m,IM2,k,j,i) = um2;
//       cons(m,IM3,k,j,i) = um3;
//       // convert scalars (if any)
//       for (int n=nhyd; n<(nhyd+nscal); ++n) {
//         cons(m,n,k,j,i) = prim(m,n,k,j,i)*cons(m,IDN,k,j,i);
//       }
     }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates only over active cells.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc, 
   			    DvceArray5D<Real> &cons)
{

  // FIXME Remove when finishing the function
  std::cout << "PrimToCons not implemented for GRMHD" << std::endl;
  std::exit(EXIT_FAILURE);

  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto coord = pmy_pack->coord.coord_data;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);

  par_for("hyd_prim2cons", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      // Extract components of metric
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                              coord.bh_spin, g_, gi_);

      const Real
        &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
        &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
        &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
        &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];

      const Real& w_d  = prim(m,IDN,k,j,i);
      const Real& w_p  = prim(m,IPR,k,j,i);
      const Real& w_ux = prim(m,IVX,k,j,i);
      const Real& w_uy = prim(m,IVY,k,j,i);
      const Real& w_uz = prim(m,IVZ,k,j,i);

      // Calculate 4-velocity
      Real alpha = sqrt(-1.0/gi_[I00]);
      Real tmp = g_[I11]*w_ux*w_ux + 2.0*g_[I12]*w_ux*w_uy + 2.0*g_[I13]*w_ux*w_uz
               + g_[I22]*w_uy*w_uy + 2.0*g_[I23]*w_uy*w_uz
               + g_[I33]*w_uz*w_uz;
      Real gamma = sqrt(1.0 + tmp);
      Real u0 = gamma/alpha;
      Real u1 = w_ux - alpha * gamma * gi_[I01];
      Real u2 = w_uy - alpha * gamma * gi_[I02];
      Real u3 = w_uz - alpha * gamma * gi_[I03];
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
      
      // TODO NEED TO ADD MHD

      Real wgas_u0 = (w_d + gamma_prime * w_p) * u0;
      u_d  = w_d * u0;
      u_e  = wgas_u0 * u_0 + w_p - u_d;  // Evolve E-D, as in SR
      u_m1 = wgas_u0 * u_1;
      u_m2 = wgas_u0 * u_2;
      u_m3 = wgas_u0 * u_3;

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
      }
    }
  );

  return;
}



