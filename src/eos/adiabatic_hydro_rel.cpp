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
#include "hydro/hydro.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor
    
AdiabaticHydroRel::AdiabaticHydroRel(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{      
  eos_data.is_adiabatic = true;
  eos_data.gamma = pin->GetReal("eos","gamma");
  eos_data.iso_cs = 0.0;
}  

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief No-Op version of MHD cons to prim functions.  Never used in Hydro.

void AdiabaticHydroRel::ConsToPrim(const DvceArray5D<Real> &cons,
         const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc)
{
}

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic adiabatic hydro

void AdiabaticHydroRel::ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim)
{

  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_adi = eos_data.gamma;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;
  Real ee_min = pfloor_/gm1;

    // Parameters
    int const max_iterations = 25;
    Real const tol = 1.0e-12;
    Real const v_sq_max = 1.0 - 1.0e-12;


    // Primitive inversion following Wolfgang Kastaun's algorithm
    // This references https://arxiv.org/pdf/1306.4953.pdf


  par_for("hyd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
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

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
      w_d = u_d;

      // apply energy floor
      u_e = (u_e > ee_min) ?  u_e : ee_min;


      // Recasting all variables

      auto q = u_e/u_d; // (C2)

      auto r = sqrt(SQR(cons(m, IM1,k,j,i))    // (C2)
	          + SQR(cons(m, IM2,k,j,i)) 
		  + SQR(cons(m, IM3,k,j,i)))/u_d;

      auto kk = r/(1.+q);  // (C2)

      // Enforce lower velocity bound
      // Obeying this bound combined with a floor on 
      // p will guarantuee "some" result of the inversion

      kk = fmin(2.* sqrt(v_sq_max)/(1.+v_sq_max), kk); // (C13)

      // Compute bracket
      auto zm = 0.5*kk/sqrt(1. - 0.25*kk*kk); // (C23)
      auto zp = k/sqrt(1-kk*kk);             // (C23)

      // Evaluate master function
      Real fm,fp;      
      {
	auto &z = zm;
	auto &f = fm;

	auto const W = sqrt(1. + z*z); // (C15)

	w_d = u_d/W; // (C15)

	auto eps = W*q - z*r + z*z / (1.+W); // (C16)

	//NOTE: The following generalizes to ANY equation of state
	eps = fmax(pfloor_/w_d/gm1, eps); // (C18)
	w_p = w_d*gm1*eps;
	auto const h = (1. + eps) * ( 1. +  w_p/(w_d*(1.+eps))); // (C1) & (C21)

	f = z - r/h; // (C22)
      }

      {
	auto &z = zp;
	auto &f = fp;

	auto const W = sqrt(1. + z*z); // (C15)

	w_d = u_d/W; // (C15)

	auto eps = W*q - z*r + z*z / (1.+W); // (C16)

	//NOTE: The following generalizes to ANY equation of state
	eps = fmax(pfloor_/w_d/gm1, eps); // (C18)
	w_p = w_d*gm1*eps;
	auto const h = (1. + eps) * ( 1. +  w_p/(w_d*(1.+eps))); // (C1) & (C21)

	f = z - r/h; // (C22)
      }

      //For simplicity on the GPU, use the false position method


      Real z,h;
      for(int ii=0; ii< max_iterations; ++ii){

	z =  (zm*fp - zp*fm)/(fp-fm);

	auto const W = sqrt(1. + z*z); // (C15)

	w_d = u_d/W; // (C15)

	auto eps = W*q - z*r + z*z / (1.+W); // (C16)

	//NOTE: The following generalizes to ANY equation of state
	eps = fmax(pfloor_/w_d/gm1, eps); // (C18)
	w_p = w_d*gm1*eps;
	h = (1. + eps) * ( 1. +  w_p/(w_d*(1.+eps))); // (C1) & (C21)

	auto f = z - r/h; // (C22)

	// NOTE: both z and f are of order unity
	if((fabs(zm-zp) < tol ) || (fabs(f) < tol )){
	    break;
	}

	if(f * fp < 0.){
	   zm = zp;
	   fm = fp;
	   zp = z;
	   fp = f;
	}
	else{
	   fm = 0.5*fm;
	   zp = z;
	   fp = f;
	}

      }

    auto const conv = 1./(h*u_d); // (C26)
    w_vx = conv * u_m1;           // (C26)
    w_vy = conv * u_m2;           // (C26)
    w_vz = conv * u_m3;           // (C26)


    // TODO error handling

      if (false)
      {
	Real gamma_adi = gm1+1.;
	Real rho_eps = w_p / gm1;
	//FIXME ERM: Only ideal fluid for now
        Real wgas = w_d + gamma_adi / gm1 *w_p;
	
	auto gamma = sqrt(1. +z*z);
        cons(m,IDN,k,j,i) = w_d * gamma;
        cons(m,IEN,k,j,i) = wgas*gamma*gamma - w_p - w_d * gamma; //rho_eps * gamma_sq + (w_p + cons(IDN,k,j,i)/(gamma+1.))*(v_sq*gamma_sq);
        cons(m,IM1,k,j,i) = wgas * gamma * w_vx;
        cons(m,IM2,k,j,i) = wgas * gamma * w_vy;
        cons(m,IM3,k,j,i) = wgas * gamma * w_vz;
      }

      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }
    }
  );

  return;
}
