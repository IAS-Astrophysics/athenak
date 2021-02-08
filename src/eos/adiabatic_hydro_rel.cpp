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

  Real mm_sq_ee_sq_max = 1.0 - 1.0e-12;  // max. of squared momentum over energy

    // Parameters
    int const max_iterations = 15;
    Real const tol = 1.0e-12;
    Real const pgas_uniform_min = 1.0e-12;
    Real const a_min = 1.0e-12;
    Real const v_sq_max = 1.0 - 1.0e-12;
    Real const rr_max = 1.0 - 1.0e-12;


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
///      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
//      w_d = u_d;

      // apply energy floor
//      u_e = (u_e > ee_min) ?  u_e : ee_min;
//      w_p = pfloor_;

      Real ee = u_d + u_e;

      Real m2 = SQR(cons(m, IM1,k,j,i)) + SQR(cons(m, IM2,k,j,i)) + SQR(cons(m, IM3,k,j,i));

      Real m2_max = mm_sq_ee_sq_max * SQR(ee);
      if( m2 > m2_max){
	Real factor = std::sqrt(m2_max/m2);
	u_m1*= factor;
	u_m2*= factor;
	u_m3*= factor;
      }

    bool failed= false;



    // Calculate functions of conserved quantities
    Real pgas_min = -ee;
    pgas_min = fmax(pfloor_, pgas_uniform_min);

    // Iterate until convergence
    Real pgas[3];
    pgas[0] =  pgas_min; // Do we have a previous step
    int n;
    for (n = 0; n < max_iterations; ++n) {
      // Step 1: Calculate cubic coefficients
      Real a;
      if (n%3 != 2) {
	a = ee + pgas[n%3];      // (NH 5.7)
	a = fmax(a, a_min);
      }

      // Step 2: Calculate correct root of cubic equation
      Real v_sq;
      if (n%3 != 2) {
	v_sq = m2 / SQR(a);                                     // (NH 5.2)
	v_sq = fmin(fmax(v_sq, 0.0), v_sq_max);
	Real gamma_sq = 1.0/(1.0-v_sq);                            // (NH 3.1)
	Real gamma = sqrt(gamma_sq);                          // (NH 3.1)
	Real wgas = a/gamma_sq;                                    // (NH 5.1)
	Real rho = u_d/gamma;                                       // (NH 4.5)
	pgas[(n+1)%3] = gm1/(gm1+1.) * (wgas - rho);  // (NH 4.1)
	pgas[(n+1)%3] = fmax(pgas[(n+1)%3], pgas_min);
      }

      // Step 3: Check for convergence
      if (n%3 != 2) {
	if (pgas[(n+1)%3] > pgas_min && fabs(pgas[(n+1)%3]-pgas[n%3]) < tol) {
	  break;
	}
      }

      // Step 4: Calculate Aitken accelerant and check for convergence
      if (n%3 == 2) {
	Real rr = (pgas[2] - pgas[1]) / (pgas[1] - pgas[0]);  // (NH 7.1)
	if ((rr!=rr) || fabs(rr) > rr_max) {
	  continue;
	}
	pgas[0] = pgas[1] + (pgas[2] - pgas[1]) / (1.0 - rr);  // (NH 7.2)
	pgas[0] = fmax(pgas[0], pgas_min);
	if (pgas[0] > pgas_min && fabs(pgas[0]-pgas[2]) < tol) {
	  break;
	}
      }
    }

    // Step 5: Set primitives
    if (n == max_iterations) {
      failed = true;
    }
    w_p = pgas[(n+1)%3];
//    if (!std::isfinite(w_p)) {
//      failed=true;
//    }
    Real a = ee + w_p;                   // (NH 5.7)
    a = fmax(a, a_min);
    Real v_sq = m2 / SQR(a);                      // (NH 5.2)
    v_sq = fmin(fmax(v_sq, 0.0), v_sq_max);
    Real gamma_sq = 1.0/(1.0-v_sq);                  // (NH 3.1)
    Real gamma = std::sqrt(gamma_sq);                // (NH 3.1)
    w_d = u_d/gamma;                      // (NH 4.5)
//    if (!std::isfinite(w_d)) {
//      failed=true;
//    }
    w_vx = gamma * u_m1 / a;           // (NH 4.6)
    w_vy = gamma * u_m2 / a;           // (NH 4.6)
    w_vz = gamma * u_m3 / a;           // (NH 4.6)

//    if (!std::isfinite(w_vx) || !std::isfinite(w_vy) || !std::isfinite(w_vz)) {
//      failed = true;
//    }

      // apply pressure floor, correct total energy
//      u_e = (w_p > pfloor_) ?  u_e : ((pfloor_/gm1) + e_k);
//      w_p = (w_p > pfloor_) ?  w_p : pfloor_;


    // TODO error handling

      if (false)
      {
	Real gamma_adi = gm1+1.;
	Real rho_eps = w_p / gm1;
	//FIXME ERM: Only ideal fluid for now
        Real wgas = w_d + gamma_adi / gm1 *w_p;
	
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
