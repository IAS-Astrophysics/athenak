//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_srmhd.cpp
//! \brief derived class that implements ideal gas EOS in special relativistic mhd

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealSRMHD::IdealSRMHD(MeshBlockPack *pp,
                       ParameterInput *pin) : EquationOfState(pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;

  // Read flags specifying which variable to use in primitives
  // if nothing set in input file, use e as default
  if (!(pin->DoesParameterExist("mhd","use_e")) &&
      !(pin->DoesParameterExist("mhd","use_t")) ) {
    eos_data.use_e = true;
    eos_data.use_t = false;
  } else {
    eos_data.use_e = pin->GetOrAddBoolean("mhd","use_e",false);
    eos_data.use_t = pin->GetOrAddBoolean("mhd","use_t",false);
  }
  if (!(eos_data.use_e) && !(eos_data.use_t)) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Both use_e and use_t set to false" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eos_data.use_e && eos_data.use_t) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Both use_e and use_t set to true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation49()
//! \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
//! The root fa(mu)==0 of this function corresponds to the upper bracket for
//! solving Equation44

KOKKOS_INLINE_FUNCTION
Real Equation49(const Real mu, const Real b2,
                const Real rpar, const Real r, const Real q) {
  Real const x = 1./(1.+mu*b2);                  // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar); // (38)
  return mu*sqrt(1.+rbar) - 1.;
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation44()
//! \brief Inline function to compute function f(mu) defined in eq. 44 of Kastaun et al.
//! The ConsToPRim algorithms finds the root of this function f(mu)=0

KOKKOS_INLINE_FUNCTION
Real Equation44(const Real mu, const Real b2, const Real rpar, const Real r, const Real q,
                const Real ud, const Real pfloor, const Real gm1) {
  Real const x = 1./(1.+mu*b2);                  // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar); // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)

  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar))); // (32)
  Real w = sqrt(1.+z2);

  Real const wd = ud/w;                           // (34)
  Real eps = w*(qbar - mu*rbar)+  z2/(w+1.);

  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(pfloor/(wd*gm1), eps);                          //
  Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (43)

  return mu - 1./(h/w + rbar*mu); // (45)
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables for an ideal gas in SR mhd.
//! Implementation follows Wolfgang Kastaun's algorithm described in Appendix C of
//! Galeazzi et al., PhysRevD, 88, 064009 (2013).  Roots of "master function" (eq. C22)
//! found by false position method.
//!
//! In SR mhd, the conserved variables are: (D, E - D, m^i), where
//!    D = \gamma \rho is the density in the lab frame,
//!    \gamma = (1 + u^2)^{1/2} = (1 - v^2)^{-1/2} is the Lorentz factor,
//!    u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is 3-vel),
//!    \rho is the comoving/fluid frame mass density,
//!    E = \gamma^2 w - P_g is the total energy,
//!    w = \rho + [\Gamma / (\Gamma - 1)] P_g is the total enthalpy,
//!    \Gamma is the adiabatic index, P_g is the gas pressure
//!    m^i = \gamma w u^i are components of the momentum in the lab frame.
//! Note we evolve (E-D). This improves accuracy/stability in high-density regions.
//!
//! In SR mhd, the primitive variables are: (\rho, P_gas, u^i).
//! Note components of the 4-velocity (not 3-velocity) are stored in the primitive
//! variables because tests show it is better to reconstruct the 4-vel.
//!
//! This function operates over entire MeshBlock, including ghost cells.

void IdealSRMHD::ConsToPrim(DvceArray5D<Real> &cons,
         const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;
  Real ee_min = pfloor_/gm1;
  bool &use_e = eos_data.use_e;

  Real mm_sq_ee_sq_max = 1.0 - 1.0e-12;  // max. of squared momentum over energy

  // Parameters
  int const max_iterations = 15;
  Real const tol = 1.0e-12;
  Real const pgas_uniform_min = 1.0e-12;
  Real const a_min = 1.0e-12;
  Real const v_sq_max = 1.0 - 1.0e-12;
  Real const rr_max = 1.0 - 1.0e-12;

  par_for("srmhd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m, IDN,k,j,i);
    Real& u_m1 = cons(m, IM1,k,j,i);
    Real& u_m2 = cons(m, IM2,k,j,i);
    Real& u_m3 = cons(m, IM3,k,j,i);
    Real& u_e  = cons(m, IEN,k,j,i);

    Real& w_d  = prim(m, IDN,k,j,i);
    Real& w_ux = prim(m, IVX,k,j,i);
    Real& w_uy = prim(m, IVY,k,j,i);
    Real& w_uz = prim(m, IVZ,k,j,i);

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
    // Real ee_min = pfloor_/gm1;
    // u_e = (u_e > ee_min) ?  u_e : ee_min;

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
      // NOTE(@ermost): both z and f are of order unity
      if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )) {
        break;
      }
      // assign zm-->zp if root bracketed by [z,zp]
      if (f * fp < 0.0) {
        zm = zp;
        fm = fp;
        zp = z;
        fp = f;
      } else {  // assign zp-->z if root bracketed by [zm,z]
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
      if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )) {
        break;
      }
      // assign zm-->zp if root bracketed by [z,zp]
      if (f * fp < 0.0) {
        zm = zp;
        fm = fp;
        zp = z;
        fp = f;
      } else {  // assign zp-->z if root bracketed by [zm,z]
        fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
        zp = z;
        fp = f;
      }
    }

    Real &mu = z;
    Real const x = 1./(1.+mu*b2);                              // (26)
    Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);             // (38)
    Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
    // rbar = sqrt(rbar);

    Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));           // (32)
    Real w = sqrt(1.+z2);

    w_d = u_d/w;                  // (34)
    Real eps = w*(qbar - mu*rbar)+  z2/(w+1.);

    //NOTE: The following generalizes to ANY equation of state
    eps = fmax(pfloor_/(w_d*gm1), eps);
    Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));  // (43)
    if (use_e) {
      Real& w_e  = prim(m,IEN,k,j,i);
      w_e = w_d*eps;
    } else {
      Real& w_t  = prim(m,ITM,k,j,i);
      w_t = gm1*eps;  // TODO(@user):  is this the correct expression?
    }

    Real const conv = w/(h*w + b2); // (C26)
    w_ux = conv * ( u_m1/u_d + bx * rpar/(h*w));  // (C26)
    w_uy = conv * ( u_m2/u_d + by * rpar/(h*w));  // (C26)
    w_uz = conv * ( u_m3/u_d + bz * rpar/(h*w));  // (C26)

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
    }
  });

  // TODO(@user): error handling

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for SR mhd. Operates
//! only over active cells.
//! Recall in SR mhd the conserved variables are: (D, E-D, m^i, bcc),
//!              and the primitive variables are: (\rho, P_gas, u^i).

void IdealSRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
            DvceArray5D<Real> &cons) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(gm1);
  bool &use_e = eos_data.use_e;

  par_for("srmhd_prim2cons", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    Real& u_m1 = cons(m,IM1,k,j,i);
    Real& u_m2 = cons(m,IM2,k,j,i);
    Real& u_m3 = cons(m,IM3,k,j,i);

    const Real& w_d  = prim(m,IDN,k,j,i);
    const Real& w_ux = prim(m,IVX,k,j,i);
    const Real& w_uy = prim(m,IVY,k,j,i);
    const Real& w_uz = prim(m,IVZ,k,j,i);
    const Real& bcc1 = bcc(m,IBX,k,j,i);
    const Real& bcc2 = bcc(m,IBY,k,j,i);
    const Real& bcc3 = bcc(m,IBZ,k,j,i);

    Real w_p;
    if (use_e) {
      const Real& w_e  = prim(m,IEN,k,j,i);
      w_p = w_e*gm1;
    } else {
      const Real& w_t  = prim(m,ITM,k,j,i);
      w_p = w_t*w_d;
    }

    // Calculate Lorentz factor
    Real u0 = sqrt(1.0 + SQR(w_ux) + SQR(w_uy) + SQR(w_uz));

    // Calculate 4-magnetic field
    Real b0 = bcc1*w_ux + bcc2*w_uy + bcc3*w_uz;
    Real b1 = (bcc1 + b0 * w_ux) / u0;
    Real b2 = (bcc2 + b0 * w_uy) / u0;
    Real b3 = (bcc3 + b0 * w_uz) / u0;
    Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

    // Set conserved quantities
    Real wtot_u02 = (w_d + gamma_prime * w_p + b_sq) * u0 * u0;
    u_d  = w_d * u0;
    u_e  = wtot_u02 - b0 * b0 - (w_p + 0.5*b_sq) - u_d;  // In SR, evolve E - D
    u_m1 = wtot_u02 * w_ux / u0 - b0 * b1;
    u_m2 = wtot_u02 * w_uy / u0 - b0 * b2;
    u_m3 = wtot_u02 * w_uz / u0 - b0 * b3;

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
    }
  });

  return;
}

