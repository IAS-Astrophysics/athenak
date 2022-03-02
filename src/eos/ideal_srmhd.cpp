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

IdealSRMHD::IdealSRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//--------------------------------------------------------------------------------------
//! \fn void PrimToConsSingle()
//! \brief Converts primitive into conserved variables in SRMHD.
//! Operates on only one active cell.

KOKKOS_INLINE_FUNCTION
void PrimToConsSingle(const Real &gammap, const Real &bcc1, const Real &bcc2,
                      const Real &bcc3, const HydPrim1D &w, HydCons1D &u) {
  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));

  // Calculate 4-magnetic field
  Real b0 = bcc1*w.vx + bcc2*w.vy + bcc3*w.vz;
  Real b1 = (bcc1 + b0 * w.vx) / u0;
  Real b2 = (bcc2 + b0 * w.vy) / u0;
  Real b3 = (bcc3 + b0 * w.vz) / u0;
  Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

  // Set conserved quantities
  Real wtot_u02 = (w.d + gammap * w.p + b_sq) * u0 * u0;
  u.d  = w.d * u0;
  u.e  = wtot_u02 - b0 * b0 - (w.p + 0.5*b_sq) - u.d;  // In SR, evolve E - D
  u.mx = wtot_u02 * w.vx / u0 - b0 * b1;
  u.my = wtot_u02 * w.vy / u0 - b0 * b2;
  u.mz = wtot_u02 * w.vz / u0 - b0 * b3;
  return;
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
//! This function operates over range of cells given in argument list.

void IdealSRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(gm1);

  Real &dfloor_ = eos_data.dfloor;
  Real &pfloor_ = eos_data.pfloor;

  // Parameters
  int const max_iterations = 15;
  Real const tol = 1.0e-12;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, maxit_=0;
  Kokkos::parallel_reduce("hyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum_d, int &sum_e, int &max_iter) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    const Real& u_m1 = cons(m,IM1,k,j,i);
    const Real& u_m2 = cons(m,IM2,k,j,i);
    const Real& u_m3 = cons(m,IM3,k,j,i);

    Real& w_d  = prim(m,IDN,k,j,i);
    Real& w_ux = prim(m,IVX,k,j,i);
    Real& w_uy = prim(m,IVY,k,j,i);
    Real& w_uz = prim(m,IVZ,k,j,i);
    Real& w_e  = prim(m,IEN,k,j,i);

    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc(m,IBX,k,j,i);
    Real& w_by = bcc(m,IBY,k,j,i);
    Real& w_bz = bcc(m,IBZ,k,j,i);
    w_bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
    w_by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
    w_bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

    // apply density floor, without changing momentum or energy
    bool floor_hit = false;
    if (u_d < dfloor_) {
      u_d = dfloor_;
      sum_d++;
      floor_hit = true;
    }

    // apply energy floor
    // Real ee_min = pfloor_/gm1;
    // u_e = (u_e > ee_min) ?  u_e : ee_min;

    // Recast all variables (eq 22-24)
    Real q = u_e/u_d;  // q is (E-D)/D, and we evolve u_e = E-D
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

    {int iter;
    for (iter=0; iter < iterations; ++iter) {
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
    max_iter = (iter > max_iter)? iter : max_iter;
    }

    // Found brackets. Now find solution in bounded interval, again using the
    // false position method
    zm= 0.;
    zp= z;

    // Evaluate master function (eq 44) at bracket values
    fm = Equation44(zm, b2, rpar, r, q, u_d, pfloor_, gm1);
    fp = Equation44(zp, b2, rpar, r, q, u_d, pfloor_, gm1);

    iterations = max_iterations;
    if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
      iterations = -1;
    }
    z = 0.5*(zm + zp);

    {int iter;
    for (iter=0; iter < iterations; ++iter) {
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
    max_iter = (iter > max_iter)? iter : max_iter;
    }

    // iterations ended, compute primitives from resulting value of z
    Real &mu = z;
    Real const x = 1./(1.+mu*b2);                              // (26)
    Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);             // (38)
    Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
    // rbar = sqrt(rbar);

    Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));           // (32)
    Real w = sqrt(1.+z2);

    w_d = u_d/w;                                               // (34)
    Real eps = w*(qbar - mu*rbar)+  z2/(w+1.);
    Real epsmin = pfloor_/(w_d*gm1);
    if (eps <= epsmin) {
      eps = epsmin;
      sum_e++;
      floor_hit = true;
    }

    //NOTE: The following generalizes to ANY equation of state
    Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));  // (43)
    w_e = w_d*eps;

    Real const conv = w/(h*w + b2); // (C26)
    w_ux = conv * ( u_m1/u_d + bx * rpar/(h*w));  // (C26)
    w_uy = conv * ( u_m2/u_d + by * rpar/(h*w));  // (C26)
    w_uz = conv * ( u_m3/u_d + bz * rpar/(h*w));  // (C26)

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
    }

    // reset conserved variables if floor is hit
    if (floor_hit) {
      HydPrim1D w;
      w.d  = w_d;
      w.vx = w_ux;
      w.vy = w_uy;
      w.vz = w_uz;
      w.p = w_e*gm1;

      HydCons1D u;
      PrimToConsSingle(gamma_prime, w_bx, w_by, w_bz, w, u);

      cons(m,IDN,k,j,i) = u.d;
      cons(m,IEN,k,j,i) = u.e;
      cons(m,IM1,k,j,i) = u.mx;
      cons(m,IM2,k,j,i) = u.my;
      cons(m,IM3,k,j,i) = u.mz;
      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        cons(m,n,k,j,i) = prim(m,n,k,j,i)*cons(m,IDN,k,j,i);
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Max<int>(maxit_));

  // store counters
  pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
  pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for SR mhd. Operates over range
//! of cells given in argument list.
//! Recall in SR mhd the conserved variables are: (D, E-D, m^i, bcc),
//!              and the primitive variables are: (\rho, P_gas, u^i).

void IdealSRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(gm1);

  par_for("srmhd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    Real& u_m1 = cons(m,IM1,k,j,i);
    Real& u_m2 = cons(m,IM2,k,j,i);
    Real& u_m3 = cons(m,IM3,k,j,i);

    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.p = prim(m,IEN,k,j,i)*gm1;
    const Real& bcc1 = bcc(m,IBX,k,j,i);
    const Real& bcc2 = bcc(m,IBY,k,j,i);
    const Real& bcc3 = bcc(m,IBZ,k,j,i);

    HydCons1D u;
    PrimToConsSingle(gamma_prime, bcc1, bcc2, bcc3, w, u);

    // Set conserved quantities
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IEN,k,j,i) = u.e;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
    }
  });

  return;
}
