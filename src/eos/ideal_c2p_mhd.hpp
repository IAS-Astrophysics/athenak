#ifndef EOS_IDEAL_C2P_MHD_HPP_
#define EOS_IDEAL_C2P_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.hpp
//! \brief Various inline functions that transform a single state of conserved variables
//! into primitive variables (and the reverse, primitive to conserved) for MHD
//! with an ideal gas EOS. Versions for both non-relativistic and relativistic fluids are
//! provided.

//----------------------------------------------------------------------------------------
//! \!fn void SingleC2P_IdealMHD()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Note input CONSERVED state contains cell-centered magnetic
//! fields, but PRIMITIVE state returned through arguments does not.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealMHD(MHDCons1D &u, const EOS_Data &eos,
                        HydPrim1D &w,
                        bool &dfloor_used, bool &efloor_used, bool &tfloor_used) {
  const Real &dfloor_ = eos.dfloor;
  Real efloor = eos.pfloor/(eos.gamma - 1.0);
  Real tfloor = eos.tfloor;
  Real sfloor = eos.sfloor;
  Real gm1 = eos.gamma - 1.0;

  // apply density floor, without changing momentum or energy
  if (u.d < dfloor_) {
    u.d = dfloor_;
    dfloor_used = true;
  }
  w.d = u.d;

  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  // set internal energy, apply floor, correcting total energy
  Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  Real e_m = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  w.e = (u.e - e_k - e_m);
  if (w.e < efloor) {
    w.e = efloor;
    u.e = efloor + e_k + e_m;
    efloor_used = true;
  }
  // apply temperature floor
  if (gm1*w.e*di < tfloor) {
    w.e = w.d*tfloor/gm1;
    u.e = w.e + e_k + e_m;
    tfloor_used =true;
  }
  // apply entropy floor
  Real spe_over_eps = gm1/pow(w.d, gm1);
  Real spe = spe_over_eps*w.e*di;
  if (spe <= sfloor) {
    w.e = w.d*sfloor/spe_over_eps;
    efloor_used = true;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealMHD()
//! \brief Converts single state of primitive variables into conserved variables for
//! non-relativistic MHD with an ideal gas EOS.  Note input PRIMITIVE state contains
//! cell-centered magnetic fields, but CONSERVED state returned via arguments does not.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealMHD(const MHDPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e  = w.e + 0.5*(w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)) +
                        (SQR(w.bx) + SQR(w.by) + SQR(w.bz)) );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation49()
//! \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
//! The root fa(mu)==0 of this function corresponds to the upper bracket for
//! solving Equation44

KOKKOS_INLINE_FUNCTION
Real Equation49(const Real mu, const Real b2, const Real rp, const Real r, const Real q) {
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  Real rbar = (x*x*r*r + mu*x*(1.0 + x)*rp*rp); // (38)
  return mu*sqrt(1.0 + rbar) - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation44()
//! \brief Inline function to compute function f(mu) defined in eq. 44 of Kastaun et al.
//! The ConsToPrim algorithms finds the root of this function f(mu)=0

KOKKOS_INLINE_FUNCTION
Real Equation44(const Real mu, const Real b2, const Real rpar, const Real r, const Real q,
                const Real u_d,  EOS_Data eos) {
  Real const x = 1./(1.+mu*b2);                    // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);   // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar))); // (32)
  Real w = sqrt(1.+z2);
  Real const wd = u_d/w;                           // (34)
  Real eps = w*(qbar - mu*rbar) + z2/(w+1.);
  Real const gm1 = eos.gamma - 1.0;
  Real epsmin = fmax(eos.pfloor/(wd*gm1), eos.sfloor*pow(wd, gm1)/gm1);
  eps = fmax(eps, epsmin);
  Real const h = 1.0 + eos.gamma*eps;              // (43)
  return mu - 1./(h/w + rbar*mu);                  // (45)
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic MHD with an ideal gas EOS. Note input CONSERVED state contains
//! cell-centered magnetic fields, but PRIMITIVE state returned via arguments does not.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2, Real rpar,
                          HydPrim1D &w, bool &dfloor_used, bool &efloor_used,
                          bool &c2p_failure, int &max_iter) {
  // Parameters
  const int max_iterations = 25;
  const Real tol = 1.0e-12;
  const Real gm1 = eos.gamma - 1.0;

  // apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // apply energy floor
  if (u.e < (eos.pfloor/gm1 + 0.5*b2)) {
    u.e = eos.pfloor/gm1 + 0.5*b2;
    efloor_used = true;
  }

  // Recast all variables (eq 22-24)
  Real q = u.e/u.d;
  Real r = sqrt(s2)/u.d;
  Real isqrtd = 1.0/sqrt(u.d);
  Real bx = u.bx*isqrtd;
  Real by = u.by*isqrtd;
  Real bz = u.bz*isqrtd;

  // normalize b2 and rpar as well since they contain b
  b2 /= u.d;
  rpar *= isqrtd;

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

  int iter;
  for (iter=0; iter<iterations; ++iter) {
    z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = Equation49(z, b2, rpar, r, q);
    // Quit if convergence reached
    // NOTE(@ermost): both z and f are of order unity
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }
    // assign zm-->zp if root bracketed by [z,zp]
    if (f*fp < 0.0) {
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
  max_iter = (iter > max_iter) ? iter : max_iter;

  // Found brackets. Now find solution in bounded interval, again using the
  // false position method
  zm= 0.;
  zp= z;

  // Evaluate master function (eq 44) at bracket values
  fm = Equation44(zm, b2, rpar, r, q, u.d, eos);
  fp = Equation44(zp, b2, rpar, r, q, u.d, eos);

  iterations = max_iterations;
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  z = 0.5*(zm + zp);

  for (iter=0; iter<iterations; ++iter) {
    z = (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = Equation44(z, b2, rpar, r, q, u.d, eos);
    // Quit if convergence reached
    // NOTE: both z and f are of order unity
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }
    // assign zm-->zp if root bracketed by [z,zp]
    if (f*fp < 0.0) {
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
  max_iter = (iter > max_iter) ? iter : max_iter;

  // check if convergence is established within max_iterations.  If not, trigger a C2P
  // failure and return floored density, pressure, and primitive velocities. Future
  // development may trigger averaging of (successfully inverted) neighbors in the event
  // of a C2P failure.
  if (max_iter==max_iterations) {
    w.d = eos.dfloor;
    w.e = eos.pfloor/gm1;
    w.vx = 0.0;
    w.vy = 0.0;
    w.vz = 0.0;
    c2p_failure = true;
    return;
  }

  // iterations ended, compute primitives from resulting value of z
  Real &mu = z;
  Real const x = 1./(1.+mu*b2);                               // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);              // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar - rpar*rpar)); // (31)
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));            // (32)
  Real lor = sqrt(1.0 + z2);

  // compute density then apply floor
  Real dens = u.d/lor;
  if (dens < eos.dfloor) {
    dens = eos.dfloor;
    dfloor_used = true;
  }

  // compute specific internal energy density then apply floors
  Real eps = lor*(qbar - mu*rbar) + z2/(lor + 1.0);
  Real epsmin = fmax(eos.pfloor/(dens*gm1), eos.sfloor*pow(dens, gm1)/gm1);
  if (eps <= epsmin) {
    eps = epsmin;
    efloor_used = true;
  }

  // set parameters required for velocity inversion
  Real const h = 1.0 + eos.gamma*eps;  // (43)
  Real const conv = lor/(h*lor + b2);  // (C26)

  // set primitive variables
  w.d  = dens;
  w.vx = conv*(u.mx/u.d + bx*rpar/(h*lor));  // (C26)
  w.vy = conv*(u.my/u.d + by*rpar/(h*lor));  // (C26)
  w.vz = conv*(u.mz/u.d + bz*rpar/(h*lor));  // (C26)
  w.e  = dens*eps;

  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealSRMHD()
//! \brief Converts single set of primitive into conserved variables in SRMHD.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealSRMHD(const MHDPrim1D &w, const Real gam, HydCons1D &u) {
  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));

  // Calculate 4-magnetic field
  Real b0 = w.bx*w.vx + w.by*w.vy + w.bz*w.vz;
  Real b1 = (w.bx + b0 * w.vx) / u0;
  Real b2 = (w.by + b0 * w.vy) / u0;
  Real b3 = (w.bz + b0 * w.vz) / u0;
  Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

  // Set conserved quantities
  Real wtot_u02 = (w.d + gam * w.e + b_sq) * u0 * u0;
  u.d  = w.d * u0;
  u.e  = wtot_u02 - b0 * b0 - ((gam-1.0)*w.e + 0.5*b_sq) - u.d;  // In SR, evolve E - D
  u.mx = wtot_u02 * w.vx / u0 - b0 * b1;
  u.my = wtot_u02 * w.vy / u0 - b0 * b2;
  u.mz = wtot_u02 * w.vz / u0 - b0 * b3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TransformToSRMHD()
//! \brief Converts single state of conserved variables in GR MHD into conserved
//! variables for special relativistic MHD with an ideal gas EOS. This allows
//! the ConsToPrim() function in GR MHD to use SingleP2C_IdealSRMHD() function.

KOKKOS_INLINE_FUNCTION
void TransformToSRMHD(const MHDCons1D &u, Real glower[][4], Real gupper[][4],
                      Real &s2, Real &b2, Real &rpar, MHDCons1D &u_sr) {
  // Need to multiply the conserved density by alpha, so that it
  // contains a lorentz factor
  Real alpha = sqrt(-1.0/gupper[0][0]);
  u_sr.d = u.d*alpha;

  // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
  // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
  // We are also evolving T^t_t + D as conserved variable, so must convert to E
  u_sr.e = gupper[0][0]*(u.e - u.d) +
           gupper[0][1]*u.mx + gupper[0][2]*u.my + gupper[0][3]*u.mz;

  // This is only true if sqrt{-g}=1!
  u_sr.e *= (-1./gupper[0][0]);  // Multiply by alpha^2

  // Subtract density for consistency with the rest of the algorithm
  u_sr.e -= u_sr.d;

  // Need to treat the conserved momenta. Also they lack an alpha
  // This is only true if sqrt{-g}=1!
  Real m1l = u.mx*alpha;
  Real m2l = u.my*alpha;
  Real m3l = u.mz*alpha;

  // Need to raise indices on u_m1, which transforms using the spatial 3-metric.
  // Store in u_sr.  This is slightly more involved
  //
  // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
  //       g^0i = beta^i/alpha^2
  //       g^00 = -1/ alpha^2
  // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
  u_sr.mx = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  u_sr.my = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  u_sr.mz = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  // Compute (S^i S_i) (eqn C2)
  s2 = (m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz);

  // load magnetic fields into SR conserved state. Also they lack an alpha
  // This is only true if sqrt{-g}=1!
  u_sr.bx = alpha*u.bx;
  u_sr.by = alpha*u.by;
  u_sr.bz = alpha*u.bz;

  b2 = glower[1][1]*SQR(u_sr.bx) + glower[2][2]*SQR(u_sr.by) + glower[3][3]*SQR(u_sr.bz) +
       2.0*(u_sr.bx*(glower[1][2]*u_sr.by + glower[1][3]*u_sr.bz) +
                     glower[2][3]*u_sr.by*u_sr.bz);
  rpar = (u_sr.bx*m1l +  u_sr.by*m2l +  u_sr.bz*m3l)/u_sr.d;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn voidSingleP2C_IdealGRMHD()
//! \brief Converts single set of primitive into conserved variables in GRMHD.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealGRMHD(const Real glower[][4], const Real gupper[][4],
                          const MHDPrim1D &w, const Real gam, HydCons1D &u) {
  // Calculate 4-velocity (exploiting symmetry of metric)
  Real q = glower[1][1]*w.vx*w.vx +2.0*glower[1][2]*w.vx*w.vy +2.0*glower[1][3]*w.vx*w.vz
         + glower[2][2]*w.vy*w.vy +2.0*glower[2][3]*w.vy*w.vz
         + glower[3][3]*w.vz*w.vz;
  Real alpha = sqrt(-1.0/gupper[0][0]);
  Real gamma = sqrt(1.0 + q);
  Real u0 = gamma / alpha;
  Real u1 = w.vx - alpha * gamma * gupper[0][1];
  Real u2 = w.vy - alpha * gamma * gupper[0][2];
  Real u3 = w.vz - alpha * gamma * gupper[0][3];

  // lower vector indices
  Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
  Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
  Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

  // Calculate 4-magnetic field
  Real b0 = u_1*w.bx + u_2*w.by + u_3*w.bz;
  Real b1 = (w.bx + b0 * u1) / u0;
  Real b2 = (w.by + b0 * u2) / u0;
  Real b3 = (w.bz + b0 * u3) / u0;

  // lower vector indices
  Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
  Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
  Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
  Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
  Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  Real wtot = w.d + gam * w.e + b_sq;
  Real ptot = (gam-1.0)*w.e + 0.5 * b_sq;
  u.d  = w.d * u0;
  u.e  = wtot * u0 * u_0 - b0 * b_0 + ptot + u.d;  // evolve T^t_t + D
  u.mx = wtot * u0 * u_1 - b0 * b_1;
  u.my = wtot * u0 * u_2 - b0 * b_2;
  u.mz = wtot * u0 * u_3 - b0 * b_3;
  return;
}

///----------------------------------------------------------------------------------------
//! \fn bool GetPrimEntropyFix()
//! \brief Inline function to compute density, pressure, and Lorentz factor from total entropy.
//! See details in Appendix~C of the thesis by Lizhong Zhang (LZ).

KOKKOS_INLINE_FUNCTION
bool GetPrimEntropyFix(const MHDCons1D &u, const Real s_tot, EOS_Data eos,
                       const Real s2, const Real b2, const Real rpar,
                       const Real ll, Real &rho, Real &pgas, Real &gamma)
{
  Real tt = rpar * u.d; // (LZ C9)
  Real v_sq = ( s2*SQR(ll) + SQR(tt)*(2*ll+b2) ) / SQR(ll*(ll+b2)); // (LZ C12)
  const Real v_sq_max = 1. - 1./SQR(eos.gamma_max);
  if (v_sq > v_sq_max) {
    // Adopt zero velocity for initial guess if velocity ceiling is reached.
    // The velocity ceiling as initial guess is usually less numerically stable
    // due to the existence of some local minimums in ultra-relativistic regime.
    v_sq = 0.;
  }
  v_sq  = fmax(v_sq, static_cast<Real>(0.0));
  gamma = 1./sqrt(1-v_sq);
  rho   = u.d/gamma;
  pgas  = s_tot/u.d * pow(rho, eos.gamma);
  if (isfinite(rho)  && (rho > 0)
   && isfinite(pgas) && (pgas > 0)
   && isfinite(gamma)) {
    // primitives are physical
    return true;
  } else {
    return false;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool GetVelEntropyFix()
//! \brief Inline function to compute velocity in entropy fix
//! See details in Appendix~C of LZ's thesis and Newman & Hamlin (NH, 2014)

KOKKOS_INLINE_FUNCTION
bool GetVelEntropyFix(const MHDCons1D &u, const Real b2, const Real rpar,
                      const Real ll, const Real gamma, Real &u1, Real &u2, Real &u3) {
  Real tt = rpar * u.d; // (LZ C9)
  Real v1 = (u.mx + tt*u.bx/ll) / (ll + b2); // (NH 3.7, 4.6, 4.7, 4.8 & 5.1)
  Real v2 = (u.my + tt*u.by/ll) / (ll + b2); // (NH 3.7, 4.6, 4.7, 4.8 & 5.1)
  Real v3 = (u.mz + tt*u.bz/ll) / (ll + b2); // (NH 3.7, 4.6, 4.7, 4.8 & 5.1)
  u1 = gamma * v1;
  u2 = gamma * v2;
  u3 = gamma * v3;
  if (isfinite(u1) && isfinite(u2) && isfinite(u3)
      && (SQR(v1)+SQR(v2)+SQR(v3) < 1)) {
    // velocity is physical
    return true;
  } else return false;
}

//----------------------------------------------------------------------------------------
//! \fn bool EquationC13()
//! \brief Inline function to compute target function for Newton-Raphson iteration
//! See details in Appendix~C of LZ's thesis

KOKKOS_INLINE_FUNCTION
bool EquationC13(const MHDCons1D &u, const Real s_tot, EOS_Data eos,
                 const Real s2, const Real b2, const Real rpar,
                 const Real rho, const Real pgas, const Real gamma,
                 const Real ll, Real &ff, Real &dff)
{
  Real tt = rpar * u.d; // (LZ C9)
  Real gamma_adi = eos.gamma;
  Real gm1 = gamma_adi - 1;
  Real gp1 = gamma_adi + 1;

  Real v_sq = 1. - 1./SQR(gamma);
  ff = u.d*pgas / pow(rho, gamma_adi) - s_tot;                      // (LZ C13)
  Real dv_sq = -2. / pow(ll*(ll+b2), 3);                            // (LZ C14d)
  dv_sq *= SQR(tt) * (3*ll*(ll+b2) + SQR(b2)) + s2*pow(ll, 3);      // (LZ C14d)
  Real dpgas = gm1/gamma_adi * (1-v_sq + (0.5*u.d*gamma-ll)*dv_sq); // (LZ C14b)
  Real drho = -0.5*u.d*gamma*dv_sq;                                 // (LZ C14c)
  dff = u.d/pow(rho, gamma_adi)*dpgas;                              // (LZ C14a)
  dff -= gamma_adi*u.d*pgas/pow(rho, gp1)*drho;                     // (LZ C14a)

  if (isfinite(ff) && isfinite(dff)) {
  	return true;
  } else return false;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD_EntropyFix()
//! \brief Converts single state of entropy-based conserved variables into primitive variables for
//! special relativistic MHD with an ideal gas EOS. Note input CONSERVED state contains
//! cell-centered magnetic fields, but PRIMITIVE state returned via arguments does not.
//! The algorithm follows Mignone & McKinney (MM, 2007) but modified for the entropy case by LZ.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD_EntropyFix(MHDCons1D &u, Real& s_tot, const EOS_Data &eos,
                                     Real s2, Real b2, Real rpar, HydPrim1D &w, HydPrim1D &w_old,
                                     bool &dfloor_used, bool &efloor_used,
                                     bool &c2p_failure, int &max_iter) {
  // Parameters
  const int max_iterations = 25;
  const Real tol = 1.0e-12;
  const Real gm1 = eos.gamma - 1.0;

  // Initialize variables
  bool flag = false;
  Real ll_;
  Real rho_, pgas_, gamma_, u1_, u2_, u3_;

  // Apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // Apply energy floor
  if (u.e < (eos.pfloor/gm1 + 0.5*b2)) {
    u.e = eos.pfloor/gm1 + 0.5*b2;
    efloor_used = true;
  }

  // Step 1: Set up initial guess
  if (!c2p_failure) {
    // use the previously computed velocity for initial guess
    u1_ = w.vx;
    u2_ = w.vy;
    u3_ = w.vz;
  } else {
    // use the old-timestep velocity for initial guess
    u1_ = w_old.vx;
    u2_ = w_old.vy;
    u3_ = w_old.vz;
  }
  gamma_ = sqrt(1+SQR(u1_)+SQR(u2_)+SQR(u3_));
  rho_   = u.d/gamma_;
  pgas_  = s_tot * pow(u.d, gm1) / pow(gamma_, gm1+1);
  ll_ = u.d*gamma_ + (gm1+1)/gm1 * pgas_ * SQR(gamma_);
  if (isfinite(ll_) && (ll_ > 0)
      && isfinite(rho_) && (rho_ > 0)
      && isfinite(pgas_) && (pgas_ > 0)){
    flag = true;
  }

  // backup initial guess
  Real ee = u.e + u.d;
  if (!flag) {
    // this guess is based on (d, m^i, e): (MM A27) set f=0 and assume v=1
    Real c2 = 3.;
    Real c1 = 4. * (b2 - ee);
    Real c0 = s2 + SQR(b2) - 2*b2*ee;

    Real ll_a, ll_b;
    Real delta_sq = SQR(c1)-4*c2*c0;
    if (delta_sq >= 0) { // (dens, mom^i, etot) have real solution
      Real delta = sqrt(delta_sq);
      if (c1 >= 0) {
        ll_a = (-c1 - delta) / (2*c2);
        ll_b = (2*c0) / (-c1 - delta);
      } else {
        ll_a = (2*c0) / (-c1 + delta);
        ll_b = (-c1 + delta) / (2*c2);
      }
      ll_ = fmax(ll_a, ll_b);
      if (isfinite(ll_) && (ll_ > 0)) {
        // ll_ is physical
        flag = GetPrimEntropyFix(u, s_tot, eos, s2, b2, rpar, ll_, rho_, pgas_, gamma_);
      }
    } // endif
  } // endif backup initial guess

  // Step 2: Newton-Raphson iteration to solve primitives
  if (flag) {
  	int n;
    for (n = 0; n < max_iterations; ++n) {
      Real ff_, dff_;
      flag = EquationC13(u, s_tot, eos, s2, b2, rpar, rho_, pgas_, gamma_, ll_, ff_, dff_);
      if (!flag) break; // ff_ and dff_ are invalid

      Real ll_new = ll_ - ff_/dff_;
      flag = GetPrimEntropyFix(u, s_tot, eos, s2, b2, rpar, ll_new, rho_, pgas_, gamma_);
      if (!flag) break; // new primitives are not physical

      if (fabs(ll_new-ll_) < tol) {
        // converging within tolerance
        ll_ = ll_new;
        break;
      } else ll_ = ll_new; // continue iteration
    } // endfor
    if (n == max_iterations) flag = false; // maximum iteration
    max_iter = n;
  } // endif
  if (flag) flag = GetVelEntropyFix(u, b2, rpar, ll_, gamma_, u1_, u2_, u3_);

  // Step 3: Apply density and energy floors
  if (rho_ < eos.dfloor) {
    rho_ = eos.dfloor;
    dfloor_used = true;
  }

  Real pgas_min = fmax(eos.pfloor, eos.sfloor*pow(rho_, eos.gamma));
  if (pgas_ < pgas_min) {
    pgas_ = pgas_min;
    efloor_used = true;
  }

  // Step 4: Save primitive variables
  w.d  = rho_;
  w.e  = pgas_/gm1;
  w.vx = u1_;
  w.vy = u2_;
  w.vz = u3_;
  c2p_failure = !flag;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD_NH()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic MHD with an ideal gas EOS. Note input CONSERVED state contains
//! cell-centered magnetic fields, but PRIMITIVE state returned via arguments does not.
//! The implemented algorithm follows Newman & Hamlin (2014).

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD_NH(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2, Real rpar,
                             HydPrim1D &w, HydPrim1D &w_old, bool &dfloor_used, bool &efloor_used,
                             bool &c2p_failure, int &max_iter) {
  // parameters
  const int max_iterations = 25;
  const Real a_min = 1.0e-12; // a>0 due to its definition a=ee+pgas+0.5*bb_sq
  const Real rr_max = 1.0 - 1.0e-12; // allowed maximum of Lipshitz parameter
  const Real tol = 1.0e-12;
  c2p_failure = false;
  const Real gm1 = eos.gamma - 1.0;
  const Real v_sq_max = 1. - 1./SQR(eos.gamma_max);

  // extract conserved values
  const Real &mm1 = u.mx;
  const Real &mm2 = u.my;
  const Real &mm3 = u.mz;
  const Real &bb1 = u.bx;
  const Real &bb2 = u.by;
  const Real &bb3 = u.bz;
  const Real &mm_sq = s2;
  const Real &bb_sq = b2;
  const Real tt = rpar * u.d;

  // apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // apply energy floor
  if (u.e < (eos.pfloor/gm1 + 0.5*b2)) {
    u.e = eos.pfloor/gm1 + 0.5*b2;
    efloor_used = true;
  }

  // extract the rest of conserved values
  const Real &dd  = u.d;
  const Real ee  = u.e + u.d;

  // calculate functions of conserved quantities
  Real d = 0.5 * (mm_sq * bb_sq - SQR(tt)); // (NH 5.7)
  d = fmax(d, 0.0);
  Real pgas_min = cbrt(27.0/4.0 * d) - ee - 0.5*bb_sq;
  pgas_min = fmax(pgas_min, eos.pfloor);

  // iterate until convergence
  Real pgas[3];
  pgas[0] = fmax(gm1*w_old.e, pgas_min);
  int n;
  for (n = 0; n < max_iterations; ++n) {
    Real a;
    Real phi, eee, ll, v_sq;
    if (n%3 != 2) {
      // Step 1: Calculate cubic coefficients
      a = ee + pgas[n%3] + 0.5*bb_sq;  // (NH 5.7)
      a = fmax(a, a_min);

      // Step 2: Calculate correct root of cubic equation
      phi = acos(1.0/a * sqrt(27.0*d/(4.0*a)));                               // (NH 5.10)
      eee = a/3.0 - 2.0/3.0 * a * cos(2.0/3.0 * (phi+M_PI));                  // (NH 5.11)
      ll = eee - bb_sq;                                                       // (NH 5.5)
      v_sq = (mm_sq*SQR(ll) + SQR(tt)*(bb_sq+2.0*ll)) / SQR(ll * (bb_sq+ll)); // (NH 5.2)
      v_sq = fmin(fmax(v_sq, static_cast<Real>(0.0)), v_sq_max);
      Real gamma_sq = 1.0/(1.0-v_sq);                                         // (NH 3.1)
      Real gamma = sqrt(gamma_sq);                                            // (NH 3.1)
      Real wgas = ll/gamma_sq;                                                // (NH 5.1)
      Real rho = dd/gamma;                                                    // (NH 4.5)
      pgas[(n+1)%3] = gm1/(gm1+1) * (wgas - rho);                             // (NH 4.1)
      pgas[(n+1)%3] = fmax(pgas[(n+1)%3], pgas_min);

      // Step 3: Check for convergence
      if (pgas[(n+1)%3] > pgas_min && fabs(pgas[(n+1)%3]-pgas[n%3]) < tol) {
        break;
      }
    } // endfor n

    // Step 4: Calculate Aitken accelerant and check for convergence
    if (n%3 == 2) {
      Real rr = (pgas[2] - pgas[1]) / (pgas[1] - pgas[0]);  // (NH 7.1)
      if (!isfinite(rr) || fabs(rr) > rr_max) {
        continue; // invalid Lipshitz parameter, start with next loop
      }
      pgas[0] = pgas[1] + (pgas[2] - pgas[1]) / (1.0 - rr); // (NH 7.2)
      pgas[0] = fmax(pgas[0], pgas_min);
      if (pgas[0] > pgas_min && fabs(pgas[0]-pgas[2]) < tol) {
        break;
      }
    }
  } // endfor n
  max_iter = n;

  // Step 5: Set primitives
  if (n == max_iterations) {
    c2p_failure = true; // reach max iteration number
  }
  Real pgas_ret = pgas[(n+1)%3];
  if (!isfinite(pgas_ret) || (pgas_ret<=0)) {
    c2p_failure = true; // solution is not physical
  }
  Real a = ee + pgas_ret + 0.5*bb_sq;                                          // (NH 5.7)
  a = fmax(a, a_min);
  Real phi = acos(1.0/a * sqrt(27.0*d/(4.0*a)));                               // (NH 5.10)
  Real eee = a/3.0 - 2.0/3.0 * a * cos(2.0/3.0 * (phi+M_PI));                  // (NH 5.11)
  Real ll = eee - bb_sq;                                                       // (NH 5.5)
  Real v_sq = (mm_sq*SQR(ll) + SQR(tt)*(bb_sq+2.0*ll)) / SQR(ll * (bb_sq+ll)); // (NH 5.2)
  v_sq = fmin(fmax(v_sq, static_cast<Real>(0.0)), v_sq_max);
  Real gamma_sq = 1.0/(1.0-v_sq);                                              // (NH 3.1)
  Real gamma = sqrt(gamma_sq);                                                 // (NH 3.1)
  Real rho_ret = dd/gamma;                                                     // (NH 4.5)
  if (!isfinite(rho_ret) || (rho_ret <= 0)) {
    c2p_failure = true; // solution is not physical
  }
  Real ss = tt/ll;                          // (NH 4.8)
  Real v1 = (mm1 + ss*bb1) / (ll + bb_sq);  // (NH 4.6)
  Real v2 = (mm2 + ss*bb2) / (ll + bb_sq);  // (NH 4.6)
  Real v3 = (mm3 + ss*bb3) / (ll + bb_sq);  // (NH 4.6)
  Real u1_ret = gamma*v1;                   // (NH 3.3)
  Real u2_ret = gamma*v2;                   // (NH 3.3)
  Real u3_ret = gamma*v3;                   // (NH 3.3)
  if (!isfinite(u1_ret) || !isfinite(u2_ret) || !isfinite(u3_ret)
      || (SQR(v1)+SQR(v2)+SQR(v3) > 1)) {
    c2p_failure = true; // solution is not physical
  }

  // if c2p fails, return floored density, pressure, and primitive velocities.
  if (c2p_failure) {
    w.d = eos.dfloor;
    w.e = eos.pfloor/gm1;
    w.vx = 0.0;
    w.vy = 0.0;
    w.vz = 0.0;
    return;
  }

  // compute density then apply floor
  if (rho_ret < eos.dfloor) {
    rho_ret = eos.dfloor;
    dfloor_used = true;
  }

  // compute specific internal energy density then apply floors
  Real eps = pgas_ret/(rho_ret*gm1);
  Real epsmin = fmax(eos.pfloor/(rho_ret*gm1), eos.sfloor*pow(rho_ret, gm1)/gm1);
  if (eps <= epsmin) {
    eps = epsmin;
    efloor_used = true;
  }

  // set primitive variables
  w.d  = rho_ret;
  w.vx = u1_ret;
  w.vy = u2_ret;
  w.vz = u3_ret;
  w.e  = rho_ret*eps;

  return;
}

#endif // EOS_IDEAL_C2P_MHD_HPP_
