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

#endif // EOS_IDEAL_C2P_MHD_HPP_
