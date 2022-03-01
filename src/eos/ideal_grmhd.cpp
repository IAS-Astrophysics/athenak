//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd

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

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//--------------------------------------------------------------------------------------
//! \fn void PrimToConsSingle()
//! \brief Converts primitive into conserved variables in GRMHD.
//! Operates on only one active cell.

KOKKOS_INLINE_FUNCTION
void PrimToConsSingle(const Real g_[], const Real gi_[], const Real &gammap,
                      const Real &bcc1, const Real &bcc2, const Real &bcc3,
                      const HydPrim1D &w, HydCons1D &u) {
  const Real
    &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
    &g_10 = g_[I01], &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
    &g_20 = g_[I02], &g_21 = g_[I12], &g_22 = g_[I22], &g_23 = g_[I23],
    &g_30 = g_[I03], &g_31 = g_[I13], &g_32 = g_[I23], &g_33 = g_[I33];

  // Calculate 4-velocity
  Real alpha = sqrt(-1.0/gi_[I00]);
  Real tmp = g_[I11]*w.vx*w.vx + 2.0*g_[I12]*w.vx*w.vy + 2.0*g_[I13]*w.vx*w.vz
           + g_[I22]*w.vy*w.vy + 2.0*g_[I23]*w.vy*w.vz
           + g_[I33]*w.vz*w.vz;
  Real gg = sqrt(1.0 + tmp);
  Real u0 = gg/alpha;
  Real u1 = w.vx - alpha * gg * gi_[I01];
  Real u2 = w.vy - alpha * gg * gi_[I02];
  Real u3 = w.vz - alpha * gg * gi_[I03];
  // lower vector indices
  Real u_0 = g_00*u0 + g_01*u1 + g_02*u2 + g_03*u3;
  Real u_1 = g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3;
  Real u_2 = g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3;
  Real u_3 = g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3;

  // Calculate 4-magnetic field
  Real b0 = g_[I01]*u0*bcc1 + g_[I02]*u0*bcc2 + g_[I03]*u0*bcc3
          + g_[I11]*u1*bcc1 + g_[I12]*u1*bcc2 + g_[I13]*u1*bcc3
          + g_[I12]*u2*bcc1 + g_[I22]*u2*bcc2 + g_[I23]*u2*bcc3
          + g_[I13]*u3*bcc1 + g_[I23]*u3*bcc2 + g_[I33]*u3*bcc3;
  Real b1 = (bcc1 + b0 * u1) / u0;
  Real b2 = (bcc2 + b0 * u2) / u0;
  Real b3 = (bcc3 + b0 * u3) / u0;
  // lower vector indices
  Real b_0 = g_00*b0 + g_01*b1 + g_02*b2 + g_03*b3;
  Real b_1 = g_10*b0 + g_11*b1 + g_12*b2 + g_13*b3;
  Real b_2 = g_20*b0 + g_21*b1 + g_22*b2 + g_23*b3;
  Real b_3 = g_30*b0 + g_31*b1 + g_32*b2 + g_33*b3;
  Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  Real wtot = w.d + gammap * w.p + b_sq;
  Real ptot = w.p + 0.5 * b_sq;
  u.d  = w.d * u0;
  u.e  = wtot * u0 * u_0 - b0 * b_0 + ptot - u.d;  // evolve E-D, as in SR
  u.mx = wtot * u0 * u_1 - b0 * b_1;
  u.my = wtot * u0 * u_2 - b0 * b_2;
  u.mz = wtot * u0 * u_3 - b0 * b_3;
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
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &rmin = pmy_pack->pcoord->coord_data.bh_rmin;
  Real &dfloor_ = eos_data.dfloor;
  Real &pfloor_ = eos_data.pfloor;
  Real ee_min = pfloor_/gm1;

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

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    // Extract components of metric
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, false, spin, g_, gi_);

    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    // Only execute cons2prim if outside excised region
    bool floor_hit = false;
    if (rad > rmin) {
      // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
      // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
      // We are also evolving (E-D) as conserved variable, so must convert to E
      Real ue_sr = gi_[I00]*(u_e+u_d) + gi_[I01]*u_m1 + gi_[I02]*u_m2 + gi_[I03]*u_m3;

      // This is only true if sqrt{-g}=1!
      ue_sr *= (-1./gi_[I00]);  // Multiply by alpha^2

      // Need to multiply the conserved density by alpha, so that it
      // contains a lorentz factor
      Real alpha = sqrt(-1.0/gi_[I00]);
      Real ud_sr = u_d*alpha;

      // Subtract density for consistency with the rest of the algorithm
      ue_sr -= ud_sr;

      // Need to treat the conserved momenta. Also they lack an alpha
      // This is only true if sqrt{-g}=1!
      Real um1_sr = u_m1*alpha;
      Real um2_sr = u_m2*alpha;
      Real um3_sr = u_m3*alpha;

      // apply density floor, without changing momentum or energy
      if (ud_sr < dfloor_) {
        ud_sr = dfloor_;
        sum_d++;
        floor_hit = true;
      }

      // apply energy floor
      if (ue_sr < pfloor_/gm1) {
        ue_sr = pfloor_/gm1;
        sum_e++;
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
      Real m1u = ((gi_[I11] - gi_[I01]*gi_[I01]/gi_[I00])*um1_sr +
                  (gi_[I12] - gi_[I01]*gi_[I02]/gi_[I00])*um2_sr +
                  (gi_[I13] - gi_[I01]*gi_[I03]/gi_[I00])*um3_sr);  // (C26)

      Real m2u = ((gi_[I12] - gi_[I01]*gi_[I02]/gi_[I00])*um1_sr +
                  (gi_[I22] - gi_[I02]*gi_[I02]/gi_[I00])*um2_sr +
                  (gi_[I23] - gi_[I02]*gi_[I03]/gi_[I00])*um3_sr);  // (C26)

      Real m3u = ((gi_[I13] - gi_[I01]*gi_[I03]/gi_[I00])*um1_sr +
                  (gi_[I23] - gi_[I02]*gi_[I03]/gi_[I00])*um2_sr +
                  (gi_[I33] - gi_[I03]*gi_[I03]/gi_[I00])*um3_sr);  // (C26)

      // Recast all variables (eq 22-24)
      // Variables q and r defined in anonymous namspace: global this file
      Real q = ue_sr/ud_sr;
      Real r = sqrt(um1_sr*m1u + um2_sr*m2u + um3_sr*m3u)/ud_sr;

      Real sqrtd = sqrt(ud_sr);
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
      Real rpar = (bx*um1_sr +  by*um2_sr +  bz*um3_sr)/ud_sr;

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
        } else { // assign zp-->z if root bracketed by [zm,z]
          fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
          zp = z;
          fp = f;
        }
      }
      max_iter = (iter > max_iter)? iter : max_iter;
      }

      zm= 0.;
      zp= z;

      // Evaluate master function (eq 44) at bracket values
      fm = Equation44(zm, b2, rpar, r, q, ud_sr, pfloor_, gm1);
      fp = Equation44(zp, b2, rpar, r, q, ud_sr, pfloor_, gm1);

      // For simplicity on the GPU, find roots using the false position method
      iterations = max_iterations;
      // If bracket within tolerances, don't bother doing any iterations
      if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
        iterations = -1;
      }
      z = 0.5*(zm + zp);

      {int iter;
      for (int iter=0; iter < iterations; ++iter) {
        z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = Equation44(z, b2, rpar, r, q, ud_sr, pfloor_, gm1);

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
        // assign zp-->z if root bracketed by [zm,z]
        } else {
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
            //  rbar = sqrt(rbar);

      Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));           // (32)
      Real w = sqrt(1.+z2);

      w_d = ud_sr/w;                                             // (34)
      Real eps = w*(qbar - mu*rbar)+  z2/(w+1.);
      Real epsmin = pfloor_/(w_d*gm1);
      if (eps <= epsmin) {
        eps = epsmin;
        sum_e++;
        floor_hit = true;
      }

      Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));   // (43)
      w_e = w_d*eps;

      Real const conv = w/(h*w + b2); // (C26)
      w_ux = conv * ( m1u/ud_sr + bx * rpar/(h*w));           // (C26)
      w_uy = conv * ( m2u/ud_sr + by * rpar/(h*w));           // (C26)
      w_uz = conv * ( m3u/ud_sr + bz * rpar/(h*w));           // (C26)

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }
    }

    // reset conserved variables inside excised regions or if floor is hit
    if (rad <= rmin || floor_hit) {
      HydPrim1D w;
      w.d  = w_d;
      w.vx = w_ux;
      w.vy = w_uy;
      w.vz = w_uz;
      w.p = w_e*gm1;

      HydCons1D u;
      PrimToConsSingle(g_, gi_, gamma_prime, w_bx, w_by, w_bz, w, u);

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
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = (eos_data.gamma - 1.0);
  Real gamma_prime = eos_data.gamma/(gm1);

  par_for("grmhd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, false, spin, g_, gi_);

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
    PrimToConsSingle(g_, gi_, gamma_prime, bcc1, bcc2, bcc3, w, u);

    // Set conserved quantities
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IEN,k,j,i) = u.e;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;

    // Extract conserved quantities
    Real& u_d    = cons(m,IDN,k,j,i);
    Real& u_t0_0 = cons(m,IEN,k,j,i);
    Real& u_t0_1 = cons(m,IM1,k,j,i);
    Real& u_t0_2 = cons(m,IM2,k,j,i);
    Real& u_t0_3 = cons(m,IM3,k,j,i);

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*u_d;
    }
  });

  return;
}
