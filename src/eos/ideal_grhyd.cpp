//========================================================================================
// Athena++ (Kokkos version) astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grhyd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRHydro::IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//--------------------------------------------------------------------------------------
//! \fn void PrimToConsSingle()
//! \brief Converts single set of primitive into conserved variables.

KOKKOS_INLINE_FUNCTION
void PrimToConsSingle(const Real g_[], const Real gi_[], const Real &gammap,
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
  Real u_0 = g_00*u0 + g_01*u1 + g_02*u2 + g_03*u3;
  Real u_1 = g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3;
  Real u_2 = g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3;
  Real u_3 = g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3;

  Real wgas_u0 = (w.d + gammap * w.p) * u0;
  u.d  = w.d * u0;
  u.e  = wgas_u0 * u_0 + w.p + u.d;  // evolve T^t_t + D
  u.mx = wgas_u0 * u_1;
  u.my = wgas_u0 * u_2;
  u.mz = wgas_u0 * u_3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationC22()
//! \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
//! The ConsToPrim algorithms finds the root of this function f(z)=0

KOKKOS_INLINE_FUNCTION
Real EquationC22(Real z, Real &u_d, Real q, Real r, Real gm1, Real pfloor) {
  Real const w = sqrt(1.0 + z*z);  // (C15)
  Real const wd = u_d/w;  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w);  // (C16)

  //NOTE: The following generalizes to ANY equation of state
  eps = fmax(pfloor/(wd*gm1), eps);  // (C18)
  Real const h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps));  // (C1) & (C21)

  return (z - r/h); // (C22)
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &cc_mask_ = pmy_pack->pcoord->cc_mask;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  Real &dfloor_ = eos_data.dfloor;
  Real &pfloor_ = eos_data.pfloor;

  // Parameters
  int const max_iterations = 25;
  Real const tol = 1.0e-12;
  Real const v_sq_max = 1.0 - tol;

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

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Extract components of metric
    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, g_, gi_);

    // Only execute cons2prim if outside excised region
    bool fixup_hit = false;

    // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
    // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
    // We are also evolving T^t_t + D as conserved variable, so must convert to E
    Real ue_sr = gi_[I00]*(u_e-u_d) + gi_[I01]*u_m1 + gi_[I02]*u_m2 + gi_[I03]*u_m3;

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
      fixup_hit = true;
    }

    // apply energy floor
    if (ue_sr < pfloor_/gm1) {
      ue_sr = pfloor_/gm1;
      sum_e++;
      fixup_hit = true;
    }

    // Recast all variables (eq C2)
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

    Real q = ue_sr/ud_sr;
    Real r = sqrt(um1_sr*m1u + um2_sr*m2u + um3_sr*m3u)/ud_sr;

    // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
    // the value of p will guarantee "some" result of the inversion
    Real kk = fmin(2.* sqrt(v_sq_max)/(1.0 + v_sq_max) - tol, r/(1.+q));

    // Compute bracket (C23)
    auto zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
    auto zp = kk/sqrt(1.0 - kk*kk);

    // Evaluate master function (eq C22) at bracket values
    Real fm = EquationC22(zm, ud_sr, q, r, gm1, pfloor_);
    Real fp = EquationC22(zp, ud_sr, q, r, gm1, pfloor_);

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
      Real f = EquationC22(z, ud_sr, q, r, gm1, pfloor_);

      // Quit if convergence reached
      // NOTE: both z and f are of order unity
      if ((fabs(zm-zp) < tol ) || (fabs(f) < tol )) {
        break;
      }

      if (f*fp < 0.0) {  // assign zm-->zp if root bracketed by [z,zp]
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
    Real const w = sqrt(1.0 + z*z);  // (C15)
    w_d = ud_sr/w;                   // (C15)
    Real eps = w*q - z*r + (z*z)/(1.0 + w);  // (C16)
    Real epsmin = pfloor_/(w_d*gm1);
    if (eps <= epsmin) {                     // C18
      eps = epsmin;
      sum_e++;
      fixup_hit = true;
    }

    Real h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps)); // (C1) & (C21)
    w_e = w_d*eps;

    Real const conv = 1.0/(h*ud_sr); // (C26)
    w_ux = conv*m1u;  // (C26)
    w_uy = conv*m2u;  // (C26)
    w_uz = conv*m3u;  // (C26)

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
    }

    // if excising, handle r_ks < 0.5*(r_inner + r_outer)
    if (excise) {
      if (cc_mask_(m,k,j,i)) {
        w_d = dexcise_;
        w_ux = 0.0;
        w_uy = 0.0;
        w_uz = 0.0;
        w_e = pexcise_/gm1;
        fixup_hit = true;
      }
    }

    // reset conserved variables inside excised regions or if floor is hit
    if (fixup_hit) {
      HydPrim1D w;
      w.d  = w_d;
      w.vx = w_ux;
      w.vy = w_uy;
      w.vz = w_uz;
      w.p = w_e*gm1;

      HydCons1D u;
      PrimToConsSingle(g_, gi_, gamma_prime, w, u);

      cons(m,IDN,k,j,i) = u.d;
      cons(m,IEN,k,j,i) = u.e;
      cons(m,IM1,k,j,i) = u.mx;
      cons(m,IM2,k,j,i) = u.my;
      cons(m,IM3,k,j,i) = u.mz;
      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
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

void IdealGRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);

  par_for("grhyd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
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
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, g_, gi_);

    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.p = prim(m,IEN,k,j,i)*gm1;

    HydCons1D u;
    PrimToConsSingle(g_, gi_, gamma_prime, w, u);

    // Set conserved quantities
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IEN,k,j,i) = u.e;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*u.d;
    }
  });

  return;
}
