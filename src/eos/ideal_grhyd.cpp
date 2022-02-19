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

  // Read flags specifying which variable to use in primitives
  // if nothing set in input file, use e as default
  if (!(pin->DoesParameterExist("hydro","use_e")) &&
      !(pin->DoesParameterExist("hydro","use_t")) ) {
    eos_data.use_e = true;
    eos_data.use_t = false;
  } else {
    eos_data.use_e = pin->GetOrAddBoolean("hydro","use_e",false);
    eos_data.use_t = pin->GetOrAddBoolean("hydro","use_t",false);
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
  u.e  = wgas_u0 * u_0 + w.p - u.d;  // Evolve E-D as in SR
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
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &rmin = pmy_pack->pcoord->coord_data.bh_rmin;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;
  Real gamma_prime = eos_data.gamma/(eos_data.gamma - 1.0);
  Real &pfloor_ = eos_data.pfloor;
  Real &dfloor_ = eos_data.dfloor;
  bool &use_e = eos_data.use_e;

  // Parameters
  int const max_iterations = 25;
  Real const tol = 1.0e-12;
  Real const v_sq_max = 1.0 - tol;

  par_for("grhyd_con2prim", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    Real& u_m1 = cons(m,IM1,k,j,i);
    Real& u_m2 = cons(m,IM2,k,j,i);
    Real& u_m3 = cons(m,IM3,k,j,i);

    Real& w_d  = prim(m,IDN,k,j,i);
    Real& w_ux = prim(m,IVX,k,j,i);
    Real& w_uy = prim(m,IVY,k,j,i);
    Real& w_uz = prim(m,IVZ,k,j,i);

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    bool floor_hit = false;

    // Extract components of metric
    Real g_[NMETRIC], gi_[NMETRIC];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, false, spin, g_, gi_);

    // Only execute cons2prim if outside excised region
    if (rad > rmin) {
      // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
      // alpha^2 T^{tt}.J.  Therefore compute T^{tt} = g^0\mu T^t_\mu
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

      // Recast all variables (eq C2)
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

      Real q = ue_tmp/ud_tmp;
      Real r = sqrt(um1_tmp*m1u + um2_tmp*m2u + um3_tmp*m3u)/ud_tmp;

      // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
      // the value of p will guarantee "some" result of the inversion
      Real kk = fmin(2.* sqrt(v_sq_max)/(1.0 + v_sq_max) - tol, r/(1.+q));

      // Compute bracket (C23)
      auto zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
      auto zp = kk/sqrt(1.0 - kk*kk);

      // Evaluate master function (eq C22) at bracket values
      Real fm = EquationC22(zm, ud_tmp, q, r, gm1, pfloor_);
      Real fp = EquationC22(zp, ud_tmp, q, r, gm1, pfloor_);

      // For simplicity on the GPU, find roots using the false position method
      int iterations = max_iterations;
      // If bracket within tolerances, don't bother doing any iterations
      if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
        iterations = -1;
      }
      Real z = 0.5*(zm + zp);

      for (int ii=0; ii < iterations; ++ii) {
        z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = EquationC22(z, ud_tmp, q, r, gm1, pfloor_);

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

      // iterations ended, compute primitives from resulting value of z
      Real const w = sqrt(1.0 + z*z);  // (C15)
      w_d = ud_tmp/w;  // (C15)
      Real eps = w*q - z*r + (z*z)/(1.0 + w);  // (C16)
      eps = fmax(pfloor_/w_d/gm1, eps);  // (C18)
      Real h = (1.0 + eps) * (1.0 + (gm1*eps)/(1.0+eps)); // (C1) & (C21)
      if (use_e) {
        Real& w_e  = prim(m,IEN,k,j,i);
        w_e = w_d*eps;
      } else {
        Real& w_t  = prim(m,ITM,k,j,i);
        w_t = gm1*eps;  // TODO(@user):  is this the correct expression?
      }

      Real const conv = 1.0/(h*ud_tmp); // (C26)
      w_ux = conv*m1u;  // (C26)
      w_uy = conv*m2u;  // (C26)
      w_uz = conv*m3u;  // (C26)

      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }
    }

    // reset conserved variables inside excised regions or if floor is hit
    Real w_p;
    if (use_e) {
      const Real& w_e  = prim(m,IEN,k,j,i);
      w_p = w_e*gm1;
    } else {
      const Real& w_t  = prim(m,ITM,k,j,i);
      w_p = w_t*gm1*w_d;
    }
    if (rad <= rmin || floor_hit) {
      HydPrim1D w;
      w.d  = w_d;
      w.vx = w_ux;
      w.vy = w_uy;
      w.vz = w_uz;
      w.p  = w_p;

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
  });

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
  bool &use_e = eos_data.use_e;

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
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, false, spin, g_, gi_);

    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    if (use_e) {
      w.p = prim(m,IEN,k,j,i)*gm1;
    } else {
      w.p = prim(m,IEN,k,j,i)*w.d;
    }

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
