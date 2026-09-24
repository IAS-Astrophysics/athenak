#ifndef RADIATION_M1_RADIATION_M1_PHOTON_SOLVER_HPP_
#define RADIATION_M1_RADIATION_M1_PHOTON_SOLVER_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! Optional smooth-branch reorganizations of the coupled gray photon source solve.

#include "radiation_m1/radiation_m1_calc_closure.hpp"
#include "radiation_m1/radiation_m1_photon_opacities.hpp"

namespace radiationm1 {

// An alternative may not silently cross a floor or realizability correction.
KOKKOS_INLINE_FUNCTION
bool photon_unfloored(const SrcParams &s, const Real E,
                     const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F,
                     const RadiationM1Params &p) {
  if (!Kokkos::isfinite(E) || E <= p.rad_E_floor) return false;
  for (int a = 0; a < 4; ++a) {
    if (!Kokkos::isfinite(F(a))) return false;
  }
  const Real f2 = tensor_dot(s.g_uu, F, F);
  return Kokkos::isfinite(f2) && f2 <= E*E*(1.0-p.rad_eps);
}

// T is eliminated using the contraction of the four moment equations with u^a:
//   T = T0 - (gamma-1)/(rho sqrt(gamma)) [Delta E - v^i Delta F_i].
// This identity assumes frozen fluid velocity/metric and inactive radiation floors.
// It is algebraically equivalent to the legacy thermal equation at a moment root;
// away from a root its residual differs by the same contraction of moment residuals.
struct PhotonReducedSystem {
  SrcParams &s;
  const RadiationM1Params &p;
  const Real rho, gm1, vol, t0, arad, thermal_tolerance;
  const bool coupled;

  KOKKOS_INLINE_FUNCTION
  bool Evaluate(const Real x[5], Real r[5], Real &temperature, Real &chi,
                Real &thermal, Real &flux_factor) const {
    s.E = s.Estar*x[0];
    pack_F_d(-s.alp*s.n_u(1), -s.alp*s.n_u(2), -s.alp*s.n_u(3),
             s.Estar*x[1], s.Estar*x[2], s.Estar*x[3], s.F_d);
    if (!photon_unfloored(s, s.E, s.F_d, p)) return false;
    Real heat = s.E-s.Estar;
    for (int a = 1; a < 4; ++a) heat -= s.v_u(a)*(s.F_d(a)-s.Fstar_d(a));
    temperature = t0-gm1*heat/(rho*vol);
    if (!(temperature > 0.0) || !Kokkos::isfinite(temperature)) return false;
    // Tightening the internal closure accuracy makes numerical derivatives smooth
    // enough to certify the original thermal residual; no user tolerance is relaxed.
    auto cp = p;
    cp.closure_epsilon = Kokkos::fmin(p.closure_epsilon,
        0.05*Kokkos::fmin(p.source_epsrel, thermal_tolerance));
    if (coupled) {
      if (!Kokkos::isfinite(x[4]) || x[4] < 0.0 || x[4] > 1.0) return false;
      chi = closure_fun(x[4], p.closure_type);
      apply_closure(s.g_dd, s.g_uu, s.n_d, s.W, s.u_u, s.v_d, s.proj_ud,
                    s.E, s.F_d, chi, s.P_dd, p);
    } else {
      calc_closure(BrentFunctor{}, s.g_dd, s.g_uu, s.n_d, s.W, s.u_u, s.v_d,
                   s.proj_ud, s.E, s.F_d, chi, s.P_dd, cp, p.closure_type);
    }
    assemble_rT(s.n_d, s.E, s.F_d, s.P_dd, s.T_dd);
    s.J = calc_J_from_rT(s.T_dd, s.u_u);
    calc_H_from_rT(s.T_dd, s.u_u, s.proj_ud, s.H_d);
    if (!photon_unfloored(s, s.J, s.H_d, p)) return false;
    const Real h2 = tensor_dot(s.g_uu, s.H_d, s.H_d);
    flux_factor = Kokkos::sqrt(Kokkos::fmax(0.0, h2))/s.J;
    const Real emission = s.kabs*arad*SQR(SQR(temperature))*vol;
    if (!Kokkos::isfinite(emission)) return false;
    calc_rad_sources(emission, s.kabs, s.kscat, s.u_d, s.J, s.H_d, s.S_d);
    s.Edot = calc_rE_source(s.alp, s.n_u, s.S_d);
    calc_rF_source(s.alp, s.gamma_ud, s.S_d, s.tS_d);
    r[0] = (s.E-s.Estar-s.cdt*s.Edot)/s.Estar;
    for (int a = 1; a < 4; ++a) {
      r[a] = (s.F_d(a)-s.Fstar_d(a)-s.cdt*s.tS_d(a))/s.Estar;
    }
    r[4] = coupled ? x[4]-flux_factor : 0.0;
    thermal = temperature-t0 + s.cdt*s.alp/s.W*s.kabs*gm1/rho*
        (arad*SQR(SQR(temperature))-s.J/vol);
    for (int a = 0; a < 5; ++a) {
      if (!Kokkos::isfinite(r[a])) return false;
    }
    return Kokkos::isfinite(thermal);
  }

  // Derivative of Evaluate's normalized residual at the CURRENT evaluated state.
  // Fluid/metric fields and opacity coefficients are fixed in this inner solve.
  // No floors or clipping are differentiated: Evaluate rejects those states.
  KOKKOS_INLINE_FUNCTION
  bool AnalyticJacobian(const Real x[5], const Real temperature,
                        Real jac[5][5]) const {
    if (!coupled || p.closure_type != Minerbo) return false;
    const Real f2 = tensor_dot(s.g_uu, s.F_d, s.F_d);
    const Real h2 = tensor_dot(s.g_uu, s.H_d, s.H_d);
    // Pthin has no unique directional derivative at F=0; |H| is not
    // differentiable at H=0. Use the established numerical path nearby.
    if (!(f2 > 1.e-24*s.E*s.E) || !(h2 > 1.e-24*s.J*s.J)) return false;
    const Real hnorm = Kokkos::sqrt(h2);
    const Real xi = x[4];
    const Real chi = closure_fun(xi, Minerbo);
    const Real thick = 1.5*(1.0-chi), thin = 1.0-thick;
    const Real dthin = 1.5*(12.0*xi-6.0*xi*xi+24.0*xi*xi*xi)/15.0;
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> pthin{}, pthick{};
    calc_Pthin(s.g_uu, s.E, s.F_d, pthin);
    calc_Pthick(s.g_dd, s.g_uu, s.n_d, s.W, s.v_d, s.E, s.F_d, pthick);
    for (int col = 0; col < 5; ++col) {
      const Real de = col == 0 ? s.Estar : 0.0;
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> df{}, dh{}, ds{};
      pack_F_d(-s.alp*s.n_u(1), -s.alp*s.n_u(2), -s.alp*s.n_u(3),
               col == 1 ? s.Estar : 0.0, col == 2 ? s.Estar : 0.0,
               col == 3 ? s.Estar : 0.0, df);
      AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> dp{}, dstress{};
      if (col == 4) {
        for (int a = 0; a < 4; ++a) {
          for (int b = a; b < 4; ++b) {
            dp(a,b) = dthin*(pthin(a,b)-pthick(a,b));
          }
        }
      } else {
        // Pthick is linear in E,F at fixed metric and velocity.
        calc_Pthick(s.g_dd, s.g_uu, s.n_d, s.W, s.v_d, de, df, dp);
        const Real df2 = 2.0*tensor_dot(s.g_uu, s.F_d, df);
        for (int a = 0; a < 4; ++a) {
          for (int b = a; b < 4; ++b) {
            const Real dpt = (de/f2)*s.F_d(a)*s.F_d(b)
                + (s.E/f2)*(df(a)*s.F_d(b)+s.F_d(a)*df(b))
                - pthin(a,b)*(df2/f2);
            dp(a,b) = thick*dp(a,b)+thin*dpt;
          }
        }
      }
      assemble_rT(s.n_d, de, df, dp, dstress);
      const Real dj = calc_J_from_rT(dstress, s.u_u);
      calc_H_from_rT(dstress, s.u_u, s.proj_ud, dh);
      Real dt = de;
      for (int a = 1; a < 4; ++a) dt -= s.v_u(a)*df(a);
      dt *= -gm1/(rho*vol);
      const Real demission = 4.0*s.kabs*arad*vol*
                             temperature*temperature*temperature*dt;
      // The four-force is linear in emission, J and H at fixed opacity/u.
      calc_rad_sources(demission, s.kabs, s.kscat, s.u_d, dj, dh, ds);
      jac[0][col] = (col == 0 ? 1.0 : 0.0)
          - s.cdt*calc_rE_source(s.alp, s.n_u, ds)/s.Estar;
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> dfdot{};
      calc_rF_source(s.alp, s.gamma_ud, ds, dfdot);
      for (int a = 1; a < 4; ++a) {
        jac[a][col] = (a == col ? 1.0 : 0.0)-s.cdt*dfdot(a)/s.Estar;
      }
      jac[4][col] = (col == 4 ? 1.0 : 0.0)
          - tensor_dot(s.g_uu, s.H_d, dh)/(s.J*hnorm)
          + (hnorm/s.J)*(dj/s.J);
      for (int a = 0; a < 5; ++a) {
        if (!Kokkos::isfinite(jac[a][col])) return false;
      }
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  Real Norm(const Real x[5], const Real r[5], Real t, Real thermal) const {
    Real norm = 0.0;
    // Require residuals, not merely small Newton steps, including the ORIGINAL
    // thermal equation (the algebraic elimination alone is not a convergence test).
    for (int a = 0; a < 4; ++a) {
      const Real tol = p.source_epsabs/s.Estar + p.source_epsrel*Kokkos::abs(x[a]);
      norm = Kokkos::fmax(norm, Kokkos::abs(r[a])/tol);
    }
    const Real ttol = thermal_tolerance*Kokkos::fmax(
        Kokkos::fmax(t, t0), 1.0e-100);
    norm = Kokkos::fmax(norm, Kokkos::abs(thermal)/ttol);
    if (coupled) {
      norm = Kokkos::fmax(norm, Kokkos::abs(r[4])/
          Kokkos::fmin(p.closure_epsilon, thermal_tolerance));
    }
    return norm;
  }
};

// Partial-pivot elimination of a dimensionless 4x4 or 5x5 Jacobian.
KOKKOS_INLINE_FUNCTION
bool photon_linear_solve(Real a[5][5], Real b[5], const int n) {
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k+1; i < n; ++i) {
      if (Kokkos::abs(a[i][k]) > Kokkos::abs(a[pivot][k])) pivot = i;
    }
    if (!Kokkos::isfinite(a[pivot][k]) || Kokkos::abs(a[pivot][k]) < 1.0e-30) {
      return false;
    }
    if (pivot != k) {
      for (int j = k; j < n; ++j) {
        const Real tmp = a[k][j]; a[k][j] = a[pivot][j]; a[pivot][j] = tmp;
      }
      const Real tmp = b[k]; b[k] = b[pivot]; b[pivot] = tmp;
    }
    for (int i = k+1; i < n; ++i) {
      const Real factor = a[i][k]/a[k][k];
      for (int j = k+1; j < n; ++j) a[i][j] -= factor*a[k][j];
      b[i] -= factor*b[k];
    }
  }
  for (int i = n-1; i >= 0; --i) {
    for (int j = i+1; j < n; ++j) b[i] -= a[i][j]*b[j];
    b[i] /= a[i][i];
    if (!Kokkos::isfinite(b[i])) return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool photon_reduced_solve(SrcParams &s, const RadiationM1Params &p,
                          const PhotonOpacityParams &photon, const Real rho,
                          const Real gm1, const Real vol, const Real t0,
                          Real &E, AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F,
                          Real &chi, Real &temperature, int &iterations) {
  // Eddington has a fixed closure, hence both alternatives use four unknowns.
  if ((p.closure_type != Minerbo && p.closure_type != Eddington) ||
      !photon_unfloored(s, s.Estar, s.Fstar_d, p) || !(rho > 0.0) ||
      !(gm1 > 0.0) || !(vol > 0.0) || !(t0 > 0.0) || !(s.kabs >= 0.0) ||
      !(s.kabs+s.kscat >= 0.0)) return false;
  const bool coupled = p.photon_source_solver == PhotonCoupled &&
                       p.closure_type == Minerbo;
  PhotonReducedSystem system{s, p, rho, gm1, vol, t0, photon.arad,
                             photon.source_tolerance, coupled};
  const int n = coupled ? 5 : 4;
  Real x[5] = {E/s.Estar, F(1)/s.Estar, F(2)/s.Estar, F(3)/s.Estar, 0.5};
  // Seed xi from the existing physical closure, not from another root branch.
  auto initialize = [&]() {
    PhotonReducedSystem reduced{s, p, rho, gm1, vol, t0, photon.arad,
                                photon.source_tolerance, false};
    Real r[5], thermal, flux_factor, c;
    if (!reduced.Evaluate(x, r, temperature, c, thermal, flux_factor)) return false;
    x[4] = flux_factor;
    return true;
  };
  if (!initialize()) {
    x[0] = 1.0;
    for (int a = 1; a < 4; ++a) x[a] = s.Fstar_d(a)/s.Estar;
    if (!initialize()) return false;
  }
  Real r[5], thermal, flux_factor;
  for (int iter = 0; iter < p.source_maxiter; ++iter) {
    ++iterations;
    if (!system.Evaluate(x, r, temperature, chi, thermal, flux_factor)) return false;
    const Real norm = system.Norm(x, r, temperature, thermal);
    if (norm <= 1.0) {
      // Check the physical closure branch with the established bracketed solve.
      // Re-evaluate the ORIGINAL four source and thermal equations at that closure.
      PhotonReducedSystem reference{s, p, rho, gm1, vol, t0, photon.arad,
                                    photon.source_tolerance, false};
      Real rr[5], tr, cr, hr, xr;
      if (!reference.Evaluate(x, rr, tr, cr, hr, xr) ||
          Kokkos::abs(cr-chi) > 2.0*p.closure_epsilon ||
          reference.Norm(x, rr, tr, hr) > 1.0) return false;
      E = s.E;
      F = s.F_d;
      temperature = tr;
      chi = cr;
      return true;
    }
    Real jac[5][5] = {}, step[5] = {};
    const bool analytic = p.photon_analytic_jacobian &&
                          system.AnalyticJacobian(x, temperature, jac);
    for (int column = 0; !analytic && column < n; ++column) {
      const Real h = 1.0e-5*Kokkos::fmax(1.0, Kokkos::abs(x[column]));
      Real xp[5], xm[5], rp[5], rm[5], tp, cp, hp, fp;
      for (int a = 0; a < 5; ++a) xp[a] = xm[a] = x[a];
      xp[column] += h;
      xm[column] -= h;
      const bool plus = system.Evaluate(xp, rp, tp, cp, hp, fp);
      const bool minus = system.Evaluate(xm, rm, tp, cp, hp, fp);
      if (!plus && !minus) return false;
      for (int row = 0; row < n; ++row) {
        jac[row][column] = plus && minus ? (rp[row]-rm[row])/(2.0*h) :
            (plus ? (rp[row]-r[row])/h : (r[row]-rm[row])/h);
      }
    }
    for (int a = 0; a < n; ++a) step[a] = -r[a];
    if (!photon_linear_solve(jac, step, n)) return false;
    bool accepted = false;
    Real length = 1.0;
    for (int ls = 0; ls < 24; ++ls, length *= 0.5) {
      Real trial[5], rt[5], tt, ct, ht, ft;
      for (int a = 0; a < 5; ++a) trial[a] = x[a]+length*step[a];
      if (system.Evaluate(trial, rt, tt, ct, ht, ft) &&
          system.Norm(trial, rt, tt, ht) < norm*(1.0-1.0e-4*length)) {
        for (int a = 0; a < 5; ++a) x[a] = trial[a];
        accepted = true;
        break;
      }
    }
    if (!accepted) return false;
  }
  return false;
}

}  // namespace radiationm1
#endif  // RADIATION_M1_RADIATION_M1_PHOTON_SOLVER_HPP_
