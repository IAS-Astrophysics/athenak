//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_core.cpp
//  \brief Unit tests for the pointwise kernels of the grey M1 solver: closure function,
//         lab/fluid-frame projections, closure rootfinders, floors, matter sources,
//         the implicit source solve, and the small dense linear algebra behind it.
//
//  Sections 1-9 call KOKKOS_INLINE_FUNCTIONs that act on a single point directly on the
//  host, with hand-built metrics and fluid states.  Section 10 drives the real flux
//  kernel, RadiationM1::CalculateFluxes, on the tiny 1D mesh of the input file: the
//  pgen writes a chosen metric and radiation state into the module's arrays (ghost
//  zones included), calls the kernel once and compares the face fluxes with values
//  derived by hand.  No EOS table, opacity library or time step is involved.
//
//  ## What is checked
//
//   1. Closure function: Minerbo/Eddington/Thin limits, dclosure_dxi.
//   2. Lab-frame pressure: thin limit P = E F F / F^2, thick limit P = J/3 (g + n n) at
//      rest, apply_closure interpolates linearly in chi.
//   3. Frames: with the fluid at rest J = E and H = F; for a boosted fluid
//      J = W^2 (E - 2 v.F + v v P) and H.u = 0.
//   4. Rootfinders: Brent on x^2 - 2; the Brent and Newton closure solvers return a chi
//      that is self-consistent (chi = closure_fun(|H|/J)) and agree with each other.
//   5. Floors: E is raised to the floor, causal F is untouched, acausal F becomes causal.
//   6. Sources at rest: Edot = alpha (eta - kabs E), Fdot = -alpha (kabs + kscat) F.
//   7. source_update_ll: SrcThin is the explicit update from the old state, an
//      intermediate optical depth gives backward Euler at rest and a zero residual
//      q - q* - cdt S[q] for a boosted fluid, very large optical depth returns SrcEquil.
//   8. Linear algebra: QR = A, Q^T Q = 1, qr_R_solve; Hybridsj on a cubic system.
//   9. ADM geometry, as assembled in RadiationM1::TimeUpdate for a general lapse, shift
//      and non-diagonal 3-metric: g_ab g^bc = delta, n.n = -1, gamma^ij from g^ij,
//      u.u = -1 and u = W (n + v), F^a n_a = 0, volume form.
//  10. Fluxes (CalculateFluxes): a uniform state in a curved, shifted metric gives the
//      same flux on every face (free-stream preservation) equal to
//      alpha gamma^xj F_j - beta^x E; for a fluid at rest in that metric the F_j flux
//      is alpha P^x_j - beta^x F_j for the thin and thick closures, and the N flux
//      alpha N F^x / E - beta^x N; a free-streaming step is upwinded in both
//      directions; the Rusanov speed is alpha sqrt(gamma^xx) + |beta^x| for either
//      sign of the shift; smooth data get the unlimited central flux and a sawtooth
//      the Rusanov flux; the numerical diffusion is reduced by 1/(kappa dx) in
//      optically thick cells, kappa = absorption + scattering.
//
//  Checks tagged [KNOWN] document defects found in the pre-merge audit (2026-11).  They
//  are printed but do not fail the run unless <problem>/strict_known = true; once a
//  defect is fixed the corresponding check should be promoted to a regular one.
//
//  Run via tst/inputs/ut_rad_m1_core.athinput; exits non-zero on failure, which is what
//  tst/test_suite/unit_tests/test_rad_m1_core_cpu.py checks.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_calc_closure.hpp"
#include "radiation_m1/radiation_m1_closure.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_linalg.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_roots_brent.hpp"
#include "radiation_m1/radiation_m1_roots_hybridsj.hpp"
#include "radiation_m1/radiation_m1_sources.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"

namespace {

using namespace radiationm1;  // NOLINT(build/namespaces)

using Vec4 = AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>;
using Sym4 = AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>;
using Mix4 = AthenaPointTensor<Real, TensorSymm::NONE, 4, 2>;

int nfail = 0, nrun = 0, nknown = 0;
bool strict_known = false;

void Report(const std::string &what, bool ok, bool known, const std::string &detail) {
  ++nrun;
  const char *tag = ok ? "  [ ok ] " : (known ? "  [KNOWN] " : "  [FAIL] ");
  if (!ok) {
    if (known) {
      ++nknown;
      if (strict_known) ++nfail;
    } else {
      ++nfail;
    }
  }
  std::cout << tag << std::left << std::setw(58) << what << std::right << detail
            << std::endl;
}

void CheckClose(const std::string &what, Real got, Real want, Real rtol,
                Real atol = 0.0, bool known = false) {
  bool ok = std::isfinite(got) &&
            std::abs(got - want) <= atol + rtol * std::abs(want);
  std::ostringstream s;
  s << std::scientific << std::setprecision(10) << " got " << got << "  want " << want;
  Report(what, ok, known, s.str());
}

void CheckTrue(const std::string &what, bool ok, bool known = false) {
  Report(what, ok, known, "");
}

//! A point of spacetime: flat spatial metric, lapse alpha, zero shift, and a fluid
//! moving with 3-velocity v along x.  Mirrors the setup in RadiationM1::TimeUpdate.
struct Point {
  Real alpha, W;
  Sym4 g_dd{}, g_uu{};
  Vec4 n_d{}, n_u{}, u_d{}, u_u{}, v_d{}, v_u{};
  Mix4 gamma_ud{}, proj_ud{};

  Point(Real alpha_, Real vx) : alpha(alpha_) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        g_dd(a, b) = (a == b) ? 1.0 : 0.0;
        g_uu(a, b) = (a == b) ? 1.0 : 0.0;
      }
    }
    g_dd(0, 0) = -alpha * alpha;
    g_uu(0, 0) = -1.0 / (alpha * alpha);
    pack_n_d(alpha, n_d);
    tensor_contract(g_uu, n_d, n_u);
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        gamma_ud(a, b) = (a == b) + n_u(a) * n_d(b);
      }
    }
    W = get_w_lorentz(vx / std::sqrt(1.0 - vx * vx), 0.0, 0.0, g_dd);
    pack_u_u(W / alpha, W * vx, 0.0, 0.0, u_u);
    pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), alpha, 0.0, 0.0, 0.0, v_u);
    tensor_contract(g_dd, u_u, u_d);
    tensor_contract(g_dd, v_u, v_d);
    calc_proj(u_d, u_u, proj_ud);
  }
};

RadiationM1Params DefaultParams() {
  RadiationM1Params p{};
  p.closure_type = Minerbo;
  p.closure_solver = ClosureBrent;
  p.closure_newton_bracket = true;
  p.closure_epsilon = 1e-12;   // tighter than the runtime default (1e-5) so the
  p.closure_maxiter = 200;     // checks below measure the algebra, not the tolerance
  p.rad_E_floor = 1e-30;
  p.rad_N_floor = 1e-30;
  p.rad_eps = 1e-14;
  p.source_epsabs = 1e-15;
  p.source_epsrel = 1e-12;
  p.source_maxiter = 200;
  p.source_thin_limit = 1.0 / 3.0;
  p.source_thick_limit = 20.0;
  p.source_scat_limit = -1.0;
  return p;
}

//! J and |H| of a lab-frame state (E, F_d) closed with the given chi.
void FluidMoments(const Point &pt, Real E, const Vec4 &F_d, Real chi,
                  const RadiationM1Params &p, Real &J, Real &Hnorm, Real &Hdotu) {
  Sym4 P_dd{}, T_dd{};
  apply_closure(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d, pt.proj_ud, E, F_d, chi,
                P_dd, p);
  assemble_rT(pt.n_d, E, F_d, P_dd, T_dd);
  J = calc_J_from_rT(T_dd, pt.u_u);
  Vec4 H_d{};
  calc_H_from_rT(T_dd, pt.u_u, pt.proj_ud, H_d);
  Hnorm = std::sqrt(std::max(0.0, tensor_dot(pt.g_uu, H_d, H_d)));
  Hdotu = tensor_dot(H_d, pt.u_u);
}

//! Residual of the implicit source problem, |q - q* - cdt S[q]| / E*, evaluated with
//! the given closure.  Zero at the solution of source_update_ll.
Real SourceResidual(const Point &pt, Real cdt, Real Estar, const Vec4 &Fstar_d, Real eta,
                    Real kabs, Real kscat, Real Enew, const Vec4 &Fnew_d,
                    const RadiationM1Params &p, RadiationM1Closure closure) {
  SrcParams sp(cdt, pt.alpha, pt.g_dd, pt.g_uu, pt.n_d, pt.n_u, pt.gamma_ud, pt.u_d,
               pt.u_u, pt.v_d, pt.v_u, pt.proj_ud, pt.W, Estar, Fstar_d, 0.0, eta, kabs,
               kscat);
  Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
  prepare(BrentFunctor{}, q, sp, p, closure);
  Real r2 = 0.0;
  Real f[4] = {q[0] - Estar - cdt * sp.Edot,
               q[1] - Fstar_d(1) - cdt * sp.tS_d(1),
               q[2] - Fstar_d(2) - cdt * sp.tS_d(2),
               q[3] - Fstar_d(3) - cdt * sp.tS_d(3)};
  for (int i = 0; i < 4; ++i) r2 += f[i] * f[i];
  return std::sqrt(r2) / Estar;
}

//------------------------------------------------------------------------------------
void TestClosureFunction() {
  std::cout << "-- 1. closure function" << std::endl;
  CheckClose("Minerbo chi(0) = 1/3", closure_fun(0.0, Minerbo), 1.0 / 3.0, 1e-15);
  CheckClose("Minerbo chi(1) = 1", closure_fun(1.0, Minerbo), 1.0, 1e-15);
  CheckClose("Minerbo chi(0.5)", closure_fun(0.5, Minerbo), 1.0 / 3.0 + 0.25 * 6.5 / 15.0,
             1e-15);
  CheckClose("Eddington chi = 1/3", closure_fun(0.7, Eddington), 1.0 / 3.0, 1e-15);
  CheckClose("Thin chi = 1", closure_fun(0.2, Thin), 1.0, 1e-15);

  bool monotone = true, bounded = true;
  Real prev = closure_fun(0.0, Minerbo);
  Real maxerr = 0.0;
  const Real h = 1e-6;
  for (int n = 1; n <= 100; ++n) {
    const Real xi = 0.01 * n;
    const Real c = closure_fun(xi, Minerbo);
    monotone = monotone && (c >= prev);
    bounded = bounded && (c >= 1.0 / 3.0) && (c <= 1.0);
    prev = c;
    if (xi < 1.0) {
      const Real fd = (closure_fun(xi + h, Minerbo) - closure_fun(xi - h, Minerbo)) /
                      (2 * h);
      maxerr = std::max(maxerr, std::abs(fd - dclosure_dxi(xi)));
    }
  }
  CheckTrue("Minerbo chi monotone on [0,1]", monotone);
  CheckTrue("Minerbo chi in [1/3, 1] on [0,1]", bounded);
  CheckClose("dclosure_dxi matches finite difference (max err)", maxerr, 0.0, 0.0, 1e-8);
}

//------------------------------------------------------------------------------------
void TestPressure() {
  std::cout << "-- 2. lab-frame radiation pressure" << std::endl;
  const RadiationM1Params p = DefaultParams();
  Point pt(1.0, 0.0);
  const Real E = 2.0;
  Vec4 F_d{};
  pack_F_d(0.0, 0.0, 0.0, 0.6 * E, 0.0, 0.0, F_d);

  Sym4 Pthin{}, Pthick{}, Pmix{};
  calc_Pthin(pt.g_uu, E, F_d, Pthin);
  CheckClose("thin: P_xx = E for F along x", Pthin(1, 1), E, 1e-15);
  CheckClose("thin: P_yy = 0", Pthin(2, 2), 0.0, 0.0, 1e-15);

  calc_Pthick(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.v_d, E, F_d, Pthick);
  CheckClose("thick at rest: P_xx = E/3", Pthick(1, 1), E / 3.0, 1e-15);
  CheckClose("thick at rest: P_zz = E/3", Pthick(3, 3), E / 3.0, 1e-15);
  CheckClose("thick at rest: P_xy = 0", Pthick(1, 2), 0.0, 0.0, 1e-15);
  CheckClose("thick at rest: P_00 = 0", Pthick(0, 0), 0.0, 0.0, 1e-15);

  apply_closure(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d, pt.proj_ud, E, F_d,
                1.0 / 3.0, Pmix, p);
  CheckClose("apply_closure(chi=1/3) = thick", Pmix(1, 1), Pthick(1, 1), 1e-14);
  apply_closure(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d, pt.proj_ud, E, F_d, 1.0,
                Pmix, p);
  CheckClose("apply_closure(chi=1) = thin", Pmix(1, 1), Pthin(1, 1), 1e-14);
  const Real chi = 0.6;
  const Real dthick = 1.5 * (1.0 - chi);
  apply_closure(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d, pt.proj_ud, E, F_d, chi,
                Pmix, p);
  CheckClose("apply_closure interpolates linearly in chi", Pmix(1, 1),
             dthick * Pthick(1, 1) + (1.0 - dthick) * Pthin(1, 1), 1e-14);
}

//------------------------------------------------------------------------------------
void TestFrames() {
  std::cout << "-- 3. lab <-> fluid frame" << std::endl;
  const RadiationM1Params p = DefaultParams();
  const Real E = 1.5;
  Vec4 F_d{};
  pack_F_d(0.0, 0.0, 0.0, 0.4, -0.3, 0.2, F_d);

  {
    Point pt(1.3, 0.0);
    Sym4 P_dd{}, T_dd{};
    apply_closure(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d, pt.proj_ud, E, F_d, 0.5,
                  P_dd, p);
    assemble_rT(pt.n_d, E, F_d, P_dd, T_dd);
    Vec4 H_d{};
    calc_H_from_rT(T_dd, pt.u_u, pt.proj_ud, H_d);
    CheckClose("at rest: J = E", calc_J_from_rT(T_dd, pt.u_u), E, 1e-14);
    CheckClose("at rest: H_x = F_x", H_d(1), F_d(1), 1e-14);
    CheckClose("at rest: H_y = F_y", H_d(2), F_d(2), 1e-14);
    CheckClose("at rest: H_0 = 0", H_d(0), 0.0, 0.0, 1e-15);
  }
  {
    const Real v = 0.3;
    Point pt(1.0, v);
    CheckClose("boost: u.u = -1", tensor_dot(pt.u_d, pt.u_u), -1.0, 1e-14);
    Sym4 P_dd{}, T_dd{};
    apply_closure(pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d, pt.proj_ud, E, F_d, 0.5,
                  P_dd, p);
    assemble_rT(pt.n_d, E, F_d, P_dd, T_dd);
    const Real J = calc_J_from_rT(T_dd, pt.u_u);
    const Real Jan = pt.W * pt.W * (E - 2.0 * v * F_d(1) + v * v * P_dd(1, 1));
    CheckClose("boost: J = W^2 (E - 2 v.F + v v P)", J, Jan, 1e-13);
    Vec4 H_d{};
    calc_H_from_rT(T_dd, pt.u_u, pt.proj_ud, H_d);
    CheckClose("boost: H.u = 0", tensor_dot(H_d, pt.u_u), 0.0, 0.0, 1e-14);
  }
}

//------------------------------------------------------------------------------------
void TestRootfinders() {
  std::cout << "-- 4. rootfinders and closure solve" << std::endl;
  {
    auto f = [](Real x) { return x * x - 2.0; };
    Real lo = 0.0, hi = 2.0, root = 0.0;
    BrentState st{};
    MathSignal ierr = BrentInitialize(f, lo, hi, root, st);
    int iter = 0;
    do {
      ++iter;
      BrentIterate(f, lo, hi, root, st);
      ierr = BrentTestInterval(lo, hi, 1e-14, 0.0);
    } while (ierr == LinalgContinue && iter < 100);
    CheckClose("Brent: root of x^2 - 2 on [0,2]", root, std::sqrt(2.0), 1e-12);
    Real lo2 = 2.0, hi2 = 3.0, root2 = 0.0;
    BrentState st2{};
    CheckTrue("Brent: same-sign bracket -> LinalgEinval",
              BrentInitialize(f, lo2, hi2, root2, st2) == LinalgEinval);
  }

  struct Case { const char *name; Real alpha, v, E, Fx, Fy; };
  const Case cases[] = {
      {"rest, f=0.5", 1.0, 0.0, 1.0, 0.5, 0.0},
      {"rest, f=0.95", 1.0, 0.0, 1.0, 0.0, 0.95},
      {"boost v=0.3, f=0.6", 1.0, 0.3, 1.0, 0.6, 0.0},
      {"boost v=0.5, f=0.4 oblique", 1.2, 0.5, 2.0, -0.5, 0.6},
  };
  for (const Case &c : cases) {
    Point pt(c.alpha, c.v);
    Vec4 F_d{};
    pack_F_d(0.0, 0.0, 0.0, c.Fx, c.Fy, 0.0, F_d);
    Real chi_solver[2]{};
    for (int s = 0; s < 2; ++s) {
      RadiationM1Params p = DefaultParams();
      p.closure_solver = (s == 0) ? ClosureBrent : ClosureNewton;
      Real chi = -1.0;
      Sym4 P_dd{};
      calc_closure(BrentFunctor{}, pt.g_dd, pt.g_uu, pt.n_d, pt.W, pt.u_u, pt.v_d,
                   pt.proj_ud, c.E, F_d, chi, P_dd, p, Minerbo);
      chi_solver[s] = chi;
      Real J, Hn, Hu;
      FluidMoments(pt, c.E, F_d, chi, p, J, Hn, Hu);
      const std::string solver = (s == 0) ? "Brent " : "Newton";
      CheckClose(solver + " closure self-consistent, " + c.name, chi,
                 closure_fun(Hn / J, Minerbo), 1e-8);
    }
    CheckClose(std::string("Brent and Newton agree, ") + c.name, chi_solver[1],
               chi_solver[0], 1e-8);
  }
}

//------------------------------------------------------------------------------------
void TestFloors() {
  std::cout << "-- 5. floors" << std::endl;
  const RadiationM1Params p = DefaultParams();
  Point pt(1.0, 0.0);
  {
    Real E = -1.0;
    Vec4 F_d{};
    apply_floor(pt.g_uu, E, F_d, p);
    CheckClose("negative E raised to rad_E_floor", E, p.rad_E_floor, 1e-15);
  }
  {
    Real E = 1.0;
    Vec4 F_d{};
    pack_F_d(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, F_d);
    apply_floor(pt.g_uu, E, F_d, p);
    CheckClose("causal F untouched", F_d(1), 0.5, 1e-15);
  }
  {
    // a large rad_eps, so that the margin below |F| = E is resolved
    RadiationM1Params pe = p;
    pe.rad_eps = 1e-2;
    Real E = 1.0;
    Vec4 F_d{};
    pack_F_d(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, F_d);
    apply_floor(pt.g_uu, E, F_d, pe);
    const Real F2 = tensor_dot(pt.g_uu, F_d, F_d);
    CheckTrue("acausal |F| = 2E: F^2 <= (1 - rad_eps) E^2",
              F2 <= (1.0 - pe.rad_eps) * E * E * (1.0 + 1e-14));
    CheckClose("acausal |F| = 2E rescaled to sqrt(1 - rad_eps) E", std::sqrt(F2),
               std::sqrt(0.99), 1e-14);
  }
  {
    Real E = 1.0;
    Vec4 F_d{};
    pack_F_d(0.0, 0.0, 0.0, std::numeric_limits<Real>::quiet_NaN(), 0.0, 0.0, F_d);
    apply_floor(pt.g_uu, E, F_d, p);
    CheckTrue("NaN flux: every F_a set to 0 and E unchanged",
              F_d(0) == 0.0 && F_d(1) == 0.0 && F_d(2) == 0.0 && F_d(3) == 0.0 &&
                  E == 1.0);
  }
}

//------------------------------------------------------------------------------------
void TestSources() {
  std::cout << "-- 6. matter sources at rest" << std::endl;
  const Real alpha = 1.5, eta = 0.7, kabs = 2.0, kscat = 3.0, E = 0.4;
  Point pt(alpha, 0.0);
  Vec4 H_d{}, S_d{}, tS_d{};
  pack_F_d(0.0, 0.0, 0.0, 0.1, -0.2, 0.05, H_d);
  calc_rad_sources(eta, kabs, kscat, pt.u_d, E, H_d, S_d);
  CheckClose("Edot = alpha (eta - kabs J)", calc_rE_source(alpha, pt.n_u, S_d),
             alpha * (eta - kabs * E), 1e-14);
  calc_rF_source(alpha, pt.gamma_ud, S_d, tS_d);
  CheckClose("Fdot_x = -alpha (kabs + kscat) F_x", tS_d(1),
             -alpha * (kabs + kscat) * H_d(1), 1e-14);
  CheckClose("Fdot_z = -alpha (kabs + kscat) F_z", tS_d(3),
             -alpha * (kabs + kscat) * H_d(3), 1e-14);
}

//------------------------------------------------------------------------------------
void TestSourceUpdate() {
  std::cout << "-- 7. source_update_ll" << std::endl;
  const RadiationM1Params p = DefaultParams();
  const Real eta = 0.8, kscat = 0.5;

  // thin regime: explicit update evaluated at the old state
  {
    const Real alpha = 1.2, cdt = 0.01, kabs = 1.0;
    Point pt(alpha, 0.0);
    const Real Eold = 0.3, Estar = 0.35;
    Vec4 Fold{}, Fstar{}, Fnew{};
    pack_F_d(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, Fold);
    pack_F_d(0.0, 0.0, 0.0, 0.12, 0.0, 0.0, Fstar);
    Real chi = 1.0 / 3.0, Enew = Estar;
    SrcSignal sig = source_update_ll(BrentFunctor{}, HybridsjFunctor{}, cdt, alpha,
                                     pt.g_dd, pt.g_uu, pt.n_d, pt.n_u, pt.gamma_ud,
                                     pt.u_d, pt.u_u, pt.v_d, pt.v_u, pt.proj_ud, pt.W,
                                     Eold, Fold, Estar, Fstar, eta, kabs, kscat, chi,
                                     Enew, Fnew, p, Minerbo);
    CheckTrue("cdt*kappa < 1/3 returns SrcThin", sig == SrcThin);
    CheckClose("thin: Enew = E* + cdt alpha (eta - kabs Eold)", Enew,
               Estar + cdt * alpha * (eta - kabs * Eold), 1e-13);
    CheckClose("thin: Fnew = F* - cdt alpha (kabs + kscat) Fold", Fnew(1),
               Fstar(1) - cdt * alpha * (kabs + kscat) * Fold(1), 1e-13);
  }

  // intermediate regime at rest: exact backward Euler
  {
    const Real alpha = 1.2, cdt = 1.0, kabs = 2.0;
    Point pt(alpha, 0.0);
    const Real Estar = 0.35;
    Vec4 Fstar{}, Fnew{};
    pack_F_d(0.0, 0.0, 0.0, 0.12, -0.05, 0.0, Fstar);
    Real chi = 1.0 / 3.0, Enew = Estar;
    Fnew = Fstar;
    SrcSignal sig = source_update_ll(BrentFunctor{}, HybridsjFunctor{}, cdt, alpha,
                                     pt.g_dd, pt.g_uu, pt.n_d, pt.n_u, pt.gamma_ud,
                                     pt.u_d, pt.u_u, pt.v_d, pt.v_u, pt.proj_ud, pt.W,
                                     Estar, Fstar, Estar, Fstar, eta, kabs, kscat, chi,
                                     Enew, Fnew, p, Minerbo);
    CheckTrue("implicit at rest returns SrcOk", sig == SrcOk);
    CheckClose("implicit at rest: E = (E* + cdt a eta)/(1 + cdt a kabs)", Enew,
               (Estar + cdt * alpha * eta) / (1.0 + cdt * alpha * kabs), 1e-10);
    CheckClose("implicit at rest: F = F*/(1 + cdt a (kabs + kscat))", Fnew(1),
               Fstar(1) / (1.0 + cdt * alpha * (kabs + kscat)), 1e-10);
  }

  // intermediate regime, boosted fluid: residual of the nonlinear system vanishes
  {
    const Real alpha = 1.0, cdt = 1.0, kabs = 2.0;
    Point pt(alpha, 0.3);
    const Real Estar = 1.0;
    Vec4 Fstar{}, Fnew{};
    pack_F_d(0.0, 0.0, 0.0, 0.85, 0.0, 0.0, Fstar);
    for (int s = 0; s < 2; ++s) {
      const RadiationM1Closure cl = (s == 0) ? Minerbo : Eddington;
      Real chi = 1.0 / 3.0, Enew = Estar;
      Fnew = Fstar;
      SrcSignal sig = source_update_ll(BrentFunctor{}, HybridsjFunctor{}, cdt, alpha,
                                       pt.g_dd, pt.g_uu, pt.n_d, pt.n_u, pt.gamma_ud,
                                       pt.u_d, pt.u_u, pt.v_d, pt.v_u, pt.proj_ud, pt.W,
                                       Estar, Fstar, Estar, Fstar, eta, kabs, kscat, chi,
                                       Enew, Fnew, p, cl);
      const Real res = SourceResidual(pt, cdt, Estar, Fstar, eta, kabs, kscat, Enew, Fnew,
                                      p, cl);
      if (s == 0) {
        CheckTrue("implicit boosted (Minerbo) returns SrcOk", sig == SrcOk);
        CheckClose("implicit boosted (Minerbo): residual", res, 0.0, 0.0, 1e-9);
      } else {
        // The Eddington retry in source_update relies on this: the solver must honour
        // the closure it is passed, not m1_params.closure_type.
        CheckClose("implicit boosted solved with requested closure (Eddington)", res,
                   0.0, 0.0, 1e-9);
      }
    }
  }

  // very optically thick: hand over to the equilibrium treatment
  {
    const Real alpha = 1.0, cdt = 1.0, kabs = 30.0;
    Point pt(alpha, 0.0);
    Vec4 Fstar{}, Fnew{};
    Real chi = 1.0 / 3.0, Estar = 0.5, Enew = Estar;
    SrcSignal sig = source_update_ll(BrentFunctor{}, HybridsjFunctor{}, cdt, alpha,
                                     pt.g_dd, pt.g_uu, pt.n_d, pt.n_u, pt.gamma_ud,
                                     pt.u_d, pt.u_u, pt.v_d, pt.v_u, pt.proj_ud, pt.W,
                                     Estar, Fstar, Estar, Fstar, eta, kabs, kscat, chi,
                                     Enew, Fnew, p, Minerbo);
    CheckTrue("cdt^2 kabs (kabs + kscat) > thick_limit^2 returns SrcEquil",
              sig == SrcEquil);
  }
}

//------------------------------------------------------------------------------------
void TestLinalg() {
  std::cout << "-- 8. linear algebra and Hybridsj" << std::endl;
  constexpr int N = M1_MULTIROOTS_DIM;
  const Real A[N][N] = {{4.0, 1.0, -2.0, 0.5},
                        {1.0, 3.0, 0.0, -1.0},
                        {-2.0, 0.0, 5.0, 1.5},
                        {0.5, -1.0, 1.5, 2.0}};
  Real Q[N][N], R[N][N];
  qr_factorize(A, Q, R);
  Real err_qr = 0.0, err_orth = 0.0, err_tri = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Real qr = 0.0, qtq = 0.0;
      for (int k = 0; k < N; ++k) {
        qr += Q[i][k] * R[k][j];
        qtq += Q[k][i] * Q[k][j];
      }
      err_qr = std::max(err_qr, std::abs(qr - A[i][j]));
      err_orth = std::max(err_orth, std::abs(qtq - (i == j ? 1.0 : 0.0)));
      if (i > j) err_tri = std::max(err_tri, std::abs(R[i][j]));
    }
  }
  CheckClose("QR = A (max err)", err_qr, 0.0, 0.0, 1e-13);
  CheckClose("Q^T Q = 1 (max err)", err_orth, 0.0, 0.0, 1e-13);
  CheckClose("R upper triangular (max sub-diagonal)", err_tri, 0.0, 0.0, 1e-15);

  const Real xs[N] = {1.0, -2.0, 0.5, 3.0};
  Real b[N]{}, qtb[N]{}, x[N]{};
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) b[i] += A[i][j] * xs[j];
  }
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) qtb[i] += Q[k][i] * b[k];
  }
  qr_R_solve(R, qtb, x);
  Real err_x = 0.0;
  for (int i = 0; i < N; ++i) err_x = std::max(err_x, std::abs(x[i] - xs[i]));
  CheckClose("qr_R_solve recovers x from A x = b (max err)", err_x, 0.0, 0.0, 1e-12);

  // Hybridsj on f_i = x_i^3 - a_i
  const Real a[N] = {8.0, 27.0, 1.0, 0.125};
  auto fdf = [&a](const Real xx[N], Real f[N], Real J[N][N]) {
    for (int i = 0; i < N; ++i) {
      f[i] = xx[i] * xx[i] * xx[i] - a[i];
      for (int j = 0; j < N; ++j) J[i][j] = (i == j) ? 3.0 * xx[i] * xx[i] : 0.0;
    }
  };
  HybridsjState st{};
  HybridsjParams hp{};
  for (int i = 0; i < N; ++i) hp.x[i] = 1.0;
  HybridsjInitialize(fdf, st, hp);
  MathSignal ierr;
  int iter = 0;
  do {
    HybridsjIterate(fdf, st, hp);
    ierr = HybridsjTestDelta(hp.dx, hp.x, 1e-15, 1e-13);
  } while (ierr == LinalgContinue && ++iter < 200);
  const Real roots[N] = {2.0, 3.0, 1.0, 0.5};
  Real err_h = 0.0;
  for (int i = 0; i < N; ++i) err_h = std::max(err_h, std::abs(hp.x[i] - roots[i]));
  CheckClose("Hybridsj solves x_i^3 = a_i (max err)", err_h, 0.0, 0.0, 1e-10);
}


//------------------------------------------------------------------------------------
// A general spacetime point: lapse, shift and a non-diagonal 3-metric.
struct Geometry {
  Real alpha = 1.3;
  Real beta[3] = {0.2, -0.1, 0.05};
  Real gam[3][3] = {{1.2, 0.1, 0.02}, {0.1, 1.1, 0.05}, {0.02, 0.05, 0.9}};
};

//! Inverse and determinant of a symmetric 3x3 matrix by cofactors (independent of the
//! adm:: helpers under test).
Real Inverse3(const Real a[3][3], Real inv[3][3]) {
  const Real det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                   a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                   a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
  inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
  inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
  inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  return det;
}

void TestADM() {
  std::cout << "-- 9. ADM geometry as used by M1" << std::endl;
  const Geometry geo;
  const Real a = geo.alpha;
  const Real *b = geo.beta;
  const auto &gm = geo.gam;

  // 4-metric and inverse, exactly as RadiationM1::TimeUpdate assembles them
  Real garr_dd[16], garr_uu[16];
  adm::SpacetimeMetric(a, b[0], b[1], b[2], gm[0][0], gm[0][1], gm[0][2], gm[1][1],
                       gm[1][2], gm[2][2], garr_dd);
  adm::SpacetimeUpperMetric(a, b[0], b[1], b[2], gm[0][0], gm[0][1], gm[0][2], gm[1][1],
                            gm[1][2], gm[2][2], garr_uu);
  Sym4 g_dd{}, g_uu{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      g_dd(i, j) = garr_dd[i + j * 4];
      g_uu(i, j) = garr_uu[i + j * 4];
    }
  }
  Real err_id = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 4; ++k) {
      Real s = 0.0;
      for (int j = 0; j < 4; ++j) s += g_dd(i, j) * g_uu(j, k);
      err_id = std::max(err_id, std::abs(s - (i == k ? 1.0 : 0.0)));
    }
  }
  CheckClose("g_ab g^bc = delta (max err)", err_id, 0.0, 0.0, 1e-14);

  Real bd[3] = {0, 0, 0}, b2 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) bd[i] += gm[i][j] * b[j];
    b2 += bd[i] * b[i];
  }
  CheckClose("g_00 = -alpha^2 + beta_k beta^k", g_dd(0, 0), -a * a + b2, 1e-14);
  CheckClose("g_0x = beta_x", g_dd(0, 1), bd[0], 1e-14);
  CheckClose("g^00 = -1/alpha^2", g_uu(0, 0), -1.0 / (a * a), 1e-14);
  CheckClose("g^0y = beta^y/alpha^2", g_uu(0, 2), b[1] / (a * a), 1e-14);

  // 3-metric inverse as reconstructed in TimeUpdate: gamma^ij = g^ij + beta^i beta^j/a^2
  Real ginv[3][3];
  const Real det = Inverse3(gm, ginv);
  Real err_gi = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const Real gij = g_uu(i + 1, j + 1) + b[i] * b[j] / (a * a);
      err_gi = std::max(err_gi, std::abs(gij - ginv[i][j]));
    }
  }
  CheckClose("gamma^ij = g^ij + beta^i beta^j / alpha^2 (max err)", err_gi, 0.0, 0.0,
             1e-14);
  CheckClose("volform = sqrt(SpatialDet)",
             std::sqrt(adm::SpatialDet(gm[0][0], gm[0][1], gm[0][2], gm[1][1], gm[1][2],
                                       gm[2][2])),
             std::sqrt(det), 1e-14);

  // normal
  Vec4 n_d{}, n_u{};
  pack_n_d(a, n_d);
  tensor_contract(g_uu, n_d, n_u);
  CheckClose("n.n = -1", tensor_dot(n_d, n_u), -1.0, 1e-14);
  CheckClose("n^0 = 1/alpha", n_u(0), 1.0 / a, 1e-14);
  CheckClose("n^x = -beta^x/alpha", n_u(1), -b[0] / a, 1e-14);

  // fluid four-velocity from the primitive z^i = W v^i, as in TimeUpdate
  const Real z[3] = {0.4, -0.2, 0.1};
  const Real W = get_w_lorentz(z[0], z[1], z[2], g_dd);
  Vec4 u_u{}, u_d{}, v_u{};
  pack_u_u(W / a, z[0] - W * b[0] / a, z[1] - W * b[1] / a, z[2] - W * b[2] / a, u_u);
  pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), a, b[0], b[1], b[2], v_u);
  tensor_contract(g_dd, u_u, u_d);
  CheckClose("u.u = -1", tensor_dot(u_d, u_u), -1.0, 1e-14);
  Real err_unv = 0.0;
  for (int i = 0; i < 4; ++i) {
    err_unv = std::max(err_unv, std::abs(u_u(i) - W * (n_u(i) + v_u(i))));
  }
  CheckClose("u = W (n + v) (max err)", err_unv, 0.0, 0.0, 1e-14);

  // lab-frame flux: F_0 = beta^i F_i makes F orthogonal to n
  Vec4 F_d{}, F_u{};
  pack_F_d(b[0], b[1], b[2], 0.3, -0.2, 0.15, F_d);
  tensor_contract(g_uu, F_d, F_u);
  CheckClose("F^a n_a = 0", tensor_dot(F_u, n_d), 0.0, 0.0, 1e-15);
  Real F2_3 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) F2_3 += ginv[i][j] * F_d(i + 1) * F_d(j + 1);
  }
  CheckClose("g^ab F_a F_b = gamma^ij F_i F_j", tensor_dot(g_uu, F_d, F_d), F2_3, 1e-14);
}

//------------------------------------------------------------------------------------
// Section 10: the flux kernel.  The pgen fills the module's arrays on the host, so these
// checks are meant for CPU builds (the test is registered as *_cpu).

struct FluxSetup {
  Real alpha = 1.0;
  Real beta[3] = {0.0, 0.0, 0.0};
  Real gam[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Real zx = 0.0;          // fluid W v^x
  bool matter = false;    // matter_sources: switches on the 1/(kappa dx) correction
  Real kappa = 0.0;       // abs_1 in every cell
  Real kscat = 0.0;       // scat_1 in every cell
};

struct CellState { Real E, Fx, Fy, Fz, chi; };

//! Writes the setup and a per-cell state (cell index n counted from the first ghost
//! cell) into the module, runs CalculateFluxes once, and returns the x1-face fluxes of
//! species 0 for faces is..ie+1 as flux[var][face - is] (var 4, the N flux, only when
//! the build evolves N).  N is set equal to E in every cell.
template <class StateFn>
void RunFluxKernel(MeshBlockPack *pmbp, const FluxSetup &fs, StateFn state,
                   std::vector<std::vector<Real>> &flux) {
  auto *pr = pmbp->pradm1;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ncells1 = indcs.nx1 + 2 * indcs.ng;
  const int is = indcs.is, ie = indcs.ie;
  const int nmb = pmbp->nmb_thispack;
  const int nvars = pr->nvars, nspecies = pr->nspecies;

  auto adm_h = Kokkos::create_mirror_view(pmbp->padm->u_adm);
  auto u0_h = Kokkos::create_mirror_view(pr->u0);
  auto w0_h = Kokkos::create_mirror_view(pr->w0);
  auto chi_h = Kokkos::create_mirror_view(pr->chi);
  auto abs_h = Kokkos::create_mirror_view(pr->abs_1);
  auto scat_h = Kokkos::create_mirror_view(pr->scat_1);
  using A = adm::ADM;
  const int gidx[3][3] = {{A::I_ADM_GXX, A::I_ADM_GXY, A::I_ADM_GXZ},
                          {A::I_ADM_GXY, A::I_ADM_GYY, A::I_ADM_GYZ},
                          {A::I_ADM_GXZ, A::I_ADM_GYZ, A::I_ADM_GZZ}};
  const int bidx[3] = {A::I_ADM_BETAX, A::I_ADM_BETAY, A::I_ADM_BETAZ};
  for (int m = 0; m < nmb; ++m) {
    for (int i = 0; i < ncells1; ++i) {
      for (int v = 0; v < A::nadm; ++v) adm_h(m, v, 0, 0, i) = 0.0;
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) adm_h(m, gidx[p][q], 0, 0, i) = fs.gam[p][q];
        adm_h(m, bidx[p], 0, 0, i) = fs.beta[p];
      }
      adm_h(m, A::I_ADM_ALPHA, 0, 0, i) = fs.alpha;
      adm_h(m, A::I_ADM_PSI4, 0, 0, i) = 1.0;

      w0_h(m, IDN, 0, 0, i) = 1.0;
      w0_h(m, IVX, 0, 0, i) = fs.zx;
      w0_h(m, IVY, 0, 0, i) = 0.0;
      w0_h(m, IVZ, 0, 0, i) = 0.0;
      w0_h(m, IEN, 0, 0, i) = 1.0;

      const CellState c = state(i);
      for (int nu = 0; nu < nspecies; ++nu) {
        u0_h(m, CombinedIdx(nu, M1_E_IDX, nvars), 0, 0, i) = c.E;
        u0_h(m, CombinedIdx(nu, M1_FX_IDX, nvars), 0, 0, i) = c.Fx;
        u0_h(m, CombinedIdx(nu, M1_FY_IDX, nvars), 0, 0, i) = c.Fy;
        u0_h(m, CombinedIdx(nu, M1_FZ_IDX, nvars), 0, 0, i) = c.Fz;
        if (nvars > M1_N_IDX) u0_h(m, CombinedIdx(nu, M1_N_IDX, nvars), 0, 0, i) = c.E;
        chi_h(m, nu, 0, 0, i) = c.chi;
        abs_h(m, nu, 0, 0, i) = fs.kappa;
        scat_h(m, nu, 0, 0, i) = fs.kscat;
      }
    }
  }
  Kokkos::deep_copy(pmbp->padm->u_adm, adm_h);
  Kokkos::deep_copy(pr->u0, u0_h);
  Kokkos::deep_copy(pr->w0, w0_h);
  Kokkos::deep_copy(pr->chi, chi_h);
  Kokkos::deep_copy(pr->abs_1, abs_h);
  Kokkos::deep_copy(pr->scat_1, scat_h);
  pr->params.matter_sources = fs.matter;

  (void) pr->CalculateFluxes(nullptr, 0);
  Kokkos::fence();

  auto flx_h = Kokkos::create_mirror_view(pr->uflx.x1f);
  Kokkos::deep_copy(flx_h, pr->uflx.x1f);
  const int nret = (nvars > M1_N_IDX) ? 5 : 4;
  flux.assign(nret, std::vector<Real>(ie + 2 - is));
  for (int v = 0; v < nret; ++v) {
    for (int f = is; f <= ie + 1; ++f) {
      flux[v][f - is] = flx_h(0, CombinedIdx(0, v, nvars), 0, 0, f);
    }
  }
}

void TestFluxes(MeshBlockPack *pmbp) {
  std::cout << "-- 10. flux kernel (RadiationM1::CalculateFluxes)" << std::endl;
  if (pmbp->pradm1 == nullptr || pmbp->padm == nullptr || !pmbp->pmesh->one_d) {
    CheckTrue("flux tests need a 1D mesh with <adm> and <radiation_m1>", false);
    return;
  }
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is;
  const int nface = indcs.nx1 + 1;
  const Real dx = pmbp->pmesh->mesh_size.dx1;
  const Real saved_minmod = pmbp->pradm1->params.minmod_theta;
  const bool saved_matter = pmbp->pradm1->params.matter_sources;
  pmbp->pradm1->params.minmod_theta = 1.0;
  std::vector<std::vector<Real>> flx;

  // (a) uniform state in a curved, shifted metric with a moving fluid
  {
    FluxSetup fs;
    const Geometry geo;
    fs.alpha = geo.alpha;
    for (int p = 0; p < 3; ++p) {
      fs.beta[p] = geo.beta[p];
      for (int q = 0; q < 3; ++q) fs.gam[p][q] = geo.gam[p][q];
    }
    fs.zx = 0.2;
    const CellState c{1.0, 0.3, 0.1, -0.05, 0.5};
    RunFluxKernel(pmbp, fs, [&](int) { return c; }, flx);
    Real spread = 0.0;
    for (int v = 0; v < 4; ++v) {
      for (int f = 0; f < nface; ++f) {
        spread = std::max(spread, std::abs(flx[v][f] - flx[v][0]));
      }
    }
    CheckClose("uniform state: identical flux on every face (max diff)", spread, 0.0,
               0.0, 1e-14);
    Real ginv[3][3];
    Inverse3(geo.gam, ginv);
    const Real Fup_x = ginv[0][0] * c.Fx + ginv[0][1] * c.Fy + ginv[0][2] * c.Fz;
    CheckClose("uniform state: E flux = alpha gamma^xj F_j - beta^x E", flx[0][0],
               geo.alpha * Fup_x - geo.beta[0] * c.E, 1e-13);
  }

  // (a2) fluid at rest in the same metric: the F_j flux is alpha P^x_j - beta^x F_j with
  // P^x_j = gamma^xk P_kj, thin P_ij = E F_i F_j / F^2 and thick P_ij = (E/3) gamma_ij.
  // F is not along x, so P^x_j != P^j_x.  With N = E the N flux is alpha F^x - beta^x E.
  {
    FluxSetup fs;
    const Geometry geo;
    fs.alpha = geo.alpha;
    for (int p = 0; p < 3; ++p) {
      fs.beta[p] = geo.beta[p];
      for (int q = 0; q < 3; ++q) fs.gam[p][q] = geo.gam[p][q];
    }
    Real ginv[3][3];
    Inverse3(geo.gam, ginv);
    const Real E = 1.0, Fd[3] = {0.3, 0.1, -0.05};
    Real Fu[3] = {0.0, 0.0, 0.0}, F2 = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) Fu[i] += ginv[i][k] * Fd[k];
      F2 += Fu[i] * Fd[i];
    }
    for (int thin = 0; thin < 2; ++thin) {
      const CellState c{E, Fd[0], Fd[1], Fd[2], thin ? 1.0 : 1.0 / 3.0};
      RunFluxKernel(pmbp, fs, [&](int) { return c; }, flx);
      const std::string tag = thin ? "thin" : "thick";
      for (int j = 0; j < 3; ++j) {
        const Real Pxj = thin ? E * Fu[0] * Fd[j] / F2 : (j == 0 ? E / 3.0 : 0.0);
        CheckClose("fluid at rest, curved metric, " + tag + ": F_" + "xyz"[j] +
                       " flux = alpha P^x_j - beta^x F_j",
                   flx[1 + j][0], geo.alpha * Pxj - geo.beta[0] * Fd[j], 1e-13, 1e-15);
      }
      if (flx.size() > 4) {
        CheckClose("fluid at rest, curved metric, " + tag + ": N flux", flx[4][0],
                   geo.alpha * Fu[0] - geo.beta[0] * E, 1e-13);
      }
    }
  }

  // (b) free-streaming step, thin closure, flat space: the flux is upwinded
  const int s = 4;  // the step sits on face is+s (between cells is+s-1 and is+s)
  for (int dirn = 0; dirn < 2; ++dirn) {
    const Real sgn = (dirn == 0) ? 1.0 : -1.0;
    FluxSetup fs;
    RunFluxKernel(pmbp, fs, [&](int i) {
      const Real E = (i < is + s) ? 1.0 : 0.5;
      return CellState{E, sgn * E, 0.0, 0.0, 1.0};
    }, flx);
    const Real up = (dirn == 0) ? 1.0 : 0.5;
    const std::string tag = (dirn == 0) ? "right-moving" : "left-moving";
    CheckClose("free streaming " + tag + " step: E flux is the upwind value",
               flx[0][s], sgn * up, 1e-14);
    CheckClose("free streaming " + tag + " step: F_x flux is the upwind value",
               flx[1][s], up, 1e-14);
  }

  // (c) Rusanov speed alpha sqrt(gamma^xx) + |beta^x| from the diffusive part of the
  // flux, for both signs of the shift
  for (const Real bx : {0.3, -0.3}) {
    FluxSetup fs;
    fs.alpha = 2.0;
    fs.beta[0] = bx;
    fs.gam[0][0] = 4.0;
    const Real EL = 1.0, ER = 0.5;
    RunFluxKernel(pmbp, fs, [&](int i) {
      return CellState{(i < is + s) ? EL : ER, 0.0, 0.0, 0.0, 1.0 / 3.0};
    }, flx);
    const Real central = -fs.beta[0] * (EL + ER) / 2.0;
    const Real c = -2.0 * (flx[0][s] - central) / (ER - EL);
    CheckClose(std::string("Rusanov speed = alpha sqrt(gamma^xx) + |beta^x|, beta^x ") +
                   (bx > 0 ? "> 0" : "< 0"),
               c, fs.alpha * std::sqrt(1.0 / fs.gam[0][0]) + std::abs(fs.beta[0]), 1e-13);
  }

  // (d) limiter: smooth data get the central flux, a sawtooth gets the Rusanov flux
  {
    FluxSetup fs;
    const Real E = 1.0;
    RunFluxKernel(pmbp, fs, [&](int i) {
      return CellState{E, 0.1 + 0.01 * i, 0.0, 0.0, 1.0 / 3.0};
    }, flx);
    Real err_c = 0.0, err_e = 0.0;
    for (int f = 0; f < nface; ++f) {
      const int i = is + f;
      err_c = std::max(err_c, std::abs(flx[1][f] - E / 3.0));
      err_e = std::max(err_e, std::abs(flx[0][f] - (0.2 + 0.01 * (2 * i - 1)) / 2.0));
    }
    CheckClose("linear F_x: central (unlimited) F_x flux, no diffusion (max err)", err_c,
               0.0, 0.0, 1e-14);
    CheckClose("linear F_x: E flux = face average of F_x (max err)", err_e, 0.0, 0.0,
               1e-14);

    RunFluxKernel(pmbp, fs, [&](int i) {
      return CellState{E, 0.1 + 0.02 * ((i % 2 == 0) ? 1.0 : -1.0), 0.0, 0.0,
                       1.0 / 3.0};
    }, flx);
    Real err_s = 0.0;
    for (int f = 0; f < nface; ++f) {
      const int i = is + f;
      const Real dF = 0.04 * ((i % 2 == 0) ? 1.0 : -1.0);  // F_x(i) - F_x(i-1)
      err_s = std::max(err_s, std::abs(flx[1][f] - (E / 3.0 - dF / 2.0)));
    }
    CheckClose("sawtooth F_x: Rusanov F_x flux (max err)", err_s, 0.0, 0.0, 1e-14);
  }

  // (e) optically thick: diffusion reduced by A = 1/(kappa dx), kappa = abs + scat
  {
    FluxSetup fs;
    fs.matter = true;
    fs.kappa = 5.0 / dx;
    fs.kscat = 5.0 / dx;
    RunFluxKernel(pmbp, fs, [&](int i) {
      const Real E = (i < is + s) ? 1.0 : 0.5;
      return CellState{E, E, 0.0, 0.0, 1.0};
    }, flx);
    const Real central = 0.75, rusanov = 1.0, A = 0.1;
    CheckClose("thick cells: flux = central - (1/(kappa dx)) (central - Rusanov)",
               flx[0][s], central - A * (central - rusanov), 1e-13);
  }

  pmbp->pradm1->params.minmod_theta = saved_minmod;
  pmbp->pradm1->params.matter_sources = saved_matter;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::RadM1Core()
//! \brief Unit tests of the pointwise M1 kernels; exits non-zero on failure.

void ProblemGenerator::RadM1Core(ParameterInput *pin, const bool restart) {
  if (restart) return;
  strict_known = pin->GetOrAddBoolean("problem", "strict_known", false);

  std::cout << "=== rad_m1_core: pointwise M1 kernels ===" << std::endl;
  TestClosureFunction();
  TestPressure();
  TestFrames();
  TestRootfinders();
  TestFloors();
  TestSources();
  TestSourceUpdate();
  TestLinalg();
  TestADM();
  TestFluxes(pmy_mesh_->pmb_pack);

  std::cout << "=== rad_m1_core: " << (nrun - nfail - nknown) << "/" << nrun
            << " checks passed, " << nknown << " known defect(s)"
            << (strict_known ? " (counted as failures)" : " (not counted)") << std::endl;
  if (nfail > 0) {
    std::cout << "### FATAL ERROR: " << nfail << " rad_m1_core check(s) failed"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
