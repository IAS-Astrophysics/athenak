//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License
//========================================================================================
//! \file z4c_critical_kerr_formation.cpp
//! \brief Compact helical conformal free data for native CTS solve and Z4c evolution.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <Kokkos_MathematicalFunctions.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "z4c/id_solve.hpp"
#include "z4c/z4c.hpp"

namespace {

constexpr int kMaxRadialModes = 8;

struct CriticalKerrOptions {
  int ell = 2;
  int emm = 2;
  int radial_modes = 1;
  Real amplitude = 1.0e-3;
  Real omega = 0.0;
  Real support_radius = 5.0;
  Real bump_steepness = 0.35;
  Real phase = 0.0;
  Real helicity = 1.0;
  Real ingoing_sign = 1.0;
  Real lapse = 1.0;
  Real center[3] = {0.0, 0.0, 0.0};
  Real radial_coeff[kMaxRadialModes] = {1.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0};
  bool use_precollapsed_lapse = false;
};

KOKKOS_INLINE_FUNCTION
int SymIdx(const int a, const int b) {
  const int lo = a < b ? a : b;
  const int hi = a < b ? b : a;
  if (lo == 0 && hi == 0) return 0;
  if (lo == 0 && hi == 1) return 1;
  if (lo == 0 && hi == 2) return 2;
  if (lo == 1 && hi == 1) return 3;
  if (lo == 1 && hi == 2) return 4;
  return 5;
}

KOKKOS_INLINE_FUNCTION
Real Det3(const Real g[3][3]) {
  return g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1]) -
         g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0]) +
         g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
}

KOKKOS_INLINE_FUNCTION
void Inverse3(const Real g[3][3], const Real det, Real gi[3][3]) {
  const Real inv_det = 1.0/det;
  gi[0][0] =  (g[1][1]*g[2][2] - g[1][2]*g[2][1])*inv_det;
  gi[0][1] = -(g[0][1]*g[2][2] - g[0][2]*g[2][1])*inv_det;
  gi[0][2] =  (g[0][1]*g[1][2] - g[0][2]*g[1][1])*inv_det;
  gi[1][0] = gi[0][1];
  gi[1][1] =  (g[0][0]*g[2][2] - g[0][2]*g[2][0])*inv_det;
  gi[1][2] = -(g[0][0]*g[1][2] - g[0][2]*g[1][0])*inv_det;
  gi[2][0] = gi[0][2];
  gi[2][1] = gi[1][2];
  gi[2][2] =  (g[0][0]*g[1][1] - g[0][1]*g[1][0])*inv_det;
}

KOKKOS_INLINE_FUNCTION
void LegendreAndDerivative(const int n, const Real x, Real &p, Real &dpdx) {
  if (n == 0) {
    p = 1.0;
    dpdx = 0.0;
    return;
  }
  Real p_nm2 = 1.0;
  Real dp_nm2 = 0.0;
  Real p_nm1 = x;
  Real dp_nm1 = 1.0;
  if (n == 1) {
    p = p_nm1;
    dpdx = dp_nm1;
    return;
  }
  for (int l = 2; l <= n; ++l) {
    const Real lr = static_cast<Real>(l);
    p = ((2.0*lr - 1.0)*x*p_nm1 - (lr - 1.0)*p_nm2)/lr;
    dpdx = ((2.0*lr - 1.0)*(p_nm1 + x*dp_nm1) -
            (lr - 1.0)*dp_nm2)/lr;
    p_nm2 = p_nm1;
    dp_nm2 = dp_nm1;
    p_nm1 = p;
    dp_nm1 = dpdx;
  }
}

KOKKOS_INLINE_FUNCTION
void RadialProfileAndDerivative(const Real r, const CriticalKerrOptions opt,
                                Real &value, Real &dr_value) {
  value = 0.0;
  dr_value = 0.0;
  if (!(opt.support_radius > 0.0) || r <= 0.0 || r >= opt.support_radius) {
    return;
  }

  const Real rho = r/opt.support_radius;
  const Real one_minus_rho = 1.0 - rho;
  const Real denom = rho*one_minus_rho;
  if (denom <= 0.0) return;

  const Real bump =
      Kokkos::exp(-opt.bump_steepness*(1.0/denom - 4.0));
  const Real dbump_dr =
      bump*opt.bump_steepness*(1.0 - 2.0*rho)/
      (rho*rho*one_minus_rho*one_minus_rho*opt.support_radius);
  const Real x = 2.0*rho - 1.0;
  const int modes = opt.radial_modes < kMaxRadialModes
                        ? opt.radial_modes
                        : kMaxRadialModes;

  for (int n = 0; n < modes; ++n) {
    const Real coeff = opt.radial_coeff[n];
    if (coeff == 0.0) continue;
    Real legendre = 0.0;
    Real dlegendre_dx = 0.0;
    LegendreAndDerivative(n, x, legendre, dlegendre_dx);
    value += coeff*bump*legendre;
    dr_value += coeff*(dbump_dr*legendre +
                       bump*dlegendre_dx*2.0/opt.support_radius);
  }
  value *= opt.amplitude;
  dr_value *= opt.amplitude;
}

CriticalKerrOptions ReadOptions(ParameterInput *pin) {
  CriticalKerrOptions opt;
  opt.ell = pin->GetOrAddInteger("problem", "ell", opt.ell);
  opt.emm = pin->GetOrAddInteger("problem", "m", opt.emm);
  opt.radial_modes = pin->GetOrAddInteger("problem", "radial_modes", opt.radial_modes);
  opt.amplitude = pin->GetOrAddReal("problem", "amplitude", opt.amplitude);
  opt.omega = pin->GetOrAddReal("problem", "omega", opt.omega);
  opt.support_radius = pin->GetOrAddReal("problem", "support_radius", opt.support_radius);
  opt.bump_steepness = pin->GetOrAddReal("problem", "bump_steepness", opt.bump_steepness);
  opt.phase = pin->GetOrAddReal("problem", "phase", opt.phase);
  opt.helicity = pin->GetOrAddReal("problem", "helicity", opt.helicity);
  opt.ingoing_sign = pin->GetOrAddReal("problem", "ingoing_sign", opt.ingoing_sign);
  opt.lapse = pin->GetOrAddReal("problem", "initial_lapse", opt.lapse);
  opt.center[0] = pin->GetOrAddReal("problem", "center_x", opt.center[0]);
  opt.center[1] = pin->GetOrAddReal("problem", "center_y", opt.center[1]);
  opt.center[2] = pin->GetOrAddReal("problem", "center_z", opt.center[2]);
  opt.use_precollapsed_lapse =
      pin->GetOrAddBoolean("problem", "use_precollapsed_lapse",
                           opt.use_precollapsed_lapse);
  opt.radial_modes = std::max(1, std::min(kMaxRadialModes, opt.radial_modes));
  for (int n = 0; n < kMaxRadialModes; ++n) {
    const std::string suffix = std::to_string(n);
    opt.radial_coeff[n] =
        pin->GetOrAddReal("problem", "radial_coeff_" + suffix, opt.radial_coeff[n]);
  }

  if (opt.ell != 2 || opt.emm != 2 || !(opt.support_radius > 0.0) ||
      opt.bump_steepness < 0.0 || !(opt.lapse > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "z4c_critical_kerr_formation currently implements the regular "
              << "solid-harmonic ell=m=2 seed only, with support_radius>0, "
              << "bump_steepness>=0, and initial_lapse>0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return opt;
}

void FillCriticalKerrFreeDataADM(MeshBlockPack *pmbp, const CriticalKerrOptions opt) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int isg = is - indcs.ng;
  const int ieg = ie + indcs.ng;
  const int jsg = js - indcs.ng;
  const int jeg = je + indcs.ng;
  const int ksg = ks - indcs.ng;
  const int keg = ke + indcs.ng;
  const int nmb = pmbp->nmb_thispack;
  auto &adm_vars = pmbp->padm->adm;

  par_for("critical_kerr_free_data_adm", DevExeSpace(), 0, nmb - 1,
          ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max) - opt.center[0];
    const Real y = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max) - opt.center[1];
    const Real z = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max) - opt.center[2];
    const Real r = Kokkos::sqrt(x*x + y*y + z*z);

    Real f = 0.0;
    Real df_dr = 0.0;
    RadialProfileAndDerivative(r, opt, f, df_dr);

    // Regular ell=m=2 solid-harmonic quadrature seed from the notes:
    // Re(Y22) ~ x^2-y^2 and Im(Y22) ~ 2xy, with constant STF polarizations.
    const Real inv_r0_sq = 1.0/(opt.support_radius*opt.support_radius);
    const Real solid_c = (x*x - y*y)*inv_r0_sq;
    const Real solid_s = 2.0*x*y*inv_r0_sq;
    const Real phase = opt.omega*r + opt.phase;
    const Real cos_phase = Kokkos::cos(phase);
    const Real sin_phase = Kokkos::sin(phase);

    const Real pol_e[3][3] = {{1.0, 0.0, 0.0},
                              {0.0, -1.0, 0.0},
                              {0.0, 0.0, 0.0}};
    const Real pol_b[3][3] = {{0.0, 1.0, 0.0},
                              {1.0, 0.0, 0.0},
                              {0.0, 0.0, 0.0}};

    Real h[3][3];
    Real udot_cov[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const Real y_e = solid_c*pol_e[a][b];
        const Real y_b = solid_s*pol_b[a][b];
        const Real tensor = cos_phase*y_e + opt.helicity*sin_phase*y_b;
        const Real dt_tensor =
            -opt.omega*sin_phase*y_e + opt.helicity*opt.omega*cos_phase*y_b;
        h[a][b] = f*tensor;
        udot_cov[a][b] = opt.ingoing_sign*(df_dr*tensor + f*dt_tensor);
      }
    }

    Real q[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real h2 = 0.0;
        for (int c = 0; c < 3; ++c) h2 += h[a][c]*h[c][b];
        q[a][b] = (a == b ? 1.0 : 0.0) + h[a][b] + 0.5*h2;
      }
    }

    const Real q_det = Det3(q);
    const Real det_scale = q_det > 0.0 ? Kokkos::pow(q_det, -1.0/3.0) : 1.0;
    Real gbar[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        gbar[a][b] = det_scale*q[a][b];
      }
    }

    Real gbar_inv[3][3];
    Inverse3(gbar, Det3(gbar), gbar_inv);
    Real trace_udot = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        trace_udot += gbar_inv[a][b]*udot_cov[a][b];
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        udot_cov[a][b] -= (1.0/3.0)*gbar[a][b]*trace_udot;
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = gbar[a][b];
        // The native CTS BuildFreeData computes ubar^ij_TF = 2 alpha Ahat^ij
        // from this seed K_ij when beta=0, so K_ij = 0.5 ubar_ij.
        adm_vars.vK_dd(m, a, b, k, j, i) = 0.5*udot_cov[a][b];
      }
      adm_vars.beta_u(m, a, k, j, i) = 0.0;
    }
    adm_vars.psi4(m, k, j, i) = 1.0;
    adm_vars.alpha(m, k, j, i) = opt.lapse;
  });
}

void ConvertADMToZ4cAndConstraints(MeshBlockPack *pmbp, ParameterInput *pin,
                                   const int fd_stencil) {
  switch (fd_stencil) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "z4c_critical_kerr_formation supports Z4c fd_stencil = 2, 3, or 4"
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pmbp->pz4c->Z4cToADM(pmbp);
}

} // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr || pmbp->pid_solve == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "z4c_critical_kerr_formation requires <z4c>, <adm>, and "
              << "<id_solve> blocks so the native CTS solver can run before Z4c."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const CriticalKerrOptions options = ReadOptions(pin);
  FillCriticalKerrFreeDataADM(pmbp, options);
  if (options.use_precollapsed_lapse) {
    pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  }
  ConvertADMToZ4cAndConstraints(pmbp, pin, pmbp->pz4c->opt.fd_stencil);

  std::cout << "Initialized critical Kerr conformal CTS free data. "
            << "Native <id_solve> will solve the constraints before Z4c evolution."
            << std::endl;
}
