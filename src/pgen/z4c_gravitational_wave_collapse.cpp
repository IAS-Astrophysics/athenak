//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License
//========================================================================================
//! \file z4c_gravitational_wave_collapse.cpp
//! \brief Parametrized vacuum gravitational-wave collapse data for Z4c.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include <Kokkos_MathematicalFunctions.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

namespace {

constexpr int kMaxRadialModes = 8;
constexpr const char *kWaveBlock = "problem_gw_collapse";
constexpr const char *kDomainBlock = "problem_gw_collapse_domain";

struct WaveCollapseOptions {
  int ell = 2;
  int emm = 2;
  int radial_modes = 1;
  Real amplitude = 1.0e-3;
  Real omega = 0.0;
  Real support_radius = 8.0;
  Real radial_center = 4.0;
  Real radial_width = 1.5;
  Real bump_steepness = 4.0;
  Real ingoing_weight = 1.0;
  Real rotation_weight = 1.0;
  Real helicity = 1.0;
  Real plus_weight = 1.0;
  Real cross_weight = 0.0;
  Real phase = 0.0;
  Real extrinsic_sign = -0.5;
  Real center[3] = {0.0, 0.0, 0.0};
  Real metric_coeff[kMaxRadialModes] = {1.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0};
  Real momentum_coeff[kMaxRadialModes] = {1.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0};
  bool use_precollapsed_lapse = false;
};

struct ShellBand {
  Real inner = 0.0;
  Real outer = 0.0;
  int reflevel = 0;
};

struct WaveCollapseDomainOptions {
  Real center[3] = {0.0, 0.0, 0.0};
  Real filled_radius = 2.0;
  int filled_reflevel = 1;
  int shell_count = 0;
  ShellBand shell[8];
};

WaveCollapseDomainOptions g_domain_options;

KOKKOS_INLINE_FUNCTION
Real safe_radius(const Real x, const Real y, const Real z) {
  return Kokkos::sqrt(x*x + y*y + z*z);
}

KOKKOS_INLINE_FUNCTION
void radial_profile_and_derivative(const Real r, const WaveCollapseOptions opt,
                                   const bool momentum_profile, Real &value,
                                   Real &dr_value) {
  value = 0.0;
  dr_value = 0.0;
  if (!(opt.support_radius > 0.0) || r >= opt.support_radius) {
    return;
  }

  const Real x = r/opt.support_radius;
  const Real one_minus_x2 = 1.0 - x*x;
  if (one_minus_x2 <= 0.0) {
    return;
  }

  const Real bump =
      Kokkos::exp(-opt.bump_steepness*x*x/one_minus_x2);
  const Real dbump_dx =
      bump*(-2.0*opt.bump_steepness*x)/(one_minus_x2*one_minus_x2);
  Real gaussian = 1.0;
  Real dgaussian_dr = 0.0;
  if (opt.radial_width > 0.0) {
    const Real scaled = (r - opt.radial_center)/opt.radial_width;
    gaussian = Kokkos::exp(-0.5*scaled*scaled);
    dgaussian_dr = -scaled*gaussian/opt.radial_width;
  }

  const int modes = opt.radial_modes < kMaxRadialModes
                        ? opt.radial_modes
                        : kMaxRadialModes;
  for (int n = 0; n < modes; ++n) {
    const Real coeff =
        momentum_profile ? opt.momentum_coeff[n] : opt.metric_coeff[n];
    if (coeff == 0.0) {
      continue;
    }
    const int power = opt.ell + n;
    Real x_power = 1.0;
    for (int p = 0; p < power; ++p) {
      x_power *= x;
    }
    Real dx_power = 0.0;
    if (power == 0) {
      dx_power = 0.0;
    } else if (x == 0.0) {
      dx_power = power == 1 ? 1.0 : 0.0;
    } else {
      dx_power = static_cast<Real>(power)*x_power/x;
    }
    const Real basis = x_power*bump*gaussian;
    const Real dbasis_dr =
        ((dx_power*bump + x_power*dbump_dx)*gaussian)/
            opt.support_radius +
        x_power*bump*dgaussian_dr;
    value += coeff*basis;
    dr_value += coeff*dbasis_dr;
  }
  value *= opt.amplitude;
  dr_value *= opt.amplitude;
}

KOKKOS_INLINE_FUNCTION
void angular_tensors(const Real x, const Real y, const Real z,
                     const WaveCollapseOptions opt, Real real_tensor[3][3],
                     Real imag_tensor[3][3], Real &angular_weight) {
  const Real r = safe_radius(x, y, z);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      real_tensor[a][b] = 0.0;
      imag_tensor[a][b] = 0.0;
    }
  }
  angular_weight = 0.0;
  if (r == 0.0) {
    return;
  }

  const Real rho = Kokkos::sqrt(x*x + y*y);
  const Real inv_r = 1.0/r;
  const Real sin_theta = rho*inv_r;
  const Real cos_theta = z*inv_r;
  const Real inv_rho = rho > 1.0e-14 ? 1.0/rho : 0.0;
  const Real cos_phi = rho > 1.0e-14 ? x*inv_rho : 1.0;
  const Real sin_phi = rho > 1.0e-14 ? y*inv_rho : 0.0;

  Real sin_power = 1.0;
  for (int p = 0; p < opt.ell; ++p) {
    sin_power *= sin_theta;
  }
  angular_weight = sin_power;
  if (angular_weight == 0.0) {
    return;
  }

  const Real e_theta[3] = {cos_theta*cos_phi, cos_theta*sin_phi,
                           -sin_theta};
  const Real e_phi[3] = {-sin_phi, cos_phi, 0.0};
  const Real phase = static_cast<Real>(opt.emm)*Kokkos::atan2(y, x) + opt.phase;
  const Real cos_phase = Kokkos::cos(phase);
  const Real sin_phase = Kokkos::sin(phase);

  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const Real plus =
          e_theta[a]*e_theta[b] - e_phi[a]*e_phi[b];
      const Real cross =
          e_theta[a]*e_phi[b] + e_phi[a]*e_theta[b];
      real_tensor[a][b] =
          angular_weight*(opt.plus_weight*plus*cos_phase +
                          opt.helicity*opt.cross_weight*cross*sin_phase);
      imag_tensor[a][b] =
          angular_weight*(opt.plus_weight*plus*sin_phase -
                          opt.helicity*opt.cross_weight*cross*cos_phase);
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real det3(const Real g[3][3]) {
  return g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1]) -
         g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0]) +
         g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
}

WaveCollapseOptions ReadOptions(ParameterInput *pin) {
  WaveCollapseOptions opt;
  opt.ell = pin->GetOrAddInteger(kWaveBlock, "ell", opt.ell);
  opt.emm = pin->GetOrAddInteger(kWaveBlock, "m", opt.ell);
  opt.radial_modes = pin->GetOrAddInteger(kWaveBlock, "radial_modes", 1);
  opt.amplitude = pin->GetOrAddReal(kWaveBlock, "amplitude", opt.amplitude);
  opt.omega = pin->GetOrAddReal(kWaveBlock, "omega", opt.omega);
  opt.support_radius =
      pin->GetOrAddReal(kWaveBlock, "support_radius", opt.support_radius);
  opt.radial_center =
      pin->GetOrAddReal(kWaveBlock, "radial_center", opt.radial_center);
  opt.radial_width =
      pin->GetOrAddReal(kWaveBlock, "radial_width", opt.radial_width);
  opt.bump_steepness =
      pin->GetOrAddReal(kWaveBlock, "bump_steepness",
                        opt.bump_steepness);
  opt.ingoing_weight =
      pin->GetOrAddReal(kWaveBlock, "ingoing_weight",
                        opt.ingoing_weight);
  opt.rotation_weight =
      pin->GetOrAddReal(kWaveBlock, "rotation_weight",
                        opt.rotation_weight);
  opt.helicity = pin->GetOrAddReal(kWaveBlock, "helicity", opt.helicity);
  opt.plus_weight =
      pin->GetOrAddReal(kWaveBlock, "plus_weight", opt.plus_weight);
  opt.cross_weight =
      pin->GetOrAddReal(kWaveBlock, "cross_weight", opt.cross_weight);
  opt.phase = pin->GetOrAddReal(kWaveBlock, "phase", opt.phase);
  opt.extrinsic_sign =
      pin->GetOrAddReal(kWaveBlock, "extrinsic_sign",
                        opt.extrinsic_sign);
  opt.center[0] = pin->GetOrAddReal(kWaveBlock, "center_x", opt.center[0]);
  opt.center[1] = pin->GetOrAddReal(kWaveBlock, "center_y", opt.center[1]);
  opt.center[2] = pin->GetOrAddReal(kWaveBlock, "center_z", opt.center[2]);
  opt.use_precollapsed_lapse =
      pin->GetOrAddBoolean(kWaveBlock, "use_precollapsed_lapse",
                           opt.use_precollapsed_lapse);
  opt.radial_modes = std::max(1, std::min(kMaxRadialModes, opt.radial_modes));
  for (int n = 0; n < kMaxRadialModes; ++n) {
    const std::string suffix = std::to_string(n);
    opt.metric_coeff[n] =
        pin->GetOrAddReal(kWaveBlock, "metric_coeff_" + suffix,
                          opt.metric_coeff[n]);
    opt.momentum_coeff[n] =
        pin->GetOrAddReal(kWaveBlock, "momentum_coeff_" + suffix,
                          opt.momentum_coeff[n]);
  }
  if (opt.ell < 2 || opt.emm < 1 || opt.emm > opt.ell ||
      !(opt.support_radius > 0.0) || opt.radial_width < 0.0 ||
      opt.bump_steepness < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "Invalid <" << kWaveBlock << "> parameters. Require ell>=2, "
              << "1<=m<=ell, support_radius>0, radial_width>=0, and "
              << "bump_steepness>=0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return opt;
}

WaveCollapseDomainOptions ReadDomainOptions(ParameterInput *pin,
                                            const WaveCollapseOptions &wave) {
  WaveCollapseDomainOptions domain;
  domain.center[0] = wave.center[0];
  domain.center[1] = wave.center[1];
  domain.center[2] = wave.center[2];
  domain.filled_radius =
      pin->GetOrAddReal(kDomainBlock, "filled_radius",
                        domain.filled_radius);
  domain.filled_reflevel =
      pin->GetOrAddInteger(kDomainBlock, "filled_reflevel",
                           domain.filled_reflevel);
  domain.shell_count =
      std::min(8, pin->GetOrAddInteger(kDomainBlock,
                                       "shell_count", 0));
  for (int shell = 0; shell < domain.shell_count; ++shell) {
    const std::string suffix = std::to_string(shell);
    domain.shell[shell].inner = pin->GetOrAddReal(
        kDomainBlock, "shell_" + suffix + "_inner",
        domain.filled_radius);
    domain.shell[shell].outer = pin->GetOrAddReal(
        kDomainBlock, "shell_" + suffix + "_outer",
        domain.shell[shell].inner);
    domain.shell[shell].reflevel = pin->GetOrAddInteger(
        kDomainBlock, "shell_" + suffix + "_reflevel",
        domain.filled_reflevel);
    if (domain.shell[shell].outer < domain.shell[shell].inner) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << kDomainBlock << " shell_" << suffix
                << " has outer < inner." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return domain;
}

void FillWaveCollapseADM(MeshBlockPack *pmbp, const WaveCollapseOptions opt) {
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
  auto &adm = pmbp->padm->adm;

  par_for("pgen_gw_collapse_adm", DevExeSpace(), 0, nmb - 1, ksg, keg,
          jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max) - opt.center[0];
    const Real y = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max) - opt.center[1];
    const Real z = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max) - opt.center[2];
    const Real r = safe_radius(x, y, z);

    Real metric_radial = 0.0;
    Real d_metric_radial = 0.0;
    Real momentum_radial = 0.0;
    Real d_momentum_radial = 0.0;
    radial_profile_and_derivative(r, opt, false, metric_radial,
                                  d_metric_radial);
    radial_profile_and_derivative(r, opt, true, momentum_radial,
                                  d_momentum_radial);

    Real real_tensor[3][3];
    Real imag_tensor[3][3];
    Real angular_weight = 0.0;
    angular_tensors(x, y, z, opt, real_tensor, imag_tensor, angular_weight);

    Real g[3][3];
    Real h[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        h[a][b] = metric_radial*real_tensor[a][b];
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real h2 = 0.0;
        for (int c = 0; c < 3; ++c) {
          h2 += h[a][c]*h[c][b];
        }
        g[a][b] = (a == b ? 1.0 : 0.0) + h[a][b] + 0.5*h2;
      }
    }

    const Real determinant = det3(g);
    const Real det_scale =
        determinant > 0.0 ? Kokkos::pow(determinant, -1.0/3.0) : 1.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        g[a][b] *= det_scale;
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        const Real dt_gamma =
            opt.ingoing_weight*d_momentum_radial*real_tensor[a][b] +
            opt.rotation_weight*opt.omega*static_cast<Real>(opt.emm)*
                momentum_radial*imag_tensor[a][b];
        adm.g_dd(m, a, b, k, j, i) = g[a][b];
        adm.vK_dd(m, a, b, k, j, i) = opt.extrinsic_sign*dt_gamma;
      }
    }
    adm.psi4(m, k, j, i) = 1.0;
    adm.alpha(m, k, j, i) = 1.0;
  });
}

Real block_min_radius(const Real xmin, const Real xmax, const Real ymin,
                      const Real ymax, const Real zmin, const Real zmax,
                      const Real center[3]) {
  const Real dx = center[0] < xmin ? xmin - center[0]
                : center[0] > xmax ? center[0] - xmax
                                    : 0.0;
  const Real dy = center[1] < ymin ? ymin - center[1]
                : center[1] > ymax ? center[1] - ymax
                                    : 0.0;
  const Real dz = center[2] < zmin ? zmin - center[2]
                : center[2] > zmax ? center[2] - zmax
                                    : 0.0;
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

Real block_max_radius(const Real xmin, const Real xmax, const Real ymin,
                      const Real ymax, const Real zmin, const Real zmax,
                      const Real center[3]) {
  Real max_r2 = 0.0;
  for (const Real x : {xmin, xmax}) {
    for (const Real y : {ymin, ymax}) {
      for (const Real z : {zmin, zmax}) {
        const Real dx = x - center[0];
        const Real dy = y - center[1];
        const Real dz = z - center[2];
        max_r2 = std::max(max_r2, dx*dx + dy*dy + dz*dz);
      }
    }
  }
  return std::sqrt(max_r2);
}

void WaveCollapseRefinementCondition(MeshBlockPack *pmbp) {
  Mesh *pmesh = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  const WaveCollapseDomainOptions domain = g_domain_options;

  for (int m = 0; m < nmb; ++m) {
    const int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;
    const Real xmin = size.h_view(m).x1min;
    const Real xmax = size.h_view(m).x1max;
    const Real ymin = size.h_view(m).x2min;
    const Real ymax = size.h_view(m).x2max;
    const Real zmin = size.h_view(m).x3min;
    const Real zmax = size.h_view(m).x3max;
    const Real min_r =
        block_min_radius(xmin, xmax, ymin, ymax, zmin, zmax, domain.center);
    const Real max_r =
        block_max_radius(xmin, xmax, ymin, ymax, zmin, zmax, domain.center);
    int desired_level = -1;
    if (min_r <= domain.filled_radius) {
      desired_level = std::max(desired_level, domain.filled_reflevel);
    }
    for (int shell = 0; shell < domain.shell_count; ++shell) {
      if (max_r >= domain.shell[shell].inner &&
          min_r <= domain.shell[shell].outer) {
        desired_level = std::max(desired_level, domain.shell[shell].reflevel);
      }
    }
    if (desired_level < 0) {
      refine_flag.h_view(m + mbs) = -1;
    } else if (level < desired_level) {
      refine_flag.h_view(m + mbs) = 1;
    } else if (level > desired_level) {
      refine_flag.h_view(m + mbs) = -1;
    } else {
      refine_flag.h_view(m + mbs) = 0;
    }
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func = WaveCollapseRefinementCondition;
  if (restart) {
    return;
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "z4c_gravitational_wave_collapse requires a <z4c> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const WaveCollapseOptions options = ReadOptions(pin);
  g_domain_options = ReadDomainOptions(pin, options);
  FillWaveCollapseADM(pmbp, options);
  if (options.use_precollapsed_lapse) {
    pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  }

  switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "Z4c ADMToZ4c supports nghost = 2, 3, or 4" << std::endl;
      std::exit(EXIT_FAILURE);
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
  }
  std::cout << "Initialized parametrized Z4c gravitational-wave collapse data."
            << std::endl;
}
