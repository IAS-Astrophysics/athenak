//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2026 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file superposed_boosted_puncture.cpp
//! \brief Superpose two nonspinning Schwarzschild punctures boosted along signed x1.
//! This is approximate initial data; the binary constraints are not solved here.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"

namespace {
void SuperposedRefinement(MeshBlockPack *pack) { pack->pz4c->pamr->Refine(pack); }
} // namespace

void ProblemGenerator::Z4cSuperposedPunctures(ParameterInput *pin, const bool restart) {
  user_ref_func = SuperposedRefinement;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->pz4c == nullptr || !pmy_mesh_->three_d) {
    std::cerr << "Superposed punctures require three dimensions and <z4c>." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto indcs = pmy_mesh_->mb_indcs;
  auto size = pack->pmb->mb_size;
  auto adm = pack->padm->adm;
  int nmb = pack->nmb_thispack;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ng = indcs.ng;
  par_for("superposed flat background", DevExeSpace(), 0, nmb-1,
      ks-ng, indcs.ke+ng, js-ng, indcs.je+ng, is-ng, indcs.ie+ng,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm.alpha(m,k,j,i) = 1.0;
    for (int a=0; a<3; ++a) {
      adm.beta_u(m,a,k,j,i) = 0.0;
      for (int b=a; b<3; ++b) {
        adm.g_dd(m,a,b,k,j,i) = (a == b ? 1.0 : 0.0);
        adm.vK_dd(m,a,b,k,j,i) = 0.0;
      }
    }
  });
  for (int hole=1; hole<=2; ++hole) {
    std::string prefix = "punc_" + std::to_string(hole);
    Real m0 = pin->GetOrAddReal("problem", prefix+"_rest_mass", 1.0);
    Real vel = pin->GetOrAddReal("problem", prefix+"_velocity_x1", 0.0);
    Real cx = pin->GetOrAddReal("problem", prefix+"_center_x1",
                                hole == 1 ? -2.0 : 2.0);
    Real cy = pin->GetOrAddReal("problem", prefix+"_center_x2", 0.0);
    Real cz = pin->GetOrAddReal("problem", prefix+"_center_x3", 0.0);
    Real floor = pin->GetOrAddReal("problem", "puncture_radius_floor", 1e-8);
    if (!std::isfinite(m0) || m0 <= 0 || !std::isfinite(vel) || std::abs(vel) >= 1 ||
        !std::isfinite(floor) || floor <= 0 || !std::isfinite(cx) ||
        !std::isfinite(cy) || !std::isfinite(cz)) {
      std::cerr << "Invalid puncture mass, velocity, center, or radius floor." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real Gamma = 1.0/std::sqrt(1.0-vel*vel);
    par_for("add boosted puncture", DevExeSpace(), 0, nmb-1,
        ks-ng, indcs.ke+ng, js-ng, indcs.je+ng, is-ng, indcs.ie+ng,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max)-cx;
      Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max)-cy;
      Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max)-cz;
        // Coordinates in comoving frame (x0)
        // Lorentz transformation along x-direction
        Real x0 = Gamma * x;  // At t = 0
        Real y0 = y;
        Real z0 = z;

        // Radial coordinate in comoving frame
        Real r0 = fmax(floor, std::sqrt(x0*x0 + y0*y0 + z0*z0));

        // Compute psi0 and its derivative
        Real psi0 = 1.0 + m0 / (2.0 * r0);
        Real psi0_prime = -m0 / (2.0 * r0 * r0);

        // Compute A and its derivative
        Real A = 1.0 - m0 / (2.0 * r0);
        Real A_prime = m0 / (2.0 * r0 * r0);

        // Compute alpha0 and its derivative
        Real alpha0 = A / psi0;
        Real alpha0_prime = (A_prime * psi0 - A * psi0_prime) / (psi0 * psi0);

        // Compute psi0^4 and alpha0^2
        Real psi0_4 = psi0 * psi0 * psi0 * psi0;
        Real alpha0_2 = alpha0 * alpha0;

        // Compute B0^2 and B0
        Real B0_squared = Gamma * Gamma * (1.0 - vel * vel * alpha0_2 / psi0_4);
        Real B0 = std::sqrt(B0_squared);

        Real den_beta = psi0_4 - alpha0_2 * vel * vel;

        // Spatial metric components gamma_{ij}
        adm.g_dd(m, 0, 0, k, j, i) += psi0_4 * B0_squared - 1;
        adm.g_dd(m, 1, 1, k, j, i) += psi0_4 - 1;
        adm.g_dd(m, 2, 2, k, j, i) += psi0_4 - 1;
        adm.g_dd(m, 0, 1, k, j, i) = 0.0;
        adm.g_dd(m, 0, 2, k, j, i) = 0.0;
        adm.g_dd(m, 1, 2, k, j, i) = 0.0;

        // Compute extrinsic curvature components
        // Compute s and its derivative
        Real s = den_beta;  // s = psi0^4 - alpha0^2 * v^2
        Real psi0_prime_4 = 4.0 * psi0 * psi0 * psi0 * psi0_prime;
        Real alpha0_prime_2 = 2.0 * alpha0 * alpha0_prime;
        Real s_prime = psi0_prime_4 - alpha0_prime_2 * vel * vel;
        Real ln_s_prime = s_prime / s;

        // K_xx
        Real prefactor_xx = Gamma * Gamma * B0 * x * vel / r0;
        Real expr_xx = 2.0 * alpha0_prime - (alpha0 / 2.0) * ln_s_prime;
        adm.vK_dd(m, 0, 0, k, j, i) += prefactor_xx * expr_xx;

        // K_yy and K_zz
        Real num_yy = 2.0 * Gamma * Gamma * x * vel * alpha0 * psi0_prime;
        Real den_yy = psi0 * B0 * r0;
        adm.vK_dd(m, 1, 1, k, j, i) += num_yy / den_yy;
        adm.vK_dd(m, 2, 2, k, j, i) += num_yy / den_yy;

        // K_xy and K_xz
        Real prefactor_xy = B0 * vel / r0;
        Real expr_xy = alpha0_prime - (alpha0 / 2.0) * ln_s_prime;
        adm.vK_dd(m, 0, 1, k, j, i) += prefactor_xy * y * expr_xy;
        adm.vK_dd(m, 0, 2, k, j, i) += prefactor_xy * z * expr_xy;

        // K_yz is zero due to symmetry
        adm.vK_dd(m, 1, 2, k, j, i) += 0.0;
    });
  }
  switch (ng) {
    case 2: pack->pz4c->ADMToZ4c<2>(pack, pin); break;
    case 3: pack->pz4c->ADMToZ4c<3>(pack, pin); break;
    case 4: pack->pz4c->ADMToZ4c<4>(pack, pin); break;
  }
  pack->pz4c->Z4cToADM(pack);
  pack->pz4c->GaugePreCollapsedLapse(pack, pin);
}
