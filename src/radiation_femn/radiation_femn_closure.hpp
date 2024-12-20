//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_closure.cpp
//  \brief Functions to calculate M1 closure

#ifndef RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_
#define RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_

#include "athena.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_closure.hpp"
#include "adm/adm.hpp"

namespace radiationfemn {
// Apply M1 closure
KOKKOS_INLINE_FUNCTION
void ApplyM1Closure(TeamMember_t member, int num_points, int m, int nuen, int kk,
                    int jj, int ii, DvceArray5D<Real> f, ScrArray1D<Real> f_scratch,
                    M1Closure m1_closure, ClosureFunc m1_closure_fun,
                    Real rad_E_floor = 1e-15, Real rad_eps = 1e-5) {
  Real E = Kokkos::sqrt(4. * M_PI) * f(m, nuen + 0, kk, jj, ii);         // 00
  Real Fx = -Kokkos::sqrt(4. * M_PI / 3.0) * f(m, nuen + 3, kk, jj, ii); // 11
  Real Fy = -Kokkos::sqrt(4. * M_PI / 3.0) * f(m, nuen + 1, kk, jj, ii); // 1-1
  Real Fz = Kokkos::sqrt(4. * M_PI / 3.0) * f(m, nuen + 2, kk, jj, ii);  // 10
  Real F2 = Fx * Fx + Fy * Fy + Fz * Fz;

  E = Kokkos::fmax(E, rad_E_floor);
  Real lim = E * E * (1. - rad_eps);
  if (F2 > lim) {
    Real fac = lim / F2;
    Fx = fac * Fx;
    Fy = fac * Fy;
    Fz = fac * Fz;
  }
  F2 = Fx * Fx + Fy * Fy + Fz * Fz;
  Real Fnorm = Kokkos::sqrt(F2);

  // Normalized flux
  Real fx = Fx / E;
  Real fy = Fy / E;
  Real fz = Fz / E;
  Real fnorm = Fnorm / E;
  Real fixed_fnorm = Kokkos::fmin(1.0, fnorm);

  // Flux direction
  Real nx = (fnorm > 0) ? fx / fnorm : 0;
  Real ny = (fnorm > 0) ? fy / fnorm : 0;
  Real nz = (fnorm > 0) ? fz / fnorm : 0;

  Real xi = fixed_fnorm;
  Real chi = 1.0 / 3.0 + xi * xi * (6.0 - 2.0 * xi + 6.0 * xi * xi) / 15.0;

  if (m1_closure_fun == ClosureFunc::Eddington) {
    chi = 1. / 3.;
  }
  if (m1_closure_fun == ClosureFunc::Thin) {
    chi = 1;
  }
  if (m1_closure_fun == ClosureFunc::Kershaw) {
    chi = 1.0 / 3.0 + 2.0 / 3.0 * xi * xi;
  }

  chi = Kokkos::fmin(chi, 1);
  chi = Kokkos::fmax(1. / 3., chi);
  Real a = (1. - chi) / 2.;
  Real b = (3. * chi - 1.) / 2.;

  Real Pxx = 0;
  Real Pyy = 0;
  Real Pzz = 0;
  Real Pxy = 0;
  Real Pxz = 0;
  Real Pyz = 0;

  if (m1_closure == M1Closure::Charon) {
    Pxx = a * E + b * nx * nx * E;
    Pyy = a * E + b * ny * ny * E;
    Pzz = a * E + b * nz * nz * E;
    Pxy = b * nx * ny * E;
    Pxz = b * nx * nz * E;
    Pyz = b * ny * nz * E;
  } else if (m1_closure == M1Closure::Shibata) {
    Pxx = a * E + b * nx * nx * fnorm * E;
    Pyy = a * E + b * ny * ny * fnorm * E;
    Pzz = a * E + b * nz * nz * fnorm * E;
    Pxy = b * nx * ny * fnorm * E;
    Pxz = b * nx * nz * fnorm * E;
    Pyz = b * ny * nz * fnorm * E;
  } else {
    Pxx = a * E + b * fx * fx * E;
    Pyy = a * E + b * fy * fy * E;
    Pzz = a * E + b * fz * fz * E;
    Pxy = b * fx * fy * E;
    Pxz = b * fx * fz * E;
    Pyz = b * fy * fz * E;
  }
  f_scratch(0) = E / Kokkos::sqrt(4. * M_PI);                             // (0,0)
  f_scratch(1) = -Fy / Kokkos::sqrt(4. * M_PI / 3.0);                     // (1,-1)
  f_scratch(2) = Fz / Kokkos::sqrt(4. * M_PI / 3.0);                      // (1,0)
  f_scratch(3) = -Fx / Kokkos::sqrt(4. * M_PI / 3.0);                     // (1,1)
  f_scratch(4) = Kokkos::sqrt(60. * M_PI) * Pxy / (4. * M_PI);            // (2, -2)
  f_scratch(5) = -Kokkos::sqrt(60. * M_PI) * Pyz / (4. * M_PI);           // (2, -1)
  f_scratch(6) = Kokkos::sqrt(5. * M_PI) * (3. * Pzz - E) / (4. * M_PI);  // (2, 0)
  f_scratch(7) = -Kokkos::sqrt(60. * M_PI) * Pxz / (4. * M_PI);           // (2, 1)
  f_scratch(8) = Kokkos::sqrt(15. * M_PI) * (Pxx - Pyy) / (4. * M_PI);    // (2, 2)

  f(m, nuen + 0, kk, jj, ii) = f_scratch(0);
  f(m, nuen + 1, kk, jj, ii) = f_scratch(1);
  f(m, nuen + 2, kk, jj, ii) = f_scratch(2);
  f(m, nuen + 3, kk, jj, ii) = f_scratch(3);
  f(m, nuen + 4, kk, jj, ii) = f_scratch(4);
  f(m, nuen + 5, kk, jj, ii) = f_scratch(5);
  f(m, nuen + 6, kk, jj, ii) = f_scratch(6);
  f(m, nuen + 7, kk, jj, ii) = f_scratch(7);
  f(m, nuen + 8, kk, jj, ii) = f_scratch(8);
}

// Apply M1 closure in the thin limit
KOKKOS_INLINE_FUNCTION
void ApplyM1ClosureThin(TeamMember_t member, int num_points, int m, int nuen,
                    int kk, int jj, int ii,
                    adm::ADM::ADM_vars adm,
                    AthenaTensor4d<Real, TensorSymm::NONE, 4, 2> tetr_mu_muhat0,
                    AthenaTensor4d<Real, TensorSymm::NONE, 4, 1> u_mu,
                    DvceArray5D<Real> f,
                    ScrArray1D<Real> f_scratch,
                    M1Closure m1_closure, ClosureFunc m1_closure_fun,
                    Real rad_E_floor = 1e-15, Real rad_eps = 1e-5) {

  Real g_dd[16];
  adm::SpacetimeMetric(adm.alpha(m, kk, jj, ii),
                                         adm.beta_u(m, 0, kk, jj, ii),
                                         adm.beta_u(m, 1, kk, jj, ii),
                                         adm.beta_u(m, 2, kk, jj, ii),
                                         adm.g_dd(m, 0, 0, kk, jj, ii),
                                         adm.g_dd(m, 0, 1, kk, jj, ii),
                                         adm.g_dd(m, 0, 2, kk, jj, ii),
                                         adm.g_dd(m, 1, 1, kk, jj, ii),
                                         adm.g_dd(m, 1, 2, kk, jj, ii),
                                         adm.g_dd(m, 2, 2, kk, jj, ii),
                                         g_dd);

  Real J = Kokkos::sqrt(4. * M_PI) * f(m, nuen + 0, kk, jj, ii); // 00
  Real Hx_u = -Kokkos::sqrt(4. * M_PI / 3.0) * f(m, nuen + 3, kk, jj, ii); // 11
  Real Hy_u = -Kokkos::sqrt(4. * M_PI / 3.0) * f(m, nuen + 1, kk, jj, ii); // 1-1
  Real Hz_u = Kokkos::sqrt(4. * M_PI / 3.0) * f(m, nuen + 2, kk, jj, ii); // 10
  Real H2 = Hx_u * Hx_u + Hy_u * Hy_u + Hz_u * Hz_u;
  Real u_tetr_u[4] = {1, 0, 0, 0};

  J = Kokkos::fmax(J, rad_E_floor);
  Real lim = J * J * (1. - rad_eps);
  if (H2 > lim) {
    Real fac = lim / H2;
    Hx_u = fac * Hx_u;
    Hy_u = fac * Hy_u;
    Hz_u = fac * Hz_u;
  }
  H2 = Hx_u * Hx_u + Hy_u * Hy_u + Hz_u * Hz_u;
  Real Hnorm = Kokkos::sqrt(H2);
  Real H_u[4] = {0, Hx_u, Hy_u, Hz_u};

  // Normalized flux
  Real hx_u = Hx_u / J;
  Real hy_u = Hy_u / J;
  Real hz_u = Hz_u / J;
  Real hnorm = Hnorm / J;
  Real fixed_hnorm = Kokkos::fmin(1.0, hnorm);

  Real n_u[4] = {
    1. / adm.alpha(m, kk, jj, ii),
    1. / adm.alpha(m, kk, jj, ii) * adm.beta_u(m, 0, kk, jj, ii),
    1. / adm.alpha(m, kk, jj, ii) * adm.beta_u(m, 1, kk, jj, ii),
    1. / adm.alpha(m, kk, jj, ii) * adm.beta_u(m, 2, kk, jj, ii)
  };

  Real n_tetr_u[4] = {0, 0, 0, 0};
  for (int muhat = 0; muhat < 4; muhat++) {
    for (int mu = 0; mu < 4; mu++) {
      Real sign_factor = (muhat == 0) ? -1. : +1.;
      Real tetr_muhat_mu = 0;
      for (int nu = 0; nu < 4; ++nu) {
        tetr_muhat_mu += sign_factor
          * g_dd[mu + 4 * nu]
          * tetr_mu_muhat0(m, nu, muhat, kk, jj, ii);
      }
      n_tetr_u[muhat] += tetr_muhat_mu * n_u[mu];
    }
  }

  Real nH = n_tetr_u[1] * Hx_u + n_tetr_u[2] * Hy_u + n_tetr_u[3] * Hz_u;
  Real W = u_mu(m, 0, kk, jj, ii);

  Real E = (nH - W * J) * (nH - W * J) / J;
  Real Fu = nH + E * W - W * J;
  Real F[4];
  for (int alp = 0; alp < 4; alp++) {
    F[alp] = H_u[alp] - W * E * n_u[alp] + Fu * n_tetr_u[alp] + W * W * E * u_tetr_u[alp]
      - 2. * W * Fu * u_tetr_u[alp] + Fu * Fu * u_tetr_u[alp] / E;
    F[alp] = F[alp] / (W - Fu / E);
  }

  Real T_uu[4][4];
  for (int muhat = 0; muhat < 4; muhat++) {
    for (int nuhat = 0; nuhat < 4; nuhat++) {
      T_uu[muhat][nuhat] = E * n_tetr_u[muhat] * n_tetr_u[nuhat]
        + F[muhat] * n_tetr_u[nuhat] + n_tetr_u[muhat] * F[nuhat]
        + F[muhat] * F[nuhat] / E;
    }
  }

  Real Kxx_uu = T_uu[1][1];
  Real Kyy_uu = T_uu[2][2];
  Real Kzz_uu = T_uu[3][3];
  Real Kxy_uu = T_uu[1][2];
  Real Kxz_uu = T_uu[1][3];
  Real Kyz_uu = T_uu[2][3];

  f_scratch(0) = J / Kokkos::sqrt(4. * M_PI);                             // (0,0)
  f_scratch(1) = -Hy_u / Kokkos::sqrt(4. * M_PI / 3.0);                     // (1,-1)
  f_scratch(2) = Hz_u / Kokkos::sqrt(4. * M_PI / 3.0);                      // (1,0)
  f_scratch(3) = -Hx_u / Kokkos::sqrt(4. * M_PI / 3.0);                     // (1,1)
  f_scratch(4) = Kokkos::sqrt(60. * M_PI) * Kxy_uu / (4. * M_PI);            // (2, -2)
  f_scratch(5) = -Kokkos::sqrt(60. * M_PI) * Kyz_uu / (4. * M_PI);           // (2, -1)
  f_scratch(6) = Kokkos::sqrt(5. * M_PI) * (3. * Kzz_uu - J) / (4. * M_PI);  // (2, 0)
  f_scratch(7) = -Kokkos::sqrt(60. * M_PI) * Kxz_uu / (4. * M_PI);           // (2, 1)
  f_scratch(8) = Kokkos::sqrt(15. * M_PI) * (Kxx_uu - Kyy_uu) / (4. * M_PI);    // (2, 2)

  f(m, nuen + 0, kk, jj, ii) = f_scratch(0);
  f(m, nuen + 1, kk, jj, ii) = f_scratch(1);
  f(m, nuen + 2, kk, jj, ii) = f_scratch(2);
  f(m, nuen + 3, kk, jj, ii) = f_scratch(3);
  f(m, nuen + 4, kk, jj, ii) = f_scratch(4);
  f(m, nuen + 5, kk, jj, ii) = f_scratch(5);
  f(m, nuen + 6, kk, jj, ii) = f_scratch(6);
  f(m, nuen + 7, kk, jj, ii) = f_scratch(7);
  f(m, nuen + 8, kk, jj, ii) = f_scratch(8);
}
// Apply closure along the x direction
KOKKOS_INLINE_FUNCTION
void ApplyClosureX(TeamMember_t member, int num_species, int num_energy_bins,
                   int num_points, int m, int nuidx, int enidx,
                   int kk, int jj, int ii,
                   adm::ADM::ADM_vars adm,
                   AthenaTensor4d<Real, TensorSymm::NONE, 4, 2> tetr_mu_muhat0,
                   AthenaTensor4d<Real, TensorSymm::NONE, 4, 1> u_mu,
                   DvceArray5D<Real> f, ScrArray1D<Real> f0_scratch,
                   ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2,
                   ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1,
                   ScrArray1D<Real> f0_scratch_m2, bool m1_flag,
                   M1Closure m1_closure, ClosureFunc m1_closure_fun) {
  if (m1_flag && m1_closure_fun != ClosureFunc::Thin) {
    const int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 1, f, f0_scratch_p1,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 2, f, f0_scratch_p2,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 3, f, f0_scratch_p3,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii - 1, f, f0_scratch_m1,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii - 2, f, f0_scratch_m2,
                   m1_closure, m1_closure_fun);
  } else if (m1_flag && m1_closure_fun == ClosureFunc::Thin){
    const int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1ClosureThin(member, num_points, m, nuen, kk, jj, ii, adm, tetr_mu_muhat0,
                       u_mu, f, f0_scratch, m1_closure, m1_closure_fun);
    ApplyM1ClosureThin(member, num_points, m, nuen, kk, jj, ii + 1, adm, tetr_mu_muhat0,
                       u_mu, f, f0_scratch_p1, m1_closure, m1_closure_fun);
    ApplyM1ClosureThin(member, num_points, m, nuen, kk, jj, ii + 2, adm, tetr_mu_muhat0,
                       u_mu, f, f0_scratch_p2, m1_closure, m1_closure_fun);
    ApplyM1ClosureThin(member, num_points, m, nuen, kk, jj, ii + 3, adm, tetr_mu_muhat0,
                       u_mu, f, f0_scratch_p3, m1_closure, m1_closure_fun);
    ApplyM1ClosureThin(member, num_points, m, nuen, kk, jj, ii - 1, adm, tetr_mu_muhat0,
                       u_mu, f, f0_scratch_m1, m1_closure, m1_closure_fun);
    ApplyM1ClosureThin(member, num_points, m, nuen, kk, jj, ii - 2, adm, tetr_mu_muhat0,
                       u_mu, f, f0_scratch_m2, m1_closure, m1_closure_fun);
  } else {
    const int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      const int nuenang =
          IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk, jj, ii + 1);
      f0_scratch_p2(idx) = f(m, nuenang, kk, jj, ii + 2);
      f0_scratch_p3(idx) = f(m, nuenang, kk, jj, ii + 3);
      f0_scratch_m1(idx) = f(m, nuenang, kk, jj, ii - 1);
      f0_scratch_m2(idx) = f(m, nuenang, kk, jj, ii - 2);
    });
  }
}

// Apply closure along the y direction
KOKKOS_INLINE_FUNCTION
void ApplyClosureY(TeamMember_t member, int num_species, int num_energy_bins,
                   int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii,
                   DvceArray5D<Real> f, ScrArray1D<Real> f0_scratch,
                   ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2,
                   ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1,
                   ScrArray1D<Real> f0_scratch_m2, bool m1_flag,
                   M1Closure m1_closure, ClosureFunc m1_closure_fun) {
  if (m1_flag) {
    const int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 1, ii, f, f0_scratch_p1,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 2, ii, f, f0_scratch_p2,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 3, ii, f, f0_scratch_p3,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj - 1, ii, f, f0_scratch_m1,
                   m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj - 2, ii, f, f0_scratch_m2,
                   m1_closure, m1_closure_fun);
  } else {
    const int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      const int nuenang =
          IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk, jj + 1, ii);
      f0_scratch_p2(idx) = f(m, nuenang, kk, jj + 2, ii);
      f0_scratch_p3(idx) = f(m, nuenang, kk, jj + 3, ii);
      f0_scratch_m1(idx) = f(m, nuenang, kk, jj - 1, ii);
      f0_scratch_m2(idx) = f(m, nuenang, kk, jj - 2, ii);
    });
  }
}

// Apply closure along the z direction
KOKKOS_INLINE_FUNCTION
void ApplyClosureZ(TeamMember_t member, int num_species, int num_energy_bins,
                   int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii,
                   DvceArray5D<Real> f, ScrArray1D<Real> f0_scratch,
                   ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2,
                   ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1,
                   ScrArray1D<Real> f0_scratch_m2, bool m1_flag,
                   M1Closure m1_closure, ClosureFunc m1_closure_fun) {
  if (m1_flag) {
    const int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii,
                   f, f0_scratch, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk + 1, jj, ii,
                   f, f0_scratch_p1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk + 2, jj, ii,
                   f, f0_scratch_p2, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk + 3, jj, ii,
                   f, f0_scratch_p3, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk - 1, jj, ii,
                   f, f0_scratch_m1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, +num_points, m, nuen, kk - 2, jj, ii,
                   f, f0_scratch_m2, m1_closure, m1_closure_fun);
  } else {
    const int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      const int nuenang =
          IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk + 1, jj, ii);
      f0_scratch_p2(idx) = f(m, nuenang, kk + 2, jj, ii);
      f0_scratch_p3(idx) = f(m, nuenang, kk + 3, jj, ii);
      f0_scratch_m1(idx) = f(m, nuenang, kk - 1, jj, ii);
      f0_scratch_m2(idx) = f(m, nuenang, kk - 2, jj, ii);
    });
  }
}

// Apply floor to M1 quantities
KOKKOS_INLINE_FUNCTION
void ApplyM1Floor(TeamMember_t member, ScrArray1D<Real> x,
                  const Real rad_E_floor, const Real rad_eps) {
  Real E = Kokkos::sqrt(4. * M_PI) * x(0);          // (0,0)
  Real Fx = -Kokkos::sqrt(4. * M_PI / 3.0) * x(3);  // (1, 1)
  Real Fy = -Kokkos::sqrt(4. * M_PI / 3.0) * x(1);  // (1, -1)
  Real Fz = Kokkos::sqrt(4. * M_PI / 3.0) * x(2);   // (1, 0)
  Real F2 = Fx * Fx + Fy * Fy + Fz * Fz;

  E = Kokkos::fmax(E, rad_E_floor);
  Real lim = E * E * (1. - rad_eps);
  if (F2 > lim) {
    Real fac = lim / F2;
    Fx = fac * Fx;
    Fy = fac * Fy;
    Fz = fac * Fz;
  }
  x(0) = E / Kokkos::sqrt(4. * M_PI);                             // (0,0)
  x(1) = -Fy / Kokkos::sqrt(4. * M_PI / 3.0);                     // (1,-1)
  x(2) = Fz / Kokkos::sqrt(4. * M_PI / 3.0);                      // (1,0)
  x(3) = -Fx / Kokkos::sqrt(4. * M_PI / 3.0);                     // (1,1)
}
} // namespace radiationfemn
#endif //RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_
