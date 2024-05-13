//
// Created by maitraya on 12/12/23.
//

#ifndef ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_
#define ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_

#include "athena.hpp"

namespace radiationfemn {

KOKKOS_INLINE_FUNCTION
void ApplyFEMNFPNClosure(TeamMember_t member, int num_points, int m, int en, int kk, int jj, int ii, DvceArray5D<Real> f, ScrArray1D<Real> f_scratch) {

  int nang1 = num_points - 1;
  par_for_inner(member, 0, nang1, [&](const int idx) {
    f_scratch(idx) = f(m, en * num_points + idx, kk, jj, ii);
  });

}

KOKKOS_INLINE_FUNCTION
void ApplyM1Closure(TeamMember_t member, int num_points, int m, int en, int kk, int jj, int ii, DvceArray5D<Real> f, ScrArray1D<Real> f_scratch,
                    M1Closure m1_closure, ClosureFunc m1_closure_fun) {

  Real E = Kokkos::sqrt(4. * M_PI) * f(m, en * num_points + 0, kk, jj, ii);          // (0,0)
  Real Fx = -Kokkos::sqrt(4. * M_PI / 3.0) * f(m, en * num_points + 3, kk, jj, ii);  // (1, 1)
  Real Fy = -Kokkos::sqrt(4. * M_PI / 3.0) * f(m, en * num_points + 1, kk, jj, ii);  // (1, -1)
  Real Fz = Kokkos::sqrt(4. * M_PI / 3.0) * f(m, en * num_points + 2, kk, jj, ii);   // (1, 0)
  Real F2 = Fx * Fx + Fy * Fy + Fz * Fz;
  Real Fnorm = Kokkos::sqrt(F2);

  if (E < 1e-14 || Fnorm < 1e-14) {
    f_scratch(0) = f(m, en * num_points + 0, kk, jj, ii);
    f_scratch(1) = f(m, en * num_points + 1, kk, jj, ii);
    f_scratch(2) = f(m, en * num_points + 2, kk, jj, ii);
    f_scratch(3) = f(m, en * num_points + 3, kk, jj, ii);
    f_scratch(4) = 0.;
    f_scratch(5) = 0.;
    f_scratch(6) = 0.;
    f_scratch(7) = 0.;
    f_scratch(8) = 0.;
  } else {

    // Normalized flux
    Real fx = Fx / E;
    Real fy = Fy / E;
    Real fz = Fz / E;
    Real fnorm = Fnorm / E;
    Real fixed_fnorm = Kokkos::fmin(1.0, fnorm);

    // Flux direction
    Real nx = fx / fnorm;
    Real ny = fy / fnorm;
    Real nz = fz / fnorm;


    Real chi = (3. + 4. * fixed_fnorm * fixed_fnorm) / (5. + 2. * Kokkos::sqrt(4. - 3. * fixed_fnorm * fixed_fnorm));

    if (m1_closure_fun == ClosureFunc::Eddington) {
      chi = 1. / 3.;
    }
    if (m1_closure_fun == ClosureFunc::Thin) {
      chi = 1;
    }
    if (m1_closure_fun == ClosureFunc::Kershaw) {
      chi = 1; // @TODO: fix later
    }

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
      Pxx = a * E + b * Fx * Fx / Fnorm;
      Pyy = a * E + b * Fy * Fy / Fnorm;
      Pzz = a * E + b * Fz * Fz / Fnorm;
      Pxy = b * Fx * Fy / Fnorm;
      Pxz = b * Fx * Fz / Fnorm;
      Pyz = b * Fy * Fz / Fnorm;
    } else {
      Pxx = a * E + b * nx * nx * (F2 / E);
      Pyy = a * E + b * ny * ny * (F2 / E);
      Pzz = a * E + b * nz * nz * (F2 / E);
      Pxy = b * nx * ny * (F2 / E);
      Pxz = b * nx * nz * (F2 / E);
      Pyz = b * ny * nz * (F2 / E);
    }

    f_scratch(0) = f(m, en * num_points + 0, kk, jj, ii);
    f_scratch(1) = f(m, en * num_points + 1, kk, jj, ii);
    f_scratch(2) = f(m, en * num_points + 2, kk, jj, ii);
    f_scratch(3) = f(m, en * num_points + 3, kk, jj, ii);
    f_scratch(4) = Kokkos::sqrt(60. * M_PI) * Pxy / (4. * M_PI);            // (2, -2)
    f_scratch(5) = -Kokkos::sqrt(60. * M_PI) * Pyz / (4. * M_PI);           // (2, -1)
    f_scratch(6) = Kokkos::sqrt(5. * M_PI) * (3. * Pzz - E) / (4. * M_PI);   // (2, 0)
    f_scratch(7) = -Kokkos::sqrt(60. * M_PI) * Pxz / (4. * M_PI);           // (2, 1)
    f_scratch(8) = Kokkos::sqrt(15. * M_PI) * (Pxx - Pyy) / (4. * M_PI);    // (2, 2)
  }

}

KOKKOS_INLINE_FUNCTION
void ApplyClosureX(TeamMember_t member, int num_species, int num_energy_bins, int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii,
                   DvceArray5D<Real> f, ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2,
                   ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1, ScrArray1D<Real> f0_scratch_m2, bool m1_flag, M1Closure m1_closure, ClosureFunc m1_closure_fun) {
  if (m1_flag) {
    int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 1, f, f0_scratch_p1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 2, f, f0_scratch_p2, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 3, f, f0_scratch_p3, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii - 1, f, f0_scratch_m1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii - 2, f, f0_scratch_m2, m1_closure, m1_closure_fun);
  } else {
    int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      int nuenang = IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk, jj, ii + 1);
      f0_scratch_p2(idx) = f(m, nuenang, kk, jj, ii + 2);
      f0_scratch_p3(idx) = f(m, nuenang, kk, jj, ii + 3);
      f0_scratch_m1(idx) = f(m, nuenang, kk, jj, ii - 1);
      f0_scratch_m2(idx) = f(m, nuenang, kk, jj, ii - 2);
    });
  }
}

KOKKOS_INLINE_FUNCTION
void ApplyClosureY(TeamMember_t member, int num_species, int num_energy_bins, int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii,
                   DvceArray5D<Real> f, ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2,
                   ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1, ScrArray1D<Real> f0_scratch_m2, bool m1_flag, M1Closure m1_closure, ClosureFunc m1_closure_fun) {
  if (m1_flag) {
    int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 1, ii, f, f0_scratch_p1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 2, ii, f, f0_scratch_p2, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 3, ii, f, f0_scratch_p3, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj - 1, ii, f, f0_scratch_m1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj - 2, ii, f, f0_scratch_m2, m1_closure, m1_closure_fun);
  } else {
    int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      int nuenang = IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk, jj + 1, ii);
      f0_scratch_p2(idx) = f(m, nuenang, kk, jj + 2, ii);
      f0_scratch_p3(idx) = f(m, nuenang, kk, jj + 3, ii);
      f0_scratch_m1(idx) = f(m, nuenang, kk, jj - 1, ii);
      f0_scratch_m2(idx) = f(m, nuenang, kk, jj - 2, ii);
    });
  }
}

KOKKOS_INLINE_FUNCTION
void ApplyClosureZ(TeamMember_t member, int num_species, int num_energy_bins, int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii,
                   DvceArray5D<Real> f, ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2,
                   ScrArray1D<Real> f0_scratch_p3, ScrArray1D<Real> f0_scratch_m1, ScrArray1D<Real> f0_scratch_m2, bool m1_flag, M1Closure m1_closure, ClosureFunc m1_closure_fun) {
  if (m1_flag) {
    int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk + 1, jj, ii, f, f0_scratch_p1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk + 2, jj, ii, f, f0_scratch_p2, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk + 3, jj, ii, f, f0_scratch_p3, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk - 1, jj, ii, f, f0_scratch_m1, m1_closure, m1_closure_fun);
    ApplyM1Closure(member, num_points, m, nuen, kk - 2, jj, ii, f, f0_scratch_m2, m1_closure, m1_closure_fun);
  } else {
    int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      int nuenang = IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk + 1, jj, ii);
      f0_scratch_p2(idx) = f(m, nuenang, kk + 2, jj, ii);
      f0_scratch_p3(idx) = f(m, nuenang, kk + 3, jj, ii);
      f0_scratch_m1(idx) = f(m, nuenang, kk - 1, jj, ii);
      f0_scratch_m2(idx) = f(m, nuenang, kk - 2, jj, ii);
    });
  }
}

}
#endif //ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_
