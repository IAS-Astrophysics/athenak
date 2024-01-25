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
void ApplyM1Closure(TeamMember_t member, int num_points, int m, int en, int kk, int jj, int ii, DvceArray5D<Real> f, ScrArray1D<Real> f_scratch) {

  Real E = sqrt(4. * M_PI) * f(m, en * num_points + 0, kk, jj, ii);          // (0,0)
  Real Fx = -sqrt(4. * M_PI / 3.0) * f(m, en * num_points + 3, kk, jj, ii);  // (1, 1)
  Real Fy = -sqrt(4. * M_PI / 3.0) * f(m, en * num_points + 1, kk, jj, ii);  // (1, -1)
  Real Fz = sqrt(4. * M_PI / 3.0) * f(m, en * num_points + 2, kk, jj, ii);   // (1, 0)
  Real Fnorm = sqrt(Fx * Fx + Fy * Fy + Fz * Fz);

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
    Real fixed_fnorm = fmin(1.0, fnorm);

    // Flux direction
    Real nx = fx / fnorm;
    Real ny = fy / fnorm;
    Real nz = fz / fnorm;

    // Eddington factor and closure
    Real chi = (3. + 4. * fixed_fnorm * fixed_fnorm) / (5. + 2. * sqrt(4. - 3. * fixed_fnorm * fixed_fnorm));
    Real a = (1. - chi) / 2.;
    Real b = (3. * chi - 1.) / 2.;

    // P_{ij} = [a \delta_{ij} + b n_i n_j] E
    Real Pxx = (a + b * nx * nx) * E;
    Real Pyy = (a + b * ny * ny) * E;
    Real Pzz = (a + b * nz * nz) * E;
    Real Pxy = b * nx * ny * E;
    Real Pxz = b * nx * nz * E;
    Real Pyz = b * ny * nz * E;

    f_scratch(0) = f(m, en * num_points + 0, kk, jj, ii);
    f_scratch(1) = f(m, en * num_points + 1, kk, jj, ii);
    f_scratch(2) = f(m, en * num_points + 2, kk, jj, ii);
    f_scratch(3) = f(m, en * num_points + 3, kk, jj, ii);
    f_scratch(4) = sqrt(60. * M_PI) * Pxy / (4. * M_PI);            // (2, -2)
    f_scratch(5) = -sqrt(60. * M_PI) * Pyz / (4. * M_PI);           // (2, -1)
    f_scratch(6) = sqrt(5. * M_PI) * (3. * Pzz - E) / (4. * M_PI);   // (2, 0)
    f_scratch(7) = -sqrt(60. * M_PI) * Pxz / (4. * M_PI);           // (2, 1)
    f_scratch(8) = sqrt(15. * M_PI) * (Pxx - Pyy) / (4. * M_PI);    // (2, 2)
  }

}

KOKKOS_INLINE_FUNCTION
void ApplyClosure(TeamMember_t member, int num_points, int m, int en, int kk, int jj, int ii, DvceArray5D<Real> f, ScrArray1D<Real> f_scratch, bool m1_flag) {
  if (m1_flag) {
    ApplyM1Closure(member, num_points, m, en, kk, jj, ii, f, f_scratch);
  } else {
    ApplyFEMNFPNClosure(member, num_points, m, en, kk, jj, ii, f, f_scratch);
  }
}

KOKKOS_INLINE_FUNCTION
void ApplyClosureX(TeamMember_t member, int num_species, int num_energy_bins, int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii, DvceArray5D<Real> f,
                   ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2, ScrArray1D<Real> f0_scratch_p3, bool m1_flag) {
  if (m1_flag) {
    int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 1, f, f0_scratch_p1);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 2, f, f0_scratch_p2);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii + 3, f, f0_scratch_p3);
  } else {
    int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      int nuenang = IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk, jj, ii + 1);
      f0_scratch_p2(idx) = f(m, nuenang, kk, jj, ii + 2);
      f0_scratch_p3(idx) = f(m, nuenang, kk, jj, ii + 3);
    });
  }
}

KOKKOS_INLINE_FUNCTION
void ApplyClosureY(TeamMember_t member, int num_species, int num_energy_bins, int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii, DvceArray5D<Real> f,
                   ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2, ScrArray1D<Real> f0_scratch_p3, bool m1_flag) {
  if (m1_flag) {
    int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 1, ii, f, f0_scratch_p1);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 2, ii, f, f0_scratch_p2);
    ApplyM1Closure(member, num_points, m, nuen, kk, jj + 3, ii, f, f0_scratch_p3);
  } else {
    int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      int nuenang = IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk, jj + 1, ii);
      f0_scratch_p2(idx) = f(m, nuenang, kk, jj + 2, ii);
      f0_scratch_p3(idx) = f(m, nuenang, kk, jj + 3, ii);
    });
  }
}

KOKKOS_INLINE_FUNCTION
void ApplyClosureZ(TeamMember_t member, int num_species, int num_energy_bins, int num_points, int m, int nuidx, int enidx, int kk, int jj, int ii, DvceArray5D<Real> f,
                   ScrArray1D<Real> f0_scratch, ScrArray1D<Real> f0_scratch_p1, ScrArray1D<Real> f0_scratch_p2, ScrArray1D<Real> f0_scratch_p3, bool m1_flag) {
  if (m1_flag) {
    int nuen = nuidx * num_energy_bins * num_points + enidx * num_points;
    ApplyM1Closure(member, num_points, m, nuen, kk, jj, ii, f, f0_scratch);
    ApplyM1Closure(member, num_points, m, nuen, kk + 1, jj, ii, f, f0_scratch_p1);
    ApplyM1Closure(member, num_points, m, nuen, kk + 2, jj, ii, f, f0_scratch_p2);
    ApplyM1Closure(member, num_points, m, nuen, kk + 3, jj, ii, f, f0_scratch_p3);
  } else {
    int nang1 = num_points - 1;
    par_for_inner(member, 0, nang1, [&](const int idx) {
      int nuenang = IndicesUnited(nuidx, enidx, idx, num_species, num_energy_bins, num_points);
      f0_scratch(idx) = f(m, nuenang, kk, jj, ii);
      f0_scratch_p1(idx) = f(m, nuenang, kk + 1, jj, ii);
      f0_scratch_p2(idx) = f(m, nuenang, kk + 2, jj, ii);
      f0_scratch_p3(idx) = f(m, nuenang, kk + 3, jj, ii);
    });
  }
}

}
#endif //ATHENA_SRC_RADIATION_FEMN_RADIATION_FEMN_CLOSURE_HPP_
