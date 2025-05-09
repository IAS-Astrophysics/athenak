#ifndef RADIATION_M1_FERMI_HPP
#define RADIATION_M1_FERMI_HPP

//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_fermi.hpp
//! \brief Approximates Fermi integrals following Takahashi, K., El Eid, M. F.,
//! Hillebrandt, W, A&A 67, 185 (1978)
//!        Note: adapted from Athena++

#include "athena.hpp"
#define ETA0 1.0e-3

namespace Fermi {
// FERMI 2
KOKKOS_INLINE_FUNCTION Real fermi2p(Real eta) {
  Real k21 = 3.2899;
  Real k22 = 1.8246;

  Real x = 1.0 / eta;
  return (1.0 / 3.0 + k21 * x * x) / (1.0 - Kokkos::exp(-k22 * eta));
}

KOKKOS_INLINE_FUNCTION Real fermi2m(Real eta) {
  Real a21 = 0.1092;
  Real a22 = 0.8908;

  return 2.0 / (1.0 + a21 * Kokkos::exp(a22 * eta));
}

KOKKOS_INLINE_FUNCTION Real fermi2(Real eta) {
  return (eta > ETA0) ? (eta * eta * eta) * fermi2p(eta) : exp(eta) * fermi2m(eta);
}

// FERMI 3
KOKKOS_INLINE_FUNCTION Real fermi3p(Real eta) {
  Real k31 = 4.9348;
  Real k32 = 11.3644;
  Real k33 = 1.9039;

  Real x = 1.0 / eta;
  Real x2 = x * x;
  Real x4 = x2 * x2;

  return (0.25 + k31 * x2 + k32 * x4) / (1.0 + Kokkos::exp(-k33 * eta));
}

KOKKOS_INLINE_FUNCTION Real fermi3m(Real eta) {
  Real a31 = 0.0559;
  Real a32 = 0.9069;

  return 6.0 / (1.0 + a31 * Kokkos::exp(a32 * eta));
}

KOKKOS_INLINE_FUNCTION Real fermi3(Real eta) {
  return (eta > ETA0) ? (eta * eta * eta * eta) * fermi3p(eta) : exp(eta) * fermi3m(eta);
}

}  // namespace Fermi

#undef ETA0

#endif