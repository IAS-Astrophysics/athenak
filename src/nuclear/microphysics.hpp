#ifndef NUCLEAR_MICROPHYSICS_HPP_
#define NUCLEAR_MICROPHYSICS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file microphysics.hpp
//! \brief Defines nuclear::Thermo and nuclear::burn<EOSType>(), filling in the
//! remaining thermodynamic state from a given pair of inputs selected by eos_input_t.

#include <cmath>
#include <type_traits>

#include "athena.hpp"
#include "units/units.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "nuclear/network/network.hpp"

namespace nuclear {

// Physical constants (cgs). `inline constexpr` (C++17 inline variables) gives
// this header a single definition across every translation unit that
// includes it, avoiding an ODR/multiple-definition link error.
inline constexpr Real kB = units::Units::k_boltzmann_cgs;
inline constexpr Real m_u = units::Units::atomic_mass_unit_cgs;
inline constexpr Real hbar = units::Units::red_plank_constant_cgs;

//! \struct Thermo
//! \brief Thermodynamic state used by burn(). Which of rho/t/p/e/s are
//! "given" vs "derived" depends on which eos_input_t tag burn() is called with.
struct Thermo {
  Real rho{};
  Real t{};
  Real mu{};
  Real xn[NSPECIES]{};
  Real p{};
  Real e{};
  Real s{};
  Real cv{};
  Real abar{};
  Real zbar{};
  Real dedA{};
  Real dedZ{};
};

// --- C++17 "does this type have member X" detection idiom ---
// (std::void_t-based SFINAE; the C++17-compatible equivalent of a C++20
// `concept has_pressure = requires(T t) { t.p; };`.) Each has_X<T> is a
// std::true_type/std::false_type; has_X_v<T> is the bool to use in
// `if constexpr`, matching how the concept itself used to be used directly.
template <typename T, typename = void>
struct has_energy : std::false_type {};
template <typename T>
struct has_energy<T, std::void_t<decltype(std::declval<T>().e)>> : std::true_type {};
template <typename T>
inline constexpr bool has_energy_v = has_energy<T>::value;

template <typename T, typename = void>
struct has_pressure : std::false_type {};
template <typename T>
struct has_pressure<T, std::void_t<decltype(std::declval<T>().p)>> : std::true_type {};
template <typename T>
inline constexpr bool has_pressure_v = has_pressure<T>::value;

template <typename T, typename = void>
struct has_entropy : std::false_type {};
template <typename T>
struct has_entropy<T, std::void_t<decltype(std::declval<T>().s)>> : std::true_type {};
template <typename T>
inline constexpr bool has_entropy_v = has_entropy<T>::value;

template <typename T, typename = void>
struct has_xn : std::false_type {};
template <typename T>
struct has_xn<T, std::void_t<decltype(std::declval<T>().xn)>> : std::true_type {};
template <typename T>
inline constexpr bool has_xn_v = has_xn<T>::value;

template <typename T, typename = void>
struct has_cv : std::false_type {};
template <typename T>
struct has_cv<T, std::void_t<decltype(std::declval<T>().cv)>> : std::true_type {};
template <typename T>
inline constexpr bool has_cv_v = has_cv<T>::value;

template <typename T, typename = void>
struct has_abar : std::false_type {};
template <typename T>
struct has_abar<T, std::void_t<decltype(std::declval<T>().abar)>> : std::true_type {};
template <typename T>
inline constexpr bool has_abar_v = has_abar<T>::value;

template <typename T, typename = void>
struct has_zbar : std::false_type {};
template <typename T>
struct has_zbar<T, std::void_t<decltype(std::declval<T>().zbar)>> : std::true_type {};
template <typename T>
inline constexpr bool has_zbar_v = has_zbar<T>::value;

template <typename T, typename = void>
struct has_dedA : std::false_type {};
template <typename T>
struct has_dedA<T, std::void_t<decltype(std::declval<T>().dedA)>> : std::true_type {};
template <typename T>
inline constexpr bool has_dedA_v = has_dedA<T>::value;

template <typename T, typename = void>
struct has_dedZ : std::false_type {};
template <typename T>
struct has_dedZ<T, std::void_t<decltype(std::declval<T>().dedZ)>> : std::true_type {};
template <typename T>
inline constexpr bool has_dedZ_v = has_dedZ<T>::value;

enum eos_input_t : int {
  eos_input_rt = 0,
  eos_input_ps,
  eos_input_re,
  eos_input_pt,
};

//! \fn burn
//! \brief Given a thermodynamic input pair (selected at runtime by `input`)
//! and composition, fills in the remaining fields of `state` using `eos`.
//! Always resolves to a canonical (rho, t, xn[]) state first (inverting the
//! given pair if needed), then runs the shared rho/T-based computation for
//! whichever of p/e/s are present on the state type.
//!
//! `eos` must already have SetNSpecies() called on it -- construct it once on
//! the host (it is not safe to default-construct an EOS<> object on-device;
//! its constructor is not KOKKOS_INLINE_FUNCTION) and capture it by value into
//! whatever Kokkos kernel calls burn() per cell. burn() itself only ever calls
//! eos.SetBaryonMass()/.Get*(), all of which are already KOKKOS_INLINE_FUNCTION.
//!
//! Everything here is plain cgs (matching kB/m_u/hbar above, and `eos`'s
//! default-constructed IdealGas state): never call eos.SetCodeUnitSystem() or
//! eos.SetEOSUnitSystem() anywhere -- doing so would break the invariant this
//! file relies on, namely that EOS<>::Get*()'s internal unit-conversion
//! factors are all identity because code_units == eos_units (IdealGas's
//! default). That's what lets raw cgs values flow through Get*() unconverted.
template <typename EOSType>
KOKKOS_INLINE_FUNCTION
void burn(eos_input_t input, Thermo &state,
          Primitive::EOS<EOSType, Primitive::ResetFloor> &eos) {
 
  if constexpr (has_abar_v<Thermo>) {
    // abar (mean atomic mass number): 1/Abar = Sum X_n/A_n -- same formula
    // as mu below (equal to it, under this neutral-gas model), but kept as
    // its own field since it's a pure composition quantity.
    Real abar_inv = 0.0;
    for (int n=0; n<NSPECIES; n++) {
      abar_inv += state.xn[n] * aion_inv(n);
    }
   state.abar = 1.0 / abar_inv;
  }


  if constexpr (has_zbar_v<Thermo>) {
    // Zbar/Abar = Sum X_n*Z_n/A_n
    Real zbar = 0.0;
    for (int n=0; n<NSPECIES; n++) {
      zbar += state.xn[n] * zion(n) * aion_inv(n);
    }
   state.zbar = state.abar * zbar;
  }

  // Mean molecular weight (neutral gas): 1/mu = Sum X_n / A_n
  if constexpr (has_xn_v<Thermo>) {
    Real mu_inv = 0.0;
    for (int n = 0; n < NSPECIES; n++) {
      mu_inv += state.xn[n] * aion_inv(n);
    }
    state.mu = 1.0 / mu_inv;
  }

  Real mb = state.mu * m_u;
  eos.SetBaryonMass(mb);

  switch (input) {
    case eos_input_rt:
      // rho, t already given directly -- nothing to invert.
      break;

    case eos_input_ps: {
      // Invert (p, s) -> (rho, t) [Sackur-Tetrode monatomic ideal gas]
      Real fac_rho = pow(1.0 / (2.0 * M_PI * hbar * hbar), 0.6);
      state.rho = pow(state.p, 0.6) * pow(mb, 1.6) * fac_rho *
                  exp(1.0 - 2.0 * mb * state.s / (5.0 * kB));

      Real fac_t = pow(2.0 * M_PI * hbar * hbar / mb, 0.6);
      state.t = pow(state.p, 0.4) * fac_t *
                exp(2.0 * mb * state.s / (5.0* kB) - 1.0) / kB;
      break;
    }

    case eos_input_re: {
      // Invert specific internal energy -> t, at fixed rho. GetTemperatureFromE
      // expects an energy DENSITY (rest-mass included), not specific internal
      // energy: e_density = rho*(1 + specific internal energy).
      Real n = state.rho / mb;
      Real e_density = state.rho * (1.0 + state.e);
      state.t = eos.GetTemperatureFromE(n, e_density, state.xn) / kB;
      break;
    }

    case eos_input_pt: {
      // Invert (p, t) -> rho [ideal gas law]
      state.rho = mb * state.p / (kB * state.t);
      break;
    }
  }

  Real n = state.rho / mb;
  Real t_energy = kB * state.t;

  if constexpr (has_cv_v<Thermo>) {
    // Specific heat (de/dT at constant rho and composition), monatomic ideal gas:
    // cv = (3/2) kB / (mu m_u)  [erg/g/K]
    state.cv = 1.5 * kB / (state.mu * m_u);
  }



  if constexpr (has_energy_v<Thermo>) {
    state.e = eos.GetSpecificInternalEnergy(n, t_energy, state.xn);
  }

  if constexpr (has_entropy_v<Thermo>) {
    Real rhoinv = 1.0 / state.rho;
    Real fac = pow(1.0 / (2.0 * M_PI * hbar * hbar), 1.5);
    state.s = kB / mb * (2.5 + log(pow(mb, 2.5) * rhoinv * pow(t_energy, 1.5) * fac));
  }

  if constexpr (has_pressure_v<Thermo>) {
    state.p = eos.GetPressure(n, t_energy, state.xn);
  }


  if constexpr (has_dedA_v<Thermo>) {
    // Monatomic ideal gas: e = 1.5*kB*T/(Abar*m_u) -> de/dAbar = -1.5*kB*T/(Abar^2*m_u)
    state.dedA = -1.5 * kB * state.t / (state.abar * state.abar * m_u);
  }


  if constexpr (has_dedZ_v<Thermo>) {
    state.dedZ = 0.0;
  }


}

}  // namespace nuclear

#endif  // NUCLEAR_MICROPHYSICS_HPP_
