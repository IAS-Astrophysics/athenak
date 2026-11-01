#ifndef NUCLEAR_MICROPHYSICS_HPP_
#define NUCLEAR_MICROPHYSICS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file microphysics.hpp
//! \brief Defines nuclear::Thermo and nuclear::burn() for the Helmholtz EOS, filling
//! in the remaining thermodynamic state from a given pair of inputs selected by
//! eos_input_t. The IdealGas overload is in nuclear/microphysics_idealgas.hpp.

#include <cmath>
#include <limits>
#include <type_traits>

#include "athena.hpp"
#include "units/units.hpp"
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/eos_helmholtz.hpp"
#include "nuclear/network/network.hpp"

namespace nuclear {

// Physical constants (cgs). `inline constexpr` (C++17 inline variables) gives
// this header a single definition across every translation unit that
// includes it, avoiding an ODR/multiple-definition link error.
inline constexpr Real kB = units::Units::k_boltzmann_cgs;
inline constexpr Real m_u = units::Units::atomic_mass_unit_cgs;
inline constexpr Real hbar = units::Units::red_plank_constant_cgs;
inline constexpr Real c_light = units::Units::speed_of_light_cgs;
inline constexpr Real MeV_erg = 1.6021766208e-6;  // same value as Primitive::MakeCGS()
// Helmholtz EOS number densities are in fm^-3 (nuclear units): 1 fm^-3 = 1e39 cm^-3
inline constexpr Real cm3_per_fm3 = 1.0e39;

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
  Real dedX[NSPECIES]{};
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
struct has_dedX : std::false_type {};
template <typename T>
struct has_dedX<T, std::void_t<decltype(std::declval<T>().dedX)>> : std::true_type {};
template <typename T>
inline constexpr bool has_dedX_v = has_dedX<T>::value;

enum eos_input_t : int {
  eos_input_rt = 0,
  eos_input_ps,
  eos_input_re,
  eos_input_pt,
  eos_input_rp,
};

//! \fn composition
//! \brief Fills the composition-only fields of `state` (abar, zbar, mu) from
//! state.xn[]. Shared by every burn() overload.
KOKKOS_INLINE_FUNCTION
void composition(Thermo &state) {
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
}

//! \fn burn
//! \brief Helmholtz EOS. Given a thermodynamic input pair (selected at runtime
//! by `input`) and composition, fills in the remaining fields of `state` using
//! `eos`. Always resolves to a canonical (rho, t, xn[]) state first (inverting
//! the given pair if needed), then runs the shared rho/T-based computation for
//! whichever of p/e/s are present on the state type.
//!
//! Everything here is plain cgs, T in K (matching kB/m_u/c_light above). `eos`
//! must be a shallow copy of the DynGRMHD EOS (it shares the table) with
//! SetCodeUnitSystem(Primitive::MakeCGS()) called on it, made once on the host
//! and captured by value into whatever Kokkos kernel calls burn() per cell:
//!   auto eos = static_cast<dyngr::DynGRMHDPS<Primitive::EOSHelmholtz,
//!                 Primitive::ResetFloor>*>(pmbp->pdyngr)->eos.ps.GetEOS();
//!   eos.SetCodeUnitSystem(Primitive::MakeCGS());
//! The wrapper then converts to and from the table's nuclear units. The number
//! density n it takes is in EOS units, n = rho/GetBaryonMass().
//!
//! Input pairs: rt (rho, t), ps (p, s), re (rho, e), pt (p, t), rp (rho, p).
//! The (p, s) and (p, t) inversions give NaN if p is not reachable, or if the
//! isentrope leaves the table; the caller has to check for it.
//!
//! The IdealGas overload lives in nuclear/microphysics_idealgas.hpp.
KOKKOS_INLINE_FUNCTION
void burn(eos_input_t input, Thermo &state,
          const Primitive::EOS<Primitive::EOSHelmholtz, Primitive::ResetFloor> &eos) {
  composition(state);

  // Baryon mass [g]: a fixed convention read from the table (it sets the
  // rest-mass/eps split, so it must not be changed), which also carries the
  // fm^-3 -> cm^-3 factor.
  const Real mb = eos.GetBaryonMass();

  switch (input) {
    case eos_input_rt:
      // rho, t already given directly -- nothing to invert.
      break;

    case eos_input_ps: {
      // Table inversion on the isentrope; gives t at the solution too.
      state.rho = eos.GetDensityFromPressureEntropy(state.p, state.s, state.xn,
                                                    &state.t) * mb;
      break;
    }

    case eos_input_re: {
      // Invert specific internal energy -> t, at fixed rho. Takes eps directly, so
      // the rest-mass energy density rho*c^2 is never formed (no cancellation).
      Real n = state.rho / mb;
      state.t = eos.GetTemperatureFromSpecificInternalEnergy(n, state.e, state.xn);
      break;
    }

    case eos_input_pt: {
      // Table inversion at fixed t
      state.rho = eos.GetDensityFromPressureTemperature(state.p, state.t,
                                                        state.xn) * mb;
      break;
    }

    case eos_input_rp: {
      // Invert pressure -> t, at fixed rho (as DynGRMHD gets T from its primitives)
      Real n = state.rho / mb;
      state.t = eos.GetTemperatureFromP(n, state.p, state.xn);
      break;
    }
  }

  Real n = state.rho / mb;

  if constexpr (has_cv_v<Thermo>) {
    state.cv = eos.GetSpecificHeat(n, state.t, state.xn);
  }

  if constexpr (has_energy_v<Thermo>) {
    state.e = eos.GetSpecificInternalEnergy(n, state.t, state.xn);
  }

  if constexpr (has_entropy_v<Thermo>) {
    state.s = eos.GetEntropy(n, state.t, state.xn);
  }

  if constexpr (has_pressure_v<Thermo>) {
    state.p = eos.GetPressure(n, state.t, state.xn);
  }

  // de/dX_n at fixed (rho, T): only the nuclear binding term, which dominates
  // (~1e18 vs ~1e16 erg/g for the ion and electron thermal terms at 1e9 K).
  // The EOS's rest mass per baryon is mbar = m_u + Sum X_i dm_i/A_i, and eps
  // carries (mbar/mb - 1)*c^2, so de/dX_n = dm_n*c^2/(A_n*mb), with dm_n the
  // atomic mass excess. mb here is the baryon mass in g; GetBaryonMass() also
  // carries the fm^-3 -> cm^-3 density factor.
  if constexpr (has_dedX_v<Thermo>) {
    const Real mb_g = mb / cm3_per_fm3;
    for(int n=0; n<NSPECIES; n++) {
      state.dedX[n] = dmass_ions(n) * MeV_erg * aion_inv(n) / mb_g;
    }
  }
}

}  // namespace nuclear

#endif  // NUCLEAR_MICROPHYSICS_HPP_
