#ifndef EOS_PRIMITIVE_SOLVER_EOS_TRANSITION_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_TRANSITION_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_transition.hpp
//  \brief Device port of EOSTransition: a density/temperature blend between the
//         CompOSE (NSE) table and the Helmholtz (out-of-NSE) thermal EOS.
//
//  The two sub-EOSs are held by value; their tables live in Kokkos Views so the
//  whole policy is device-copyable. All log/exp math internal to the blend uses
//  natural logs; compose grid nodes and table values are decoded via the
//  compose LogPolicy (log2_/exp2_).

#include <cassert>
#include <string>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"
#include "logs.hpp"
#include "numtools_root.hpp"
#include "eos_compose.hpp"
#include "eos_helmholtz.hpp"

namespace Primitive {

template<typename LogPolicy>
class EOSTransition : public EOSPolicyInterface, public LogPolicy,
                      public SupportsEntropy, public SupportsChemicalPotentials {
 private:
  using LogPolicy::log2_;
  using LogPolicy::exp2_;

 protected:
  EOSTransition() {
    n_species = 7;
    eos_units = MakeNuclear();
    m_initialized = false;

    min_Y[SCYE] = 0.0;  min_Y[SCXN] = 0.0;  min_Y[SCXP] = 0.0;
    min_Y[SCXA] = 0.0;  min_Y[SCXH] = 0.0;  min_Y[SCAH] = 1.0;  min_Y[SCEB] = 0.0;
    max_Y[SCYE] = 1.0;  max_Y[SCXN] = 1.0;  max_Y[SCXP] = 1.0;
    max_Y[SCXA] = 1.0;  max_Y[SCXH] = 1.0;  max_Y[SCAH] = 300.0; max_Y[SCEB] = 1e-2;

    m_min_h          = std::numeric_limits<Real>::max();
    m_trans_T_width  = std::numeric_limits<Real>::quiet_NaN();
    m_trans_ln_width = std::numeric_limits<Real>::quiet_NaN();
    trans_T_start    = std::numeric_limits<Real>::quiet_NaN();
    trans_T_end      = std::numeric_limits<Real>::quiet_NaN();
    trans_ln_start   = std::numeric_limits<Real>::quiet_NaN();
    trans_ln_end     = std::numeric_limits<Real>::quiet_NaN();
    m_helm_n_max     = std::numeric_limits<Real>::quiet_NaN();
    m_helm_T_max     = std::numeric_limits<Real>::quiet_NaN();
    m_helm_n_ramp_dec = 1.0;
    m_helm_T_ramp_dec = 0.5;
    m_ln_n_h0 = std::numeric_limits<Real>::quiet_NaN();
    m_ln_T_h0 = std::numeric_limits<Real>::quiet_NaN();
    m_id_ln_n_ramp = std::numeric_limits<Real>::quiet_NaN();
    m_id_ln_T_ramp = std::numeric_limits<Real>::quiet_NaN();
    root.iterations = 500;
  }

  //--------------------------------------------------------------------------
  // Transition weight: 1 => pure compose (NSE), 0 => pure Helmholtz.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real TransitionFactor(Real n, Real T) const {
    return TransitionFactor(n, T, Kokkos::log(n), Kokkos::log(T));
  }
  KOKKOS_INLINE_FUNCTION Real TransitionFactor(Real n, Real T, Real ln, Real lT) const {
    if ((n > m_helm_n_max) || (T > m_helm_T_max)) {
      return 1.0;
    }
    if ((n < compose_eos.min_n) || (T < compose_eos.min_T)) {
      return 0.0;
    }
    Real u = Kokkos::fmin(1.0, Kokkos::fmax(0.0, (T - trans_T_end)/m_trans_T_width));
    u *= Kokkos::fmin(1.0, Kokkos::fmax(0.0, (ln - trans_ln_end)/m_trans_ln_width));
    Real v_n = Kokkos::fmin(1.0, Kokkos::fmax(0.0, (ln - m_ln_n_h0)*m_id_ln_n_ramp));
    Real v_T = Kokkos::fmin(1.0, Kokkos::fmax(0.0, (lT - m_ln_T_h0)*m_id_ln_T_ramp));
    return 1.0 - (1.0 - u)*(1.0 - v_n)*(1.0 - v_T);
  }

  //--------------------------------------------------------------------------
  // Fast path: for n above the Helmholtz cutoff the weight is identically 1,
  // so delegate straight to the compose table with a clamped Ye.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION bool InteriorYq(Real n, const Real *Y, Real &yq) const {
    if (n <= m_helm_n_max) {
      return false;
    }
    yq = Kokkos::fmax(min_Y[SCYE], Kokkos::fmin(Y[SCYE], max_Y[SCYE]));
    return true;
  }

  //--------------------------------------------------------------------------
  // Clamp + normalize the composition, remapping unphysical states to free
  // nucleons. Returns the pre-normalization mass-fraction sum.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real SanitizeMassFractions(const Real *Y, Real *Y_norm) const {
    for (int i = 0; i < SCNVAR; ++i) {
      Y_norm[i] = Kokkos::fmax(min_Y[i], Kokkos::fmin(Y[i], max_Y[i]));
    }
    Real Xsum = Y_norm[SCXN] + Y_norm[SCXP] + Y_norm[SCXA] + Y_norm[SCXH];
    if (Xsum <= 0.0) {
      Y_norm[SCYE] = 0.5;
      Y_norm[SCXN] = 0.5;
      Y_norm[SCXP] = 0.5;
      Y_norm[SCXA] = 0.0;
      Y_norm[SCXH] = 0.0;
      return 1.0;
    }
    Y_norm[SCXN] /= Xsum;
    Y_norm[SCXP] /= Xsum;
    Y_norm[SCXA] /= Xsum;
    Y_norm[SCXH] /= Xsum;
    const Real q_h = Y_norm[SCYE] - Y_norm[SCXP] - 0.5*Y_norm[SCXA];
    if (q_h < -0.05 || q_h > 0.5*Y_norm[SCXH] + 0.05) {
      Y_norm[SCXN] = 1.0 - Y_norm[SCYE];
      Y_norm[SCXP] = Y_norm[SCYE];
      Y_norm[SCXA] = 0.0;
      Y_norm[SCXH] = 0.0;
    }
    return Xsum;
  }

  //--------------------------------------------------------------------------
  // Thermodynamic quantities (blended).
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.Pressure(n, T, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.Pressure(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.Pressure(n, T, Yn); }
    return helmholtz_eos.Pressure(n, T, Yn)*(1 - w) + compose_eos.Pressure(n, T, Yn)*w;
  }

  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.Entropy(n, T, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.Entropy(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.Entropy(n, T, Yn); }
    return helmholtz_eos.Entropy(n, T, Yn)*(1 - w) + compose_eos.Entropy(n, T, Yn)*w;
  }

  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T,
                                                     const Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.SpecificInternalEnergy(n, T, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.SpecificInternalEnergy(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.SpecificInternalEnergy(n, T, Yn); }
    return helmholtz_eos.SpecificInternalEnergy(n, T, Yn)*(1 - w) +
           compose_eos.SpecificInternalEnergy(n, T, Yn)*w;
  }

  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    return (SpecificInternalEnergy(n, T, Y) + 1.0)*mb*n;
  }

  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    return (Pressure(n, T, Y) + Energy(n, T, Y))/n;
  }

  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.SoundSpeed(n, T, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.SoundSpeed(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.SoundSpeed(n, T, Yn); }
    return helmholtz_eos.SoundSpeed(n, T, Yn)*(1 - w) +
           compose_eos.SoundSpeed(n, T, Yn)*w;
  }

  KOKKOS_INLINE_FUNCTION Real BaryonChemicalPotential(Real n, Real T, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.BaryonChemicalPotential(n, T, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.BaryonChemicalPotential(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.BaryonChemicalPotential(n, T, Yn); }
    return helmholtz_eos.BaryonChemicalPotential(n, T, Yn)*(1 - w) +
           compose_eos.BaryonChemicalPotential(n, T, Yn)*w;
  }

  KOKKOS_INLINE_FUNCTION Real ChargeChemicalPotential(Real n, Real T, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.ChargeChemicalPotential(n, T, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.ChargeChemicalPotential(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.ChargeChemicalPotential(n, T, Yn); }
    return helmholtz_eos.ChargeChemicalPotential(n, T, Yn)*(1 - w) +
           compose_eos.ChargeChemicalPotential(n, T, Yn)*w;
  }

  KOKKOS_INLINE_FUNCTION Real ElectronLeptonChemicalPotential(Real n, Real T,
                                                              Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) {
      return compose_eos.ElectronLeptonChemicalPotential(n, T, &yq);
    }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    Real w = TransitionFactor(n, T);
    if (w == 1.0) { return compose_eos.ElectronLeptonChemicalPotential(n, T, Yn); }
    if (w == 0.0) { return helmholtz_eos.ElectronLeptonChemicalPotential(n, T, Yn); }
    return helmholtz_eos.ElectronLeptonChemicalPotential(n, T, Yn)*(1 - w) +
           compose_eos.ElectronLeptonChemicalPotential(n, T, Yn)*w;
  }

  //--------------------------------------------------------------------------
  // Temperature inversions.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    if (n > m_helm_n_max) { return compose_eos.TemperatureFromE(n, e, Y); }
    return TemperatureFromEps(n, e/(mb*n) - 1.0, Y);
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromEps(Real n, Real eps, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.TemperatureFromEps(n, eps, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    if (n < compose_eos.min_n) { return helmholtz_eos.TemperatureFromEps(n, eps, Yn); }
    Real eps_min = MinimumSpecificInternalEnergy(n, Yn);
    Real eps_max = MaximumSpecificInternalEnergy(n, Yn);
    if (eps <= eps_min) {
      return (Kokkos::log(n) >= m_ln_n_h0) ? compose_eos.min_T : min_T;
    }
    if (eps >= eps_max) { return max_T; }

    if (Kokkos::log(n) < m_ln_n_h0) {
      Real T_h = helmholtz_eos.TemperatureFromEps(n, eps, Yn);
      if (TransitionFactor(n, T_h) == 0.0) { return T_h; }
    }
    Real T_b = temperature_from_var_trans(EOSCompOSE<LogPolicy>::ECLOGE,
                                          Kokkos::log(n*mb*(1 + eps)), n, Yn);
    if (!Kokkos::isnan(T_b)) { return T_b; }
    Real T_c = compose_eos.TemperatureFromEps(n, eps, Yn);
    return T_c;
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    Real yq;
    if (InteriorYq(n, Y, yq)) { return compose_eos.TemperatureFromP(n, p, &yq); }
    Real Yn[SCNVAR];
    SanitizeMassFractions(Y, Yn);
    if (n < compose_eos.min_n) { return helmholtz_eos.TemperatureFromP(n, p, Yn); }
    Real p_min = MinimumPressure(n, Yn);
    Real p_max = MaximumPressure(n, Yn);
    if (p <= p_min) {
      return (Kokkos::log(n) >= m_ln_n_h0) ? compose_eos.min_T : min_T;
    }
    if (p >= p_max) { return max_T; }

    if (Kokkos::log(n) < m_ln_n_h0) {
      Real T_h = helmholtz_eos.TemperatureFromP(n, p, Yn);
      if (TransitionFactor(n, T_h) == 0.0) { return T_h; }
    }
    Real T_b = temperature_from_var_trans(EOSCompOSE<LogPolicy>::ECLOGP,
                                          Kokkos::log(p), n, Yn);
    if (!Kokkos::isnan(T_b)) { return T_b; }
    return compose_eos.TemperatureFromP(n, p, Yn);
  }

  //--------------------------------------------------------------------------
  // Extrema (used by the temperature inversions and the error policy).
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const { return m_min_h; }

  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return (Kokkos::log(n) >= m_ln_n_h0) ? Pressure(n, compose_eos.min_T, Y)
                                         : Pressure(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    return Pressure(n, max_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MinimumEntropy(Real n, Real *Y) const {
    return (Kokkos::log(n) >= m_ln_n_h0) ? Entropy(n, compose_eos.min_T, Y)
                                         : Entropy(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumEntropy(Real n, Real *Y) const {
    return Entropy(n, max_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MinimumSpecificInternalEnergy(Real n, Real *Y) const {
    return (Kokkos::log(n) >= m_ln_n_h0) ? SpecificInternalEnergy(n, compose_eos.min_T, Y)
                                         : SpecificInternalEnergy(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumSpecificInternalEnergy(Real n, Real *Y) const {
    return SpecificInternalEnergy(n, max_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return (MinimumSpecificInternalEnergy(n, Y) + 1.0)*mb*n;
  }
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    return (MaximumSpecificInternalEnergy(n, Y) + 1.0)*mb*n;
  }

  //--------------------------------------------------------------------------
  // Neutrino equilibrium: only meaningful in the NSE interior, so delegate.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION int BetaEquilibriumTrapped(Real n, Real e, Real *Yl,
      Real &T_eq, Real *Y_eq, Real T_guess, Real *Y_guess) const {
    return compose_eos.BetaEquilibriumTrapped(n, e, Yl, T_eq, Y_eq, T_guess, Y_guess);
  }
  KOKKOS_INLINE_FUNCTION void TrappedNeutrinos(Real n, Real T, Real *Y,
      Real n_nu[3], Real e_nu[3]) const {
    compose_eos.TrappedNeutrinos(n, T, Y, n_nu, e_nu);
  }

 public:
  //--------------------------------------------------------------------------
  // Public wrappers so external drivers (e.g. RHINE) can query the blend
  // weight and sanitize a composition without touching the EOS internals.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real GetTransitionFactor(Real n, Real T) const {
    return TransitionFactor(n, T);
  }
  KOKKOS_INLINE_FUNCTION Real GetSanitizedMassFractions(const Real *Y,
                                                        Real *Y_norm) const {
    return SanitizeMassFractions(Y, Y_norm);
  }
  //! Raw baryon mass in EOS (nuclear, MeV) units, for per-baryon energetics.
  KOKKOS_INLINE_FUNCTION Real GetBaryonMassMeV() const { return mb; }

  //--------------------------------------------------------------------------
  // Composition accessors (NSE composition where w==1, advected otherwise).
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real FrYn(Real n, Real T, Real *Y) const {
    return (TransitionFactor(n, T) == 1.0) ? compose_eos.FrYn(n, T, Y) : Y[SCXN];
  }
  KOKKOS_INLINE_FUNCTION Real FrYp(Real n, Real T, Real *Y) const {
    return (TransitionFactor(n, T) == 1.0) ? compose_eos.FrYp(n, T, Y) : Y[SCXP];
  }
  KOKKOS_INLINE_FUNCTION Real FrXa(Real n, Real T, Real *Y) const {
    return (TransitionFactor(n, T) == 1.0) ? compose_eos.FrXa(n, T, Y) : Y[SCXA];
  }
  KOKKOS_INLINE_FUNCTION Real FrXh(Real n, Real T, Real *Y) const {
    return (TransitionFactor(n, T) == 1.0) ? compose_eos.FrXh(n, T, Y) : Y[SCXH];
  }
  KOKKOS_INLINE_FUNCTION Real AN(Real n, Real T, Real *Y) const {
    return (TransitionFactor(n, T) == 1.0) ? compose_eos.AN(n, T, Y) : Y[SCAH];
  }
  KOKKOS_INLINE_FUNCTION Real ZN(Real n, Real T, Real *Y) const {
    if (TransitionFactor(n, T) == 1.0) { return compose_eos.ZN(n, T, Y); }
    return (Y[SCXH] > 0) ? (Y[SCYE] - Y[SCXP] - Y[SCXA]/2)/Y[SCXH]*Y[SCAH] : 0.0;
  }
  KOKKOS_INLINE_FUNCTION Real InteractionPotentialDifference(Real n, Real T,
                                                             Real *Y) const {
    Real w = TransitionFactor(n, T);
    if (w == 0.0) { return 0.0; }
    return w*compose_eos.InteractionPotentialDifference(n, T, Y);
  }

  /// NSE binding energy per baryon (used by the RHINE coupling).
  KOKKOS_INLINE_FUNCTION Real GetNSEBindingEnergy(Real n, Real T, Real *Y) const {
    if (n > helmholtz_eos.max_n || T > helmholtz_eos.max_T) {
      return 0.0;
    } else if (n < compose_eos.min_n || T < compose_eos.min_T) {
      return Y[SCEB];
    }
    Real Y_NSE[SCNVAR];
    Y_NSE[SCYE] = Y[SCYE];
    Y_NSE[SCXN] = compose_eos.FrYn(n, T, Y);
    Y_NSE[SCXP] = compose_eos.FrYp(n, T, Y);
    Y_NSE[SCXA] = compose_eos.FrXa(n, T, Y);
    Y_NSE[SCXH] = compose_eos.FrXh(n, T, Y);
    Y_NSE[SCAH] = compose_eos.AN(n, T, Y);
    Y_NSE[SCEB] = 0.0;

    Real eps_helm = helmholtz_eos.SpecificInternalEnergy(n, T, Y_NSE);
    Real eps_comp = compose_eos.SpecificInternalEnergy(n, T, Y);

    Real Zh = (Y_NSE[SCXH] > 0)
              ? (Y_NSE[SCYE] - Y_NSE[SCXP] - Y_NSE[SCXA]/2)/Y_NSE[SCXH]*Y_NSE[SCAH]
              : 0.0;
    Real Ah = Y_NSE[SCAH];
    Real const mH   = mp + me;
    Real const yq_h = (Ah > 0) ? Zh/Ah : 0.5;
    Real min_EB = (Y_NSE[SCXN]*mn + Y_NSE[SCXP]*mH + Y_NSE[SCXA]*(ma + 2*me)/4 +
                   Y_NSE[SCXH]*mFe/56.0)/mb - 1;
    Real max_EB = (Y_NSE[SCXN]*mn + Y_NSE[SCXP]*mH + Y_NSE[SCXA]*(mn + mH)/2 +
                   Y_NSE[SCXH]*((1 - yq_h)*mn + yq_h*mH))/mb - 1;
    return Kokkos::fmax(min_EB, Kokkos::fmin(eps_comp - eps_helm, max_EB));
  }

  KOKKOS_INLINE_FUNCTION bool IsInitialized() const { return m_initialized; }

  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
    assert(n <= MAX_SPECIES && n >= 0);
    n_species = n;
  }

  /// Effective low/high-density and temperature validity limits for the
  /// error policy. ld_* are the ramp starts; hd_* the compose table edges.
  KOKKOS_INLINE_FUNCTION void GetTableBoundaries(Real &ld_n, Real &hd_n,
                                                 Real &ld_t, Real &hd_t) const {
    ld_n = Kokkos::exp(m_ln_n_h0);
    ld_t = Kokkos::exp(m_ln_T_h0);
    hd_n = compose_eos.min_n;
    hd_t = compose_eos.min_T;
  }

  //--------------------------------------------------------------------------
  // Host-side setup (defined in eos_transition.cpp).
  //--------------------------------------------------------------------------
  void InitializeTables(std::string fname, std::string helm_fname);
  void SetTransition(Real n_start, Real n_end, Real T_start, Real T_end);
  void SetHelmholtzTMax(Real T_max) { m_helm_T_max = T_max; if (m_initialized) update_bounds(); }
  void SetHelmholtzNMax(Real n_max) { m_helm_n_max = n_max; if (m_initialized) update_bounds(); }
  void SetHelmholtzRampDecades(Real n_dec, Real T_dec) {
    m_helm_n_ramp_dec = n_dec; m_helm_T_ramp_dec = T_dec;
    if (m_initialized) update_bounds();
  }

 private:
  void SetBaryonMass(Real new_mb);
  void update_bounds();

  //--------------------------------------------------------------------------
  // Strip solver: invert log(P) or log(e) for temperature across the blended
  // region. Returns NaN if no root is bracketed. All returned/compared logs
  // are natural; compose grid nodes and table values are decoded via exp2_.
  //--------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION Real temperature_from_var_trans(int iv, Real var,
                                                         Real n, Real *Y) const {
    int in, iy;
    Real wn0, wn1, wy0, wy1;
    Real Yq = Y[SCYE];
    Real const ln_n = Kokkos::log(n);
    compose_eos.weight_idx_ln(&wn0, &wn1, &in, log2_(n));
    compose_eos.weight_idx_yq(&wy0, &wy1, &iy, Yq);

    // Natural log(T) at compose grid node it.
    auto lT_node = [=](int it) {
      return Kokkos::log(exp2_(compose_eos.m_log_t(it)));
    };
    // Compose table value (log2 of P or e) interpolated in (n, Ye) at node it.
    auto vcomp_node = [=](int it) {
      return wn0*(wy0*compose_eos.m_table(iv, in+0, iy+0, it) +
                  wy1*compose_eos.m_table(iv, in+0, iy+1, it)) +
             wn1*(wy0*compose_eos.m_table(iv, in+1, iy+0, it) +
                  wy1*compose_eos.m_table(iv, in+1, iy+1, it));
    };
    auto f = [=](int it) {
      Real lT = lT_node(it);
      Real T  = Kokkos::exp(lT);
      Real w  = TransitionFactor(n, T, ln_n, lT);
      Real var_helm = (iv == EOSCompOSE<LogPolicy>::ECLOGP)
                    ? helmholtz_eos.Pressure(n, T, Y)
                    : helmholtz_eos.Energy(n, T, Y);
      Real var_comp = exp2_(vcomp_node(it));
      return var - Kokkos::log(var_helm*(1 - w) + var_comp*w);
    };

    int nt_max = static_cast<int>(compose_eos.m_nt) - 1;
    int ilo_0 = (ln_n >= m_ln_n_h0) ? 0 : comp_it_trans_end;
    int ihi_0 = (ln_n < trans_ln_start)
              ? ((comp_it_helm_tmax < nt_max) ? comp_it_helm_tmax : nt_max)
              : comp_it_trans_start;
    if (ihi_0 < ilo_0 + 1) { ihi_0 = ilo_0 + 1; }

    int ilo = ilo_0;
    int ihi = ihi_0;
    Real flo = f(ilo);
    Real fhi = flo;
    bool bracketed = false;
    for (int i = ilo_0; i < ihi_0; ++i) {
      Real fnext = f(i + 1);
      if (flo*fnext <= 0) {
        ilo = i; ihi = i + 1; fhi = fnext; bracketed = true;
        break;
      }
      flo = fnext;
    }
    if (!bracketed) { return std::numeric_limits<Real>::quiet_NaN(); }

    Real ltlo = lT_node(ilo);
    Real lthi = lT_node(ihi);
    if (flo == 0) { return Kokkos::exp(ltlo); }
    if (fhi == 0) { return Kokkos::exp(lthi); }

    Real v0 = vcomp_node(ilo);
    Real v1 = vcomp_node(ihi);
    Real dv = v1 - v0;
    Real lt0 = ltlo;
    Real dlt = lthi - ltlo;

    Real wt;
    Real lb = 0.0;
    Real ub = 1.0;
    root.FalsePositionModified(RootFunction, lb, ub, wt, 1e-13, 1e-15,
                               iv, ilo, v0, dv, lt0, dlt, n, ln_n, Y, var, this);
    return Kokkos::exp(lt0 + wt*dlt);
  }

  //--------------------------------------------------------------------------
  // Root functor for the strip solver (blended log(var) residual in wt).
  //--------------------------------------------------------------------------
  class RootFunctor {
   public:
    KOKKOS_INLINE_FUNCTION
    Real operator()(Real wt, int iv, int it, Real v0, Real dv, Real lt0, Real dlt,
                    Real n, Real ln, Real *Y, Real var,
                    const EOSTransition *peos) const {
      Real lt = lt0 + wt*dlt;
      Real t  = Kokkos::exp(lt);
      Real f_trans = peos->TransitionFactor(n, t, ln, lt);
      Real var_helm = (iv == EOSCompOSE<LogPolicy>::ECLOGP)
                    ? peos->helmholtz_eos.Pressure(n, t, Y)
                    : peos->helmholtz_eos.Energy(n, t, Y);
      Real var_comp = peos->exp2_(v0 + wt*dv);
      Real var_pt   = Kokkos::log(var_comp*f_trans + var_helm*(1 - f_trans));
      return (var - var_pt)/var;
    }
  };

 protected:
  EOSCompOSE<LogPolicy> compose_eos;
  EOSHelmholtz          helmholtz_eos;

  Real trans_T_start, trans_T_end, trans_ln_start, trans_ln_end;
  Real m_trans_T_width, m_trans_ln_width;
  Real m_min_h;
  bool m_initialized;

  Real m_helm_n_max, m_helm_T_max;
  Real m_helm_n_ramp_dec, m_helm_T_ramp_dec;
  Real m_ln_n_h0, m_ln_T_h0, m_id_ln_n_ramp, m_id_ln_T_ramp;
  int  comp_it_trans_start, comp_it_trans_end, comp_it_helm_tmax;

  NumTools::Root root;
  RootFunctor RootFunction;

  static constexpr Real mn  = EOSHelmholtz::mn;
  static constexpr Real mp  = EOSHelmholtz::mp;
  static constexpr Real ma  = EOSHelmholtz::ma;
  static constexpr Real me  = EOSHelmholtz::me;
  static constexpr Real mFe = 52103.06261020851;  // atomic mass of 56Fe [MeV]
};

}  // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_EOS_TRANSITION_HPP_
