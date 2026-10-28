#ifndef EOS_PRIMITIVE_SOLVER_EOS_HELMHOLTZ_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_HELMHOLTZ_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_helmholtz.hpp
//  \brief Device port of EOSHelmholtz: a thermal EOS built on a tabulated
//         electron gas (2D in log n_e, log T) plus analytic ion/radiation terms.
//
//  The electron-gas table lives in Kokkos Views and evaluation is device
//  callable. Y[] holds the mass fractions of the network species (ScalarVariables);
//  Ye, the ion number per baryon and the rest mass per baryon are derived from them
//  and the per-species (A, Z, mass excess) set by SetSpecies.

///  \warning Assumes the table is uniformly spaced in log n_e and log T.

#include <cassert>
#include <string>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"

namespace Primitive {

class EOSHelmholtz : public EOSPolicyInterface, public SupportsEntropy,
                     public SupportsMassFractions, public SupportsTemperatureFromEntropy,
                     public SupportsSpecificHeat, public SupportsDensityFromPressure {

 public:
  enum TableVariables {
    ECLOGP   = 0,  //! log of electron pressure / 1 MeV fm^-3
    ECENT    = 1,  //! electron entropy per baryon [kb]
    ECLOGEPS = 2,  //! log of electron specific internal energy
    ECETA    = 3,  //! electron degeneracy parameter
    ECDEPSDT = 4,
    ECDPDN   = 5,
    ECDPDT   = 6,
    ECNVARS  = 7
  };

  /// Reads the table file (host only). The Ye range, and with it the density
  /// bounds, follows from the species list, so SetSpecies must come first.
  void ReadTableFromFile(std::string fname);

  /// Species list behind Y[]: mass number, charge number and ATOMIC mass excess
  /// [MeV] of each of the n species (host only). Atomic, because the electron
  /// table carries no rest mass and charge neutrality supplies exactly Z
  /// electrons per nucleus. A and Z need not be integers (effective species).
  /// g is the ground-state spin degeneracy 2J+1 entering the Sackur-Tetrode
  /// entropy; nullptr means 1. Sets n_species; validates, echoes on rank 0.
  void SetSpecies(int n, const Real *A, const Real *Z, const Real *dm,
                  const Real *g = nullptr);

  /// The species count is fixed by SetSpecies; anything else is a parfile error.
  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) { assert(n == n_species); }

  /// Set the baryon mass.
  ///
  /// N.B. mb is a convention, not a physical mass: the reference mass per
  /// baryon that defines rho = mb*n and the rest-mass/eps split.
  KOKKOS_INLINE_FUNCTION void SetBaryonMass(Real new_mb) { mb = new_mb; }

  /// Enable/disable the ion Coulomb (OCP) correction to P, eps and s.
  KOKKOS_INLINE_FUNCTION void SetCoulomb(bool use) { use_coulomb = use; }
  KOKKOS_INLINE_FUNCTION bool GetCoulomb() const { return use_coulomb; }

  KOKKOS_INLINE_FUNCTION bool IsInitialized() const { return m_initialized; }

 protected:
  EOSHelmholtz() :
      m_log_ne("helm log ne", 1),
      m_log_t("helm log T", 1),
      m_t("helm T", 1),
      m_table("helm table", 1, 1, 1) {
    eos_units = MakeNuclear();
    m_initialized = false;
    m_id_log_ne = std::numeric_limits<Real>::quiet_NaN();
    m_id_log_t  = std::numeric_limits<Real>::quiet_NaN();
    m_nn = 0;
    m_nt = 0;
    m_ne_min = std::numeric_limits<Real>::quiet_NaN();
    m_ne_max = std::numeric_limits<Real>::quiet_NaN();
    for (int s = 0; s < MAX_SPECIES; ++s) {
      min_Y[s] = 0.0;
      max_Y[s] = 1.0;
    }
    n_species = 0;  // SetSpecies must be called before the table is read
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    return TemperatureFromEps(n, e/(mb*n) - 1.0, Y);
  }

  //! Temperature from specific internal energy. guess_it, when given,
  //! warm-starts the bracket hunt with the table index found by a previous
  //! call (the c2p iteration hot path).
  KOKKOS_INLINE_FUNCTION Real TemperatureFromEps(Real n, Real eps, Real *Y,
                                                 int *guess_it = nullptr) const {
    // Lazy bounds: floor-clamped states (the atmosphere) hit the min early-out
    // and never pay for the max bound.
    Real eps_min = MinimumInternalEnergy(n, Y);
    if (eps <= eps_min) { return min_T; }
    Real eps_max = MaximumInternalEnergy(n, Y);
    if (eps >= eps_max) { return max_T; }
    return temperature_from_var(ECLOGEPS, eps, n, Y, guess_it);
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    // Lazy bounds, as in TemperatureFromEps.
    Real p_min = MinimumPressure(n, Y);
    if (p <= p_min) { return min_T; }
    Real p_max = MaximumPressure(n, Y);
    if (p >= p_max) { return max_T; }
    return temperature_from_var(ECLOGP, p, n, Y);
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromEntropy(Real n, Real s, Real *Y) const {
    Real s_min = MinimumEntropy(n, Y);
    Real s_max = MaximumEntropy(n, Y);
    return (s <= s_min) ? min_T
         : (s >= s_max) ? max_T
                        : temperature_from_var(ECENT, s, n, Y);
  }

  //! Density at which the pressure is p at fixed T. NaN if p is not bracketed by
  //  the table's density range at this composition.
  KOKKOS_INLINE_FUNCTION Real DensityFromPressureTemperature(Real p, Real T,
                                                             Real *Y) const {
    return density_from_pressure(p, Y, [=](Real) { return T; });
  }

  //! Density on the isentrope s at which the pressure is p; the temperature there
  //  through T. NaN (both) if p is not bracketed, or if s is outside the entropies
  //  attainable at the solution (TemperatureFromEntropy clamps to min_T/max_T there,
  //  so that n would not be on the isentrope).
  KOKKOS_INLINE_FUNCTION Real DensityFromPressureEntropy(Real p, Real s, Real *Y,
                                                         Real *T = nullptr) const {
    Real n = density_from_pressure(p, Y, [&](Real nt) {
      return TemperatureFromEntropy(nt, s, Y);
    });
    Real Ts = Kokkos::Experimental::quiet_NaN_v<Real>;
    if (Kokkos::isfinite(n)) {
      if (s >= Entropy(n, min_T, Y) && s <= Entropy(n, max_T, Y)) {
        Ts = TemperatureFromEntropy(n, s, Y);
      } else {
        n = Kokkos::Experimental::quiet_NaN_v<Real>;
      }
    }
    if (T != nullptr) { *T = Ts; }
    return n;
  }

  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T,
                                                     const Real *Y) const {
    return eval_lin_at_nty(ECLOGEPS, n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    return (SpecificInternalEnergy(n, T, Y) + 1.0)*n*mb;
  }

  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    return eval_lin_at_nty(ECLOGP, n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    return eval_lin_at_nty(ECENT, n, T, Y);
  }

  //! (d eps/dT) at fixed n, Y [1/MeV]: table channel plus radiation, ion and Coulomb.
  KOKKOS_INLINE_FUNCTION Real SpecificHeat(Real n, Real T, Real *Y) const {
    return eval_lin_at_nty(ECDEPSDT, n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    return (Pressure(n, T, Y) + Energy(n, T, Y))/n;
  }

  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    // Timmes & Arnett (1999) Gamma_1 from the (n, T) derivatives. One table
    // point (log n_e, log T) and one Coulomb prep serve all five channels.
    const CoulombPrep cp = coulomb_prep(n, Y);
    const Real ln_ne = Kokkos::log(n*cp.comp.ye);
    const Real lT    = Kokkos::log(T);
    auto ch = [&](int vi) {
      return add_rad_ion_lin(vi, eval_at_lnty(vi, ln_ne, lT), n, T, lT, Y, cp);
    };
    Real pres    = ch(ECLOGP);
    Real eps     = ch(ECLOGEPS);
    Real dpresdt = ch(ECDPDT);                              // dP/dT [fm^-3]
    Real dpresdn = ch(ECDPDN);                              // dP/dn [MeV]
    Real cv      = ch(ECDEPSDT);                            // deps/dT [1/MeV]
    Real chit    = T/pres*dpresdt;
    Real chin    = n/pres*dpresdn;
    Real x       = pres*chit/(n*mb*T*cv);
    Real gam1    = chit*x + chin;
    Real z = 1.0 + (1.0 + eps)*n*mb/pres;
    return Kokkos::sqrt(gam1/z);
  }

  KOKKOS_INLINE_FUNCTION Real ElectronChemicalPotential(Real n, Real T, Real *Y) const {
    return eval_lin_at_nty(ECETA, n, T, Y)*T + me;
  }

  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const { return m_min_h; }
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    return Pressure(n, max_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return Energy(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    return Energy(n, max_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MinimumInternalEnergy(Real n, Real *Y) const {
    return SpecificInternalEnergy(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumInternalEnergy(Real n, Real *Y) const {
    return SpecificInternalEnergy(n, max_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MinimumEntropy(Real n, Real *Y) const {
    return Entropy(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumEntropy(Real n, Real *Y) const {
    return Entropy(n, max_T, Y);
  }

 private:
  //! Per-baryon sums over the species: electrons, ions, charged ions, and the
  //! rest mass per baryon in units of mb minus one (mbar = m_u + sum X_i dm_i/A_i).
  //! The last is the binding-energy term that used to be its own advected scalar.
  struct Composition {
    Real ye;        // electrons per baryon
    Real inv_abar;  // ions per baryon = 1/Abar
    Real y_chg;     // charged ions per baryon (free neutrons carry no charge)
    Real eb;        // mbar/mb - 1
  };
  //! The sums are taken over X/sum(X): limited reconstructions of the individual
  //! species do not sum to 1 at a smeared composition contact, and since Ye
  //! inherits that error the electron pressure would too. A non-positive sum (a
  //! floor-overwritten cell) falls back to pure species 0.
  KOKKOS_INLINE_FUNCTION Composition composition(const Real *Y) const {
    Composition c = {0.0, 0.0, 0.0, 0.0};
    Real xsum = 0.0;
    for (int s = 0; s < n_species; ++s) { xsum += Y[s]; }
    Real dmoa = 0.0;
    if (xsum > 0.0) {
      const Real inv = 1.0/xsum;
      for (int s = 0; s < n_species; ++s) {
        const Real xi = Y[s]*inv;
        const Real yi = xi*m_inv_A[s];
        c.inv_abar += yi;
        c.ye       += xi*m_zoa[s];
        dmoa       += xi*m_dmoa[s];
        if (m_Z[s] > 0.0) { c.y_chg += yi; }
      }
    } else {
      c.inv_abar = m_inv_A[0];
      c.ye       = m_zoa[0];
      dmoa       = m_dmoa[0];
      if (m_Z[0] > 0.0) { c.y_chg = c.inv_abar; }
    }
    c.eb = (m_u + dmoa)/mb - 1.0;
    return c;
  }

  // Classical one-component-plasma Coulomb correction for a single mean ion averaged
  // over the charged species (free neutrons carry no charge). Yakovlev & Shalybkov
  // (1989) fits with the Timmes Helmholtz coefficients.
  // Returns the per-ion energy u = E_C/(N_i T), du/dGamma and the per-ion entropy s
  // [kB]; y_chg = 0 means no correction. Not applied to the chemical potentials.
  struct CoulombTerms {
    Real y_chg;  // charged ions per baryon
    Real gamma;  // plasma coupling parameter
    Real u;      // E_C per ion in units of T
    Real du;     // du/dGamma
    Real ucv;    // u - Gamma du, written without the O(Gamma) cancellation (enters
                 // d/dT of the Coulomb energy and pressure)
    Real s;      // S_C per ion in kB
  };
  // The (n, Y)-only part of the ion and Coulomb terms. Gamma = Gamma_T/T at fixed
  // (n, Y), so the ion-sphere cbrt and log(Gamma_T) are taken once per prep and every
  // temperature then costs a division, and for Gamma < 1 one exp in place of the pow.
  // Callers that evaluate many temperatures at one (n, Y) (the inversions, the
  // multi-channel SoundSpeed) build the prep once.
  struct CoulombPrep {
    Composition comp;
    Real y_chg;       // charged ions per baryon entering the correction; 0 = off
    Real gamma_T;     // Gamma*T = zbar^2 e^2/a_i [MeV]
    Real ln_gamma_T;  // log(gamma_T)
  };
  KOKKOS_INLINE_FUNCTION CoulombPrep coulomb_prep(Real n, const Real *Y) const {
    CoulombPrep cp = {composition(Y), 0.0, 0.0, 0.0};
    if (!use_coulomb) { return cp; }
    const Real y_chg = cp.comp.y_chg;
    if (!(y_chg > 1e-30) || !(cp.comp.ye > 0.0)) {
      return cp;
    }
    Real zbar  = cp.comp.ye/y_chg;  // charge neutrality
    Real a_i   = Kokkos::cbrt(3.0/(4.0*M_PI*n*y_chg));
    cp.y_chg      = y_chg;
    cp.gamma_T    = zbar*zbar*esqu/a_i;
    cp.ln_gamma_T = Kokkos::log(cp.gamma_T);
    return cp;
  }
  //! lT = log(T). p_e, when known (the P channel), saves the electron-pressure lookup
  //! of the positivity guard.
  KOKKOS_INLINE_FUNCTION CoulombTerms coulomb_terms(const CoulombPrep &cp, Real n,
                                                     Real T, Real lT, const Real *Y,
                                                     Real p_e = -1.0) const {
    CoulombTerms c = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (!(cp.y_chg > 0.0) || !(T > 0.0)) {
      return c;
    }
    const Real y_chg = cp.y_chg;
    const Real gamma = cp.gamma_T/T;
    const Real lng   = cp.ln_gamma_T - lT;  // log(gamma)
    constexpr Real a1 = -0.898004, b1 = 0.96786, c1 = 0.220703,
                   d1 = -0.86097, e1 = 2.5269;
    constexpr Real a2 = 0.29561, b2 = 1.9885, c2 = 0.288675;
    if (gamma >= 1.0) {
      Real x = Kokkos::sqrt(Kokkos::sqrt(gamma));  // gamma^(1/4)
      c.u  = a1*gamma + b1*x + c1/x + d1;
      c.du = a1 + 0.25*(b1*x - c1/x)/gamma;
      c.ucv = 0.75*b1*x + 1.25*c1/x + d1;
      c.s  = -(3.0*b1*x - 5.0*c1/x + d1*(lng - 1.0) - e1);
    } else {
      Real x = gamma*Kokkos::sqrt(gamma);  // gamma^(3/2)
      Real y = Kokkos::exp(b2*lng);        // gamma^b2
      c.u  = -3.0*c2*x + a2*y;
      c.du = (-4.5*c2*x + a2*b2*y)/gamma;
      c.ucv = 1.5*c2*x + a2*(1.0 - b2)*y;
      c.s  = -(c2*x - a2*(b2 - 1.0)/b2*y);
    }
    const Real pcoul = n*y_chg*T*c.u/3.0;
    const Real p_ir  = asol/3.0*T*T*T*T + n*cp.comp.inv_abar*T;
    if (p_ir + pcoul <= 0.0) {
      if (p_e < 0.0) {
        p_e = Kokkos::exp(eval_at_lnty(ECLOGP, Kokkos::log(n*cp.comp.ye), lT));
      }
      if (p_e + p_ir + pcoul <= 0.0) {
        return CoulombTerms{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      }
    }
    c.y_chg = y_chg;
    c.gamma = gamma;
    return c;
  }

  /// Add the analytic radiation + ion terms to an electron-gas table value.
  /// lT = log(T); cp = coulomb_prep(n, Y). The log-stored channels (ECLOGP,
  /// ECLOGEPS) return the total itself, not its log: every caller wants the value
  /// or compares linearly, so no exp(log(.)) round trip.
  KOKKOS_INLINE_FUNCTION Real add_rad_ion_lin(int vi, Real var, Real n, Real T,
                                              Real lT, const Real *Y,
                                              const CoulombPrep &cp) const {
    const Real Ye      = cp.comp.ye;
    const Real eps_fac = Ye/mb;
    switch (vi) {
      case ECLOGP: {
        Real pele = Kokkos::exp(var);
        Real ptot = pele + (asol/3.0*T*T*T*T + n*cp.comp.inv_abar*T);
        if (use_coulomb) {
          const CoulombTerms c = coulomb_terms(cp, n, T, lT, Y, pele);
          Real pcoul = n*c.y_chg*T*c.u/3.0;
          if (ptot + pcoul > 0.0) { ptot += pcoul; }
        }
        return ptot;
      }
      case ECENT: {
        Real srad = 4.0*asol/3.0*T*T*T/n;
        // Sackur-Tetrode per species; a vanishing species contributes 0, not 0*inf.
        // Ground-state spin degeneracy only (no excited states): fine below a few GK
        // for light nuclei, exact for n and p.
        Real xsum = 0.0;
        for (int s = 0; s < n_species; ++s) { xsum += Y[s]; }
        const Real xnorm = (xsum > 0.0) ? 1.0/xsum : 0.0;
        Real sion = 0.0;
        for (int s = 0; s < n_species; ++s) {
          const Real yi = ((xsum > 0.0) ? Y[s]*xnorm : (s == 0 ? 1.0 : 0.0))*m_inv_A[s];
          if (yi > 0.0) {
            sion += yi*(2.5 - Kokkos::log(n*yi/m_g[s]*Kokkos::pow(sac_const/(m_mion[s]*T),
                                                                    1.5)));
          }
        }
        Real scoul = 0.0;
        if (use_coulomb) {
          const CoulombTerms c = coulomb_terms(cp, n, T, lT, Y);
          scoul = c.y_chg*c.s;
        }
        return Ye*var + srad + sion + scoul;
      }
      case ECLOGEPS: {
        Real erad  = asol*T*T*T*T/(n*mb);
        Real eion  = 1.5*T*cp.comp.inv_abar/mb;
        Real ebind = cp.comp.eb;
        Real etot  = eps_fac*Kokkos::exp(var) + erad + eion + ebind;
        if (use_coulomb) {
          const CoulombTerms c = coulomb_terms(cp, n, T, lT, Y);
          Real ecoul = c.y_chg*T*c.u/mb;
          if (etot + ecoul > 0.0) { etot += ecoul; }
        }
        return etot;
      }
      case ECDEPSDT: {
        Real deraddt = 4.0*asol*T*T*T/(n*mb);
        Real deiondt = 1.5*cp.comp.inv_abar/mb;
        Real decouldt = 0.0;
        if (use_coulomb) {
          // d/dT of y_chg*T*u(Gamma)/mb with dGamma/dT = -Gamma/T, included under the
          // same condition as the Coulomb energy in ECLOGEPS (which needs eps itself,
          // one more table lookup)
          const CoulombTerms c = coulomb_terms(cp, n, T, lT, Y);
          if (c.y_chg > 0.0) {
            Real eps_e = Kokkos::exp(eval_at_lnty(ECLOGEPS, Kokkos::log(n*Ye), lT));
            Real etot  = eps_fac*eps_e + asol*T*T*T*T/(n*mb) + 1.5*T*cp.comp.inv_abar/mb
                       + cp.comp.eb;
            Real ecoul = c.y_chg*T*c.u/mb;
            if (etot + ecoul > 0.0) { decouldt = c.y_chg*c.ucv/mb; }
          }
        }
        return eps_fac*var + deraddt + deiondt + decouldt;
      }
      case ECDPDN: {
        Real dpiondn = T*cp.comp.inv_abar;
        Real dpcouldn = 0.0;
        if (use_coulomb) {
          // d/dn of n*y_chg*T*u(Gamma)/3 with dGamma/dn = Gamma/(3n)
          const CoulombTerms c = coulomb_terms(cp, n, T, lT, Y);
          dpcouldn = c.y_chg*T*(c.u + c.gamma*c.du/3.0)/3.0;
        }
        return Ye*var + dpiondn + dpcouldn;
      }
      case ECDPDT: {
        Real dpraddt = 4.0/3.0*asol*T*T*T;
        Real dpiondt = n*cp.comp.inv_abar;
        Real dpcouldt = 0.0;
        if (use_coulomb) {
          // d/dT of n*y_chg*T*u(Gamma)/3 with dGamma/dT = -Gamma/T
          const CoulombTerms c = coulomb_terms(cp, n, T, lT, Y);
          dpcouldt = n*c.y_chg*c.ucv/3.0;
        }
        return var + dpraddt + dpiondt + dpcouldt;
      }
      case ECETA:
      default:
        return var;
    }
  }

  /// Full channel value at (n, T, Y); ECLOGP/ECLOGEPS return P and eps themselves.
  KOKKOS_INLINE_FUNCTION Real eval_lin_at_nty(int vi, Real n, Real T,
                                              const Real *Y) const {
    const Real lT = Kokkos::log(T);
    const CoulombPrep cp = coulomb_prep(n, Y);
    Real var = eval_at_lnty(vi, Kokkos::log(n*cp.comp.ye), lT);
    if (vi == ECETA) { return var; }  // no analytic terms
    return add_rad_ion_lin(vi, var, n, T, lT, Y, cp);
  }

  /// Pressure and energy density with ln_ne = log(n Ye), lT = log(T) and the prep
  /// supplied by the caller (temperature scans at fixed n, Y).
  KOKKOS_INLINE_FUNCTION Real pressure_at(Real n, Real T, Real ln_ne, Real lT,
                                          const Real *Y, const CoulombPrep &cp) const {
    return add_rad_ion_lin(ECLOGP, eval_at_lnty(ECLOGP, ln_ne, lT), n, T, lT, Y, cp);
  }
  KOKKOS_INLINE_FUNCTION Real energy_at(Real n, Real T, Real ln_ne, Real lT,
                                        const Real *Y, const CoulombPrep &cp) const {
    return (add_rad_ion_lin(ECLOGEPS, eval_at_lnty(ECLOGEPS, ln_ne, lT),
                            n, T, lT, Y, cp) + 1.0)*n*mb;
  }

  KOKKOS_INLINE_FUNCTION void weight_idx_ln(Real *w0, Real *w1, int *in,
                                            Real log_n) const {
    *in = (log_n - m_log_ne(0))*m_id_log_ne;
    if (*in > m_nn - 2) {
      *in = m_nn - 2;
    } else if (*in < 0) {
      *in = 0;
    }
    *w1 = (log_n - m_log_ne(*in))*m_id_log_ne;
    *w0 = 1.0 - (*w1);
  }

  KOKKOS_INLINE_FUNCTION void weight_idx_lt(Real *w0, Real *w1, int *it,
                                            Real log_t) const {
    *it = (log_t - m_log_t(0))*m_id_log_t;
    if (*it > m_nt - 2) {
      *it = m_nt - 2;
    } else if (*it < 0) {
      *it = 0;
    }
    *w1 = (log_t - m_log_t(*it))*m_id_log_t;
    *w0 = 1.0 - (*w1);
  }

  /// Electron-gas-only bilinear interpolation (no analytic terms).
  KOKKOS_INLINE_FUNCTION Real eval_at_lnty(int iv, Real log_n, Real log_t) const {
    int in, it;
    Real wn0, wn1, wt0, wt1;
    weight_idx_ln(&wn0, &wn1, &in, log_n);
    weight_idx_lt(&wt0, &wt1, &it, log_t);
    return wn0*(wt0*m_table(iv, in+0, it+0) + wt1*m_table(iv, in+0, it+1)) +
           wn1*(wt0*m_table(iv, in+1, it+0) + wt1*m_table(iv, in+1, it+1));
  }

  /// Invert one table variable for temperature at fixed (n, Y). Returns NaN on
  /// bracketing failure. var is the target value itself (P, eps or s), not its log:
  /// the root of var - value is the root of log var - log value, and staying linear
  /// saves a log per evaluation. The analytic terms make f nonlinear in log T inside a
  /// cell, so refine with a false-position (Anderson-Bjorck) iteration.
  KOKKOS_INLINE_FUNCTION Real temperature_from_var(int iv, Real var, Real n,
                                                   const Real *Y,
                                                   int *guess_it = nullptr) const {
    // (n, Y)-only parts of the ion/Coulomb terms, identical at every T.
    const CoulombPrep cp = coulomb_prep(n, Y);
    int in;
    Real wn0, wn1;
    weight_idx_ln(&wn0, &wn1, &in, Kokkos::log(n*cp.comp.ye));

    auto f = [=](int it) {
      Real var_pt = wn0*m_table(iv, in+0, it) + wn1*m_table(iv, in+1, it);
      var_pt = add_rad_ion_lin(iv, var_pt, n, m_t(it), m_log_t(it), Y, cp);
      return var - var_pt;
    };

    int ilo = 0;
    int ihi = m_nt - 1;
    Real flo, fhi;
    bool bracketed = false;

    // Hunt locally around the warm-start index first. f = var - var_pt is
    // monotone in it for these channels, so f < 0 puts the root to the left.
    // A miss (stale index, or one left behind by another call) fails the sign
    // check and falls through to the full search below, so any int is safe.
    if (guess_it != nullptr && *guess_it >= 0 && *guess_it < m_nt - 1) {
      int itg = *guess_it;
      Real fl = f(itg);
      Real fh = f(itg + 1);
      if (fl*fh <= 0) {
        ilo = itg; ihi = itg + 1; flo = fl; fhi = fh; bracketed = true;
      } else if (fl < 0 && itg > 0) {              // try shifting left
        Real fl_minus = f(itg - 1);
        if (fl_minus*fl <= 0) {
          ilo = itg - 1; ihi = itg; flo = fl_minus; fhi = fl; bracketed = true;
        }
      } else if (fh > 0 && itg + 2 < m_nt) {       // try shifting right
        Real fh_plus = f(itg + 2);
        if (fh*fh_plus <= 0) {
          ilo = itg + 1; ihi = itg + 2; flo = fh; fhi = fh_plus; bracketed = true;
        }
      }
    }

    if (!bracketed) {
      ilo = 0;
      ihi = m_nt - 1;
      flo = f(ilo);
      fhi = f(ihi);
      while (flo*fhi > 0) {
        if (ilo == ihi - 1) {
          break;
        } else {
          ilo += 1;
          flo = f(ilo);
        }
      }
    }
    if (flo*fhi > 0) {
      return Kokkos::Experimental::quiet_NaN_v<Real>;
    }
    while (ihi - ilo > 1) {
      int ip = ilo + (ihi - ilo)/2;
      Real fp = f(ip);
      if (fp*flo <= 0) {
        ihi = ip;
        fhi = fp;
      } else {
        ilo = ip;
        flo = fp;
      }
    }
    if (guess_it != nullptr) { *guess_it = ilo; }
    Real lthi = m_log_t(ihi);
    Real ltlo = m_log_t(ilo);
    if (flo == 0) { return m_t(ilo); }
    if (fhi == 0) { return m_t(ihi); }

    Real const v_lo = wn0*m_table(iv, in+0, ilo) + wn1*m_table(iv, in+1, ilo);
    Real const v_hi = wn0*m_table(iv, in+0, ihi) + wn1*m_table(iv, in+1, ihi);
    auto g = [=](Real lt) {
      Real wt     = (lt - ltlo)/(lthi - ltlo);
      Real var_pt = (1.0 - wt)*v_lo + wt*v_hi;
      return var - add_rad_ion_lin(iv, var_pt, n, Kokkos::exp(lt), lt, Y, cp);
    };

    Real la = ltlo, lb_ = lthi;
    Real fa = flo, fb = fhi;
    Real lt   = la - fa*(lb_ - la)/(fb - fa);
    int side  = 0;
    for (int i = 0; i < 50; ++i) {
      Real ft = g(lt);
      if (ft == 0.0) { break; }
      if (ft*fa > 0) {
        if (side == 1) {
          Real m = 1.0 - ft/fa;
          fb = (m > 0) ? fb*m : 0.5*fb;
        }
        la = lt; fa = ft; side = 1;
      } else {
        if (side == -1) {
          Real m = 1.0 - ft/fb;
          fa = (m > 0) ? fa*m : 0.5*fa;
        }
        lb_ = lt; fb = ft; side = -1;
      }
      Real lt_new = la - fa*(lb_ - la)/(fb - fa);
      if (Kokkos::fabs(lt_new - lt) <= 1e-13*(Kokkos::fabs(lt_new) + 1e-13)) {
        lt = lt_new;
        break;
      }
      lt = lt_new;
    }
    return Kokkos::exp(lt);
  }

  /// Invert the pressure for the density at fixed composition, with the temperature
  /// given by T_of(n) (a constant, or T(n, s) on an isentrope). Same scheme as
  /// temperature_from_var with the roles swapped: bracket between neighbouring n_e
  /// nodes by bisection on the node index, then false position (Anderson-Bjorck) in
  /// ln n inside that cell, where the electron interpolant is smooth. Returns NaN if p
  /// is not bracketed by the table's density range at this composition. Meant for
  /// initial data: every residual costs a full pressure evaluation, and on an
  /// isentrope a temperature inversion as well. P(n) need not be monotonic (the
  /// Coulomb positivity guard switches at low T); the result is then one of several
  /// solutions.
  template <class TFn>
  KOKKOS_INLINE_FUNCTION Real density_from_pressure(Real p, Real *Y, TFn&& T_of) const {
    const Real nan = Kokkos::Experimental::quiet_NaN_v<Real>;
    const Real ye = composition(Y).ye;
    if (!(ye > 0.0) || !(p > 0.0)) { return nan; }
    const Real lye = Kokkos::log(ye);  // ln n = ln n_e - ln Ye

    auto g = [&](Real ln) {
      Real n = Kokkos::exp(ln);
      return Pressure(n, T_of(n), Y) - p;
    };
    auto f = [&](int in) { return g(m_log_ne(in) - lye); };

    int ilo = 0, ihi = m_nn - 1;
    Real flo = f(ilo), fhi = f(ihi);
    if (!(flo*fhi <= 0.0)) { return nan; }  // no sign change, or NaN
    while (ihi - ilo > 1) {
      int ip = ilo + (ihi - ilo)/2;
      Real fp = f(ip);
      if (fp*flo <= 0) {
        ihi = ip;
        fhi = fp;
      } else {
        ilo = ip;
        flo = fp;
      }
    }
    if (flo == 0) { return Kokkos::exp(m_log_ne(ilo) - lye); }
    if (fhi == 0) { return Kokkos::exp(m_log_ne(ihi) - lye); }

    Real la = m_log_ne(ilo) - lye, lb_ = m_log_ne(ihi) - lye;
    Real fa = flo, fb = fhi;
    Real ln   = la - fa*(lb_ - la)/(fb - fa);
    int side  = 0;
    for (int i = 0; i < 50; ++i) {
      Real fn = g(ln);
      if (fn == 0.0) { break; }
      if (fn*fa > 0) {
        if (side == 1) {
          Real m = 1.0 - fn/fa;
          fb = (m > 0) ? fb*m : 0.5*fb;
        }
        la = ln; fa = fn; side = 1;
      } else {
        if (side == -1) {
          Real m = 1.0 - fn/fb;
          fa = (m > 0) ? fa*m : 0.5*fa;
        }
        lb_ = ln; fb = fn; side = -1;
      }
      Real ln_new = la - fa*(lb_ - la)/(fb - fa);
      if (Kokkos::fabs(ln_new - ln) <= 1e-13*(Kokkos::fabs(ln_new) + 1.0)) {
        ln = ln_new;
        break;
      }
      ln = ln_new;
    }
    // P(n) jumps where the Coulomb positivity guard switches, and the iteration can
    // converge onto that jump (or run out of steps) without P = p. A finite n is
    // returned only if it solves the equation.
    if (!(Kokkos::fabs(g(ln)) <= 1e-10*p)) { return nan; }
    return Kokkos::exp(ln);
  }

 private:
  Real m_id_log_ne, m_id_log_t;
  int m_nn, m_nt;
  Real m_ne_min, m_ne_max;  // electron-density edges of the table [fm^-3]
  Real m_min_h;
  bool m_initialized;
  bool use_coulomb = true;  // ion Coulomb (OCP) correction

  // Species list (see SetSpecies) and the per-species factors the composition
  // sums use: 1/A, Z/A, dm/A [MeV] and the nuclear mass A m_u + dm - Z me [MeV].
  Real m_A[MAX_SPECIES], m_Z[MAX_SPECIES], m_dm[MAX_SPECIES], m_g[MAX_SPECIES];
  Real m_inv_A[MAX_SPECIES], m_zoa[MAX_SPECIES], m_dmoa[MAX_SPECIES], m_mion[MAX_SPECIES];

  /// Density bounds from the table edges and the Ye range of the species list.
  void update_density_bounds();

  // Table storage on device.
  DvceArray1D<Real> m_log_ne;
  DvceArray1D<Real> m_log_t;
  DvceArray1D<Real> m_t;       // exp(m_log_t), so scans need no exp per node
  DvceArray3D<Real> m_table;   // (ECNVARS, m_nn, m_nt)

 public:
  static constexpr Real hbarc = 197.3269804;  // MeV fm
  static constexpr Real asol = M_PI*M_PI/(15.0*hbarc*hbarc*hbarc);  // (MeV fm)^-3
  static constexpr Real sac_const = hbarc*hbarc*2.0*M_PI;           // (MeV fm)^2
  static constexpr Real me = 0.5109989461;   // MeV
  static constexpr Real esqu = 1.4399764;    // e^2 [MeV fm]
  static constexpr Real m_u = 931.49410242;  // atomic mass unit [MeV], CODATA 2018
};

}  // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_EOS_HELMHOLTZ_HPP_
