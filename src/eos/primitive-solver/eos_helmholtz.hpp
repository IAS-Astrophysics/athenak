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
//  Used only as a sub-component of EOSTransition (never a standalone EOSPolicy).
//  The electron-gas table lives in Kokkos Views and evaluation is device
//  callable. Composition is passed through Y[] via the ScalarVariables indices.

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

template<typename LogPolicy> class EOSTransition;

class EOSHelmholtz : public EOSPolicyInterface {
  template<typename LP> friend class EOSTransition;

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

  EOSHelmholtz() :
      m_log_ne("helm log ne", 1),
      m_log_t("helm log T", 1),
      m_table("helm table", 1, 1, 1) {
    n_species = 1;
    eos_units = MakeNuclear();
    m_initialized = false;
    m_id_log_ne = std::numeric_limits<Real>::quiet_NaN();
    m_id_log_t  = std::numeric_limits<Real>::quiet_NaN();
    m_nn = 0;
    m_nt = 0;
    mn = mn_codata;
    mp = mp_codata;

    min_Y[SCYE] = 0.0;  min_Y[SCXN] = 0.0;  min_Y[SCXP] = 0.0;
    min_Y[SCXA] = 0.0;  min_Y[SCXH] = 0.0;  min_Y[SCAH] = 1.0;
    max_Y[SCYE] = 1.0;  max_Y[SCXN] = 1.0;  max_Y[SCXP] = 1.0;
    max_Y[SCXA] = 1.0;  max_Y[SCXH] = 1.0;  max_Y[SCAH] = 500.0;
  }

  /// Reads the table file (host only). Ye range comes from the compose table.
  void ReadTableFromFile(std::string fname, Real min_Ye, Real max_Ye);

  /// Set the baryon mass.
  ///
  /// N.B. mb is a convention, not a physical mass: the reference mass per
  /// baryon that defines rho = mb*n and the rest-mass/eps split.
  KOKKOS_INLINE_FUNCTION void SetBaryonMass(Real new_mb) { mb = new_mb; }

  /// Set the physical nucleon masses [MeV].
  ///
  /// Used by EOSTransition to adopt the values carried by the compose table so
  /// that both halves of the blend build their chemical potentials and
  /// rest-mass zero points from the same constants.
  KOKKOS_INLINE_FUNCTION void SetNucleonMasses(Real new_mn, Real new_mp) {
    mn = new_mn;
    mp = new_mp;
  }

  KOKKOS_INLINE_FUNCTION bool IsInitialized() const { return m_initialized; }

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
    return temperature_from_var(ECLOGEPS, Kokkos::log(eps), n, Y, guess_it);
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    // Lazy bounds, as in TemperatureFromEps.
    Real p_min = MinimumPressure(n, Y);
    if (p <= p_min) { return min_T; }
    Real p_max = MaximumPressure(n, Y);
    if (p >= p_max) { return max_T; }
    return temperature_from_var(ECLOGP, Kokkos::log(p), n, Y);
  }

  KOKKOS_INLINE_FUNCTION Real TemperatureFromEntropy(Real n, Real s, Real *Y) const {
    Real s_min = MinimumEntropy(n, Y);
    Real s_max = MaximumEntropy(n, Y);
    return (s <= s_min) ? min_T
         : (s >= s_max) ? max_T
                        : temperature_from_var(ECENT, s, n, Y);
  }

  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return Kokkos::exp(eval_at_nty(ECLOGEPS, n, T, Y));
  }

  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, Real *Y) const {
    return (SpecificInternalEnergy(n, T, Y) + 1.0)*n*mb;
  }

  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    return Kokkos::exp(eval_at_nty(ECLOGP, n, T, Y));
  }

  KOKKOS_INLINE_FUNCTION Real Abar(Real n, Real T, Real *Y) const {
    return 1.0/inverse_abar(Y);
  }

  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    return eval_at_nty(ECENT, n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    return (Pressure(n, T, Y) + Energy(n, T, Y))/n;
  }

  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    // Timmes & Arnett (1999) Gamma_1 from the (n, T) derivatives.
    Real pres    = Pressure(n, T, Y);
    Real eps     = SpecificInternalEnergy(n, T, Y);
    Real dpresdt = eval_at_nty(ECDPDT, n, T, Y);            // dP/dT [fm^-3]
    // dpdn channel is dP_ele/dn_e; n_e = Ye*n_b introduces the Ye factor.
    Real dpele_dne = eval_at_lnty(ECDPDN, Kokkos::log(n*Y[SCYE]), Kokkos::log(T));
    Real dpresdn   = Y[SCYE]*dpele_dne + T*inverse_abar(Y);  // dP/dn [MeV]
    Real cv      = eval_at_nty(ECDEPSDT, n, T, Y);          // deps/dT [1/MeV]
    Real chit    = T/pres*dpresdt;
    Real chin    = n/pres*dpresdn;
    Real x       = pres*chit/(n*mb*T*cv);
    Real gam1    = chit*x + chin;
    Real z = 1.0 + (1.0 + eps)*n*mb/pres;
    return Kokkos::sqrt(gam1/z);
  }

  // Yn/Yp are floored: min_Y[SCXN] is 0 and frozen-out ejecta legitimately
  // reach it, which would send mu -> -inf and overflow the nurates degeneracy
  // parameters. The floor is far below any composition the rates resolve.
  KOKKOS_INLINE_FUNCTION Real NeutronChemicalPotential(Real n, Real T, Real *Y) const {
    Real Yn = Kokkos::fmax(Y[SCXN], min_Y_free);
    return mn + T*Kokkos::log(n*Yn/2*Kokkos::pow(sac_const/(mn*T), 1.5));
  }

  KOKKOS_INLINE_FUNCTION Real ProtonChemicalPotential(Real n, Real T, Real *Y) const {
    Real Yp = Kokkos::fmax(Y[SCXP], min_Y_free);
    return mp + T*Kokkos::log(n*Yp/2*Kokkos::pow(sac_const/(mp*T), 1.5));
  }

  KOKKOS_INLINE_FUNCTION Real ElectronChemicalPotential(Real n, Real T, Real *Y) const {
    return eval_at_nty(ECETA, n, T, Y)*T + me;
  }

  KOKKOS_INLINE_FUNCTION Real BaryonChemicalPotential(Real n, Real T, Real *Y) const {
    return NeutronChemicalPotential(n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real ChargeChemicalPotential(Real n, Real T, Real *Y) const {
    return ProtonChemicalPotential(n, T, Y) - NeutronChemicalPotential(n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real ElectronLeptonChemicalPotential(Real n, Real T,
                                                              Real *Y) const {
    return ElectronChemicalPotential(n, T, Y) + ChargeChemicalPotential(n, T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const { return m_min_h; }
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, min_T, Y);
  }
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    return Pressure(n, max_T, Y);
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
  KOKKOS_INLINE_FUNCTION Real inverse_abar(const Real *Y) const {
    Real abar = Y[SCXN] + Y[SCXP] + Y[SCXA]/4 +
                ((Y[SCXH] > 0.0) ? Y[SCXH]/Y[SCAH] : 0.0);
    if (abar <= 0.0) {
      Kokkos::printf("EOSHelmholtz::inverse_abar: invalid mass fractions, "
                     "sum is %.5e\n", abar);
      return 1.0;
    }
    return abar;
  }

  /// Add the analytic radiation + ion terms to an electron-gas table value.
  KOKKOS_INLINE_FUNCTION Real add_rad_ion(int vi, Real var, Real n, Real T,
                                          const Real *Y) const {
    const Real Ye      = Y[SCYE];
    const Real eps_fac = Ye/mb;
    switch (vi) {
      case ECLOGP: {
        Real prad = asol/3.0*T*T*T*T;
        Real pion = n*inverse_abar(Y)*T;
        return Kokkos::log(Kokkos::exp(var) + prad + pion);
      }
      case ECENT: {
        Real srad = 4.0*asol/3.0*T*T*T/n;
        Real Yn = Y[SCXN];
        Real Yp = Y[SCXP];
        Real Ya = Y[SCXA]/4;
        Real Yh = Y[SCXH]/Y[SCAH];
        Real sn = (Yn > 0.0) ?
          Yn*(2.5 - Kokkos::log(n*Yn/g_n*Kokkos::pow(sac_const/(mn*T), 1.5))) : 0.0;
        Real sp = (Yp > 0.0) ?
          Yp*(2.5 - Kokkos::log(n*Yp/g_p*Kokkos::pow(sac_const/(mp*T), 1.5))) : 0.0;
        Real sa = (Ya > 0.0) ?
          Ya*(2.5 - Kokkos::log(n*Ya/g_a*Kokkos::pow(sac_const/(ma*T), 1.5))) : 0.0;
        Real sh = 0.0;
        if (Yh > 0.0) {
          Real mbar = mb*(1 + Y[SCEB]);
          Real mh   = (mbar - Yn*mn - Yp*mp - Ya*ma)/Yh;
          if (mh > 0.0) {
            sh = Yh*(2.5 - Kokkos::log(n*Yh/g_h*Kokkos::pow(sac_const/(mh*T), 1.5)));
          }
        }
        return Ye*var + srad + sn + sp + sa + sh;
      }
      case ECLOGEPS: {
        Real erad  = asol*T*T*T*T/(n*mb);
        Real eion  = 1.5*T*inverse_abar(Y)/mb;
        Real ebind = Y[SCEB];
        return Kokkos::log(eps_fac*Kokkos::exp(var) + erad + eion + ebind);
      }
      case ECDEPSDT: {
        Real deraddt = 4.0*asol*T*T*T/(n*mb);
        Real deiondt = 1.5*inverse_abar(Y)/mb;
        return eps_fac*var + deraddt + deiondt;
      }
      case ECDPDN: {
        Real dpiondn = T*inverse_abar(Y);
        return Ye*var + dpiondn;
      }
      case ECDPDT: {
        Real dpraddt = 4.0/3.0*asol*T*T*T;
        Real dpiondt = n*inverse_abar(Y);
        return var + dpraddt + dpiondt;
      }
      case ECETA:
      default:
        return var;
    }
  }

  KOKKOS_INLINE_FUNCTION Real eval_at_nty(int vi, Real n, Real T, const Real *Y) const {
    Real var = eval_at_lnty(vi, Kokkos::log(n*Y[SCYE]), Kokkos::log(T));
    return add_rad_ion(vi, var, n, T, Y);
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
  /// bracketing failure. The analytic terms make f nonlinear in log T inside a
  /// cell, so refine with a false-position (Anderson-Bjorck) iteration.
  KOKKOS_INLINE_FUNCTION Real temperature_from_var(int iv, Real var, Real n,
                                                   const Real *Y,
                                                   int *guess_it = nullptr) const {
    int in;
    Real wn0, wn1;
    weight_idx_ln(&wn0, &wn1, &in, Kokkos::log(n*Y[SCYE]));

    auto f = [=](int it) {
      Real var_pt = wn0*m_table(iv, in+0, it) + wn1*m_table(iv, in+1, it);
      var_pt = add_rad_ion(iv, var_pt, n, Kokkos::exp(m_log_t(it)), Y);
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
    if (flo == 0) { return Kokkos::exp(ltlo); }
    if (fhi == 0) { return Kokkos::exp(lthi); }

    Real const v_lo = wn0*m_table(iv, in+0, ilo) + wn1*m_table(iv, in+1, ilo);
    Real const v_hi = wn0*m_table(iv, in+0, ihi) + wn1*m_table(iv, in+1, ihi);
    auto g = [=](Real lt) {
      Real wt     = (lt - ltlo)/(lthi - ltlo);
      Real var_pt = (1.0 - wt)*v_lo + wt*v_hi;
      return var - add_rad_ion(iv, var_pt, n, Kokkos::exp(lt), Y);
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

 private:
  Real m_id_log_ne, m_id_log_t;
  int m_nn, m_nt;
  static constexpr Real m_min_h = 0.0;
  bool m_initialized;

  // Table storage on device.
  DvceArray1D<Real> m_log_ne;
  DvceArray1D<Real> m_log_t;
  DvceArray3D<Real> m_table;   // (ECNVARS, m_nn, m_nt)

 public:
  static constexpr Real hbarc = 197.3269804;  // MeV fm
  static constexpr Real asol = M_PI*M_PI/(15.0*hbarc*hbarc*hbarc);  // (MeV fm)^-3
  static constexpr Real min_Y_free = 1.0e-30;  // floor for Yn/Yp in log()
  static constexpr Real sac_const = hbarc*hbarc*2.0*M_PI;           // (MeV fm)^2
  static constexpr Real me = 0.5109989461;   // MeV
  // Physical nucleon masses [MeV]. The CODATA values are the defaults; when the
  // Helmholtz EOS is driven by EOSTransition these are replaced by the values
  // carried by the compose table (SetNucleonMasses), since the table energies
  // and chemical potentials were built with them.
  static constexpr Real mn_codata = 939.5654133;  // MeV
  static constexpr Real mp_codata = 938.2720813;  // MeV
  Real mn, mp;                                    // MeV
  static constexpr Real ma = 3727.379378;    // MeV
  static constexpr int g_n = 2;
  static constexpr int g_p = 2;
  static constexpr int g_a = 1;
  static constexpr int g_h = 1;
};

}  // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_EOS_HELMHOLTZ_HPP_
