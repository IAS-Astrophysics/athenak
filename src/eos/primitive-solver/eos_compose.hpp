#ifndef EOS_PRIMITIVE_SOLVER_EOS_COMPOSE_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_COMPOSE_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_compose.hpp
//  \brief Defines EOSTable, which stores information from a tabulated
//         equation of state in CompOSE format.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb, log t, and yq

#include <cstdio>
#include <string>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
#include "../../bns_nurates_fdi_ns.hpp"
#include "ps_types.hpp"
#include "eos_policy_interface.hpp"
#include "unit_system.hpp"
#include "logs.hpp"

namespace Primitive {

template<typename LogPolicy>
class EOSCompOSE : public EOSPolicyInterface, public LogPolicy, public SupportsEntropy,
                   public SupportsChemicalPotentials {
 private:
  using LogPolicy::log2_;
  using LogPolicy::exp2_;

 public:
  enum TableVariables {
    ECLOGP  = 0,  //! log (pressure / 1 MeV fm^-3)
    ECENT   = 1,  //! entropy per baryon [kb]
    ECMUB   = 2,  //! baryon chemical potential [MeV]
    ECMUQ   = 3,  //! charge chemical potential [MeV]
    ECMUL   = 4,  //! lepton chemical potential [MeV]
    ECLOGE  = 5,  //! log (total energy density / 1 MeV fm^-3)
    ECCS    = 6,  //! sound speed [c]
    ECYP    = 7,  //! proton fraction
    ECYN    = 8,  //! neutron fraction
    ECNVARS = 9
  };

  //! Where a BetaEquilibriumTrapped answer came from, reported through that
  //! function's optional status argument.
  enum NeutrinoEquilibriumStatus {
    NUEQ_INTERIOR    = 0,  //! interior solution of the full 2D system
    NUEQ_CONSTRAINED = 1,  //! solution constrained to a table edge (KKT point)
    NUEQ_ENERGY_ONLY = 2,  //! energy equation alone, solved for T at frozen Y_e
    NUEQ_FAILED      = 3,  //! no solution; T_eq is a clamped endpoint or the guess
    NUEQ_UNSUPPORTED = 4   //! split electron-pair weights without bns_nurates
  };

 protected:
  /// Constructor
  EOSCompOSE() :
      m_log_nb("log nb",1),
      m_yq("yq",1),
      m_log_t("log T",1),
      m_table("EoS table",1,1,1,1) {
    n_species = 1;
    eos_units = MakeNuclear();
    m_initialized = false;

    // These will be set properly when the table is read
    m_id_log_nb = std::numeric_limits<Real>::quiet_NaN();
    m_id_log_t = std::numeric_limits<Real>::quiet_NaN();
    m_id_yq = std::numeric_limits<Real>::quiet_NaN();
    m_nn = 0;
    m_nt = 0;
    m_ny = 0;
    m_min_h = std::numeric_limits<Real>::max();
    mb =    std::numeric_limits<Real>::quiet_NaN();
    min_n = std::numeric_limits<Real>::quiet_NaN();
    max_n = std::numeric_limits<Real>::quiet_NaN();
    min_T = std::numeric_limits<Real>::quiet_NaN();
    max_T = std::numeric_limits<Real>::quiet_NaN();
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = std::numeric_limits<Real>::quiet_NaN();
      max_Y[i] = std::numeric_limits<Real>::quiet_NaN();
    }

    // Defaults for neutrino equilibrium solver
    nu_2DNR_eps_lim  = 1.e-7;
    nu_2DNR_n_max    = 100;
    nu_bis_n_cut_max = 8;
    nu_grad_cells    = 1.0;
    nu_1D_bis_n_max  = 60;
  }

/*
  /// Destructor
  ~EOSCompOSE();
*/

  /// Temperature from energy density
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    assert (m_initialized);
    if (n < min_n) {
      return min_T;
    } else if (e <= 0.0) {
      return min_T;
    }
    Real log_e = log2_(e);
    return temperature_from_var(ECLOGE, log_e, n, Y[0]);
  }

  /// Calculate the temperature using.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    assert (m_initialized);
    if (n < min_n) {
      return min_T;
    } else if (p <= MinimumPressure(n, Y)) {
      return min_T;
    }
    Real log_p = log2_(p);
    return temperature_from_var(ECLOGP, log_p, n, Y[0]);
  }

  /// Calculate the energy density using.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    assert (m_initialized);
    Real log_e = eval_at_nty(ECLOGE, n, T, Y[0]);
    return exp2_(log_e);
  }

  /// Calculate the pressure using.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    Real log_p = eval_at_nty(ECLOGP, n, T, Y[0]);
    return exp2_(log_p);
  }

  /// Calculate the entropy per baryon using.
  KOKKOS_INLINE_FUNCTION Real Entropy(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    return eval_at_nty(ECENT, n, T, Y[0]);
  }

  /// Calculate the enthalpy per baryon using.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    Real const P = Pressure(n, T, Y);
    Real const e = Energy(n, T, Y);
    return (P + e)/n;
  }

  /// Calculate the sound speed.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    return eval_at_nty(ECCS, n, T, Y[0]);
  }

  /// Calculate the specific internal energy per unit mass
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return Energy(n, T, Y)/(mb*n) - 1;
  }

  /// Calculate the baryon chemical potential
  KOKKOS_INLINE_FUNCTION Real BaryonChemicalPotential(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    return eval_at_nty(ECMUB, n, T, Y[0]);
  }

  /// Calculate the charge chemical potential
  KOKKOS_INLINE_FUNCTION Real ChargeChemicalPotential(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    return eval_at_nty(ECMUQ, n, T, Y[0]);
  }

  /// Calculate the electron-lepton chemical potential
  KOKKOS_INLINE_FUNCTION Real ElectronLeptonChemicalPotential(Real n, Real T,
                                                              Real *Y) const {
    assert (m_initialized);
    return eval_at_nty(ECMUL, n, T, Y[0]);
  }

  /// Calculate the proton fraction
  KOKKOS_INLINE_FUNCTION Real ProtonFraction(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    // Use ternary instead of fmax: on IEEE-compliant hardware (Intel SYCL),
    // fmax(0, NaN) = NaN, so a NaN from the table lookup would propagate.
    Real yp = eval_at_nty(ECYP, n, T, Y[0]);
    return (yp > 0.0) ? yp : 0.0;
  }

  /// Calculate the neutron fraction
  KOKKOS_INLINE_FUNCTION Real NeutronFraction(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    // Use ternary instead of fmax: on IEEE-compliant hardware (Intel SYCL),
    // fmax(0, NaN) = NaN, so a NaN from the table lookup would propagate.
    Real yn = eval_at_nty(ECYN, n, T, Y[0]);
    return (yn > 0.0) ? yn : 0.0;
  }

  /// Calculate hot (neutrino trapped) beta equilibrium T_eq and Y_eq given n, e, and Yl
  //
  //  If status is not null, it receives one of NeutrinoEquilibriumStatus telling the
  //  caller where the answer came from. A constrained (table edge) solution and the
  //  energy-only fallback both count as success, so a nonzero return means no usable
  //  equilibrium was found at all.
  KOKKOS_INLINE_FUNCTION int BetaEquilibriumTrapped(Real n, Real e, Real *Yl, Real &T_eq,
                                                     Real *Y_eq, Real T_guess,
                                                     Real *Y_guess,
                                                     int *status = nullptr) const {
    const Real w_one[PEQ_NWEIGHTS] = {1.0, 1.0, 1.0, 1.0, 1.0};
    return BetaEquilibriumPartial(n, e, Yl, w_one, T_eq, Y_eq,
                                  T_guess, Y_guess, status);
  }

  /// Calculate partially-equilibrated T_eq and Y_eq: the one-parameter family of which
  /// BetaEquilibriumTrapped is the fully-trapped endpoint.
  //
  //  w holds one weight per neutrino channel, indexed by PeqWeightIndex (ps_types.hpp).
  //  All five equal to 1 is the fully trapped equilibrium, all five equal to 0 leaves
  //  the matter state untouched.
  //
  //  The caller must build the right-hand sides with the *same* weights it passes here,
  //
  //      e_rhs   = e_matter + w[PEQ_W1_NUE]*J_nue + w[PEQ_W1_ANUE]*J_anue
  //                         + w[PEQ_W1_X]*J_x
  //      Yl_rhs  = Y_e      + (w[PEQ_W0_NUE]*N_nue - w[PEQ_W0_ANUE]*N_anue)/n
  //
  //  where J and N are the neutrino energy and number densities the matter is currently
  //  in contact with. Weights and right-hand sides that disagree solve a problem that is
  //  not on the family and has no physical interpretation.
  KOKKOS_INLINE_FUNCTION int BetaEquilibriumPartial(Real n, Real e_rhs, Real *Yl_rhs,
                                                     const Real w[PEQ_NWEIGHTS],
                                                     Real &T_eq, Real *Y_eq,
                                                     Real T_guess, Real *Y_guess,
                                                     int *status = nullptr) const {
#if !ENABLE_NURATES
    // Unequal weights within a pair need bns_nurates' Fermi-Dirac integrals, whose
    // headers are on the include path only under ENABLE_NURATES. Report the request as
    // unsupported; answering the equal-weight problem instead would be a different
    // scheme under this function's name.
    if (w[PEQ_W1_NUE] != w[PEQ_W1_ANUE] || w[PEQ_W0_NUE] != w[PEQ_W0_ANUE]) {
      // T_eq and Y_eq are in-out: every other exit writes them, and the eos.hpp wrapper
      // rescales T_eq unconditionally, so leaving them alone would hand the caller its
      // own guess multiplied by a unit conversion.
      T_eq = T_guess;
      Y_eq[0] = Y_guess[0];
      if (status != nullptr) {
        *status = NUEQ_UNSUPPORTED;
      }
      return 1;
    }
#endif
    const int n_at = 16;
    Real vec_guess[n_at][2] = {
      {1.00e0, 1.00e0},
      {0.90e0, 1.25e0},
      {0.90e0, 1.10e0},
      {0.90e0, 1.00e0},
      {0.90e0, 0.90e0},
      {0.90e0, 0.75e0},
      {0.75e0, 1.25e0},
      {0.75e0, 1.10e0},
      {0.75e0, 1.00e0},
      {0.75e0, 0.90e0},
      {0.75e0, 0.75e0},
      {0.50e0, 1.25e0},
      {0.50e0, 1.10e0},
      {0.50e0, 1.00e0},
      {0.50e0, 0.90e0},
      {0.50e0, 0.75e0},
    };

    // ierr = 0    Equilibrium found
    // ierr = 1    Equilibrium not found
    int ierr = 1;
    int eq_status = NUEQ_FAILED;

    Real x0[2], x1[2]; // T,Ye guess and T,Ye result

    // A constrained (table edge) solution found along the way, kept as a fallback in
    // case no interior solution turns up.
    bool have_kkt = false;
    Real x_kkt[2] = {0.0};

    for (int na = 0; na < n_at; ++na) {
      x0[0] = vec_guess[na][0] * T_guess;
      x0[1] = vec_guess[na][1] * Y_guess[0];

      int ierr_try = trapped_equilibrium_2DNR(n, e_rhs, Yl_rhs[0], w, x0, x1);

      if (ierr_try == 0) {
        ierr = 0;
        eq_status = NUEQ_INTERIOR;
        break;
      }
      // ierr = 2 is a KKT point: the iterate is pinned to a table boundary and the
      // Newton step points out of the domain, so this is a legitimate constrained
      // solution rather than a failure. Keep the first one, but let the remaining
      // guesses look for an interior root.
      if (ierr_try == 2 && !have_kkt) {
        x_kkt[0] = x1[0];
        x_kkt[1] = x1[1];
        have_kkt = true;
      }
    }

    if (ierr == 0) {          // Success: interior solution
      T_eq = x1[0];
      // Maybe in the future we could explicitly conserve the lepton numbers
      Y_eq[0] = x1[1];
    } else if (have_kkt) {    // Success: constrained solution on a table edge
      T_eq = x_kkt[0];
      Y_eq[0] = x_kkt[1];
      ierr = 0;
      eq_status = NUEQ_CONSTRAINED;
    } else {
      // The 2D solve failed from every guess. Do not simply return (T_guess,
      // Y_guess): that is the *unequilibrated* matter state, and this function is only
      // called where the local blackbody is already known to be a bad description, so
      // handing it back gives an emissivity that can be off by any factor, in either
      // direction, depending on the sign of the energy imbalance. Freeze Y_e instead
      // and solve the energy equation alone for T, which bisection can always do
      // provided the state is inside the table.
      T_eq = T_guess;
      Y_eq[0] = Y_guess[0];
      ierr = trapped_equilibrium_1D(n, e_rhs, w, Y_guess[0], T_eq);
      eq_status = (ierr == 0) ? NUEQ_ENERGY_ONLY : NUEQ_FAILED;
    }

    if (status != nullptr) {
      *status = eq_status;
    }

    return ierr;
  }

  /// Calculate trapped neutrino net number and energy densities
  KOKKOS_INLINE_FUNCTION void TrappedNeutrinos(Real n, Real T, Real *Y, Real n_nu[3],
                                                Real e_nu[3]) const {
    Real mu_le = ElectronLeptonChemicalPotential(n, T, Y);
    Real eta_e = mu_le/T;
    Real eta_e2 = eta_e*eta_e;

    Real eta_m = 0.0;
    Real eta_m2 = 0.0;

    Real eta_t = 0.0;
    Real eta_t2 = 0.0;

    Real T3 = T*T*T;
    Real T4 = T3*T;

    // n_nu_e   - n_anu_e   [fm^-3]
    n_nu[0] = nu_n_prefactor * T3 * (eta_e * (pi2 + eta_e2));
    // n_nu_mu  - n_anu_mu  [fm^-3]
    n_nu[1] = nu_n_prefactor * T3 * (eta_m * (pi2 + eta_m2));
    // n_nu_tau - n_anu_tau [fm^-3]
    n_nu[2] = nu_n_prefactor * T3 * (eta_t * (pi2 + eta_t2));

    // e_nu_e   + e_anu_e   [MeV fm^-3]
    e_nu[0] = nu_e_prefactor * T4 * (nu_7pi4_60 + 0.5*eta_e2*(pi2 + 0.5*eta_e2));
    // e_nu_mu  + e_anu_mu  [MeV fm^-3]
    e_nu[1] = nu_e_prefactor * T4 * (nu_7pi4_60 + 0.5*eta_m2*(pi2 + 0.5*eta_m2));
    // e_nu_tau + e_anu_tau [MeV fm^-3]
    e_nu[2] = nu_e_prefactor * T4 * (nu_7pi4_60 + 0.5*eta_t2*(pi2 + 0.5*eta_t2));

    return;
  }

  /// Get the minimum enthalpy per baryon.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    assert (m_initialized);
    return m_min_h;
  }

  /// Get the minimum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, min_T, Y);
  }

  /// Get the maximum pressure at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    return Pressure(n, max_T, Y);
  }

  /// Get the minimum energy at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return Energy(n, min_T, Y);
  }

  /// Get the maximum energy at a given density and composition
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    return Energy(n, max_T, Y);
  }

 public:
  /// Reads the table file.
  void ReadTableFromFile(std::string fname);

  /// Get the raw number density
  KOKKOS_INLINE_FUNCTION DvceArray1D<Real> const GetRawLogNumberDensity() const {
    return m_log_nb;
  }

  /// Get the raw charge fraction
  KOKKOS_INLINE_FUNCTION DvceArray1D<Real> const GetRawYq() const {
    return m_yq;
  }
  /// Get the raw temperature
  KOKKOS_INLINE_FUNCTION DvceArray1D<Real> const GetRawLogTemperature() const {
    return m_log_t;
  }
  /// Get the raw table data
  KOKKOS_INLINE_FUNCTION DvceArray4D<Real> const GetRawTable() const {
    return m_table;
  }

  // Indexing used to access the data
  KOKKOS_INLINE_FUNCTION ptrdiff_t index(int iv, int in, int iy, int it) const {
    return it + m_nt*(iy + m_ny*(in + m_nn*iv));
  }

  /// Check if the EOS has been initialized properly.
  KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
    return m_initialized;
  }

  /// Set the number of species. Throw an exception if
  /// the number of species is invalid.
  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
    // Number of species must be within limits
    assert (n<=MAX_SPECIES && n>=0);

    // Only 1 species is implemented for tables
    assert (n == 1);

    n_species = n;
    return;
  }

  /// Set the EOS unit system.
  KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(UnitSystem units) {
    eos_units = units;
  }

 private:
  /// Low level evaluation function, not intended for outside use
  KOKKOS_INLINE_FUNCTION Real eval_at_nty(int vi, Real n, Real T, Real Yq) const {
    Real log_n = log2_(n);
    Real log_T = log2_(T);
    return eval_at_lnty(vi, log_n, log_T, Yq);
  }
  /// Low level evaluation function, not intended for outside use
  KOKKOS_INLINE_FUNCTION Real eval_at_lnty(int iv, Real log_n, Real log_t, Real yq)
      const {
    int in, iy, it;
    Real wn0, wn1, wy0, wy1, wt0, wt1;


    weight_idx_ln(&wn0, &wn1, &in, log_n);
    weight_idx_yq(&wy0, &wy1, &iy, yq);
    weight_idx_lt(&wt0, &wt1, &it, log_t);

    return
      wn0 * (wy0 * (wt0 * m_table(iv, in+0, iy+0, it+0)   +
                    wt1 * m_table(iv, in+0, iy+0, it+1))  +
             wy1 * (wt0 * m_table(iv, in+0, iy+1, it+0)   +
                    wt1 * m_table(iv, in+0, iy+1, it+1))) +
      wn1 * (wy0 * (wt0 * m_table(iv, in+1, iy+0, it+0)   +
                    wt1 * m_table(iv, in+1, iy+0, it+1))  +
             wy1 * (wt0 * m_table(iv, in+1, iy+1, it+0)   +
                    wt1 * m_table(iv, in+1, iy+1, it+1)));
  }

  /// Evaluate interpolation weight for density
  KOKKOS_INLINE_FUNCTION void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n)
      const {
    *in = (log_n - m_log_nb(0))*m_id_log_nb;
    // Clamp in. Note that we check m_nn - 2, not m_nn - 1, because all calculations will
    // use in and in+1.
    if (*in < 0) {
      *in = 0;
    } else if (*in > static_cast<int>(m_nn) - 2) {
      *in = m_nn - 2;
    }
    *w1 = (log_n - m_log_nb(*in))*m_id_log_nb;
    *w0 = 1.0 - (*w1);
    return;
  }
  /// Evaluate interpolation weight for composition
  KOKKOS_INLINE_FUNCTION void weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const {
    *iy = (yq - m_yq(0))*m_id_yq;
    // Clamp iy. See weight_idx_ln.
    if (*iy < 0) {
      *iy = 0;
    } else if (*iy > static_cast<int>(m_ny) - 2) {
      *iy = m_ny - 2;
    }
    *w1 = (yq - m_yq(*iy))*m_id_yq;
    *w0 = 1.0 - (*w1);
    return;
  }

  /// Evaluate interpolation weight for temperature
  KOKKOS_INLINE_FUNCTION void weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t)
      const {
    *it = (log_t - m_log_t(0))*m_id_log_t;
    // Clamp it. See weight_idx_ln.
    if (*it < 0) {
      *it = 0;
    } else if (*it > static_cast<int>(m_nt) - 2) {
      *it = m_nt - 2;
    }
    *w1 = (log_t - m_log_t(*it))*m_id_log_t;
    *w0 = 1.0 - (*w1);
    return;
  }

  /// Low level function, not intended for outside use
  KOKKOS_INLINE_FUNCTION Real temperature_from_var(int iv, Real var, Real n, Real Yq)
      const {
    int in, iy;
    Real wn0, wn1, wy0, wy1;
    Real log_n = log2_(n);
    weight_idx_ln(&wn0, &wn1, &in, log_n);
    weight_idx_yq(&wy0, &wy1, &iy, Yq);

    auto f = [=](int it){
      Real var_pt =
        wn0 * (wy0 * m_table(iv, in+0, iy+0, it)  +
               wy1 * m_table(iv, in+0, iy+1, it)) +
        wn1 * (wy0 * m_table(iv, in+1, iy+0, it)  +
               wy1 * m_table(iv, in+1, iy+1, it));

      return var - var_pt;
    };

    int ilo = 0;
    int ihi = m_nt-1;
    Real flo = f(ilo);
    Real fhi = f(ihi);
    while (flo*fhi>0) {
      if (ilo == ihi - 1) {
        break;
      } else {
        ilo += 1;
        flo = f(ilo);
      }
    }

    if (flo*fhi>0.0 && (iv==ECLOGP || iv==ECLOGE)) {
      /*if (iv == ECLOGE) {
        Real vlo = eval_at_nty(iv,n,min_T,Yq);
        Real vhi = eval_at_nty(iv,n,max_T,Yq);
        Kokkos::printf("Testing maxima and minima:\n"
                       "  iv = %i\n"
                       "  var = %20.17g\n"
                       "  minimum: %20.17g\n"
                       "  maximum: %20.17g\n",
                       iv, var, vlo, vhi);
      }*/
      //if (var <= eval_at_nty(iv,n,min_T,Yq)) {
      if (f(0) <= 0) {
        return min_T;
      } else if (f(m_nt-1) >= 0) {  // else if (var >= eval_at_nty(iv,n,max_T,Yq)) {
        return max_T;
      }
    }

    if (flo*fhi > 0) {
      int imin = 0;
      Real fmin = f(imin);
      Kokkos::printf("There's a problem with temperature bracketing!\n"
                     "  iv = %i\n"
                     "  var = %20.17g\n"
                     "  n = %20.17g\n"
                     "  Yq = %20.17g\n"
                     "  imin = %i\n"
                     "  ilo = %i\n"
                     "  ihi = %i\n"
                     "  fmin = %20.17g\n"
                     "  flo = %20.17g\n"
                     "  fhigh = %20.17g\n", iv, var, n , Yq, imin, ilo, ihi, fmin, flo,
                     fhi);
      assert(flo*fhi <= 0);
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
    assert(ihi - ilo == 1);
    Real lthi = m_log_t[ihi];
    Real ltlo = m_log_t[ilo];

    //Real lt = m_log_t[ilo] - flo*(lthi - ltlo)/(fhi - flo);
    Real lt = (ltlo*fhi - lthi*flo)/(fhi - flo);
    return exp2_(lt);
  }

  /// Low level functions for neutrino equilibrium, not intended for outside use.
  /// See BetaEquilibriumPartial for what the weights mean.
  KOKKOS_INLINE_FUNCTION int trapped_equilibrium_2DNR(Real n, Real e_rhs, Real Yle_rhs,
                                                       const Real w[PEQ_NWEIGHTS],
                                                       Real x0[2], Real x1[2]) const {
    int ierr = 1;

    // initialize the solution
    x1[0] = x0[0];
    x1[1] = x0[1];
    bool KKT = false;

    //compute the initial residuals
    Real y[2] = {0.0};
    func_eq_weak(n,e_rhs,Yle_rhs,w,x1,y);

    // compute the error from the residuals
    Real err = error_func_eq_weak(Yle_rhs,y);

    // initialize the iteration variables
    int n_iter = 0;
    Real J[2][2] = {};
    Real invJ[2][2] = {};
    Real dx1[2] = {0.0};
    Real dxa[2] = {0.0};
    Real norm[2] = {0.0};
    Real x1_tmp[2] = {0.0};

    // loop until a low enough residual is found or until  a too
    // large number of steps has been performed
    while (err>nu_2DNR_eps_lim && n_iter<=nu_2DNR_n_max && !KKT) {
      // compute the Jacobian
      ierr = jacobi_eq_weak(n,e_rhs,Yle_rhs,w,x1,J);
      if (ierr != 0) {
        return ierr;
      }

      // compute and check the determinant of the Jacobian
      Real det = J[0][0]*J[1][1] - J[0][1]*J[1][0];
      if (det==0.0) {
        ierr = 1;
        return ierr;
      }

      // invert the Jacobian
      inv_jacobi(det,J,invJ);

      // compute the next step
      dx1[0] = - (invJ[0][0]*y[0] + invJ[0][1]*y[1]);
      dx1[1] = - (invJ[1][0]*y[0] + invJ[1][1]*y[1]);

      // check if we are the boundary of the table
      if (x1[0] == min_T) {
        norm[0] = -1.0;
      } else if (x1[0] == max_T) {
        norm[0] = 1.0;
      } else {
        norm[0] = 0.0;
      }

      if (x1[1] == min_Y[0]) {
        norm[1] = -1.0;
      } else if (x1[1] == max_Y[0]) {
        norm[1] = 1.0;
      } else {
        norm[1] = 0.0;
      }

      // Take the part of the gradient that is active (pointing within the eos domain)
      Real scal = norm[0]*norm[0] + norm[1]*norm[1];
      if (scal <= 0.5) { // this can only happen if norm = (0, 0)
        scal = 1.0;
      }
      dxa[0] = dx1[0] - (dx1[0]*norm[0] + dx1[1]*norm[1])*norm[0]/scal;
      dxa[1] = dx1[1] - (dx1[0]*norm[0] + dx1[1]*norm[1])*norm[1]/scal;

      if ((dxa[0]*dxa[0] + dxa[1]*dxa[1]) <
          (nu_2DNR_eps_lim*nu_2DNR_eps_lim * (dx1[0]*dx1[0] + dx1[1]*dx1[1]))) {
        KKT = true;
        ierr = 2;
        return ierr;
      }

      // Backtracking line search along the Newton direction. Every trial point must
      // be measured from the iterate as it was on entry: advancing x1 inside the loop
      // makes each pass step *further* along dx1 rather than retreating, visiting
      // x + dx, x + 1.5 dx, x + 1.75 dx, ... -> x + 2 dx. That safeguard only engages
      // once the full step has already failed to reduce the residual, i.e. exactly
      // when a genuine backtrack is needed, and if no pass improves the error the loop
      // leaves x1 at the last and worst trial point.
      const Real x_in[2] = {x1[0], x1[1]};
      const Real err_old = err;
      Real fac_cut = 1.0;
      bool improved = false;

      for (int n_cut = 0; n_cut <= nu_bis_n_cut_max; ++n_cut, fac_cut *= 0.5) {
        // the variation of x1 is divided by a power of 2 if the
        // error is not decreasing along the gradient direction
        x1_tmp[0] = x_in[0] + (dx1[0]*fac_cut);
        x1_tmp[1] = x_in[1] + (dx1[1]*fac_cut);

        // check if the next step calculation had problems
        if (isnan(x1_tmp[0]) || isnan(x1_tmp[1])) {
          ierr = 1;
          return ierr;
        }

        // tabBoundsFlag = enforceTableBounds(rho, x1_tmp[0], x1_tmp[1]);
        // Use ternary instead of fmin/fmax: on IEEE-compliant hardware
        // (Intel SYCL), fmax(NaN, x) = NaN, so a NaN iterate would not be
        // clamped back to table bounds and would poison subsequent iterations.
        x1_tmp[0] = (x1_tmp[0] > min_T) ? x1_tmp[0] : min_T;
        x1_tmp[0] = (x1_tmp[0] < max_T) ? x1_tmp[0] : max_T;
        x1_tmp[1] = (x1_tmp[1] > min_Y[0]) ? x1_tmp[1] : min_Y[0];
        x1_tmp[1] = (x1_tmp[1] < max_Y[0]) ? x1_tmp[1] : max_Y[0];

        // compute the residuals and the error for the trial point
        Real y_tmp[2] = {0.0};
        func_eq_weak(n,e_rhs,Yle_rhs,w,x1_tmp,y_tmp);
        Real err_tmp = error_func_eq_weak(Yle_rhs,y_tmp);

        // accept the first cut that makes progress
        if (err_tmp < err_old) {
          x1[0] = x1_tmp[0];
          x1[1] = x1_tmp[1];
          y[0] = y_tmp[0];
          y[1] = y_tmp[1];
          err = err_tmp;
          improved = true;
          break;
        }
      }

      if (!improved) {
        // No cut along the Newton direction reduces the residual. Leave x1 at the
        // iterate that got here and let the caller's restart ladder try a different
        // initial guess, rather than accepting a worse point.
        ierr = 1;
        return ierr;
      }

      // update the iteration
      n_iter += 1;
    }

    if (err <= nu_2DNR_eps_lim) {
      ierr = 0;
    } else {
      ierr = 1;
    }

    return ierr;
  }

  /// Energy residual of the partial equilibrium at fixed Y_e. This is the second
  /// component of func_eq_weak, which does not depend on Yle_rhs.
  KOKKOS_INLINE_FUNCTION Real energy_eq_weak(Real n, Real e_rhs,
                                              const Real w[PEQ_NWEIGHTS],
                                              Real T, Real Ye) const {
    Real x[2] = {T, Ye};
    Real y[2] = {0.0};
    // Yle_rhs enters y[0] only, so its value is irrelevant here. The whole weight array
    // goes through, so this is the same energy equation the 2D solve uses.
    func_eq_weak(n, e_rhs, 1.0, w, x, y);
    return y[1];
  }

  /// Fallback for when the 2D Newton solve fails from every guess: freeze Y_e and
  /// solve the energy equation alone for T, by bisection in log2(T) -- the variable
  /// the table is uniform in. Bisection needs only a sign change, not monotonicity,
  /// so it cannot fail as long as the state is inside the table.
  ///
  /// Returns 0 if a root was bracketed and refined. Returns 1 if the residual has no
  /// sign change across the table, in which case the state is out of range and T_eq
  /// is set to whichever endpoint comes closest, or if the endpoint residuals are
  /// NaN, in which case T_eq is left as the caller set it.
  KOKKOS_INLINE_FUNCTION int trapped_equilibrium_1D(Real n, Real e_rhs,
                                                     const Real w[PEQ_NWEIGHTS],
                                                     Real Ye, Real &T_eq) const {
    Real ya = energy_eq_weak(n, e_rhs, w, min_T, Ye);
    Real yb = energy_eq_weak(n, e_rhs, w, max_T, Ye);

    if (isnan(ya) || isnan(yb)) {
      return 1;
    }

    if (ya*yb > 0.0) {
      T_eq = (abs(ya) < abs(yb)) ? min_T : max_T;
      return 1;
    }

    Real la = log2_(min_T);
    Real lb = log2_(max_T);

    for (int n_bis = 0; n_bis < nu_1D_bis_n_max; ++n_bis) {
      Real lm = 0.5*(la + lb);
      Real Tm = exp2_(lm);
      Real ym = energy_eq_weak(n, e_rhs, w, Tm, Ye);

      if (isnan(ym)) {
        break;
      }
      if (abs(ym) <= nu_2DNR_eps_lim) {
        T_eq = Tm;
        return 0;
      }
      if (ya*ym <= 0.0) {
        lb = lm;
        yb = ym;
      } else {
        la = lm;
        ya = ym;
      }
    }

    T_eq = exp2_(0.5*(la + lb));
    return 0;
  }

  /// Pair mean of the two weights a channel's species carry, and half their
  /// difference. The residual and the Jacobian must agree on these, so they are defined
  /// here and nowhere else.
  KOKKOS_INLINE_FUNCTION Real peq_mean(const Real w[PEQ_NWEIGHTS], int i_nu,
                                        int i_anu) const {
    return 0.5*(w[i_nu] + w[i_anu]);
  }

  KOKKOS_INLINE_FUNCTION Real peq_half_diff(const Real w[PEQ_NWEIGHTS], int i_nu,
                                             int i_anu) const {
    return 0.5*(w[i_nu] - w[i_anu]);
  }

  // The three Fermi-Dirac combinations that carry the half-difference of a pair's
  // weights. Each is the partner of a combination available in closed form:
  //
  //   F_1(eta) + F_1(-eta) = pi^2/6 + eta^2/2
  //   F_2(eta) - F_2(-eta) = eta (pi^2 + eta^2)/3
  //   F_3(eta) + F_3(-eta) = 7pi^4/60 + pi^2 eta^2/2 + eta^4/4
  //
  // which are Bernoulli polynomials, from the polylogarithm inversion formula, and are
  // the exact terms the residuals carry. bns_nurates' FDI_p* reflect on these same
  // polynomials, so one rational evaluation at negative argument supplies the whole of
  // the remaining, decaying part.
  //
  // eta2 and P3 are passed in because the residual has them already.
  //
  // Without bns_nurates all three return 0.0, which is safe only because
  // BetaEquilibriumPartial refuses unequal weights within a pair in that build: the
  // half-difference multiplying them is then exactly zero. Keep those two guards
  // together.

  /// F_2(eta) + F_2(-eta), even.
  KOKKOS_INLINE_FUNCTION Real fermi_sum_2(Real eta, Real eta2) const {
#if ENABLE_NURATES
    const Real a = Kokkos::fabs(eta);
    return 2.0*bns_nurates::FDI_p2(-a) + a*(pi2 + eta2)/3.0;
#else
    static_cast<void>(eta); static_cast<void>(eta2);
    return 0.0;
#endif
  }

  /// F_3(eta) - F_3(-eta), odd. P3 = F_3(eta) + F_3(-eta).
  KOKKOS_INLINE_FUNCTION Real fermi_diff_3(Real eta, Real P3) const {
#if ENABLE_NURATES
    const Real d = P3 - 2.0*bns_nurates::FDI_p3(-Kokkos::fabs(eta));
    return (eta < 0.0) ? -d : d;
#else
    static_cast<void>(eta); static_cast<void>(P3);
    return 0.0;
#endif
  }

  /// F_1(eta) - F_1(-eta), odd. Equals (dS_2/deta)/2, by F_k' = k F_{k-1}.
  KOKKOS_INLINE_FUNCTION Real fermi_diff_1(Real eta, Real eta2) const {
#if ENABLE_NURATES
    const Real d = nu_pi2_6 + 0.5*eta2 -
                   2.0*bns_nurates::FDI_p1(-Kokkos::fabs(eta));
    return (eta < 0.0) ? -d : d;
#else
    static_cast<void>(eta); static_cast<void>(eta2);
    return 0.0;
#endif
  }

  // Each residual pairs a polynomial, which is the exact pair combination the closed
  // forms give, with the combination that is not polynomial, which carries the
  // half-difference of the pair's two weights:
  //
  //   lepton:  wbar_0 (N_+ - N_-) + dw_0 (N_+ + N_-),  P_2 exact, S_2 from FDI_p2
  //   energy:  wbar_1 (J_+ + J_-) + dw_1 (J_+ - J_-),  P_3 exact, D_3 from FDI_p3
  //
  // Both terms are evaluated unconditionally. Equal weights make dw exactly zero and
  // the second term contributes nothing, so a branch on it would only trade three
  // rational evaluations for warp divergence. |eta| stays far below the argument at
  // which S_2 ~ |eta|^3/3 could overflow, so there is no 0 * inf to guard against.
  KOKKOS_INLINE_FUNCTION void func_eq_weak(Real n, Real e_rhs, Real Yle_rhs,
                                            const Real w[PEQ_NWEIGHTS],
                                            Real x[2], Real y[2]) const {
    Real T = x[0];

    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = x[1];

    Real mu_l = ElectronLeptonChemicalPotential(n, T, Y);
    Real e = Energy(n, T, Y);
    Real eta = mu_l/T;
    Real eta2 = eta*eta;

    const Real wbar_E = peq_mean(w, PEQ_W1_NUE, PEQ_W1_ANUE);
    const Real dw_E = peq_half_diff(w, PEQ_W1_NUE, PEQ_W1_ANUE);
    const Real wbar_L = peq_mean(w, PEQ_W0_NUE, PEQ_W0_ANUE);
    const Real dw_L = peq_half_diff(w, PEQ_W0_NUE, PEQ_W0_ANUE);
    const Real w_E_x = w[PEQ_W1_X];

    Real t3 = T*T*T;
    Real t4 = t3*T;

    // The factor 3 on nu_n_prefactor undoes the 1/3 it carries for P_2.
    const Real P3 = nu_7pi4_60 + 0.5*eta2*(pi2 + 0.5*eta2);
    y[0] = Y[0] + (wbar_L*nu_n_prefactor*t3*eta*(pi2 + eta2) +
                   dw_L*3.0*nu_n_prefactor*t3*fermi_sum_2(eta, eta2))/n - Yle_rhs;
    y[1] = (e + nu_e_prefactor*t4*(wbar_E*P3 + w_E_x*nu_7pi4_30 +
                                   dw_E*fermi_diff_3(eta, P3)))/e_rhs - 1.0;

    return;
  }

  /// Scalar error the Newton iteration converges on. The energy residual is already
  /// relative, the lepton one is not, hence the division by Yle_rhs on that alone.
  KOKKOS_INLINE_FUNCTION Real error_func_eq_weak(Real Yle_rhs, Real y[2]) const {
    Real err = abs(y[0]/Yle_rhs) + abs(y[1]);
    return err;
  }

  KOKKOS_INLINE_FUNCTION int jacobi_eq_weak(Real n, Real e_rhs, Real Yle_rhs,
                                             const Real w[PEQ_NWEIGHTS],
                                             Real x[2], Real J[2][2]) const {
    int ierr = 0;

    Real T = x[0];
    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = x[1];

    if (isnan(T)) {
      ierr = 1;
      return ierr;
    }

    Real mu_l = ElectronLeptonChemicalPotential(n, T, Y);
    Real eta = mu_l/T;
    Real eta2 = eta*eta;

    if (isnan(eta)) {
      ierr = 1;
      return ierr;
    }

    Real detadt,detadye,dedt,dedye;
    ierr = eta_e_gradient(n,T,Y,eta,detadt,detadye,dedt,dedye);
    if (ierr != 0) {
      return ierr;
    }

    Real T2 = T*T;
    Real T3 = T2*T;
    Real T4 = T3*T;

    const Real wbar_E = peq_mean(w, PEQ_W1_NUE, PEQ_W1_ANUE);
    const Real dw_E = peq_half_diff(w, PEQ_W1_NUE, PEQ_W1_ANUE);
    const Real wbar_L = peq_mean(w, PEQ_W0_NUE, PEQ_W0_ANUE);
    const Real dw_L = peq_half_diff(w, PEQ_W0_NUE, PEQ_W0_ANUE);
    const Real w_E_x = w[PEQ_W1_X];

    // The derivatives of the half-difference terms are exact: F_k' = k F_{k-1} gives
    // dS_2/deta = 2 D_1 and dD_3/deta = 3 S_2, so S_2 serves both the lepton row and
    // the energy row and nothing is differentiated by hand.
    const Real P3 = nu_7pi4_60 + 0.5*eta2*(pi2 + 0.5*eta2);
    const Real S2 = fermi_sum_2(eta, eta2);
    const Real D3 = fermi_diff_3(eta, P3);
    const Real dS2 = 2.0*fermi_diff_1(eta, eta2);
    const Real c_n3 = 3.0*nu_n_prefactor;

    J[0][0] = (wbar_L*nu_n_prefactor*T2*(3.e0*eta*(pi2+eta2) +
                                         T*(pi2+3.e0*eta2)*detadt) +
               dw_L*c_n3*(3.0*T2*S2 + T3*dS2*detadt))/n;
    J[0][1] = 1.e0 + (wbar_L*nu_n_prefactor*T3*(pi2+3.e0*eta2)*detadye +
                      dw_L*c_n3*T3*dS2*detadye)/n;

    // nu_7pi4_15 is 4x the nu_e pair term and nu_14pi4_15 is 4x the heavy pair term, so
    // they take wbar_E and w_E_x respectively; the eta-dependent terms are the nu_e
    // pair's alone.
    J[1][0] = (dedt + nu_e_prefactor*T3*(wbar_E*nu_7pi4_15 + w_E_x*nu_14pi4_15 +
               wbar_E*2.e0*eta2*(pi2+0.5*eta2) +
               wbar_E*eta*T*(pi2+eta2)*detadt) +
               dw_E*nu_e_prefactor*(4.0*T3*D3 + 3.0*T4*S2*detadt))/e_rhs;
    J[1][1] = (dedye + wbar_E*nu_e_prefactor*T4*eta*(pi2+eta2)*detadye +
               dw_E*nu_e_prefactor*3.0*T4*S2*detadye)/e_rhs;

    return ierr;
  }

  KOKKOS_INLINE_FUNCTION int eta_e_gradient(Real n, Real T, Real *Y, Real eta,
                                             Real &deta_dT, Real &deta_dYe, Real &de_dT,
                                             Real &de_dYe) const {
    int ierr=1;

    // Tie the finite-difference steps to the table spacing. The interpolation is
    // trilinear in (log2 nb, Yq, log2 T), so a step narrower than one cell just
    // returns that cell's slope: the derivative comes out piecewise constant, with an
    // O(10%) jump at every cell boundary, and Newton sees a non-Lipschitz Jacobian. A
    // secant of fixed width in the table's own variables is continuous and piecewise
    // linear in the evaluation point instead, because the interpolant is continuous.
    // The narrower step buys precision the table does not contain.
    const Real Ye_delta = 0.5*nu_grad_cells/m_id_yq;
    const Real T_fac = Kokkos::exp2(0.5*nu_grad_cells/m_id_log_t);

    Real Y1[MAX_SPECIES] = {0.0};
    Real Y2[MAX_SPECIES] = {0.0};

    // Use ternary instead of fmax/fmin: on IEEE-compliant hardware (Intel
    // SYCL), fmax(NaN, x) = NaN, so NaN inputs would propagate through.
    Real Ye_lo = Y[0] - Ye_delta;
    Y1[0] = (Ye_lo > min_Y[0]) ? Ye_lo : min_Y[0];
    Real mu_l1 = ElectronLeptonChemicalPotential(n, T, Y1);
    Real e1 = Energy(n, T, Y1);

    Real Ye_hi = Y[0] + Ye_delta;
    Y2[0] = (Ye_hi < max_Y[0]) ? Ye_hi : max_Y[0];
    Real mu_l2 = ElectronLeptonChemicalPotential(n, T, Y2);
    Real e2 = Energy(n, T, Y2);

    Real dmu_l_dYe = (mu_l2-mu_l1)/(Y2[0] - Y1[0]);
    de_dYe         = (e2-e1)/(Y2[0] - Y1[0]);

    // The secant is symmetric in log2(T), and the divisions below use the actual
    // T2 - T1, so clamping at a table edge degrades gracefully to a one-sided
    // difference of the correct width.
    Real T_lo = T/T_fac;
    Real T1 = (T_lo > min_T) ? T_lo : min_T;
    mu_l1 = ElectronLeptonChemicalPotential(n, T1, Y);
    e1 = Energy(n, T1, Y);

    Real T_hi = T*T_fac;
    Real T2 = (T_hi < max_T) ? T_hi : max_T;
    mu_l2 = ElectronLeptonChemicalPotential(n, T2, Y);
    e2 = Energy(n, T2, Y);

    Real dmu_l_dT   = (mu_l2 - mu_l1)/(T2 - T1);
    de_dT          = (e2 - e1)/(T2 - T1);

    // eta = mu_le/T, so d(eta)/dT = (1/T) dmu_le/dT - mu_le/T^2 = (dmu_le/dT - eta)/T
    deta_dT  = (dmu_l_dT - eta )/T; // [1/MeV]
    deta_dYe = dmu_l_dYe/T;      // [-]

    if (isnan(deta_dT)||isnan(deta_dYe)||isnan(de_dT)||isnan(de_dYe)) {
      ierr = 1;
    } else {
      ierr = 0;
    }

    return ierr;
  }

  KOKKOS_INLINE_FUNCTION void inv_jacobi(Real det, Real J[2][2], Real invJ[2][2]) const {
    Real inv_det = 1.0/det;
    invJ[0][0] =  J[1][1]*inv_det;
    invJ[1][1] =  J[0][0]*inv_det;
    invJ[0][1] = -J[0][1]*inv_det;
    invJ[1][0] = -J[1][0]*inv_det;
  }

 private:
  // Inverse of table spacing
  Real m_id_log_nb, m_id_yq, m_id_log_t;
  // Table size
  size_t m_nn, m_nt, m_ny;
  // Minimum enthalpy per baryon
  Real m_min_h;

  // bool to protect against access of uninitialized table and prevent repeated reading
  // of table
  bool m_initialized;

  // Table storage on DEVICE.
  DvceArray1D<Real> m_log_nb;
  DvceArray1D<Real> m_yq;
  DvceArray1D<Real> m_log_t;
  DvceArray4D<Real> m_table;

 private:
  // Neutrino equilibrium parameters
  Real nu_2DNR_eps_lim; // tolerance in 2D NR (required for 1e-12 err in T)
  int nu_2DNR_n_max;    // Newton-Raphson max number of iterations
  int nu_bis_n_cut_max; // Bisection max number of iterations
  // Width of the Jacobian finite differences, in table cells. One cell is the
  // resolution of the interpolant itself; see eta_e_gradient.
  Real nu_grad_cells;
  // Max bisections in the 1D energy-only fallback. 60 halvings of the log2(T) range
  // take the interval below double round-off, so this is a hard backstop, not a
  // tolerance: the loop normally exits on the residual.
  int nu_1D_bis_n_max;

  // Neutrino equilibrium physical constants
  const Real hc_mevfm = 1.23984172e3;           // hc    [MeV fm] (not reduced)
  const Real pi       = 3.14159265358979323846; // pi    [-]
  const Real pi2      = pi*pi;                  // pi**2 [-]
  const Real pi4      = pi2*pi2;                // pi**4 [-]

  // 4/3 *pi/(hc)**3 [1/MeV^3/fm^3]
  const Real nu_n_prefactor = 4.0/3.0*pi/(hc_mevfm*hc_mevfm*hc_mevfm);
  // 4*pi/(hc)**3    [1/MeV^3 fm^3]
  const Real nu_e_prefactor = 4.0*pi/(hc_mevfm*hc_mevfm*hc_mevfm);

  const Real nu_pi2_6 = pi2/6.0;         // pi**2/6     [-]
  const Real nu_7pi4_60 = 7.0*pi4/60.0;  // 7*pi**4/60  [-]
  const Real nu_7pi4_30 = 7.0*pi4/30.0;  // 7*pi**4/30  [-]
  const Real nu_7pi4_15 = 7.0*pi4/15.0;  // 7*pi**4/15  [-]
  const Real nu_14pi4_15 = 14.0*pi4/15.0; // 14*pi**4/15 [-]
};

}; // namespace Primitive

#endif //EOS_PRIMITIVE_SOLVER_EOS_COMPOSE_HPP_
