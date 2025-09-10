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

#include <string>
#include <limits>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
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
    ECNVARS = 7
  };

 protected:
  /// Constructor
  EOSCompOSE() :
      m_log_nb("log nb",1),
      m_log_t("log T",1),
      m_yq("yq",1),
      m_table("EoS table",1,1,1,1) {
    n_species = 1;
    eos_units = MakeNuclear();
    m_initialized = false;

    // These will be set properly when the table is read
    m_id_log_nb = std::numeric_limits<Real>::quiet_NaN();
    m_id_log_t = std::numeric_limits<Real>::quiet_NaN();
    m_id_yq = std::numeric_limits<Real>::quiet_NaN();
    m_nn = std::numeric_limits<int>::quiet_NaN();
    m_nt = std::numeric_limits<int>::quiet_NaN();
    m_ny = std::numeric_limits<int>::quiet_NaN();
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
    } else if (e <= MinimumEnergy(n, Y)) {
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

  /// Calculate hot (neutrino trapped) beta equilibrium T_eq and Y_eq given n, e, and Yl
  KOKKOS_INLINE_FUNCTION int BetaEquilibriumTrapped(Real n, Real e, Real *Yl, Real &T_eq,
      Real *Y_eq, Real T_guess, Real *Y_guess) const {
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
    int na = 0; // counter for the number of attempts

    Real x0[2], x1[2]; // T,Ye guess and T,Ye result

    while (ierr!=0 && na<n_at) {
      x0[0] = vec_guess[na][0] * T_guess;
      x0[1] = vec_guess[na][1] * Y_guess[0];

      ierr = trapped_equilibrium_2DNR(n, e, Yl[0], x0, x1);

      na += 1;
    }

    if (ierr==0) { // Success
      T_eq = x1[0];
      Y_eq[0] = x1[1];
    } else {      // Failure
      T_eq = T_guess;       // Set results to guesses
      Y_eq[0] = Y_guess[0];
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
    } else if (*in > m_nn - 2) {
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
    } else if (*iy > m_ny - 2) {
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
    } else if (*it > m_nt - 2) {
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
      if (f(0) <= 0) {
        return min_T;
      } else if (f(m_nt-1) >= 0) {
        return max_T;
      }
    }

    if (flo*fhi > 0) {
      int imin = 0;
      Real fmin = f(imin);
      Kokkos::printf("There's a problem with temperature bracketing!\n" // NOLINT
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

    Real lt = m_log_t[ilo] - flo*(lthi - ltlo)/(fhi - flo);
    return exp2_(lt);
  }

  /// Low level functions for neutrino equilibrium, not intended for outside use
  KOKKOS_INLINE_FUNCTION int trapped_equilibrium_2DNR(Real n, Real e, Real Yle,
                                                      Real x0[2], Real x1[2]) const {
    int ierr = 1;

    // initialize the solution
    x1[0] = x0[0];
    x1[1] = x0[1];
    bool KKT = false;

    //compute the initial residuals
    Real y[2] = {0.0};
    func_eq_weak(n,e,Yle,x1,y);

    // compute the error from the residuals
    Real err = error_func_eq_weak(Yle,e,y);

    // initialize the iteration variables
    int n_iter = 0;
    Real J[2][2] = {0.0};
    Real invJ[2][2] = {0.0};
    Real dx1[2] = {0.0};
    Real dxa[2] = {0.0};
    Real norm[2] = {0.0};
    Real x1_tmp[2] = {0.0};

    // loop until a low enough residual is found or until  a too
    // large number of steps has been performed
    while (err>nu_2DNR_eps_lim && n_iter<=nu_2DNR_n_max && !KKT) {
      // compute the Jacobian
      ierr = jacobi_eq_weak(n,e,Yle,x1,J);
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

      int n_cut = 0;
      Real fac_cut = 1.0;
      Real err_old = err;

      while (n_cut <= nu_bis_n_cut_max && err >= err_old) {
        // the variation of x1 is divided by an powers of 2 if the
        // error is not decreasing along the gradient direction

        x1_tmp[0] = x1[0] + (dx1[0]*fac_cut);
        x1_tmp[1] = x1[1] + (dx1[1]*fac_cut);

        // check if the next step calculation had problems
        if (isnan(x1_tmp[0])) {
          ierr = 1;
          return ierr;
        }

        // tabBoundsFlag = enforceTableBounds(rho, x1_tmp[0], x1_tmp[1]);
        x1_tmp[0] = fmin(fmax(x1_tmp[0],min_T),max_T);
        x1_tmp[1] = fmin(fmax(x1_tmp[1],min_Y[0]),max_Y[0]);

        // assign the new point
        x1[0] = x1_tmp[0];
        x1[1] = x1_tmp[1];

        // compute the residuals for the new point
        func_eq_weak(n,e,Yle,x1,y);

        // compute the error
        err = error_func_eq_weak(Yle,e,y);

        // update the bisection cut along the gradient
        n_cut += 1;
        fac_cut *= 0.5;
      }

      // update the iteration
      n_iter += 1;
    }

    if (n_iter <= nu_2DNR_n_max) {
      ierr = 0;
    } else {
      ierr = 1;
    }

    return ierr;
  }

  KOKKOS_INLINE_FUNCTION void func_eq_weak(Real n, Real e_eq, Real Yle, Real x[2],
                                           Real y[2]) const {
    Real T = x[0];

    Real Y[MAX_SPECIES] = {0.0};
    Y[0] = x[1];

    Real mu_l = ElectronLeptonChemicalPotential(n, T, Y);
    Real e = Energy(n, T, Y);
    Real eta = mu_l/T;
    Real eta2 = eta*eta;

    Real t3 = T*T*T;
    Real t4 = t3*T;
    y[0] = Y[0] + nu_n_prefactor*t3*eta*(pi2 + eta2)/n - Yle;
    y[1] = (e+nu_e_prefactor*t4*((nu_7pi4_60+0.5*eta2*(pi2+0.5*eta2))+nu_7pi4_30)) /
           e_eq - 1.0;

    return;
  }

  KOKKOS_INLINE_FUNCTION Real error_func_eq_weak(Real Yle, Real e_eq, Real y[2]) const {
    Real err = abs(y[0]/Yle) + abs(y[1]/1.0);
    return err;
  }

  KOKKOS_INLINE_FUNCTION int jacobi_eq_weak(Real n, Real e_eq, Real Yle, Real x[2],
                                            Real J[2][2]) const {
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
    // Real T4 = T3*T;

    J[0][0] = nu_n_prefactor/n*T2*(3.e0*eta*(pi2+eta2)+T*(pi2+3.e0*eta2)*detadt);
    J[0][1] = 1.e0+nu_n_prefactor/n*T3*(pi2+3.e0*eta2)*detadye;

    J[1][0] = (dedt+nu_e_prefactor*T3*(nu_7pi4_15+nu_14pi4_15+2.e0*eta2*(pi2+0.5*eta2) +
                                       eta*T*(pi2+eta2)*detadt))/e_eq;
    J[1][1] = (dedye+nu_e_prefactor*T3*eta*(pi2+eta2)*detadye)/e_eq;

    return ierr;
  }

  KOKKOS_INLINE_FUNCTION int eta_e_gradient(Real n, Real T, Real *Y, Real eta,
      Real &deta_dT, Real &deta_dYe, Real &de_dT, Real &de_dYe) const {
    int ierr=1;

    const Real Ye_delta = 0.005;
    const Real T_delta = 0.01;

    Real Y1[MAX_SPECIES] = {0.0};
    Real Y2[MAX_SPECIES] = {0.0};

    Y1[0] = fmax(Y[0] - Ye_delta, min_Y[0]);
    Real mu_l1 = ElectronLeptonChemicalPotential(n, T, Y1);
    Real e1 = Energy(n, T, Y1);

    Y2[0] = fmin(Y[0] + Ye_delta, max_Y[0]);
    Real mu_l2 = ElectronLeptonChemicalPotential(n, T, Y2);
    Real e2 = Energy(n, T, Y2);

    Real dmu_l_dYe = (mu_l2-mu_l1)/(Y2[0] - Y1[0]);
    de_dYe         = (e2-e1)/(Y2[0] - Y1[0]);

    Real T1 = fmax(T - T_delta, min_T);
    mu_l1 = ElectronLeptonChemicalPotential(n, T1, Y);
    e1 = Energy(n, T1, Y);

    Real T2 = fmin(T + T_delta, max_T);
    mu_l2 = ElectronLeptonChemicalPotential(n, T2, Y);
    e2 = Energy(n, T2, Y);

    Real dmu_l_dT   = (mu_l2 - mu_l1)/(T2 - T1);
    de_dT          = (e2 - e1)/(T2 - T1);

    deta_dT  = (dmu_l_dT - eta )/T; // [1/MeV] TODO: Check
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
  int m_nn, m_nt, m_ny;
  // Minimum enthalpy per baryon
  Real m_min_h;

  // bool to protect against access of uninitialised table and prevent repeated reading
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

  // Neutrino equilibrium physical constants
  const Real hc_mevfm = 1.23984172e3;           // hc    [MeV fm] (not reduced)
  const Real pi       = 3.14159265358979323846; // pi    [-]
  const Real pi2      = pi*pi;                  // pi**2 [-]
  const Real pi4      = pi2*pi2;                // pi**4 [-]

  // 4/3 *pi/(hc)**3 [1/MeV^3/fm^3]
  const Real nu_n_prefactor = 4.0/3.0*pi/(hc_mevfm*hc_mevfm*hc_mevfm);
  // 4*pi/(hc)**3    [1/MeV^3 fm^3]
  const Real nu_e_prefactor = 4.0*pi/(hc_mevfm*hc_mevfm*hc_mevfm);

  const Real nu_7pi4_60 = 7.0*pi4/60.0;  // 7*pi**4/60  [-]
  const Real nu_7pi4_30 = 7.0*pi4/30.0;  // 7*pi**4/30  [-]
  const Real nu_7pi4_15 = 7.0*pi4/15.0;  // 7*pi**4/15  [-]
  const Real nu_14pi4_15 = 14.0*pi4/15.0; // 14*pi**4/15 [-]
};

}; // namespace Primitive

#endif //EOS_PRIMITIVE_SOLVER_EOS_COMPOSE_HPP_
