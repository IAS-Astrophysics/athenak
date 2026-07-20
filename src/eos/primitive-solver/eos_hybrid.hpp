#ifndef EOS_PRIMITIVE_SOLVER_EOS_HYBRID_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_HYBRID_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_hybrid.hpp
//  \brief Defines EOSTable, which stores information from a 1D tabulated
//         equation of state in CompOSE format, with a thermal Gamma-Law component.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb

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
class EOSHybrid : public EOSPolicyInterface, public LogPolicy {
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
  EOSHybrid() :
      m_log_nb("log nb",1),
      m_table("EoS table",1,1) {
    n_species = 0;
    eos_units = MakeNuclear();
    m_initialized = false;

    // These will be set properly when the table is read
    m_id_log_nb = std::numeric_limits<Real>::quiet_NaN();
    m_nn = std::numeric_limits<int>::quiet_NaN();
    m_min_h = std::numeric_limits<Real>::max();
    mb =    std::numeric_limits<Real>::quiet_NaN();

    min_n = std::numeric_limits<Real>::quiet_NaN();
    max_n = std::numeric_limits<Real>::quiet_NaN();
    min_T = 0.0;
    max_T = std::numeric_limits<Real>::max();
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = 0.0;
      max_Y[i] = 1.0;
    }
  }

/*
  /// Destructor
  ~EOSHybrid();
*/

  /// Temperature from energy density.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    assert (m_initialized);
    if (n < min_n) {
      // If density is OOB then return minimum temperature
      return min_T;
    } else if (e <= MinimumEnergy(n, Y)) {
      // If energy is OOB then return minimum temperature
      return min_T;
    }

    Real e_cold = ColdEnergy(n);
    Real T = gamma_th_m1*(e-e_cold)/n;
    return Kokkos::fmax(T,min_T);
  }

  /// Calculate the temperature using.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    assert (m_initialized);
    if (n < min_n) {
      // If density is OOB then return minimum temperature
      return min_T;
    } else if (p <= MinimumPressure(n, Y)) {
      // If pressure is OOB then return minimum temperature
      return min_T;
    }

    Real p_cold = ColdPressure(n);
    Real T = (p-p_cold)/n;
    return Kokkos::fmax(T,min_T);
  }

  /// Calculate the energy density using.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    Real e_cold = ColdEnergy(n);
    Real e_th   = n*T/gamma_th_m1;
    return e_cold + e_th;
  }

  /// Calculate the pressure using.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    Real p_cold = ColdPressure(n);
    Real p_th   = n*T;
    return p_cold + p_th;
  }

  /// Calculate the enthalpy per baryon using.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    Real const P = Pressure(n, T, Y);
    Real const e = Energy(n, T, Y);
    return (P + e)/n;
  }

  /// Calculate the sound speed.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    Real H_cold = ColdEnthalpy(n);
    Real H_th   = (gamma_th*T)/(gamma_th_m1);

    Real Hcs2_cold = pow(ColdSoundSpeed(n),2.0)*H_cold;
    Real Hcs2_th   = gamma_th*T;

    return sqrt((Hcs2_cold + Hcs2_th)/(H_cold + H_th));
  }

  /// Calculate the specific internal energy per unit mass.
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    return Energy(n, T, Y)/(mb*n) - 1;
  }

  /// Calculate energy for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdEnergy(Real n) const {
    assert (m_initialized);
    return exp2_(eval_at_n(ECLOGE, n));
  }

  /// Calculate pressure for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdPressure(Real n) const {
    assert (m_initialized);
    return exp2_(eval_at_n(ECLOGP, n));
  }

  /// Calculate enthalpy for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdEnthalpy(Real n) const {
    assert (m_initialized);
    Real const p_cold = ColdPressure(n);
    Real const e_cold = ColdEnergy(n);
    return (p_cold + e_cold)/n;
  }

  /// Calculate sound speed for the cold part.
  KOKKOS_INLINE_FUNCTION Real ColdSoundSpeed(Real n) const {
    assert (m_initialized);
    return eval_at_n(ECCS, n);
  }

  /// Get the minimum enthalpy per baryon.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    assert (m_initialized);
    return m_min_h;
  }

  /// Get the minimum pressure at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, min_T, Y);
  }

  /// Get the maximum pressure at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max!
    return max_T;
  }

  /// Get the minimum energy at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return Energy(n, min_T, Y);
  }

  /// Get the maximum energy at a given density and composition.
  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    // Note that max_T is already set to numeric_limits<Real>::max!
    return max_T;
  }

 public:
  /// Reads the table file.
  void ReadTableFromFile(std::string fname);

  /// Get the raw number density.
  KOKKOS_INLINE_FUNCTION DvceArray1D<Real> const GetRawLogNumberDensity() const {
    return m_log_nb;
  }

  /// Get the raw table data.
  KOKKOS_INLINE_FUNCTION DvceArray2D<Real> const GetRawTable() const {
    return m_table;
  }

  // Indexing used to access the data.
  KOKKOS_INLINE_FUNCTION ptrdiff_t index(int iv, int in) const {
    return in + m_nn*iv;
  }

  /// Check if the EOS has been initialized properly.
  KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
    return m_initialized;
  }

  /// Set the number of species. Throw an exception if
  /// the number of species is invalid. TODO
  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
    // Number of species must be within limits
    assert (n<=MAX_SPECIES && n>=0);

    n_species = n;
    return;
  }

  /// Set the adiabatic constant for the thermal part.
  /// Gamma is limited to the range 1.00001 <= g <= 2.0.
  KOKKOS_INLINE_FUNCTION void SetThermalGamma(Real g) {
    gamma_th = (g <= 1.00001) ? 1.00001 : ((g >= 2.0) ? 2.0 : g);
    gamma_th_m1 = gamma_th - 1.0;
  }

  /// Get the adiabatic constant for the thermal part.
  KOKKOS_INLINE_FUNCTION Real GetThermalGamma() const {
    return gamma_th;
  }

  /// Set the EOS unit system.
  KOKKOS_INLINE_FUNCTION void SetEOSUnitSystem(UnitSystem units) {
    eos_units = units;
  }

 private:
  /// Low level evaluation function, not intended for outside use.
  KOKKOS_INLINE_FUNCTION Real eval_at_n(int vi, Real n) const {
    Real log_n = log2_(n);
    return eval_at_ln(vi, log_n);
  }

  /// Low level evaluation function, not intended for outside use.
  KOKKOS_INLINE_FUNCTION Real eval_at_ln(int iv, Real log_n)
      const {
    int in;
    Real wn0, wn1;

    weight_idx_ln(&wn0, &wn1, &in, log_n);

    return
      wn0 * m_table(iv, in+0) +
      wn1 * m_table(iv, in+1);
  }

  /// Evaluate interpolation weight for density.
  KOKKOS_INLINE_FUNCTION void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n)
      const {
    *in = (log_n - m_log_nb(0))*m_id_log_nb;
    *w1 = (log_n - m_log_nb(*in))*m_id_log_nb;
    *w0 = 1.0 - (*w1);
    return;
  }

 private:
  // Inverse of table spacing
  Real m_id_log_nb;
  // Table size
  int m_nn;
  // Minimum enthalpy per baryon
  Real m_min_h;

  // bool to protect against access of uninitialised table and prevent repeated reading
  // of table
  bool m_initialized;

  // Table storage on DEVICE.
  DvceArray1D<Real> m_log_nb;
  DvceArray2D<Real> m_table;

  // Thermal Gamma
  Real gamma_th;
  Real gamma_th_m1;
};

}; // namespace Primitive

#endif //EOS_PRIMITIVE_SOLVER_EOS_HYBRID_HPP_
