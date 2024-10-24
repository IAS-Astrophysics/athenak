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
class EOSCompOSE : public EOSPolicyInterface, public LogPolicy {
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
  }

/*
  /// Destructor
  ~EOSCompOSE();
*/

  /// Temperature from energy density
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    assert (m_initialized);
    Real log_e = log2_(e);
    return temperature_from_var(ECLOGE, log_e, n, Y[0]);
  }

  /// Calculate the temperature using.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    assert (m_initialized);
    if (n < min_n || p < MinimumPressure(n, Y)) {
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

/* Not needed until neutrinos are added
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
  KOKKOS_INLINE_FUNCTION Real ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) const {
    assert (m_initialized);
    return eval_at_nty(ECMUL, n, T, Y[0]);
  }
*/

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
    *w1 = (log_n - m_log_nb(*in))*m_id_log_nb;
    *w0 = 1.0 - (*w1);
    return;
  }
  /// Evaluate interpolation weight for composition
  KOKKOS_INLINE_FUNCTION void weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const {
    *iy = (yq - m_yq(0))*m_id_yq;
    *w1 = (yq - m_yq(*iy))*m_id_yq;
    *w0 = 1.0 - (*w1);
    return;
  }

  /// Evaluate interpolation weight for temperature
  KOKKOS_INLINE_FUNCTION void weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t)
      const {
    *it = (log_t - m_log_t(0))*m_id_log_t;
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
    
    if (flo*fhi>0.0 && iv==ECLOGP || iv==ECLOGE) {
      if (var <= eval_at_nty(iv,n,min_T,Yq)) {
        return min_T;
      } else if (var >= eval_at_nty(iv,n,max_T,Yq)) {
        return max_T;
      }
    }
    
    assert(flo*fhi <= 0);
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
};

}; // namespace Primitive

#endif //EOS_PRIMITIVE_SOLVER_EOS_COMPOSE_HPP_
