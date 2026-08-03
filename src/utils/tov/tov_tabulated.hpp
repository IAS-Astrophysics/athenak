#ifndef UTILS_TOV_TOV_TABULATED_HPP_
#define UTILS_TOV_TOV_TABULATED_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_tabulated.hpp
//  \brief Tabulated EOS for use with TOVStar
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "utils/tr_table.hpp"
#include "tov_utils.hpp"
#include "eos/primitive-solver/unit_system.hpp"

namespace tov {

class TabulatedEOS {
 private:
  DualArray1D<Real> m_log_rho;
  DualArray1D<Real> m_log_p;
  DualArray1D<Real> m_log_e;
  DualArray1D<Real> m_ye;
  DualArray1D<Real> m_xn, m_xp, m_xa, m_xh, m_ah;

  Real dlrho;
  Real lrho_min;
  Real lrho_max;
  Real lP_min;
  Real lP_max;
  Real le_min;
  Real le_max;

  bool has_ye = false;
  bool has_comp = false;
  Real ye_atmosphere;

  size_t m_nn;

  //static const Real fm_to_Msun = 6.771781959609192e-19
  //static const Real MeV_to_Msun = 8.962968324680417e-61
  static constexpr Real ener_to_geo = 2.8863099290608455e-6;

 public:
  explicit TabulatedEOS(ParameterInput* pin) {
    const std::string fname = pin->GetString("problem", "table");

    TableReader::Table table;

    auto read_result = table.ReadTable(fname);
    if (read_result.error != TableReader::ReadResult::SUCCESS) {
      std::cout << read_result.message << std::endl
                << "TOV EOS table could not be read.\n";
      std::exit(EXIT_FAILURE);
    }

    // Unit conversions
    Primitive::UnitSystem unit_geo = Primitive::MakeGeometricSolar();
    Primitive::UnitSystem unit_nuc = Primitive::MakeNuclear();

    auto test_field = [](bool test, const std::string name) -> void {
      if (test) {
        return;
      } else {
        std::stringstream ss;
        ss << "Table is missing key '" << name << "'\n";
        throw std::runtime_error(ss.str());
      }
    };

    // TODO(JMF) Check that table has right fields and dimensions
    auto& table_scalars = table.GetScalars();
    test_field(table_scalars.count("mn") > 0, "mn");
    Real mb = table_scalars.at("mn");
    // rho = n_b*m_b must use the same m_b as the EOS the star is evolved with,
    // or the star is not in hydrostatic equilibrium with its own EOS. The EOS is
    // built before the pgen, so <mhd> bmass is already set where it applies.
    Real mb_rho = pin->GetOrAddReal("mhd", "bmass", mb);

    // Get table dimensions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;
    has_ye = table.HasField("Y[e]");
    has_comp = table.HasField("Y[n]") && table.HasField("Y[p]") &&
               table.HasField("Y[He4]") && table.HasField("A[N]") &&
               table.HasField("Y[N]");

    // Allocate storage
    Kokkos::realloc(m_log_rho, m_nn);
    Kokkos::realloc(m_log_p, m_nn);
    Kokkos::realloc(m_log_e, m_nn);
    if (has_ye) {Kokkos::realloc(m_ye, m_nn);}
    if (has_comp) {
      Kokkos::realloc(m_xn, m_nn);
      Kokkos::realloc(m_xp, m_nn);
      Kokkos::realloc(m_xa, m_nn);
      Kokkos::realloc(m_xh, m_nn);
      Kokkos::realloc(m_ah, m_nn);
    }

    // Read rho
    test_field(table.HasField("nb"), "nb");
    Real * table_nb = table["nb"];
    for (size_t in = 0; in < m_nn; in++) {
      //m_log_rho.h_view(in) = log(table_nb[in]*mb*ener_to_geo);
      m_log_rho.h_view(in) = log(table_nb[in]*mb_rho*
                                 unit_nuc.MassDensityConversion(unit_geo));
    }
    dlrho = m_log_rho.h_view(1)-m_log_rho.h_view(0);
    lrho_min = m_log_rho.h_view(0);
    lrho_max = m_log_rho.h_view(m_nn-1);

    // Read pressure
    test_field(table.HasField("Q1"), "Q1");
    Real * table_Q1 = table["Q1"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_p.h_view(in) = log(table_Q1[in]*table_nb[in]*
                                unit_nuc.EnergyDensityConversion(unit_geo));
    }
    lP_min = m_log_p.h_view(0);
    lP_max = m_log_p.h_view(m_nn-1);

    // Read energy
    test_field(table.HasField("Q7"), "Q7");
    Real * table_Q7 = table["Q7"];
    for (size_t in = 0; in < m_nn; in++) {
      m_log_e.h_view(in) = log(mb*(table_Q7[in] + 1.)*table_nb[in]*
                                    unit_nuc.EnergyDensityConversion(unit_geo));
    }
    le_min = m_log_e.h_view(0);
    le_max = m_log_e.h_view(m_nn-1);

    // Read electron fraction (optional)
    if (has_ye) {
      Real * table_ye = table["Y[e]"];
      for (size_t in = 0; in < m_nn; in++) {
        m_ye.h_view(in) = table_ye[in];
      }
    }

    // Read composition (optional); Y -> X as in EOSCompOSE.
    if (has_comp) {
      Real * t_yn = table["Y[n]"];
      Real * t_yp = table["Y[p]"];
      Real * t_ya = table["Y[He4]"];
      Real * t_an = table["A[N]"];
      Real * t_yN = table["Y[N]"];
      const char* light[3] = {"Y[H2]", "Y[H3]", "Y[He3]"};
      const Real amass[3] = {2.0, 3.0, 3.0};
      for (size_t in = 0; in < m_nn; in++) {
        Real xa = t_ya[in]*4.0;
        for (int l = 0; l < 3; ++l) {
          if (table.HasField(light[l])) { xa += table[light[l]][in]*amass[l]; }
        }
        m_xn.h_view(in) = fmax(0.0, fmin(t_yn[in], 1.0));
        m_xp.h_view(in) = fmax(0.0, fmin(t_yp[in], 1.0));
        m_xa.h_view(in) = fmax(0.0, fmin(xa, 1.0));
        m_ah.h_view(in) = fmax(1.0, t_an[in]);
        m_xh.h_view(in) = fmax(0.0, fmin(m_ah.h_view(in)*t_yN[in], 1.0));
      }
    }

    std::cout << "Loaded table " << fname << std::endl
              << "  rho = [" << exp(lrho_min) << ", " << exp(lrho_max) << "]" << std::endl
              << "  P = [" << exp(lP_min) << ", " << exp(lP_max) << "]" << std::endl;

    ye_atmosphere = pin->GetOrAddReal("mhd", "s0_atmosphere",0.5);

    // Sync the views to the GPU
    m_log_rho.template modify<HostMemSpace>();
    m_log_p.template modify<HostMemSpace>();
    m_log_e.template modify<HostMemSpace>();
    if (has_ye) {m_ye.template modify<HostMemSpace>();}
    if (has_comp) {
      m_xn.template modify<HostMemSpace>();
      m_xp.template modify<HostMemSpace>();
      m_xa.template modify<HostMemSpace>();
      m_xh.template modify<HostMemSpace>();
      m_ah.template modify<HostMemSpace>();
    }

    m_log_rho.template sync<DevExeSpace>();
    m_log_p.template sync<DevExeSpace>();
    m_log_e.template sync<DevExeSpace>();
    if (has_ye) {m_ye.template sync<DevExeSpace>();}
    if (has_comp) {
      m_xn.template sync<DevExeSpace>();
      m_xp.template sync<DevExeSpace>();
      m_xa.template sync<DevExeSpace>();
      m_xh.template sync<DevExeSpace>();
      m_ah.template sync<DevExeSpace>();
    }
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetPFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return 0.0;
    }
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& lp_view = GetView<loc>(m_log_p);
    return exp(Interpolate(lrho, lrho_view(lb), lrho_view(ub), lp_view(lb), lp_view(ub)));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetEFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return 0.0;
    }
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& le_view = GetView<loc>(m_log_e);
    return exp(Interpolate(lrho, lrho_view(lb), lrho_view(ub), le_view(lb), le_view(ub)));
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromE(Real e) const {
    Real le = log(e);

    // Guard against negative total energy
    // densities and garbage values.
    if (le < le_min || e < 0.0) {
      return 0.0;
    }

    return GetRhoFromVar<loc>(le, m_log_e);
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetYeFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min || !has_ye) {
      return ye_atmosphere;
    }
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    auto& lrho_view = GetView<loc>(m_log_rho);
    auto& ye_view = GetView<loc>(m_ye);
    return Interpolate(lrho, lrho_view(lb), lrho_view(ub), ye_view(lb), ye_view(ub));
  }

  //! Mass fractions and mean heavy mass number at a given density, using the same
  //! Y -> X conventions as EOSCompOSE (Xa folds in the light nuclei, Xh = A[N]*Y[N]).
  //! Returns false if the table carries no composition, leaving the arguments alone.
  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  bool GetCompositionFromRho(Real rho, Real &xn, Real &xp, Real &xa,
                             Real &xh, Real &ah) const {
    if (!has_comp) { return false; }
    Real lrho = fmax(log(rho), lrho_min);
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    auto& lr = GetView<loc>(m_log_rho);
    xn = Interpolate(lrho, lr(lb), lr(ub), GetView<loc>(m_xn)(lb), GetView<loc>(m_xn)(ub));
    xp = Interpolate(lrho, lr(lb), lr(ub), GetView<loc>(m_xp)(lb), GetView<loc>(m_xp)(ub));
    xa = Interpolate(lrho, lr(lb), lr(ub), GetView<loc>(m_xa)(lb), GetView<loc>(m_xa)(ub));
    xh = Interpolate(lrho, lr(lb), lr(ub), GetView<loc>(m_xh)(lb), GetView<loc>(m_xh)(ub));
    ah = Interpolate(lrho, lr(lb), lr(ub), GetView<loc>(m_ah)(lb), GetView<loc>(m_ah)(ub));
    return true;
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    Real lP = log(P);
    // If the pressure is below the minimum of the table, we return zero density.
    if (lP < lP_min) {
      return 0.0;
    }
    return GetRhoFromVar<loc>(lP, m_log_p);
  }

 private:
  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  auto& GetView(const DualArray1D<Real>& arr) const {
    if constexpr (loc == LocationTag::Host) {
      return arr.h_view;
    } else {
      return arr.d_view;
    }
  }

  // Use bisection on a specified variable to find the corresponding density.
  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromVar(Real var, const DualArray1D<Real>& arr_var) const {
    int lb = 0;
    int ub = m_nn-1;
    auto& v_view = GetView<loc>(arr_var);
    auto& lrho_view = GetView<loc>(m_log_rho);
    // Do a binary search for the lower and upper indices of arr_var
    while (ub - lb > 1) {
      int idx = (lb + ub)/2;
      if (v_view(idx) > var) {
        ub = idx;
      } else {
        lb = idx;
      }
    }
    return exp(Interpolate(var, v_view(lb), v_view(ub), lrho_view(lb), lrho_view(ub)));
  }
};


} // namespace tov

#endif // UTILS_TOV_TOV_TABULATED_HPP_
