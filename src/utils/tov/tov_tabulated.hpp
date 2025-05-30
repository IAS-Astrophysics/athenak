#ifndef UTILS_TOV_TABULATED_HPP_
#define UTILS_TOV_TABULATED_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_tabulated.hpp
//  \brief Tabulated EOS for use with TOVStar
#include <stdexcept>
#include <sstream>

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

  Real dlrho;
  Real lrho_min;
  Real lrho_max;
  Real lP_min;
  Real lP_max;

  Real ye_atmosphere;

  std::string fname;
  size_t m_nn;

  //static const Real fm_to_Msun = 6.771781959609192e-19
  //static const Real MeV_to_Msun = 8.962968324680417e-61
  static constexpr Real ener_to_geo = 2.8863099290608455e-6;

 public:
  explicit TabulatedEOS(ParameterInput* pin) {
    fname = pin->GetString("problem", "table");

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

    // Get table dimensions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;

    // Allocate storage
    Kokkos::realloc(m_log_rho, m_nn);
    Kokkos::realloc(m_log_p, m_nn);
    Kokkos::realloc(m_log_e, m_nn);
    Kokkos::realloc(m_ye, m_nn);

    // Read rho
    test_field(table.HasField("nb"), "nb");
    Real * table_nb = table["nb"];
    for (size_t in = 0; in < m_nn; in++) {
      //m_log_rho.h_view(in) = log(table_nb[in]*mb*ener_to_geo);
      m_log_rho.h_view(in) = log(table_nb[in]*mb*
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

    // Read electron fraction
    test_field(table.HasField("Y[e]"), "Y[e]");
    Real * table_ye = table["Y[e]"];
    for (size_t in = 0; in < m_nn; in++) {
      m_ye.h_view(in) = table_ye[in];
    }

    std::cout << "Loaded table " << fname << std::endl
              << "  rho = [" << exp(lrho_min) << ", " << exp(lrho_max) << "]" << std::endl
              << "  P = [" << exp(lP_min) << ", " << exp(lP_max) << "]" << std::endl;

    ye_atmosphere = pin->GetOrAddReal("mhd", "s0_atmosphere",0.5);

    // Sync the views to the GPU
    m_log_rho.template modify<HostMemSpace>();
    m_log_p.template modify<HostMemSpace>();
    m_log_e.template modify<HostMemSpace>();
    m_ye.template modify<HostMemSpace>();

    m_log_rho.template sync<DevExeSpace>();
    m_log_p.template sync<DevExeSpace>();
    m_log_e.template sync<DevExeSpace>();
    m_ye.template sync<DevExeSpace>();
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
    if constexpr (loc == LocationTag::Host) {
      return exp(Interpolate(lrho, m_log_rho.h_view(lb), m_log_rho.h_view(ub),
                              m_log_p.h_view(lb), m_log_p.h_view(ub)));
    } else {
      return exp(Interpolate(lrho, m_log_rho.d_view(lb), m_log_rho.d_view(ub),
                              m_log_p.d_view(lb), m_log_p.d_view(ub)));
    }
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
    if constexpr (loc == LocationTag::Host) {
      return exp(Interpolate(lrho, m_log_rho.h_view(lb), m_log_rho.h_view(ub),
                              m_log_e.h_view(lb), m_log_e.h_view(ub)));
    } else {
      return exp(Interpolate(lrho, m_log_rho.d_view(lb), m_log_rho.d_view(ub),
                              m_log_e.d_view(lb), m_log_e.d_view(ub)));
    }
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetYeFromRho(Real rho) const {
    Real lrho = log(rho);
    if (lrho < lrho_min) {
      return ye_atmosphere;
    }
    int lb = static_cast<int>((lrho-lrho_min)/dlrho);
    int ub = lb + 1;
    if constexpr (loc == LocationTag::Host) {
      return Interpolate(lrho, m_log_rho.h_view(lb), m_log_rho.h_view(ub),
                          m_ye.h_view(lb), m_ye.h_view(ub));
    } else {
      return Interpolate(lrho, m_log_rho.d_view(lb), m_log_rho.d_view(ub),
                          m_ye.d_view(lb), m_ye.d_view(ub));
    }
  }

  template<LocationTag loc>
  KOKKOS_INLINE_FUNCTION
  Real GetRhoFromP(Real P) const {
    Real lP = log(P);
    int lb = 0;
    int ub = m_nn-1;
    // If the pressure is below the minimum of the table, we return zero density.
    if (lP < lP_min) {
      return 0.0;
    }
    // Do a binary search for the lower and upper indices of the pressure
    if constexpr (loc == LocationTag::Host) {
      while (ub - lb > 1) {
        int idx = (lb + ub)/2;
        if (m_log_p.h_view(idx) > lP) {
          ub = idx;
        } else {
          lb = idx;
        }
      }
      return exp(Interpolate(lP, m_log_p.h_view(lb), m_log_p.h_view(ub),
                              m_log_rho.h_view(lb), m_log_rho.h_view(ub)));
    } else {
      while (ub - lb > 1) {
        int idx = (lb + ub)/2;
        if (m_log_p.d_view(idx) > lP) {
          ub = idx;
        } else {
          lb = idx;
        }
      }
      return exp(Interpolate(lP, m_log_p.d_view(lb), m_log_p.d_view(ub),
                              m_log_rho.d_view(lb), m_log_rho.d_view(ub)));
    }
  }
};


} // namespace tov

#endif // UTILS_TOV_TABULATED_HPP_
