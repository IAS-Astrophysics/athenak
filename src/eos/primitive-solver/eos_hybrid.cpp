//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_hybrid.cpp
//  \brief Implementation of EOSHybrid

#include <math.h>

#include <cassert>
#include <cstdio>
#include <limits>
#include <iostream>
#include <cstddef>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "eos_hybrid.hpp"
#include "utils/tr_table.hpp"
#include "logs.hpp"

namespace Primitive {

template<typename LogPolicy>
void EOSHybrid<LogPolicy>::ReadTableFromFile(std::string fname) {
  if (m_initialized==false) {
    TableReader::Table table;
    auto read_result = table.ReadTable(fname);
    if (read_result.error != TableReader::ReadResult::SUCCESS) {
      std::cout << "Table could not be read.\n" << std::flush;
      abort();
    }
    // Make sure table has correct dimensions
    assert(table.GetNDimensions()==1);
    // TODO(PH) check that required fields are present?

    // Read baryon (neutron) mass
    auto& table_scalars = table.GetScalars();
    mb = table_scalars.at("mn");

    // Get table dimensions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;

    // (Re)Allocate device storage
    Kokkos::realloc(m_log_nb, m_nn);
    Kokkos::realloc(m_table, ECNVARS, m_nn);

    // Create host storage to read into
    HostArray1D<Real>::HostMirror host_log_nb = create_mirror_view(m_log_nb);
    HostArray2D<Real>::HostMirror host_table =  create_mirror_view(m_table);

    { // read nb
      Real * table_nb = table["nb"];

      for (size_t in=0; in<m_nn; ++in) {
        host_log_nb(in) = log2_(table_nb[in]);
      }

      m_id_log_nb = 1.0/(host_log_nb(1) - host_log_nb(0));
      min_n = table_nb[0]*(1 + 1e-15);
      max_n = table_nb[m_nn-1]*(1 - 1e-15);
    }

    { // Read Q1 -> log(P)
      Real * table_Q1 = table["Q1"];
      for (size_t in=0; in<m_nn; ++in) {
        Real p_current = table_Q1[in]*exp2_(host_log_nb(in));
        host_table(ECLOGP,in) = log2_(p_current);
      }
    }

    { // Read Q2 -> S
      Real * table_Q2 = table["Q2"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECENT,in) = table_Q2[in];
      }
    }

    { // Read Q3-> mu_b
      Real * table_Q3 = table["Q3"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECMUB,in) = (table_Q3[in]+1)*mb;
      }
    }

    { // Read Q4-> mu_q
      Real * table_Q4 = table["Q4"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECMUQ,in) = table_Q4[in]*mb;
      }
    }

    { // Read Q5-> mu_le
      Real * table_Q5 = table["Q5"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECMUL,in) = table_Q5[in]*mb;
      }
    }

    { // Read Q7-> log(e)
      Real * table_Q7 = table["Q7"];
      for (size_t in=0; in<m_nn; ++in) {
        Real e_current = mb*(table_Q7[in] + 1)*exp2_(host_log_nb(in));
        host_table(ECLOGE,in) = log2_(e_current);
      }
    }

    { // Read cs2-> cs
      Real * table_cs2 = table["cs2"];
      for (size_t in=0; in<m_nn; ++in) {
        host_table(ECCS,in) = sqrt(table_cs2[in]);
      }
    }

    // Copy from host to device
    Kokkos::deep_copy(m_log_nb, host_log_nb);
    Kokkos::deep_copy(m_table,  host_table);

    m_initialized = true;

    m_min_h = std::numeric_limits<Real>::max();
    // New form of bound based on properties of NQT functions and their
    // departure from 'true' log behaviour
    for (size_t in = 0; in < m_nn-1; ++in) {
      Real const nb = exp2_(host_log_nb(in));
      Real log2_e_in   = host_table(ECLOGE,in);
      Real log2_e_inp1 = host_table(ECLOGE,in+1);
      Real pow_e = Kokkos::fabs(log2_e_inp1-log2_e_in);

      Real log2_p_in   = host_table(ECLOGP,in);
      Real log2_p_inp1 = host_table(ECLOGP,in+1);
      Real pow_p = Kokkos::fabs(log2_p_inp1-log2_p_in);

      Real k0 =  3.696e-3; // Exact number rounded up
      Real k1 = -9.709e-3; // Exact number rounded down

      Real fac_e = (1-k0)*Kokkos::exp2(pow_e*k1); // N.B. not exp2_
      Real fac_p = (1-k0)*Kokkos::exp2(pow_p*k1);

      Real e_over_n_min = fac_e*Kokkos::fmin(exp2_(log2_e_in)/nb,
                                            exp2_(log2_e_inp1)/exp2_(host_log_nb(in+1)));

      Real p_over_n_min = fac_p*Kokkos::fmin(exp2_(log2_p_in)/nb,
                                            exp2_(log2_p_inp1)/exp2_(host_log_nb(in+1)));

      m_min_h = Kokkos::fmin(m_min_h,e_over_n_min+p_over_n_min);
    }

    if (m_min_h <= 0.0) {
      Kokkos::abort("There was a problem computing the minimum enthalpy in the table!");
    }
  } // if (m_initialized==false)
}

template class EOSHybrid<NormalLogs>;
template class EOSHybrid<NQTLogs>;

} // namespace Primitive
