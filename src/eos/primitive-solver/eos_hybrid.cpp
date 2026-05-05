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
    int in_min;
    // Compute minimum enthalpy
    for (int in = 0; in < m_nn; ++in) {
      Real const nb = exp2_(host_log_nb(in));
      // This would use GPU memory, and we are currently on the CPU, so Enthalpy is
      // hardcoded
      Real e = exp2_(host_table(ECLOGE,in));
      Real p = exp2_(host_table(ECLOGP,in));
      Real h = (e + p) / nb;
      if (h < m_min_h) {
        m_min_h = h;
        in_min = in;
      }
    }
    // Because the enthalpy is (e + P)/n (i.e., nonlinear), we cannot guarantee that the
    // minimum at table points is actually the minimum allowed by the EOS once we
    // interpolate between points. Finding the true minimum would require some sort of
    // optimization algorithm, which is difficult because all the interpolation operations
    // are on the GPU. However, we can compute a conservative lower bound on min_h by
    // computing the derivative at the estimated minimum. It should have O(dlog n)
    // behavior, so it should also be a conservative bound for the NQTs, which will have
    // O((dlog n)^2) behavior.
    //
    // We estimate the minimum as h(n_k) - dlog n |dh/dn|_{n_k}, where we take the
    // maximum of |dh/dn| using the left and right estimates.
    Real loge = host_table(ECLOGE,in_min);
    Real logp = host_table(ECLOGP,in_min);
    Real nb = exp2_(host_log_nb(in_min));
    Real e = exp2_(loge);
    Real p = exp2_(logp);
    Real loge_left = loge;
    Real logp_left = logp;
    if (in_min > 0) {
      loge_left = host_table(ECLOGE,in_min-1);
      logp_left = host_table(ECLOGP,in_min-1);
    }
    Real dhdn = Kokkos::fabs(((loge - loge_left)*m_id_log_nb - 1.0)*e +
                             ((logp - logp_left)*m_id_log_nb - 1.0)*p)/nb;
    Real loge_right = loge;
    Real logp_right = logp;
    if (in_min < m_nn-1) {
      loge_right = host_table(ECLOGE,in_min+1);
      logp_right = host_table(ECLOGP,in_min+1);
    }
    dhdn = Kokkos::fmax(Kokkos::fabs(((loge_right - loge)*m_id_log_nb - 1.0)*e +
                        ((logp_right - logp)*m_id_log_nb - 1.0)*p)/nb, dhdn);
    m_min_h -= dhdn/m_id_log_nb;
    if (m_min_h <= 0.0) {
      Kokkos::abort("There was a problem computing the minimum enthalpy in the table!");
    }
  } // if (m_initialized==false)
}

template class EOSHybrid<NormalLogs>;
template class EOSHybrid<NQTLogs>;

} // namespace Primitive
