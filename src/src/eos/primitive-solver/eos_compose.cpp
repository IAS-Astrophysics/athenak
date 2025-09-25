//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_compose.cpp
//  \brief Implementation of EOSCompose

#include <math.h>

#include <cassert>
#include <cstdio>
#include <limits>
#include <iostream>
#include <cstddef>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "eos_compose.hpp"
#include "utils/tr_table.hpp"
#include "logs.hpp"

namespace Primitive {

template<typename LogPolicy>
void EOSCompOSE<LogPolicy>::ReadTableFromFile(std::string fname) {
  if (m_initialized==false) {
    TableReader::Table table;
    auto read_result = table.ReadTable(fname);
    if (read_result.error != TableReader::ReadResult::SUCCESS) {
      std::cout << read_result.message << "\n"
                << "Table could not be read.\n";
      std::exit(EXIT_FAILURE);
    }
    // Make sure table has correct dimentions
    assert(table.GetNDimensions()==3);
    // TODO(PH) check that required fields are present?

    // Read baryon (neutron) mass
    auto& table_scalars = table.GetScalars();
    mb = table_scalars.at("mn");

    // Get table dimesnions
    auto& point_info = table.GetPointInfo();
    m_nn = point_info[0].second;
    m_ny = point_info[1].second;
    m_nt = point_info[2].second;

    // (Re)Allocate device storage
    Kokkos::realloc(m_log_nb, m_nn);
    Kokkos::realloc(m_yq,     m_ny);
    Kokkos::realloc(m_log_t,  m_nt);
    Kokkos::realloc(m_table, ECNVARS, m_nn, m_ny, m_nt);

    // Create host storage to read into
    HostArray1D<Real>::HostMirror host_log_nb = create_mirror_view(m_log_nb);
    HostArray1D<Real>::HostMirror host_yq =     create_mirror_view(m_yq);
    HostArray1D<Real>::HostMirror host_log_t =  create_mirror_view(m_log_t);
    HostArray4D<Real>::HostMirror host_table =  create_mirror_view(m_table);

    // Note that the some quantities are perturbed down slightly from what the top of
    // the table allows. This is because a lot of the interpolation operations need
    // n[i], n[i+1], yq[j], and yq[j+1], where i and j are the indices providing the
    // nearest table values at or below a specified i and yq.
    { // read nb
      Real * table_nb = table["nb"];

      for (size_t in=0; in<m_nn; ++in) {
        host_log_nb(in) = log2_(table_nb[in]);
      }

      m_id_log_nb = 1.0/(host_log_nb(1) - host_log_nb(0));
      min_n = table_nb[0];
      max_n = table_nb[m_nn-1];
    }

    { // read yq
      Real * table_yq = table["yq"];
      for (size_t iy=0; iy<m_ny; ++iy) {
        host_yq(iy) = table_yq[iy];
      }
      m_id_yq = 1.0/(host_yq(1) - host_yq(0));
      min_Y[0] = table_yq[0];
      max_Y[0] = table_yq[m_ny-1];
    }

    { // read T
      Real * table_t = table["t"];

      for (size_t it=0; it<m_nt; ++it) {
        host_log_t(it) = log2_(table_t[it]);
      }

      m_id_log_t = 1.0/(host_log_t(1) - host_log_t(0));
      min_T = table_t[0];
      max_T = table_t[m_nt-1];
    }

    { // Read Q1 -> log(P)
      Real * table_Q1 = table["Q1"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            Real p_current = table_Q1[iflat]*exp2_(host_log_nb(in));
            host_table(ECLOGP,in,iy,it) = log2_(p_current);
          }
        }
      }
    }

    { // Read Q2 -> S
      Real * table_Q2 = table["Q2"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            host_table(ECENT,in,iy,it) = table_Q2[iflat];
          }
        }
      }
    }

    { // Read Q3-> mu_b
      Real * table_Q3 = table["Q3"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            host_table(ECMUB,in,iy,it) = (table_Q3[iflat]+1)*mb;
          }
        }
      }
    }

    { // Read Q4-> mu_q
      Real * table_Q4 = table["Q4"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            host_table(ECMUQ,in,iy,it) = table_Q4[iflat]*mb;
          }
        }
      }
    }

    { // Read Q5-> mu_le
      Real * table_Q5 = table["Q5"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            host_table(ECMUL,in,iy,it) = table_Q5[iflat]*mb;
          }
        }
      }
    }

    { // Read Q7-> log(e)
      Real * table_Q7 = table["Q7"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            Real e_current = mb*(table_Q7[iflat] + 1)*exp2_(host_log_nb(in));
            host_table(ECLOGE,in,iy,it) = log2_(e_current);
          }
        }
      }
    }

    { // Read cs2-> cs
      Real * table_cs2 = table["cs2"];
      for (size_t in=0; in<m_nn; ++in) {
        for (size_t iy=0; iy<m_ny; ++iy) {
          for (size_t it=0; it<m_nt; ++it) {
            size_t iflat = it + m_nt*(iy + m_ny*in);
            host_table(ECCS,in,iy,it) = sqrt(table_cs2[iflat]);
          }
        }
      }
    }

    // Copy from host to device
    Kokkos::deep_copy(m_log_nb, host_log_nb);
    Kokkos::deep_copy(m_yq,     host_yq);
    Kokkos::deep_copy(m_log_t,  host_log_t);
    Kokkos::deep_copy(m_table,  host_table);

    m_initialized = true;

    m_min_h = std::numeric_limits<Real>::max();
    // Compute minimum enthalpy
    for (int in = 0; in < m_nn; ++in) {
      Real const nb = exp2_(host_log_nb(in));
      for (int iy = 0; iy < m_ny; ++iy) {
        for (int it = 0; it < m_nt; ++it) {
          // This would use GPU memory, and we are currently on the CPU, so Enthalpy is
          // hardcoded
          Real e = exp2_(host_table(ECLOGE,in,iy,it));
          Real p = exp2_(host_table(ECLOGP,in,iy,it));
          Real h = (e + p) / nb;
          m_min_h = fmin(m_min_h, h);
        }
      }
    }
  } // if (m_initialized==false)
}

template class EOSCompOSE<NormalLogs>;
template class EOSCompOSE<NQTLogs>;

} // namespace Primitive
