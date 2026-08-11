//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_helmholtz.cpp
//  \brief Host-side .athtab reader for the device EOSHelmholtz table.

#include <math.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "eos_helmholtz.hpp"
#include "utils/tr_table.hpp"

namespace Primitive {

void EOSHelmholtz::ReadTableFromFile(std::string fname, Real min_Ye, Real max_Ye) {
  if (m_initialized) {
    return;
  }
  TableReader::Table table;
  auto read_result = table.ReadTable(fname);
  if (read_result.error != TableReader::ReadResult::SUCCESS) {
    std::cout << read_result.message << "\n"
              << "Helmholtz table could not be read.\n";
    std::exit(EXIT_FAILURE);
  }
  assert(table.GetNDimensions() == 2);

  auto& point_info = table.GetPointInfo();
  m_nn = point_info[0].second;
  m_nt = point_info[1].second;

  // The atomic mass unit used as the reference mass by the table generator.
  auto& scalars = table.GetScalars();
  const Real mb_file = scalars.at("mb");
  mb = mb_file;

  Kokkos::realloc(m_log_ne, m_nn);
  Kokkos::realloc(m_log_t,  m_nt);
  Kokkos::realloc(m_table,  ECNVARS, m_nn, m_nt);

  auto host_log_ne = Kokkos::create_mirror_view(m_log_ne);
  auto host_log_t  = Kokkos::create_mirror_view(m_log_t);
  auto host_table  = Kokkos::create_mirror_view(m_table);

  { // n_e grid
    Real* ne = table["ne"];
    min_n = ne[0]/min_Ye;
    max_n = ne[m_nn-1]/max_Ye;
    for (int in = 0; in < m_nn; ++in) {
      host_log_ne(in) = log(ne[in]);
    }
    m_id_log_ne = 1.0/(host_log_ne(1) - host_log_ne(0));
  }

  { // T grid
    Real* t = table["t"];
    min_T = t[0];
    max_T = t[m_nt-1];
    for (int it = 0; it < m_nt; ++it) {
      host_log_t(it) = log(t[it]);
    }
    m_id_log_t = 1.0/(host_log_t(1) - host_log_t(0));
  }

  auto fill_channel = [&](int iv, const char* field, int mode, Real fac) {
    // mode 0: copy; mode 1: log(); mode 2: log(v*fac); mode 3: v*fac
    Real* d = table[field];
    for (int in = 0; in < m_nn; ++in) {
      for (int it = 0; it < m_nt; ++it) {
        int iflat = it + m_nt*in;
        Real v = d[iflat];
        Real out;
        switch (mode) {
          case 1: out = log(v); break;
          case 2:
            if (v <= 0.0) {
              std::cout << "Helmholtz table: non-positive value in '" << field
                        << "', cannot take log.\n";
              std::exit(EXIT_FAILURE);
            }
            out = log(v*fac);
            break;
          case 3: out = v*fac; break;
          default: out = v; break;
        }
        host_table(iv, in, it) = out;
      }
    }
  };

  fill_channel(ECLOGP,   "p",      1, 1.0);
  fill_channel(ECENT,    "s",      0, 1.0);
  fill_channel(ECLOGEPS, "eps",    2, mb_file);
  fill_channel(ECETA,    "eta",    0, 1.0);
  fill_channel(ECDEPSDT, "depsdt", 3, mb_file);
  fill_channel(ECDPDN,   "dpdn",   0, 1.0);
  fill_channel(ECDPDT,   "dpdt",   0, 1.0);

  Kokkos::deep_copy(m_log_ne, host_log_ne);
  Kokkos::deep_copy(m_log_t,  host_log_t);
  Kokkos::deep_copy(m_table,  host_table);

  max_Y[SCYE] = max_Ye;
  min_Y[SCYE] = min_Ye;
  m_initialized = true;
}

}  // namespace Primitive
