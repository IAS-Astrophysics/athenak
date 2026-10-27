//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_helmholtz.cpp
//  \brief Host-side setup of EOSHelmholtz: species list and .athtab electron table.

#include <math.h>

#include <algorithm>
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

void EOSHelmholtz::SetSpecies(int n, const Real *A, const Real *Z, const Real *dm) {
  if (n < 1 || n > MAX_SPECIES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSHelmholtz::SetSpecies: " << n << " species, MAX_SPECIES is "
              << MAX_SPECIES << std::endl;
    std::exit(EXIT_FAILURE);
  }
  n_species = n;
  m_min_h = std::numeric_limits<Real>::max();
  for (int s = 0; s < n; ++s) {
    m_A[s]     = A[s];
    m_Z[s]     = Z[s];
    m_dm[s]    = dm[s];
    m_inv_A[s] = 1.0/A[s];
    m_zoa[s]   = Z[s]/A[s];
    m_dmoa[s]  = dm[s]/A[s];
    m_mion[s]  = A[s]*m_u + dm[s] - Z[s]*me;
    // enthalpy per baryon is bounded below by the rest mass per baryon
    m_min_h = std::min(m_min_h, m_u + m_dmoa[s]);
  }
  if (m_initialized) { update_density_bounds(); }
}

void EOSHelmholtz::update_density_bounds() {
  // Conservative: n in [min_n, max_n] keeps n*Ye inside the table for every Ye the
  // species list can produce.
  Real ye_min = 1.0, ye_max = 0.0;
  for (int s = 0; s < n_species; ++s) {
    ye_min = std::min(ye_min, m_zoa[s]);
    ye_max = std::max(ye_max, m_zoa[s]);
  }
  min_n = m_ne_min/ye_min;
  max_n = m_ne_max/ye_max;
}

void EOSHelmholtz::ReadTableFromFile(std::string fname) {
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
  Kokkos::realloc(m_t,      m_nt);
  Kokkos::realloc(m_table,  ECNVARS, m_nn, m_nt);

  auto host_log_ne = Kokkos::create_mirror_view(m_log_ne);
  auto host_log_t  = Kokkos::create_mirror_view(m_log_t);
  auto host_table  = Kokkos::create_mirror_view(m_table);

  { // n_e grid
    Real* ne = table["ne"];
    m_ne_min = ne[0];
    m_ne_max = ne[m_nn-1];
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
  {
    // exp(log t) rather than t: bit-identical to the exp(m_log_t) it replaces.
    auto host_t = Kokkos::create_mirror_view(m_t);
    for (int it = 0; it < m_nt; ++it) { host_t(it) = exp(host_log_t(it)); }
    Kokkos::deep_copy(m_t, host_t);
  }
  Kokkos::deep_copy(m_table,  host_table);

  m_initialized = true;
  update_density_bounds();
}

}  // namespace Primitive
