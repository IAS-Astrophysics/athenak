//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_transition.cpp
//  \brief Host-side setup for the device EOSTransition (table read + bounds).

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "eos_transition.hpp"
#include "logs.hpp"

namespace Primitive {

template<typename LogPolicy>
void EOSTransition<LogPolicy>::SetTransition(Real n_start, Real n_end,
                                             Real T_start, Real T_end) {
  if (m_initialized) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: transition must be set before initialization."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (n_start <= n_end) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: density transition start (" << n_start
              << ") is not larger than end (" << n_end << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (T_start <= T_end) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: temperature transition start (" << T_start
              << ") is not larger than end (" << T_end << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  trans_ln_start   = std::log(n_start);
  trans_ln_end     = std::log(n_end);
  m_trans_ln_width = std::log(n_start/n_end);
  trans_T_start    = T_start;
  trans_T_end      = T_end;
  m_trans_T_width  = T_start - T_end;
}

template<typename LogPolicy>
void EOSTransition<LogPolicy>::SetBaryonMass(Real new_mb) {
  helmholtz_eos.SetBaryonMass(new_mb);
  compose_eos.SetBaryonMass(new_mb);
  min_Y[SCEB] = mFe/(56.0*new_mb) - 1.0;  // most bound nucleus is Fe-56
  max_Y[SCEB] = mn/new_mb - 1.0;          // free neutron limit
  m_min_h     = mFe/56.0;
  mb          = new_mb;
}

template<typename LogPolicy>
void EOSTransition<LogPolicy>::update_bounds() {
  min_n       = helmholtz_eos.min_n;
  min_T       = helmholtz_eos.min_T;
  max_n       = compose_eos.max_n;
  max_T       = compose_eos.max_T;
  min_Y[SCYE] = compose_eos.min_Y[SCYE];
  max_Y[SCYE] = compose_eos.max_Y[SCYE];

  if (std::isnan(m_helm_n_max)) {
    m_helm_n_max = std::min(helmholtz_eos.max_n, 1e-6);
  }
  if (std::isnan(m_helm_T_max)) {
    m_helm_T_max = std::min(helmholtz_eos.max_T, compose_eos.max_T);
  }

  Real const ln10 = std::log(10.0);
  m_ln_n_h0      = std::log(m_helm_n_max) - m_helm_n_ramp_dec*ln10;
  m_id_ln_n_ramp = 1.0/(m_helm_n_ramp_dec*ln10);
  m_ln_T_h0      = std::log(m_helm_T_max) - m_helm_T_ramp_dec*ln10;
  m_id_ln_T_ramp = 1.0/(m_helm_T_ramp_dec*ln10);

  Real const edge_tol = 1.0 - 1e-10;
  if (trans_T_end < compose_eos.min_T*edge_tol) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: trans_T_end (" << trans_T_end
              << ") below compose table minimum temperature (" << compose_eos.min_T
              << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (trans_T_start > std::min(compose_eos.max_T, helmholtz_eos.max_T)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: trans_T_start (" << trans_T_start
              << ") above joint table maximum temperature." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (std::exp(trans_ln_end) < compose_eos.min_n*edge_tol) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: trans_n_end (" << std::exp(trans_ln_end)
              << ") below compose table minimum density (" << compose_eos.min_n
              << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (std::exp(trans_ln_start) > std::min(compose_eos.max_n, helmholtz_eos.max_n)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: trans_n_start (" << std::exp(trans_ln_start)
              << ") above joint table maximum density." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (m_helm_n_max > helmholtz_eos.max_n) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: helm_n_max (" << m_helm_n_max
              << ") above helmholtz table maximum density (" << helmholtz_eos.max_n
              << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Compose temperature-grid indices. The compose grid stores log2_(T); its
  // spacing inverse is m_id_log_t, and node 0 corresponds to log2_(min_T).
  const Real lt0  = log2_(compose_eos.min_T);
  const Real idlt = compose_eos.m_id_log_t;
  comp_it_trans_start = (log2_(trans_T_start) - lt0)*idlt + 1;
  comp_it_trans_end   = (log2_(trans_T_end)   - lt0)*idlt;
  comp_it_helm_tmax   = (log2_(m_helm_T_max)  - lt0)*idlt + 1;

  int const it_max    = static_cast<int>(compose_eos.m_nt) - 1;
  comp_it_trans_end   = std::max(0, std::min(comp_it_trans_end, it_max - 1));
  comp_it_trans_start = std::max(comp_it_trans_end + 1,
                                 std::min(comp_it_trans_start, it_max));
  comp_it_helm_tmax   = std::max(comp_it_trans_end + 1,
                                 std::min(comp_it_helm_tmax, it_max));
}

template<typename LogPolicy>
void EOSTransition<LogPolicy>::InitializeTables(std::string fname,
                                                std::string helm_fname,
                                                Real baryon_mass) {
  if (m_initialized) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition: InitializeTables should only be called once."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  compose_eos.ReadTableFromFile(fname);
  if (!compose_eos.m_has_composition) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "EOSTransition requires the CompOSE composition channels Y[n], Y[p], "
              << "Y[He4], A[N], Z[N] and Y[N]; '" << fname << "' is missing at least "
              << "one of them (see the message above). Regenerate the table with "
              << "pycompose including the composition datasets." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  helmholtz_eos.ReadTableFromFile(helm_fname, compose_eos.min_Y[SCYE],
                                  compose_eos.max_Y[SCYE]);
  // Default transition strips: T in [0.5, 0.6] MeV (just below NSE dropout at
  // T9 = 7); density one decade above the compose low-density edge (fallback).
  if (std::isnan(m_trans_T_width) || std::isnan(m_trans_ln_width)) {
    SetTransition(10.0*compose_eos.min_n, compose_eos.min_n, 0.6, 0.5);
  }
  SetBaryonMass(baryon_mass);
  update_bounds();
  m_initialized = true;
}

template class EOSTransition<NormalLogs>;
template class EOSTransition<NQTLogs>;

}  // namespace Primitive
