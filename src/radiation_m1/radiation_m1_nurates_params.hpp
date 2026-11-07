#ifndef RADIATION_M1_RADIATION_M1_NURATES_PARAMS_HPP_
#define RADIATION_M1_RADIATION_M1_NURATES_PARAMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nurates_params.hpp
//! \brief bns_nurates parameters without the reaction-kernel implementations

#include "config.hpp"

#if ENABLE_NURATES

#include "athena.hpp"
#include "bns_nurates_fdi_ns.hpp"

namespace radiationm1 {

struct NuratesParams {
  Real opacity_corr_fac_max;  // maximum correction factor for optically thin regime
  Real nb_min;
  Real temp_min_mev;
  Real max_recon_temp;  // [MeV] fall back to the equilibrium distribution when the
                        // reconstructed spectral temperature exceeds this
  bool use_recon_y_fallback;  // also fall back where y >= y3 (bns_nurates' fit edge)

  bool use_abs_em;
  bool use_pair;
  bool use_brem;
  bool use_iso;
  bool use_inelastic_scatt;
  bool use_WM_ab;
  bool use_WM_sc;
  bool use_dU;
  bool use_dm_eff;
  bool use_equilibrium_distribution;
  bool use_NN_medium_corr;
  bool neglect_blocking;
  bool use_decay;
  bool use_BRT_brem;
  Real exclude_pair_brem_nb_max;  // remove pair/brem from nue/anue in eq-spectrum
                                  // cells where the baryon number density is below
                                  // this [nm^-3]; <= 0: never

  int eq_warmup_cycles;  // force use_equilibrium_distribution for the first
                         // this-many cycles on a fresh start, to avoid
                         // reconstructing the distribution from the floored,
                         // neutrinoless initial field

  Real kirchhoff_tau_trap;  // use Kirchhoff only above this effective optical
                            // depth; < 0: always Kirchhoff

  // Partially-equilibrated (T*, Ye*) emissivity predictor, always run in Kirchhoff
  // cells: the equilibrium Kirchhoff's law is handed. The trapped weak equilibrium is
  // the w -> 1 endpoint of a one-parameter family in w = a/(1+a), a = dtau*kappa_abs,
  // and the local blackbody is the w -> 0 one; the predictor evaluates the family at
  // the w the cell actually has.
  Real peq_w_floor;             // skip the cell below this weight (tier-0 gate)
  Real peq_dlnT_tol;             // skip below this predicted |dlnT| (tier-1 gate)
  Real peq_dYe_tol;              // skip below this predicted |dYe|  (tier-1 gate)

  int quad_nx;  // no. of quadrature points for 1d integration (bns_nurates)
  bns_nurates::MyQuadrature quadrature;
  int quad_nx_2;
  bns_nurates::MyQuadrature quadrature_2;
};

}  // namespace radiationm1

#endif  // ENABLE_NURATES
#endif  // RADIATION_M1_RADIATION_M1_NURATES_PARAMS_HPP_
