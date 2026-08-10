//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacities_nn.cpp
//! \brief NN emulator (batched LibTorch) as drop-in replacement for the
//!        bns_nurates quadrature kernel.
//!
//! The NN replaces only the per-cell bns_nurates() quadrature call.
//! All corr_fac / Kirchhoff post-processing is copy-pasted verbatim from
//! radiation_m1_calc_opacities_nurates.cpp.

#if ENABLE_NURATES && ENABLE_NN_OPACITY

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_nurates.hpp"

#include <vector>

namespace radiationm1 {

// 8-channel layout per species in the NN output tensor [N, 4×8 = 32].
// The model takes 8 EOS features and outputs all 4 species at once (no one-hot).
// Channels within each species block s*NN_NCH + ch:
//   0  eta_0_th          [nm^-3 s^-1]   number emissivity, thermal (pair+brem)
//   1  kappa_0_a_th      [nm^-1]        number absorption, thermal
//   2  eta_0_non_th      [nm^-3 s^-1]   number emissivity, NEPS
//   3  kappa_0_a_non_th  [nm^-1]        number absorption, NEPS
//   4  eta_th            [MeV nm^-3 s^-1] energy emissivity, thermal
//   5  kappa_a_th        [nm^-1]        energy absorption, thermal
//   6  eta_non_th        [MeV nm^-3 s^-1] energy emissivity, NEPS
//   7  kappa_a_non_th    [nm^-1]        energy absorption, NEPS
// scattering (kappa_s) is not emulated; it remains zero (iso is cheap exact).
static constexpr int NN_CH_ETA_0_TH         = 0;
static constexpr int NN_CH_KAPPA_0_A_TH     = 1;
static constexpr int NN_CH_ETA_0_NON_TH     = 2;
static constexpr int NN_CH_KAPPA_0_A_NON_TH = 3;
static constexpr int NN_CH_ETA_TH           = 4;
static constexpr int NN_CH_KAPPA_A_TH       = 5;
static constexpr int NN_CH_ETA_NON_TH       = 6;
static constexpr int NN_CH_KAPPA_A_NON_TH   = 7;
static constexpr int NN_NCH  = 8;              // channels per species
static constexpr int NN_NSP  = 4;              // species
static constexpr int NN_NOUT = NN_NSP * NN_NCH; // 32 total outputs per cell
static constexpr int NN_NEOS = 8;              // EOS input features
static constexpr int NN_NIN  = NN_NEOS;        // 8 (no one-hot encoding)

template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcOpacityNN_(Driver *pdrive, int stage) {
  assert(((nspecies == 3) || (nspecies == 4)));

  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  const int nmb = nmb1 + 1;
  auto &nspecies_ = nspecies;
  auto nvars_ = nvars;

  auto &adm = pmy_pack->padm->adm;
  auto &radiation_mask_ = radiation_mask;
  auto &m1_params_    = params;
  auto &nurates_params_ = nurates_params;

  auto &eta_0_ = eta_0;
  auto &abs_0_ = abs_0;
  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  auto &u0_ = u0;
  auto &chi_ = chi;

  DvceArray5D<Real> w0_ = w0;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmy_pack->pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.GetBaryonMass();

  auto code_units    = eos.GetCodeUnitSystem();
  auto eos_units     = eos.GetEOSUnitSystem();
  auto nurates_units = Primitive::MakeNGS();

  // Unit conversion factors (code → NGS)
  Real const unit_length       = code_units.LengthConversion(nurates_units);
  Real const unit_time         = code_units.TimeConversion(nurates_units);
  Real const unit_num_dens     = eos_units.NumberDensityConversion(nurates_units);
  Real const unit_ene_dens     = code_units.EnergyDensityConversion(nurates_units);
  Real const unit_num_dens_dot = unit_num_dens / unit_time;
  Real const unit_ene_dens_dot = unit_ene_dens / unit_time;

  // ── dimensions ──────────────────────────────────────────────────────────────
  const int nk = ke - ks + 1;
  const int nj = je - js + 1;
  const int ni = ie - is + 1;
  const int ncells_per_mb = nk * nj * ni;
  const int N_total = nmb * ncells_per_mb;

  // ── capture-friendly copies of loop bounds / scalars ───────────────────────
  const int ks_ = ks, ke_ = ke, js_ = js, je_ = je, is_ = is, ie_ = ie;
  const int nj_ = nj, ni_ = ni, ncells_ = ncells_per_mb;
  const Real unit_length_       = unit_length;
  const Real unit_num_dens_dot_ = unit_num_dens_dot;
  const Real unit_ene_dens_dot_ = unit_ene_dens_dot;
  const Real unit_num_dens_     = unit_num_dens;
  const Real beta_dt_           = beta_dt;
  const Real mb_                = mb;

  // ── 1. Device gather: EOS inputs (8 features per cell, no species tiling) ────
  // The 8→32 model takes EOS features only; species symmetry is baked in
  // structurally (anux pair/brem = nux pair/brem, NEPS predicted separately).
  //
  // Layout:  eos_dev(flat, col)     col = nb_nm3,T,ye,yn,yp,mu_n,mu_p,mu_e
  //          x_full_dev(flat, col)  col = normalized EOS (8, no one-hot)
  // Grow-only persistent scratch: (re)allocate only when the local cell count
  // exceeds the current capacity, then reuse across steps.  This removes the
  // per-step cudaMalloc/cudaFree calls that can serialize the device.  Active
  // rows are fully written before use, so no per-step zero-initialisation is needed.
  // Dynamic batch: the forward runs on all N_total local cells (no padding).
  // Inject the Kokkos CUDA stream so the LibTorch forward and the profiler events
  // run on it (the emulator TU is kept free of Kokkos headers for build speed).
  nn_emulator.SetStream(static_cast<void *>(Kokkos::Cuda().cuda_stream()));
  nn_emulator.ProfilePollAndReport();
  const bool nn_scratch_will_grow = N_total > nn_scratch_capacity_;
  const size_t nn_scratch_bytes = static_cast<size_t>(N_total) *
      (static_cast<size_t>(NN_NEOS + NN_NIN + NN_NOUT) * sizeof(float) +
       sizeof(bool) + static_cast<size_t>(8 + 16) * sizeof(Real));
  nn_emulator.ProfileBegin(N_total, nn_scratch_will_grow, nn_scratch_bytes);
  if (N_total > nn_scratch_capacity_) {
    Kokkos::realloc(nn_eos_dev_,    N_total, NN_NEOS);
    Kokkos::realloc(nn_x_full_dev_, N_total, NN_NIN);
    Kokkos::realloc(nn_valid_view_, N_total);
    Kokkos::realloc(nn_view_,       static_cast<size_t>(N_total) * NN_NOUT);
    Kokkos::realloc(nn_m1_moments_, N_total, 8);
    Kokkos::realloc(nn_non_th_buf_, N_total, 16);
    nn_scratch_capacity_ = N_total;
  }
  // Local handle-copies (share the persistent storage; NOT captured via `this`).
  auto eos_dev    = nn_eos_dev_;
  auto x_full_dev = nn_x_full_dev_;
  auto valid_view = nn_valid_view_;

  auto &radiation_mask_cap = radiation_mask_;

  Kokkos::Array<float, NN_NEOS> nn_in_mean{};
  Kokkos::Array<float, NN_NEOS> nn_in_std{};
  for (int c = 0; c < NN_NEOS; ++c) {
    nn_in_mean[c] = nn_emulator.HostInMean()[c];
    nn_in_std[c]  = nn_emulator.HostInStd()[c];
  }

  Kokkos::Profiling::pushRegion("NN::gather_pack");
  par_for(
      "radiation_m1_nn_gather_pack", DevExeSpace(), 0, nmb1, ks, ke, js, je,
      is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int flat = m * ncells_
                       + (k - ks_) * nj_ * ni_
                       + (j - js_) * ni_
                       + (i - is_);

        if (radiation_mask_cap(m, k, j, i)) {
          valid_view(flat) = false;
          for (int c = 0; c < NN_NIN; ++c) {
            eos_dev(flat, c) = 0.f;
            x_full_dev(flat, c) = 0.f;
          }
          return;
        }
        valid_view(flat) = true;

        Real nb   = w0_(m, IDN, k, j, i) / mb_;
        Real p    = w0_(m, IPR, k, j, i);
        Real Y    = w0_(m, IYF, k, j, i);
        Real T    = eos.GetTemperatureFromP(nb, p, &Y);
        Real yp   = eos.GetProtonFraction(nb, T, &Y);
        Real yn   = eos.GetNeutronFraction(nb, T, &Y);
        Real mu_b = eos.GetBaryonChemicalPotential(nb, T, &Y);
        Real mu_q = eos.GetChargeChemicalPotential(nb, T, &Y);
        Real mu_le= eos.GetElectronLeptonChemicalPotential(nb, T, &Y);

        eos_dev(flat, 0) = static_cast<float>(nb * unit_num_dens_);  // nb [nm^-3]
        eos_dev(flat, 1) = static_cast<float>(T);                    // T  [MeV]
        eos_dev(flat, 2) = static_cast<float>(Y);                    // ye
        eos_dev(flat, 3) = static_cast<float>(yn);
        eos_dev(flat, 4) = static_cast<float>(yp);
        eos_dev(flat, 5) = static_cast<float>(mu_b);                 // mu_n
        eos_dev(flat, 6) = static_cast<float>(mu_b + mu_q);          // mu_p
        eos_dev(flat, 7) = static_cast<float>(mu_le - mu_q);         // mu_e

        // Normalize: log10 for nb and T, z-score for the rest.
        x_full_dev(flat, 0) =
            (static_cast<float>(Kokkos::log10(nb * unit_num_dens_)) -
             nn_in_mean[0]) / nn_in_std[0];
        x_full_dev(flat, 1) =
            (static_cast<float>(Kokkos::log10(T)) - nn_in_mean[1]) / nn_in_std[1];
        for (int c = 2; c < NN_NIN; ++c) {
          x_full_dev(flat, c) = (eos_dev(flat, c) - nn_in_mean[c]) / nn_in_std[c];
        }
      });
  nn_emulator.ProfileMark(NNProfilePoint::gather);
  // SCALING FIX: fence removed — the torch forward now runs on the Kokkos stream
  // (see InferPrebuilt), so it is ordered after this gather with no barrier.
  // Kokkos::fence();
  Kokkos::Profiling::popRegion();

  // ── 2. GPU-resident NN inference — no PCIe transfers ─────────────────────────
  // Output layout: nn_view(flat * NN_NOUT + s*NN_NCH + ch)
  // i.e. (N_total, NN_NSP=4, NN_NCH=8) stored row-major as (N_total, 32).
  auto nn_view = nn_view_;   // persistent (grow-only) buffer, N_total × 32 in use

  Kokkos::Profiling::pushRegion("NN::InferPrebuilt");
  // GPU-resident LibTorch forward on the Kokkos CUDA stream (dynamic N_total,
  // zero-copy input).  Runs in-line with the gather (before) and readout (after)
  // on the shared stream, so no cross-stream fence/event handshake is needed.
  nn_emulator.InferPrebuilt(x_full_dev.data(), nn_view.data(), N_total);
  nn_emulator.ProfileMark(NNProfilePoint::forward);
  Kokkos::Profiling::popRegion();

  // ── 3. Readout: metric reconstruction + J/rnnu + NN→code conversion ──────────
  // Intermediate buffer storing J[0..3] and rnnu[0..3] per cell so the
  // Kirchhoff kernel does not need to redo the metric/closure computation.
  // Layout: m1_moments(flat, 0..3) = J[s], m1_moments(flat, 4..7) = rnnu[s]
  auto m1_moments = nn_m1_moments_;   // persistent (grow-only) buffer

  // Non-thermal (NEPS) components from NN, needed by the Kirchhoff kernel to
  // apply corr_fac only to the thermal part while keeping NEPS unchanged.
  // Layout [flat, col]:
  //   col  0..3  abs_0_non_th[s]  number absorption,  code units
  //   col  4..7  abs_1_non_th[s]  energy absorption,  code units
  //   col  8..11 eta_0_non_th[s]  number emissivity,  code units
  //   col 12..15 eta_1_non_th[s]  energy emissivity,  code units
  // fac×2 for nux/anux already applied to eta columns; abs columns have no fac.
  auto non_th_buf = nn_non_th_buf_;   // persistent (grow-only) buffer

  // Capture NN output normalization stats (32-dim) for fused denorm.
  // InferPrebuilt writes raw normalized values; readout applies:
  //   physical = 10^(y_norm * std + mean) = exp(LN10 * (y_norm * std + mean))
  Kokkos::Array<float, NN_NOUT> nn_out_mean_cap{}, nn_out_std_cap{};
  for (int c = 0; c < NN_NOUT; ++c) {
    nn_out_mean_cap[c] = nn_emulator.HostOutMean()[c];
    nn_out_std_cap[c]  = nn_emulator.HostOutStd()[c];
  }

  Kokkos::Profiling::pushRegion("NN::readout");
  par_for(
      "radiation_m1_nn_readout", DevExeSpace(), 0, nmb1, ks, ke, js, je,
      is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int flat = m * ncells_
                       + (k - ks_) * nj_ * ni_
                       + (j - js_) * ni_
                       + (i - is_);

        if (!valid_view(flat)) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            abs_0_(m, nuidx, k, j, i) = 0;
            eta_0_(m, nuidx, k, j, i) = 0;
            abs_1_(m, nuidx, k, j, i) = 0;
            eta_1_(m, nuidx, k, j, i) = 0;
            scat_1_(m, nuidx, k, j, i) = 0;
          }
          for (int c = 0; c < 8; ++c) m1_moments(flat, c) = 0.0;
          return;
        }

        // ── metric reconstruction (needed for J, rnnu) ─────────────────────
        Real garr_dd[16];
        Real garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
        pack_n_d(adm.alpha(m, k, j, i), n_d);
        adm::SpacetimeMetric(
            adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
            adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
            garr_dd);
        adm::SpacetimeUpperMetric(
            adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
            adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
            garr_uu);
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }

        Real gam = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        Real volform = Kokkos::sqrt(gam);

        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

        Real w_lorentz =
            get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                          w0_(m, IVZ, k, j, i), g_dd);
        pack_u_u(
            w_lorentz / adm.alpha(m, k, j, i),
            w0_(m, IVX, k, j, i) -
                w_lorentz * adm.beta_u(m, 0, k, j, i) / adm.alpha(m, k, j, i),
            w0_(m, IVY, k, j, i) -
                w_lorentz * adm.beta_u(m, 1, k, j, i) / adm.alpha(m, k, j, i),
            w0_(m, IVZ, k, j, i) -
                w_lorentz * adm.beta_u(m, 2, k, j, i) / adm.alpha(m, k, j, i),
            u_u);
        pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                 adm.beta_u(m, 2, k, j, i), v_u);
        tensor_contract(g_dd, u_u, u_d);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        Real J[4]{}, rnnu[4]{};
        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                   F_d);
          const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
          apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                        F_d, chi_(m, nuidx, k, j, i), P_dd, m1_params_);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
          assemble_rT(n_d, E, F_d, P_dd, T_dd);
          J[nuidx] = calc_J_from_rT(T_dd, u_u);
          Real Gamma =
              compute_Gamma(w_lorentz, v_u, J[nuidx], E, F_d, m1_params_);
          rnnu[nuidx] =
              u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma;
        }

        // Store J and rnnu for the Kirchhoff kernel (avoids redoing metric).
        for (int s = 0; s < nspecies_; ++s) {
          m1_moments(flat, s)     = J[s];
          m1_moments(flat, 4 + s) = rnnu[s];
        }

        // ── denormalize all 32 NN outputs and map to M1 opacity fields ──
        // physical = 10^(y_norm * std + mean) = exp(LN10 * (y_norm*std + mean))
        // Layout: nn_view[flat*32 + s*8 + ch], 8 channels per species.
        //
        // M1 fields assembled by summing thermal (pair+brem) and non-thermal
        // (NEPS) contributions.  scattering (kappa_s) is not emulated by the
        // 2D model — it remains zero here (iso scattering is fast exact 1D).
        //
        // Factor of 2 for nux/anux: bns_nurates uses "nux = mu OR tau",
        // AthenaK needs "nux = mu AND tau" (see radiation_m1_nurates.hpp).
        static constexpr Real LN10 = 2.302585092994046;
        const int base = flat * NN_NOUT;   // flat × 32
        Real nn_phys[NN_NOUT];
        for (int q = 0; q < NN_NOUT; ++q) {
          nn_phys[q] = Kokkos::exp(
              LN10 * (static_cast<Real>(nn_view(base + q))
                      * nn_out_std_cap[q] + nn_out_mean_cap[q]));
        }
        for (int s = 0; s < NN_NSP; ++s) {
          const int sb = s * NN_NCH;
          Real fac = ((s == 2) || (s == 3)) ? 2.0 : 1.0;  // nux/anux: mu-OR-tau → mu-AND-tau
          // total kappa (thermal + non-thermal), nm^-1 → code; no fac on absorption
          abs_0_(m, s, k, j, i) = (nn_phys[sb + NN_CH_KAPPA_0_A_TH] +
                                    nn_phys[sb + NN_CH_KAPPA_0_A_NON_TH])
                                   * unit_length_;
          abs_1_(m, s, k, j, i) = (nn_phys[sb + NN_CH_KAPPA_A_TH] +
                                    nn_phys[sb + NN_CH_KAPPA_A_NON_TH])
                                   * unit_length_;
          scat_1_(m, s, k, j, i) = 0.0;   // iso not in 2D model
          // total emissivity (thermal + non-thermal)
          eta_0_(m, s, k, j, i) = fac * (nn_phys[sb + NN_CH_ETA_0_TH] +
                                          nn_phys[sb + NN_CH_ETA_0_NON_TH])
                                   / unit_num_dens_dot_;
          eta_1_(m, s, k, j, i) = fac * (nn_phys[sb + NN_CH_ETA_TH] +
                                          nn_phys[sb + NN_CH_ETA_NON_TH])
                                   / unit_ene_dens_dot_;
          // non-thermal (NEPS) parts separately for Kirchhoff thermal/non-thermal split
          non_th_buf(flat, s)      = nn_phys[sb + NN_CH_KAPPA_0_A_NON_TH] * unit_length_;
          non_th_buf(flat, 4 + s)  = nn_phys[sb + NN_CH_KAPPA_A_NON_TH]   * unit_length_;
          non_th_buf(flat, 8 + s)  = fac * nn_phys[sb + NN_CH_ETA_0_NON_TH] / unit_num_dens_dot_;
          non_th_buf(flat, 12 + s) = fac * nn_phys[sb + NN_CH_ETA_NON_TH]   / unit_ene_dens_dot_;
        }
        (void)volform;
      });
  nn_emulator.ProfileMark(NNProfilePoint::readout);
  // SCALING FIX: profiling fence removed for comm/compute overlap; the next
  // kernel (1D_exact) is ordered on the same Kokkos stream.
  // Kokkos::fence();
  Kokkos::Profiling::popRegion();

  // ── 3b. 1D exact: β-processes (abs_em) + iso scattering ─────────────────────
  // bns_nurates called with only 1D processes (pair/brem/inelastic disabled).
  // Results ADDED to the NN 2D outputs already written to abs_0_, abs_1_,
  // eta_0_, eta_1_.  Iso fills scat_1_ (which was zeroed in the readout step).
  // This completes the hybrid: NN(pair+brem+NEPS) + exact(β+iso) = full opacity.
  {
    NuratesParams params_1d = nurates_params_;
    params_1d.use_pair            = false;  // NN already covers pair
    params_1d.use_brem            = false;  // NN already covers brem
    params_1d.use_inelastic_scatt = false;  // NN already covers NEPS (inelastic)

    if (params_1d.use_abs_em || params_1d.use_iso) {
      Kokkos::Profiling::pushRegion("NN::1D_exact");
      const NuratesParams params_1d_cap    = params_1d;
      const auto          code_units_cap   = code_units;
      const auto          eos_units_cap    = eos_units;
      const auto          nurates_units_cap = nurates_units;
      par_for(
          "radiation_m1_nn_1d_exact", DevExeSpace(), 0, nmb1, ks, ke, js, je,
          is, ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const int flat = m * ncells_
                           + (k - ks_) * nj_ * ni_
                           + (j - js_) * ni_
                           + (i - is_);
            if (!valid_view(flat)) return;

            // Spatial determinant for undensitizing M1 moments
            Real gam = adm::SpatialDet(
                adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
            Real inv_volform = 1.0 / Kokkos::sqrt(gam);

            // EOS state (read from gather-kernel cache; same layout as Kirchhoff)
            Real nb   = static_cast<Real>(eos_dev(flat, 0)) / unit_num_dens_;
            Real T    = static_cast<Real>(eos_dev(flat, 1));
            Real yn   = static_cast<Real>(eos_dev(flat, 3));
            Real yp   = static_cast<Real>(eos_dev(flat, 4));
            Real mu_n = static_cast<Real>(eos_dev(flat, 5));
            Real mu_p = static_cast<Real>(eos_dev(flat, 6));
            Real mu_e = static_cast<Real>(eos_dev(flat, 7));

            // Fluid-frame radiation moments (undensitized, code units)
            Real nudens_0[4]{}, nudens_1[4]{}, chi_loc[4]{};
            for (int s = 0; s < nspecies_; ++s) {
              nudens_1[s] = m1_moments(flat, s)     * inv_volform;  // J/volform
              nudens_0[s] = m1_moments(flat, 4 + s) * inv_volform;  // rnnu/volform
              chi_loc[s]  = chi_(m, s, k, j, i);
            }

            Real eta_0_loc[4]{}, eta_1_loc[4]{};
            Real abs_0_loc[4]{}, abs_1_loc[4]{};
            Real scat_0_loc[4]{}, scat_1_loc[4]{};
            // non-thermal outputs required by the signature but unused here:
            // β+iso have no NEPS component, so these stay zero.
            Real eta_1_non_th[4]{}, abs_1_non_th[4]{};
            Real eta_0_non_th[4]{}, abs_0_non_th[4]{};
            bns_nurates(nb, T, yp, yn, mu_n, mu_p, mu_e,
                        nudens_0, nudens_1, chi_loc,
                        eta_0_loc, eta_1_loc,
                        abs_0_loc, abs_1_loc,
                        scat_0_loc, scat_1_loc,
                        eta_1_non_th, abs_1_non_th,
                        eta_0_non_th, abs_0_non_th,
                        params_1d_cap, code_units_cap, eos_units_cap,
                        nurates_units_cap);

            // Add 1D contributions on top of NN 2D outputs already in the arrays.
            // bns_nurates already applies the factor-of-2 for nux/anux emissivities.
            for (int s = 0; s < nspecies_; ++s) {
              abs_0_(m, s, k, j, i) += abs_0_loc[s];
              abs_1_(m, s, k, j, i) += abs_1_loc[s];
              eta_0_(m, s, k, j, i) += eta_0_loc[s];
              eta_1_(m, s, k, j, i) += eta_1_loc[s];
              scat_1_(m, s, k, j, i) += scat_1_loc[s];
            }
          });
      // SCALING FIX: profiling fence removed for overlap; kirchhoff is ordered
      // on the same Kokkos stream.
      // Kokkos::fence();
      Kokkos::Profiling::popRegion();
    }
  }
  nn_emulator.ProfileMark(NNProfilePoint::exact_1d);

  // ── 4. Kirchhoff / corr_fac / NeutrinoDens ───────────────────────────────────
  // Reads raw opacities from output arrays + J/rnnu from m1_moments.
  // Applies NeutrinoDens + corr_fac + Kirchhoff in-place.
  Kokkos::Profiling::pushRegion("NN::kirchhoff");
  par_for(
      "radiation_m1_nn_kirchhoff", DevExeSpace(), 0, nmb1, ks, ke, js, je,
      is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const int flat = m * ncells_
                       + (k - ks_) * nj_ * ni_
                       + (j - js_) * ni_
                       + (i - is_);

        if (!valid_view(flat)) return;

        // Restore J and rnnu from intermediate buffer.
        Real J[4]{}, rnnu[4]{};
        for (int s = 0; s < nspecies_; ++s) {
          J[s]    = m1_moments(flat, s);
          rnnu[s] = m1_moments(flat, 4 + s);
        }

        // Read raw opacities written by the readout kernel.
        Real abs_0_loc[4]{}, abs_1_loc[4]{}, scat_1_loc[4]{};
        Real eta_0_loc[4]{}, eta_1_loc[4]{};
        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          abs_0_loc[nuidx]  = abs_0_(m, nuidx, k, j, i);
          abs_1_loc[nuidx]  = abs_1_(m, nuidx, k, j, i);
          scat_1_loc[nuidx] = scat_1_(m, nuidx, k, j, i);
          eta_0_loc[nuidx]  = eta_0_(m, nuidx, k, j, i);
          eta_1_loc[nuidx]  = eta_1_(m, nuidx, k, j, i);
        }

        // EOS from gather-kernel cache (no table re-evaluation).
        const Real nb  = static_cast<Real>(eos_dev(flat, 0)) / unit_num_dens_;
        const Real T   = static_cast<Real>(eos_dev(flat, 1));
        const Real Y   = static_cast<Real>(eos_dev(flat, 2));
        const Real mu_n= static_cast<Real>(eos_dev(flat, 5));
        const Real mu_p= static_cast<Real>(eos_dev(flat, 6));
        const Real mu_e= static_cast<Real>(eos_dev(flat, 7));

        // ── NeutrinoDens + corr_fac + Kirchhoff ────────────────────────────
        Real tau{}, nudens_0_trap[4]{}, nudens_1_trap[4]{},
            nudens_0_thin[4]{}, nudens_1_thin[4]{};

        if (nurates_params_.use_kirchhoff_law ||
            nurates_params_.use_equilibrium_distribution) {
          tau =
              Kokkos::min(
                  Kokkos::sqrt(abs_1_loc[0] * (abs_1_loc[0] + scat_1_loc[0])),
                  Kokkos::sqrt(abs_1_loc[1] *
                               (abs_1_loc[1] + scat_1_loc[1]))) *
              beta_dt_;

          if (nurates_params_.opacity_tau_trap >= 0 &&
              tau > nurates_params_.opacity_tau_trap) {
            Real n_nu[6] = {rnnu[0],      rnnu[1],      rnnu[2] / 2.,
                            rnnu[3] / 2., rnnu[2] / 2., rnnu[3] / 2.};
            Real Y_part[3] = {Y, 0., 0.};

            Real Y_lep[3]{};
            eos.GetLeptonFractions(nb, Y_part, n_nu, Y_lep);
            Real Y_guess[3] = {Y_lep[0], Y_lep[1], Y_lep[2]};

            Real e = eos.GetEnergy(nb, T, Y_part) + J[0] + J[1] + J[2] + J[3];

            Real temperature_trap{}, Y_e_trap[3]{};
            bool res = eos.GetBetaEquilibriumTrapped(
                nb, e, Y_lep, temperature_trap, &Y_e_trap[0], T, Y_guess);

            if (res) {
              Real e_zero = eos.GetEnergy(nb, T, Y_part);
              bool res2 = eos.GetBetaEquilibriumTrapped(
                  nb, e_zero, Y_part, temperature_trap, &Y_e_trap[0], T,
                  Y_part);
              (void)res2;
            }

            Real mu_b_eq = eos.GetBaryonChemicalPotential(
                nb, temperature_trap, &Y_e_trap[0]);
            Real mu_q_eq = eos.GetChargeChemicalPotential(
                nb, temperature_trap, &Y_e_trap[0]);
            Real mu_le_eq = eos.GetElectronLeptonChemicalPotential(
                nb, temperature_trap, &Y_e_trap[0]);

            NeutrinoDens(mu_b_eq, mu_b_eq + mu_q_eq, mu_le_eq - mu_q_eq,
                         temperature_trap,
                         nudens_0_trap[0], nudens_0_trap[1], nudens_0_trap[2],
                         nudens_1_trap[0], nudens_1_trap[1], nudens_1_trap[2],
                         nurates_params_, code_units, eos_units, nurates_units);

            nudens_0_trap[2] *= 0.5;
            nudens_1_trap[2] *= 0.5;
            nudens_0_trap[3] = nudens_0_trap[2];
            nudens_1_trap[3] = nudens_1_trap[2];
          }

          NeutrinoDens(mu_n, mu_p, mu_e, T, nudens_0_thin[0],
                       nudens_0_thin[1], nudens_0_thin[2], nudens_1_thin[0],
                       nudens_1_thin[1], nudens_1_thin[2], nurates_params_,
                       code_units, eos_units, nurates_units);

          nudens_0_thin[2] *= 0.5;
          nudens_1_thin[2] *= 0.5;
          nudens_0_thin[3] = nudens_0_thin[2];
          nudens_1_thin[3] = nudens_1_thin[2];
        }

        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          Real my_nudens_0{}, my_nudens_1{}, corr_fac{1};
          if (nurates_params_.use_kirchhoff_law ||
              nurates_params_.use_equilibrium_distribution) {
            if (nurates_params_.opacity_tau_trap < 0 ||
                tau <= nurates_params_.opacity_tau_trap) {
              my_nudens_0 = nudens_0_thin[nuidx];
              my_nudens_1 = nudens_1_thin[nuidx];
            } else if (tau > nurates_params_.opacity_tau_trap +
                               nurates_params_.opacity_tau_delta) {
              my_nudens_0 = nudens_0_trap[nuidx];
              my_nudens_1 = nudens_1_trap[nuidx];
            } else {
              Real const lam = (tau - nurates_params_.opacity_tau_trap) /
                               nurates_params_.opacity_tau_delta;
              my_nudens_0 = lam * nudens_0_trap[nuidx] +
                            (1 - lam) * nudens_0_thin[nuidx];
              my_nudens_1 = lam * nudens_1_trap[nuidx] +
                            (1 - lam) * nudens_1_thin[nuidx];
            }
          }

          corr_fac = 1.0;
          if (nurates_params_.use_equilibrium_distribution) {
            corr_fac = (J[nuidx] / rnnu[nuidx]) * (my_nudens_0 / my_nudens_1);
            if (!Kokkos::isfinite(corr_fac)) corr_fac = 1.0;
            corr_fac *= corr_fac;
            corr_fac = Kokkos::max(
                1.0 / nurates_params_.opacity_corr_fac_max,
                Kokkos::min(corr_fac, nurates_params_.opacity_corr_fac_max));
          }

          // Scattering correction: applied to ALL flavors.
          scat_1_(m, nuidx, k, j, i) = scat_1_loc[nuidx] * corr_fac;

          // CC correction (kappa ~ E^2): only charged-current species nue(0)/anue(1).
          // Heavy leptons (nux=2, anux=3) have no CC absorption, corr_ae stays 1.
          Real corr_ae = (nuidx == 0 || nuidx == 1) ? corr_fac : 1.0;

          if (nurates_params_.use_kirchhoff_law) {
            // Non-thermal (NEPS) components stored from NN readout kernel.
            // The 1D β-processes added in step 3b are thermal+CC, so they belong
            // in the thermal bucket: thermal = (total in arrays) - (NN non-th).
            Real abs_0_non_th = non_th_buf(flat, nuidx);
            Real abs_1_non_th = non_th_buf(flat, 4 + nuidx);
            Real eta_0_non_th = non_th_buf(flat, 8 + nuidx);
            Real eta_1_non_th = non_th_buf(flat, 12 + nuidx);
            // Apply corr_ae only to the thermal part; NEPS is left unchanged.
            Real abs_0_th_corr =
                Kokkos::fmax(abs_0_loc[nuidx] - abs_0_non_th, 0.0) * corr_ae;
            Real abs_1_th_corr =
                Kokkos::fmax(abs_1_loc[nuidx] - abs_1_non_th, 0.0) * corr_ae;
            abs_0_(m, nuidx, k, j, i) = abs_0_th_corr + abs_0_non_th;
            abs_1_(m, nuidx, k, j, i) = abs_1_th_corr + abs_1_non_th;
            // Kirchhoff on thermal part; non-thermal emissivity kept separate.
            eta_0_(m, nuidx, k, j, i) = (abs_0_th_corr > 0)
                ? abs_0_th_corr * my_nudens_0 + eta_0_non_th
                : eta_0_loc[nuidx];
            eta_1_(m, nuidx, k, j, i) = (abs_1_th_corr > 0)
                ? abs_1_th_corr * my_nudens_1 + eta_1_non_th
                : eta_1_loc[nuidx];
          } else {
            if (nuidx == 0 || nuidx == 1) {
              abs_0_(m, nuidx, k, j, i) = abs_0_loc[nuidx] * corr_fac;
              abs_1_(m, nuidx, k, j, i) = abs_1_loc[nuidx] * corr_fac;
              eta_0_(m, nuidx, k, j, i) = eta_0_loc[nuidx] * corr_fac;
              eta_1_(m, nuidx, k, j, i) = eta_1_loc[nuidx] * corr_fac;
            }
          }
        }
      });  // par_for kirchhoff
  nn_emulator.ProfileMark(NNProfilePoint::kirchhoff);
  // Do not fence here.  Every NN path and all downstream RadiationM1 kernels use
  // the same Kokkos CUDA stream, so stream ordering supplies the data dependency.
  // The normal nurates opacity task likewise returns asynchronously.  A global
  // Kokkos::fence() here stalled the task scheduler and all Kokkos execution
  // instances, undoing the stream/allocation fixes above.
  Kokkos::Profiling::popRegion();

  return TaskStatus::complete;
}

}  // namespace radiationm1

// Explicit instantiations — required because the template definition lives in
// this TU while the call site is in radiation_m1_calc_opacities_nurates.cpp.
template TaskStatus radiationm1::RadiationM1::CalcOpacityNN_<
    Primitive::EOSCompOSE<Primitive::NQTLogs>,
    Primitive::ResetFloor>(Driver *, int);
template TaskStatus radiationm1::RadiationM1::CalcOpacityNN_<
    Primitive::EOSCompOSE<Primitive::NormalLogs>,
    Primitive::ResetFloor>(Driver *, int);

#endif  // ENABLE_NURATES && ENABLE_NN_OPACITY
