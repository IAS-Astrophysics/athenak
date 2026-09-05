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
static constexpr int NN_NEOS = 8;              // EOS features gathered per cell
// NN input width — matches NNOpacityEmulator::N_INPUTS. eos_dev always holds all
// NN_NEOS features (the 1D/Kirchhoff reconstruction needs the chemical potentials);
// only the network input x_full_dev is reduced.
#if NN_REDUCED_INPUT
static constexpr int NN_NIN  = 3;              // reduced: nb, T, Ye
#else
static constexpr int NN_NIN  = NN_NEOS;        // 8 (full EOS feature set)
#endif

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
  // Local copy (not a reference to the member) so the eq-distribution warmup
  // override stays call-local.  On a fresh (neutrinoless) start the M1 moments
  // are floored, so reconstructing the spectrum yields garbage; force the
  // equilibrium distribution for the first eq_warmup_cycles cycles.  Feeds both
  // the 1D ComputeNuratesOpacities call and the Kirchhoff/peq reconstruction,
  // matching the standard nurates route (largesim-m1).
  NuratesParams nurates_params_ = nurates_params;
  if (pmy_pack->pmesh->ncycle < nurates_params_.eq_warmup_cycles) {
    nurates_params_.use_equilibrium_distribution = true;
  }

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
  const Real dt_full_           = pmy_pack->pmesh->dt;  // full step for the peq dtau
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
       sizeof(bool) + static_cast<size_t>(9 + 16) * sizeof(Real));
  nn_emulator.ProfileBegin(N_total, nn_scratch_will_grow, nn_scratch_bytes);
  if (N_total > nn_scratch_capacity_) {
    Kokkos::realloc(nn_eos_dev_,    N_total, NN_NEOS);
    Kokkos::realloc(nn_x_full_dev_, N_total, NN_NIN);
    Kokkos::realloc(nn_valid_view_, N_total);
    Kokkos::realloc(nn_view_,       static_cast<size_t>(N_total) * NN_NOUT);
    Kokkos::realloc(nn_m1_moments_, N_total, 9);  // 0-3 J/vf, 4-7 rnnu/vf, 8 dtau
    Kokkos::realloc(nn_non_th_buf_, N_total, 16);
    nn_scratch_capacity_ = N_total;
  }
  // Local handle-copies (share the persistent storage; NOT captured via `this`).
  auto eos_dev    = nn_eos_dev_;
  auto x_full_dev = nn_x_full_dev_;
  auto valid_view = nn_valid_view_;

  auto &radiation_mask_cap = radiation_mask_;

  // Input normalization stats are NN_NIN-dim (HostInMean/HostInStd hold N_INPUTS).
  Kokkos::Array<float, NN_NIN> nn_in_mean{};
  Kokkos::Array<float, NN_NIN> nn_in_std{};
  for (int c = 0; c < NN_NIN; ++c) {
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
          for (int c = 0; c < NN_NEOS; ++c) eos_dev(flat, c) = 0.f;
          for (int c = 0; c < NN_NIN;  ++c) x_full_dev(flat, c) = 0.f;
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

  // ── 3. Readout: metric reconstruction + M1 moments + NN→code conversion ──────
  // Intermediate buffer storing undensitized fluid-frame energy and number
  // densities so later kernels do not need to redo the metric/closure work.
  // Layout: m1_moments(flat, 0..3) = J[s]/volform,
  //         m1_moments(flat, 4..7) = rnnu[s]/volform
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
          for (int c = 0; c < 9; ++c) m1_moments(flat, c) = 0.0;
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
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
          calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
          apply_floor(g_uu, J[nuidx], H_d, m1_params_);
          Real Gamma =
              compute_Gamma(w_lorentz, v_u, J[nuidx], E, F_d, m1_params_);
          rnnu[nuidx] =
              u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma;
        }

        // Store undensitized moments for the exact-1D and Kirchhoff kernels.
        for (int s = 0; s < nspecies_; ++s) {
          m1_moments(flat, s)     = J[s] / volform;
          m1_moments(flat, 4 + s) = rnnu[s] / volform;
        }
        // Per-cell diffusion step for the partial-equilibrium predictor
        // (a FULL step; W = Lorentz factor).  Read back in the Kirchhoff kernel.
        m1_moments(flat, 8) = dt_full_ * adm.alpha(m, k, j, i) / w_lorentz;

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
              nudens_1[s] = m1_moments(flat, s);
              nudens_0[s] = m1_moments(flat, 4 + s);
              chi_loc[s]  = chi_(m, s, k, j, i);
            }

            Real eta_0_loc[4]{}, eta_1_loc[4]{};
            Real abs_0_loc[4]{}, abs_1_loc[4]{};
            Real scat_0_loc[4]{}, scat_1_loc[4]{};
            // Non-thermal outputs required by the signature but unused here:
            // beta+iso have no NEPS component, so these stay zero. There is no
            // eta_0_non_th output because NEPS conserves neutrino number.
            Real eta_1_non_th[4]{}, abs_1_non_th[4]{};
            Real abs_0_non_th[4]{};
            // Renamed from bns_nurates() by the largesim-m1 merge; now returns
            // an int fallback flag (eq-distribution fallback when reconstructed
            // T_nu > bound). The NN path doesn't track fallback stats, so the
            // return is intentionally ignored; the fallback + 1/H2 fix apply to
            // this 1D pair/brem/inelastic call automatically.
            (void)ComputeNuratesOpacities(nb, T, yp, yn, mu_n, mu_p, mu_e,
                        nudens_0, nudens_1, chi_loc,
                        eta_0_loc, eta_1_loc,
                        abs_0_loc, abs_1_loc,
                        scat_0_loc, scat_1_loc,
                        eta_1_non_th, abs_1_non_th,
                        abs_0_non_th,
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

  // ── 4. Kirchhoff / corr_fac / partial-equilibrium (T*,Ye*) / NeutrinoDens ─────
  // Reads raw opacities and undensitized radiation moments, then applies the
  // non-LTE correction, the partial-equilibrium predictor, and Kirchhoff's law
  // in-place.  Ported verbatim (adapted to the NN cached-value layout) from the
  // partial-equilibrium route in radiation_m1_calc_opacities_nurates.cpp.
  const bool peq_on_       = nurates_params_.use_partial_equilibrium;
  const Real peq_cv_eps_   = 1.0e-2;   // c_v secant half-width (matches nurates)
  const Real peq_trust_c_  = 10.0;     // trust-region multiple of the tier-1 bound
  const int  peq_max_halvings_ = 4;    // weight halvings on a rejected root
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

        // Restore undensitized radiation moments from the intermediate buffer.
        Real nudens_0[4]{}, nudens_1[4]{};
        for (int s = 0; s < nspecies_; ++s) {
          nudens_1[s] = m1_moments(flat, s);
          nudens_0[s] = m1_moments(flat, 4 + s);
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

        // ── local blackbody + non-LTE corr + (T*,Ye*) predictor + Kirchhoff ──
        // Undensitized M1 moments nudens_0/nudens_1 and the per-cell dtau come
        // from the readout kernel; opacities are NN(2D) + exact-1D(beta+iso).
        Real nudens_0_thin[4]{}, nudens_1_thin[4]{},
             nudens_0_peq[4]{},  nudens_1_peq[4]{};
        // Thermal absorption after the non-LTE correction (Kirchhoff multiplies
        // it) and the correction factor itself.  Default to a no-op if skipped.
        Real abs_0_th[4]{}, abs_1_th[4]{};
        Real corr_ae[4] = {1.0, 1.0, 1.0, 1.0};

        if (nurates_params_.use_kirchhoff_law ||
            nurates_params_.use_equilibrium_distribution) {
          // local blackbody at (T^n, Ye^n)
          NeutrinoDens(mu_n, mu_p, mu_e, T, nudens_0_thin[0], nudens_0_thin[1],
                       nudens_0_thin[2], nudens_1_thin[0], nudens_1_thin[1],
                       nudens_1_thin[2], nurates_params_, code_units, eos_units,
                       nurates_units);
          nudens_0_thin[2] *= 0.5;
          nudens_1_thin[2] *= 0.5;
          nudens_0_thin[3] = nudens_0_thin[2];
          nudens_1_thin[3] = nudens_1_thin[2];

          // Non-LTE correction (kappa ~ E_nu^2) from the LOCAL blackbody, not the
          // equilibrium the predictor settles on (the weights below are built
          // from kappa_abs, so the other choice would make kappa depend on the
          // weights that depend on kappa).  NEPS excluded from the number channel
          // (scattering conserves number); kept in the energy channel.
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            Real corr_fac = 1.0;
            if (nurates_params_.use_equilibrium_distribution) {
              corr_fac = (nudens_1[nuidx] / nudens_0[nuidx]) *
                         (nudens_0_thin[nuidx] / nudens_1_thin[nuidx]);
              if (!Kokkos::isfinite(corr_fac)) corr_fac = 1.0;
              corr_fac *= corr_fac;
              corr_fac = Kokkos::fmax(
                  1.0 / nurates_params_.opacity_corr_fac_max,
                  Kokkos::fmin(corr_fac, nurates_params_.opacity_corr_fac_max));
            }
            corr_ae[nuidx] = (nuidx == 0 || nuidx == 1) ? corr_fac : 1.0;
            scat_1_loc[nuidx] *= corr_fac;

            Real abs_0_non_th = non_th_buf(flat, nuidx);
            Real abs_1_non_th = non_th_buf(flat, 4 + nuidx);
            abs_0_th[nuidx] = Kokkos::fmax(
                abs_0_loc[nuidx] - abs_0_non_th, 0.0) * corr_ae[nuidx];
            abs_1_th[nuidx] = Kokkos::fmax(
                abs_1_loc[nuidx] - abs_1_non_th, 0.0) * corr_ae[nuidx];
            abs_0_loc[nuidx] = abs_0_th[nuidx];                     // thermal only
            abs_1_loc[nuidx] = abs_1_th[nuidx] + abs_1_non_th;      // thermal+NEPS
          }

          // Partially-equilibrated (T*, Ye*) predictor: a one-parameter family in
          // w = a/(1+a), a = dtau*kappa_abs, per channel (nu_e energy, anue
          // energy, heavy-pair energy, nu_e number, anue number).  w=1 -> trapped
          // weak equilibrium, w=0 -> local blackbody, continuous between.
          if (peq_on_) {
            const Real dtau = m1_moments(flat, 8);  // dt * alpha / W (full step)

            Real J_x  = nudens_1[2];
            Real kJ_x = abs_1_loc[2] * nudens_1[2];
            Real ks_x = abs_1_loc[2];
            int  n_x  = 1;
            if (nspecies_ > 3) {
              J_x  += nudens_1[3];
              kJ_x += abs_1_loc[3] * nudens_1[3];
              ks_x += abs_1_loc[3];
              n_x   = 2;
            }
            const Real kbar_1x = (J_x > 0.0) ? kJ_x/J_x : ks_x/n_x;

            // Split electron-pair weights (one per species).  Lumping nu_e and
            // anue under a common weight is exact only where the two paired
            // terms are equal, but they differ by exp(eta) with the lepton
            // residual their difference, so the errors reinforce.  The heavy
            // pair shares one J-weighted weight.  Mirrors the standard nurates
            // route (largesim-m1 split-weights commit).
            const Real a_1p = dtau*abs_1_loc[0];
            const Real a_1m = dtau*abs_1_loc[1];
            const Real a_1x = dtau*kbar_1x;
            const Real a_0p = dtau*abs_0_loc[0];
            const Real a_0m = dtau*abs_0_loc[1];
            const Real w_1p = a_1p/(1.0 + a_1p);
            const Real w_1m = a_1m/(1.0 + a_1m);
            const Real w_1x = a_1x/(1.0 + a_1x);
            const Real w_0p = a_0p/(1.0 + a_0p);
            const Real w_0m = a_0m/(1.0 + a_0m);

            Real T_star = T;
            Real Ye_star = Y;

            // Tier-0 gate: no EOS calls.  Ternaries (not fmax) so a NaN weight
            // gates the cell out deliberately.
            const bool w_finite = Kokkos::isfinite(w_1p) &&
                                  Kokkos::isfinite(w_1m) &&
                                  Kokkos::isfinite(w_1x) &&
                                  Kokkos::isfinite(w_0p) &&
                                  Kokkos::isfinite(w_0m);
            Real w_max = (w_1p > w_1m) ? w_1p : w_1m;
            w_max = (w_max > w_1x) ? w_max : w_1x;
            w_max = (w_max > w_0p) ? w_max : w_0p;
            w_max = (w_max > w_0m) ? w_max : w_0m;

            if (w_finite && w_max >= nurates_params_.peq_w_floor) {
              Real Y_part[3] = {Y, 0.0, 0.0};

              // Tier-1 gate: first-order bound on the excursion, from the
              // blackbody in hand and c_v.  T clamped to the table range.
              const Real T_tab_min = eos.GetMinimumTemperature()*
                                     eos_units.TemperatureConversion(code_units);
              const Real T_tab_max = eos.GetMaximumTemperature()*
                                     eos_units.TemperatureConversion(code_units);
              Real T_lo = T*(1.0 - peq_cv_eps_);
              Real T_hi = T*(1.0 + peq_cv_eps_);
              T_lo = (T_lo > T_tab_min) ? T_lo : T_tab_min;
              T_hi = (T_hi < T_tab_max) ? T_hi : T_tab_max;
              const Real cv = (T_hi > T_lo)
                  ? (eos.GetEnergy(nb, T_hi, Y_part) -
                     eos.GetEnergy(nb, T_lo, Y_part))/(T_hi - T_lo)
                  : 0.0;
              const bool cv_ok = Kokkos::isfinite(cv) && cv > 0.0;

              Real J_x_eq = nudens_1_thin[2];
              if (nspecies_ > 3) {
                J_x_eq += nudens_1_thin[3];
              }

              // Sum of absolute per-species terms (not |net|): the gate must
              // not under-estimate the move, so it cannot gate out a cell that
              // would have moved.  Split electron pair, matching nurates.
              const Real dlnT_hat =
                  cv_ok ? (w_1p*Kokkos::fabs(nudens_1_thin[0] - nudens_1[0]) +
                           w_1m*Kokkos::fabs(nudens_1_thin[1] - nudens_1[1]) +
                           w_1x*Kokkos::fabs(J_x_eq - J_x))/(T*cv)
                        : 0.0;
              const Real dYe_hat =
                  (w_0p*Kokkos::fabs(nudens_0_thin[0] - nudens_0[0]) +
                   w_0m*Kokkos::fabs(nudens_0_thin[1] - nudens_0[1]))/nb;

              if (cv_ok && !(dlnT_hat < nurates_params_.peq_dlnT_tol &&
                             dYe_hat < nurates_params_.peq_dYe_tol)) {
                // Trust region: a root far outside the linear bound is a
                // converged-but-wrong root.  Retry with halved weights on reject.
                const Real dlnT_trust = peq_trust_c_*dlnT_hat;
                const Real dYe_trust  = peq_trust_c_*dYe_hat;
                const Real dlnT_max =
                    (dlnT_trust > nurates_params_.peq_dlnT_tol)
                        ? dlnT_trust : nurates_params_.peq_dlnT_tol;
                const Real dYe_max =
                    (dYe_trust > nurates_params_.peq_dYe_tol)
                        ? dYe_trust : nurates_params_.peq_dYe_tol;

                const Real e_mat = eos.GetEnergy(nb, T, Y_part);

                Real f_soft = 1.0;
                for (int n_soft = 0; n_soft <= peq_max_halvings_;
                     ++n_soft, f_soft *= 0.5) {
                  const Real u[PEQ_NWEIGHTS] = {
                      f_soft*w_1p, f_soft*w_1m, f_soft*w_1x,
                      f_soft*w_0p, f_soft*w_0m};

                  const Real e_rhs = e_mat + u[PEQ_W1_NUE]*nudens_1[0] +
                                     u[PEQ_W1_ANUE]*nudens_1[1] + u[PEQ_W1_X]*J_x;
                  Real Yl_rhs[3] = {Y + (u[PEQ_W0_NUE]*nudens_0[0] -
                                         u[PEQ_W0_ANUE]*nudens_0[1])/nb, 0.0, 0.0};

                  Real T_try = T;
                  Real Ye_try[3] = {Y, 0.0, 0.0};
                  bool ok = eos.GetBetaEquilibriumPartial(
                      nb, e_rhs, Yl_rhs, u, T_try, &Ye_try[0], T, Y_part);

                  if (ok && Kokkos::fabs(Kokkos::log(T_try/T)) <= dlnT_max &&
                      Kokkos::fabs(Ye_try[0] - Y) <= dYe_max) {
                    T_star = T_try;
                    Ye_star = Ye_try[0];
                    break;
                  }
                }
              }
            }

            // Equilibrium the cell radiates towards.  Evaluated unconditionally:
            // a gated/unusable cell has (T*,Ye*) = (T,Y), reproducing the thin
            // limit, so the w -> 0 case needs no special path.
            Real Ye_arr[3] = {Ye_star, 0.0, 0.0};
            Real mu_b_s  = eos.GetBaryonChemicalPotential(nb, T_star, Ye_arr);
            Real mu_q_s  = eos.GetChargeChemicalPotential(nb, T_star, Ye_arr);
            Real mu_le_s = eos.GetElectronLeptonChemicalPotential(nb, T_star,
                                                                  Ye_arr);
            NeutrinoDens(mu_b_s, mu_b_s + mu_q_s, mu_le_s - mu_q_s, T_star,
                         nudens_0_peq[0], nudens_0_peq[1], nudens_0_peq[2],
                         nudens_1_peq[0], nudens_1_peq[1], nudens_1_peq[2],
                         nurates_params_, code_units, eos_units, nurates_units);
            nudens_0_peq[2] *= 0.5;
            nudens_1_peq[2] *= 0.5;
            nudens_0_peq[3] = nudens_0_peq[2];
            nudens_1_peq[3] = nudens_1_peq[2];

            // Finiteness screen: a NaN here would ride abs_*_th * my_nudens into
            // the source term.  Fall back to the local blackbody (w -> 0).
            bool peq_finite = true;
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              peq_finite = peq_finite &&
                           Kokkos::isfinite(nudens_0_peq[nuidx]) &&
                           Kokkos::isfinite(nudens_1_peq[nuidx]);
            }
            if (!peq_finite) {
              for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
                nudens_0_peq[nuidx] = nudens_0_thin[nuidx];
                nudens_1_peq[nuidx] = nudens_1_thin[nuidx];
              }
            }
          }
        }

        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          // store corrected opacities + default (bns_nurates) emissivities
          scat_1_(m, nuidx, k, j, i) = scat_1_loc[nuidx];
          abs_0_(m, nuidx, k, j, i)  = abs_0_loc[nuidx];
          abs_1_(m, nuidx, k, j, i)  = abs_1_loc[nuidx];
          eta_0_(m, nuidx, k, j, i)  = eta_0_loc[nuidx];
          eta_1_(m, nuidx, k, j, i)  = eta_1_loc[nuidx];

          Real my_nudens_0{}, my_nudens_1{};
          if (nurates_params_.use_kirchhoff_law ||
              nurates_params_.use_equilibrium_distribution) {
            if (peq_on_) {
              my_nudens_0 = nudens_0_peq[nuidx];
              my_nudens_1 = nudens_1_peq[nuidx];
            } else {
              my_nudens_0 = nudens_0_thin[nuidx];
              my_nudens_1 = nudens_1_thin[nuidx];
            }
          }

          // Limit NEPS in-scattering by the actual M1 occupation (only suppresses
          // over-emission in the decoupling region; never amplifies).
          Real eta_1_non_th = non_th_buf(flat, 12 + nuidx);
          if (nurates_params_.use_equilibrium_distribution) {
            Real f_occ_0 = (my_nudens_0 > 0.0) ? nudens_0[nuidx] / my_nudens_0
                                               : 1.0;
            if (!Kokkos::isfinite(f_occ_0)) f_occ_0 = 1.0;
            f_occ_0 = Kokkos::fmin(f_occ_0, 1.0);
            eta_1_non_th *= f_occ_0;
          }

          // Kirchhoff derives the emissivities from the corrected THERMAL
          // opacity and the equilibrium distribution; NEPS energy emission is
          // added back afterwards.  NUMBER excludes NEPS (scattering conserves
          // number).  Without Kirchhoff, bns_nurates' own emissivities stand,
          // scaled by corr_ae like the opacities.
          if (nurates_params_.use_kirchhoff_law) {
            eta_0_(m, nuidx, k, j, i) = (abs_0_th[nuidx] > 0)
                ? abs_0_th[nuidx] * my_nudens_0
                : eta_0_loc[nuidx];
            eta_1_(m, nuidx, k, j, i) = (abs_1_th[nuidx] > 0)
                ? abs_1_th[nuidx] * my_nudens_1 + eta_1_non_th
                : eta_1_loc[nuidx];
          } else {
            eta_0_(m, nuidx, k, j, i) = eta_0_loc[nuidx] * corr_ae[nuidx];
            eta_1_(m, nuidx, k, j, i) = eta_1_loc[nuidx] * corr_ae[nuidx];
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
