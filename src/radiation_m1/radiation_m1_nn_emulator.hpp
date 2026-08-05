#ifndef RADIATION_M1_NN_EMULATOR_HPP
#define RADIATION_M1_NN_EMULATOR_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nn_emulator.hpp
//! \brief Lightweight declaration for the LibTorch NN opacity emulator.

#if ENABLE_NN_OPACITY

#include <cstddef>
#include <memory>
#include <string>

#include "radiation_m1/radiation_m1_nn_forward.hpp"  // NNWeights (Kokkos, no LibTorch)

namespace radiationm1 {

// CUDA-event checkpoints used by the opt-in, asynchronously collected NN profiler.
// Keep the enum here (rather than exposing CUDA types) so radiation_m1.hpp stays light.
enum class NNProfilePoint : int {
  start = 0,
  gather,
  forward,
  readout,
  exact_1d,
  kirchhoff,
  count
};

//----------------------------------------------------------------------------------------
//! \class NNOpacityEmulator
//! \brief Batched LibTorch inference path for grey M1 neutrino opacities.
//!
//! Keep this header free of LibTorch includes.  radiation_m1.hpp is included by many
//! translation units, and pulling <torch/script.h> through it makes normal AthenaK
//! rebuilds extremely slow.  The Torch-heavy implementation lives in
//! radiation_m1_nn_emulator.cpp behind this pimpl.
class NNOpacityEmulator {
 public:
  static constexpr int N_EOS     = 8;    // EOS input features
  static constexpr int N_SPECIES = 4;    // nue, anue, nux, anux
  static constexpr int N_CH      = 8;    // channels per species
  static constexpr int N_INPUTS  = N_EOS;              // 8 (no one-hot)
  static constexpr int N_OUTPUTS = N_SPECIES * N_CH;   // 32 (4 species × 8 channels)
  // Channel layout per species [s*N_CH + ch]:
  //   0 eta_0_th   1 kappa_0_a_th   2 eta_0_non_th   3 kappa_0_a_non_th
  //   4 eta_th     5 kappa_a_th     6 eta_non_th     7 kappa_a_non_th
  static constexpr int HIDDEN    = 256;

  NNOpacityEmulator();
  ~NNOpacityEmulator();

  NNOpacityEmulator(const NNOpacityEmulator &) = delete;
  NNOpacityEmulator &operator=(const NNOpacityEmulator &) = delete;
  NNOpacityEmulator(NNOpacityEmulator &&) noexcept;
  NNOpacityEmulator &operator=(NNOpacityEmulator &&) noexcept;

  void Load(const std::string &model_path, const std::string &stats_dir,
            bool use_cuda = false);

  void Infer(const float *eos_data, float *out_data, int N) const;
  void InferPrebuilt(const float *x_full_ptr, float *nn_out_ptr, int N) const;
  void InferDevice(const float *eos_dev_ptr, float *nn_out_ptr, int N) const;

  // CUDA-graph path: capture one fixed-size forward and replay it in chunks.
  // This reduces forward dispatch overhead but imposes a 256k-row minimum-work
  // quantum. Same I/O as InferPrebuilt.
  void InferGraph(const float *x_full_ptr, float *nn_out_ptr, int N) const;

  // Direct-cuBLAS path: evaluate the network with cublasSgemm (TF32) for its
  // dense matmuls plus small Kokkos kernels for bias/GELU/LayerNorm/residual/map —
  // ALL issued on the Kokkos stream. This keeps cuBLAS GEMM throughput and removes
  // the LibTorch runtime from the steady-state forward. Requires ExtractWeights()
  // first. Same zero-copy I/O contract as InferPrebuilt (x_full: N×8, out: N×32).
  void InferCublas(const float *x_full_ptr, float *nn_out_ptr, int N) const;

  // ── Optional fused-kernel path ──────────────────────────────────────────────
  // Extract the trained weights from the loaded .pt into device-resident Views
  // (NNWeights) so the network can be evaluated inside a single Kokkos kernel
  // (see radiation_m1_nn_forward.hpp / nn_fused_kernel).  Call once, after Load().
  void ExtractWeights();
  bool WeightsReady() const;
  const NNWeights &GetWeights() const;

  bool IsLoaded() const;
  bool IsGPU() const;
  int DeviceIndex() const;

  const float *HostInMean() const;
  const float *HostInStd() const;
  const float *HostOutMean() const;
  const float *HostOutStd() const;

  // Opt-in sampled profiler.  Each hook is a predictable branch when disabled.
  // Samples use CUDA events on the Kokkos stream and are collected on
  // a later opacity call with cudaEventQuery(), never a fence/synchronize.  At the
  // reporting interval, min/mean/max phase times and allocator deltas are reduced
  // across ranks and printed by rank zero.
  void ConfigureProfiling(bool enabled, int interval, const std::string &mode);
  void ProfilePollAndReport() const;
  void ProfileBegin(int n_cells, int infer_cells, bool kokkos_scratch_grew,
                    size_t kokkos_scratch_bytes) const;
  void ProfileMark(NNProfilePoint point) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
#endif  // RADIATION_M1_NN_EMULATOR_HPP
