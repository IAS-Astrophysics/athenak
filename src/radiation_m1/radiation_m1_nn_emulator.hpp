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

namespace radiationm1 {

// Device-event checkpoints used by the opt-in, asynchronously collected NN profiler.
// Keep the enum here (rather than exposing cudaEvent_t / c10::xpu::XPUEvent) so
// radiation_m1.hpp stays light and backend-agnostic.
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
  static constexpr int N_EOS     = 8;    // EOS features gathered per cell
  static constexpr int N_SPECIES = 4;    // nue, anue, nux, anux
  // Channels per species. The reduced-output model drops the two NUMBER
  // non-thermal channels (NEPS conserves number → the M1 Kirchhoff step
  // discards them) and predicts all 4 species independently (no nux=anux copy):
  //   6 channels = eta_0_th, kappa_0_a_th, eta_th, kappa_a_th, eta_non_th,
  //                kappa_a_non_th   (number non-thermal absent)
#if NN_REDUCED_OUTPUT
  static constexpr int N_CH      = 6;
#else
  static constexpr int N_CH      = 8;
#endif
  // NN input width. The hybrid still gathers all N_EOS features (the 1D/Kirchhoff
  // reconstruction needs the chemical potentials), but the *network* input can be
  // reduced to (nb, T, Ye) since the other 5 are EOS-derived at fixed EOS.  The
  // reduced build expects a native 3-input deploy (3-dim nn2d_in_* stats + a
  // 3-input best_2d_nn.pt).  Default (flag off) keeps the 8-input model.
#if NN_REDUCED_INPUT
  static constexpr int N_INPUTS  = 3;                  // reduced: nb, T, Ye
#else
  static constexpr int N_INPUTS  = N_EOS;              // 8 (full EOS feature set)
#endif
  static constexpr int N_OUTPUTS = N_SPECIES * N_CH;   // 32 (4 species × 8 channels)
  // Channel layout per species [s*N_CH + ch]:
  //   0 eta_0_th   1 kappa_0_a_th   2 eta_0_non_th   3 kappa_0_a_non_th
  //   4 eta_th     5 kappa_a_th     6 eta_non_th     7 kappa_a_non_th
  NNOpacityEmulator();
  ~NNOpacityEmulator();

  NNOpacityEmulator(const NNOpacityEmulator &) = delete;
  NNOpacityEmulator &operator=(const NNOpacityEmulator &) = delete;
  NNOpacityEmulator(NNOpacityEmulator &&) noexcept;
  NNOpacityEmulator &operator=(NNOpacityEmulator &&) noexcept;

  // device_index MUST be the index Kokkos itself resolved to (Kokkos::device_id()),
  // never one computed independently from the MPI rank or the launcher environment:
  // AthenaK has no explicit rank-to-GPU binding anywhere in src/, so querying Kokkos
  // is the only way Torch and Kokkos are guaranteed to agree on the physical device.
  void Load(const std::string &model_path, const std::string &stats_dir,
            int device_index);

  // Device index the model was actually loaded onto (-1 before Load).  Exposed so the
  // Kokkos-side caller can assert it still matches Kokkos::device_id() before handing
  // over a stream: nothing else checks that the Torch device and the borrowed Kokkos
  // queue refer to the same physical tile, and a mismatch would submit work to one tile
  // while reading memory resident on another.
  int DeviceIndex() const;

  // Provide the device queue/stream the forward and profiler events run on.  Passed
  // as an opaque void* so this header — and the ~32 TUs that include
  // radiation_m1.hpp — stay free of Kokkos/CUDA/SYCL headers.  What the caller
  // (a Kokkos TU) must pass differs per backend, because the two runtimes' handles
  // differ in kind, not just in type:
  //   CUDA: (void*)DevExeSpace().cuda_stream()   — cudaStream_t IS a pointer, passed
  //                                                by value
  //   SYCL: (void*)&DevExeSpace().sycl_queue()   — a POINTER TO the Kokkos-owned
  //                                                sycl::queue, which must outlive
  //                                                every InferPrebuilt call (it does:
  //                                                Kokkos owns it for the run)
  // Must be called before InferPrebuilt / ProfileBegin each step.
  void SetStream(void *stream) const;

  void InferPrebuilt(const float *x_full_ptr, float *nn_out_ptr, int N) const;

  const float *HostInMean() const;
  const float *HostInStd() const;
  const float *HostOutMean() const;
  const float *HostOutStd() const;

  // Opt-in sampled profiler.  Each hook is a predictable branch when disabled.
  // Samples use device events recorded on the Kokkos stream/queue and are collected
  // on a later opacity call with a non-blocking query, never a fence/synchronize.
  // At the reporting interval, min/mean/max phase times and allocator deltas are
  // reduced across ranks and printed by rank zero.
  // ConfigureProfiling() silently declines (with a rank-0 warning) on a backend or
  // device that cannot time events — see NN_PROFILE support notes in the .cpp.
  void ConfigureProfiling(bool enabled, int interval);
  void ProfilePollAndReport() const;
  void ProfileBegin(int n_cells, bool kokkos_scratch_grew,
                    size_t kokkos_scratch_bytes) const;
  void ProfileMark(NNProfilePoint point) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
#endif  // RADIATION_M1_NN_EMULATOR_HPP
