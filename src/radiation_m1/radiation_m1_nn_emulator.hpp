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

#include <memory>
#include <string>

namespace radiationm1 {

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

  bool IsLoaded() const;
  bool IsGPU() const;
  int DeviceIndex() const;

  const float *HostInMean() const;
  const float *HostInStd() const;
  const float *HostOutMean() const;
  const float *HostOutStd() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
#endif  // RADIATION_M1_NN_EMULATOR_HPP
