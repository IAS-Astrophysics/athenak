#ifndef RADIATION_M1_NN_FORWARD_HPP
#define RADIATION_M1_NN_FORWARD_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_nn_forward.hpp
//! \brief Device-side (Kokkos) forward pass for the grey-M1 opacity NN.
//!
//! This header is intentionally free of LibTorch: it holds the network weights as
//! plain device Views (NNWeights) and evaluates the SymMLP forward as an inline
//! device function (nn_forward), so the emulator can run inside a single fused
//! Kokkos kernel with no second allocator, no second CUDA stream, and no per-step
//! host<->device transfers.  It mirrors the CompOSE EOS table pattern
//! (eos_compose.hpp): persistent device arrays evaluated inline per cell.
//!
//! It reproduces exactly the architecture trained in ml_opacity/hybrid_2d_nn
//! (model.py:SymMLP, species_heads=True):
//!   embed : Linear(8->H) + GELU
//!   blocks: n_blocks x [ LayerNorm(H) -> Linear(H,H) -> GELU -> Linear(H,H) ] + residual
//!   heads : Linear(H,8) | Linear(H,8) | Linear(H,12)  -> 28 normalised outputs
//!   map   : 28 -> 32 (anux pair/brem copied from nux; OUT_MAP_28_TO_32)
//! Inputs are the already-normalised 8 EOS features (same buffer fed to the torch
//! path); outputs are the 32 normalised log-opacities (denorm happens downstream).

#if ENABLE_NN_OPACITY

#include "athena.hpp"        // DvceArray1D/2D/3D
#include <Kokkos_Core.hpp>

namespace radiationm1 {

//! Hidden width of the trained model.  MUST match ml_opacity model.py (hidden=256).
static constexpr int NN_FWD_HIDDEN = 256;
static constexpr int NN_FWD_IN     = 8;
static constexpr int NN_FWD_OUT28  = 28;
static constexpr int NN_FWD_OUT32  = 32;

//----------------------------------------------------------------------------------------
//! \struct NNWeights
//! \brief Device-resident network weights (filled once at init from the .pt).
//!
//! Shapes follow PyTorch nn.Linear (weight is [out,in]):
struct NNWeights {
  DvceArray2D<float> W_embed;   // (H, 8)
  DvceArray1D<float> b_embed;   // (H)
  DvceArray2D<float> ln_g;      // (n_blocks, H)  LayerNorm weight (gamma)
  DvceArray2D<float> ln_b;      // (n_blocks, H)  LayerNorm bias   (beta)
  DvceArray3D<float> W1;        // (n_blocks, H, H)
  DvceArray2D<float> b1;        // (n_blocks, H)
  DvceArray3D<float> W2;        // (n_blocks, H, H)
  DvceArray2D<float> b2;        // (n_blocks, H)
  DvceArray2D<float> W_nue;     // (8, H)
  DvceArray1D<float> b_nue;     // (8)
  DvceArray2D<float> W_anue;    // (8, H)
  DvceArray1D<float> b_anue;    // (8)
  DvceArray2D<float> W_nux;     // (12, H)  nux[0-7] + anux NEPS[0-3]
  DvceArray1D<float> b_nux;     // (12)
  int n_blocks = 0;
};

//! Exact (erf-based) GELU, matching torch nn.GELU(approximate='none').
KOKKOS_INLINE_FUNCTION
float nn_gelu(float x) {
  return 0.5f * x * (1.0f + Kokkos::erf(x * 0.70710678118654752f));
}

//----------------------------------------------------------------------------------------
//! \fn nn_forward
//! \brief Evaluate the network for one cell.  xin: 8 normalised inputs;
//!        out32: 32 normalised outputs (same layout as the torch path's nn_view).
//!
//! Naive per-thread implementation (three H-length local buffers).  Correct and
//! self-contained; if raw MLP compute becomes the bottleneck, replace with a
//! team/shared-memory tiled GEMM (see notes in the PR).  eps and biased variance
//! match torch LayerNorm defaults.
KOKKOS_INLINE_FUNCTION
void nn_forward(const NNWeights &w, const float xin[NN_FWD_IN],
                float out32[NN_FWD_OUT32]) {
  constexpr int H = NN_FWD_HIDDEN;
  constexpr float eps = 1e-5f;

  float z[H];
  float a[H];
  float h[H];

  // embed: z = GELU(W_embed x + b_embed)
  for (int o = 0; o < H; ++o) {
    float acc = w.b_embed(o);
    for (int k = 0; k < NN_FWD_IN; ++k) acc += w.W_embed(o, k) * xin[k];
    z[o] = nn_gelu(acc);
  }

  // residual blocks: z += Linear2(GELU(Linear1(LayerNorm(z))))
  for (int blk = 0; blk < w.n_blocks; ++blk) {
    // LayerNorm over the H features (biased variance, eps)
    float mean = 0.0f;
    for (int o = 0; o < H; ++o) mean += z[o];
    mean /= static_cast<float>(H);
    float var = 0.0f;
    for (int o = 0; o < H; ++o) {
      const float d = z[o] - mean;
      var += d * d;
    }
    var /= static_cast<float>(H);
    const float inv = 1.0f / Kokkos::sqrt(var + eps);
    for (int o = 0; o < H; ++o) {
      a[o] = (z[o] - mean) * inv * w.ln_g(blk, o) + w.ln_b(blk, o);
    }
    // Linear1 + GELU
    for (int o = 0; o < H; ++o) {
      float acc = w.b1(blk, o);
      for (int k = 0; k < H; ++k) acc += w.W1(blk, o, k) * a[k];
      h[o] = nn_gelu(acc);
    }
    // Linear2 + residual
    for (int o = 0; o < H; ++o) {
      float acc = w.b2(blk, o);
      for (int k = 0; k < H; ++k) acc += w.W2(blk, o, k) * h[k];
      z[o] += acc;
    }
  }

  // heads -> 28 normalised outputs
  float o28[NN_FWD_OUT28];
  for (int o = 0; o < 8; ++o) {
    float acc = w.b_nue(o);
    for (int k = 0; k < H; ++k) acc += w.W_nue(o, k) * z[k];
    o28[o] = acc;
  }
  for (int o = 0; o < 8; ++o) {
    float acc = w.b_anue(o);
    for (int k = 0; k < H; ++k) acc += w.W_anue(o, k) * z[k];
    o28[8 + o] = acc;
  }
  for (int o = 0; o < 12; ++o) {
    float acc = w.b_nux(o);
    for (int k = 0; k < H; ++k) acc += w.W_nux(o, k) * z[k];
    o28[16 + o] = acc;
  }

  // 28 -> 32 (OUT_MAP_28_TO_32): nue/anue/nux identity; anux pair/brem = nux.
  for (int i = 0; i < 24; ++i) out32[i] = o28[i];   // nue(0-7) anue(8-15) nux(16-23)
  out32[24] = o28[16];  out32[25] = o28[17];         // anux ch0,1 = nux ch0,1
  out32[26] = o28[24];  out32[27] = o28[25];         // anux ch2,3 = anux NEPS
  out32[28] = o28[20];  out32[29] = o28[21];         // anux ch4,5 = nux ch4,5
  out32[30] = o28[26];  out32[31] = o28[27];         // anux ch6,7 = anux NEPS
}

//----------------------------------------------------------------------------------------
//! \fn nn_forward_team
//! \brief Team (thread-block) evaluation of the network for ONE cell, with the
//!        H-length activations held in team scratch (shared memory) and the hidden
//!        units parallelised across the team.  This avoids the per-thread
//!        local-memory spills of nn_forward() and gives a tiled matmul, while
//!        staying pure Kokkos (no LibTorch → no per-call cudaMalloc/device sync →
//!        scales like nurates/eos_compose).
//!
//! z, a, h : team-scratch pointers, H floats each.
//! out32   : device pointer to this cell's 32 outputs (e.g. nn_view.data()+f*32).
template <class TeamMember>
KOKKOS_INLINE_FUNCTION
void nn_forward_team(const TeamMember &team, const NNWeights &w,
                     const float xin[NN_FWD_IN],
                     float *z, float *a, float *h, float *out32) {
  constexpr int H = NN_FWD_HIDDEN;
  constexpr float eps = 1e-5f;

  // embed: z = GELU(W_embed x + b_embed)
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, H), [&](const int o) {
    float acc = w.b_embed(o);
    for (int k = 0; k < NN_FWD_IN; ++k) acc += w.W_embed(o, k) * xin[k];
    z[o] = nn_gelu(acc);
  });
  team.team_barrier();

  for (int blk = 0; blk < w.n_blocks; ++blk) {
    // LayerNorm statistics over z (team reductions, broadcast to all threads)
    float mean = 0.0f;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, H),
        [&](const int o, float &s) { s += z[o]; }, mean);
    mean /= static_cast<float>(H);
    float var = 0.0f;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, H),
        [&](const int o, float &s) { const float d = z[o] - mean; s += d * d; },
        var);
    var /= static_cast<float>(H);
    const float inv = 1.0f / Kokkos::sqrt(var + eps);
    // a = LayerNorm(z)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, H), [&](const int o) {
      a[o] = (z[o] - mean) * inv * w.ln_g(blk, o) + w.ln_b(blk, o);
    });
    team.team_barrier();
    // h = GELU(Linear1(a))
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, H), [&](const int o) {
      float acc = w.b1(blk, o);
      for (int k = 0; k < H; ++k) acc += w.W1(blk, o, k) * a[k];
      h[o] = nn_gelu(acc);
    });
    team.team_barrier();
    // z += Linear2(h)   (residual)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, H), [&](const int o) {
      float acc = w.b2(blk, o);
      for (int k = 0; k < H; ++k) acc += w.W2(blk, o, k) * h[k];
      z[o] += acc;
    });
    team.team_barrier();
  }

  // heads -> 28 outputs, stored in a[0..27]
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NN_FWD_OUT28),
      [&](const int o) {
        float acc;
        if (o < 8) {
          acc = w.b_nue(o);
          for (int k = 0; k < H; ++k) acc += w.W_nue(o, k) * z[k];
        } else if (o < 16) {
          const int oo = o - 8;
          acc = w.b_anue(oo);
          for (int k = 0; k < H; ++k) acc += w.W_anue(oo, k) * z[k];
        } else {
          const int oo = o - 16;
          acc = w.b_nux(oo);
          for (int k = 0; k < H; ++k) acc += w.W_nux(oo, k) * z[k];
        }
        a[o] = acc;
      });
  team.team_barrier();

  // 28 -> 32 map (single thread): anux pair/brem copied from nux.
  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    for (int i = 0; i < 24; ++i) out32[i] = a[i];   // nue anue nux (identity)
    out32[24] = a[16];  out32[25] = a[17];           // anux ch0,1 = nux ch0,1
    out32[26] = a[24];  out32[27] = a[25];           // anux ch2,3 = anux NEPS
    out32[28] = a[20];  out32[29] = a[21];           // anux ch4,5 = nux ch4,5
    out32[30] = a[26];  out32[31] = a[27];           // anux ch6,7 = anux NEPS
  });
}

}  // namespace radiationm1

#endif  // ENABLE_NN_OPACITY
#endif  // RADIATION_M1_NN_FORWARD_HPP
