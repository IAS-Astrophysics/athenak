#ifndef RHINE_RHINE_NET_HPP_
#define RHINE_RHINE_NET_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rhine_net.hpp
//! \brief Device port of the header-only RHINE neural-network r-process heating model.
//!
//! The original RHINE (https://git.gsi.de/nucastro_public/rhine) is a host-only,
//! zero-allocation C++ translation of the RHINE Fortran modules.  Here it is split into:
//!   * HostNeuralNet  - host-only parser for one model_N.txt file (uses std::ifstream).
//!   * RhineNets      - the 16 networks' weights stored in Kokkos DvceArrays, with
//!                      KOKKOS_INLINE_FUNCTION evaluation callable inside par_for kernels.
//!
//! Usage:
//!   RhineNets nets;                 // member of the RHINE class
//!   nets.InitFromFiles(models_dir); // host: parse the 16 files, deep_copy to device
//!   ...
//!   nets.run(rho, tmev, ye, yn, ya, yh, ah, drho, dt,
//!            ye0, yn0, ya0, yh0, ah0, mass0,
//!            dye, dyn, dyp, dya, dyh, dah, dma, fnu, pmode);   // device

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace rhine {

// ---------------------------------------------------------------------------
// Compile-time bounds (largest layer/input sizes across the 16 RHINE networks)
// ---------------------------------------------------------------------------
constexpr int  MAX_DIN  = 9;
constexpr int  MAX_H    = 128;   // covers both hidden layers h1 and h2
constexpr int  NUM_NETS = 16;

constexpr Real L10       = 2.302585092994046;   // ln(10)
constexpr Real EXP_CLAMP = 511.5;               // Fortran expo clamp (=5*0.1*1023)

// Physical/regime constants (see original RHINE::Model)
constexpr Real T9NSE = 7.0;
constexpr Real T9QSE = 5.0;
constexpr Real T9YA  = 3.0;
constexpr Real T9LOW = 0.1;
constexpr Real T9MEV = 11.604442475978844;
constexpr Real LT9QSE = 0.6989700043360187;     // log10(T9QSE) = log10(5)
constexpr Real LT9YA  = 0.4771212547196624;     // log10(T9YA)  = log10(3)

// ---------------------------------------------------------------------------
// Network ordering: model_1.txt ... model_16.txt map to these indices.
// ---------------------------------------------------------------------------
enum NetIndex {
  N_MA = 0,     // model_1  : average mass excess
  N_YN_QSE,     // model_2
  N_YP_QSE,     // model_3
  N_YA_QSE,     // model_4
  N_YH_QSE,     // model_5
  N_AH_QSE,     // model_6
  N_YC_QSE,     // model_7
  N_FNU,        // model_8  : fraction of neutrino losses
  N_DYE,        // model_9
  N_DYN,        // model_10
  N_DYHS,       // model_11
  N_DYHP,       // model_12
  N_DAHZ,       // model_13
  N_DAHS,       // model_14
  N_DAHP,       // model_15
  N_DAHN        // model_16
};

// ---------------------------------------------------------------------------
// Device scalar math (activation functions and helpers)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION Real safeExp(Real x) {
  x = x < -EXP_CLAMP ? -EXP_CLAMP : (x > EXP_CLAMP ? EXP_CLAMP : x);
  return Kokkos::exp(x);
}
KOKKOS_INLINE_FUNCTION Real elu1(Real x)     { return x >= 0.0 ? x : safeExp(x) - 1.0; }
KOKKOS_INLINE_FUNCTION Real sigmoid1(Real x) { return 1.0 / (1.0 + safeExp(-x)); }
KOKKOS_INLINE_FUNCTION Real exp10(Real x)    { return Kokkos::exp(x * L10); }

KOKKOS_INLINE_FUNCTION Real postprocess_para_neg1(Real y, const Real* x) {
  return y     * exp10(x[5])
       + 8.071 * exp10(x[2])
       + 7.289 * exp10(x[3])
       + 2.425 * exp10(x[4]);
}

// ---------------------------------------------------------------------------
// HostNeuralNet - host-only parser for one model_N.txt file.
// Weights are stored tightly (stride d_in / h1); transferred to padded device
// Views by RhineNets::InitFromFiles().
// ---------------------------------------------------------------------------
struct HostNeuralNet {
  int para{0};
  int d_in{0}, h1{0}, h2{0};             // d_out is always 1

  double sa[MAX_DIN], sb[MAX_DIN];       // affine input scaling
  double y_ra, y_rb;                     // output rescaling
  double W1[MAX_H * MAX_DIN];
  double W2[MAX_H * MAX_H];
  double W3[MAX_H];
  double b1[MAX_H];
  double b2[MAX_H];
  double b3;

  void load(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
      throw std::runtime_error("RHINE: cannot open " + filename);

    auto skipLine = [&]() {
      f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    };

    int dout_f;
    f >> para >> d_in >> h1 >> h2 >> dout_f;
    if (d_in > MAX_DIN || h1 > MAX_H || h2 > MAX_H)
      throw std::runtime_error(
          "RHINE: " + filename + " - layer sizes exceed MAX_DIN/MAX_H. "
          "Increase the compile-time constants in rhine_net.hpp.");

    double xmax[MAX_DIN], xmin[MAX_DIN];
    double ymax, ymin;
    for (int j = 0; j < d_in; ++j) f >> xmax[j];
    for (int j = 0; j < d_in; ++j) f >> xmin[j];
    f >> ymax;
    f >> ymin;

    for (int j = 0; j < d_in; ++j) {
      double inv = 2.0 / (xmax[j] - xmin[j]);
      sa[j] = inv;
      sb[j] = -1.0 - xmin[j] * inv;
    }
    y_ra = (ymax - ymin) * 0.5;
    y_rb =  ymin + y_ra;

    skipLine();
    for (int k = 0; k < h1; ++k)
      for (int j = 0; j < d_in; ++j)
        f >> W1[k * d_in + j];

    skipLine();
    for (int k = 0; k < h2; ++k)
      for (int j = 0; j < h1; ++j)
        f >> W2[k * h1 + j];

    skipLine();
    for (int j = 0; j < h2; ++j) f >> W3[j];

    skipLine();
    for (int j = 0; j < h1; ++j) f >> b1[j];

    skipLine();
    for (int j = 0; j < h2; ++j) f >> b2[j];

    skipLine();
    f >> b3;

    if (f.fail())
      throw std::runtime_error("RHINE: read error in " + filename);
  }
};

// ---------------------------------------------------------------------------
// RhineNets - device-resident networks + evaluation.
//
// All 16 nets are padded to MAX_H/MAX_DIN and stored contiguously so a single
// struct (captured by value into a par_for lambda) carries every weight to the
// device via shallow View copies.
// ---------------------------------------------------------------------------
struct RhineNets {
  // Per-net metadata
  DvceArray1D<int>  para, din, h1, h2;                    // (NUM_NETS)
  // Scaling / rescaling
  DvceArray2D<Real> sa, sb;                               // (NUM_NETS, MAX_DIN)
  DvceArray1D<Real> yra, yrb, b3;                         // (NUM_NETS)
  // Weights and biases (padded)
  DvceArray3D<Real> W1;                                   // (NUM_NETS, MAX_H, MAX_DIN)
  DvceArray3D<Real> W2;                                   // (NUM_NETS, MAX_H, MAX_H)
  DvceArray2D<Real> W3, b1, b2;                           // (NUM_NETS, MAX_H)

  // -----------------------------------------------------------------------
  // Host: parse the 16 files under `path` and deep-copy weights to device.
  // -----------------------------------------------------------------------
  void InitFromFiles(const std::string& path) {
    Kokkos::realloc(para, NUM_NETS);
    Kokkos::realloc(din,  NUM_NETS);
    Kokkos::realloc(h1,   NUM_NETS);
    Kokkos::realloc(h2,   NUM_NETS);
    Kokkos::realloc(sa,   NUM_NETS, MAX_DIN);
    Kokkos::realloc(sb,   NUM_NETS, MAX_DIN);
    Kokkos::realloc(yra,  NUM_NETS);
    Kokkos::realloc(yrb,  NUM_NETS);
    Kokkos::realloc(b3,   NUM_NETS);
    Kokkos::realloc(W1,   NUM_NETS, MAX_H, MAX_DIN);
    Kokkos::realloc(W2,   NUM_NETS, MAX_H, MAX_H);
    Kokkos::realloc(W3,   NUM_NETS, MAX_H);
    Kokkos::realloc(b1,   NUM_NETS, MAX_H);
    Kokkos::realloc(b2,   NUM_NETS, MAX_H);

    auto h_para = Kokkos::create_mirror_view(para);
    auto h_din  = Kokkos::create_mirror_view(din);
    auto h_h1   = Kokkos::create_mirror_view(h1);
    auto h_h2   = Kokkos::create_mirror_view(h2);
    auto h_sa   = Kokkos::create_mirror_view(sa);
    auto h_sb   = Kokkos::create_mirror_view(sb);
    auto h_yra  = Kokkos::create_mirror_view(yra);
    auto h_yrb  = Kokkos::create_mirror_view(yrb);
    auto h_b3   = Kokkos::create_mirror_view(b3);
    auto h_W1   = Kokkos::create_mirror_view(W1);
    auto h_W2   = Kokkos::create_mirror_view(W2);
    auto h_W3   = Kokkos::create_mirror_view(W3);
    auto h_b1   = Kokkos::create_mirror_view(b1);
    auto h_b2   = Kokkos::create_mirror_view(b2);

    // Zero the padded regions so device memory is deterministic (never read,
    // but keeps deep_copy from shipping uninitialised host memory).
    Kokkos::deep_copy(h_sa, 0.0); Kokkos::deep_copy(h_sb, 0.0);
    Kokkos::deep_copy(h_W1, 0.0); Kokkos::deep_copy(h_W2, 0.0);
    Kokkos::deep_copy(h_W3, 0.0); Kokkos::deep_copy(h_b1, 0.0);
    Kokkos::deep_copy(h_b2, 0.0);

    for (int n = 0; n < NUM_NETS; ++n) {
      HostNeuralNet net;
      net.load(path + "/model_" + std::to_string(n + 1) + ".txt");

      h_para(n) = net.para;
      h_din(n)  = net.d_in;
      h_h1(n)   = net.h1;
      h_h2(n)   = net.h2;
      h_yra(n)  = static_cast<Real>(net.y_ra);
      h_yrb(n)  = static_cast<Real>(net.y_rb);
      h_b3(n)   = static_cast<Real>(net.b3);

      for (int jj = 0; jj < net.d_in; ++jj) {
        h_sa(n, jj) = static_cast<Real>(net.sa[jj]);
        h_sb(n, jj) = static_cast<Real>(net.sb[jj]);
      }
      for (int kk = 0; kk < net.h1; ++kk) {
        h_b1(n, kk) = static_cast<Real>(net.b1[kk]);
        for (int jj = 0; jj < net.d_in; ++jj)
          h_W1(n, kk, jj) = static_cast<Real>(net.W1[kk * net.d_in + jj]);
      }
      for (int kk = 0; kk < net.h2; ++kk) {
        h_b2(n, kk) = static_cast<Real>(net.b2[kk]);
        for (int jj = 0; jj < net.h1; ++jj)
          h_W2(n, kk, jj) = static_cast<Real>(net.W2[kk * net.h1 + jj]);
      }
      for (int jj = 0; jj < net.h2; ++jj)
        h_W3(n, jj) = static_cast<Real>(net.W3[jj]);
    }

    Kokkos::deep_copy(para, h_para); Kokkos::deep_copy(din, h_din);
    Kokkos::deep_copy(h1, h_h1);     Kokkos::deep_copy(h2, h_h2);
    Kokkos::deep_copy(sa, h_sa);     Kokkos::deep_copy(sb, h_sb);
    Kokkos::deep_copy(yra, h_yra);   Kokkos::deep_copy(yrb, h_yrb);
    Kokkos::deep_copy(b3, h_b3);     Kokkos::deep_copy(W1, h_W1);
    Kokkos::deep_copy(W2, h_W2);     Kokkos::deep_copy(W3, h_W3);
    Kokkos::deep_copy(b1, h_b1);     Kokkos::deep_copy(b2, h_b2);
  }

  // -----------------------------------------------------------------------
  // Device: evaluate network `n` on input x[0..din(n)-1].
  // -----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  Real forward(int n, const Real* x) const {
    const int nin = din(n);
    const int nh1 = h1(n);
    const int nh2 = h2(n);

    // 1. Scale inputs (affine)
    Real xs[MAX_DIN];
    for (int jj = 0; jj < nin; ++jj) xs[jj] = x[jj] * sa(n, jj) + sb(n, jj);

    // 2. Hidden layer 1
    Real h1v[MAX_H];
    for (int kk = 0; kk < nh1; ++kk) {
      Real s = b1(n, kk);
      for (int jj = 0; jj < nin; ++jj) s += W1(n, kk, jj) * xs[jj];
      h1v[kk] = elu1(s);
    }

    // 3. Hidden layer 2
    Real h2v[MAX_H];
    for (int kk = 0; kk < nh2; ++kk) {
      Real s = b2(n, kk);
      for (int jj = 0; jj < nh1; ++jj) s += W2(n, kk, jj) * h1v[jj];
      h2v[kk] = elu1(s);
    }

    // 4. Output neuron
    Real y = b3(n);
    for (int jj = 0; jj < nh2; ++jj) y += W3(n, jj) * h2v[jj];

    // 5. Post-processing
    switch (para(n)) {
      case 1:
        y = y * yra(n) + yrb(n);
        break;
      case 10: {
        y = y * yra(n) + yrb(n);
        Real t = safeExp(-y);
        y = -Kokkos::log(Kokkos::sqrt(t * t + 1.0));
        break;
      }
      case 11: {
        y = y * yra(n) + yrb(n);
        Real t = safeExp(-y + 0.6);
        y = -Kokkos::log(Kokkos::sqrt(t * t + 1.0)) - 0.6;
        break;
      }
      case 0:
        y = sigmoid1(y);
        break;
      case 2:
        y = 0.5 * sigmoid1(y);
        break;
      case -1:
        y = postprocess_para_neg1(y, x);
        break;
      default:
        break;
    }
    return y;
  }

  // -----------------------------------------------------------------------
  // Device: QSE abundances from (rho, T, Ye) in xx[0..2].
  // -----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void getQSE(const Real* xx, Real& yn, Real& yp, Real& ya,
              Real& yh, Real& ah, Real& yc) const {
    yn = forward(N_YN_QSE, xx);
    yp = forward(N_YP_QSE, xx);
    ya = forward(N_YA_QSE, xx);
    yh = forward(N_YH_QSE, xx);
    ah = forward(N_AH_QSE, xx);
    yc = forward(N_YC_QSE, xx);
    yn = exp10(yn);
    yp = exp10(yp);
    ya = exp10(ya);
    yh = exp10(yh);
  }

  // -----------------------------------------------------------------------
  // Device: mass excess and neutrino-loss fraction.
  // -----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void getMa(const Real* x, Real zh, Real& ma, Real& fnu) const {
    { const Real x4[4] = { x[0], x[1], x[2], x[6] };
      fnu = forward(N_FNU, x4); }
    { const Real x6[6] = { zh / x[6], x[6], x[3], x[8], x[4], x[5] };
      ma  = forward(N_MA, x6); }
  }

  // -----------------------------------------------------------------------
  // Device: predicted time derivatives / QSE mapping.
  // -----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void getDerivative(const Real* xin,
                     Real& dye, Real& dyn, Real& dyp,
                     Real& dya, Real& dyh, Real& dah, Real& yc,
                     Real ye0, Real yn0, Real yp0,
                     Real ya0, Real yh0, Real ah0,
                     Real dt, int pmode) const {
    Real xx[9];
    for (int ii = 0; ii < 9; ++ii) xx[ii] = xin[ii];
    xx[3] = Kokkos::fmax(xx[3], -10.0);
    xx[4] = Kokkos::fmax(xx[4], -10.0);
    xx[7] = Kokkos::copysign(Kokkos::fmax(Kokkos::fabs(xx[7]), 0.3), xx[7]);

    { Real x4[4] = { xx[0], xx[1], xx[2], xx[6] };
      dye = exp10(forward(N_DYE, x4)); }

    if (xx[7] < 0.0) {
      xx[7] = Kokkos::log10(Kokkos::fmax(-xx[7], 0.0) + 1e-2);

      { Real x5[5] = { xx[0], xx[1], xx[2], xx[6], xx[7] };
        dyn = -exp10(forward(N_DYN, x5)); }

      { Real x7[7] = { xx[0], xx[1], xx[2], xx[3], xx[5], xx[6], xx[7] };
        if (forward(N_DYHS, x7) <= 0.5) {
          dyh = 0.0;
        } else {
          Real x6[6] = { xx[0], xx[1], xx[2], xx[5], xx[6], xx[7] };
          dyh = exp10(forward(N_DYHP, x6));
        } }

      { Real x6[6] = { xx[0], xx[1], xx[2], xx[3], xx[6], xx[7] };
        Real vz = forward(N_DAHZ, x6);
        if (vz < 0.5) {
          dah = 0.0;
        } else {
          Real vs = forward(N_DAHS, x6);
          dah = (vs > 0.5) ?  exp10(forward(N_DAHP, x6))
                           : -exp10(forward(N_DAHN, x6));
        } }
    } else {
      dyn = dyh = dah = 0.0;
    }

    dyn /= L10;
    dyh /= L10;

    dyp = (pmode == 1) ? (-yp0 * Kokkos::fabs(xin[7]))
                       : (-yp0 / dt);

    if (xx[1] > LT9YA) {
      Real yn_q, yp_q, ya_q, yh_q, ah_q, yc_q;
      getQSE(xx, yn_q, yp_q, ya_q, yh_q, ah_q, yc_q);
      yc = yc_q;
      if (xx[1] > LT9QSE) {
        const Real inv_dt = 1.0 / dt;
        dyn = (yn_q - yn0) * inv_dt;
        dyp = (yp_q - yp0) * inv_dt;
        dya = (ya_q - ya0) * inv_dt;
        dyh = (yh_q - yh0) * inv_dt;
        dah = (ah_q - ah0) * inv_dt;
        dye = (xx[2] - ye0) * inv_dt;
      } else {
        dya = (ya_q - ya0) / 1e-4;
        dyh = 0.0;
      }
    } else {
      dya = 0.0;
    }

    if (Kokkos::isnan(dye) || dye < 1e-4)    dye = 0.0;
    if (Kokkos::isnan(dyn) || xx[3] < -10.0) dyn = 0.0;
    if (Kokkos::isnan(dyh))                  dyh = 0.0;
    if (Kokkos::isnan(dah))                  dah = 0.0;
  }

  // -----------------------------------------------------------------------
  // Device: enforce mass and charge conservation on the source terms.
  // -----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  static void normalization(Real& dye, Real& dyn, Real& dyp,
                            Real& dya, Real& dyh, Real& dah, Real yc1,
                            Real ye0, Real yn0, Real yp0,
                            Real ya0, Real yh0, Real ah0,
                            Real dt, Real t9) {
    Real ye1 = ye0 + dye * dt;
    Real yn1 = yn0 + dyn * dt;
    Real yp1 = yp0 + dyp * dt;
    Real ya1 = ya0 + dya * dt;
    Real yh1 = yh0 + dyh * dt;
    Real ah1 = ah0 + dah * dt;

    ye1 = Kokkos::fmax(Kokkos::fmin(ye1, 1.0), 0.0);
    yp1 = Kokkos::fmax(Kokkos::fmin(yp1, 1.0), 0.0);
    ya1 = Kokkos::fmax(Kokkos::fmin(ya1, 0.25), 0.0);

    if (((yn1 < 0.0) || (yn1 > 1.0)) && Kokkos::fabs(dyn) > 1e-8) {
      Real dum1 = yn1;
      yn1 = Kokkos::fmax(Kokkos::fmin(yn1, 1.0), 0.0);
      if (dah * dyn < 0.0)
        ah1 = ah0 + dah * (yn1 - yn0) / (dum1 - yn0) * dt;
    }
    ah1 = Kokkos::fmax(Kokkos::fmin(ah1, 300.0), 4.0);
    yh1 = Kokkos::fmax(Kokkos::fmin(yh1, 1.0 / ah1), 0.0);

    if (t9 > T9QSE) {
      // --- Mass conservation ---
      Real dxtot = 1.0 - (yn1 + yp1 + 4.0 * ya1 + yh1 * ah1);
      Real dxn   = yn1 * yn1;
      Real dxp   = yp1 * yp1;
      Real dxa   = 16.0 * ya1 * ya1;
      Real dxh   = (yh1 * ah1) * (yh1 * ah1);
      Real dxsum = dxn + dxp + dxa + dxh;
      const Real inv_sum = 1.0 / dxsum;
      yn1 += dxtot * dxn * inv_sum;
      yp1 += dxtot * dxp * inv_sum;
      ya1 += dxtot * dxa * inv_sum * 0.25;
      yh1 += dxtot * dxh * inv_sum / ah1;

      // --- Charge conservation ---
      Real ye_trial = yp1 + 2.0 * ya1 + yc1 * ah1 * yh1;
      dxtot = ye1 - ye_trial;

      dxn = yn1 * yn1             * Kokkos::fabs(ye1);
      dxp = yp1 * yp1             * Kokkos::fabs(ye1 - 1.0);
      dxa = 16.0 * ya1 * ya1      * Kokkos::fabs(ye1 - 0.5);
      dxh = (yh1 * ah1) * (yh1 * ah1) * Kokkos::fabs(ye1 - yc1);

      Real dxinc = 0.0, dxdec = 0.0;

      if      (dxtot < 0.0) { dxinc += dxn; }
      else if (dxtot > 0.0) { dxdec += dxn; dxn = -dxn; }

      if      (dxtot * (ye1 - 1.0) < 0.0) { dxinc += dxp; }
      else if (dxtot * (ye1 - 1.0) > 0.0) { dxdec += dxp; dxp = -dxp; }

      if      (dxtot * (ye1 - 0.5) < 0.0) { dxinc += dxa; }
      else if (dxtot * (ye1 - 0.5) > 0.0) { dxdec += dxa; dxa = -dxa; }

      if      (dxtot * (ye1 - yc1) < 0.0) { dxinc += dxh; }
      else if (dxtot * (ye1 - yc1) > 0.0) { dxdec += dxh; dxh = -dxh; }

      const Real ratio = (dxdec > 0.0) ? (dxinc / dxdec) : 1.0;
      if (dxtot > 0.0)                dxn *= ratio;
      if (dxtot * (ye1 - 1.0) > 0.0)  dxp *= ratio;
      if (dxtot * (ye1 - 0.5) > 0.0)  dxa *= ratio;
      if (dxtot * (ye1 - yc1) > 0.0)  dxh *= ratio;

      dxsum = dxp + 0.5 * dxa + yc1 * dxh;
      if (dxsum != 0.0) {
        const Real inv_ds = 1.0 / dxsum;
        yn1 += dxtot * dxn * inv_ds;
        yp1 += dxtot * dxp * inv_ds;
        ya1 += dxtot * dxa * inv_ds * 0.25;
        yh1 += dxtot * dxh * inv_ds / ah1;
      }
    } else {
      Real dxtot = 1.0 - (yn1 + yp1 + 4.0 * ya1 + yh1 * ah1);
      Real dxn   = yn1 * yn1;
      Real dxh   = (yh1 * ah1) * (yh1 * ah1);
      const Real inv_ds = 1.0 / (dxn + dxh);
      dxn *= inv_ds;
      dxh *= inv_ds;
      yn1 += dxtot * dxn;

      if (yh1 > 1e-10) {
        ah1 += dxh / yh1 * dxtot;
        if      (ah1 > 300.0) { yh1 = ah1 * yh1 / 300.0; ah1 = 300.0; }
        else if (ah1 <   4.0) { yh1 = ah1 * yh1 / 4.0;   ah1 =   4.0; }
      }

      Real zh1 = (ye1 - yp1 - 2.0 * ya1) / yh1;
      zh1 = Kokkos::fmax(Kokkos::fmin(zh1, 0.55 * ah1), 0.3 * ah1);
      if (ah1 >= 80.0)
        zh1 = Kokkos::fmin(zh1, (0.4 + 1.5 / (ah1 - 70.0)) * ah1);
      ye1 = yp1 + 2.0 * ya1 + zh1 * yh1;
    }

    const Real inv_dt = 1.0 / dt;
    dye = (ye1 - ye0) * inv_dt;
    dyn = (yn1 - yn0) * inv_dt;
    dyp = (yp1 - yp0) * inv_dt;
    dya = (ya1 - ya0) * inv_dt;
    dyh = (yh1 - yh0) * inv_dt;
    dah = (ah1 - ah0) * inv_dt;
  }

  // -----------------------------------------------------------------------
  // Device: full RHINE source-term evaluation for one cell (one time step).
  //
  // Units follow the original RHINE demo: rho [g/cm^3], tmev [MeV], dt [s],
  // drho = D ln(rho)/Dt [1/s].  Outputs dX are rates [1/s] (dma [MeV/baryon/s]).
  // -----------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  void run(Real rho,  Real tmev, Real ye,
           Real yn,   Real ya,   Real yh,  Real ah,
           Real drho, Real dt,
           Real ye0,  Real yn0,  Real ya0, Real yh0, Real ah0, Real mass0,
           Real& dye, Real& dyn, Real& dyp,
           Real& dya, Real& dyh, Real& dah,
           Real& dma, Real& fnu,
           int pmode) const {
    const Real t9  = T9MEV * tmev;
    const Real yp0 = 1.0 - yn0 - 4.0 * ya0 - ah0 * yh0;
    const Real yp  = 1.0 - yn  - 4.0 * ya  - ah  * yh;

    Real zh = (ye - yp - 2.0 * ya) / (yh + 1e-25);
    zh = Kokkos::fmax(Kokkos::fmin(zh, 0.55 * ah), 0.3 * ah);
    if (ah >= 80.0) zh = Kokkos::fmin(zh, (0.4 + 1.5 / (ah - 70.0)) * ah);

    Real x[9];
    x[0] = Kokkos::log10(rho);
    x[1] = Kokkos::log10(t9);
    x[2] = ye;
    x[3] = Kokkos::log10(yn + 1e-25);
    x[4] = Kokkos::log10(ya + 1e-25);
    x[5] = Kokkos::log10(yh + 1e-25);
    x[6] = ah;
    x[7] = drho;
    x[8] = Kokkos::log10(yp + 1e-25);

    if ((x[2] > 0.45 && x[1] < LT9QSE) || t9 > T9NSE || t9 < T9LOW) {
      dye = dyn = dyp = dya = dyh = dah = dma = fnu = 0.0;
      return;
    }

    Real yc = 0.0;
    getDerivative(x, dye, dyn, dyp, dya, dyh, dah, yc,
                  ye0, yn0, yp0, ya0, yh0, ah0, dt, pmode);
    normalization(dye, dyn, dyp, dya, dyh, dah, yc,
                  ye0, yn0, yp0, ya0, yh0, ah0, dt, t9);

    Real mass1 = 0.0;
    getMa(x, zh, mass1, fnu);
    dma = (mass1 - mass0) / dt;
  }
};

}  // namespace rhine

#endif  // RHINE_RHINE_NET_HPP_
