#ifndef RECONSTRUCT_RECON_HPP_
#define RECONSTRUCT_RECON_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file recon.hpp
//! \brief Per-cell reconstruction supporting every reconstruction method (DC, PLM, PPM4,
//! PPMX, WENOZ).  Each cell produces ql at its right face and qr at its left face,
//! written to the global per-face L/R buffers.  Templated on direction
//! (ivx = IVX|IVY|IVZ); the method is selected at runtime via a grid-uniform branch
//! (coherent across threads, so effectively branch-free on GPU).

#include "athena.hpp"
#include "eos/eos.hpp"
#include "dc.hpp"     // (trivial donor-cell handled inline below)
#include "plm.hpp"    // PLM()
#include "ppm.hpp"    // PPM4(), PPMX()
#include "wenoz.hpp"  // WENOZ()

//----------------------------------------------------------------------------------------
//! \fn ReconCellT<recon,ivx>()
//! \brief Single-variable, single-cell reconstruction with the method selected at COMPILE
//! TIME.  Reads the stencil of variable `n` centered on cell (m,k,j,i) in direction `ivx`,
//! and writes ql to face (i+1) and qr to face (i) in the global per-face buffers.  Buffers
//! are indexed by the GLOBAL cell/face index (m,n,k,j,i); they are sized to the full cell
//! range (including ghosts) so any face touched by the reconstruction loop is in bounds
//! (requires nghost >= 2).  The variable index `n` is supplied by the caller (it is a
//! parallel index of the launching kernel), so this function handles exactly one variable.
//! Caller loops over the cells whose faces are needed (one extra cell on the normal axis):
//!   ivx=IVX:  i in [is-1, ie+1], (transverse j,k over their active+ghost range)
//!   ivx=IVY:  j in [js-1, je+1], (transverse i,k over their active+ghost range)
//!   ivx=IVZ:  k in [ks-1, ke+1], (transverse i,j over their active+ghost range)
//!
//! The wide (+/-2) stencil used by PPM/WENOZ requires nghost >= 3 (enforced in the
//! Hydro/MHD constructor); DC/PLM only touch the +/-1 stencil.  The method-dispatch is a
//! constexpr-if on the template parameter, so no per-cell branch survives on the device.
template <ReconstructionMethod recon, int ivx>
KOKKOS_INLINE_FUNCTION
void ReconCellT(const EOS_Data &eos, const bool apply_floors,
                const int m, const int n, const int k, const int j, const int i,
                const DvceArray5D<Real> &q,
                const DvceArray5D<Real> &wl,
                const DvceArray5D<Real> &wr) {
  // Compile-time stencil offsets along the reconstruction direction
  constexpr int di = (ivx == IVX) ? 1 : 0;
  constexpr int dj = (ivx == IVY) ? 1 : 0;
  constexpr int dk = (ivx == IVZ) ? 1 : 0;

  const Real dfloor = eos.dfloor;
  const Real efloor = eos.is_ideal ? (eos.pfloor/(eos.gamma - 1.0)) : 0.0;

  Real ql_val, qr_val;
  if constexpr (recon == ReconstructionMethod::dc) {
    ql_val = q(m, n, k, j, i);
    qr_val = q(m, n, k, j, i);
  } else if constexpr (recon == ReconstructionMethod::plm) {
    PLM(q(m, n, k - dk, j - dj, i - di),
        q(m, n, k,      j,      i),
        q(m, n, k + dk, j + dj, i + di),
        ql_val, qr_val);
  } else if constexpr (recon == ReconstructionMethod::ppm4) {
    PPM4(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
         q(m, n, k -   dk, j -   dj, i -   di),
         q(m, n, k,        j,        i),
         q(m, n, k +   dk, j +   dj, i +   di),
         q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
         ql_val, qr_val);
  } else if constexpr (recon == ReconstructionMethod::ppmx) {
    PPMX(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
         q(m, n, k -   dk, j -   dj, i -   di),
         q(m, n, k,        j,        i),
         q(m, n, k +   dk, j +   dj, i +   di),
         q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
         ql_val, qr_val);
    if (apply_floors) {
      if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
      if (eos.is_ideal && n == IEN) {
        ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
      }
    }
  } else if constexpr (recon == ReconstructionMethod::wenoz) {
    WENOZ(q(m, n, k - 2*dk, j - 2*dj, i - 2*di),
          q(m, n, k -   dk, j -   dj, i -   di),
          q(m, n, k,        j,        i),
          q(m, n, k +   dk, j +   dj, i +   di),
          q(m, n, k + 2*dk, j + 2*dj, i + 2*di),
          ql_val, qr_val);
    if (apply_floors) {
      if (n == IDN) { ql_val = fmax(ql_val, dfloor); qr_val = fmax(qr_val, dfloor); }
      if (eos.is_ideal && n == IEN) {
        ql_val = fmax(ql_val, efloor); qr_val = fmax(qr_val, efloor);
      }
    }
  } else {
    ql_val = q(m, n, k, j, i);
    qr_val = q(m, n, k, j, i);
  }
  // ql is the right-face value of cell (writes the LEFT state of face i+1);
  // qr is the left-face value of cell (writes the RIGHT state of face i).
  wl(m, n, k + dk, j + dj, i + di) = ql_val;
  wr(m, n, k, j, i) = qr_val;
}

//----------------------------------------------------------------------------------------
//! \fn ReconLaunch<R,ivx>()
//! \brief Launch the per-cell, per-variable reconstruction kernel with the method R fixed
//! at COMPILE time, so the recon dispatch happens here (host side) rather than per-cell on
//! device.  This keeps only the selected method's code in each kernel (instead of inlining
//! all methods via a runtime switch), avoiding the register/footprint bloat that would
//! defeat the purpose of templating ReconCellT.  Shared by hydro, MHD, and dyn-GRMHD.
//!
//! Defined as a function template in this header so it has external (vague) linkage in
//! every translation unit that uses it: nvcc forbids extended __device__ lambdas inside
//! functions with internal linkage (e.g. an anonymous namespace), and a single header
//! definition keeps it ODR-safe across the solvers.  The variable index `n` is the inner
//! parallel index, run over [0, nvars-1]; `apply_floors` toggles the PPMX/WENOZ floors
//! (true for fluid primitives, false for the cell-centered B field).
template <ReconstructionMethod R, int ivx>
inline void ReconLaunch(const char *name, int nmb1,
    int kl, int ku, int jl, int ju, int il, int iu,
    const EOS_Data &eos, bool apply_floors, int nvars,
    const DvceArray5D<Real> &q,
    const DvceArray5D<Real> &ql, const DvceArray5D<Real> &qr) {
  par_for(name, DevExeSpace(), 0, nmb1, 0, nvars-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      ReconCellT<R, ivx>(eos, apply_floors, m, n, k, j, i, q, ql, qr);
    });
}

//----------------------------------------------------------------------------------------
//! \fn ReconDispatch<ivx>()
//! \brief Select the (runtime, grid-uniform) reconstruction method on the host and launch
//! the compile-time reconstruction kernel for it.  Shared by hydro, MHD, and dyn-GRMHD.
template <int ivx>
inline void ReconDispatch(ReconstructionMethod recon, const char *name, int nmb1,
    int kl, int ku, int jl, int ju, int il, int iu,
    const EOS_Data &eos, bool apply_floors, int nvars,
    const DvceArray5D<Real> &q,
    const DvceArray5D<Real> &ql, const DvceArray5D<Real> &qr) {
  switch (recon) {
    case ReconstructionMethod::dc:
      ReconLaunch<ReconstructionMethod::dc,    ivx>(name, nmb1, kl, ku, jl, ju,
          il, iu, eos, apply_floors, nvars, q, ql, qr); break;
    case ReconstructionMethod::plm:
      ReconLaunch<ReconstructionMethod::plm,   ivx>(name, nmb1, kl, ku, jl, ju,
          il, iu, eos, apply_floors, nvars, q, ql, qr); break;
    case ReconstructionMethod::ppm4:
      ReconLaunch<ReconstructionMethod::ppm4,  ivx>(name, nmb1, kl, ku, jl, ju,
          il, iu, eos, apply_floors, nvars, q, ql, qr); break;
    case ReconstructionMethod::ppmx:
      ReconLaunch<ReconstructionMethod::ppmx,  ivx>(name, nmb1, kl, ku, jl, ju,
          il, iu, eos, apply_floors, nvars, q, ql, qr); break;
    case ReconstructionMethod::wenoz:
      ReconLaunch<ReconstructionMethod::wenoz, ivx>(name, nmb1, kl, ku, jl, ju,
          il, iu, eos, apply_floors, nvars, q, ql, qr); break;
    default:
      ReconLaunch<ReconstructionMethod::dc,    ivx>(name, nmb1, kl, ku, jl, ju,
          il, iu, eos, apply_floors, nvars, q, ql, qr); break;
  }
}

#endif  // RECONSTRUCT_RECON_HPP_
