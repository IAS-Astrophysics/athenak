#ifndef HYDRO_RSOLVERS_ADVECT_HYD_HPP_
#define HYDRO_RSOLVERS_ADVECT_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect_hyd.hpp
//! \brief Advection Riemann solver for pure advection problems (v = constant): computes
//! the upwind flux of each variable.

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn Advect<ivx>()
//! \brief Upwind advection flux at face (m,k,j,i) for direction ivx.  Reads L/R
//! primitives from the global per-face buffers and writes a single flux entry.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void Advect(const EOS_Data &eos,
            const int m, const int k, const int j, const int i,
            const int is, const int js, const int ks,
            const DvceArray5D<Real> &wl,
            const DvceArray5D<Real> &wr,
            const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;

  const Real wl_idn = wl(m, IDN, k, j, i);
  const Real wl_ivx = wl(m, ivx, k, j, i);

  if (wl_ivx >= 0.0) {
    flx(m, IDN, k, j, i) = wl_idn*wl_ivx;
    flx(m, ivx, k, j, i) = wl_idn*wl_ivx*wl_ivx;
    flx(m, ivy, k, j, i) = wl(m, ivy, k, j, i)*wl_ivx;
    flx(m, ivz, k, j, i) = wl(m, ivz, k, j, i)*wl_ivx;
    if (eos.is_ideal) {
      flx(m, IEN, k, j, i) = wl(m, IEN, k, j, i)*wl_ivx;
    }
  } else {
    const Real wr_idn = wr(m, IDN, k, j, i);
    const Real wr_ivx = wr(m, ivx, k, j, i);
    flx(m, IDN, k, j, i) = wr_idn*wr_ivx;
    flx(m, ivx, k, j, i) = wr_idn*wr_ivx*wr_ivx;
    flx(m, ivy, k, j, i) = wr(m, ivy, k, j, i)*wr_ivx;
    flx(m, ivz, k, j, i) = wr(m, ivz, k, j, i)*wr_ivx;
    if (eos.is_ideal) {
      flx(m, IEN, k, j, i) = wr(m, IEN, k, j, i)*wr_ivx;
    }
  }
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_ADVECT_HYD_HPP_
