#ifndef MHD_RSOLVERS_ADVECT_MHD_HPP_
#define MHD_RSOLVERS_ADVECT_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect_mhd.hpp
//! \brief MHD Riemann solver for pure advection problems (v = constant).  Simply computes
//!  the upwind flux of each variable.  Can only be used for isothermal EOS.  Per-face
//!  implementation for the split-kernel path.

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn Advect<ivx>()
//! \brief An advection Riemann solver for MHD (isothermal), single face (m,k,j,i).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void Advect(const EOS_Data &eos,
            const int m, const int k, const int j, const int i,
            const int is, const int js, const int ks,
            const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
            const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
            const DvceArray4D<Real> &bx,
            const DvceArray5D<Real> &flx,
            const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX) + 1)%3;
  constexpr int ivz = IVX + ((ivx-IVX) + 2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;


  Real bxi = bx(m,k,j,i);

  //  Compute upwind fluxes
  if (wl(m,ivx,k,j,i) >= 0.0) {
    Real wl_idn = wl(m,IDN,k,j,i);
    Real wl_ivx = wl(m,ivx,k,j,i);
    flx(m,IDN,k,j,i) = wl_idn*wl_ivx;
    flx(m,ivx,k,j,i) = wl_idn*wl_ivx*wl_ivx;
    flx(m,ivy,k,j,i) = 0.0;
    flx(m,ivz,k,j,i) = 0.0;
    ey(m,k,j,i) = -bl(m,iby,k,j,i)*wl_ivx + bxi*wl(m,ivy,k,j,i);
    ez(m,k,j,i) =  bl(m,ibz,k,j,i)*wl_ivx - bxi*wl(m,ivz,k,j,i);
  } else {
    Real wr_idn = wr(m,IDN,k,j,i);
    Real wr_ivx = wr(m,ivx,k,j,i);
    flx(m,IDN,k,j,i) = wr_idn*wr_ivx;
    flx(m,ivx,k,j,i) = wr_idn*wr_ivx*wr_ivx;
    flx(m,ivy,k,j,i) = 0.0;
    flx(m,ivz,k,j,i) = 0.0;
    ey(m,k,j,i) = -br(m,iby,k,j,i)*wr_ivx + bxi*wr(m,ivy,k,j,i);
    ez(m,k,j,i) =  br(m,ibz,k,j,i)*wr_ivx - bxi*wr(m,ivz,k,j,i);
  }
}
} // namespace mhd
#endif // MHD_RSOLVERS_ADVECT_MHD_HPP_
