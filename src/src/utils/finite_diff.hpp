#ifndef UTILS_FINITE_DIFF_HPP_
#define UTILS_FINITE_DIFF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file finite_diff.hpp
//  \brief high order finite-differencing operators.

// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.
// 1st derivative scalar
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dx(int const dir,
        const Real idx[], TYPE &quant,
        int const m,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (  -1./2.   * quant(m,k+(-1)*shiftk,
                                  j+(-1)*shiftj,
                                  i+(-1)*shifti)
               +1./2.   * quant(m,k+( 1)*shiftk,
                                  j+( 1)*shiftj,
                                  i+( 1)*shifti));
  } else if constexpr ( NGHOST == 3 ) {
    out = + ( +1./12.   * quant(m,k+(-2)*shiftk,
                                  j+(-2)*shiftj,
                                  i+(-2)*shifti)
              -1./12.   * quant(m,k+( 2)*shiftk,
                                  j+( 2)*shiftj,
                                  i+( 2)*shifti))
          + (  -2./3.   * quant(m,k+(-1)*shiftk,
                                  j+(-1)*shiftj,
                                  i+(-1)*shifti)
               +2./3.   * quant(m,k+( 1)*shiftk,
                                  j+( 1)*shiftj,
                                  i+( 1)*shifti));
  } else if constexpr ( NGHOST == 4 ) {
    out = + ( -1./60.   * quant(m,k+(-3)*shiftk,
                                  j+(-3)*shiftj,
                                  i+(-3)*shifti)
              +1./60.   * quant(m,k+( 3)*shiftk,
                                  j+( 3)*shiftj,
                                  i+( 3)*shifti))
          + ( +3./20.   * quant(m,k+(-2)*shiftk,
                                  j+(-2)*shiftj,
                                  i+(-2)*shifti)
              -3./20.   * quant(m,k+( 2)*shiftk,
                                  j+( 2)*shiftj,
                                  i+( 2)*shifti))
          + (  -3./4.   * quant(m,k+(-1)*shiftk,
                                  j+(-1)*shiftj,
                                  i+(-1)*shifti)
               +3./4.   * quant(m,k+( 1)*shiftk,
                                  j+( 1)*shiftj,
                                  i+( 1)*shifti));
  }
  return out*idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative vector
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dx(int const dir,
        const Real idx[], TYPE &quant,
        int const m, int const a,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (  -1./2.   * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
               +1./2.   * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti));
  } else if constexpr ( NGHOST == 3 ) {
    out = + ( +1./12.   * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
              -1./12.   * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (  -2./3.   * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
               +2./3.   * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti));
  } else if constexpr ( NGHOST == 4 ) {
    out = + ( -1./60.   * quant(m,a,k+(-3)*shiftk,
                                    j+(-3)*shiftj,
                                    i+(-3)*shifti)
              +1./60.   * quant(m,a,k+( 3)*shiftk,
                                    j+( 3)*shiftj,
                                    i+( 3)*shifti))
          + ( +3./20.   * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
              -3./20.   * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (  -3./4.   * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
               +3./4.   * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti));
  }
  return out*idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative 2D tensor
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dx(int const dir,
        const Real idx[], TYPE &quant,
        int const m, int const a, int const b,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (  -1./2.   * quant(m,a,b,k+(-1)*shiftk,
                                      j+(-1)*shiftj,
                                      i+(-1)*shifti)
               +1./2.   * quant(m,a,b,k+( 1)*shiftk,
                                      j+( 1)*shiftj,
                                      i+( 1)*shifti));
  } else if constexpr ( NGHOST == 3 ) {
    out = + ( +1./12.   * quant(m,a,b,k+(-2)*shiftk,
                                      j+(-2)*shiftj,
                                      i+(-2)*shifti)
              -1./12.   * quant(m,a,b,k+( 2)*shiftk,
                                      j+( 2)*shiftj,
                                      i+( 2)*shifti))
          + (  -2./3.   * quant(m,a,b,k+(-1)*shiftk,
                                      j+(-1)*shiftj,
                                      i+(-1)*shifti)
               +2./3.   * quant(m,a,b,k+( 1)*shiftk,
                                      j+( 1)*shiftj,
                                      i+( 1)*shifti));
  } else if constexpr ( NGHOST == 4 ) {
    out = + ( -1./60.   * quant(m,a,b,k+(-3)*shiftk,
                                      j+(-3)*shiftj,
                                      i+(-3)*shifti)
              +1./60.   * quant(m,a,b,k+( 3)*shiftk,
                                      j+( 3)*shiftj,
                                      i+( 3)*shifti))
          + ( +3./20.   * quant(m,a,b,k+(-2)*shiftk,
                                      j+(-2)*shiftj,
                                      i+(-2)*shifti)
              -3./20.   * quant(m,a,b,k+( 2)*shiftk,
                                      j+( 2)*shiftj,
                                      i+( 2)*shifti))
          + (  -3./4.   * quant(m,a,b,k+(-1)*shiftk,
                                      j+(-1)*shiftj,
                                      i+(-1)*shifti)
               +3./4.   * quant(m,a,b,k+( 1)*shiftk,
                                      j+( 1)*shiftj,
                                      i+( 1)*shifti));
  }
  return out*idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative scalar
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dxx(int const dir,
        const Real idx[], TYPE &quant,
        int const m,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (   +1.     * quant(m,k+(-1)*shiftk,
                                  j+(-1)*shiftj,
                                  i+(-1)*shifti)
                +1.     * quant(m,k+( 1)*shiftk,
                                  j+( 1)*shiftj,
                                  i+( 1)*shifti))
                -2.     * quant(m,k,
                                  j,
                                  i);
  } else if constexpr ( NGHOST == 3 ) {
    out = + ( -1./12.   * quant(m,k+(-2)*shiftk,
                                  j+(-2)*shiftj,
                                  i+(-2)*shifti)
              -1./12.   * quant(m,k+( 2)*shiftk,
                                  j+( 2)*shiftj,
                                  i+( 2)*shifti))
          + (  +4./3.   * quant(m,k+(-1)*shiftk,
                                  j+(-1)*shiftj,
                                  i+(-1)*shifti)
               +4./3.   * quant(m,k+( 1)*shiftk,
                                  j+( 1)*shiftj,
                                  i+( 1)*shifti))
               -5./2.   * quant(m,k,
                                  j,
                                  i);
  } else if constexpr ( NGHOST == 4 ) {
    out = + ( +1./90.   * quant(m,k+(-3)*shiftk,
                                  j+(-3)*shiftj,
                                  i+(-3)*shifti)
              +1./90.   * quant(m,k+( 3)*shiftk,
                                  j+( 3)*shiftj,
                                  i+( 3)*shifti))
          + ( -3./20.   * quant(m,k+(-2)*shiftk,
                                  j+(-2)*shiftj,
                                  i+(-2)*shifti)
              -3./20.   * quant(m,k+( 2)*shiftk,
                                  j+( 2)*shiftj,
                                  i+( 2)*shifti))
          + (  +3./2.   * quant(m,k+(-1)*shiftk,
                                  j+(-1)*shiftj,
                                  i+(-1)*shifti)
               +3./2.   * quant(m,k+( 1)*shiftk,
                                  j+( 1)*shiftj,
                                  i+( 1)*shifti))
              -49./18.  * quant(m,k,
                                  j,
                                  i);
  }
  return out*idx[dir]*idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative vector
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dxx(int const dir,
        const Real idx[], TYPE &quant,
        int const m, int const a,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (   +1.     * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
                +1.     * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti))
                -2.     * quant(m,a,k,
                                    j,
                                    i);
  } else if constexpr ( NGHOST == 3 ) {
    out = + ( -1./12.   * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
              -1./12.   * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (  +4./3.   * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
               +4./3.   * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti))
               -5./2.   * quant(m,a,k,
                                    j,
                                    i);
  } else if constexpr ( NGHOST == 4 ) {
    out = + ( +1./90.   * quant(m,a,k+(-3)*shiftk,
                                    j+(-3)*shiftj,
                                    i+(-3)*shifti)
              +1./90.   * quant(m,a,k+( 3)*shiftk,
                                    j+( 3)*shiftj,
                                    i+( 3)*shifti))
          + ( -3./20.   * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
              -3./20.   * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (  +3./2.   * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
               +3./2.   * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti))
              -49./18.  * quant(m,a,k,
                                    j,
                                    i);
  }
  return out*idx[dir]*idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative 2D tensor
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dxx(int const dir,
        const Real idx[], TYPE &quant,
        int const m, int const a, int const b,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (   +1.     * quant(m,a,b,k+(-1)*shiftk,
                                      j+(-1)*shiftj,
                                      i+(-1)*shifti)
                +1.     * quant(m,a,b,k+( 1)*shiftk,
                                      j+( 1)*shiftj,
                                      i+( 1)*shifti))
                -2.     * quant(m,a,b,k,
                                      j,
                                      i);
  } else if constexpr ( NGHOST == 3 ) {
    out = + ( -1./12.   * quant(m,a,b,k+(-2)*shiftk,
                                      j+(-2)*shiftj,
                                      i+(-2)*shifti)
              -1./12.   * quant(m,a,b,k+( 2)*shiftk,
                                      j+( 2)*shiftj,
                                      i+( 2)*shifti))
          + (  +4./3.   * quant(m,a,b,k+(-1)*shiftk,
                                      j+(-1)*shiftj,
                                      i+(-1)*shifti)
               +4./3.   * quant(m,a,b,k+( 1)*shiftk,
                                      j+( 1)*shiftj,
                                      i+( 1)*shifti))
               -5./2.   * quant(m,a,b,k,
                                      j,
                                      i);
  } else if constexpr ( NGHOST == 4 ) {
    out = + ( +1./90.   * quant(m,a,b,k+(-3)*shiftk,
                                      j+(-3)*shiftj,
                                      i+(-3)*shifti)
              +1./90.   * quant(m,a,b,k+( 3)*shiftk,
                                      j+( 3)*shiftj,
                                      i+( 3)*shifti))
          + ( -3./20.   * quant(m,a,b,k+(-2)*shiftk,
                                      j+(-2)*shiftj,
                                      i+(-2)*shifti)
              -3./20.   * quant(m,a,b,k+( 2)*shiftk,
                                      j+( 2)*shiftj,
                                      i+( 2)*shifti))
          + (  +3./2.   * quant(m,a,b,k+(-1)*shiftk,
                                      j+(-1)*shiftj,
                                      i+(-1)*shifti)
               +3./2.   * quant(m,a,b,k+( 1)*shiftk,
                                      j+( 1)*shiftj,
                                      i+( 1)*shifti))
              -49./18.  * quant(m,a,b,k,
                                      j,
                                      i);
  }
  return out*idx[dir]*idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative scalar
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dxy(int const dirx, int const diry,
        const Real idx[], TYPE &quant,
        int const m,
        int const k, int const j, int const i) {
  int const shiftxk = dirx==2;
  int const shiftxj = dirx==1;
  int const shiftxi = dirx==0;
  int const shiftyk = diry==2;
  int const shiftyj = diry==1;
  int const shiftyi = diry==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (
            + (
              + (  -1./2.  ) * (  -1./2.  ) * quant(m,k+(-1)*shiftxk + (-1)*shiftyk,
                                                      j+(-1)*shiftxj + (-1)*shiftyj,
                                                      i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -1./2.  ) * (  1./2.   ) * quant(m,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                      j+(-1)*shiftxj + ( 1)*shiftyj,
                                                      i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  1./2.   ) * (  -1./2.  ) * quant(m,k+( 1)*shiftxk + (-1)*shiftyk,
                                                      j+( 1)*shiftxj + (-1)*shiftyj,
                                                      i+( 1)*shiftxi + (-1)*shiftyi)
              + (  1./2.   ) * (  1./2.   ) * quant(m,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                      j+( 1)*shiftxj + ( 1)*shiftyj,
                                                      i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  } else if constexpr ( NGHOST == 3 ) {
    out = + (
            + (
              + (  1./12.  ) * (  1./12.  ) * quant(m,k+(-2)*shiftxk + (-2)*shiftyk,
                                                      j+(-2)*shiftxj + (-2)*shiftyj,
                                                      i+(-2)*shiftxi + (-2)*shiftyi)
              + (  1./12.  ) * ( -1./12.  ) * quant(m,k+(-2)*shiftxk + ( 2)*shiftyk,
                                                      j+(-2)*shiftxj + ( 2)*shiftyj,
                                                      i+(-2)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + ( -1./12.  ) * (  1./12.  ) * quant(m,k+( 2)*shiftxk + (-2)*shiftyk,
                                                      j+( 2)*shiftxj + (-2)*shiftyj,
                                                      i+( 2)*shiftxi + (-2)*shiftyi)
              + ( -1./12.  ) * ( -1./12.  ) * quant(m,k+( 2)*shiftxk + ( 2)*shiftyk,
                                                      j+( 2)*shiftxj + ( 2)*shiftyj,
                                                      i+( 2)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  1./12.  ) * (  -2./3.  ) * quant(m,k+(-2)*shiftxk + (-1)*shiftyk,
                                                      j+(-2)*shiftxj + (-1)*shiftyj,
                                                      i+(-2)*shiftxi + (-1)*shiftyi)
              + (  1./12.  ) * (  2./3.   ) * quant(m,k+(-2)*shiftxk + ( 1)*shiftyk,
                                                      j+(-2)*shiftxj + ( 1)*shiftyj,
                                                      i+(-2)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + ( -1./12.  ) * (  -2./3.  ) * quant(m,k+( 2)*shiftxk + (-1)*shiftyk,
                                                      j+( 2)*shiftxj + (-1)*shiftyj,
                                                      i+( 2)*shiftxi + (-1)*shiftyi)
              + ( -1./12.  ) * (  2./3.   ) * quant(m,k+( 2)*shiftxk + ( 1)*shiftyk,
                                                      j+( 2)*shiftxj + ( 1)*shiftyj,
                                                      i+( 2)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  -2./3.  ) * (  1./12.  ) * quant(m,k+(-1)*shiftxk + (-2)*shiftyk,
                                                      j+(-1)*shiftxj + (-2)*shiftyj,
                                                      i+(-1)*shiftxi + (-2)*shiftyi)
              + (  -2./3.  ) * ( -1./12.  ) * quant(m,k+(-1)*shiftxk + ( 2)*shiftyk,
                                                      j+(-1)*shiftxj + ( 2)*shiftyj,
                                                      i+(-1)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  2./3.   ) * (  1./12.  ) * quant(m,k+( 1)*shiftxk + (-2)*shiftyk,
                                                      j+( 1)*shiftxj + (-2)*shiftyj,
                                                      i+( 1)*shiftxi + (-2)*shiftyi)
              + (  2./3.   ) * ( -1./12.  ) * quant(m,k+( 1)*shiftxk + ( 2)*shiftyk,
                                                      j+( 1)*shiftxj + ( 2)*shiftyj,
                                                      i+( 1)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  -2./3.  ) * (  -2./3.  ) * quant(m,k+(-1)*shiftxk + (-1)*shiftyk,
                                                      j+(-1)*shiftxj + (-1)*shiftyj,
                                                      i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -2./3.  ) * (  2./3.   ) * quant(m,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                      j+(-1)*shiftxj + ( 1)*shiftyj,
                                                      i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  2./3.   ) * (  -2./3.  ) * quant(m,k+( 1)*shiftxk + (-1)*shiftyk,
                                                      j+( 1)*shiftxj + (-1)*shiftyj,
                                                      i+( 1)*shiftxi + (-1)*shiftyi)
              + (  2./3.   ) * (  2./3.   ) * quant(m,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                      j+( 1)*shiftxj + ( 1)*shiftyj,
                                                      i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  } else if constexpr ( NGHOST == 4 ) {
    out = + (
            + (
              + ( -1./60.  ) * ( -1./60.  ) * quant(m,k+(-3)*shiftxk + (-3)*shiftyk,
                                                      j+(-3)*shiftxj + (-3)*shiftyj,
                                                      i+(-3)*shiftxi + (-3)*shiftyi)
              + ( -1./60.  ) * (  1./60.  ) * quant(m,k+(-3)*shiftxk + ( 3)*shiftyk,
                                                      j+(-3)*shiftxj + ( 3)*shiftyj,
                                                      i+(-3)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + (  1./60.  ) * ( -1./60.  ) * quant(m,k+( 3)*shiftxk + (-3)*shiftyk,
                                                      j+( 3)*shiftxj + (-3)*shiftyj,
                                                      i+( 3)*shiftxi + (-3)*shiftyi)
              + (  1./60.  ) * (  1./60.  ) * quant(m,k+( 3)*shiftxk + ( 3)*shiftyk,
                                                      j+( 3)*shiftxj + ( 3)*shiftyj,
                                                      i+( 3)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + ( -1./60.  ) * (  3./20.  ) * quant(m,k+(-3)*shiftxk + (-2)*shiftyk,
                                                      j+(-3)*shiftxj + (-2)*shiftyj,
                                                      i+(-3)*shiftxi + (-2)*shiftyi)
              + ( -1./60.  ) * ( -3./20.  ) * quant(m,k+(-3)*shiftxk + ( 2)*shiftyk,
                                                      j+(-3)*shiftxj + ( 2)*shiftyj,
                                                      i+(-3)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  1./60.  ) * (  3./20.  ) * quant(m,k+( 3)*shiftxk + (-2)*shiftyk,
                                                      j+( 3)*shiftxj + (-2)*shiftyj,
                                                      i+( 3)*shiftxi + (-2)*shiftyi)
              + (  1./60.  ) * ( -3./20.  ) * quant(m,k+( 3)*shiftxk + ( 2)*shiftyk,
                                                      j+( 3)*shiftxj + ( 2)*shiftyj,
                                                      i+( 3)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + ( -1./60.  ) * (  -3./4.  ) * quant(m,k+(-3)*shiftxk + (-1)*shiftyk,
                                                      j+(-3)*shiftxj + (-1)*shiftyj,
                                                      i+(-3)*shiftxi + (-1)*shiftyi)
              + ( -1./60.  ) * (  3./4.   ) * quant(m,k+(-3)*shiftxk + ( 1)*shiftyk,
                                                      j+(-3)*shiftxj + ( 1)*shiftyj,
                                                      i+(-3)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  1./60.  ) * (  -3./4.  ) * quant(m,k+( 3)*shiftxk + (-1)*shiftyk,
                                                      j+( 3)*shiftxj + (-1)*shiftyj,
                                                      i+( 3)*shiftxi + (-1)*shiftyi)
              + (  1./60.  ) * (  3./4.   ) * quant(m,k+( 3)*shiftxk + ( 1)*shiftyk,
                                                      j+( 3)*shiftxj + ( 1)*shiftyj,
                                                      i+( 3)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * ( -1./60.  ) * quant(m,k+(-2)*shiftxk + (-3)*shiftyk,
                                                      j+(-2)*shiftxj + (-3)*shiftyj,
                                                      i+(-2)*shiftxi + (-3)*shiftyi)
              + (  3./20.  ) * (  1./60.  ) * quant(m,k+(-2)*shiftxk + ( 3)*shiftyk,
                                                      j+(-2)*shiftxj + ( 3)*shiftyj,
                                                      i+(-2)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + ( -3./20.  ) * ( -1./60.  ) * quant(m,k+( 2)*shiftxk + (-3)*shiftyk,
                                                      j+( 2)*shiftxj + (-3)*shiftyj,
                                                      i+( 2)*shiftxi + (-3)*shiftyi)
              + ( -3./20.  ) * (  1./60.  ) * quant(m,k+( 2)*shiftxk + ( 3)*shiftyk,
                                                      j+( 2)*shiftxj + ( 3)*shiftyj,
                                                      i+( 2)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * (  3./20.  ) * quant(m,k+(-2)*shiftxk + (-2)*shiftyk,
                                                      j+(-2)*shiftxj + (-2)*shiftyj,
                                                      i+(-2)*shiftxi + (-2)*shiftyi)
              + (  3./20.  ) * ( -3./20.  ) * quant(m,k+(-2)*shiftxk + ( 2)*shiftyk,
                                                      j+(-2)*shiftxj + ( 2)*shiftyj,
                                                      i+(-2)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + ( -3./20.  ) * (  3./20.  ) * quant(m,k+( 2)*shiftxk + (-2)*shiftyk,
                                                      j+( 2)*shiftxj + (-2)*shiftyj,
                                                      i+( 2)*shiftxi + (-2)*shiftyi)
              + ( -3./20.  ) * ( -3./20.  ) * quant(m,k+( 2)*shiftxk + ( 2)*shiftyk,
                                                      j+( 2)*shiftxj + ( 2)*shiftyj,
                                                      i+( 2)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * (  -3./4.  ) * quant(m,k+(-2)*shiftxk + (-1)*shiftyk,
                                                      j+(-2)*shiftxj + (-1)*shiftyj,
                                                      i+(-2)*shiftxi + (-1)*shiftyi)
              + (  3./20.  ) * (  3./4.   ) * quant(m,k+(-2)*shiftxk + ( 1)*shiftyk,
                                                      j+(-2)*shiftxj + ( 1)*shiftyj,
                                                      i+(-2)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + ( -3./20.  ) * (  -3./4.  ) * quant(m,k+( 2)*shiftxk + (-1)*shiftyk,
                                                      j+( 2)*shiftxj + (-1)*shiftyj,
                                                      i+( 2)*shiftxi + (-1)*shiftyi)
              + ( -3./20.  ) * (  3./4.   ) * quant(m,k+( 2)*shiftxk + ( 1)*shiftyk,
                                                      j+( 2)*shiftxj + ( 1)*shiftyj,
                                                      i+( 2)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * ( -1./60.  ) * quant(m,k+(-1)*shiftxk + (-3)*shiftyk,
                                                      j+(-1)*shiftxj + (-3)*shiftyj,
                                                      i+(-1)*shiftxi + (-3)*shiftyi)
              + (  -3./4.  ) * (  1./60.  ) * quant(m,k+(-1)*shiftxk + ( 3)*shiftyk,
                                                      j+(-1)*shiftxj + ( 3)*shiftyj,
                                                      i+(-1)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + (  3./4.   ) * ( -1./60.  ) * quant(m,k+( 1)*shiftxk + (-3)*shiftyk,
                                                      j+( 1)*shiftxj + (-3)*shiftyj,
                                                      i+( 1)*shiftxi + (-3)*shiftyi)
              + (  3./4.   ) * (  1./60.  ) * quant(m,k+( 1)*shiftxk + ( 3)*shiftyk,
                                                      j+( 1)*shiftxj + ( 3)*shiftyj,
                                                      i+( 1)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * (  3./20.  ) * quant(m,k+(-1)*shiftxk + (-2)*shiftyk,
                                                      j+(-1)*shiftxj + (-2)*shiftyj,
                                                      i+(-1)*shiftxi + (-2)*shiftyi)
              + (  -3./4.  ) * ( -3./20.  ) * quant(m,k+(-1)*shiftxk + ( 2)*shiftyk,
                                                      j+(-1)*shiftxj + ( 2)*shiftyj,
                                                      i+(-1)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  3./4.   ) * (  3./20.  ) * quant(m,k+( 1)*shiftxk + (-2)*shiftyk,
                                                      j+( 1)*shiftxj + (-2)*shiftyj,
                                                      i+( 1)*shiftxi + (-2)*shiftyi)
              + (  3./4.   ) * ( -3./20.  ) * quant(m,k+( 1)*shiftxk + ( 2)*shiftyk,
                                                      j+( 1)*shiftxj + ( 2)*shiftyj,
                                                      i+( 1)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * (  -3./4.  ) * quant(m,k+(-1)*shiftxk + (-1)*shiftyk,
                                                      j+(-1)*shiftxj + (-1)*shiftyj,
                                                      i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -3./4.  ) * (  3./4.   ) * quant(m,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                      j+(-1)*shiftxj + ( 1)*shiftyj,
                                                      i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  3./4.   ) * (  -3./4.  ) * quant(m,k+( 1)*shiftxk + (-1)*shiftyk,
                                                      j+( 1)*shiftxj + (-1)*shiftyj,
                                                      i+( 1)*shiftxi + (-1)*shiftyi)
              + (  3./4.   ) * (  3./4.   ) * quant(m,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                      j+( 1)*shiftxj + ( 1)*shiftyj,
                                                      i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  }
  return out*idx[dirx]*idx[diry];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative vector
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dxy(int const dirx, int const diry,
        const Real idx[], TYPE &quant,
        int const m, int const a,
        int const k, int const j, int const i) {
  int const shiftxk = dirx==2;
  int const shiftxj = dirx==1;
  int const shiftxi = dirx==0;
  int const shiftyk = diry==2;
  int const shiftyj = diry==1;
  int const shiftyi = diry==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (
            + (
              + (  -1./2.  ) * (  -1./2.  ) * quant(m,a,k+(-1)*shiftxk + (-1)*shiftyk,
                                                        j+(-1)*shiftxj + (-1)*shiftyj,
                                                        i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -1./2.  ) * (  1./2.   ) * quant(m,a,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                        j+(-1)*shiftxj + ( 1)*shiftyj,
                                                        i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  1./2.   ) * (  -1./2.  ) * quant(m,a,k+( 1)*shiftxk + (-1)*shiftyk,
                                                        j+( 1)*shiftxj + (-1)*shiftyj,
                                                        i+( 1)*shiftxi + (-1)*shiftyi)
              + (  1./2.   ) * (  1./2.   ) * quant(m,a,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                        j+( 1)*shiftxj + ( 1)*shiftyj,
                                                        i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  } else if constexpr ( NGHOST == 3 ) {
    out = + (
            + (
              + (  1./12.  ) * (  1./12.  ) * quant(m,a,k+(-2)*shiftxk + (-2)*shiftyk,
                                                        j+(-2)*shiftxj + (-2)*shiftyj,
                                                        i+(-2)*shiftxi + (-2)*shiftyi)
              + (  1./12.  ) * ( -1./12.  ) * quant(m,a,k+(-2)*shiftxk + ( 2)*shiftyk,
                                                        j+(-2)*shiftxj + ( 2)*shiftyj,
                                                        i+(-2)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + ( -1./12.  ) * (  1./12.  ) * quant(m,a,k+( 2)*shiftxk + (-2)*shiftyk,
                                                        j+( 2)*shiftxj + (-2)*shiftyj,
                                                        i+( 2)*shiftxi + (-2)*shiftyi)
              + ( -1./12.  ) * ( -1./12.  ) * quant(m,a,k+( 2)*shiftxk + ( 2)*shiftyk,
                                                        j+( 2)*shiftxj + ( 2)*shiftyj,
                                                        i+( 2)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  1./12.  ) * (  -2./3.  ) * quant(m,a,k+(-2)*shiftxk + (-1)*shiftyk,
                                                        j+(-2)*shiftxj + (-1)*shiftyj,
                                                        i+(-2)*shiftxi + (-1)*shiftyi)
              + (  1./12.  ) * (  2./3.   ) * quant(m,a,k+(-2)*shiftxk + ( 1)*shiftyk,
                                                        j+(-2)*shiftxj + ( 1)*shiftyj,
                                                        i+(-2)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + ( -1./12.  ) * (  -2./3.  ) * quant(m,a,k+( 2)*shiftxk + (-1)*shiftyk,
                                                        j+( 2)*shiftxj + (-1)*shiftyj,
                                                        i+( 2)*shiftxi + (-1)*shiftyi)
              + ( -1./12.  ) * (  2./3.   ) * quant(m,a,k+( 2)*shiftxk + ( 1)*shiftyk,
                                                        j+( 2)*shiftxj + ( 1)*shiftyj,
                                                        i+( 2)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  -2./3.  ) * (  1./12.  ) * quant(m,a,k+(-1)*shiftxk + (-2)*shiftyk,
                                                        j+(-1)*shiftxj + (-2)*shiftyj,
                                                        i+(-1)*shiftxi + (-2)*shiftyi)
              + (  -2./3.  ) * ( -1./12.  ) * quant(m,a,k+(-1)*shiftxk + ( 2)*shiftyk,
                                                        j+(-1)*shiftxj + ( 2)*shiftyj,
                                                        i+(-1)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  2./3.   ) * (  1./12.  ) * quant(m,a,k+( 1)*shiftxk + (-2)*shiftyk,
                                                        j+( 1)*shiftxj + (-2)*shiftyj,
                                                        i+( 1)*shiftxi + (-2)*shiftyi)
              + (  2./3.   ) * ( -1./12.  ) * quant(m,a,k+( 1)*shiftxk + ( 2)*shiftyk,
                                                        j+( 1)*shiftxj + ( 2)*shiftyj,
                                                        i+( 1)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  -2./3.  ) * (  -2./3.  ) * quant(m,a,k+(-1)*shiftxk + (-1)*shiftyk,
                                                        j+(-1)*shiftxj + (-1)*shiftyj,
                                                        i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -2./3.  ) * (  2./3.   ) * quant(m,a,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                        j+(-1)*shiftxj + ( 1)*shiftyj,
                                                        i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  2./3.   ) * (  -2./3.  ) * quant(m,a,k+( 1)*shiftxk + (-1)*shiftyk,
                                                        j+( 1)*shiftxj + (-1)*shiftyj,
                                                        i+( 1)*shiftxi + (-1)*shiftyi)
              + (  2./3.   ) * (  2./3.   ) * quant(m,a,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                        j+( 1)*shiftxj + ( 1)*shiftyj,
                                                        i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  } else if constexpr ( NGHOST == 4 ) {
    out = + (
            + (
              + ( -1./60.  ) * ( -1./60.  ) * quant(m,a,k+(-3)*shiftxk + (-3)*shiftyk,
                                                        j+(-3)*shiftxj + (-3)*shiftyj,
                                                        i+(-3)*shiftxi + (-3)*shiftyi)
              + ( -1./60.  ) * (  1./60.  ) * quant(m,a,k+(-3)*shiftxk + ( 3)*shiftyk,
                                                        j+(-3)*shiftxj + ( 3)*shiftyj,
                                                        i+(-3)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + (  1./60.  ) * ( -1./60.  ) * quant(m,a,k+( 3)*shiftxk + (-3)*shiftyk,
                                                        j+( 3)*shiftxj + (-3)*shiftyj,
                                                        i+( 3)*shiftxi + (-3)*shiftyi)
              + (  1./60.  ) * (  1./60.  ) * quant(m,a,k+( 3)*shiftxk + ( 3)*shiftyk,
                                                        j+( 3)*shiftxj + ( 3)*shiftyj,
                                                        i+( 3)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + ( -1./60.  ) * (  3./20.  ) * quant(m,a,k+(-3)*shiftxk + (-2)*shiftyk,
                                                        j+(-3)*shiftxj + (-2)*shiftyj,
                                                        i+(-3)*shiftxi + (-2)*shiftyi)
              + ( -1./60.  ) * ( -3./20.  ) * quant(m,a,k+(-3)*shiftxk + ( 2)*shiftyk,
                                                        j+(-3)*shiftxj + ( 2)*shiftyj,
                                                        i+(-3)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  1./60.  ) * (  3./20.  ) * quant(m,a,k+( 3)*shiftxk + (-2)*shiftyk,
                                                        j+( 3)*shiftxj + (-2)*shiftyj,
                                                        i+( 3)*shiftxi + (-2)*shiftyi)
              + (  1./60.  ) * ( -3./20.  ) * quant(m,a,k+( 3)*shiftxk + ( 2)*shiftyk,
                                                        j+( 3)*shiftxj + ( 2)*shiftyj,
                                                        i+( 3)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + ( -1./60.  ) * (  -3./4.  ) * quant(m,a,k+(-3)*shiftxk + (-1)*shiftyk,
                                                        j+(-3)*shiftxj + (-1)*shiftyj,
                                                        i+(-3)*shiftxi + (-1)*shiftyi)
              + ( -1./60.  ) * (  3./4.   ) * quant(m,a,k+(-3)*shiftxk + ( 1)*shiftyk,
                                                        j+(-3)*shiftxj + ( 1)*shiftyj,
                                                        i+(-3)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  1./60.  ) * (  -3./4.  ) * quant(m,a,k+( 3)*shiftxk + (-1)*shiftyk,
                                                        j+( 3)*shiftxj + (-1)*shiftyj,
                                                        i+( 3)*shiftxi + (-1)*shiftyi)
              + (  1./60.  ) * (  3./4.   ) * quant(m,a,k+( 3)*shiftxk + ( 1)*shiftyk,
                                                        j+( 3)*shiftxj + ( 1)*shiftyj,
                                                        i+( 3)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * ( -1./60.  ) * quant(m,a,k+(-2)*shiftxk + (-3)*shiftyk,
                                                        j+(-2)*shiftxj + (-3)*shiftyj,
                                                        i+(-2)*shiftxi + (-3)*shiftyi)
              + (  3./20.  ) * (  1./60.  ) * quant(m,a,k+(-2)*shiftxk + ( 3)*shiftyk,
                                                        j+(-2)*shiftxj + ( 3)*shiftyj,
                                                        i+(-2)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + ( -3./20.  ) * ( -1./60.  ) * quant(m,a,k+( 2)*shiftxk + (-3)*shiftyk,
                                                        j+( 2)*shiftxj + (-3)*shiftyj,
                                                        i+( 2)*shiftxi + (-3)*shiftyi)
              + ( -3./20.  ) * (  1./60.  ) * quant(m,a,k+( 2)*shiftxk + ( 3)*shiftyk,
                                                        j+( 2)*shiftxj + ( 3)*shiftyj,
                                                        i+( 2)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * (  3./20.  ) * quant(m,a,k+(-2)*shiftxk + (-2)*shiftyk,
                                                        j+(-2)*shiftxj + (-2)*shiftyj,
                                                        i+(-2)*shiftxi + (-2)*shiftyi)
              + (  3./20.  ) * ( -3./20.  ) * quant(m,a,k+(-2)*shiftxk + ( 2)*shiftyk,
                                                        j+(-2)*shiftxj + ( 2)*shiftyj,
                                                        i+(-2)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + ( -3./20.  ) * (  3./20.  ) * quant(m,a,k+( 2)*shiftxk + (-2)*shiftyk,
                                                        j+( 2)*shiftxj + (-2)*shiftyj,
                                                        i+( 2)*shiftxi + (-2)*shiftyi)
              + ( -3./20.  ) * ( -3./20.  ) * quant(m,a,k+( 2)*shiftxk + ( 2)*shiftyk,
                                                        j+( 2)*shiftxj + ( 2)*shiftyj,
                                                        i+( 2)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * (  -3./4.  ) * quant(m,a,k+(-2)*shiftxk + (-1)*shiftyk,
                                                        j+(-2)*shiftxj + (-1)*shiftyj,
                                                        i+(-2)*shiftxi + (-1)*shiftyi)
              + (  3./20.  ) * (  3./4.   ) * quant(m,a,k+(-2)*shiftxk + ( 1)*shiftyk,
                                                        j+(-2)*shiftxj + ( 1)*shiftyj,
                                                        i+(-2)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + ( -3./20.  ) * (  -3./4.  ) * quant(m,a,k+( 2)*shiftxk + (-1)*shiftyk,
                                                        j+( 2)*shiftxj + (-1)*shiftyj,
                                                        i+( 2)*shiftxi + (-1)*shiftyi)
              + ( -3./20.  ) * (  3./4.   ) * quant(m,a,k+( 2)*shiftxk + ( 1)*shiftyk,
                                                        j+( 2)*shiftxj + ( 1)*shiftyj,
                                                        i+( 2)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * ( -1./60.  ) * quant(m,a,k+(-1)*shiftxk + (-3)*shiftyk,
                                                        j+(-1)*shiftxj + (-3)*shiftyj,
                                                        i+(-1)*shiftxi + (-3)*shiftyi)
              + (  -3./4.  ) * (  1./60.  ) * quant(m,a,k+(-1)*shiftxk + ( 3)*shiftyk,
                                                        j+(-1)*shiftxj + ( 3)*shiftyj,
                                                        i+(-1)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + (  3./4.   ) * ( -1./60.  ) * quant(m,a,k+( 1)*shiftxk + (-3)*shiftyk,
                                                        j+( 1)*shiftxj + (-3)*shiftyj,
                                                        i+( 1)*shiftxi + (-3)*shiftyi)
              + (  3./4.   ) * (  1./60.  ) * quant(m,a,k+( 1)*shiftxk + ( 3)*shiftyk,
                                                        j+( 1)*shiftxj + ( 3)*shiftyj,
                                                        i+( 1)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * (  3./20.  ) * quant(m,a,k+(-1)*shiftxk + (-2)*shiftyk,
                                                        j+(-1)*shiftxj + (-2)*shiftyj,
                                                        i+(-1)*shiftxi + (-2)*shiftyi)
              + (  -3./4.  ) * ( -3./20.  ) * quant(m,a,k+(-1)*shiftxk + ( 2)*shiftyk,
                                                        j+(-1)*shiftxj + ( 2)*shiftyj,
                                                        i+(-1)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  3./4.   ) * (  3./20.  ) * quant(m,a,k+( 1)*shiftxk + (-2)*shiftyk,
                                                        j+( 1)*shiftxj + (-2)*shiftyj,
                                                        i+( 1)*shiftxi + (-2)*shiftyi)
              + (  3./4.   ) * ( -3./20.  ) * quant(m,a,k+( 1)*shiftxk + ( 2)*shiftyk,
                                                        j+( 1)*shiftxj + ( 2)*shiftyj,
                                                        i+( 1)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * (  -3./4.  ) * quant(m,a,k+(-1)*shiftxk + (-1)*shiftyk,
                                                        j+(-1)*shiftxj + (-1)*shiftyj,
                                                        i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -3./4.  ) * (  3./4.   ) * quant(m,a,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                        j+(-1)*shiftxj + ( 1)*shiftyj,
                                                        i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  3./4.   ) * (  -3./4.  ) * quant(m,a,k+( 1)*shiftxk + (-1)*shiftyk,
                                                        j+( 1)*shiftxj + (-1)*shiftyj,
                                                        i+( 1)*shiftxi + (-1)*shiftyi)
              + (  3./4.   ) * (  3./4.   ) * quant(m,a,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                        j+( 1)*shiftxj + ( 1)*shiftyj,
                                                        i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  }
  return out*idx[dirx]*idx[diry];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative 2D tensor
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dxy(int const dirx, int const diry,
        const Real idx[], TYPE &quant,
        int const m, int const a, int const b,
        int const k, int const j, int const i) {
  int const shiftxk = dirx==2;
  int const shiftxj = dirx==1;
  int const shiftxi = dirx==0;
  int const shiftyk = diry==2;
  int const shiftyj = diry==1;
  int const shiftyi = diry==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (
            + (
              + (  -1./2.  ) * (  -1./2.  ) * quant(m,a,b,k+(-1)*shiftxk + (-1)*shiftyk,
                                                          j+(-1)*shiftxj + (-1)*shiftyj,
                                                          i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -1./2.  ) * (  1./2.   ) * quant(m,a,b,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                          j+(-1)*shiftxj + ( 1)*shiftyj,
                                                          i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  1./2.   ) * (  -1./2.  ) * quant(m,a,b,k+( 1)*shiftxk + (-1)*shiftyk,
                                                          j+( 1)*shiftxj + (-1)*shiftyj,
                                                          i+( 1)*shiftxi + (-1)*shiftyi)
              + (  1./2.   ) * (  1./2.   ) * quant(m,a,b,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                          j+( 1)*shiftxj + ( 1)*shiftyj,
                                                          i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  } else if constexpr ( NGHOST == 3 ) {
    out = + (
            + (
              + (  1./12.  ) * (  1./12.  ) * quant(m,a,b,k+(-2)*shiftxk + (-2)*shiftyk,
                                                          j+(-2)*shiftxj + (-2)*shiftyj,
                                                          i+(-2)*shiftxi + (-2)*shiftyi)
              + (  1./12.  ) * ( -1./12.  ) * quant(m,a,b,k+(-2)*shiftxk + ( 2)*shiftyk,
                                                          j+(-2)*shiftxj + ( 2)*shiftyj,
                                                          i+(-2)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + ( -1./12.  ) * (  1./12.  ) * quant(m,a,b,k+( 2)*shiftxk + (-2)*shiftyk,
                                                          j+( 2)*shiftxj + (-2)*shiftyj,
                                                          i+( 2)*shiftxi + (-2)*shiftyi)
              + ( -1./12.  ) * ( -1./12.  ) * quant(m,a,b,k+( 2)*shiftxk + ( 2)*shiftyk,
                                                          j+( 2)*shiftxj + ( 2)*shiftyj,
                                                          i+( 2)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  1./12.  ) * (  -2./3.  ) * quant(m,a,b,k+(-2)*shiftxk + (-1)*shiftyk,
                                                          j+(-2)*shiftxj + (-1)*shiftyj,
                                                          i+(-2)*shiftxi + (-1)*shiftyi)
              + (  1./12.  ) * (  2./3.   ) * quant(m,a,b,k+(-2)*shiftxk + ( 1)*shiftyk,
                                                          j+(-2)*shiftxj + ( 1)*shiftyj,
                                                          i+(-2)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + ( -1./12.  ) * (  -2./3.  ) * quant(m,a,b,k+( 2)*shiftxk + (-1)*shiftyk,
                                                          j+( 2)*shiftxj + (-1)*shiftyj,
                                                          i+( 2)*shiftxi + (-1)*shiftyi)
              + ( -1./12.  ) * (  2./3.   ) * quant(m,a,b,k+( 2)*shiftxk + ( 1)*shiftyk,
                                                          j+( 2)*shiftxj + ( 1)*shiftyj,
                                                          i+( 2)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  -2./3.  ) * (  1./12.  ) * quant(m,a,b,k+(-1)*shiftxk + (-2)*shiftyk,
                                                          j+(-1)*shiftxj + (-2)*shiftyj,
                                                          i+(-1)*shiftxi + (-2)*shiftyi)
              + (  -2./3.  ) * ( -1./12.  ) * quant(m,a,b,k+(-1)*shiftxk + ( 2)*shiftyk,
                                                          j+(-1)*shiftxj + ( 2)*shiftyj,
                                                          i+(-1)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  2./3.   ) * (  1./12.  ) * quant(m,a,b,k+( 1)*shiftxk + (-2)*shiftyk,
                                                          j+( 1)*shiftxj + (-2)*shiftyj,
                                                          i+( 1)*shiftxi + (-2)*shiftyi)
              + (  2./3.   ) * ( -1./12.  ) * quant(m,a,b,k+( 1)*shiftxk + ( 2)*shiftyk,
                                                          j+( 1)*shiftxj + ( 2)*shiftyj,
                                                          i+( 1)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  -2./3.  ) * (  -2./3.  ) * quant(m,a,b,k+(-1)*shiftxk + (-1)*shiftyk,
                                                          j+(-1)*shiftxj + (-1)*shiftyj,
                                                          i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -2./3.  ) * (  2./3.   ) * quant(m,a,b,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                          j+(-1)*shiftxj + ( 1)*shiftyj,
                                                          i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  2./3.   ) * (  -2./3.  ) * quant(m,a,b,k+( 1)*shiftxk + (-1)*shiftyk,
                                                          j+( 1)*shiftxj + (-1)*shiftyj,
                                                          i+( 1)*shiftxi + (-1)*shiftyi)
              + (  2./3.   ) * (  2./3.   ) * quant(m,a,b,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                          j+( 1)*shiftxj + ( 1)*shiftyj,
                                                          i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  } else if constexpr ( NGHOST == 4 ) {
    out = + (
            + (
              + ( -1./60.  ) * ( -1./60.  ) * quant(m,a,b,k+(-3)*shiftxk + (-3)*shiftyk,
                                                          j+(-3)*shiftxj + (-3)*shiftyj,
                                                          i+(-3)*shiftxi + (-3)*shiftyi)
              + ( -1./60.  ) * (  1./60.  ) * quant(m,a,b,k+(-3)*shiftxk + ( 3)*shiftyk,
                                                          j+(-3)*shiftxj + ( 3)*shiftyj,
                                                          i+(-3)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + (  1./60.  ) * ( -1./60.  ) * quant(m,a,b,k+( 3)*shiftxk + (-3)*shiftyk,
                                                          j+( 3)*shiftxj + (-3)*shiftyj,
                                                          i+( 3)*shiftxi + (-3)*shiftyi)
              + (  1./60.  ) * (  1./60.  ) * quant(m,a,b,k+( 3)*shiftxk + ( 3)*shiftyk,
                                                          j+( 3)*shiftxj + ( 3)*shiftyj,
                                                          i+( 3)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + ( -1./60.  ) * (  3./20.  ) * quant(m,a,b,k+(-3)*shiftxk + (-2)*shiftyk,
                                                          j+(-3)*shiftxj + (-2)*shiftyj,
                                                          i+(-3)*shiftxi + (-2)*shiftyi)
              + ( -1./60.  ) * ( -3./20.  ) * quant(m,a,b,k+(-3)*shiftxk + ( 2)*shiftyk,
                                                          j+(-3)*shiftxj + ( 2)*shiftyj,
                                                          i+(-3)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  1./60.  ) * (  3./20.  ) * quant(m,a,b,k+( 3)*shiftxk + (-2)*shiftyk,
                                                          j+( 3)*shiftxj + (-2)*shiftyj,
                                                          i+( 3)*shiftxi + (-2)*shiftyi)
              + (  1./60.  ) * ( -3./20.  ) * quant(m,a,b,k+( 3)*shiftxk + ( 2)*shiftyk,
                                                          j+( 3)*shiftxj + ( 2)*shiftyj,
                                                          i+( 3)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + ( -1./60.  ) * (  -3./4.  ) * quant(m,a,b,k+(-3)*shiftxk + (-1)*shiftyk,
                                                          j+(-3)*shiftxj + (-1)*shiftyj,
                                                          i+(-3)*shiftxi + (-1)*shiftyi)
              + ( -1./60.  ) * (  3./4.   ) * quant(m,a,b,k+(-3)*shiftxk + ( 1)*shiftyk,
                                                          j+(-3)*shiftxj + ( 1)*shiftyj,
                                                          i+(-3)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  1./60.  ) * (  -3./4.  ) * quant(m,a,b,k+( 3)*shiftxk + (-1)*shiftyk,
                                                          j+( 3)*shiftxj + (-1)*shiftyj,
                                                          i+( 3)*shiftxi + (-1)*shiftyi)
              + (  1./60.  ) * (  3./4.   ) * quant(m,a,b,k+( 3)*shiftxk + ( 1)*shiftyk,
                                                          j+( 3)*shiftxj + ( 1)*shiftyj,
                                                          i+( 3)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * ( -1./60.  ) * quant(m,a,b,k+(-2)*shiftxk + (-3)*shiftyk,
                                                          j+(-2)*shiftxj + (-3)*shiftyj,
                                                          i+(-2)*shiftxi + (-3)*shiftyi)
              + (  3./20.  ) * (  1./60.  ) * quant(m,a,b,k+(-2)*shiftxk + ( 3)*shiftyk,
                                                          j+(-2)*shiftxj + ( 3)*shiftyj,
                                                          i+(-2)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + ( -3./20.  ) * ( -1./60.  ) * quant(m,a,b,k+( 2)*shiftxk + (-3)*shiftyk,
                                                          j+( 2)*shiftxj + (-3)*shiftyj,
                                                          i+( 2)*shiftxi + (-3)*shiftyi)
              + ( -3./20.  ) * (  1./60.  ) * quant(m,a,b,k+( 2)*shiftxk + ( 3)*shiftyk,
                                                          j+( 2)*shiftxj + ( 3)*shiftyj,
                                                          i+( 2)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * (  3./20.  ) * quant(m,a,b,k+(-2)*shiftxk + (-2)*shiftyk,
                                                          j+(-2)*shiftxj + (-2)*shiftyj,
                                                          i+(-2)*shiftxi + (-2)*shiftyi)
              + (  3./20.  ) * ( -3./20.  ) * quant(m,a,b,k+(-2)*shiftxk + ( 2)*shiftyk,
                                                          j+(-2)*shiftxj + ( 2)*shiftyj,
                                                          i+(-2)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + ( -3./20.  ) * (  3./20.  ) * quant(m,a,b,k+( 2)*shiftxk + (-2)*shiftyk,
                                                          j+( 2)*shiftxj + (-2)*shiftyj,
                                                          i+( 2)*shiftxi + (-2)*shiftyi)
              + ( -3./20.  ) * ( -3./20.  ) * quant(m,a,b,k+( 2)*shiftxk + ( 2)*shiftyk,
                                                          j+( 2)*shiftxj + ( 2)*shiftyj,
                                                          i+( 2)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  3./20.  ) * (  -3./4.  ) * quant(m,a,b,k+(-2)*shiftxk + (-1)*shiftyk,
                                                          j+(-2)*shiftxj + (-1)*shiftyj,
                                                          i+(-2)*shiftxi + (-1)*shiftyi)
              + (  3./20.  ) * (  3./4.   ) * quant(m,a,b,k+(-2)*shiftxk + ( 1)*shiftyk,
                                                          j+(-2)*shiftxj + ( 1)*shiftyj,
                                                          i+(-2)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + ( -3./20.  ) * (  -3./4.  ) * quant(m,a,b,k+( 2)*shiftxk + (-1)*shiftyk,
                                                          j+( 2)*shiftxj + (-1)*shiftyj,
                                                          i+( 2)*shiftxi + (-1)*shiftyi)
              + ( -3./20.  ) * (  3./4.   ) * quant(m,a,b,k+( 2)*shiftxk + ( 1)*shiftyk,
                                                          j+( 2)*shiftxj + ( 1)*shiftyj,
                                                          i+( 2)*shiftxi + ( 1)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * ( -1./60.  ) * quant(m,a,b,k+(-1)*shiftxk + (-3)*shiftyk,
                                                          j+(-1)*shiftxj + (-3)*shiftyj,
                                                          i+(-1)*shiftxi + (-3)*shiftyi)
              + (  -3./4.  ) * (  1./60.  ) * quant(m,a,b,k+(-1)*shiftxk + ( 3)*shiftyk,
                                                          j+(-1)*shiftxj + ( 3)*shiftyj,
                                                          i+(-1)*shiftxi + ( 3)*shiftyi)
              )
            + (
              + (  3./4.   ) * ( -1./60.  ) * quant(m,a,b,k+( 1)*shiftxk + (-3)*shiftyk,
                                                          j+( 1)*shiftxj + (-3)*shiftyj,
                                                          i+( 1)*shiftxi + (-3)*shiftyi)
              + (  3./4.   ) * (  1./60.  ) * quant(m,a,b,k+( 1)*shiftxk + ( 3)*shiftyk,
                                                          j+( 1)*shiftxj + ( 3)*shiftyj,
                                                          i+( 1)*shiftxi + ( 3)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * (  3./20.  ) * quant(m,a,b,k+(-1)*shiftxk + (-2)*shiftyk,
                                                          j+(-1)*shiftxj + (-2)*shiftyj,
                                                          i+(-1)*shiftxi + (-2)*shiftyi)
              + (  -3./4.  ) * ( -3./20.  ) * quant(m,a,b,k+(-1)*shiftxk + ( 2)*shiftyk,
                                                          j+(-1)*shiftxj + ( 2)*shiftyj,
                                                          i+(-1)*shiftxi + ( 2)*shiftyi)
              )
            + (
              + (  3./4.   ) * (  3./20.  ) * quant(m,a,b,k+( 1)*shiftxk + (-2)*shiftyk,
                                                          j+( 1)*shiftxj + (-2)*shiftyj,
                                                          i+( 1)*shiftxi + (-2)*shiftyi)
              + (  3./4.   ) * ( -3./20.  ) * quant(m,a,b,k+( 1)*shiftxk + ( 2)*shiftyk,
                                                          j+( 1)*shiftxj + ( 2)*shiftyj,
                                                          i+( 1)*shiftxi + ( 2)*shiftyi)
              )
            )
          + (
            + (
              + (  -3./4.  ) * (  -3./4.  ) * quant(m,a,b,k+(-1)*shiftxk + (-1)*shiftyk,
                                                          j+(-1)*shiftxj + (-1)*shiftyj,
                                                          i+(-1)*shiftxi + (-1)*shiftyi)
              + (  -3./4.  ) * (  3./4.   ) * quant(m,a,b,k+(-1)*shiftxk + ( 1)*shiftyk,
                                                          j+(-1)*shiftxj + ( 1)*shiftyj,
                                                          i+(-1)*shiftxi + ( 1)*shiftyi)
              )
            + (
              + (  3./4.   ) * (  -3./4.  ) * quant(m,a,b,k+( 1)*shiftxk + (-1)*shiftyk,
                                                          j+( 1)*shiftxj + (-1)*shiftyj,
                                                          i+( 1)*shiftxi + (-1)*shiftyi)
              + (  3./4.   ) * (  3./4.   ) * quant(m,a,b,k+( 1)*shiftxk + ( 1)*shiftyk,
                                                          j+( 1)*shiftxj + ( 1)*shiftyj,
                                                          i+( 1)*shiftxi + ( 1)*shiftyi)
              )
            );
  }
  return out*idx[dirx]*idx[diry];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st advective derivative scalar
template <int NGHOST, typename TYPE1, typename TYPE2>
KOKKOS_INLINE_FUNCTION
Real Lx(int const dir,
        const Real idx[], const TYPE1 &vx,
                        const TYPE2 &quant,
        int const m, int const a,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real dl, dr;
  if constexpr ( NGHOST == 2 ) {
    dl = +0.5    * quant(m,k+(-2)*shiftk,
                                       j+(-2)*shiftj,
                                       i+(-2)*shifti)
                     -2.0    * quant(m,k+(-1)*shiftk,
                                       j+(-1)*shiftj,
                                       i+(-1)*shifti)
                     +1.5    * quant(m,k,
                                       j,
                                       i);
    dr = -0.5    * quant(m,k+(2)*shiftk,
                                       j+(2)*shiftj,
                                       i+(2)*shifti)
                     +2.0    * quant(m,k+(1)*shiftk,
                                       j+(1)*shiftj,
                                       i+(1)*shifti)
                     -1.5    * quant(m,k,
                                       j,
                                       i);
  } else if constexpr ( NGHOST == 3 ) {
    dl = -1./12.   * quant(m,k+(-3)*shiftk,
                                       j+(-3)*shiftj,
                                       i+(-3)*shifti)
                   +6./12.   * quant(m,k+(-2)*shiftk,
                                       j+(-2)*shiftj,
                                       i+(-2)*shifti)
                   -18./12.  * quant(m,k+(-1)*shiftk,
                                       j+(-1)*shiftj,
                                       i+(-1)*shifti)
                   +10./12.  * quant(m,k,
                                       j,
                                       i)
                   +3./12.   * quant(m,k+(1)*shiftk,
                                       j+(1)*shiftj,
                                       i+(1)*shifti);
    dr = +1./12.   * quant(m,k+(3)*shiftk,
                                       j+(3)*shiftj,
                                       i+(3)*shifti)
                   -6./12.   * quant(m,k+(2)*shiftk,
                                       j+(2)*shiftj,
                                       i+(2)*shifti)
                   +18./12.  * quant(m,k+(1)*shiftk,
                                       j+(1)*shiftj,
                                       i+(1)*shifti)
                   -10./12.  * quant(m,k,
                                       j,
                                       i)
                   -3./12.   * quant(m,k+(-1)*shiftk,
                                       j+(-1)*shiftj,
                                       i+(-1)*shifti);
  } else if constexpr ( NGHOST == 4 ) {
    dl = +1./60.   * quant(m,k+(-4)*shiftk,
                                       j+(-4)*shiftj,
                                       i+(-4)*shifti)
                   -2./15.   * quant(m,k+(-3)*shiftk,
                                       j+(-3)*shiftj,
                                       i+(-3)*shifti)
                    +1./2.   * quant(m,k+(-2)*shiftk,
                                       j+(-2)*shiftj,
                                       i+(-2)*shifti)
                    -4./3.   * quant(m,k+(-1)*shiftk,
                                       j+(-1)*shiftj,
                                       i+(-1)*shifti)
                   +7./12.   * quant(m,k,
                                       j,
                                       i)
                    +2./5.   * quant(m,k+(1)*shiftk,
                                       j+(1)*shiftj,
                                       i+(1)*shifti)
                   -1./30.   * quant(m,k+(2)*shiftk,
                                       j+(2)*shiftj,
                                       i+(2)*shifti);
    dr = -1./60.   * quant(m,k+(4)*shiftk,
                                       j+(4)*shiftj,
                                       i+(4)*shifti)
                   +2./15.   * quant(m,k+(3)*shiftk,
                                       j+(3)*shiftj,
                                       i+(3)*shifti)
                    -1./2.   * quant(m,k+(2)*shiftk,
                                       j+(2)*shiftj,
                                       i+(2)*shifti)
                    +4./3.   * quant(m,k+(1)*shiftk,
                                       j+(1)*shiftj,
                                       i+(1)*shifti)
                   -7./12.   * quant(m,k,
                                       j,
                                       i)
                    -2./5.   * quant(m,k+(-1)*shiftk,
                                       j+(-1)*shiftj,
                                       i+(-1)*shifti)
                   +1./30.   * quant(m,k+(-2)*shiftk,
                                       j+(-2)*shiftj,
                                       i+(-2)*shifti);
  }
  return ((vx(m,a,k,j,i) < 0) ? (vx(m,a,k,j,i) * dl) : (vx(m,a,k,j,i) * dr)) * idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st advective derivative vector
template <int NGHOST, typename TYPE1, typename TYPE2>
KOKKOS_INLINE_FUNCTION
Real Lx(int const dir,
        const Real idx[], const TYPE1 &vx,
                        const TYPE2 &quant,
        int const m, int const a, int const b,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real dl, dr;
  if constexpr ( NGHOST == 2 ) {
    dl = +0.5    * quant(m,b,k+(-2)*shiftk,
                                         j+(-2)*shiftj,
                                         i+(-2)*shifti)
                     -2.0    * quant(m,b,k+(-1)*shiftk,
                                         j+(-1)*shiftj,
                                         i+(-1)*shifti)
                     +1.5    * quant(m,b,k,
                                         j,
                                         i);
    dr = -0.5    * quant(m,b,k+(2)*shiftk,
                                         j+(2)*shiftj,
                                         i+(2)*shifti)
                     +2.0    * quant(m,b,k+(1)*shiftk,
                                         j+(1)*shiftj,
                                         i+(1)*shifti)
                     -1.5    * quant(m,b,k,
                                         j,
                                         i);
  } else if constexpr ( NGHOST == 3 ) {
    dl = -1./12.   * quant(m,b,k+(-3)*shiftk,
                                         j+(-3)*shiftj,
                                         i+(-3)*shifti)
                   +6./12.   * quant(m,b,k+(-2)*shiftk,
                                         j+(-2)*shiftj,
                                         i+(-2)*shifti)
                   -18./12.  * quant(m,b,k+(-1)*shiftk,
                                         j+(-1)*shiftj,
                                         i+(-1)*shifti)
                   +10./12.  * quant(m,b,k,
                                         j,
                                         i)
                   +3./12.   * quant(m,b,k+(1)*shiftk,
                                         j+(1)*shiftj,
                                         i+(1)*shifti);
    dr = +1./12.   * quant(m,b,k+(3)*shiftk,
                                         j+(3)*shiftj,
                                         i+(3)*shifti)
                   -6./12.   * quant(m,b,k+(2)*shiftk,
                                         j+(2)*shiftj,
                                         i+(2)*shifti)
                   +18./12.  * quant(m,b,k+(1)*shiftk,
                                         j+(1)*shiftj,
                                         i+(1)*shifti)
                   -10./12.  * quant(m,b,k,
                                         j,
                                         i)
                   -3./12.   * quant(m,b,k+(-1)*shiftk,
                                         j+(-1)*shiftj,
                                         i+(-1)*shifti);
  } else if constexpr ( NGHOST == 4 ) {
    dl = +1./60.   * quant(m,b,k+(-4)*shiftk,
                                         j+(-4)*shiftj,
                                         i+(-4)*shifti)
                   -2./15.   * quant(m,b,k+(-3)*shiftk,
                                         j+(-3)*shiftj,
                                         i+(-3)*shifti)
                    +1./2.   * quant(m,b,k+(-2)*shiftk,
                                         j+(-2)*shiftj,
                                         i+(-2)*shifti)
                    -4./3.   * quant(m,b,k+(-1)*shiftk,
                                         j+(-1)*shiftj,
                                         i+(-1)*shifti)
                   +7./12.   * quant(m,b,k,
                                         j,
                                         i)
                    +2./5.   * quant(m,b,k+(1)*shiftk,
                                         j+(1)*shiftj,
                                         i+(1)*shifti)
                   -1./30.   * quant(m,b,k+(2)*shiftk,
                                         j+(2)*shiftj,
                                         i+(2)*shifti);
    dr = -1./60.   * quant(m,b,k+(4)*shiftk,
                                         j+(4)*shiftj,
                                         i+(4)*shifti)
                   +2./15.   * quant(m,b,k+(3)*shiftk,
                                         j+(3)*shiftj,
                                         i+(3)*shifti)
                    -1./2.   * quant(m,b,k+(2)*shiftk,
                                         j+(2)*shiftj,
                                         i+(2)*shifti)
                    +4./3.   * quant(m,b,k+(1)*shiftk,
                                         j+(1)*shiftj,
                                         i+(1)*shifti)
                   -7./12.   * quant(m,b,k,
                                         j,
                                         i)
                    -2./5.   * quant(m,b,k+(-1)*shiftk,
                                         j+(-1)*shiftj,
                                         i+(-1)*shifti)
                   +1./30.   * quant(m,b,k+(-2)*shiftk,
                                         j+(-2)*shiftj,
                                         i+(-2)*shifti);
  }
  return ((vx(m,a,k,j,i) < 0) ? (vx(m,a,k,j,i) * dl) : (vx(m,a,k,j,i) * dr)) * idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st advective derivative 2D tensor
template <int NGHOST, typename TYPE1, typename TYPE2>
KOKKOS_INLINE_FUNCTION
Real Lx(int const dir,
        const Real idx[], const TYPE1 &vx,
                        const TYPE2 &quant,
        int const m, int const a, int const b, int const c,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real dl, dr;
  if constexpr ( NGHOST == 2 ) {
    dl = +0.5    * quant(m,b,c,k+(-2)*shiftk,
                                           j+(-2)*shiftj,
                                           i+(-2)*shifti)
                     -2.0    * quant(m,b,c,k+(-1)*shiftk,
                                           j+(-1)*shiftj,
                                           i+(-1)*shifti)
                     +1.5    * quant(m,b,c,k,
                                           j,
                                           i);
    dr = -0.5    * quant(m,b,c,k+(2)*shiftk,
                                           j+(2)*shiftj,
                                           i+(2)*shifti)
                     +2.0    * quant(m,b,c,k+(1)*shiftk,
                                           j+(1)*shiftj,
                                           i+(1)*shifti)
                     -1.5    * quant(m,b,c,k,
                                           j,
                                           i);
  } else if constexpr ( NGHOST == 3 ) {
    dl = -1./12.   * quant(m,b,c,k+(-3)*shiftk,
                                           j+(-3)*shiftj,
                                           i+(-3)*shifti)
                   +6./12.   * quant(m,b,c,k+(-2)*shiftk,
                                           j+(-2)*shiftj,
                                           i+(-2)*shifti)
                   -18./12.  * quant(m,b,c,k+(-1)*shiftk,
                                           j+(-1)*shiftj,
                                           i+(-1)*shifti)
                   +10./12.  * quant(m,b,c,k,
                                           j,
                                           i)
                   +3./12.   * quant(m,b,c,k+(1)*shiftk,
                                           j+(1)*shiftj,
                                           i+(1)*shifti);
    dr = +1./12.   * quant(m,b,c,k+(3)*shiftk,
                                           j+(3)*shiftj,
                                           i+(3)*shifti)
                   -6./12.   * quant(m,b,c,k+(2)*shiftk,
                                           j+(2)*shiftj,
                                           i+(2)*shifti)
                   +18./12.  * quant(m,b,c,k+(1)*shiftk,
                                           j+(1)*shiftj,
                                           i+(1)*shifti)
                   -10./12.  * quant(m,b,c,k,
                                           j,
                                           i)
                   -3./12.   * quant(m,b,c,k+(-1)*shiftk,
                                           j+(-1)*shiftj,
                                           i+(-1)*shifti);
  } else if constexpr ( NGHOST == 4 ) {
    dl = +1./60.   * quant(m,b,c,k+(-4)*shiftk,
                                           j+(-4)*shiftj,
                                           i+(-4)*shifti)
                   -2./15.   * quant(m,b,c,k+(-3)*shiftk,
                                           j+(-3)*shiftj,
                                           i+(-3)*shifti)
                    +1./2.   * quant(m,b,c,k+(-2)*shiftk,
                                           j+(-2)*shiftj,
                                           i+(-2)*shifti)
                    -4./3.   * quant(m,b,c,k+(-1)*shiftk,
                                           j+(-1)*shiftj,
                                           i+(-1)*shifti)
                   +7./12.   * quant(m,b,c,k,
                                           j,
                                           i)
                    +2./5.   * quant(m,b,c,k+(1)*shiftk,
                                           j+(1)*shiftj,
                                           i+(1)*shifti)
                   -1./30.   * quant(m,b,c,k+(2)*shiftk,
                                           j+(2)*shiftj,
                                           i+(2)*shifti);
    dr = -1./60.   * quant(m,b,c,k+(4)*shiftk,
                                           j+(4)*shiftj,
                                           i+(4)*shifti)
                   +2./15.   * quant(m,b,c,k+(3)*shiftk,
                                           j+(3)*shiftj,
                                           i+(3)*shifti)
                    -1./2.   * quant(m,b,c,k+(2)*shiftk,
                                           j+(2)*shiftj,
                                           i+(2)*shifti)
                    +4./3.   * quant(m,b,c,k+(1)*shiftk,
                                           j+(1)*shiftj,
                                           i+(1)*shifti)
                   -7./12.   * quant(m,b,c,k,
                                           j,
                                           i)
                    -2./5.   * quant(m,b,c,k+(-1)*shiftk,
                                           j+(-1)*shiftj,
                                           i+(-1)*shifti)
                   +1./30.   * quant(m,b,c,k+(-2)*shiftk,
                                           j+(-2)*shiftj,
                                           i+(-2)*shifti);
  }
  return ((vx(m,a,k,j,i) < 0) ? (vx(m,a,k,j,i) * dl) : (vx(m,a,k,j,i) * dr)) * idx[dir];
}


// Reminder: this code has been generated with py/write_FD.py,
// please do modifications there.// 1st derivative vector
template <int NGHOST, typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Diss(int const dir,
        const Real idx[], TYPE &quant,
        int const m, int const a,
        int const k, int const j, int const i) {
  int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;
  Real out;
  if constexpr ( NGHOST == 2 ) {
    out = + (   +1.     * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
                +1.     * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (   -4.     * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
                -4.     * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti))
                +6.     * quant(m,a,k,
                                    j,
                                    i);
  } else if constexpr ( NGHOST == 3 ) {
    out = + (   +1.     * quant(m,a,k+(-3)*shiftk,
                                    j+(-3)*shiftj,
                                    i+(-3)*shifti)
                +1.     * quant(m,a,k+( 3)*shiftk,
                                    j+( 3)*shiftj,
                                    i+( 3)*shifti))
          + (   -6.     * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
                -6.     * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (   +15.    * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
                +15.    * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti))
                -20.    * quant(m,a,k,
                                    j,
                                    i);
  } else if constexpr ( NGHOST == 4 ) {
    out = + (   +1.     * quant(m,a,k+(-4)*shiftk,
                                    j+(-4)*shiftj,
                                    i+(-4)*shifti)
                +1.     * quant(m,a,k+( 4)*shiftk,
                                    j+( 4)*shiftj,
                                    i+( 4)*shifti))
          + (   -8.     * quant(m,a,k+(-3)*shiftk,
                                    j+(-3)*shiftj,
                                    i+(-3)*shifti)
                -8.     * quant(m,a,k+( 3)*shiftk,
                                    j+( 3)*shiftj,
                                    i+( 3)*shifti))
          + (   +28.    * quant(m,a,k+(-2)*shiftk,
                                    j+(-2)*shiftj,
                                    i+(-2)*shifti)
                +28.    * quant(m,a,k+( 2)*shiftk,
                                    j+( 2)*shiftj,
                                    i+( 2)*shifti))
          + (   -56.    * quant(m,a,k+(-1)*shiftk,
                                    j+(-1)*shiftj,
                                    i+(-1)*shifti)
                -56.    * quant(m,a,k+( 1)*shiftk,
                                    j+( 1)*shiftj,
                                    i+( 1)*shifti))
                +70.    * quant(m,a,k,
                                    j,
                                    i);
  }
  return out*idx[dir];
}

#endif // UTILS_FINITE_DIFF_HPP_
