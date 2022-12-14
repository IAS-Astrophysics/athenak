#ifndef MESH_PROLONG_HPP_
#define MESH_PROLONG_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolong.hpp
//! \brief prolongation operators for cell-centered and face-centered variables,
//! implemented as inline functions so they can be used both in Bval and AMR functions.

//----------------------------------------------------------------------------------------
//! \fn ProlongCC2()
//! \brief

KOKKOS_INLINE_FUNCTION
void ProlongCC2(const int m, const int v, const int k, const int j, const int i,
                const int finek, const int finej, const int finei,
                const bool multi_d, const bool three_d,
                const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  // calculate x1-gradient using the min-mod limiter
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  // calculate x2-gradient using the min-mod limiter
  Real dvar2 = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // calculate x1-gradient using the min-mod limiter
  Real dvar3 = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // interpolate to the finer grid
  a(m,v,finek,finej,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(m,v,finek,finej,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(m,v,finek,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(m,v,finek,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(m,v,finek+1,finej  ,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(m,v,finek+1,finej  ,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(m,v,finek+1,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(m,v,finek+1,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}

#endif // MESH_PROLONG_HPP_
