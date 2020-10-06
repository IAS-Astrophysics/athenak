//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dc.cpp
//  \brief piecewise constant (donor cell) reconstruction implemented as inline function

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn DonorCellX1()
//  \brief For each cell-centered value q(i), returns ql(i+1) and qr(i) over il to iu.
//  Therefore range of indices for which BOTH L/R states returned is il+1 to il-1
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]

KOKKOS_INLINE_FUNCTION
void DonorCell(TeamMember_t const &member, const int il, const int iu,
               const AthenaArray2DSlice<Real> &q,
               AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr)
{
  int nvar = q.extent_int(0);
  for (int n=0; n<nvar; ++n) {
    par_for_inner(member, il, iu, [&](const int i)
    {
      ql(n,i+1) = q(n,i);
      qr(n,i  ) = q(n,i);
    });
  }
  return;
}
