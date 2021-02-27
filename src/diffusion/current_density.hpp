//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current_density.hpp
//  \brief Inlined function to compute current density in 1-D pencils in i-direction

#include "athena.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn CurrentDensity()
//  \brief Calculates the three components of the current density

KOKKOS_INLINE_FUNCTION
void CurrentDensity(TeamMember_t const &member, const int m, const int k, const int j,
     const int il, const int iu, const DvceFaceFld4D<Real> &b, const MeshBlockSize &size,
     ScrArray1D<Real> &j1, ScrArray1D<Real> &j2, ScrArray1D<Real> &j3) 
{
  par_for_inner(member, il, iu, [&](const int i)
  {
    j1(i) = 0.0;
    j2(i) = -(b.x3f(m,k,j,i) - b.x3f(m,k,j,i-1))/size.dx1.d_view(m);
    j3(i) =  (b.x2f(m,k,j,i) - b.x2f(m,k,j,i-1))/size.dx1.d_view(m);
  });
  member.team_barrier();

  if (b.x1f.extent_int(2) > 1) {  // proxy for nx2gt1: 2D problems
    par_for_inner(member, il, iu, [&](const int i)
    {
      j1(i) += (b.x3f(m,k,j,i) - b.x3f(m,k,j-1,i))/size.dx2.d_view(m);
      j3(i) -= (b.x1f(m,k,j,i) - b.x1f(m,k,j-1,i))/size.dx2.d_view(m);
    });
    member.team_barrier();
  }

  if (b.x1f.extent_int(1) > 1) {  // proxy for nx3gt1: 3D problems
    par_for_inner(member, il, iu, [&](const int i)
    {
      j1(i) -= (b.x2f(m,k,j,i) - b.x2f(m,k-1,j,i))/size.dx3.d_view(m);
      j2(i) += (b.x1f(m,k,j,i) - b.x1f(m,k-1,j,i))/size.dx3.d_view(m);
    });
  }
  return;
}
