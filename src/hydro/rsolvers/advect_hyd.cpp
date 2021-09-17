//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect_hyd.cpp
//  \brief Riemann solver for pure advection problems (v = constant).  Simply computes the
//  upwind flux of each variable.  Can only be used for isothermal EOS.

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Advection
//  \brief An advection Riemann solver for hydrodynamics (isothermal)

KOKKOS_INLINE_FUNCTION
void Advect(TeamMember_t const &member, const EOS_Data eos, const CoordData &coord, 
     const int m, const int k, const int j,  const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr, DvceArray5D<Real> flx)
{
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //  Compute upwind fluxes
    if (wl(ivx,i) >= 0.0) {
      flx(m,IDN,k,j,i) = wl(IDN,i)*wl(ivx,i);
      flx(m,ivx,k,j,i) = 0.0;
      flx(m,ivy,k,j,i) = 0.0;
      flx(m,ivz,k,j,i) = 0.0;
    } else {
      flx(m,IDN,k,j,i) = wr(IDN,i)*wr(ivx,i);
      flx(m,ivx,k,j,i) = 0.0;
      flx(m,ivy,k,j,i) = 0.0;
      flx(m,ivz,k,j,i) = 0.0;
    }
  });

  return;
}

} // namespace hydro
