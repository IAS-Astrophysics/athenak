//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advect.cpp
//  \brief Riemann solver for pure advection problems (v = constant).  Simply computes the
//  upwind flux of each variable.  Can only be used for isothermal EOS.

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Advection
//  \brief An advection Riemann solver for hydrodynamics (isothermal)

KOKKOS_INLINE_FUNCTION
void Advect(TeamMember_t const &member, const EOS_Data eos,
     const int m, const int k, const int j,  const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     DvceArray5D<Real> flx)
{
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;

  par_for_inner(member, il, iu, [&](const int i)
  {
    //  Compute upwind fluxes
    if (wl(ivx,i) >= 0.0) {
      Real mxl = wl(IDN,i)*wl(ivx,i);
      flx(m,IDN,k,j,i) = mxl;
//      flx(m,ivx,k,j,i) = mxl*wl(ivx,i);
//      flx(m,ivy,k,j,i) = mxl*wl(ivy,i);
//      flx(m,ivz,k,j,i) = mxl*wl(ivz,i);
    } else {
      Real mxr = wr(IDN,i)*wr(ivx,i);
      flx(m,IDN,k,j,i) = mxr;
//      flx(m,ivx,k,j,i) = mxr*wr(ivx,i);
//      flx(m,ivy,k,j,i) = mxr*wr(ivy,i);
//      flx(m,ivz,k,j,i) = mxr*wr(ivz,i);
    }
  });

  return;
}

} // namespace hydro
