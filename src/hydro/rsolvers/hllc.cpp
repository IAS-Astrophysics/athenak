// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hllc.cpp
//  \brief HLLC Riemann solver for hydrodynamics, an extension of the HLLE fluxes to
//  include the contact wave.  Only works for adiabatic hydrodynamics.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.
//
// - P. Batten, N. Clarke, C. Lambert, and D. M. Causon, "On the Choice of Wavespeeds
//   for the HLLC Riemann Solver", SIAM J. Sci. & Stat. Comp. 18, 6, 1553-1570, (1997).

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void HLLC
//! \brief The HLLC Riemann solver for adiabatic hydrodynamics (use HLLE for isothermal)

KOKKOS_INLINE_FUNCTION
void HLLC(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j, const int il, const int iu,
     const int ivx, const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     DvceArray5D<Real> flx)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[5],wri[5];
  Real fl[5],fr[5];

  Real igm1 = 1.0/((eos.gamma) - 1.0);
  Real alpha = ((eos.gamma) + 1.0)/sqrt(2.0*(eos.gamma));

  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    wli[IPR]=wl(IPR,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    wri[IPR]=wr(IPR,i);

    //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

    // define 6 registers used below
    Real qa,qb,qc,qd,qe,qf;
    qa = eos.SoundSpeed(wli[IPR],wli[IDN]);
    qb = eos.SoundSpeed(wri[IPR],wri[IDN]);
    Real el = wli[IPR]*igm1 + 0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ]));
    Real er = wri[IPR]*igm1 + 0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ]));
    qc = 0.25*(wli[IDN] + wri[IDN])*(qa + qb);  // average density * average sound speed
    qd = 0.5 * (wli[IPR] + wri[IPR] + (wli[IVX]-wri[IVX]) * qc);  // P_mid

    //--- Step 3.  Compute sound speed in L,R

    qe = (qd <= wli[IPR]) ? 1.0 : (1.0 + alpha * ((qd / wli[IPR]) - 1.0));
    qf = (qd <= wri[IPR]) ? 1.0 : (1.0 + alpha * ((qd / wri[IPR]) - 1.0));

    //--- Step 4.  Compute the max/min wave speeds based on L/R

    qc = wli[IVX] - qa*qe;  // ql
    qd = wri[IVX] + qb*qf;  // qr

    qa = qd > 0.0 ? qd : 0.0;
    qb = qc < 0.0 ? qc : 0.0;

    //--- Step 5. Compute the contact wave speed and pressure

    qe = wli[IVX] - qc;
    qf = wri[IVX] - qd;

    qc = wli[IPR] + qe*wli[IDN]*wli[IVX];  // tl
    qd = wri[IPR] + qf*wri[IDN]*wri[IVX];  // tr

    Real ml =   wli[IDN]*qe;
    Real mr = -(wri[IDN]*qf);

    // Determine the contact wave speed...
    Real am = (qc - qd)/(ml + mr);
    // ...and the pressure at the contact surface
    Real cp = (ml*qd + mr*qc)/(ml + mr);
    cp = cp > 0.0 ? cp : 0.0;

    //--- Step 6. Compute L/R fluxes along the line bm (qb), bp (qa)

    qe = wli[IDN]*(wli[IVX] - qb);
    qf = wri[IDN]*(wri[IVX] - qa);

    fl[IDN] = qe;
    fr[IDN] = qf;

    fl[IVX] = qe*wli[IVX] + wli[IPR];
    fr[IVX] = qf*wri[IVX] + wri[IPR];

    fl[IVY] = qe*wli[IVY];
    fr[IVY] = qf*wri[IVY];

    fl[IVZ] = qe*wli[IVZ];
    fr[IVZ] = qf*wri[IVZ];

    fl[IEN] = el*(wli[IVX] - qb) + wli[IPR]*wli[IVX];
    fr[IEN] = er*(wri[IVX] - qa) + wri[IPR]*wri[IVX];

    //--- Step 8. Compute flux weights or scales

    if (am >= 0.0) {
      qc =  am/(am - qb);
      qd = 0.0;
      qe = -qb/(am - qb);
    } else {
      qc =  0.0;
      qd = -am/(qa - am);
      qe =  qa/(qa - am);
    }

    //--- Step 9. Compute the HLLC flux at interface, including weighted contribution
    // of the flux along the contact

    flx(m,IDN,k,j,i) = qc*fl[IDN] + qd*fr[IDN];
    flx(m,ivx,k,j,i) = qc*fl[IVX] + qd*fr[IVX] + qe*cp;
    flx(m,ivy,k,j,i) = qc*fl[IVY] + qd*fr[IVY];
    flx(m,ivz,k,j,i) = qc*fl[IVZ] + qd*fr[IVZ];
    flx(m,IEN,k,j,i) = qc*fl[IEN] + qd*fr[IEN] + qe*cp*am;
  });
  return;
}

} // namespace hydro
