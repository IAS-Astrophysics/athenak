#ifndef HYDRO_RSOLVERS_HLLC_HYD_HPP_
#define HYDRO_RSOLVERS_HLLC_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hllc_hyd.hpp
//! \brief HLLC Riemann solver for hydrodynamics.  Reads L/R primitives from the global
//! per-face buffers and writes a single flux entry.  Ideal gas EOS only.

#include <cmath>

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn HLLC<ivx>()
//! \brief Compute the HLLC flux at face (m,k,j,i) for direction ivx.  Only valid for
//! ideal gas EOS (caller responsibility).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLC(const EOS_Data &eos,
          const int m, const int k, const int j, const int i,
          const int is, const int js, const int ks,
          const DvceArray5D<Real> &wl,
          const DvceArray5D<Real> &wr,
          const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;

  const Real gm1   = eos.gamma - 1.0;
  const Real igm1  = 1.0 / gm1;
  const Real alpha = ((eos.gamma) + 1.0) / (2.0 * (eos.gamma));


  // L/R primitives at face
  Real wl_idn = wl(m, IDN, k, j, i);
  Real wl_ivx = wl(m, ivx, k, j, i);
  Real wl_ivy = wl(m, ivy, k, j, i);
  Real wl_ivz = wl(m, ivz, k, j, i);

  Real wr_idn = wr(m, IDN, k, j, i);
  Real wr_ivx = wr(m, ivx, k, j, i);
  Real wr_ivy = wr(m, ivy, k, j, i);
  Real wr_ivz = wr(m, ivz, k, j, i);

  Real wl_ipr = eos.IdealGasPressure(wl(m, IEN, k, j, i));
  Real wr_ipr = eos.IdealGasPressure(wr(m, IEN, k, j, i));

  // PVRS middle-state estimates (Toro 10.5.2)
  Real qa, qb, qc, qd, qe, qf;
  qa = eos.IdealHydroSoundSpeed(wl_idn, wl_ipr);
  qb = eos.IdealHydroSoundSpeed(wr_idn, wr_ipr);
  Real el = wl_ipr * igm1 + 0.5 * wl_idn * (SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
  Real er = wr_ipr * igm1 + 0.5 * wr_idn * (SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
  qc = 0.25 * (wl_idn + wr_idn) * (qa + qb);
  qd = 0.5 * (wl_ipr + wr_ipr + (wl_ivx - wr_ivx) * qc);  // P_mid

  qe = (qd <= wl_ipr) ? 1.0 : sqrt(1.0 + alpha * ((qd / wl_ipr) - 1.0));
  qf = (qd <= wr_ipr) ? 1.0 : sqrt(1.0 + alpha * ((qd / wr_ipr) - 1.0));

  // Max/min wave speeds
  qc = wl_ivx - qa * qe;  // al
  qd = wr_ivx + qb * qf;  // ar

  qa = qd > 0.0 ? qd : 1.0e-20;   // bp
  qb = qc < 0.0 ? qc : -1.0e-20;  // bm

  // Contact wave speed and pressure
  qe = wl_ivx - qc;  // vxl
  qf = wr_ivx - qd;  // vxr

  qc = wl_ipr + qe * wl_idn * wl_ivx;  // tl
  qd = wr_ipr + qf * wr_idn * wr_ivx;  // tr

  Real ml =  wl_idn * qe;
  Real mr = -(wr_idn * qf);

  Real am = (qc - qd) / (ml + mr);
  Real cp = (ml * qd + mr * qc) / (ml + mr);
  cp = cp > 0.0 ? cp : 0.0;

  // L/R fluxes along bm (qb) and bp (qa)
  qe = wl_idn * (wl_ivx - qb);
  qf = wr_idn * (wr_ivx - qa);

  Real fl_d  = qe;
  Real fr_d  = qf;
  Real fl_mx = qe * wl_ivx + wl_ipr;
  Real fr_mx = qf * wr_ivx + wr_ipr;
  Real fl_my = qe * wl_ivy;
  Real fr_my = qf * wr_ivy;
  Real fl_mz = qe * wl_ivz;
  Real fr_mz = qf * wr_ivz;
  Real fl_e  = el * (wl_ivx - qb) + wl_ipr * wl_ivx;
  Real fr_e  = er * (wr_ivx - qa) + wr_ipr * wr_ivx;

  // Flux weights
  if (am >= 0.0) {
    qc =  am / (am - qb);
    qd = 0.0;
    qe = -qb / (am - qb);
  } else {
    qc =  0.0;
    qd = -am / (qa - am);
    qe =  qa / (qa - am);
  }

  // HLLC flux at interface
  flx(m, IDN, k, j, i) = qc * fl_d  + qd * fr_d;
  flx(m, ivx, k, j, i) = qc * fl_mx + qd * fr_mx + qe * cp;
  flx(m, ivy, k, j, i) = qc * fl_my + qd * fr_my;
  flx(m, ivz, k, j, i) = qc * fl_mz + qd * fr_mz;
  flx(m, IEN, k, j, i) = qc * fl_e  + qd * fr_e  + qe * cp * am;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLC_HYD_HPP_
