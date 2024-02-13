#ifndef MHD_RSOLVERS_HLLE_GRMHD_SINGLESTATE_HPP_
#define MHD_RSOLVERS_HLLE_GRMHD_SINGLESTATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grmhd_singlestate.hpp
//! \brief HLLE Riemann solver for GR MHD

#include <cmath>      // sqrt()
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void SingleStateHLLE_GRMHD
//! \brief The HLLE Riemann solver for GR MHD for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateHLLE_GRMHD(const MHDPrim1D wl, const MHDPrim1D wr, const Real bx,
                          const Real x1v, const Real x2v, const Real x3v, const int ivx,
                          const CoordData &coord, const EOS_Data &eos, MHDCons1D &flux) {
  // Cyclic permutation of array indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  // References to left primitives
  const Real &wl_idn=wl.d;
  const Real &wl_ivx=wl.vx;
  const Real &wl_ivy=wl.vy;
  const Real &wl_ivz=wl.vz;
  const Real &wl_iby=wl.by;
  const Real &wl_ibz=wl.bz;

  // References to right primitives
  const Real &wr_idn=wr.d;
  const Real &wr_ivx=wr.vx;
  const Real &wr_ivy=wr.vy;
  const Real &wr_ivz=wr.vz;
  const Real &wr_iby=wr.by;
  const Real &wr_ibz=wr.bz;

  Real wl_ipr, wr_ipr;
  wl_ipr = eos.IdealGasPressure(wl.e);
  wr_ipr = eos.IdealGasPressure(wr.e);

  // reference to longitudinal field
  const Real &bxi = bx;

  // Extract components of metric
  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v,x2v,x3v,coord.is_minkowski, coord.bh_spin, glower, gupper);

  // Calculate 4-velocity in left state (contravariant compt)
  Real q = glower[ivx][ivx] * SQR(wl_ivx) + glower[ivy][ivy] * SQR(wl_ivy) +
           glower[ivz][ivz] * SQR(wl_ivz) + 2.0*glower[ivx][ivy] * wl_ivx * wl_ivy +
       2.0*glower[ivx][ivz] * wl_ivx * wl_ivz + 2.0*glower[ivy][ivz] * wl_ivy * wl_ivz;

  Real alpha = std::sqrt(-1.0/gupper[0][0]);
  Real gamma = sqrt(1.0 + q);
  Real uul[4];
  uul[0] = gamma / alpha;
  uul[ivx] = wl_ivx - alpha * gamma * gupper[0][ivx];
  uul[ivy] = wl_ivy - alpha * gamma * gupper[0][ivy];
  uul[ivz] = wl_ivz - alpha * gamma * gupper[0][ivz];

  // lower vector indices on left state (covariant compt)
  Real ull[4];
  ull[0]   = glower[0][0]  *uul[0]   + glower[0][ivx]*uul[ivx] +
             glower[0][ivy]*uul[ivy] + glower[0][ivz]*uul[ivz];

  ull[ivx] = glower[ivx][0]  *uul[0]   + glower[ivx][ivx]*uul[ivx] +
             glower[ivx][ivy]*uul[ivy] + glower[ivx][ivz]*uul[ivz];

  ull[ivy] = glower[ivy][0]  *uul[0]   + glower[ivy][ivx]*uul[ivx] +
             glower[ivy][ivy]*uul[ivy] + glower[ivy][ivz]*uul[ivz];

  ull[ivz] = glower[ivz][0]  *uul[0]   + glower[ivz][ivx]*uul[ivx] +
             glower[ivz][ivy]*uul[ivy] + glower[ivz][ivz]*uul[ivz];

  // calculate 4-magnetic field in left state (contravariant compt)
  Real bul[4];
  bul[0]   = ull[ivx]*bxi + ull[ivy]*wl_iby + ull[ivz]*wl_ibz;
  bul[ivx] = (bxi    + bul[0] * uul[ivx]) / uul[0];
  bul[ivy] = (wl_iby + bul[0] * uul[ivy]) / uul[0];
  bul[ivz] = (wl_ibz + bul[0] * uul[ivz]) / uul[0];

  // lower vector indices (covariant compt)
  Real bll[4];
  bll[0]   = glower[0][0]  *bul[0]   + glower[0][ivx]*bul[ivx] +
             glower[0][ivy]*bul[ivy] + glower[0][ivz]*bul[ivz];

  bll[ivx] = glower[ivx][0]  *bul[0]   + glower[ivx][ivx]*bul[ivx] +
             glower[ivx][ivy]*bul[ivy] + glower[ivx][ivz]*bul[ivz];

  bll[ivy] = glower[ivy][0]  *bul[0]   + glower[ivy][ivx]*bul[ivx] +
             glower[ivy][ivy]*bul[ivy] + glower[ivy][ivz]*bul[ivz];

  bll[ivz] = glower[ivz][0]  *bul[0]   + glower[ivz][ivx]*bul[ivx] +
             glower[ivz][ivy]*bul[ivy] + glower[ivz][ivz]*bul[ivz];

  Real bsq_l = bll[0]*bul[0] + bll[ivx]*bul[ivx] + bll[ivy]*bul[ivy] +bll[ivz]*bul[ivz];

  // Calculate 4-velocity in right state (contravariant compt)
  q = glower[ivx][ivx] * SQR(wr_ivx) + glower[ivy][ivy] * SQR(wr_ivy) +
      glower[ivz][ivz] * SQR(wr_ivz) + 2.0*glower[ivx][ivy] * wr_ivx * wr_ivy +
      2.0*glower[ivx][ivz] * wr_ivx * wr_ivz + 2.0*glower[ivy][ivz] * wr_ivy * wr_ivz;

  gamma = sqrt(1.0 + q);
  Real uur[4];
  uur[0] = gamma / alpha;
  uur[ivx] = wr_ivx - alpha * gamma * gupper[0][ivx];
  uur[ivy] = wr_ivy - alpha * gamma * gupper[0][ivy];
  uur[ivz] = wr_ivz - alpha * gamma * gupper[0][ivz];

  // lower vector indices on right state (covariant compt)
  Real ulr[4];
  ulr[0]   = glower[0][0]  *uur[0]   + glower[0][ivx]*uur[ivx] +
             glower[0][ivy]*uur[ivy] + glower[0][ivz]*uur[ivz];

  ulr[ivx] = glower[ivx][0]  *uur[0]   + glower[ivx][ivx]*uur[ivx] +
             glower[ivx][ivy]*uur[ivy] + glower[ivx][ivz]*uur[ivz];

  ulr[ivy] = glower[ivy][0]  *uur[0]   + glower[ivy][ivx]*uur[ivx] +
             glower[ivy][ivy]*uur[ivy] + glower[ivy][ivz]*uur[ivz];

  ulr[ivz] = glower[ivz][0]  *uur[0]   + glower[ivz][ivx]*uur[ivx] +
             glower[ivz][ivy]*uur[ivy] + glower[ivz][ivz]*uur[ivz];

  // Calculate 4-magnetic field in right state
  Real bur[4];
  bur[0]   = ulr[ivx]*bxi + ulr[ivy]*wr_iby + ulr[ivz]*wr_ibz;
  bur[ivx] = (bxi    + bur[0] * uur[ivx]) / uur[0];
  bur[ivy] = (wr_iby + bur[0] * uur[ivy]) / uur[0];
  bur[ivz] = (wr_ibz + bur[0] * uur[ivz]) / uur[0];

  // lower vector indices (covariant compt)
  Real blr[4];
  blr[0]   = glower[0][0]  *bur[0]   + glower[0][ivx]*bur[ivx] +
             glower[0][ivy]*bur[ivy] + glower[0][ivz]*bur[ivz];

  blr[ivx] = glower[ivx][0]  *bur[0]   + glower[ivx][ivx]*bur[ivx] +
             glower[ivx][ivy]*bur[ivy] + glower[ivx][ivz]*bur[ivz];

  blr[ivy] = glower[ivy][0]  *bur[0]   + glower[ivy][ivx]*bur[ivx] +
             glower[ivy][ivy]*bur[ivy] + glower[ivy][ivz]*bur[ivz];

  blr[ivz] = glower[ivz][0]  *bur[0]   + glower[ivz][ivx]*bur[ivx] +
             glower[ivz][ivy]*bur[ivy] + glower[ivz][ivz]*bur[ivz];

  Real bsq_r = blr[0]*bur[0] + blr[ivx]*bur[ivx] + blr[ivy]*bur[ivy] +blr[ivz]*bur[ivz];

  // Calculate wavespeeds in left state
  Real lp_l, lm_l;
  eos.IdealGRMHDFastSpeeds(wl_idn, wl_ipr, uul[0], uul[ivx], bsq_l, gupper[0][0],
                           gupper[0][ivx], gupper[ivx][ivx], lp_l, lm_l);

  // Calculate wavespeeds in right state
  Real lp_r, lm_r;
  eos.IdealGRMHDFastSpeeds(wr_idn, wr_ipr, uur[0], uur[ivx], bsq_r, gupper[0][0],
                           gupper[0][ivx], gupper[ivx][ivx], lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
  MHDCons1D du;
  Real wtot_r = wr_idn + gamma_prime * wr_ipr + bsq_r;
  Real ptot_r = wr_ipr + 0.5*bsq_r;
  Real qa = wtot_r * uur[0];
  Real wtot_l = wl_idn + gamma_prime * wl_ipr + bsq_l;
  Real ptot_l = wl_ipr + 0.5*bsq_l;
  Real qb = wtot_l * uul[0];
  du.d  = (wr_idn*uur[0]) - (wl_idn*uul[0]);
  du.mx = (qa*ulr[ivx] - bur[0]*blr[ivx]) - (qb*ull[ivx] - bul[0]*bll[ivx]);
  du.my = (qa*ulr[ivy] - bur[0]*blr[ivy]) - (qb*ull[ivy] - bul[0]*bll[ivy]);
  du.mz = (qa*ulr[ivz] - bur[0]*blr[ivz]) - (qb*ull[ivz] - bul[0]*bll[ivz]);
  du.e  = (qa*ulr[0] - bur[0]*blr[0] + ptot_r) - (qb*ull[0] - bul[0]*bll[0] + ptot_l);
  du.by = (bur[ivy]*uur[0] - bur[0]*uur[ivy]) - (bul[ivy]*uul[0] - bul[0]*uul[ivy]);
  du.bz = (bur[ivz]*uur[0] - bur[0]*uur[ivz]) - (bul[ivz]*uul[0] - bul[0]*uul[ivz]);

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fl;
  qa = wtot_l * uul[ivx];
  fl.d  = wl_idn * uul[ivx];
  fl.mx = qa * ull[ivx] - bul[ivx] * bll[ivx] + ptot_l;
  fl.my = qa * ull[ivy] - bul[ivx] * bll[ivy];
  fl.mz = qa * ull[ivz] - bul[ivx] * bll[ivz];
  fl.e  = qa * ull[0]   - bul[ivx] * bll[0];
  fl.by = bul[ivy] * uul[ivx] - bul[ivx] * uul[ivy];
  fl.bz = bul[ivz] * uul[ivx] - bul[ivx] * uul[ivz];

  // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fr;
  qa = wtot_r * uur[ivx];
  fr.d  = wr_idn * uur[ivx];
  fr.mx = qa * ulr[ivx] - bur[ivx] * blr[ivx] + ptot_r;
  fr.my = qa * ulr[ivy] - bur[ivx] * blr[ivy];
  fr.mz = qa * ulr[ivz] - bur[ivx] * blr[ivz];
  fr.e  = qa * ulr[0]   - bur[ivx] * blr[0];
  fr.by = bur[ivy] * uur[ivx] - bur[ivx] * uur[ivy];
  fr.bz = bur[ivz] * uur[ivx] - bur[ivx] * uur[ivz];

  // Calculate fluxes in HLL region
  MHDCons1D flux_hll;
  qa = lambda_r*lambda_l;
  qb = 1.0/(lambda_r - lambda_l);
  flux_hll.d  = (lambda_r*fl.d  - lambda_l*fr.d  + qa*du.d ) * qb;
  flux_hll.mx = (lambda_r*fl.mx - lambda_l*fr.mx + qa*du.mx) * qb;
  flux_hll.my = (lambda_r*fl.my - lambda_l*fr.my + qa*du.my) * qb;
  flux_hll.mz = (lambda_r*fl.mz - lambda_l*fr.mz + qa*du.mz) * qb;
  flux_hll.e  = (lambda_r*fl.e  - lambda_l*fr.e  + qa*du.e ) * qb;
  flux_hll.by = (lambda_r*fl.by - lambda_l*fr.by + qa*du.by) * qb;
  flux_hll.bz = (lambda_r*fl.bz - lambda_l*fr.bz + qa*du.bz) * qb;

  // Determine region of wavefan
  MHDCons1D *flux_interface;
  if (lambda_l >= 0.0) {  // L region
    flux_interface = &fl;
  } else if (lambda_r <= 0.0) { // R region
    flux_interface = &fr;
  } else {  // HLL region
    flux_interface = &flux_hll;
  }

  // Set fluxes
  flux.d  = flux_interface->d;
  flux.mx = flux_interface->mx;
  flux.my = flux_interface->my;
  flux.mz = flux_interface->mz;
  flux.e  = flux_interface->e;
  flux.by = -flux_interface->by;
  flux.bz = flux_interface->bz;

  // We evolve tau = T^t_t + D
  flux.e  += flux.d;
  return;
}

} // namespace mhd
#endif // MHD_RSOLVERS_HLLE_GRMHD_SINGLESTATE_HPP_
