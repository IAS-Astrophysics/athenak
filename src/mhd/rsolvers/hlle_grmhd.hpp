#ifndef MHD_RSOLVERS_HLLE_GRMHD_HPP_
#define MHD_RSOLVERS_HLLE_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grmhd.hpp
//! \brief HLLE Riemann solver for general relativistic MHD.
//!
//! Notes:
//!  - cf. HLLE solver in hlle_mhd_rel_no_transform.cpp in Athena++

#include <cmath>      // sqrt()

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void HLLE_GR
//! \brief

KOKKOS_INLINE_FUNCTION
void HLLE_GR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  // Cyclic permutation of array indices corresponding to velocity/b_field components
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;

  const Real gm1 = (eos.gamma - 1.0);
  const Real gamma_prime = eos.gamma/(gm1);
  auto &flat = coord.is_minkowski;
  auto &spin = coord.bh_spin;

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  par_for_inner(member, il, iu, [&](const int i) {
    // References to left primitives
    Real &wl_idn=wl(IDN,i);
    Real &wl_ivx=wl(ivx,i);
    Real &wl_ivy=wl(ivy,i);
    Real &wl_ivz=wl(ivz,i);
    Real &wl_iby=bl(iby,i);
    Real &wl_ibz=bl(ibz,i);

    // References to right primitives
    Real &wr_idn=wr(IDN,i);
    Real &wr_ivx=wr(ivx,i);
    Real &wr_ivy=wr(ivy,i);
    Real &wr_ivz=wr(ivz,i);
    Real &wr_iby=br(iby,i);
    Real &wr_ibz=br(ibz,i);

    Real wl_ipr, wr_ipr;
    wl_ipr = eos.IdealGasPressure(wl(IEN,i));
    wr_ipr = eos.IdealGasPressure(wr(IEN,i));

    // reference to longitudinal field
    Real &bxi = bx(m,k,j,i);

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v,x2v,x3v;
    if (ivx == IVX) {
      x1v = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else if (ivx == IVY) {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
      x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    } else {
      x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      x3v = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);
    }
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

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

    // lower vector indices (covariant compt)
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

    // lower vector indices (covariant compt)
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
    flx(m,IDN,k,j,i) = flux_interface->d;
    flx(m,ivx,k,j,i) = flux_interface->mx;
    flx(m,ivy,k,j,i) = flux_interface->my;
    flx(m,ivz,k,j,i) = flux_interface->mz;
    flx(m,IEN,k,j,i) = flux_interface->e;

    ey(m,k,j,i) = -flux_interface->by;
    ez(m,k,j,i) =  flux_interface->bz;

    // We evolve tau = T^t_t + D
    flx(m,IEN,k,j,i) += flx(m,IDN,k,j,i);
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLE_GRMHD_HPP_
