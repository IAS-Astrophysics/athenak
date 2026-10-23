//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_fluxes.cpp
//! \brief Functions for updating and storing fluxes during zooming

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateFluxes()
//! \brief Update electric fields after masking

void CyclicZoom::UpdateFluxes(Driver *pdriver) {
  // call MHD functions to update electric fields in all MeshBlocks
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  (void) pmhd->InitRecv(pdriver, 1);  // stage = 1
  (void) pmhd->CopyCons(pdriver, 1);  // stage = 1: copy u0 to u1
  if (pmesh->pmb_pack->pdyngr != nullptr) {
    (void) pmesh->pmb_pack->pdyngr->Fluxes(pdriver, 1);
  } else {
    (void) pmhd->Fluxes(pdriver, 1);
  }
  // (void) pmhd->RestrictU(this, 0);
  // TODO(@mhguo): may only send/recv electric fields if possible
  (void) pmhd->SendFlux(pdriver, 1);  // stage = 1
  (void) pmhd->RecvFlux(pdriver, 1);  // stage = 1
  (void) pmhd->SendU(pdriver, 1);
  (void) pmhd->RecvU(pdriver, 1);
  (void) pmhd->CornerE(pdriver, 1);
  (void) pmhd->EField(pdriver, 1);
  (void) pmhd->SendE(pdriver, 1);
  (void) pmhd->RecvE(pdriver, 1);
  (void) pmhd->SendB(pdriver, 1);
  (void) pmhd->RecvB(pdriver, 1);
  (void) pmhd->ClearSend(pdriver, 1); // stage = 1
  (void) pmhd->ClearRecv(pdriver, 1); // stage = 1
  if (verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Calculated electric fields after AMR" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::StoreFluxes()
//! \brief Update electric fields after masking

void CyclicZoom::StoreFluxes() {
  // update electric fields in zoom region
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->mblid_eachzmb[zm+zmbs];
    // pzdata->UpdateElectricFieldsInZoomRegion(m, zm);
    auto efld = pmesh->pmb_pack->pmhd->efld;
    pzdata->StoreEFieldsAfterAMR(zm, m, efld);
  }
  // limit electric fields if needed
  pzdata->LimitEFields();
  if (verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Updated electric fields in zoom region" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreEFieldsBeforeAMR()
//! \brief Store coarse electric fields in zoom data zm from MeshBlock m

void ZoomData::StoreEFieldsBeforeAMR(int zm, int m, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  auto ef1 = efld.x1e;
  auto ef2 = efld.x2e;
  auto ef3 = efld.x3e;
  auto e1 = efld_pre.x1e;
  auto e2 = efld_pre.x2e;
  auto e3 = efld_pre.x3e;
  // update coarse electric fields
  par_for("zoom_store_efld_before",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*ci - cis;  // correct when cis=is
    int fj = 2*cj - cjs;  // correct when cjs=js
    int fk = 2*ck - cks;  // correct when cks=ks
    e1(zm,ck,cj,ci) = 0.5*(ef1(m,fk,fj,fi) + ef1(m,fk,fj,fi+1));
    e2(zm,ck,cj,ci) = 0.5*(ef2(m,fk,fj,fi) + ef2(m,fk,fj+1,fi));
    e3(zm,ck,cj,ci) = 0.5*(ef3(m,fk,fj,fi) + ef3(m,fk+1,fj,fi));
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::CorrectEFieldsFromFiner()
//! \brief Correct coarse electric fields in zoom data zm in MeshBlock m from finer zoom
//! data zmf on previous level

void ZoomData::CorrectEFieldsFromFiner(int zm, int m, int zmf, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto &size = pzoom->pmesh->pmb_pack->pmb->mb_size;
  int &cis = indcs.cis;
  int &cjs = indcs.cjs;
  int &cks = indcs.cks;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  auto ef1 = efld.x1e;
  auto ef2 = efld.x2e;
  auto ef3 = efld.x3e;
  auto e1 = efld_pre.x1e;
  auto e2 = efld_pre.x2e;
  auto e3 = efld_pre.x3e;
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank]; // global id start of dvce
  auto &zlloc = pzmesh->lloc_eachzmb[zmf+zmbs];
  int ox1 = ((zlloc.lx1 & 1) == 1);
  int ox2 = ((zlloc.lx2 & 1) == 1);
  int ox3 = ((zlloc.lx3 & 1) == 1);
  int hcnx1 = cnx1 / 2, hcnx2 = cnx2 / 2, hcnx3 = cnx3 / 2;
  int ccis = cis + ox1 * hcnx1;
  int ccjs = cjs + ox2 * hcnx2;
  int ccks = cks + ox3 * hcnx3;
  int ccie = ccis + hcnx1 - 1;
  int ccje = ccjs + hcnx2 - 1;
  int ccke = ccks + hcnx3 - 1;
  auto zregion = pzoom->old_zregion; // previous zoom region
  // update coarse electric fields
  par_for("zoom_finer_efld1",DevExeSpace(), ccks, ccke+1, ccjs, ccje+1, ccis, ccie,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    Real x1v = CellCenterX(ci-cis, cnx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2f = LeftEdgeX  (cj-cjs, cnx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3f = LeftEdgeX  (ck-cks, cnx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if (zregion.IsInRegion(x1v, x2f, x3f)) {
      int fi = 2*(ci - ccis) + cis;
      int fj = 2*(cj - ccjs) + cjs;
      int fk = 2*(ck - ccks) + cks;
      e1(zm,ck,cj,ci) = 0.5*(ef1(zmf,fk,fj,fi) + ef1(zmf,fk,fj,fi+1));
    }
    if (zregion.IsInRegion(x1v, x2f, x3f, zregion.flx.r)) {
      e1(zm,ck,cj,ci) = 0.0;
    }
  });
  par_for("zoom_finer_efld2",DevExeSpace(), ccks, ccke+1, ccjs, ccje, ccis, ccie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    Real x1f = LeftEdgeX  (ci-cis, cnx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(cj-cjs, cnx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3f = LeftEdgeX  (ck-cks, cnx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if (zregion.IsInRegion(x1f, x2v, x3f)) {
      int fi = 2*(ci - ccis) + cis;
      int fj = 2*(cj - ccjs) + cjs;
      int fk = 2*(ck - ccks) + cks;
      e2(zm,ck,cj,ci) = 0.5*(ef2(zmf,fk,fj,fi) + ef2(zmf,fk,fj+1,fi));
    }
    if (zregion.IsInRegion(x1f, x2v, x3f, zregion.flx.r)) {
      e2(zm,ck,cj,ci) = 0.0;
    }
  });
  par_for("zoom_finer_efld3",DevExeSpace(), ccks, ccke, ccjs, ccje+1, ccis, ccie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    Real x1f = LeftEdgeX  (ci-cis, cnx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2f = LeftEdgeX  (cj-cjs, cnx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(ck-cks, cnx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if (zregion.IsInRegion(x1f, x2f, x3v)) {
      int fi = 2*(ci - ccis) + cis;
      int fj = 2*(cj - ccjs) + cjs;
      int fk = 2*(ck - ccks) + cks;
      e3(zm,ck,cj,ci) = 0.5*(ef3(zmf,fk,fj,fi) + ef3(zmf,fk+1,fj,fi));
    }
    if (zregion.IsInRegion(x1f, x2f, x3v, zregion.flx.r)) {
      e3(zm,ck,cj,ci) = 0.0;
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreEFieldsAfterAMR()
//! \brief Store electric fields in zoom data zm from MeshBlock m

void ZoomData::StoreEFieldsAfterAMR(int zm, int m, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  auto ef1 = efld.x1e;
  auto ef2 = efld.x2e;
  auto ef3 = efld.x3e;
  auto ep1 = efld_pre.x1e;
  auto ep2 = efld_pre.x2e;
  auto ep3 = efld_pre.x3e;
  auto ea1 = efld_aft.x1e;
  auto ea2 = efld_aft.x2e;
  auto ea3 = efld_aft.x3e;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank]; // global id start of dvce
  auto &zlloc = pzmesh->lloc_eachzmb[zm+zmbs];
  int ox1 = ((zlloc.lx1 & 1) == 1);
  int ox2 = ((zlloc.lx2 & 1) == 1);
  int ox3 = ((zlloc.lx3 & 1) == 1);
  auto w_ = pzoom->pmesh->pmb_pack->pmhd->w0;
  auto u_ = pzoom->pmesh->pmb_pack->pmhd->u0;
  // update electric fields
  par_for("zoom_store_efld_after",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    // int fi = 2*ci - cis;  // correct when cis=is
    // int fj = 2*cj - cjs;  // correct when cjs=js
    // int fk = 2*ck - cks;  // correct when cks=ks
    int i = ci + ox1 * cnx1;
    int j = cj + ox2 * cnx2;
    int k = ck + ox3 * cnx3;
    ea1(zm,ck,cj,ci) = ef1(m,k,j,i);
    ea2(zm,ck,cj,ci) = ef2(m,k,j,i);
    ea3(zm,ck,cj,ci) = ef3(m,k,j,i);
    de1(zm,ck,cj,ci) = ep1(zm,ck,cj,ci) - ef1(m,k,j,i);
    de2(zm,ck,cj,ci) = ep2(zm,ck,cj,ci) - ef2(m,k,j,i);
    de3(zm,ck,cj,ci) = ep3(zm,ck,cj,ci) - ef3(m,k,j,i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZoomData::LimitEFields()
//! \brief Limit electric fields in zoom data zm

void ZoomData::LimitEFields() {
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1p1 = indcs.cnx1+1, cnx2p1 = indcs.cnx2+1, cnx3p1 = indcs.cnx3+1;
  const int nmkji = pzmesh->nzmb_thisdvce*cnx3p1*cnx2p1*cnx1p1;
  const int nkji = cnx3p1*cnx2p1*cnx1p1;
  const int nji  = cnx2p1*cnx1p1;
  const int ni = cnx1p1;
  auto e01 = efld_aft.x1e;
  auto e02 = efld_aft.x2e;
  auto e03 = efld_aft.x3e;
  Real emax1 = 0.0;
  Real emax2 = 0.0;
  Real emax3 = 0.0;
  Kokkos::parallel_reduce("zoom-max-emf",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_e1, Real &max_e2, Real &max_e3) {
    int zm = (idx)/nkji;
    int ck = (idx - zm*nkji)/nji;
    int cj = (idx - zm*nkji - ck*nji)/ni;
    int ci = (idx - zm*nkji - ck*nji - cj*ni) + cis;
    ck += cks;
    cj += cjs;
    max_e1 = fmax(max_e1, fabs(e01(zm,ck,cj,ci)));
    max_e2 = fmax(max_e2, fabs(e02(zm,ck,cj,ci)));
    max_e3 = fmax(max_e3, fabs(e03(zm,ck,cj,ci)));
  }, Kokkos::Max<Real>(emax1), Kokkos::Max<Real>(emax2),Kokkos::Max<Real>(emax3));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &emax1, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax2, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax3, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (pzoom->verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: LimitEFields: emax1=" << emax1
              << ", emax2=" << emax2
              << ", emax3=" << emax3 << std::endl;
  }
  Real emax = fmax(fmax(emax1, emax2), emax3);
  emax *= pzoom->zemf.emf_fmax; // limiting factor
  if (pzoom->verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: LimitEFields: emax for limiting = " << emax << std::endl;
  }
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  int nzmbm1 = pzmesh->nzmb_thisdvce - 1;

  par_for("zoom_limit_efld",DevExeSpace(), 0, nzmbm1, cks,cke+1, cjs,cje+1, cis,cie+1,
  KOKKOS_LAMBDA(const int zm, const int ck, const int cj, const int ci) {
    // use copy sign function to avoid nan issue
    if (fabs(de1(zm,ck,cj,ci)) > emax) {
      de1(zm,ck,cj,ci) = copysign(emax, de1(zm,ck,cj,ci));
    }
    if (fabs(de2(zm,ck,cj,ci)) > emax) {
      de2(zm,ck,cj,ci) = copysign(emax, de2(zm,ck,cj,ci));
    }
    if (fabs(de3(zm,ck,cj,ci)) > emax) {
      de3(zm,ck,cj,ci) = copysign(emax, de3(zm,ck,cj,ci));
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::AccumulateFluxes()
//! \brief Sum the fluxes into the zoom sphere through the faces between cells inside
//! (r <= radius, as in the mask) and outside it, for this stage of the time integrator:
//! rest mass D, Ye*D, energy -sqrt(-g) T^i_t = alpha*(F_tau + F_D) - beta^j F_S_j, and
//! J_z = sqrt(-g) T^i_phi = x F_S_y - y F_S_x about the zoom centre. Fluxes are weighted
//! by their weight in the final update of the step, so the sums are the amounts crossing
//! the faces over the step. dyn_grmhd only. Uses the sphere of the mask in zone > 0 and
//! the flux_r_in sphere in zone 0.

void CyclicZoom::AccumulateFluxes(Driver *pdrive, int stage) {
  auto pmbp = pmesh->pmb_pack;
  if (pmbp->pdyngr == nullptr || stage < 1 || stage > pdrive->nexp_stages) return;
  // the after-time-integrator task list applies the BCs again with stage = 1
  if (pmesh->ncycle == hlast_cycle && stage <= hlast_stage) return;
  hlast_cycle = pmesh->ncycle;
  hlast_stage = stage;

  Real rs;
  Real *acc;
  if (zstate.zone > 0) {
    rs = zregion.radius;
    acc = hflx_mask;
  } else if (flux_r_in > 0.0) {
    rs = flux_r_in;
    acc = hflx_in;
  } else {
    return;
  }
  // weight of this stage's fluxes in u^(n+1) = ... + beta_s*dt*prod_{t>s}(gam0_t)*L_s
  Real wght = pdrive->beta[stage-1]*pmesh->dt;
  for (int s = stage; s < pdrive->nexp_stages; ++s) {
    wght *= pdrive->gam0[s];
  }

  auto &indcs = pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto &flx1 = pmbp->pmhd->uflx.x1f;
  auto &flx2 = pmbp->pmhd->uflx.x2f;
  auto &flx3 = pmbp->pmhd->uflx.x3f;
  auto &adm = pmbp->padm->adm;
  int iyf = (pmbp->pmhd->nscalars > 0) ? pmbp->pmhd->nmhd : -1;
  Real xc[3] = {zregion.x1c, zregion.x2c, zregion.x3c};
  Real rs2 = SQR(rs);

  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce("zoom_flux_sums",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, (pmbp->nmb_thispack)*nkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real xv[3];
    xv[0] = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    xv[1] = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    xv[2] = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    if (SQR(xv[0]-xc[0]) + SQR(xv[1]-xc[1]) + SQR(xv[2]-xc[2]) > rs2) return;
    Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};

    array_sum::GlobalSum fsum;
    for (int d = 0; d < 3; ++d) {
      for (int side = 0; side < 2; ++side) {
        // skip faces whose neighbour is also inside the sphere
        Real sh = (side == 0) ? -1.0 : 1.0;
        Real xn[3] = {xv[0], xv[1], xv[2]};
        xn[d] += sh*dx[d];
        if (SQR(xn[0]-xc[0]) + SQR(xn[1]-xc[1]) + SQR(xn[2]-xc[2]) <= rs2) continue;
        // face and neighbour indices
        int fi = i, fj = j, fk = k;
        int ni = i, nj = j, nk = k;
        if (d == 0) {fi += side; ni += (side == 0) ? -1 : 1;}
        if (d == 1) {fj += side; nj += (side == 0) ? -1 : 1;}
        if (d == 2) {fk += side; nk += (side == 0) ? -1 : 1;}
        auto flx = [&](int n) {
          return (d == 0) ? flx1(m,n,fk,fj,fi) :
                 (d == 1) ? flx2(m,n,fk,fj,fi) : flx3(m,n,fk,fj,fi);
        };
        // inward = -outward, and the outward normal is -e_d (side 0) or +e_d (side 1)
        Real area = dx[0]*dx[1]*dx[2]/dx[d];
        Real sgn = -sh*area;
        Real xf[3] = {xv[0], xv[1], xv[2]};
        xf[d] += 0.5*sh*dx[d];
        Real alp = 0.5*(adm.alpha(m,k,j,i) + adm.alpha(m,nk,nj,ni));
        Real bx = 0.5*(adm.beta_u(m,0,k,j,i) + adm.beta_u(m,0,nk,nj,ni));
        Real by = 0.5*(adm.beta_u(m,1,k,j,i) + adm.beta_u(m,1,nk,nj,ni));
        Real bz = 0.5*(adm.beta_u(m,2,k,j,i) + adm.beta_u(m,2,nk,nj,ni));
        Real fd = flx(IDN);
        fsum.the_array[0] += sgn*fd;
        fsum.the_array[1] += (iyf >= 0) ? sgn*flx(iyf) : 0.0;
        fsum.the_array[2] += sgn*(alp*(flx(IEN) + fd)
                                  - (bx*flx(IM1) + by*flx(IM2) + bz*flx(IM3)));
        fsum.the_array[3] += sgn*((xf[0] - xc[0])*flx(IM2) - (xf[1] - xc[1])*flx(IM1));
      }
    }
    mb_sum += fsum;
  }, Kokkos::Sum<array_sum::GlobalSum>(sums));

  for (int n = 0; n < 4; ++n) {
    acc[n] += wght*sums.the_array[n];
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::LoadHistoryData()
//! \brief Fill the zoom history (summed over ranks by the history output) and reset the
//! accumulators. Amounts are those since the previous output, positive into the sphere.

void CyclicZoom::LoadHistoryData(HistoryData *pdata) {
  pdata->nhist = 12;
  pdata->label[0] = "zone";
  pdata->label[1] = "r-mask";
  pdata->label[2] = "dM-mask";
  pdata->label[3] = "dYeM-mask";
  pdata->label[4] = "dE-mask";
  pdata->label[5] = "dJz-mask";
  pdata->label[6] = "r-in";
  pdata->label[7] = "dM-in";
  pdata->label[8] = "dYeM-in";
  pdata->label[9] = "dE-in";
  pdata->label[10] = "dJz-in";
  pdata->label[11] = "dM-trans";
  bool root = (global_variable::my_rank == 0);
  pdata->hdata[0] = root ? static_cast<Real>(zstate.zone) : 0.0;
  pdata->hdata[1] = (root && zstate.zone > 0) ? zregion.radius : 0.0;
  pdata->hdata[6] = root ? flux_r_in : 0.0;
  for (int n = 0; n < 4; ++n) {
    pdata->hdata[2+n] = hflx_mask[n];
    pdata->hdata[7+n] = hflx_in[n];
    hflx_mask[n] = 0.0;
    hflx_in[n] = 0.0;
  }
  pdata->hdata[11] = hdm_trans;
  hdm_trans = 0.0;
  return;
}
