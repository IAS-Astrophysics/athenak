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

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateFluxes()
//! \brief Update electric fields after masking

void CyclicZoom::UpdateFluxes(Driver *pdriver) {
  // call MHD functions to update electric fields in all MeshBlocks
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  (void) pmhd->InitRecv(pdriver, 1);  // stage = 1
  (void) pmhd->CopyCons(pdriver, 1);  // stage = 1: copy u0 to u1
  (void) pmhd->Fluxes(pdriver, 1);
  // (void) pmhd->RestrictU(this, 0);
  // TODO(@mhguo): may only send/recv electric fields if possible
  (void) pmhd->SendFlux(pdriver, 1);  // stage = 1
  (void) pmhd->RecvFlux(pdriver, 1);  // stage = 1
  (void) pmhd->SendU(pdriver, 1);
  (void) pmhd->RecvU(pdriver, 1);
  (void) pmhd->CornerE(pdriver, 1);
  (void) pmhd->EFieldSrc(pdriver, 1);
  (void) pmhd->SendE(pdriver, 1);
  (void) pmhd->RecvE(pdriver, 1);
  (void) pmhd->SendB(pdriver, 1);
  (void) pmhd->RecvB(pdriver, 1);
  (void) pmhd->ClearSend(pdriver, 1); // stage = 1
  (void) pmhd->ClearRecv(pdriver, 1); // stage = 1
  if (verbose && global_variable::my_rank == 0) {
    std::cout << " CyclicZoom: Calculated electric fields after AMR" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::StoreFluxes()
//! \brief Update electric fields after masking

void CyclicZoom::StoreFluxes() {
  // update electric fields in zoom region
  // TODO(@mhguo): only stored the emf, may need to limit de to emin/max
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
  par_for("zoom-update-efld",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
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
//! \fn void ZoomData::StoreEFieldsFromFiner()
//! \brief Store coarse electric fields in zoom data zmc from finer zoom data zm on
//! previous level

void ZoomData::StoreEFieldsFromFiner(int zmc, int zmf, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
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
  // update coarse electric fields
  par_for("zoom-finer-efld1",DevExeSpace(), ccks, ccke+1, ccjs, ccje+1, ccis, ccie,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*(ci - ccis) + cis;
    int fj = 2*(cj - ccjs) + cjs;
    int fk = 2*(ck - ccks) + cks;
    e1(zmc,ck,cj,ci) = 0.5*(ef1(zmf,fk,fj,fi) + ef1(zmf,fk,fj,fi+1));
  });
  par_for("zoom-finer-efld2",DevExeSpace(), ccks, ccke+1, ccjs, ccje, ccis, ccie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*(ci - ccis) + cis;
    int fj = 2*(cj - ccjs) + cjs;
    int fk = 2*(ck - ccks) + cks;
    e2(zmc,ck,cj,ci) = 0.5*(ef2(zmf,fk,fj,fi) + ef2(zmf,fk,fj+1,fi));
  });
  par_for("zoom-finer-efld3",DevExeSpace(), ccks, ccke, ccjs, ccje+1, ccis, ccie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*(ci - ccis) + cis;
    int fj = 2*(cj - ccjs) + cjs;
    int fk = 2*(ck - ccks) + cks;
    e3(zmc,ck,cj,ci) = 0.5*(ef3(zmf,fk,fj,fi) + ef3(zmf,fk+1,fj,fi));
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
  par_for("zoom-update-efld",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
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
    std::cout << "ZoomData::LimitEFields: emax1=" << emax1
              << ", emax2=" << emax2
              << ", emax3=" << emax3 << std::endl;
  }
  Real emax = fmax(fmax(emax1, emax2), emax3);
  emax *= pzoom->zemf.emf_fmax; // limiting factor
  if (pzoom->verbose && global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: emax for limiting = " << emax << std::endl;
  }
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  int nzmbm1 = pzmesh->nzmb_thisdvce - 1;

  par_for("zoom-limit-efld",DevExeSpace(), 0, nzmbm1, cks,cke+1, cjs,cje+1, cis,cie+1,
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
