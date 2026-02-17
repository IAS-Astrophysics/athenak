//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_srcterms.cpp
//! \brief Implements cyclic zoom source terms

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "mhd/mhd.hpp"

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SourceTermsFC()
//! \brief Add delta E field from small scale

void CyclicZoom::SourceTermsFC(DvceEdgeFld4D<Real> efld) {
  // only apply when add_emf is true
  if (!zemf.add_emf) return;
  // apply only when zone > 0
  if (zstate.zone == 0) return;
  if (zstate.zone > zemf.emf_zmax) return;
  if (zamr.zooming_out || zamr.zooming_in) return;
  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank];
  for (int zm = 0; zm < pzmesh->nzmb_thisdvce; ++zm) {
    int m = pzmesh->mblid_eachzmb[zm+zmbs];
    // pzdata->UpdateElectricFieldsInZoomRegion(m, zm);
    // pzdata->StoreEFields(zm, m);
    pzdata->AddSrcTermsFC(m, zm, efld);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::AddSrcTermsFC()
//! \brief Add delta E field from small scale for one meshblock

void ZoomData::AddSrcTermsFC(int m, int zm, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto &size = pzoom->pmesh->pmb_pack->pmb->mb_size;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  // int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  // int nzmb1 = pzmesh->nzmb_thisdvce - 1;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  auto ef1 = efld.x1e;
  auto ef2 = efld.x2e;
  auto ef3 = efld.x3e;
  auto ep1 = efld_pre.x1e;
  auto ep2 = efld_pre.x2e;
  auto ep3 = efld_pre.x3e;
  auto ea1 = efld_aft.x1e;
  auto ea2 = efld_aft.x2e;
  auto ea3 = efld_aft.x3e;

  int zmbs = pzmesh->gzms_eachdvce[global_variable::my_rank]; // global id start of dvce
  auto &zlloc = pzmesh->lloc_eachzmb[zm+zmbs];
  int ox1 = ((zlloc.lx1 & 1) == 1);
  int ox2 = ((zlloc.lx2 & 1) == 1);
  int ox3 = ((zlloc.lx3 & 1) == 1);
  auto zregion = pzoom->zregion;

  auto w_ = pzoom->pmesh->pmb_pack->pmhd->w0;
  auto u_ = pzoom->pmesh->pmb_pack->pmhd->u0;

  // Create independent execution space instances for concurrent kernel launches
  DevExeSpace exec1, exec2, exec3;

  // Launch three kernels in parallel using separate execution spaces
  par_for("apply-emf-x1", exec1, cks, cke+1, cjs, cje+1, cis, cie,
  KOKKOS_LAMBDA(int ck, int cj, int ci) {
    int i = ci + ox1 * cnx1;
    int j = cj + ox2 * cnx2;
    int k = ck + ox3 * cnx3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX  (j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3f = LeftEdgeX  (k-ks, nx3, x3min, x3max);

    // apply to zoom region
    if (zregion.IsInZoomRegion(x1v, x2f, x3f)) {
        ef1(m,k,j,i) += de1(zm,ck,cj,ci);
    }
  });

  par_for("apply-emf-x2", exec2, cks, cke+1, cjs, cje, cis, cie+1,
  KOKKOS_LAMBDA(int ck, int cj, int ci) {
    int i = ci + ox1 * cnx1;
    int j = cj + ox2 * cnx2;
    int k = ck + ox3 * cnx3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX  (i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3f = LeftEdgeX  (k-ks, nx3, x3min, x3max);

    // apply to zoom region
    if (zregion.IsInZoomRegion(x1f, x2v, x3f)) {
        ef2(m,k,j,i) += de2(zm,ck,cj,ci);
    }
  });

  par_for("apply-emf-x3", exec3, cks, cke, cjs, cje+1, cis, cie+1,
  KOKKOS_LAMBDA(int ck, int cj, int ci) {
    int i = ci + ox1 * cnx1;
    int j = cj + ox2 * cnx2;
    int k = ck + ox3 * cnx3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX  (i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX  (j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // apply to zoom region
    if (zregion.IsInZoomRegion(x1f, x2f, x3v)) {
        ef3(m,k,j,i) += de3(zm,ck,cj,ci);
    }
  });

  // Fence only needed if subsequent code depends on results being ready
  // Can be omitted if Kokkos will fence automatically before next synchronization point
  exec1.fence();
  exec2.fence();
  exec3.fence();

  return;
}
