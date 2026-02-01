//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_efld.cpp
//  \brief Functions for edge-centered fields in ZoomData class

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

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
  // TODO(@mhguo): add even finer electric fields
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreFinerEFields()
//! \brief Store coarse electric fields in zoom data zmc from finer zoom data zm on previous level

void ZoomData::StoreFinerEFields(int zmc, int zm, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  auto ef1 = efld.x1e;
  auto ef2 = efld.x2e;
  auto ef3 = efld.x3e;
  auto e1 = efld_pre.x1e;
  auto e2 = efld_pre.x2e;
  auto e3 = efld_pre.x3e;
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank]; // global id start of dvce
  auto &zlloc = pzmesh->lloc_eachzmb[zm+zmbs];
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
  printf("ZoomData::StoreFinerEFields zmc=%d zm=%d ox1=%d ox2=%d ox3=%d ccis=%d ccie=%d ccjs=%d ccje=%d ccks=%d ccke=%d\n",
         zmc, zm, ox1, ox2, ox3, ccis, ccie, ccjs, ccje, ccks, ccke);
  // update coarse electric fields
  par_for("zoom-finer-efld1",DevExeSpace(), ccks, ccke+1, ccjs, ccje+1, ccis, ccie,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*(ci - ccis) + cis;
    int fj = 2*(cj - ccjs) + cjs;
    int fk = 2*(ck - ccks) + cks;
    e1(zmc,ck,cj,ci) = 0.5*(ef1(zm,fk,fj,fi) + ef1(zm,fk,fj,fi+1));
  });
  par_for("zoom-finer-efld2",DevExeSpace(), ccks, ccke+1, ccjs, ccje, ccis, ccie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*(ci - ccis) + cis;
    int fj = 2*(cj - ccjs) + cjs;
    int fk = 2*(ck - ccks) + cks;
    e2(zmc,ck,cj,ci) = 0.5*(ef2(zm,fk,fj,fi) + ef2(zm,fk,fj+1,fi));
  });
  par_for("zoom-finer-efld3",DevExeSpace(), ccks, ccke, ccjs, ccje+1, ccis, ccie+1,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*(ci - ccis) + cis;
    int fj = 2*(cj - ccjs) + cjs;
    int fk = 2*(ck - ccks) + cks;
    e3(zmc,ck,cj,ci) = 0.5*(ef3(zm,fk,fj,fi) + ef3(zm,fk+1,fj,fi));
  });

  // print debug info
    // if (fi == cis && fj == cjs && fk == cks) {
    //   printf("ZoomData::StoreFinerEFields zmc=%d zm=%d fi=%d fj=%d fk=%d ef1=%e ef2=%e ef3=%e\n",
    //          zmc, zm, fi, fj, fk,
    //          ef1(zmc,fk,fj,fi), ef2(zmc,fk,fj,fi), ef3(zmc,fk,fj,fi));
    // }
    // if (fi == cie && fj == cje && fk == cke) {
    //   printf("ZoomData::StoreFinerEFields zmc=%d zm=%d fi=%d fj=%d fk=%d ef1=%e ef2=%e ef3=%e\n",
    //          zmc, zm, fi, fj, fk,
    //          ef1(zmc,fk,fj,fi), ef2(zmc,fk,fj,fi), ef3(zmc,fk,fj,fi));
    // }
    // int ckm = (cks + cke) / 2, cjm = (cjs + cje) / 2, cim = (cis + cie) / 2;
    // if (fk == ckm && fj == cjm && fi == cim) {
    //   printf("ZoomData::StoreFinerEFields zmc=%d zm=%d fi=%d fj=%d fk=%d ef1=%e ef2=%e ef3=%e\n",
    //          zmc, zm, fi, fj, fk,
    //          ef1(zmc,fk,fj,fi), ef2(zmc,fk,fj,fi), ef3(zmc,fk,fj,fi));
    // }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::StoreEFieldsAfterAMR()
//! \brief Store electric fields in zoom data zm from MeshBlock m

void ZoomData::StoreEFieldsAfterAMR(int zm, int m, DvceEdgeFld4D<Real> efld) {
  auto &indcs = pzoom->pmesh->mb_indcs;
  auto &size = pzoom->pmesh->pmb_pack->pmb->mb_size;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  // int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
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
  int zmbs = pzmesh->gids_eachdvce[global_variable::my_rank]; // global id start of dvce
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

    // debugging info
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f = LeftEdgeX  (i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f = LeftEdgeX  (j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f = LeftEdgeX  (k-ks, nx3, x3min, x3max);

    Real rad = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);

    // print debug info
    if (ck==cks && cj==cjs) {
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k-1=%d j=%d i=%d rad=%e w0=%e w1=%e  w2=%e  w3=%e  w4=%e\n",
             zm, m, k-1, j, i, rad, w_(m,IDN,k-1,j,i), w_(m,IVX,k-1,j,i), w_(m,IVY,k-1,j,i), w_(m,IVZ,k-1,j,i), w_(m,IEN,k-1,j,i));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e w0=%e w1=%e  w2=%e  w3=%e  w4=%e\n",
             zm, m, k, j, i, rad, w_(m,IDN,k,j,i), w_(m,IVX,k,j,i), w_(m,IVY,k,j,i), w_(m,IVZ,k,j,i), w_(m,IEN,k,j,i));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep1=%e, ef1=%e de1=%e\n",
             zm, m, k, j, i, rad, ep1(zm,ck,cj,ci), ef1(m,k,j,i), de1(zm,ck,cj,ci));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep2=%e, ef2=%e de2=%e\n",
             zm, m, k, j, i, rad, ep2(zm,ck,cj,ci), ef2(m,k,j,i), de2(zm,ck,cj,ci));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep3=%e, ef3=%e de3=%e\n",
             zm, m, k, j, i, rad, ep3(zm,ck,cj,ci), ef3(m,k,j,i), de3(zm,ck,cj,ci));
    }
    if (ck==cke && cj==cje && ci==cie) {
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep1=%e, ef1=%e de1=%e\n",
             zm, m, k, j, i, rad, ep1(zm,ck,cj,ci), ef1(m,k,j,i), de1(zm,ck,cj,ci));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep2=%e, ef2=%e de2=%e\n",
             zm, m, k, j, i, rad, ep2(zm,ck,cj,ci), ef2(m,k,j,i), de2(zm,ck,cj,ci));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep3=%e, ef3=%e de3=%e\n",
             zm, m, k, j, i, rad, ep3(zm,ck,cj,ci), ef3(m,k,j,i), de3(zm,ck,cj,ci));
    }
    int ckm = (cks + cke + 1) / 2, cjm = (cjs + cje + 1) / 2, cim = (cis + cie + 1) / 2;
    if (ck==ckm && cj==cjm && ci==cim) {
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep1=%e, ef1=%e de1=%e\n",
             zm, m, k, j, i, rad, ep1(zm,ck,cj,ci), ef1(m,k,j,i), de1(zm,ck,cj,ci));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep2=%e, ef2=%e de2=%e\n",
             zm, m, k, j, i, rad, ep2(zm,ck,cj,ci), ef2(m,k,j,i), de2(zm,ck,cj,ci));
      printf("ZoomData::StoreEFieldsAfterAMR zm=%d m=%d k=%d j=%d i=%d rad=%e ep3=%e, ef3=%e de3=%e\n",
             zm, m, k, j, i, rad, ep3(zm,ck,cj,ci), ef3(m,k,j,i), de3(zm,ck,cj,ci));
    }
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
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: local emax1=" << emax1
              << ", emax2=" << emax2
              << ", emax3=" << emax3 << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &emax1, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax2, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax3, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: emax1=" << emax1
              << ", emax2=" << emax2
              << ", emax3=" << emax3 << std::endl;
  }
  Real emax = fmax(fmax(emax1, emax2), emax3);
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: emax for limiting = " << emax << std::endl;
  }
  // Real elimit = 0.1; // TODO(@mhguo): make this a parameter
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  int nzmbm1 = pzmesh->nzmb_thisdvce - 1;
  // debug info
  emax1 = 0.0;
  emax2 = 0.0;
  emax3 = 0.0;
  Kokkos::parallel_reduce("zoom-max-delta-emf",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_de1, Real &max_de2, Real &max_de3) {
    int zm = (idx)/nkji;
    int ck = (idx - zm*nkji)/nji;
    int cj = (idx - zm*nkji - ck*nji)/ni;
    int ci = (idx - zm*nkji - ck*nji - cj*ni) + cis;
    ck += cks;
    cj += cjs;
    max_de1 = fmax(max_de1, fabs(de1(zm,ck,cj,ci)));
    max_de2 = fmax(max_de2, fabs(de2(zm,ck,cj,ci)));
    max_de3 = fmax(max_de3, fabs(de3(zm,ck,cj,ci)));
  }, Kokkos::Max<Real>(emax1), Kokkos::Max<Real>(emax2),Kokkos::Max<Real>(emax3));
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: pre-limit local de1max=" << emax1
              << ", de2max=" << emax2
              << ", de3max=" << emax3 << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &emax1, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax2, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax3, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: pre-limit de1max=" << emax1
              << ", de2max=" << emax2
              << ", de3max=" << emax3 << std::endl;
  }

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

  // debug info
  emax1 = 0.0;
  emax2 = 0.0;
  emax3 = 0.0;
  Kokkos::parallel_reduce("zoom-max-delta-emf",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_de1, Real &max_de2, Real &max_de3) {
    int zm = (idx)/nkji;
    int ck = (idx - zm*nkji)/nji;
    int cj = (idx - zm*nkji - ck*nji)/ni;
    int ci = (idx - zm*nkji - ck*nji - cj*ni) + cis;
    ck += cks;
    cj += cjs;
    max_de1 = fmax(max_de1, fabs(de1(zm,ck,cj,ci)));
    max_de2 = fmax(max_de2, fabs(de2(zm,ck,cj,ci)));
    max_de3 = fmax(max_de3, fabs(de3(zm,ck,cj,ci)));
  }, Kokkos::Max<Real>(emax1), Kokkos::Max<Real>(emax2),Kokkos::Max<Real>(emax3));
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: post-limit local de1max=" << emax1
              << ", de2max=" << emax2
              << ", de3max=" << emax3 << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &emax1, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax2, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &emax3, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "ZoomData::LimitEFields: post-limit de1max=" << emax1
              << ", de2max=" << emax2
              << ", de3max=" << emax3 << std::endl;
  }

  return;
}
