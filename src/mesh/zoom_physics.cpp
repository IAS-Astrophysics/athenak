//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_physics.cpp
//! \brief Functions to handle cyclic zoom physics

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/cyclic_zoom.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
// TODO(@mhguo): check whehther all above includes are necessary


//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::BoundaryConditions()
//! \brief User-defined boundary conditions

void CyclicZoom::BoundaryConditions()
{
  if (!zoom_bcs) return;
  if (zstate.zone == 0) return;
  auto &indcs = pmesh->mb_indcs;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;

  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb = pmesh->pmb_pack->nmb_thispack;
  bool is_gr = pmesh->pmb_pack->pcoord->is_general_relativistic;

  // Select either Hydro or MHD
  Real gamma = 0.0;
  DvceArray5D<Real> u0_, w0_, bcc;
  bool is_mhd = (pmesh->pmb_pack->pmhd != nullptr);
  if (pmesh->pmb_pack->phydro != nullptr) {
    gamma = pmesh->pmb_pack->phydro->peos->eos_data.gamma;
    u0_ = pmesh->pmb_pack->phydro->u0;
    w0_ = pmesh->pmb_pack->phydro->w0;
  } else if (pmesh->pmb_pack->pmhd != nullptr) {
    gamma = pmesh->pmb_pack->pmhd->peos->eos_data.gamma;
    u0_ = pmesh->pmb_pack->pmhd->u0;
    w0_ = pmesh->pmb_pack->pmhd->w0;
    bcc = pmesh->pmb_pack->pmhd->bcc0;
  }
  Real gm1 = gamma - 1.0;

  auto cu0 = pzdata->coarse_u0;
  auto cw0 = pzdata->coarse_w0;

  Real rzoom = zregion.radius;
  int zid = pzmesh->nleaf*(zstate.zone-1);
  auto &flat = pmesh->pmb_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmesh->pmb_pack->pcoord->coord_data.bh_spin;
  par_for("fixed_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    if (rad < rzoom) {
      bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
      bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
      int leaf_id = 1*x1r + 2*x2r + 4*x3r;
      int zm = zid + leaf_id;
      int ci = i - cnx1 * x1l;
      int cj = j - cnx2 * x2l;
      int ck = k - cnx3 * x3l;
      if (is_mhd) {
        w0_(m,IDN,k,j,i) = cw0(zm,IDN,ck,cj,ci);
        w0_(m,IM1,k,j,i) = cw0(zm,IM1,ck,cj,ci);
        w0_(m,IM2,k,j,i) = cw0(zm,IM2,ck,cj,ci);
        w0_(m,IM3,k,j,i) = cw0(zm,IM3,ck,cj,ci);
        w0_(m,IEN,k,j,i) = cw0(zm,IEN,ck,cj,ci);

        // Load single state of primitive variables
        MHDPrim1D w;
        w.d  = w0_(m,IDN,k,j,i);
        w.vx = w0_(m,IVX,k,j,i);
        w.vy = w0_(m,IVY,k,j,i);
        w.vz = w0_(m,IVZ,k,j,i);
        w.e  = w0_(m,IEN,k,j,i);

        // load cell-centered fields into primitive state
        w.bx = bcc(m,IBX,k,j,i);
        w.by = bcc(m,IBY,k,j,i);
        w.bz = bcc(m,IBZ,k,j,i);

        // call p2c function
        HydCons1D u;
        if (is_gr) {
          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
          SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
        } else {
          SingleP2C_IdealMHD(w, u);
        }

        // store conserved quantities in 3D array
        u0_(m,IDN,k,j,i) = u.d;
        u0_(m,IM1,k,j,i) = u.mx;
        u0_(m,IM2,k,j,i) = u.my;
        u0_(m,IM3,k,j,i) = u.mz;
        u0_(m,IEN,k,j,i) = u.e;
      } else {
        u0_(m,IDN,k,j,i) = cu0(zm,IDN,ck,cj,ci);
        u0_(m,IM1,k,j,i) = cu0(zm,IM1,ck,cj,ci);
        u0_(m,IM2,k,j,i) = cu0(zm,IM2,ck,cj,ci);
        u0_(m,IM3,k,j,i) = cu0(zm,IM3,ck,cj,ci);
        u0_(m,IEN,k,j,i) = cu0(zm,IEN,ck,cj,ci);
      }
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::FixEField()
//! \brief Modify E field on the zoomed grid

// TODO(@mhguo): may change FixEField to a more general name
void CyclicZoom::FixEField(DvceEdgeFld4D<Real> emf) {
  if (emf_flag == 0) {
    // MeanEField(emf);
  } else if (emf_flag == 1) {
    // AddEField(emf);
  } else if (emf_flag == 2) {
    AddDeltaEField(emf);
  } else if (emf_flag == 3) {
    AddDeltaEField(emf); // adaptive
  } else {
    std::cerr << "Error: CyclicZoom::FixEField() failed: emf_flag = " << emf_flag << std::endl;
    std::exit(1);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::AddDeltaEField()
//! \brief Add delta E field from small scale or adaptive E field

// TODO(@mhguo): check the corner case in ghost zones
void CyclicZoom::AddDeltaEField(DvceEdgeFld4D<Real> emf) {
  if (zstate.zone == 0) return;
  auto &indcs = pmesh->mb_indcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb1 = pmesh->pmb_pack->nmb_thispack-1;
  if (zamr.first_emf) {
    UpdateDeltaEField(emf);
    SyncZoomEField(pzdata->emf0,pzmesh->nleaf*(zstate.zone-1));
    SyncZoomEField(pzdata->delta_efld,pzmesh->nleaf*(zstate.zone-1));
    SetMaxEField();
    if (dump_diag) {
      pzdata->DumpData();
    }
    // TODO(@mhguo): print basic information or not
    zamr.first_emf = false;
  }
  auto de1 = pzdata->delta_efld.x1e;
  auto de2 = pzdata->delta_efld.x2e;
  auto de3 = pzdata->delta_efld.x3e;
  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;
  Real rzoom = zregion.radius;

  int zid = pzmesh->nleaf*(zstate.zone-1);
  Real f0 = emf_f0; //(rad-rzoom)/rzoom;
  Real f1 = emf_f1; //(rzoom-rad)/rzoom;
  if (zstate.zone > emf_zmax) {
    f1 = 0.0;
  }
  Real emax1 = emf_fmax*pzdata->max_emf0(zstate.zone-1,0);
  Real emax2 = emf_fmax*pzdata->max_emf0(zstate.zone-1,1);
  Real emax3 = emf_fmax*pzdata->max_emf0(zstate.zone-1,2);
  par_for("apply-emf", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1 ,is-1, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
    bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
    int leaf_id = 1*x1r + 2*x2r + 4*x3r;
    int zm = zid + leaf_id;
    int ci = i - cnx1 * x1l;
    int cj = j - cnx2 * x2l;
    int ck = k - cnx3 * x3l;
    if (sqrt(SQR(x1v)+SQR(x2f)+SQR(x3f)) < rzoom) {
      // ef1(m,k,j,i) = f0*ef1(m,k,j,i) + f1*de1(zm,ck,cj,ci);
      // limit de1 to be between -emax1 and emax1
      ef1(m,k,j,i) = f0*ef1(m,k,j,i) + f1*fmax(-emax1, fmin(emax1, de1(zm,ck,cj,ci)));
    }
    if (sqrt(SQR(x1f)+SQR(x2v)+SQR(x3f)) < rzoom) {
      // ef2(m,k,j,i) = f0*ef2(m,k,j,i) + f1*de2(zm,ck,cj,ci);
      // limit de2 to be between -emax2 and emax2
      ef2(m,k,j,i) = f0*ef2(m,k,j,i) + f1*fmax(-emax2, fmin(emax2, de2(zm,ck,cj,ci)));
    }
    if (sqrt(SQR(x1f)+SQR(x2f)+SQR(x3v)) < rzoom) {
      // ef3(m,k,j,i) = f0*ef3(m,k,j,i) + f1*de3(zm,ck,cj,ci);
      // limit de3 to be between -emax3 and emax3
      ef3(m,k,j,i) = f0*ef3(m,k,j,i) + f1*fmax(-emax3, fmin(emax3, de3(zm,ck,cj,ci)));
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateDeltaEField()
//! \brief Update delta E field on the zoomed grid

void CyclicZoom::UpdateDeltaEField(DvceEdgeFld4D<Real> emf) {
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: UpdateDeltaEField" << std::endl;
  }
  auto &indcs = pmesh->mb_indcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  auto e1 = pzdata->efld.x1e;
  auto e2 = pzdata->efld.x2e;
  auto e3 = pzdata->efld.x3e;
  auto e01 = pzdata->emf0.x1e;
  auto e02 = pzdata->emf0.x2e;
  auto e03 = pzdata->emf0.x3e;
  auto de1 = pzdata->delta_efld.x1e;
  auto de2 = pzdata->delta_efld.x2e;
  auto de3 = pzdata->delta_efld.x3e;
  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;
  int zid = pzmesh->nleaf*(zstate.zone-1);
  Real rin = zregion.radius;
  for (int m=0; m<nmb; ++m) {
    if (pmesh->lloc_eachmb[m+mbs].level == zamr.level) {
      Real &x1min = size.h_view(m).x1min;
      Real &x1max = size.h_view(m).x1max;
      Real &x2min = size.h_view(m).x2min;
      Real &x2max = size.h_view(m).x2max;
      Real &x3min = size.h_view(m).x3min;
      Real &x3max = size.h_view(m).x3max;
      Real ax1min = x1min*x1max>0.0? fmin(fabs(x1min), fabs(x1max)) : 0.0;
      Real ax2min = x2min*x2max>0.0? fmin(fabs(x2min), fabs(x2max)) : 0.0;
      Real ax3min = x3min*x3max>0.0? fmin(fabs(x3min), fabs(x3max)) : 0.0;
      Real rad_min = sqrt(SQR(ax1min)+SQR(ax2min)+SQR(ax3min));
      if (rad_min < rin) {
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        // update delta electric fields
        par_for("zoom-update-efld",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
        KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
          int fi = ci + cnx1 * x1l; // correct if cis = is
          int fj = cj + cnx2 * x2l; // correct if cjs = js
          int fk = ck + cnx3 * x3l; // correct if cks = ks
          e01(zm,ck,cj,ci) = ef1(m,fk,fj,fi);
          e02(zm,ck,cj,ci) = ef2(m,fk,fj,fi);
          e03(zm,ck,cj,ci) = ef3(m,fk,fj,fi);
          de1(zm,ck,cj,ci) = e1(zm,ck,cj,ci) - ef1(m,fk,fj,fi);
          de2(zm,ck,cj,ci) = e2(zm,ck,cj,ci) - ef2(m,fk,fj,fi);
          de3(zm,ck,cj,ci) = e3(zm,ck,cj,ci) - ef3(m,fk,fj,fi);
        });
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SyncZoomEField()
//! \brief Syncronize variables between different ranks

void CyclicZoom::SyncZoomEField(DvceEdgeFld4D<Real> emf, int zid) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: SyncZoomEField" << std::endl;
  }
  // broadcast zoom data
  auto &indcs = pmesh->mb_indcs;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  int e1_slice_size = (n_ccells3+1) * (n_ccells2+1) * n_ccells1;
  int e2_slice_size = (n_ccells3+1) * n_ccells2 * (n_ccells1+1);
  int e3_slice_size = n_ccells3 * (n_ccells2+1) * (n_ccells1+1);

  // int zid = nleaf*(zstate.zone-1);
  for (int leaf=0; leaf<pzmesh->nleaf; ++leaf) {
    // determine which rank is the "root" rank
    int zm = zid + leaf;
    int x1r = (leaf%2 == 1); int x2r = (leaf%4 > 1); int x3r = (leaf > 3);
    int zm_rank = 0;
    for (int m=0; m<pmesh->nmb_total; ++m) {
      auto lloc = pmesh->lloc_eachmb[m];
      // TODO(@mhguo): not working for half domain
      if (lloc.level == zamr.level) {
        if ((lloc.lx1 == pow(2,zamr.level-1)+x1r-1) &&
            (lloc.lx2 == pow(2,zamr.level-1)+x2r-1) &&
            (lloc.lx3 == pow(2,zamr.level-1)+x3r-1)) {
          zm_rank = pmesh->rank_eachmb[m];
          // print basic information
          // std::cout << "CyclicZoom: Syncing delta efield for zoom meshblock " << zm
          //           << " from rank " << zm_rank << std::endl;
        }
      }
    }
    // It looks device to device communication is not supported, so copy to host first
    auto harr_4d = pzdata->harr_4d;
    Kokkos::realloc(harr_4d, 1, n_ccells3+1, n_ccells2+1, n_ccells1);
    auto e1_slice = Kokkos::subview(emf.x1e, Kokkos::make_pair(zm,zm+1),
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(harr_4d, e1_slice);
    MPI_Bcast(harr_4d.data(), e1_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(e1_slice, harr_4d);

    Kokkos::realloc(harr_4d, 1, n_ccells3+1, n_ccells2, n_ccells1+1);
    auto e2_slice = Kokkos::subview(emf.x2e, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(harr_4d, e2_slice);
    MPI_Bcast(harr_4d.data(), e2_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(e2_slice, harr_4d);

    Kokkos::realloc(harr_4d, 1, n_ccells3, n_ccells2+1, n_ccells1+1);
    auto e3_slice = Kokkos::subview(emf.x3e, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(harr_4d, e3_slice);
    MPI_Bcast(harr_4d.data(), e3_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(e3_slice, harr_4d);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SetMaxEField()
//! \brief Set maximum E field on the zoomed grid

void CyclicZoom::SetMaxEField() {
  if (zstate.zone == 0) return;
  auto pm = pmesh;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, cnx1p1 = indcs.cnx1+1;
  int js = indcs.js, cnx2p1 = indcs.cnx2+1;
  int ks = indcs.ks, cnx3p1 = indcs.cnx3+1;
  const int zid = pzmesh->nleaf*(zstate.zone-1);
  const int nmkji = pzmesh->nleaf*cnx3p1*cnx2p1*cnx1p1;
  const int nkji = cnx3p1*cnx2p1*cnx1p1;
  const int nji  = cnx2p1*cnx1p1;
  const int ni = cnx1p1;
  auto e01 = pzdata->emf0.x1e;
  auto e02 = pzdata->emf0.x2e;
  auto e03 = pzdata->emf0.x3e;
  // debuging
  Real demax1 = 0.0;
  Real demax2 = 0.0;
  Real demax3 = 0.0;
  auto de1 = pzdata->delta_efld.x1e;
  auto de2 = pzdata->delta_efld.x2e;
  auto de3 = pzdata->delta_efld.x3e;
  Kokkos::parallel_reduce("max_de_1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_e1, Real &max_e2, Real &max_e3) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + is;
    k += ks;
    j += js;
    m += zid;
    max_e1 = fmax(max_e1, fabs(de1(m,k,j,i)));
    max_e2 = fmax(max_e2, fabs(de2(m,k,j,i)));
    max_e3 = fmax(max_e3, fabs(de3(m,k,j,i)));
  }, Kokkos::Max<Real>(demax1), Kokkos::Max<Real>(demax2),Kokkos::Max<Real>(demax3));
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: MaxEField: max_de = " << demax1 << " " << demax2 << " " << demax3
              << std::endl;
  }
  Real emax1 = 0.0;
  Real emax2 = 0.0;
  Real emax3 = 0.0;
  Kokkos::parallel_reduce("max_emf0_1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_e1, Real &max_e2, Real &max_e3) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + is;
    k += ks;
    j += js;
    m += zid;
    max_e1 = fmax(max_e1, fabs(e01(m,k,j,i)));
    max_e2 = fmax(max_e2, fabs(e02(m,k,j,i)));
    max_e3 = fmax(max_e3, fabs(e03(m,k,j,i)));
  }, Kokkos::Max<Real>(emax1), Kokkos::Max<Real>(emax2),Kokkos::Max<Real>(emax3));
  pzdata->max_emf0(zstate.zone-1,0) = emax1;
  pzdata->max_emf0(zstate.zone-1,1) = emax2;
  pzdata->max_emf0(zstate.zone-1,2) = emax3;
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: MaxEField: max_emf0 = " << emax1 << " " << emax2 << " " << emax3
              << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::NewTimeStep()
//! \brief New time step for zoom

Real CyclicZoom::NewTimeStep(Mesh* pm) {
  Real dt = pm->dt/pm->cfl_no;
  if (!zoom_dt) return dt;
  bool &is_gr = pmesh->pmb_pack->pcoord->is_general_relativistic;
  bool is_mhd = (pmesh->pmb_pack->pmhd != nullptr);
  dt = (is_gr)? GRTimeStep(pm) : dt; // replace dt with GRTimeStep
  // TODO(@mhguo): 1. EMFTimeStep is too small, 2. we may use v=c instead
  // Real dt_emf = dt;
  // if (emf_dt && is_mhd) {
  //   dt_emf = EMFTimeStep(pm);
  //   if (ndiag > 0 && (pm->ncycle % ndiag == 0) && (zstate.zone > 0)) {
  //     if (dt_emf < dt) {
  //       std::cout << "CyclicZoom: dt_emf = " << dt_emf << " dt = " << dt
  //                 << " on rank " << global_variable::my_rank << std::endl;
  //     }
  //   }
  // }
  // dt = fmin(dt_emf, dt); // get minimum of EMFTimeStep and dt
  return dt;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::GRTimeStep()
//! \brief New time step for GR zoom, only for GR since others are already handled

Real CyclicZoom::GRTimeStep(Mesh* pm) {
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  auto &flat = pmesh->pmb_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmesh->pmb_pack->pcoord->coord_data.bh_spin;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  const int nmkji = (pmesh->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  bool is_hydro = (pmesh->pmb_pack->phydro != nullptr);
  bool is_mhd = (pmesh->pmb_pack->pmhd != nullptr);

  if (is_hydro) {
    auto &w0_ = pmesh->pmb_pack->phydro->w0;
    auto &eos = pmesh->pmb_pack->phydro->peos->eos_data;

    // find smallest dx/(v +/- Cs) in each direction for hydrodynamic problems
    Kokkos::parallel_reduce("ZHydroNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

      // Use the GR sound speed to compute the time step
      // References to left primitives
      Real &wd = w0_(m,IDN,k,j,i);
      Real &ux = w0_(m,IVX,k,j,i);
      Real &uy = w0_(m,IVY,k,j,i);
      Real &uz = w0_(m,IVZ,k,j,i);

      // FIXME ERM: Ideal fluid for now
      Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

      // Extract components of metric
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      // Calculate 4-velocity (contravariant compt)
      Real q = glower[IVX][IVX] * SQR(ux) + glower[IVY][IVY] * SQR(uy) +
               glower[IVZ][IVZ] * SQR(uz) + 2.0*glower[IVX][IVY] * ux * uy +
           2.0*glower[IVX][IVZ] * ux * uz + 2.0*glower[IVY][IVZ] * uy * uz;

      Real alpha = std::sqrt(-1.0/gupper[0][0]);
      Real gamma = sqrt(1.0 + q);
      Real uu[4];
      uu[0] = gamma / alpha;
      uu[IVX] = ux - alpha * gamma * gupper[0][IVX];
      uu[IVY] = uy - alpha * gamma * gupper[0][IVY];
      uu[IVZ] = uz - alpha * gamma * gupper[0][IVZ];

      // Calculate wavespeeds
      Real lm, lp;
      eos.IdealGRHydroSoundSpeeds(wd, p, uu[0], uu[IVX], gupper[0][0],
                                  gupper[0][IVX], gupper[IVX][IVX], lp, lm);
      max_dv1 = fmax(fabs(lm), lp);

      eos.IdealGRHydroSoundSpeeds(wd, p, uu[0], uu[IVY], gupper[0][0],
                                  gupper[0][IVY], gupper[IVY][IVY], lp, lm);
      max_dv2 = fmax(fabs(lm), lp);

      eos.IdealGRHydroSoundSpeeds(wd, p, uu[0], uu[IVZ], gupper[0][0],
                                  gupper[0][IVZ], gupper[IVZ][IVZ], lp, lm);
      max_dv3 = fmax(fabs(lm), lp);

      min_dt1 = fmin((size.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((size.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((size.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  } else if (is_mhd) {
    auto &w0_ = pmesh->pmb_pack->pmhd->w0;
    auto &eos = pmesh->pmb_pack->pmhd->peos->eos_data;
    auto &bcc0_ = pmesh->pmb_pack->pmhd->bcc0;

    // find smallest dx/(v +/- Cf) in each direction for mhd problems
    Kokkos::parallel_reduce("ZMHDNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

      // Use the GR fast magnetosonic speed to compute the time step
      // References to left primitives
      Real &wd = w0_(m,IDN,k,j,i);
      Real &ux = w0_(m,IVX,k,j,i);
      Real &uy = w0_(m,IVY,k,j,i);
      Real &uz = w0_(m,IVZ,k,j,i);
      Real &bcc1 = bcc0_(m,IBX,k,j,i);
      Real &bcc2 = bcc0_(m,IBY,k,j,i);
      Real &bcc3 = bcc0_(m,IBZ,k,j,i);

      // FIXME ERM: Ideal fluid for now
      Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

      // Extract components of metric
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      // Calculate 4-velocity (contravariant compt)
      Real q = glower[IVX][IVX] * SQR(ux) + glower[IVY][IVY] * SQR(uy) +
               glower[IVZ][IVZ] * SQR(uz) + 2.0*glower[IVX][IVY] * ux * uy +
           2.0*glower[IVX][IVZ] * ux * uz + 2.0*glower[IVY][IVZ] * uy * uz;

      Real alpha = std::sqrt(-1.0/gupper[0][0]);
      Real gamma = sqrt(1.0 + q);
      Real uu[4];
      uu[0] = gamma / alpha;
      uu[IVX] = ux - alpha * gamma * gupper[0][IVX];
      uu[IVY] = uy - alpha * gamma * gupper[0][IVY];
      uu[IVZ] = uz - alpha * gamma * gupper[0][IVZ];

      // lower vector indices (covariant compt)
      Real ul[4];
      ul[0]   = glower[0][0]  *uu[0]   + glower[0][IVX]*uu[IVX] +
                glower[0][IVY]*uu[IVY] + glower[0][IVZ]*uu[IVZ];

      ul[IVX] = glower[IVX][0]  *uu[0]   + glower[IVX][IVX]*uu[IVX] +
                glower[IVX][IVY]*uu[IVY] + glower[IVX][IVZ]*uu[IVZ];

      ul[IVY] = glower[IVY][0]  *uu[0]   + glower[IVY][IVX]*uu[IVX] +
                glower[IVY][IVY]*uu[IVY] + glower[IVY][IVZ]*uu[IVZ];

      ul[IVZ] = glower[IVZ][0]  *uu[0]   + glower[IVZ][IVX]*uu[IVX] +
                glower[IVZ][IVY]*uu[IVY] + glower[IVZ][IVZ]*uu[IVZ];


      // Calculate 4-magnetic field in right state
      Real bu[4];
      bu[0]   = ul[IVX]*bcc1 + ul[IVY]*bcc2 + ul[IVZ]*bcc3;
      bu[IVX] = (bcc1 + bu[0] * uu[IVX]) / uu[0];
      bu[IVY] = (bcc2 + bu[0] * uu[IVY]) / uu[0];
      bu[IVZ] = (bcc3 + bu[0] * uu[IVZ]) / uu[0];

      // lower vector indices (covariant compt)
      Real bl[4];
      bl[0]   = glower[0][0]  *bu[0]   + glower[0][IVX]*bu[IVX] +
                glower[0][IVY]*bu[IVY] + glower[0][IVZ]*bu[IVZ];

      bl[IVX] = glower[IVX][0]  *bu[0]   + glower[IVX][IVX]*bu[IVX] +
                glower[IVX][IVY]*bu[IVY] + glower[IVX][IVZ]*bu[IVZ];

      bl[IVY] = glower[IVY][0]  *bu[0]   + glower[IVY][IVX]*bu[IVX] +
                glower[IVY][IVY]*bu[IVY] + glower[IVY][IVZ]*bu[IVZ];

      bl[IVZ] = glower[IVZ][0]  *bu[0]   + glower[IVZ][IVX]*bu[IVX] +
                glower[IVZ][IVY]*bu[IVY] + glower[IVZ][IVZ]*bu[IVZ];

      Real b_sq = bl[0]*bu[0] + bl[IVX]*bu[IVX] + bl[IVY]*bu[IVY] +bl[IVZ]*bu[IVZ];

      // Calculate wavespeeds
      Real lm, lp;
      eos.IdealGRMHDFastSpeeds(wd, p, uu[0], uu[IVX], b_sq, gupper[0][0],
                               gupper[0][IVX], gupper[IVX][IVX], lp, lm);
      max_dv1 = fmax(fabs(lm), lp);

      eos.IdealGRMHDFastSpeeds(wd, p, uu[0], uu[IVY], b_sq, gupper[0][0],
                               gupper[0][IVY], gupper[IVY][IVY], lp, lm);
      max_dv2 = fmax(fabs(lm), lp);

      eos.IdealGRMHDFastSpeeds(wd, p, uu[0], uu[IVZ], b_sq, gupper[0][0],
                               gupper[0][IVZ], gupper[IVZ][IVZ], lp, lm);
      max_dv3 = fmax(fabs(lm), lp);

      min_dt1 = fmin((size.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((size.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((size.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  Real dtnew = dt1;
  if (pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::EMFTimeStep()
//! \brief New time step for emf in zoom

// TODO(@mhguo): not working now, need to update
Real CyclicZoom::EMFTimeStep(Mesh* pm) {
  if (zstate.zone == 0) return std::numeric_limits<float>::max();
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  const int nmkji = (pmesh->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto &eos = pmesh->pmb_pack->pmhd->peos->eos_data;
  auto &bcc0_ = pmesh->pmb_pack->pmhd->bcc0;

  auto de1 = pzdata->delta_efld.x1e;
  auto de2 = pzdata->delta_efld.x2e;
  auto de3 = pzdata->delta_efld.x3e;
  Real rzoom = zregion.radius;

  int zid = pzmesh->nleaf*(zstate.zone-1);
  Real &f0 = emf_f0; //(rad-rzoom)/rzoom;
  Real &f1 = emf_f1; //(rzoom-rad)/rzoom;

  // find smallest dx*|B/E| in each direction for mhd problems
  Kokkos::parallel_reduce("ZEMFNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    // Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;
    Real max_de1 = 0.0, max_de2 = 0.0, max_de3 = 0.0;

    // Use the GR fast magnetosonic speed to compute the time step
    // References to left primitives
    Real &bcc1 = bcc0_(m,IBX,k,j,i);
    Real &bcc2 = bcc0_(m,IBY,k,j,i);
    Real &bcc3 = bcc0_(m,IBZ,k,j,i);

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
    bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
    int leaf_id = 1*x1r + 2*x2r + 4*x3r;
    int zm = zid + leaf_id;
    int ci = i - cnx1 * x1l;
    int cj = j - cnx2 * x2l;
    int ck = k - cnx3 * x3l;
    // should be face centered or edge centered, but use cell centered for now
    if (rad < rzoom) {
      max_de1 = fmax(fabs(de1(zm,ck,cj,ci)), fmax(fabs(de1(zm,ck+1,cj,ci)),
                fmax(fabs(de1(zm,ck,cj+1,ci)), fabs(de1(zm,ck+1,cj+1,ci)))));
      max_de2 = fmax(fabs(de2(zm,ck,cj,ci)), fmax(fabs(de2(zm,ck+1,cj,ci)),
                fmax(fabs(de2(zm,ck,cj,ci+1)), fabs(de2(zm,ck+1,cj,ci+1)))));
      max_de3 = fmax(fabs(de3(zm,ck,cj,ci)), fmax(fabs(de3(zm,ck,cj+1,ci)),
                fmax(fabs(de3(zm,ck,cj,ci+1)), fabs(de3(zm,ck,cj+1,ci+1)))));
    }
    Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
    min_dt1 = fmin(fabs(bcc1)/(max_de2/dx3+max_de3/dx2), min_dt1);
    min_dt2 = fmin(fabs(bcc2)/(max_de3/dx1+max_de1/dx3), min_dt2);
    min_dt3 = fmin(fabs(bcc3)/(max_de1/dx2+max_de2/dx1), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  Real dtnew = dt1;
  if (pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return dtnew;
}
