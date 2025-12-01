//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_refinement.cpp
//! \brief Functions to handle cyclic zoom mesh refinement

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
//! \fn void CyclicZoom::WorkBeforeAMR()
//! \brief Work before zooming: store zoom region
void CyclicZoom::WorkBeforeAMR() {
  // zoom state has been updated in SetRefinementFlags()
  if (zamr.zooming_out) {
    UpdateVariables();
    // TODO(@mhguo): only store the data needed on this rank instead of holding all
    SyncVariables();
    UpdateGhostVariables();
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Store variables before zooming" << std::endl;
    }
    zamr.zooming_out = false;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::WorkAfterAMR()
//! \brief Work after zooming: initialize zoom region, store emf if needed
void CyclicZoom::WorkAfterAMR() {
  if (zamr.zooming_in) {
    ApplyVariables();
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: Apply variables after zooming" << std::endl;
    }
    zamr.zooming_in = false;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateVariables()
//! \brief Update variables before zooming

void CyclicZoom::UpdateVariables() {
  auto &indcs = pmesh->mb_indcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  DvceArray5D<Real> u, w;
  if (pmesh->pmb_pack->phydro != nullptr) {
    u = pmesh->pmb_pack->phydro->u0;
    w = pmesh->pmb_pack->phydro->w0;
  } else if (pmesh->pmb_pack->pmhd != nullptr) {
    u = pmesh->pmb_pack->pmhd->u0;
    w = pmesh->pmb_pack->pmhd->w0;
  }
  auto cu = pzdata->coarse_u0, cw = pzdata->coarse_w0;
  // zoom state has updated
  int zid = pzmesh->nleaf*(zstate.zone - 1);
  int nlf = pzmesh->nleaf;
  Real rzoom = zregion.radius;
  int nvar = pzdata->nvars;
  Real rin = zregion.radius;
  // TODO(@mhguo): it looks 0.8*rzoom works, but ideally should use edge center
  Real refac = re_fac; // r < refac*rzoom
  Real r0ef = r0_efld; // r < r0_efld

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
        auto des_slice = Kokkos::subview(pzdata->u0, Kokkos::make_pair(zm,zm+1),
                                         Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto src_slice = Kokkos::subview(u, Kokkos::make_pair(m,m+1),
                                         Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        des_slice = Kokkos::subview(pzdata->w0, Kokkos::make_pair(zm,zm+1),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        src_slice = Kokkos::subview(w, Kokkos::make_pair(m,m+1),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        par_for("zoom-update",DevExeSpace(), 0,nvar-1, cks,cke, cjs,cje, cis,cie,
        KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          int finei = 2*i - cis;  // correct if cis = is
          int finej = 2*j - cjs;  // correct if cjs = js
          int finek = 2*k - cks;  // correct if cks = ks
          cu(zm,n,k,j,i) =
              0.125*(u(m,n,finek  ,finej  ,finei) + u(m,n,finek  ,finej  ,finei+1)
                  + u(m,n,finek  ,finej+1,finei) + u(m,n,finek  ,finej+1,finei+1)
                  + u(m,n,finek+1,finej,  finei) + u(m,n,finek+1,finej,  finei+1)
                  + u(m,n,finek+1,finej+1,finei) + u(m,n,finek+1,finej+1,finei+1));
          cw(zm,n,k,j,i) =
              0.125*(w(m,n,finek  ,finej  ,finei) + w(m,n,finek  ,finej  ,finei+1)
                  + w(m,n,finek  ,finej+1,finei) + w(m,n,finek  ,finej+1,finei+1)
                  + w(m,n,finek+1,finej,  finei) + w(m,n,finek+1,finej,  finei+1)
                  + w(m,n,finek+1,finej+1,finei) + w(m,n,finek+1,finej+1,finei+1));
        });
        UpdateHydroVariables(zm, m);
        if (pmesh->pmb_pack->pmhd != nullptr && fix_efield) {
          DvceEdgeFld4D<Real> emf = pmesh->pmb_pack->pmhd->efld;
          auto e1 = pzdata->efld.x1e;
          auto e2 = pzdata->efld.x2e;
          auto e3 = pzdata->efld.x3e;
          auto ef1 = emf.x1e;
          auto ef2 = emf.x2e;
          auto ef3 = emf.x3e;
          // update coarse electric fields
          par_for("zoom-update-efld",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int finei = 2*i - cis;  // correct when cis=is
            int finej = 2*j - cjs;  // correct when cjs=js
            int finek = 2*k - cks;  // correct when cks=ks
            e1(zm,k,j,i) = 0.5*(ef1(m,finek,finej,finei) + ef1(m,finek,finej,finei+1));
            e2(zm,k,j,i) = 0.5*(ef2(m,finek,finej,finei) + ef2(m,finek,finej+1,finei));
            e3(zm,k,j,i) = 0.5*(ef3(m,finek,finej,finei) + ef3(m,finek+1,finej,finei));

            // TODO(@mhguo): it looks 0.8*rzoom works, but ideally should use edge center
            Real x1v = CellCenterX(i-cis, cnx1, x1min, x1max);
            Real x2v = CellCenterX(j-cjs, cnx2, x2min, x2max);
            Real x3v = CellCenterX(k-cks, cnx3, x3min, x3max);
            Real x1f = LeftEdgeX  (i-cis, cnx1, x1min, x1max);
            Real x2f = LeftEdgeX  (j-cjs, cnx2, x2min, x2max);
            Real x3f = LeftEdgeX  (k-cks, cnx3, x3min, x3max);
            Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
            Real rade1 = sqrt(SQR(x1v)+SQR(x2f)+SQR(x3f));
            Real rade2 = sqrt(SQR(x1f)+SQR(x2v)+SQR(x3f));
            Real rade3 = sqrt(SQR(x1f)+SQR(x2f)+SQR(x3v));
            if (zid>0) {
              int zmp = zm-nlf;
              int prei = finei - cnx1 * x1l;
              int prej = finej - cnx2 * x2l;
              int prek = finek - cnx3 * x3l;
              if (rade1 < refac*rzoom) {
                e1(zm,k,j,i) = 0.5*(e1(zmp,prek,prej,prei) + e1(zmp,prek,prej,prei+1));
              }
              if (rade2 < refac*rzoom) {
                e2(zm,k,j,i) = 0.5*(e2(zmp,prek,prej,prei) + e2(zmp,prek,prej+1,prei));
              }
              if (rade3 < refac*rzoom) {
                e3(zm,k,j,i) = 0.5*(e3(zmp,prek,prej,prei) + e3(zmp,prek+1,prej,prei));
              }
            }
            if (rade1 < r0ef) {e1(zm,k,j,i) = 0.0;}
            if (rade2 < r0ef) {e2(zm,k,j,i) = 0.0;}
            if (rade3 < r0ef) {e3(zm,k,j,i) = 0.0;}
          });
        }
        std::cout << "CyclicZoom: Update variables for zoom meshblock " << zm << std::endl;
      }
    }
  }
  // if (zid != nleaf*(zstate.zone+1)) {
  //   std::cerr << "Error: CyclicZoom::UpdateVariables() failed: zid = " << zid <<
  //                " zone = " << zstate.zone << " level = " << zamr.level << std::endl;
  //   std::exit(1);
  // }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateHydroVariables()
//! \brief Update hydro variables using conserved hydro variables

void CyclicZoom::UpdateHydroVariables(int zm, int m) {
  auto &indcs = pmesh->mb_indcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  DvceArray5D<Real> u0_, w0_;
  bool is_gr = pmesh->pmb_pack->pcoord->is_general_relativistic;
  auto peos = (pmesh->pmb_pack->pmhd != nullptr)? pmesh->pmb_pack->pmhd->peos : pmesh->pmb_pack->phydro->peos;
  auto eos = peos->eos_data;
  if (pmesh->pmb_pack->phydro != nullptr) {
    u0_ = pmesh->pmb_pack->phydro->u0;
    w0_ = pmesh->pmb_pack->phydro->w0;
  } else if (pmesh->pmb_pack->pmhd != nullptr) {
    u0_ = pmesh->pmb_pack->pmhd->u0;
    w0_ = pmesh->pmb_pack->pmhd->w0;
  }
  auto cw = pzdata->coarse_w0;
  Real &x1min = size.h_view(m).x1min;
  Real &x1max = size.h_view(m).x1max;
  Real &x2min = size.h_view(m).x2min;
  Real &x2max = size.h_view(m).x2max;
  Real &x3min = size.h_view(m).x3min;
  Real &x3max = size.h_view(m).x3max;
  auto &flat = pmesh->pmb_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmesh->pmb_pack->pcoord->coord_data.bh_spin;
  par_for("zoom-update-cwu",DevExeSpace(), cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*ci - cis;  // correct when cis=is
    int fj = 2*cj - cjs;  // correct when cjs=js
    int fk = 2*ck - cks;  // correct when cks=ks
    cw(zm,IDN,ck,cj,ci) = 0.0;
    cw(zm,IM1,ck,cj,ci) = 0.0;
    cw(zm,IM2,ck,cj,ci) = 0.0;
    cw(zm,IM3,ck,cj,ci) = 0.0;
    cw(zm,IEN,ck,cj,ci) = 0.0;
    Real glower[4][4], gupper[4][4];
    // Step 1: compute coarse-grained hydro conserved variables
    for (int ii=0; ii<2; ++ii) {
      for (int jj=0; jj<2; ++jj) {
        for (int kk=0; kk<2; ++kk) {
          // Load single state of primitive variables
          HydPrim1D w;
          w.d  = w0_(m,IDN,fk+kk,fj+jj,fi+ii);
          w.vx = w0_(m,IVX,fk+kk,fj+jj,fi+ii);
          w.vy = w0_(m,IVY,fk+kk,fj+jj,fi+ii);
          w.vz = w0_(m,IVZ,fk+kk,fj+jj,fi+ii);
          w.e  = w0_(m,IEN,fk+kk,fj+jj,fi+ii);

          // call p2c function
          HydCons1D u;
          if (is_gr) {
            Real x1v = CellCenterX(fi+ii-is, nx1, x1min, x1max);
            Real x2v = CellCenterX(fj+jj-js, nx2, x2min, x2max);
            Real x3v = CellCenterX(fk+kk-ks, nx3, x3min, x3max);
            ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
            SingleP2C_IdealGRHyd(glower, gupper, w, eos.gamma, u);
          } else {
            SingleP2C_IdealHyd(w, u);
          }

          // store conserved quantities using cw
          cw(zm,IDN,ck,cj,ci) += 0.125*u.d;
          cw(zm,IM1,ck,cj,ci) += 0.125*u.mx;
          cw(zm,IM2,ck,cj,ci) += 0.125*u.my;
          cw(zm,IM3,ck,cj,ci) += 0.125*u.mz;
          cw(zm,IEN,ck,cj,ci) += 0.125*u.e;
        }
      }
    }
    // Step 2: convert coarse-grained hydro conserved variables to primitive variables
    // Shall we add excision?
    // load single state conserved variables
    HydCons1D u;
    u.d  = cw(zm,IDN,ck,cj,ci);
    u.mx = cw(zm,IM1,ck,cj,ci);
    u.my = cw(zm,IM2,ck,cj,ci);
    u.mz = cw(zm,IM3,ck,cj,ci);
    u.e  = cw(zm,IEN,ck,cj,ci);

    HydPrim1D w;
    if (is_gr) {
      // Extract components of metric
      Real x1v = CellCenterX(ci-cis, cnx1, x1min, x1max);
      Real x2v = CellCenterX(cj-cjs, cnx2, x2min, x2max);
      Real x3v = CellCenterX(ck-cks, cnx3, x3min, x3max);
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      HydCons1D u_sr;
      Real s2;
      TransformToSRHyd(u,glower,gupper,s2,u_sr);
      bool dfloor_used=false, efloor_used=false;
      bool c2p_failure=false;
      int iter_used=0;
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                        dfloor_used, efloor_used, c2p_failure, iter_used);
      // apply velocity ceiling if necessary
      Real tmp = glower[1][1]*SQR(w.vx)
                + glower[2][2]*SQR(w.vy)
                + glower[3][3]*SQR(w.vz)
                + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
                + 2.0*glower[2][3]*w.vy*w.vz;
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    } else {
      bool dfloor_used=false, efloor_used=false, tfloor_used=false;
      SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
    }
    cw(zm,IDN,ck,cj,ci) = w.d;
    cw(zm,IVX,ck,cj,ci) = w.vx;
    cw(zm,IVY,ck,cj,ci) = w.vy;
    cw(zm,IVZ,ck,cj,ci) = w.vz;
    cw(zm,IEN,ck,cj,ci) = w.e;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SyncVariables()
//! \brief Syncronize variables between different ranks

void CyclicZoom::SyncVariables() {
#if MPI_PARALLEL_ENABLED
  // broadcast zoom data
  auto &indcs = pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  int &nvars = pzdata->nvars;
  int u0_slice_size = nvars * ncells1 * ncells2 * ncells3;
  int w0_slice_size = nvars * ncells1 * ncells2 * ncells3;
  int cu_slice_size = nvars * n_ccells1 * n_ccells2 * n_ccells3;
  int cw_slice_size = nvars * n_ccells1 * n_ccells2 * n_ccells3;
  // zoom state has updated
  int &nleaf = pzmesh->nleaf;
  int zid = nleaf*(zstate.zone - 1);
  for (int leaf=0; leaf<nleaf; ++leaf) {
    // determine which rank is the "root" rank
    int zm = zid + leaf;
    int x1r = (leaf%2 == 1); int x2r = (leaf%4 > 1); int x3r = (leaf > 3);
    int zm_rank = 0;
    for (int m=0; m<pmesh->nmb_total; ++m) {
      auto lloc = pmesh->lloc_eachmb[m];
      if (lloc.level == zamr.level) {
        if ((lloc.lx1 == pow(2,zamr.level-1)+x1r-1) &&
            (lloc.lx2 == pow(2,zamr.level-1)+x2r-1) &&
            (lloc.lx3 == pow(2,zamr.level-1)+x3r-1)) {
          zm_rank = pmesh->rank_eachmb[m];
          // print basic information
          // std::cout << "CyclicZoom: Syncing variables for zoom meshblock " << zm
          //           << " from rank " << zm_rank << std::endl;
        }
      }
    }
    // It looks device to device communication is not supported, so copy to host first
    // TODO(@mhguo): check whether this auto works
    auto harr_5d = pzdata->harr_5d;
    Kokkos::realloc(harr_5d, 1, nvars, ncells3, ncells2, ncells1);
    auto u0_slice = Kokkos::subview(pzdata->u0, Kokkos::make_pair(zm,zm+1),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, u0_slice);
    MPI_Bcast(harr_5d.data(), u0_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(u0_slice, harr_5d);

    auto w0_slice = Kokkos::subview(pzdata->w0, Kokkos::make_pair(zm,zm+1),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, w0_slice);
    MPI_Bcast(harr_5d.data(), w0_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(w0_slice, harr_5d);

    Kokkos::realloc(harr_5d, 1, nvars, n_ccells3, n_ccells2, n_ccells1);
    auto cu_slice = Kokkos::subview(pzdata->coarse_u0, Kokkos::make_pair(zm,zm+1),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, cu_slice);
    MPI_Bcast(harr_5d.data(), cu_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(cu_slice, harr_5d);

    auto cw_slice = Kokkos::subview(pzdata->coarse_w0, Kokkos::make_pair(zm,zm+1),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, cw_slice);
    MPI_Bcast(harr_5d.data(), cw_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(cw_slice, harr_5d);
  }
  SyncZoomEField(pzdata->efld,nleaf*zstate.zone - 1);
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateGhostVariables()
//! \brief Update variables in ghost cells between different meshblocks

// TODO(@mhguo): add emf?
void CyclicZoom::UpdateGhostVariables() {
  auto &indcs = pmesh->mb_indcs;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  auto u = pzdata->u0, w = pzdata->w0;
  auto cu = pzdata->coarse_u0, cw = pzdata->coarse_w0;
  int &nvar = pzdata->nvars;
  int &nleaf = pzmesh->nleaf;
  // zoom state has updated
  int zid = nleaf*(zstate.zone - 1);
  for (int leaf=0; leaf<nleaf; ++leaf) {
    // determine which rank is the "root" rank
    int zm = zid + leaf;
    int x1r = (leaf%2 == 1); int x2r = (leaf%4 > 1); int x3r = (leaf > 3);
    for (int sleaf=0; sleaf<nleaf; ++sleaf) { // source leaf index
      if (leaf == sleaf) continue;
      int sm = zid + sleaf;
      int sx1r = (sleaf%2 == 1); int sx2r = (sleaf%4 > 1); int sx3r = (sleaf > 3);
      int si = (x1r == sx1r)? 0 : (x1r? nx1 : -nx1);
      int sj = (x2r == sx2r)? 0 : (x2r? nx2 : -nx2);
      int sk = (x3r == sx3r)? 0 : (x3r? nx3 : -nx3);
      int il = (x1r == sx1r)? is : (x1r? (is - ng) : (ie + 1));
      int iu = (x1r == sx1r)? ie : (x1r? (is - 1) : (ie + ng));
      int jl = (x2r == sx2r)? js : (x2r? (js - ng) : (je + 1));
      int ju = (x2r == sx2r)? je : (x2r? (js - 1) : (je + ng));
      int kl = (x3r == sx3r)? ks : (x3r? (ks - ng) : (ke + 1));
      int ku = (x3r == sx3r)? ke : (x3r? (ks - 1) : (ke + ng));
      par_for("zoom-comm",DevExeSpace(), 0, nvar-1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        u(zm,n,k,j,i) = u(sm,n,k+sk,j+sj,i+si);
        w(zm,n,k,j,i) = w(sm,n,k+sk,j+sj,i+si);
      });
      si = (x1r == sx1r)? 0 : (x1r? cnx1 : -cnx1);
      sj = (x2r == sx2r)? 0 : (x2r? cnx2 : -cnx2);
      sk = (x3r == sx3r)? 0 : (x3r? cnx3 : -cnx3);
      il = (x1r == sx1r)? cis : (x1r? (cis - ng) : (cie + 1));
      iu = (x1r == sx1r)? cie : (x1r? (cis - 1) : (cie + ng));
      jl = (x2r == sx2r)? cjs : (x2r? (cjs - ng) : (cje + 1));
      ju = (x2r == sx2r)? cje : (x2r? (cjs - 1) : (cje + ng));
      kl = (x3r == sx3r)? cks : (x3r? (cks - ng) : (cke + 1));
      ku = (x3r == sx3r)? cke : (x3r? (cks - 1) : (cke + ng));
      par_for("zoom-comm-c",DevExeSpace(), 0, nvar-1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        cu(zm,n,k,j,i) = cu(sm,n,k+sk,j+sj,i+si);
        cw(zm,n,k,j,i) = cw(sm,n,k+sk,j+sj,i+si);
      });
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::ApplyVariables()
//! \brief Apply finer level variables to coarser level

void CyclicZoom::ApplyVariables() {
  auto &indcs = pmesh->mb_indcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  auto eos = (pmesh->pmb_pack->pmhd != nullptr)? pmesh->pmb_pack->pmhd->peos->eos_data : 
              pmesh->pmb_pack->phydro->peos->eos_data;
  Real gamma = eos.gamma;
  DvceArray5D<Real> u_, w_;
  if (pmesh->pmb_pack->phydro != nullptr) {
    u_ = pmesh->pmb_pack->phydro->u0;
    w_ = pmesh->pmb_pack->phydro->w0;
  } else if (pmesh->pmb_pack->pmhd != nullptr) {
    u_ = pmesh->pmb_pack->pmhd->u0;
    w_ = pmesh->pmb_pack->pmhd->w0;
  }
  Real rzoom = zregion.radius;
  auto &flat = pmesh->pmb_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmesh->pmb_pack->pcoord->coord_data.bh_spin;
  auto u0_ = pzdata->u0, w0_ = pzdata->w0;
  int &nleaf = pzmesh->nleaf;
  int zid = nleaf*zstate.zone;
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
      if (rad_min < zregion.radius) {
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        if (pmesh->pmb_pack->phydro != nullptr) { // TODO: this seems wrong, may use 2*r_zoom
          auto src_slice = Kokkos::subview(u0_, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          auto des_slice = Kokkos::subview(u_, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          Kokkos::deep_copy(des_slice, src_slice);
          src_slice = Kokkos::subview(w0_, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          des_slice = Kokkos::subview(w_, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          Kokkos::deep_copy(des_slice, src_slice);
        } else if (pmesh->pmb_pack->pmhd != nullptr) {
          auto b = pmesh->pmb_pack->pmhd->b0;
          par_for("zoom_apply", DevExeSpace(),ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
          KOKKOS_LAMBDA(int k, int j, int i) {
            Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
            Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
            Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
            Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
            if (rad < 2.0*rzoom) { // apply to 2*rzoom since rzoom is already updated
              w_(m,IDN,k,j,i) = w0_(zm,IDN,k,j,i);
              w_(m,IM1,k,j,i) = w0_(zm,IM1,k,j,i);
              w_(m,IM2,k,j,i) = w0_(zm,IM2,k,j,i);
              w_(m,IM3,k,j,i) = w0_(zm,IM3,k,j,i);
              w_(m,IEN,k,j,i) = w0_(zm,IEN,k,j,i);
              Real glower[4][4], gupper[4][4];
              ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

              // Load single state of primitive variables
              MHDPrim1D w;
              w.d  = w_(m,IDN,k,j,i);
              w.vx = w_(m,IVX,k,j,i);
              w.vy = w_(m,IVY,k,j,i);
              w.vz = w_(m,IVZ,k,j,i);
              w.e  = w_(m,IEN,k,j,i);

              // load cell-centered fields into primitive state
              // use simple linear average of face-centered fields as bcc is not updated
              w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
              w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
              w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

              // call p2c function
              HydCons1D u;
              SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

              // store conserved quantities in 3D array
              u_(m,IDN,k,j,i) = u.d;
              u_(m,IM1,k,j,i) = u.mx;
              u_(m,IM2,k,j,i) = u.my;
              u_(m,IM3,k,j,i) = u.mz;
              u_(m,IEN,k,j,i) = u.e;
            }
          });
        }
        std::cout << "CyclicZoom: Apply variables for zoom meshblock " << zm << std::endl;
      }
    }
  }
  // if (zid != nleaf*(zstate.zone+1)) {
  //   std::cerr << "Error: CyclicZoom::ApplyVariables() failed: zid = " << zid <<
  //                " zone = " << zstate.zone << " level = " << zamr.level << std::endl;
  //   std::exit(1);
  // }

  return;
}
