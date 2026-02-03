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
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
// TODO(@mhguo): check whehther all above includes are necessary

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

  // TODO(@mhguo): this is wrong!!! need to fix or remove
  int zid = (zstate.zone-1);
  Real &f0 = zemf.emf_f0; //(rad-rzoom)/rzoom;
  Real &f1 = zemf.emf_f1; //(rzoom-rad)/rzoom;

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
