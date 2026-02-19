//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_newdt.cpp
//! \brief function to compute MHD timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
// \!fn void MHD::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for MHD problems

TaskStatus MHD::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &w0_ = w0;
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &is_special_relativistic_ = pmy_pack->pcoord->is_special_relativistic;
  auto &is_general_relativistic_ = pmy_pack->pcoord->is_general_relativistic;
  auto &is_dynamical_relativistic_ = pmy_pack->pcoord->is_dynamical_relativistic;
  auto &gr_dt_ = gr_dt;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  if (pdrive->time_evolution == TimeEvolution::kinematic) {
    // find smallest (dx/v) in each direction for advection problems
    Kokkos::parallel_reduce("MHDNudt1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      min_dt1 = fmin((mbsize.d_view(m).dx1/fabs(w0_(m,IVX,k,j,i))), min_dt1);
      min_dt2 = fmin((mbsize.d_view(m).dx2/fabs(w0_(m,IVY,k,j,i))), min_dt2);
      min_dt3 = fmin((mbsize.d_view(m).dx3/fabs(w0_(m,IVZ,k,j,i))), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  } else {
    // find smallest dx/(v +/- Cf) in each direction for mhd problems
    auto &bcc0_ = bcc0;

    Kokkos::parallel_reduce("MHDNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

      if (is_dynamical_relativistic_) {
        max_dv1 = 1.0;
        max_dv2 = 1.0;
        max_dv3 = 1.0;
      // timestep in GR MHD
      } else if (is_general_relativistic_) {
        if (!gr_dt_) {
          max_dv1 = 1.0;
          max_dv2 = 1.0;
          max_dv3 = 1.0;
        } else {
          // Use the GR fast magnetosonic speed to compute the time step
          // References to left primitives
          Real &wd = w0_(m,IDN,k,j,i);
          Real &ux = w0_(m,IVX,k,j,i);
          Real &uy = w0_(m,IVY,k,j,i);
          Real &uz = w0_(m,IVZ,k,j,i);
          Real &bcc1 = bcc0_(m,IBX,k,j,i);
          Real &bcc2 = bcc0_(m,IBY,k,j,i);
          Real &bcc3 = bcc0_(m,IBZ,k,j,i);

          // FIXME MG: Ideal fluid for now
          Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

          // Extract components of metric
          Real &x1min = mbsize.d_view(m).x1min;
          Real &x1max = mbsize.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          Real &x2min = mbsize.d_view(m).x2min;
          Real &x2max = mbsize.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
          Real &x3min = mbsize.d_view(m).x3min;
          Real &x3max = mbsize.d_view(m).x3max;
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
        }
      // timestep in SR MHD
      } else if (is_special_relativistic_) {
        Real &wd = w0_(m,IDN,k,j,i);
        Real &ux = w0_(m,IVX,k,j,i);
        Real &uy = w0_(m,IVY,k,j,i);
        Real &uz = w0_(m,IVZ,k,j,i);
        Real &bcc1 = bcc0_(m,IBX,k,j,i);
        Real &bcc2 = bcc0_(m,IBY,k,j,i);
        Real &bcc3 = bcc0_(m,IBZ,k,j,i);

        Real v2 = SQR(ux) + SQR(uy) + SQR(uz);
        Real lor = sqrt(1.0 + v2);
        // FIXME ERM: Ideal fluid for now
        Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
        // Calculate 4-magnetic field in left state
        Real b_0 = bcc1*ux + bcc2*uy + bcc3*uz;
        Real b_1 = (bcc1 + b_0 * ux) / lor;
        Real b_2 = (bcc2 + b_0 * uy) / lor;
        Real b_3 = (bcc3 + b_0 * uz) / lor;
        Real b_sq = -SQR(b_0) + SQR(b_1) + SQR(b_2) + SQR(b_3);

        Real lm, lp;
        eos.IdealSRMHDFastSpeeds(wd, p, ux, lor, b_sq, lp, lm);
        max_dv1 = fmax(fabs(lm), lp);

        eos.IdealSRMHDFastSpeeds(wd, p, uy, lor, b_sq, lp, lm);
        max_dv2 = fmax(fabs(lm), lp);

        eos.IdealSRMHDFastSpeeds(wd, p, uz, lor, b_sq, lp, lm);
        max_dv3 = fmax(fabs(lm), lp);
      // timestep in Newtonian MHD
      } else {
        Real &w_d = w0_(m,IDN,k,j,i);
        Real &w_bx = bcc0_(m,IBX,k,j,i);
        Real &w_by = bcc0_(m,IBY,k,j,i);
        Real &w_bz = bcc0_(m,IBZ,k,j,i);
        Real cf;
        if (eos.is_ideal) {
          Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
          cf = eos.IdealMHDFastSpeed(w_d, p, w_bx, w_by, w_bz);
          max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cf;
          cf = eos.IdealMHDFastSpeed(w_d, p, w_by, w_bz, w_bx);
          max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cf;
          cf = eos.IdealMHDFastSpeed(w_d, p, w_bz, w_bx, w_by);
          max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cf;
        } else {
          cf = eos.IdealMHDFastSpeed(w_d, w_bx, w_by, w_bz);
          max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cf;
          cf = eos.IdealMHDFastSpeed(w_d, w_by, w_bz, w_bx);
          max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cf;
          cf = eos.IdealMHDFastSpeed(w_d, w_bz, w_bx, w_by);
          max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cf;
        }
      }
      min_dt1 = fmin((mbsize.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((mbsize.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((mbsize.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  // compute timestep for diffusion
  if (pcond != nullptr) {
    pcond->NewTimeStep(w0, peos->eos_data);
  }
  // compute source terms timestep
  if (psrc != nullptr) {
    psrc->NewTimeStep(w0, peos->eos_data);
  }

  return TaskStatus::complete;
}
} // namespace mhd
