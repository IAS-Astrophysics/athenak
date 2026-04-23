//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//! \brief function to compute hydro timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for hydrodynamic problems

TaskStatus Hydro::NewTimeStep(Driver *pdrive, int stage) {
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
  auto &eos = pmy_pack->phydro->peos->eos_data;
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
    Kokkos::parallel_reduce("HydroNudt1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
    // find smallest dx/(v +/- Cs) in each direction for hydrodynamic problems
    Kokkos::parallel_reduce("HydroNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
      } else if (is_general_relativistic_) {
        if (!gr_dt_) {
          max_dv1 = 1.0;
          max_dv2 = 1.0;
          max_dv3 = 1.0;
        } else {
          // Use the GR sound speed to compute the time step
          // References to left primitives
          Real &wd = w0_(m,IDN,k,j,i);
          Real &ux = w0_(m,IVX,k,j,i);
          Real &uy = w0_(m,IVY,k,j,i);
          Real &uz = w0_(m,IVZ,k,j,i);

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
        }
      } else if (is_special_relativistic_) {
        Real v2 = SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i)) + SQR(w0_(m,IVZ,k,j,i));
        Real lor = sqrt(1.0 + v2);
        // FIXME ERM: Ideal fluid for now
        Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

        Real lm, lp;
        eos.IdealSRHydroSoundSpeeds(w0_(m,IDN,k,j,i), p, w0_(m,IVX,k,j,i), lor, lp, lm);
        max_dv1 = fmax(fabs(lm), lp);

        eos.IdealSRHydroSoundSpeeds(w0_(m,IDN,k,j,i), p, w0_(m,IVY,k,j,i), lor, lp, lm);
        max_dv2 = fmax(fabs(lm), lp);

        eos.IdealSRHydroSoundSpeeds(w0_(m,IDN,k,j,i), p, w0_(m,IVZ,k,j,i), lor, lp, lm);
        max_dv3 = fmax(fabs(lm), lp);
      } else {
        Real cs;
        if (eos.is_ideal) {
          Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
          cs = eos.IdealHydroSoundSpeed(w0_(m,IDN,k,j,i), p);
        } else         {
          cs = eos.iso_cs;
        }
        max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cs;
        max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cs;
        max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cs;
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
} // namespace hydro
