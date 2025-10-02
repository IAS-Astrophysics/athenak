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

      if (is_general_relativistic_ || is_dynamical_relativistic_) {
        max_dv1 = 1.0;
        max_dv2 = 1.0;
        max_dv3 = 1.0;
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
