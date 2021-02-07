//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_newdt.cpp
//  \brief function to compute MHD timestep across all MeshBlock(s) in a MeshBlockPack

#include <limits>
#include <math.h>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
// \!fn void MHD::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlock for MHD problems

TaskStatus MHD::NewTimeStep(Driver *pdriver, int stage)
{
  if (stage != pdriver->nstages) return TaskStatus::complete; // only execute last stage
  
  int is = pmy_pack->mb_cells.is; int nx1 = pmy_pack->mb_cells.nx1;
  int js = pmy_pack->mb_cells.js; int nx2 = pmy_pack->mb_cells.nx2;
  int ks = pmy_pack->mb_cells.ks; int nx3 = pmy_pack->mb_cells.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  if (pdriver->time_evolution == TimeEvolution::kinematic) {
    auto &w0_ = w0;
    auto &mbsize = pmy_pack->pmb->mbsize;
    const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;

    // find smallest (dx/v) in each direction for advection problems
    Kokkos::parallel_reduce("MHDNudt1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
      {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      min_dt1 = fmin((mbsize.dx1.d_view(m)/fabs(w0_(m,IVX,k,j,i))), min_dt1);
      min_dt2 = fmin((mbsize.dx2.d_view(m)/fabs(w0_(m,IVY,k,j,i))), min_dt2);
      min_dt3 = fmin((mbsize.dx3.d_view(m)/fabs(w0_(m,IVZ,k,j,i))), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
 
  } else {
    auto &w0_ = w0;
    auto &b0_ = b0;
    auto &bcc0_ = bcc0;
    auto &eos = pmy_pack->pmhd->peos->eos_data;
    auto &mbsize = pmy_pack->pmb->mbsize;
    const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;

    // find smallest dx/(v +/- C_fast) in each direction for MHD problems
    Kokkos::parallel_reduce("MHDNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
    { 
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real &w_d = w0_(m,IDN,k,j,i);
      // following cannot be references since they are equated to other arrays below!
      Real w_bx = b0_.x1f(m,k,j,i);
      Real w_by = bcc0_(m,IBY,k,j,i);
      Real w_bz = bcc0_(m,IBZ,k,j,i);
      Real cf;
      if (eos.is_adiabatic) { 
        Real &w_p = w0_(m,IPR,k,j,i);
        cf = eos.FastMagnetosonicSpeed(w_d,w_p,w_bx,w_by,w_bz);
      } else {
        cf = eos.FastMagnetosonicSpeed(w_d,w_bx,w_by,w_bz);
      }
      min_dt1 = fmin((mbsize.dx1.d_view(m)/(fabs(w0_(m,IVX,k,j,i)) + cf)), min_dt1);

      w_bx = b0_.x2f(m,k,j,i);
      w_by = bcc0_(m,IBZ,k,j,i);
      w_bz = bcc0_(m,IBX,k,j,i);
      if (eos.is_adiabatic) { 
        Real &w_p = w0_(m,IPR,k,j,i);
        cf = eos.FastMagnetosonicSpeed(w_d,w_p,w_bx,w_by,w_bz);
      } else {
        cf = eos.FastMagnetosonicSpeed(w_d,w_bx,w_by,w_bz);
      }
      min_dt2 = fmin((mbsize.dx2.d_view(m)/(fabs(w0_(m,IVY,k,j,i)) + cf)), min_dt2);

      w_bx = b0_.x3f(m,k,j,i);
      w_by = bcc0_(m,IBX,k,j,i);
      w_bz = bcc0_(m,IBY,k,j,i);
      if (eos.is_adiabatic) { 
        Real &w_p = w0_(m,IPR,k,j,i);
        cf = eos.FastMagnetosonicSpeed(w_d,w_p,w_bx,w_by,w_bz);
      } else {
        cf = eos.FastMagnetosonicSpeed(w_d,w_bx,w_by,w_bz);
      }
      min_dt3 = fmin((mbsize.dx3.d_view(m)/(fabs(w0_(m,IVZ,k,j,i)) + cf)), min_dt3);

    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->nx2gt1) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->nx3gt1) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace mhd
