//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_newdt.cpp
//! \brief function to compute z4c timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "z4c.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
//! \fn void Z4c::NewTimeStep()
//! \brief calculate the minimum timestep within a MeshBlockPack for z4c problems

TaskStatus Z4c::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;

  auto opt = this->opt;
  auto z4c = this->z4c;
  Kokkos::parallel_reduce("Z4c dt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;

    if (opt.telegraph_lapse) {
      int i = idx%nx1 + indcs.is;
      int j = (idx/nx1)%nx2 + indcs.js;
      int k = (idx/(nx1*nx2))%nx3 + indcs.ks;
      Real gxx=z4c.g_dd(m,0,0,k,j,i), gxy=z4c.g_dd(m,0,1,k,j,i);
      Real gxz=z4c.g_dd(m,0,2,k,j,i), gyy=z4c.g_dd(m,1,1,k,j,i);
      Real gyz=z4c.g_dd(m,1,2,k,j,i), gzz=z4c.g_dd(m,2,2,k,j,i);
      Real det = adm::SpatialDet(gxx,gxy,gxz,gyy,gyz,gzz);
      Real scale = pow(z4c.chi(m,k,j,i),-4.0/opt.chi_psi_power)/det;
      Real inv[3] = {(gyy*gzz-gyz*gyz)*scale, (gxx*gzz-gxz*gxz)*scale,
                      (gxx*gyy-gxy*gxy)*scale};
      Real speed[3];
      for (int a=0; a<3; ++a) {
        speed[a] = fmax(1.0, fabs(opt.lapse_advect*z4c.beta_u(m,a,k,j,i))+
                      sqrt(opt.telegraph_kappa/opt.telegraph_tau*inv[a]));
      }
      min_dt1 = fmin(mbsize.d_view(m).dx1/speed[0], min_dt1);
      min_dt2 = fmin(mbsize.d_view(m).dx2/speed[1], min_dt2);
      min_dt3 = fmin(mbsize.d_view(m).dx3/speed[2], min_dt3);
      min_dt1 = fmin(opt.telegraph_tau,min_dt1);
    }
    min_dt1 = fmin((mbsize.d_view(m).dx1), min_dt1);
    min_dt2 = fmin((mbsize.d_view(m).dx2), min_dt2);
    min_dt3 = fmin((mbsize.d_view(m).dx3), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace z4c
