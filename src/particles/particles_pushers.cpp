//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/adm.hpp"
#include "particles.hpp"

namespace particles {
namespace {

KOKKOS_INLINE_FUNCTION
void SpatialInverseFromADM(const adm::ADM::ADM_vars &adm_vars, int m, int k, int j, int i,
                           Real ginv[3][3]) {
  Real gxx = adm_vars.g_dd(m,0,0,k,j,i);
  Real gxy = adm_vars.g_dd(m,0,1,k,j,i);
  Real gxz = adm_vars.g_dd(m,0,2,k,j,i);
  Real gyy = adm_vars.g_dd(m,1,1,k,j,i);
  Real gyz = adm_vars.g_dd(m,1,2,k,j,i);
  Real gzz = adm_vars.g_dd(m,2,2,k,j,i);
  Real det = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
  Real uxx, uxy, uxz, uyy, uyz, uzz;
  adm::SpatialInv(1.0/det, gxx, gxy, gxz, gyy, gyz, gzz,
                  &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
  ginv[0][0] = uxx; ginv[0][1] = uxy; ginv[0][2] = uxz;
  ginv[1][0] = uxy; ginv[1][1] = uyy; ginv[1][2] = uyz;
  ginv[2][0] = uxz; ginv[2][1] = uyz; ginv[2][2] = uzz;
}

KOKKOS_INLINE_FUNCTION
Real NullMomentumNorm(const Real ginv[3][3], const Real kd[3]) {
  Real ksq = 0.0;
  for (int a=0; a<3; ++a) {
    for (int b=0; b<3; ++b) {
      ksq += ginv[a][b]*kd[a]*kd[b];
    }
  }
  return sqrt(fmax(ksq, 1.0e-300));
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int ie = indcs.ie;
  int je = indcs.je;
  int ke = indcs.ke;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = (pmy_pack->pmesh->dt);
  auto gids = pmy_pack->gids;
  int nmb = pmy_pack->nmb_thispack;

  switch (pusher) {
    case ParticlesPusher::drift:

      par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
      KOKKOS_LAMBDA(const int p) {
        int m = pi(PGID,p) - gids;
        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        pr(IPX,p) += 0.5*dt_*pr(IPVX,p);

        if (multi_d) {
          int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
          pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
        }

        if (three_d) {
          int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
          pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
        }
      });

    break;
  case ParticlesPusher::null_geodesic:
    {
    auto &adm_vars = pmy_pack->padm->adm;
    par_for("part_null_geodesic",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID,p) - gids;
      if ((m < 0) || (m >= nmb)) return;

      int i = static_cast<int>((pr(IPX,p) - mbsize.d_view(m).x1min)/
                               mbsize.d_view(m).dx1) + is;
      i = (i < is) ? is : ((i > ie) ? ie : i);
      int j = js;
      if (multi_d) {
        j = static_cast<int>((pr(IPY,p) - mbsize.d_view(m).x2min)/
                             mbsize.d_view(m).dx2) + js;
        j = (j < js) ? js : ((j > je) ? je : j);
      }
      int k = ks;
      if (three_d) {
        k = static_cast<int>((pr(IPZ,p) - mbsize.d_view(m).x3min)/
                             mbsize.d_view(m).dx3) + ks;
        k = (k < ks) ? ks : ((k > ke) ? ke : k);
      }

      Real kd[3] = {pr(IPVX,p), pr(IPVY,p), three_d ? pr(IPVZ,p) : 0.0};
      Real ginv[3][3];
      SpatialInverseFromADM(adm_vars, m, k, j, i, ginv);
      Real knorm = NullMomentumNorm(ginv, kd);

      Real ku[3] = {0.0, 0.0, 0.0};
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          ku[a] += ginv[a][b]*kd[b];
        }
      }
      Real vx = adm_vars.alpha(m,k,j,i)*ku[0]/knorm - adm_vars.beta_u(m,0,k,j,i);
      Real vy = adm_vars.alpha(m,k,j,i)*ku[1]/knorm - adm_vars.beta_u(m,1,k,j,i);
      Real vz = adm_vars.alpha(m,k,j,i)*ku[2]/knorm - adm_vars.beta_u(m,2,k,j,i);

      Real dk[3] = {0.0, 0.0, 0.0};
      const int qlo[3] = {
        (i-1 < is) ? is : i-1,
        (j-1 < js) ? js : j-1,
        (k-1 < ks) ? ks : k-1
      };
      const int qhi[3] = {
        (i+1 > ie) ? ie : i+1,
        (j+1 > je) ? je : j+1,
        (k+1 > ke) ? ke : k+1
      };
      const Real ddx[3] = {
        (qhi[0] == qlo[0]) ? 1.0 : (qhi[0]-qlo[0])*mbsize.d_view(m).dx1,
        (qhi[1] == qlo[1]) ? 1.0 : (qhi[1]-qlo[1])*mbsize.d_view(m).dx2,
        (qhi[2] == qlo[2]) ? 1.0 : (qhi[2]-qlo[2])*mbsize.d_view(m).dx3
      };

      for (int q=0; q<3; ++q) {
        if ((q == 1 && !(multi_d)) || (q == 2 && !(three_d))) continue;
        int il = (q == 0) ? qlo[0] : i;
        int ih = (q == 0) ? qhi[0] : i;
        int jl = (q == 1) ? qlo[1] : j;
        int jh = (q == 1) ? qhi[1] : j;
        int kl = (q == 2) ? qlo[2] : k;
        int kh = (q == 2) ? qhi[2] : k;

        Real ginv_l[3][3], ginv_h[3][3];
        SpatialInverseFromADM(adm_vars, m, kl, jl, il, ginv_l);
        SpatialInverseFromADM(adm_vars, m, kh, jh, ih, ginv_h);
        Real d_alpha = (adm_vars.alpha(m,kh,jh,ih) -
                        adm_vars.alpha(m,kl,jl,il))/ddx[q];
        Real grad_beta_k = 0.0;
        for (int a=0; a<3; ++a) {
          Real d_beta = (adm_vars.beta_u(m,a,kh,jh,ih) -
                         adm_vars.beta_u(m,a,kl,jl,il))/ddx[q];
          grad_beta_k += d_beta*kd[a];
        }
        Real d_ginv_kk = 0.0;
        for (int a=0; a<3; ++a) {
          for (int b=0; b<3; ++b) {
            d_ginv_kk += ((ginv_h[a][b] - ginv_l[a][b])/ddx[q])*kd[a]*kd[b];
          }
        }
        dk[q] = -d_alpha*knorm - 0.5*adm_vars.alpha(m,k,j,i)*d_ginv_kk/knorm
                + grad_beta_k;
      }

      pr(IPX,p) += dt_*vx;
      if (multi_d) pr(IPY,p) += dt_*vy;
      if (three_d) pr(IPZ,p) += dt_*vz;
      pr(IPVX,p) += dt_*dk[0];
      if (multi_d) pr(IPVY,p) += dt_*dk[1];
      if (three_d) pr(IPVZ,p) += dt_*dk[2];
    });

    break;
    }
  default:
    break;
  }

  return TaskStatus::complete;
}
} // namespace particles
