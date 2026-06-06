//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tetrad.cpp
//  \brief sets orthonormal tetrad

#include <math.h>
#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "dyn_radiation.hpp"
#include "dyn_radiation/dyn_radiation_tetrad.hpp"

namespace dyn_radiation {

constexpr Real adm_metric_floor = 1.0e-30;

KOKKOS_INLINE_FUNCTION
int Sym3Index(const int a, const int b) {
  const int i = (a < b) ? a : b;
  const int j = (a < b) ? b : a;
  if (i == 0 && j == 0) return 0;
  if (i == 0 && j == 1) return 1;
  if (i == 0 && j == 2) return 2;
  if (i == 1 && j == 1) return 3;
  if (i == 1 && j == 2) return 4;
  return 5;
}

KOKKOS_INLINE_FUNCTION
void BuildADMSpatialTriad(const Real gxx, const Real gxy, const Real gxz,
                          const Real gyy, const Real gyz, const Real gzz,
                          Real e[3][3]) {
  // Cholesky factorization gamma_ij = L_iA L_jA.  The columns of e=L^{-T}
  // map orthonormal-frame direction cosines into coordinate spatial vectors.
  const Real l00 = sqrt(fmax(gxx, adm_metric_floor));
  const Real l10 = gxy/l00;
  const Real l20 = gxz/l00;
  const Real l11 = sqrt(fmax(gyy - SQR(l10), adm_metric_floor));
  const Real l21 = (gyz - l20*l10)/l11;
  const Real l22 = sqrt(fmax(gzz - SQR(l20) - SQR(l21), adm_metric_floor));

  e[0][0] = 1.0/l00;
  e[1][0] = 0.0;
  e[2][0] = 0.0;

  e[0][1] = -l10/(l00*l11);
  e[1][1] = 1.0/l11;
  e[2][1] = 0.0;

  e[0][2] = l10*l21/(l00*l11*l22) - l20/(l00*l22);
  e[1][2] = -l21/(l11*l22);
  e[2][2] = 1.0/l22;
}

KOKKOS_INLINE_FUNCTION
void BuildADMCoTriad(const Real gxx, const Real gxy, const Real gxz,
                     const Real gyy, const Real gyz, const Real gzz,
                     Real co[3][3]) {
  // Cholesky factor gamma_ij = L_iA L_jA.  co[A][i] = L_iA maps
  // coordinate spatial vectors into the Eulerian orthonormal frame.
  const Real l00 = sqrt(fmax(gxx, adm_metric_floor));
  const Real l10 = gxy/l00;
  const Real l20 = gxz/l00;
  const Real l11 = sqrt(fmax(gyy - SQR(l10), adm_metric_floor));
  const Real l21 = (gyz - l20*l10)/l11;
  const Real l22 = sqrt(fmax(gzz - SQR(l20) - SQR(l21), adm_metric_floor));

  co[0][0] = l00;
  co[0][1] = l10;
  co[0][2] = l20;

  co[1][0] = 0.0;
  co[1][1] = l11;
  co[1][2] = l21;

  co[2][0] = 0.0;
  co[2][1] = 0.0;
  co[2][2] = l22;
}

KOKKOS_INLINE_FUNCTION
void BuildADMCoTriadDerivative(const Real gxx, const Real gxy, const Real gxz,
                               const Real gyy, const Real gyz, const Real gzz,
                               const Real dgxx, const Real dgxy, const Real dgxz,
                               const Real dgyy, const Real dgyz, const Real dgzz,
                               Real dco[3][3]) {
  const Real l00 = sqrt(fmax(gxx, adm_metric_floor));
  const Real l10 = gxy/l00;
  const Real l20 = gxz/l00;
  const Real l11 = sqrt(fmax(gyy - SQR(l10), adm_metric_floor));
  const Real l21 = (gyz - l20*l10)/l11;
  const Real l22 = sqrt(fmax(gzz - SQR(l20) - SQR(l21), adm_metric_floor));

  const Real dl00 = 0.5*dgxx/l00;
  const Real dl10 = (dgxy - l10*dl00)/l00;
  const Real dl20 = (dgxz - l20*dl00)/l00;
  const Real dl11 = 0.5*(dgyy - 2.0*l10*dl10)/l11;
  const Real dl21 = (dgyz - dl20*l10 - l20*dl10 - l21*dl11)/l11;
  const Real dl22 = 0.5*(dgzz - 2.0*l20*dl20 - 2.0*l21*dl21)/l22;

  dco[0][0] = dl00;
  dco[0][1] = dl10;
  dco[0][2] = dl20;

  dco[1][0] = 0.0;
  dco[1][1] = dl11;
  dco[1][2] = dl21;

  dco[2][0] = 0.0;
  dco[2][1] = 0.0;
  dco[2][2] = dl22;
}

KOKKOS_INLINE_FUNCTION
Real ADMDetSqrt(const Real gxx, const Real gxy, const Real gxz,
                const Real gyy, const Real gyz, const Real gzz) {
  return sqrt(fmax(adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz), adm_metric_floor));
}

KOKKOS_INLINE_FUNCTION
void BuildADMEulerianTetrad(const Real alpha, const Real beta[3], const Real g3d[6],
                            Real e[4][4], Real e_cov[4][4]) {
  Real triad[3][3];
  BuildADMSpatialTriad(g3d[S11], g3d[S12], g3d[S13],
                       g3d[S22], g3d[S23], g3d[S33], triad);

  for (int a=0; a<4; ++a) {
    for (int mu=0; mu<4; ++mu) {
      e[a][mu] = 0.0;
      e_cov[a][mu] = 0.0;
    }
  }

  e[0][0] = 1.0/alpha;
  for (int d=0; d<3; ++d) {
    e[0][d+1] = -beta[d]/alpha;
  }
  for (int a=0; a<3; ++a) {
    for (int d=0; d<3; ++d) {
      e[a+1][d+1] = triad[d][a];
    }
  }

  Real g4[16];
  adm::SpacetimeMetric(alpha, beta[0], beta[1], beta[2],
                       g3d[S11], g3d[S12], g3d[S13],
                       g3d[S22], g3d[S23], g3d[S33], g4);
  for (int a=0; a<4; ++a) {
    for (int mu=0; mu<4; ++mu) {
      for (int nu=0; nu<4; ++nu) {
        e_cov[a][mu] += g4[4*mu + nu]*e[a][nu];
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void BuildADMFaceTransportCoeffs(const int dir, const Real alpha, const Real beta[3],
                                 const Real g3d[6], Real coeff[4]) {
  Real triad[3][3];
  BuildADMSpatialTriad(g3d[S11], g3d[S12], g3d[S13],
                       g3d[S22], g3d[S23], g3d[S33], triad);
  coeff[0] = -beta[dir];
  for (int a=0; a<3; ++a) {
    coeff[a+1] = alpha*triad[dir][a];
  }
}

KOKKOS_INLINE_FUNCTION
Real ADMGeodesicAngularSpeed(const Real ell[3], const Real unit_flux[2],
                             const Real alpha, const Real beta[3],
                             const Real triad[3][3], const Real cotriad[3][3],
                             const Real grad_alpha[3],
                             const Real grad_beta[3][3],
                             const Real grad_guu[3][3][3],
                             const Real grad_cotriad[3][3][3],
                             const Real dt_cotriad[3][3]) {
  const Real ell0 = ell[0];
  const Real ell1 = ell[1];
  const Real ell2 = ell[2];
  Real p[3];
  Real s[3];
  for (int i=0; i<3; ++i) {
    p[i] = cotriad[0][i]*ell0 + cotriad[1][i]*ell1 + cotriad[2][i]*ell2;
    s[i] = triad[i][0]*ell0 + triad[i][1]*ell1 + triad[i][2]*ell2;
  }

  Real v[3] = {alpha*s[0] - beta[0],
               alpha*s[1] - beta[1],
               alpha*s[2] - beta[2]};

  Real q[3];
  for (int i=0; i<3; ++i) {
    Real pdot = -grad_alpha[i];
    for (int j=0; j<3; ++j) {
      pdot += p[j]*grad_beta[i][j];
      for (int k=0; k<3; ++k) {
        pdot -= 0.5*alpha*p[j]*p[k]*grad_guu[i][j][k];
      }
    }
    Real frame_adv = 0.0;
    for (int b=0; b<3; ++b) {
      Real dcov = dt_cotriad[b][i];
      for (int d=0; d<3; ++d) {
        dcov += v[d]*grad_cotriad[d][b][i];
      }
      frame_adv += dcov*ell[b];
    }
    q[i] = pdot - frame_adv;
  }

  Real elldot[3];
  for (int a=0; a<3; ++a) {
    elldot[a] = triad[0][a]*q[0] + triad[1][a]*q[1] + triad[2][a]*q[2];
  }

  // Remove roundoff-level radial drift so the angular flux is tangent to S^2.
  Real radial = ell0*elldot[0] + ell1*elldot[1] + ell2*elldot[2];
  for (int a=0; a<3; ++a) {
    elldot[a] -= radial*ell[a];
  }

  const Real sin2 = fmax(1.0 - SQR(ell2), 1.0e-300);
  const Real theta_dot = -elldot[2]/sqrt(sin2);
  const Real sin2_psi_dot = ell0*elldot[1] - ell1*elldot[0];
  return theta_dot*unit_flux[0] + sin2_psi_dot*unit_flux[1];
}

KOKKOS_INLINE_FUNCTION
void BuildADMGeodesicAngularCoeffs(const Real alpha, const Real beta[3],
                                   const Real triad[3][3], const Real cotriad[3][3],
                                   const Real grad_alpha[3],
                                   const Real grad_beta[3][3],
                                   const Real grad_guu[3][3][3],
                                   const Real grad_cotriad[3][3][3],
                                   const Real dt_cotriad[3][3],
                                   Real q0[3], Real q1[3][3], Real q2[3][3][3]) {
  for (int i=0; i<3; ++i) {
    q0[i] = -grad_alpha[i];
    for (int a=0; a<3; ++a) {
      q1[i][a] = -dt_cotriad[a][i];
      q2[i][a][0] = 0.0;
      q2[i][a][1] = 0.0;
      q2[i][a][2] = 0.0;
    }
  }

  for (int i=0; i<3; ++i) {
    for (int a=0; a<3; ++a) {
      for (int j=0; j<3; ++j) {
        q1[i][a] += cotriad[a][j]*grad_beta[i][j];
      }
      for (int d=0; d<3; ++d) {
        q1[i][a] += beta[d]*grad_cotriad[d][a][i];
      }
    }
  }

  for (int i=0; i<3; ++i) {
    for (int a=0; a<3; ++a) {
      for (int b=0; b<3; ++b) {
        Real guu_coeff = 0.0;
        for (int j=0; j<3; ++j) {
          for (int k=0; k<3; ++k) {
            guu_coeff += cotriad[a][j]*cotriad[b][k]*grad_guu[i][j][k];
          }
        }
        q2[i][a][b] -= 0.5*alpha*guu_coeff;
        for (int d=0; d<3; ++d) {
          q2[i][a][b] -= alpha*triad[d][a]*grad_cotriad[d][b][i];
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real ADMGeodesicAngularSpeedFromCoeffs(const Real ell[3], const Real unit_flux[2],
                                       const Real triad[3][3],
                                       const Real q0[3], const Real q1[3][3],
                                       const Real q2[3][3][3]) {
  const Real ell0 = ell[0];
  const Real ell1 = ell[1];
  const Real ell2 = ell[2];
  Real q[3];
  for (int i=0; i<3; ++i) {
    q[i] = q0[i];
    for (int a=0; a<3; ++a) {
      q[i] += q1[i][a]*ell[a];
      for (int b=0; b<3; ++b) {
        q[i] += q2[i][a][b]*ell[a]*ell[b];
      }
    }
  }

  Real elldot[3];
  for (int a=0; a<3; ++a) {
    elldot[a] = triad[0][a]*q[0] + triad[1][a]*q[1] + triad[2][a]*q[2];
  }

  Real radial = ell0*elldot[0] + ell1*elldot[1] + ell2*elldot[2];
  for (int a=0; a<3; ++a) {
    elldot[a] -= radial*ell[a];
  }

  const Real sin2 = fmax(1.0 - SQR(ell2), 1.0e-300);
  const Real theta_dot = -elldot[2]/sqrt(sin2);
  const Real sin2_psi_dot = ell0*elldot[1] - ell1*elldot[0];
  return theta_dot*unit_flux[0] + sin2_psi_dot*unit_flux[1];
}

//----------------------------------------------------------------------------------------
//! \fn  void DynRadiation::SetOrthonormalTetrad()
//! \brief Set orthonormal tetrad data

void DynRadiation::SetOrthonormalTetrad() {
  auto &size = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;

  int nang1 = prgeo->nangles - 1;
  auto &num_neighbors_ = prgeo->num_neighbors;
  auto nh_c_ = nh_c;

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;

  if (!angular_frame_initialized) {
    // define tetrad frame
    for (int n=0; n<=nang1; ++n) {
      nh_c.h_view(n,0) = 1.0;
      nh_c.h_view(n,1) = prgeo->cart_pos.h_view(n,0);
      nh_c.h_view(n,2) = prgeo->cart_pos.h_view(n,1);
      nh_c.h_view(n,3) = prgeo->cart_pos.h_view(n,2);
      if (angular_fluxes) {
        for (int nb=0; nb<num_neighbors_.h_view(n); ++nb) {
          nh_f.h_view(n,nb,0) = 1.0;
          nh_f.h_view(n,nb,1) = prgeo->cart_pos_mid.h_view(n,nb,0);
          nh_f.h_view(n,nb,2) = prgeo->cart_pos_mid.h_view(n,nb,1);
          nh_f.h_view(n,nb,3) = prgeo->cart_pos_mid.h_view(n,nb,2);
        }
        if (num_neighbors_.h_view(n)==5) {
          nh_f.h_view(n,5,0) = (FLT_MAX);
          nh_f.h_view(n,5,1) = (FLT_MAX);
          nh_f.h_view(n,5,2) = (FLT_MAX);
          nh_f.h_view(n,5,3) = (FLT_MAX);
        }
      }
    }
    nh_c.template modify<HostMemSpace>();
    nh_c.template sync<DevExeSpace>();
    nh_f.template modify<HostMemSpace>();
    nh_f.template sync<DevExeSpace>();
    angular_frame_initialized = true;
  }

  if (use_adm_geometry) {
    auto &adm_ = pmy_pack->padm->adm;
    auto tet_c_ = tet_c;
    auto tetcov_c_ = tetcov_c;
    auto sqrt_detg_c_ = sqrt_detg_c;
    auto adm_alpha_c_ = adm_alpha_c;
    auto adm_beta_u_c_ = adm_beta_u_c;
    auto adm_g_dd_c_ = adm_g_dd_c;
    auto adm_g_uu_c_ = adm_g_uu_c;
    auto adm_K_dd_c_ = adm_K_dd_c;
    auto adm_cotriad_c_ = adm_cotriad_c;
    par_for("dynrad_adm_tet_c",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real beta[3] = {adm_.beta_u(m,0,k,j,i),
                      adm_.beta_u(m,1,k,j,i),
                      adm_.beta_u(m,2,k,j,i)};
      Real g3d[6] = {adm_.g_dd(m,0,0,k,j,i), adm_.g_dd(m,0,1,k,j,i),
                     adm_.g_dd(m,0,2,k,j,i), adm_.g_dd(m,1,1,k,j,i),
                     adm_.g_dd(m,1,2,k,j,i), adm_.g_dd(m,2,2,k,j,i)};
      Real e[4][4], e_cov[4][4];
      BuildADMEulerianTetrad(adm_.alpha(m,k,j,i), beta, g3d, e, e_cov);
      Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                  g3d[S22], g3d[S23], g3d[S33]);
      Real detginv = 1.0/fmax(detg, adm_metric_floor);
      Real guu[6];
      adm::SpatialInv(detginv, g3d[S11], g3d[S12], g3d[S13],
                      g3d[S22], g3d[S23], g3d[S33],
                      &guu[S11], &guu[S12], &guu[S13],
                      &guu[S22], &guu[S23], &guu[S33]);
      Real cotriad[3][3];
      BuildADMCoTriad(g3d[S11], g3d[S12], g3d[S13],
                      g3d[S22], g3d[S23], g3d[S33], cotriad);
      for (int d1=0; d1<4; ++d1) {
        for (int d2=0; d2<4; ++d2) {
          tet_c_   (m,d1,d2,k,j,i) = e[d1][d2];
          tetcov_c_(m,d1,d2,k,j,i) = e_cov[d1][d2];
        }
      }
      sqrt_detg_c_(m,k,j,i) = sqrt(fmax(detg, adm_metric_floor));
      adm_alpha_c_(m,k,j,i) = adm_.alpha(m,k,j,i);
      for (int a=0; a<3; ++a) {
        adm_beta_u_c_(m,a,k,j,i) = beta[a];
        for (int b=0; b<3; ++b) {
          adm_g_dd_c_(m,a,b,k,j,i) = adm_.g_dd(m,a,b,k,j,i);
          adm_g_uu_c_(m,a,b,k,j,i) = guu[Sym3Index(a,b)];
          adm_K_dd_c_(m,a,b,k,j,i) = adm_.vK_dd(m,a,b,k,j,i);
          adm_cotriad_c_(m,a,b,k,j,i) = cotriad[a][b];
        }
      }
    });

    auto adm_grad_alpha_c_ = adm_grad_alpha_c;
    auto adm_grad_beta_u_c_ = adm_grad_beta_u_c;
    auto adm_grad_g_dd_c_ = adm_grad_g_dd_c;
    auto adm_grad_g_uu_c_ = adm_grad_g_uu_c;
    auto adm_grad_cotriad_c_ = adm_grad_cotriad_c;
    auto adm_dt_cotriad_c_ = adm_dt_cotriad_c;
    bool multi_d = pmy_pack->pmesh->multi_d;
    bool three_d = pmy_pack->pmesh->three_d;
    par_for("dynrad_adm_grad_cache",DevExeSpace(),
    0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      int im[3] = {(i > 0) ? i-1 : i, j, k};
      int ip[3] = {(i < n1-1) ? i+1 : i, j, k};
      int jm[3] = {i, (j > 0) ? j-1 : j, k};
      int jp[3] = {i, (j < n2-1) ? j+1 : j, k};
      int km[3] = {i, j, (k > 0) ? k-1 : k};
      int kp[3] = {i, j, (k < n3-1) ? k+1 : k};
      Real inv_dx[3] = {
        1.0/(size.d_view(m).dx1*((ip[0] == im[0]) ? 1.0 : static_cast<Real>(ip[0]-im[0]))),
        multi_d ? 1.0/(size.d_view(m).dx2*((jp[1] == jm[1]) ? 1.0 :
                      static_cast<Real>(jp[1]-jm[1]))) : 0.0,
        three_d ? 1.0/(size.d_view(m).dx3*((kp[2] == km[2]) ? 1.0 :
                       static_cast<Real>(kp[2]-km[2]))) : 0.0
      };
      int lo[3][3] = {{im[0], im[1], im[2]}, {jm[0], jm[1], jm[2]},
                      {km[0], km[1], km[2]}};
      int hi[3][3] = {{ip[0], ip[1], ip[2]}, {jp[0], jp[1], jp[2]},
                      {kp[0], kp[1], kp[2]}};

      Real grad_gdd[3][3][3];
      Real grad_beta[3][3];
      for (int d=0; d<3; ++d) {
        const bool active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
        adm_grad_alpha_c_(m,d,k,j,i) = active ?
          (adm_alpha_c_(m,hi[d][2],hi[d][1],hi[d][0]) -
           adm_alpha_c_(m,lo[d][2],lo[d][1],lo[d][0]))*inv_dx[d] : 0.0;
        for (int a=0; a<3; ++a) {
          grad_beta[d][a] = active ?
            (adm_beta_u_c_(m,a,hi[d][2],hi[d][1],hi[d][0]) -
             adm_beta_u_c_(m,a,lo[d][2],lo[d][1],lo[d][0]))*inv_dx[d] : 0.0;
          adm_grad_beta_u_c_(m,3*d+a,k,j,i) = grad_beta[d][a];
          for (int b=0; b<3; ++b) {
            const int sym = Sym3Index(a,b);
            grad_gdd[d][a][b] = active ?
              (adm_g_dd_c_(m,a,b,hi[d][2],hi[d][1],hi[d][0]) -
               adm_g_dd_c_(m,a,b,lo[d][2],lo[d][1],lo[d][0]))*inv_dx[d] : 0.0;
            adm_grad_g_dd_c_(m,6*d+sym,k,j,i) = grad_gdd[d][a][b];
            adm_grad_g_uu_c_(m,6*d+sym,k,j,i) = active ?
              (adm_g_uu_c_(m,a,b,hi[d][2],hi[d][1],hi[d][0]) -
               adm_g_uu_c_(m,a,b,lo[d][2],lo[d][1],lo[d][0]))*inv_dx[d] : 0.0;
          }
        }
      }

      for (int d=0; d<3; ++d) {
        for (int a=0; a<3; ++a) {
          for (int b=0; b<3; ++b) {
            adm_grad_cotriad_c_(m,9*d+3*a+b,k,j,i) =
              ((d == 0) || (d == 1 && multi_d) || (d == 2 && three_d)) ?
              (adm_cotriad_c_(m,a,b,hi[d][2],hi[d][1],hi[d][0]) -
               adm_cotriad_c_(m,a,b,lo[d][2],lo[d][1],lo[d][0]))*inv_dx[d] : 0.0;
          }
        }
      }

      Real beta_d[3] = {0.0, 0.0, 0.0};
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          beta_d[a] += adm_g_dd_c_(m,a,b,k,j,i)*adm_beta_u_c_(m,b,k,j,i);
        }
      }
      Real grad_beta_d[3][3];
      for (int d=0; d<3; ++d) {
        for (int a=0; a<3; ++a) {
          grad_beta_d[d][a] = 0.0;
          for (int b=0; b<3; ++b) {
            grad_beta_d[d][a] += grad_gdd[d][a][b]*adm_beta_u_c_(m,b,k,j,i)
                               + adm_g_dd_c_(m,a,b,k,j,i)*grad_beta[d][b];
          }
        }
      }

      Real dgdt[3][3];
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          Real conn_ab = 0.0;
          Real conn_ba = 0.0;
          for (int c=0; c<3; ++c) {
            Real gamma_cab = 0.0;
            Real gamma_cba = 0.0;
            for (int e=0; e<3; ++e) {
              gamma_cab += 0.5*adm_g_uu_c_(m,c,e,k,j,i)*
                (grad_gdd[a][b][e] + grad_gdd[b][a][e] - grad_gdd[e][a][b]);
              gamma_cba += 0.5*adm_g_uu_c_(m,c,e,k,j,i)*
                (grad_gdd[b][a][e] + grad_gdd[a][b][e] - grad_gdd[e][b][a]);
            }
            conn_ab += gamma_cab*beta_d[c];
            conn_ba += gamma_cba*beta_d[c];
          }
          Real d_a_beta_b = grad_beta_d[a][b] - conn_ab;
          Real d_b_beta_a = grad_beta_d[b][a] - conn_ba;
          dgdt[a][b] = -2.0*adm_alpha_c_(m,k,j,i)*adm_K_dd_c_(m,a,b,k,j,i)
                     + d_a_beta_b + d_b_beta_a;
        }
      }

      Real dco_dt[3][3];
      BuildADMCoTriadDerivative(adm_g_dd_c_(m,0,0,k,j,i),
                                adm_g_dd_c_(m,0,1,k,j,i),
                                adm_g_dd_c_(m,0,2,k,j,i),
                                adm_g_dd_c_(m,1,1,k,j,i),
                                adm_g_dd_c_(m,1,2,k,j,i),
                                adm_g_dd_c_(m,2,2,k,j,i),
                                dgdt[0][0], dgdt[0][1], dgdt[0][2],
                                dgdt[1][1], dgdt[1][2], dgdt[2][2],
                                dco_dt);
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          adm_dt_cotriad_c_(m,3*a+b,k,j,i) = dco_dt[a][b];
        }
      }

    });

    if (is_hydro_enabled || is_mhd_enabled) {
      auto norm_to_tet_ = norm_to_tet;
      par_for("dynrad_adm_norm_to_tet",DevExeSpace(),
      0,(nmb-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        for (int a=0; a<4; ++a) {
          for (int b=0; b<4; ++b) {
            norm_to_tet_(m,a,b,k,j,i) = 0.0;
          }
        }
        norm_to_tet_(m,0,0,k,j,i) = 1.0;
        for (int a=0; a<3; ++a) {
          for (int b=0; b<3; ++b) {
            norm_to_tet_(m,a+1,b+1,k,j,i) = adm_cotriad_c_(m,a,b,k,j,i);
          }
        }
      });
    }

    if (angular_fluxes) {
      auto uflux = prgeo->unit_flux;
      auto nh_f_ = nh_f;
      auto na_ = na;
      par_for("dynrad_adm_na",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real beta[3] = {adm_beta_u_c_(m,0,k,j,i), adm_beta_u_c_(m,1,k,j,i),
                        adm_beta_u_c_(m,2,k,j,i)};
        Real triad[3][3];
        Real cotriad[3][3];
        for (int a=0; a<3; ++a) {
          for (int b=0; b<3; ++b) {
            triad[b][a] = tet_c_(m,a+1,b+1,k,j,i);
            cotriad[a][b] = adm_cotriad_c_(m,a,b,k,j,i);
          }
        }
        Real grad_alpha[3] = {adm_grad_alpha_c_(m,0,k,j,i),
                              adm_grad_alpha_c_(m,1,k,j,i),
                              adm_grad_alpha_c_(m,2,k,j,i)};
        Real grad_beta[3][3];
        Real grad_guu[3][3][3];
        Real grad_cotriad[3][3][3];
        Real dt_cotriad[3][3];
        for (int d=0; d<3; ++d) {
          for (int a=0; a<3; ++a) {
            grad_beta[d][a] = adm_grad_beta_u_c_(m,3*d+a,k,j,i);
            for (int b=0; b<3; ++b) {
              grad_guu[d][a][b] = adm_grad_g_uu_c_(m,6*d+Sym3Index(a,b),k,j,i);
              grad_cotriad[d][a][b] = adm_grad_cotriad_c_(m,9*d+3*a+b,k,j,i);
            }
          }
        }
        for (int a=0; a<3; ++a) {
          for (int b=0; b<3; ++b) {
            dt_cotriad[a][b] = adm_dt_cotriad_c_(m,3*a+b,k,j,i);
          }
        }
        Real q0[3];
        Real q1[3][3];
        Real q2[3][3][3];
        BuildADMGeodesicAngularCoeffs(adm_alpha_c_(m,k,j,i), beta, triad, cotriad,
                                      grad_alpha, grad_beta, grad_guu, grad_cotriad,
                                      dt_cotriad, q0, q1, q2);
        for (int n=0; n<=nang1; ++n) {
          for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
            Real ell[3] = {nh_f_.d_view(n,nb,1), nh_f_.d_view(n,nb,2),
                           nh_f_.d_view(n,nb,3)};
            Real edge_flux[2] = {uflux.d_view(n,nb,0), uflux.d_view(n,nb,1)};
            na_(m,n,k,j,i,nb) =
              ADMGeodesicAngularSpeedFromCoeffs(ell, edge_flux, triad, q0, q1, q2);
          }
        }
      });
    }

    auto tet_d1_x1f_ = tet_d1_x1f;
    auto sqrt_detg_x1f_ = sqrt_detg_x1f;
    par_for("dynrad_adm_x1f",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,(ie+1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real g3d[6], beta[3], alpha;
      adm::Face1Metric(m, k, j, i, adm_.g_dd, adm_.beta_u, adm_.alpha, g3d, beta, alpha);
      Real coeff[4];
      BuildADMFaceTransportCoeffs(0, alpha, beta, g3d, coeff);
      for (int d=0; d<4; ++d) { tet_d1_x1f_(m,d,k,j,i) = coeff[d]; }
      sqrt_detg_x1f_(m,k,j,i) = ADMDetSqrt(g3d[S11], g3d[S12], g3d[S13],
                                           g3d[S22], g3d[S23], g3d[S33]);
    });

    auto tet_d2_x2f_ = tet_d2_x2f;
    auto sqrt_detg_x2f_ = sqrt_detg_x2f;
    if (pmy_pack->pmesh->multi_d) {
      par_for("dynrad_adm_x2f",DevExeSpace(),0,(nmb-1),ks,ke,js,(je+1),is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real g3d[6], beta[3], alpha;
        adm::Face2Metric(m, k, j, i, adm_.g_dd, adm_.beta_u, adm_.alpha, g3d, beta, alpha);
        Real coeff[4];
        BuildADMFaceTransportCoeffs(1, alpha, beta, g3d, coeff);
        for (int d=0; d<4; ++d) { tet_d2_x2f_(m,d,k,j,i) = coeff[d]; }
        sqrt_detg_x2f_(m,k,j,i) = ADMDetSqrt(g3d[S11], g3d[S12], g3d[S13],
                                             g3d[S22], g3d[S23], g3d[S33]);
      });
    }

    auto tet_d3_x3f_ = tet_d3_x3f;
    auto sqrt_detg_x3f_ = sqrt_detg_x3f;
    if (pmy_pack->pmesh->three_d) {
      par_for("dynrad_adm_x3f",DevExeSpace(),0,(nmb-1),ks,(ke+1),js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real g3d[6], beta[3], alpha;
        adm::Face3Metric(m, k, j, i, adm_.g_dd, adm_.beta_u, adm_.alpha, g3d, beta, alpha);
        Real coeff[4];
        BuildADMFaceTransportCoeffs(2, alpha, beta, g3d, coeff);
        for (int d=0; d<4; ++d) { tet_d3_x3f_(m,d,k,j,i) = coeff[d]; }
        sqrt_detg_x3f_(m,k,j,i) = ADMDetSqrt(g3d[S11], g3d[S12], g3d[S13],
                                             g3d[S22], g3d[S23], g3d[S33]);
      });
    }

    return;
  }

  // set tetrad components
  auto tet_c_ = tet_c;
  auto tetcov_c_ = tetcov_c;
  par_for("tet_c/tetcov_c",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d1=0; d1<4; ++d1) {
      for (int d2=0; d2<4; ++d2) {
        tet_c_   (m,d1,d2,k,j,i) = e[d1][d2];
        tetcov_c_(m,d1,d2,k,j,i) = e_cov[d1][d2];
      }
    }
  });

  // set tetrad components (subset) at x1f
  auto tet_d1_x1f_ = tet_d1_x1f;
  par_for("tet_d1_x1f",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,n1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1f,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1f,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1f,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d1_x1f_(m,d,k,j,i) = e[d][1]; }
  });

  // set tetrad components (subset) at x2f
  auto tet_d2_x2f_ = tet_d2_x2f;
  par_for("tet_d2_x2f",DevExeSpace(),0,(nmb-1),0,(n3-1),0,n2,0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2f,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2f,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2f,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d2_x2f_(m,d,k,j,i) = e[d][2]; }
  });

  // set tetrad components (subset) at x3f
  auto tet_d3_x3f_ = tet_d3_x3f;
  par_for("tet_d3_x3f",DevExeSpace(),0,(nmb-1),0,n3,0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3f,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3f,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3f,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d3_x3f_(m,d,k,j,i) = e[d][3]; }
  });

  // Calculate n^angle
  if (angular_fluxes) {
    auto uflux = prgeo->unit_flux;
    auto nh_f_ = nh_f;
    auto na_ = na;
    par_for("na",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      Real dgx[4][4], dgy[4][4], dgz[4][4];
      ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
      Real e[4][4], e_cov[4][4], omega[4][4][4];
      ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
      for (int n=0; n<=nang1; ++n) {
        for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
          Real iszetaf = 1.0/sqrt(1.0 - SQR(nh_f_.d_view(n,nb,3)));
          Real na1 = 0.0; Real na2 = 0.0;
          for (int q=0; q<4; ++q) {
            for (int p=0; p<4; ++p) {
              Real nhfqp = nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p);
              na1 += (nhfqp*(nh_f_.d_view(n,nb,0)*omega[3][q][p] -
                             nh_f_.d_view(n,nb,3)*omega[0][q][p]));
              na2 += (nhfqp*(nh_f_.d_view(n,nb,2)*omega[1][q][p] -
                             nh_f_.d_view(n,nb,1)*omega[2][q][p]));
            }
          }
          na_(m,n,k,j,i,nb) = iszetaf*na1*uflux.d_view(n,nb,0)+na2*uflux.d_view(n,nb,1);
        }
      }
    });
  }

  // set transformation between normal and tetrad frame
  if (is_hydro_enabled || is_mhd_enabled) {
    auto norm_to_tet_ = norm_to_tet;
    par_for("norm_to_tet",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      Real dgx[4][4], dgy[4][4], dgz[4][4];
      ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
      Real e[4][4], e_cov[4][4], omega[4][4][4];
      ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);

      // Minkowski metric
      Real eta[4][4] = {0.0};
      eta[0][0] = -1.0;
      eta[1][1] = 1.0;
      eta[2][2] = 1.0;
      eta[3][3] = 1.0;

      // Calculate normal-to-coordinate transformation
      Real norm_to_coord[4][4] = {0.0};
      Real alpha = 1.0/sqrt(-gupper[0][0]);
      norm_to_coord[0][0] = 1.0/alpha;
      norm_to_coord[1][0] = -alpha*gupper[0][1];
      norm_to_coord[2][0] = -alpha*gupper[0][2];
      norm_to_coord[3][0] = -alpha*gupper[0][3];
      norm_to_coord[1][1] = 1.0;
      norm_to_coord[2][2] = 1.0;
      norm_to_coord[3][3] = 1.0;

      for (int d1=0; d1<4; ++d1) {
        for (int d2=0; d2<4; ++d2) {
          norm_to_tet_(m,d1,d2,k,j,i) = 0.0;
          for (int p=0; p<4; ++p) {
            for (int q=0; q<4; ++q) {
              norm_to_tet_(m,d1,d2,k,j,i) += eta[d1][p]*e_cov[p][q]*norm_to_coord[q][d2];
            }
          }
        }
      }
    });
  }

  return;
}

} // namespace dyn_radiation
