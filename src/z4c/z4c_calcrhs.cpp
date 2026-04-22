//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \fn TaskStatus Z4c::CalcRHS
//! \brief Computes the wave equation RHS

#include <math.h>

//#include <algorithm>
//#include <cinttypes>
#include <iostream>
//#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

namespace {

struct PointRHS {
  Real chi = 0.0;
  Real vKhat = 0.0;
  Real vTheta = 0.0;
  Real alpha = 0.0;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> vGam_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> beta_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> vB_d;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> vA_dd;

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    chi = 0.0;
    vKhat = 0.0;
    vTheta = 0.0;
    alpha = 0.0;
    vGam_u.ZeroClear();
    beta_u.ZeroClear();
    vB_d.ZeroClear();
    g_dd.ZeroClear();
    vA_dd.ZeroClear();
  }
};

template <int NGHOST, typename State>
KOKKOS_INLINE_FUNCTION
void ComputePointwiseRHS(const State &state, const Z4c::Options &opt,
                         const Tmunu::Tmunu_vars &tmunu, bool include_matter,
                         Real kappa1_eff, Real time,
                         Real dx1, Real dx2, Real dx3,
                         const int m, const int k, const int j, const int i,
                         PointRHS &out) {
  out.ZeroClear();

  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> DA_u;

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> A_uu;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> AA_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Rphi_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddalpha_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddphi_dd;

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;

  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> ddbeta_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dphi_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddalpha_dd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dB_dd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddchi_dd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
  AthenaPointTensor<Real, TensorSymm::ISYM2, 3, 3> ddbeta_ddu;
  AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Lbeta_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LB_d;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Lg_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> LA_dd;

  Real idx[] = {1.0/dx1, 1.0/dx2, 1.0/dx3};

  Real Lalpha = 0.0;
  Real Lchi = 0.0;
  Real LKhat = 0.0;
  Real LTheta = 0.0;

  Real detg = 0.0;
  Real chi_guarded = 0.0;
  Real oopsi4 = 0.0;
  Real AA = 0.0;
  Real R = 0.0;
  Real Ht = 0.0;
  Real K = 0.0;
  Real S = 0.0;
  Real Ddalpha = 0.0;
  Real dbeta = 0.0;
  Real dB = 0.0;

  Lbeta_u.ZeroClear();
  LB_d.ZeroClear();
  LGam_u.ZeroClear();
  Gamma_u.ZeroClear();
  DA_u.ZeroClear();
  ddbeta_d.ZeroClear();
  Lg_dd.ZeroClear();
  LA_dd.ZeroClear();
  AA_dd.ZeroClear();
  R_dd.ZeroClear();
  A_uu.ZeroClear();
  Gamma_udd.ZeroClear();

  for (int a = 0; a < 3; ++a) {
    dalpha_d(a) = Dx<NGHOST>(a, idx, state.alpha, m, k, j, i);
    dchi_d(a) = Dx<NGHOST>(a, idx, state.chi, m, k, j, i);
    dKhat_d(a) = Dx<NGHOST>(a, idx, state.vKhat, m, k, j, i);
    dTheta_d(a) = Dx<NGHOST>(a, idx, state.vTheta, m, k, j, i);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    dbeta_du(b,a) = Dx<NGHOST>(b, idx, state.beta_u, m, a, k, j, i);
    dGam_du(b,a) = Dx<NGHOST>(b, idx, state.vGam_u, m, a, k, j, i);
    dB_dd(b,a) = Dx<NGHOST>(b, idx, state.vB_d, m, a, k, j, i);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, state.g_dd, m, a, b, k, j, i);
  }

  for (int a = 0; a < 3; ++a) {
    ddalpha_dd(a,a) = Dxx<NGHOST>(a, idx, state.alpha, m, k, j, i);
    ddchi_dd(a,a) = Dxx<NGHOST>(a, idx, state.chi, m, k, j, i);

    for (int b = a + 1; b < 3; ++b) {
      ddalpha_dd(a,b) = Dxy<NGHOST>(a, b, idx, state.alpha, m, k, j, i);
      ddchi_dd(a,b) = Dxy<NGHOST>(a, b, idx, state.chi, m, k, j, i);
    }
  }

  for (int c = 0; c < 3; ++c)
  for (int a = 0; a < 3; ++a) {
    ddbeta_ddu(a,a,c) = Dxx<NGHOST>(a, idx, state.beta_u, m, c, k, j, i);
    for (int b = a + 1; b < 3; ++b) {
      ddbeta_ddu(a,b,c) = Dxy<NGHOST>(a, b, idx, state.beta_u, m, c, k, j, i);
    }
  }

  for (int c = 0; c < 3; ++c)
  for (int d = c; d < 3; ++d)
  for (int a = 0; a < 3; ++a) {
    ddg_dddd(a,a,c,d) = Dxx<NGHOST>(a, idx, state.g_dd, m, c, d, k, j, i);
    for (int b = a + 1; b < 3; ++b) {
      ddg_dddd(a,b,c,d) = Dxy<NGHOST>(a, b, idx, state.g_dd, m, c, d, k, j, i);
    }
  }

  for (int a = 0; a < 3; ++a) {
    Lalpha += Lx<NGHOST>(a, idx, state.beta_u, state.alpha, m, a, k, j, i);
    Lchi += Lx<NGHOST>(a, idx, state.beta_u, state.chi, m, a, k, j, i);
    LKhat += Lx<NGHOST>(a, idx, state.beta_u, state.vKhat, m, a, k, j, i);
    LTheta += Lx<NGHOST>(a, idx, state.beta_u, state.vTheta, m, a, k, j, i);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    Lbeta_u(b) += Lx<NGHOST>(a, idx, state.beta_u, state.beta_u, m, a, b, k, j, i);
    LGam_u(b) += Lx<NGHOST>(a, idx, state.beta_u, state.vGam_u, m, a, b, k, j, i);
    if (opt.telegraph_lapse) {
      LB_d(b) += Lx<NGHOST>(a, idx, state.beta_u, state.vB_d, m, a, b, k, j, i);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    Lg_dd(a,b) += Lx<NGHOST>(c, idx, state.beta_u, state.g_dd, m, c, a, b, k, j, i);
    LA_dd(a,b) += Lx<NGHOST>(c, idx, state.beta_u, state.vA_dd, m, c, a, b, k, j, i);
  }

  K = state.vKhat(m,k,j,i) + 2.0 * state.vTheta(m,k,j,i);

  detg = adm::SpatialDet(state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
                         state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
                         state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i));
  adm::SpatialInv(1.0/detg,
                  state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
                  state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
                  state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i),
                  &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
                  &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    dB += g_uu(a,b) * dB_dd(a,b);
  }

  for (int c = 0; c < 3; ++c)
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    Gamma_ddd(c,a,b) = 0.5 * (dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
  }
  for (int c = 0; c < 3; ++c)
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int d = 0; d < 3; ++d) {
    Gamma_udd(c,a,b) += g_uu(c,d) * Gamma_ddd(d,a,b);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    Gamma_u(a) += g_uu(b,c) * Gamma_udd(a,b,c);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    for (int c = 0; c < 3; ++c) {
      R_dd(a,b) += 0.5 * (state.g_dd(m,c,a,k,j,i) * dGam_du(b,c) +
                          state.g_dd(m,c,b,k,j,i) * dGam_du(a,c) +
                          Gamma_u(c) * (Gamma_ddd(a,b,c) + Gamma_ddd(b,a,c)));
    }
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      R_dd(a,b) -= 0.5 * g_uu(c,d) * ddg_dddd(c,d,a,b);
    }
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
    for (int e = 0; e < 3; ++e) {
      R_dd(a,b) += g_uu(c,d) * (
          Gamma_udd(e,c,a) * Gamma_ddd(b,e,d) +
          Gamma_udd(e,c,b) * Gamma_ddd(a,e,d) +
          Gamma_udd(e,a,d) * Gamma_ddd(e,c,b));
    }
  }

  chi_guarded = (state.chi(m,k,j,i) > opt.chi_div_floor)
                    ? state.chi(m,k,j,i) : opt.chi_div_floor;
  oopsi4 = pow(chi_guarded, -4.0/opt.chi_psi_power);
  for (int a = 0; a < 3; ++a) {
    dphi_d(a) = dchi_d(a)/(chi_guarded * opt.chi_psi_power);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    Ddphi_dd(a,b) = ddchi_dd(a,b)/(chi_guarded * opt.chi_psi_power) -
                    opt.chi_psi_power * dphi_d(a) * dphi_d(b);
    for (int c = 0; c < 3; ++c) {
      Ddphi_dd(a,b) -= Gamma_udd(c,a,b) * dphi_d(c);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    Rphi_dd(a,b) = 4.0 * dphi_d(a) * dphi_d(b) - 2.0 * Ddphi_dd(a,b);
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      Rphi_dd(a,b) -= 2.0 * state.g_dd(m,a,b,k,j,i) * g_uu(c,d) *
                      (Ddphi_dd(c,d) + 2.0 * dphi_d(c) * dphi_d(d));
    }
  }

  if (include_matter) {
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      S += oopsi4 * g_uu(a,b) * tmunu.S_dd(m,a,b,k,j,i);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    Ddalpha_dd(a,b) = ddalpha_dd(a,b)
                    - 2.0 * (dphi_d(a) * dalpha_d(b) + dphi_d(b) * dalpha_d(a));
    for (int c = 0; c < 3; ++c) {
      Ddalpha_dd(a,b) -= Gamma_udd(c,a,b) * dalpha_d(c);
      for (int d = 0; d < 3; ++d) {
        Ddalpha_dd(a,b) += 2.0 * state.g_dd(m,a,b,k,j,i) * g_uu(c,d)
                         * dphi_d(c) * dalpha_d(d);
      }
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    Ddalpha += oopsi4 * g_uu(a,b) * Ddalpha_dd(a,b);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c)
  for (int d = 0; d < 3; ++d) {
    AA_dd(a,b) += g_uu(c,d) * state.vA_dd(m,a,c,k,j,i) * state.vA_dd(m,d,b,k,j,i);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    AA += g_uu(a,b) * AA_dd(a,b);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c)
  for (int d = 0; d < 3; ++d) {
    A_uu(a,b) += g_uu(a,c) * g_uu(b,d) * state.vA_dd(m,c,d,k,j,i);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      DA_u(a) -= (3.0/2.0) * A_uu(a,b) * dchi_d(b) / chi_guarded;
      DA_u(a) -= (1.0/3.0) * g_uu(a,b) * (2.0 * dKhat_d(b) + dTheta_d(b));
    }
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      DA_u(a) += Gamma_udd(a,b,c) * A_uu(b,c);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    R += oopsi4 * g_uu(a,b) * (R_dd(a,b) + Rphi_dd(a,b));
  }

  Ht = R + (2.0/3.0) * SQR(K) - AA;

  for (int a = 0; a < 3; ++a) {
    dbeta += dbeta_du(a,a);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    ddbeta_d(a) += (1.0/3.0) * ddbeta_ddu(a,b,b);
  }

  Lchi += (1.0/6.0) * opt.chi_psi_power * chi_guarded * dbeta;

  for (int a = 0; a < 3; ++a) {
    LGam_u(a) += (2.0/3.0) * Gamma_u(a) * dbeta;
    for (int b = 0; b < 3; ++b) {
      LGam_u(a) += g_uu(a,b) * ddbeta_d(b) - Gamma_u(b) * dbeta_du(b,a);
      for (int c = 0; c < 3; ++c) {
        LGam_u(a) += g_uu(b,c) * ddbeta_ddu(b,c,a);
      }
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    Lg_dd(a,b) -= (2.0/3.0) * state.g_dd(m,a,b,k,j,i) * dbeta;
    for (int c = 0; c < 3; ++c) {
      Lg_dd(a,b) += dbeta_du(a,c) * state.g_dd(m,b,c,k,j,i);
      Lg_dd(a,b) += dbeta_du(b,c) * state.g_dd(m,a,c,k,j,i);
    }
  }
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    LA_dd(a,b) -= (2.0/3.0) * state.vA_dd(m,a,b,k,j,i) * dbeta;
    for (int c = 0; c < 3; ++c) {
      LA_dd(a,b) += dbeta_du(b,c) * state.vA_dd(m,a,c,k,j,i);
      LA_dd(a,b) += dbeta_du(a,c) * state.vA_dd(m,b,c,k,j,i);
    }
  }

  out.vKhat = -Ddalpha + state.alpha(m,k,j,i) * (AA + (1.0/3.0) * SQR(K))
            + LKhat + kappa1_eff * (1.0 - opt.damp_kappa2) *
              state.alpha(m,k,j,i) * state.vTheta(m,k,j,i);
  if (include_matter) {
    out.vKhat += 4.0 * M_PI * state.alpha(m,k,j,i) * (S + tmunu.E(m,k,j,i));
  }

  out.chi = Lchi - (1.0/6.0) * opt.chi_psi_power *
            chi_guarded * state.alpha(m,k,j,i) * K;

  out.vTheta = LTheta + state.alpha(m,k,j,i) *
               (0.5 * Ht - (2.0 + opt.damp_kappa2) * kappa1_eff *
                state.vTheta(m,k,j,i));
  if (include_matter) {
    out.vTheta -= 8.0 * M_PI * state.alpha(m,k,j,i) * tmunu.E(m,k,j,i);
  }
  out.vTheta *= opt.use_z4c;

  for (int a = 0; a < 3; ++a) {
    out.vGam_u(a) = 2.0 * state.alpha(m,k,j,i) * DA_u(a) + LGam_u(a);
    out.vGam_u(a) -= 2.0 * state.alpha(m,k,j,i) * kappa1_eff *
                     (state.vGam_u(m,a,k,j,i) - Gamma_u(a));
    for (int b = 0; b < 3; ++b) {
      out.vGam_u(a) -= 2.0 * A_uu(a,b) * dalpha_d(b);
      if (include_matter) {
        out.vGam_u(a) -= 16.0 * M_PI * state.alpha(m,k,j,i) *
                         g_uu(a,b) * tmunu.S_d(m,b,k,j,i);
      }
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    out.g_dd(a,b) = -2.0 * state.alpha(m,k,j,i) * state.vA_dd(m,a,b,k,j,i) +
                    Lg_dd(a,b);
    out.vA_dd(a,b) = oopsi4 *
                     (-Ddalpha_dd(a,b) + state.alpha(m,k,j,i) * (R_dd(a,b) + Rphi_dd(a,b)));
    out.vA_dd(a,b) -= (1.0/3.0) * state.g_dd(m,a,b,k,j,i) *
                      (-Ddalpha + state.alpha(m,k,j,i) * R);
    out.vA_dd(a,b) += state.alpha(m,k,j,i) *
                      (K * state.vA_dd(m,a,b,k,j,i) - 2.0 * AA_dd(a,b));
    out.vA_dd(a,b) += LA_dd(a,b);
    if (include_matter) {
      out.vA_dd(a,b) -= 8.0 * M_PI * state.alpha(m,k,j,i) *
                        (oopsi4 * tmunu.S_dd(m,a,b,k,j,i) -
                         (1.0/3.0) * S * state.g_dd(m,a,b,k,j,i));
    }
  }

  Real const f = opt.lapse_oplog * opt.lapse_harmonicf
               + opt.lapse_harmonic * state.alpha(m,k,j,i);
  out.alpha = opt.lapse_advect * Lalpha
            - f * state.alpha(m,k,j,i) * state.vKhat(m,k,j,i);
  if (opt.slow_start_lapse) {
    Real W2 = (state.chi(m,k,j,i) > opt.chi_min_floor)
                ? state.chi(m,k,j,i) : opt.chi_min_floor;
    Real W = pow(W2, 0.5);
    out.alpha += opt.ssl_damping_amp * (W - state.alpha(m,k,j,i)) *
                 pow(W, opt.ssl_damping_index) *
                 exp(-0.5 * pow(time / opt.ssl_damping_time, 2));
  }
  if (opt.telegraph_lapse) {
    Real W = (state.chi(m,k,j,i) > 0.0) ? state.chi(m,k,j,i) : 0.0;
    out.alpha += W * dB;
    for (int a = 0; a < 3; ++a) {
      out.vB_d(a) = opt.lapse_advect * LB_d(a) +
                    (1.0/opt.telegraph_tau) *
                    (-state.vB_d(m,a,k,j,i) + opt.telegraph_kappa * dalpha_d(a));
    }
  }

  for (int a = 0; a < 3; ++a) {
    out.beta_u(a) =
        (1.0 - opt.sss_damping_amp * exp(-0.5 * pow(time / opt.sss_damping_time, 2))) *
        opt.shift_ggamma * state.vGam_u(m,a,k,j,i)
        + opt.shift_advect * Lbeta_u(a);
    out.beta_u(a) -= opt.shift_eta * state.beta_u(m,a,k,j,i);
  }
  for (int a = 0; a < 3; ++a) {
    out.beta_u(a) += opt.shift_alpha2ggamma *
                     SQR(state.alpha(m,k,j,i)) * state.vGam_u(m,a,k,j,i);
    for (int b = 0; b < 3; ++b) {
      out.beta_u(a) += opt.shift_hh * state.alpha(m,k,j,i) *
                       chi_guarded *
                       (0.5 * state.alpha(m,k,j,i) * dchi_d(b) - dalpha_d(b)) *
                       g_uu(a,b);
    }
  }
}

} // namespace

template <int NGHOST>
//! \fn void Z4c::CalcRHS(Driver *pdriver, int stage)
//! \brief compute rhs of the z4c equations
TaskStatus Z4c::CalcRHS(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto *pz4c = pmy_pack->pz4c;
  auto &z4c = pz4c->z4c;
  auto &full = pz4c->full;
  auto &bg = pz4c->bg;
  auto &rhs = pz4c->rhs;
  auto &opt = pz4c->opt;
  
  Real time = pmy_pack->pmesh->time;
  bool use_analytic_background =
      pz4c->use_analytic_background && pz4c->SetADMBackground != nullptr;
  if (use_analytic_background) {
    pz4c->PrescribeGaugeResidual();
    pz4c->UpdateBackgroundState(time);
    pz4c->ReconstructFullState();
  }
  
  bool is_vacuum = (pmy_pack->ptmunu == nullptr) ? true : false;
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmy_pack->ptmunu->tmunu;

  // Gaussian roll for kappa1 (host-side; capture by value into kernels)

  Real kappa1_effective = opt.damp_kappa1;
  if (opt.roll_kappa && time >= opt.kappa_roll_start_time) {
    // Gaussian stitch: S(t0)=1, S→0 as t→\infty
    Real s = (time - opt.kappa_roll_start_time) / opt.roll_window;
    Real S = exp(-2.30258509299 * s * s);  // smooth, C^\infty falloff
    // prefactor chosen to have S=0.1 at the end of the roll_window
    kappa1_effective = opt.target_kappa1
                      + (opt.damp_kappa1 - opt.target_kappa1) * S;
  }
  const Real kappa1_eff = kappa1_effective;

  // ===================================================================================
  // Main RHS calculation
  //
  par_for("z4c rhs loop",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    PointRHS rhs_full;
    ComputePointwiseRHS<NGHOST>(
        use_analytic_background ? full : z4c, opt, tmunu, !is_vacuum,
        kappa1_eff, time,
        size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
        m, k, j, i, rhs_full);

    if (use_analytic_background) {
      PointRHS rhs_bg;
      ComputePointwiseRHS<NGHOST>(
          bg, opt, tmunu, false, kappa1_eff, time,
          size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
          m, k, j, i, rhs_bg);

      rhs.vKhat(m,k,j,i) = rhs_full.vKhat - rhs_bg.vKhat;
      rhs.chi(m,k,j,i) = rhs_full.chi - rhs_bg.chi;
      rhs.vTheta(m,k,j,i) = rhs_full.vTheta - rhs_bg.vTheta;
      rhs.alpha(m,k,j,i) = 0.0;
      for (int a = 0; a < 3; ++a) {
        rhs.vGam_u(m,a,k,j,i) = rhs_full.vGam_u(a) - rhs_bg.vGam_u(a);
        rhs.beta_u(m,a,k,j,i) = 0.0;
        rhs.vB_d(m,a,k,j,i) = 0.0;
      }
      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        rhs.g_dd(m,a,b,k,j,i) = rhs_full.g_dd(a,b) - rhs_bg.g_dd(a,b);
        rhs.vA_dd(m,a,b,k,j,i) = rhs_full.vA_dd(a,b) - rhs_bg.vA_dd(a,b);
      }
    } else {
      rhs.vKhat(m,k,j,i) = rhs_full.vKhat;
      rhs.chi(m,k,j,i) = rhs_full.chi;
      rhs.vTheta(m,k,j,i) = rhs_full.vTheta;
      rhs.alpha(m,k,j,i) = rhs_full.alpha;
      for (int a = 0; a < 3; ++a) {
        rhs.vGam_u(m,a,k,j,i) = rhs_full.vGam_u(a);
        rhs.beta_u(m,a,k,j,i) = rhs_full.beta_u(a);
        rhs.vB_d(m,a,k,j,i) = rhs_full.vB_d(a);
      }
      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        rhs.g_dd(m,a,b,k,j,i) = rhs_full.g_dd(a,b);
        rhs.vA_dd(m,a,b,k,j,i) = rhs_full.vA_dd(a,b);
      }
    }
  });

  // ===================================================================================
  // Add dissipation for stability
  //
  Real &diss = pmy_pack->pz4c->diss;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u_rhs = pmy_pack->pz4c->u_rhs;
  par_for("K-O Dissipation",
  DevExeSpace(),0,nmb-1,0,nz4c-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    for(int a = 0; a < 3; ++a) {
      u_rhs(m,n,k,j,i) += Diss<NGHOST>(a, idx, u0, m, n, k, j, i)*diss;
    }
  });

  return TaskStatus::complete;
}

template TaskStatus Z4c::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<4>(Driver *pdriver, int stage);
} // namespace z4c
