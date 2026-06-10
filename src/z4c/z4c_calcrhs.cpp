//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \fn TaskStatus Z4c::CalcRHS
//! \brief Computes the wave equation RHS

#include <math.h>

#include <algorithm>
#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

namespace {

struct GeometryData {
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
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> DA_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> ddbeta_d;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Lbeta_u;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LB_d;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> A_uu;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> AA_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Rphi_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddalpha_dd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Lg_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> LA_dd;

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Lalpha = 0.0;
    Lchi = 0.0;
    LKhat = 0.0;
    LTheta = 0.0;
    detg = 0.0;
    chi_guarded = 0.0;
    oopsi4 = 0.0;
    AA = 0.0;
    R = 0.0;
    Ht = 0.0;
    K = 0.0;
    S = 0.0;
    Ddalpha = 0.0;
    dbeta = 0.0;
    dB = 0.0;
    Gamma_u.ZeroClear();
    DA_u.ZeroClear();
    dalpha_d.ZeroClear();
    dchi_d.ZeroClear();
    dKhat_d.ZeroClear();
    dTheta_d.ZeroClear();
    ddbeta_d.ZeroClear();
    LGam_u.ZeroClear();
    Lbeta_u.ZeroClear();
    LB_d.ZeroClear();
    g_uu.ZeroClear();
    A_uu.ZeroClear();
    AA_dd.ZeroClear();
    R_dd.ZeroClear();
    Rphi_dd.ZeroClear();
    Ddalpha_dd.ZeroClear();
    dbeta_du.ZeroClear();
    Lg_dd.ZeroClear();
    LA_dd.ZeroClear();
  }
};

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
void ComputeGeometryData(const State &state, const Z4c::Options &opt,
                         const Tmunu::Tmunu_vars &tmunu, bool include_matter,
                         Real dx1, Real dx2, Real dx3,
                         const int m, const int k, const int j, const int i,
                         GeometryData &geo) {
  geo.ZeroClear();

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddphi_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddalpha_dd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddchi_dd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dB_dd;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
  AthenaPointTensor<Real, TensorSymm::ISYM2, 3, 3> ddbeta_ddu;
  AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

  Real idx[] = {1.0/dx1, 1.0/dx2, 1.0/dx3};

  for (int a = 0; a < 3; ++a) {
    geo.dalpha_d(a) = Dx<NGHOST>(a, idx, state.alpha, m, k, j, i);
    geo.dchi_d(a) = Dx<NGHOST>(a, idx, state.chi, m, k, j, i);
    geo.dKhat_d(a) = Dx<NGHOST>(a, idx, state.vKhat, m, k, j, i);
    geo.dTheta_d(a) = Dx<NGHOST>(a, idx, state.vTheta, m, k, j, i);
  }

  Gamma_udd.ZeroClear();

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.dbeta_du(b,a) = Dx<NGHOST>(b, idx, state.beta_u, m, a, k, j, i);
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
    geo.Lalpha += Lx<NGHOST>(a, idx, state.beta_u, state.alpha, m, a, k, j, i);
    geo.Lchi += Lx<NGHOST>(a, idx, state.beta_u, state.chi, m, a, k, j, i);
    geo.LKhat += Lx<NGHOST>(a, idx, state.beta_u, state.vKhat, m, a, k, j, i);
    geo.LTheta += Lx<NGHOST>(a, idx, state.beta_u, state.vTheta, m, a, k, j, i);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.Lbeta_u(b) += Lx<NGHOST>(a, idx, state.beta_u, state.beta_u, m, a, b, k, j, i);
    geo.LGam_u(b) += Lx<NGHOST>(a, idx, state.beta_u, state.vGam_u, m, a, b, k, j, i);
    if (opt.telegraph_lapse) {
      geo.LB_d(b) += Lx<NGHOST>(a, idx, state.beta_u, state.vB_d, m, a, b, k, j, i);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    geo.Lg_dd(a,b) += Lx<NGHOST>(c, idx, state.beta_u, state.g_dd, m, c, a, b, k, j, i);
    geo.LA_dd(a,b) += Lx<NGHOST>(c, idx, state.beta_u, state.vA_dd, m, c, a, b, k, j, i);
  }

  geo.K = state.vKhat(m,k,j,i) + 2.0 * state.vTheta(m,k,j,i);

  geo.detg = adm::SpatialDet(state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
                             state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
                             state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i));
  adm::SpatialInv(1.0/geo.detg,
                  state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
                  state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
                  state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i),
                  &geo.g_uu(0,0), &geo.g_uu(0,1), &geo.g_uu(0,2),
                  &geo.g_uu(1,1), &geo.g_uu(1,2), &geo.g_uu(2,2));

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.dB += geo.g_uu(a,b) * dB_dd(a,b);
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
    Gamma_udd(c,a,b) += geo.g_uu(c,d) * Gamma_ddd(d,a,b);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    geo.Gamma_u(a) += geo.g_uu(b,c) * Gamma_udd(a,b,c);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    for (int c = 0; c < 3; ++c) {
      geo.R_dd(a,b) += 0.5 * (state.g_dd(m,c,a,k,j,i) * dGam_du(b,c) +
                              state.g_dd(m,c,b,k,j,i) * dGam_du(a,c) +
                              geo.Gamma_u(c) * (Gamma_ddd(a,b,c) + Gamma_ddd(b,a,c)));
    }
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      geo.R_dd(a,b) -= 0.5 * geo.g_uu(c,d) * ddg_dddd(c,d,a,b);
    }
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
    for (int e = 0; e < 3; ++e) {
      geo.R_dd(a,b) += geo.g_uu(c,d) * (
          Gamma_udd(e,c,a) * Gamma_ddd(b,e,d) +
          Gamma_udd(e,c,b) * Gamma_ddd(a,e,d) +
          Gamma_udd(e,a,d) * Gamma_ddd(e,c,b));
    }
  }

  geo.chi_guarded = (state.chi(m,k,j,i) > opt.chi_div_floor)
                        ? state.chi(m,k,j,i) : opt.chi_div_floor;
  geo.oopsi4 = pow(geo.chi_guarded, -4.0/opt.chi_psi_power);
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dphi_d;
  for (int a = 0; a < 3; ++a) {
    dphi_d(a) = geo.dchi_d(a) / (geo.chi_guarded * opt.chi_psi_power);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    Ddphi_dd(a,b) = ddchi_dd(a,b) / (geo.chi_guarded * opt.chi_psi_power) -
                    opt.chi_psi_power * dphi_d(a) * dphi_d(b);
    for (int c = 0; c < 3; ++c) {
      Ddphi_dd(a,b) -= Gamma_udd(c,a,b) * dphi_d(c);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    geo.Rphi_dd(a,b) = 4.0 * dphi_d(a) * dphi_d(b) - 2.0 * Ddphi_dd(a,b);
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      geo.Rphi_dd(a,b) -= 2.0 * state.g_dd(m,a,b,k,j,i) * geo.g_uu(c,d) *
                          (Ddphi_dd(c,d) + 2.0 * dphi_d(c) * dphi_d(d));
    }
  }

  if (include_matter) {
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      geo.S += geo.oopsi4 * geo.g_uu(a,b) * tmunu.S_dd(m,a,b,k,j,i);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.Ddalpha_dd(a,b) = ddalpha_dd(a,b) -
                          2.0 * (dphi_d(a) * geo.dalpha_d(b) + dphi_d(b) * geo.dalpha_d(a));
    for (int c = 0; c < 3; ++c) {
      geo.Ddalpha_dd(a,b) -= Gamma_udd(c,a,b) * geo.dalpha_d(c);
      for (int d = 0; d < 3; ++d) {
        geo.Ddalpha_dd(a,b) += 2.0 * state.g_dd(m,a,b,k,j,i) * geo.g_uu(c,d) *
                               dphi_d(c) * geo.dalpha_d(d);
      }
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.Ddalpha += geo.oopsi4 * geo.g_uu(a,b) * geo.Ddalpha_dd(a,b);
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c)
  for (int d = 0; d < 3; ++d) {
    geo.AA_dd(a,b) += geo.g_uu(c,d) * state.vA_dd(m,a,c,k,j,i) * state.vA_dd(m,d,b,k,j,i);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.AA += geo.g_uu(a,b) * geo.AA_dd(a,b);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b)
  for (int c = 0; c < 3; ++c)
  for (int d = 0; d < 3; ++d) {
    geo.A_uu(a,b) += geo.g_uu(a,c) * geo.g_uu(b,d) * state.vA_dd(m,c,d,k,j,i);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      geo.DA_u(a) -= (3.0/2.0) * geo.A_uu(a,b) * geo.dchi_d(b) / geo.chi_guarded;
      geo.DA_u(a) -= (1.0/3.0) * geo.g_uu(a,b) *
                     (2.0 * geo.dKhat_d(b) + geo.dTheta_d(b));
    }
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      geo.DA_u(a) += Gamma_udd(a,b,c) * geo.A_uu(b,c);
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.R += geo.oopsi4 * geo.g_uu(a,b) * (geo.R_dd(a,b) + geo.Rphi_dd(a,b));
  }

  geo.Ht = geo.R + (2.0/3.0) * SQR(geo.K) - geo.AA;

  for (int a = 0; a < 3; ++a) {
    geo.dbeta += geo.dbeta_du(a,a);
  }
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    geo.ddbeta_d(a) += (1.0/3.0) * ddbeta_ddu(a,b,b);
  }

  geo.Lchi += (1.0/6.0) * opt.chi_psi_power * geo.chi_guarded * geo.dbeta;

  for (int a = 0; a < 3; ++a) {
    geo.LGam_u(a) += (2.0/3.0) * geo.Gamma_u(a) * geo.dbeta;
    for (int b = 0; b < 3; ++b) {
      geo.LGam_u(a) += geo.g_uu(a,b) * geo.ddbeta_d(b) - geo.Gamma_u(b) * geo.dbeta_du(b,a);
      for (int c = 0; c < 3; ++c) {
        geo.LGam_u(a) += geo.g_uu(b,c) * ddbeta_ddu(b,c,a);
      }
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    geo.Lg_dd(a,b) -= (2.0/3.0) * state.g_dd(m,a,b,k,j,i) * geo.dbeta;
    for (int c = 0; c < 3; ++c) {
      geo.Lg_dd(a,b) += geo.dbeta_du(a,c) * state.g_dd(m,b,c,k,j,i);
      geo.Lg_dd(a,b) += geo.dbeta_du(b,c) * state.g_dd(m,a,c,k,j,i);
    }
  }
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    geo.LA_dd(a,b) -= (2.0/3.0) * state.vA_dd(m,a,b,k,j,i) * geo.dbeta;
    for (int c = 0; c < 3; ++c) {
      geo.LA_dd(a,b) += geo.dbeta_du(b,c) * state.vA_dd(m,a,c,k,j,i);
      geo.LA_dd(a,b) += geo.dbeta_du(a,c) * state.vA_dd(m,b,c,k,j,i);
    }
  }
}

template <typename State>
KOKKOS_INLINE_FUNCTION
void BuildStandardPointwiseRHS(const State &state, const Z4c::Options &opt,
                               const Tmunu::Tmunu_vars &tmunu, bool include_matter,
                               Real kappa1_eff, Real time,
                               const int m, const int k, const int j, const int i,
                               const GeometryData &geo, PointRHS &out) {
  out.ZeroClear();

  out.vKhat = -geo.Ddalpha + state.alpha(m,k,j,i) *
              (geo.AA + (1.0/3.0) * SQR(geo.K)) +
              geo.LKhat + kappa1_eff * (1.0 - opt.damp_kappa2) *
              state.alpha(m,k,j,i) * state.vTheta(m,k,j,i);
  if (include_matter) {
    out.vKhat += 4.0 * M_PI * state.alpha(m,k,j,i) *
                 (geo.S + tmunu.E(m,k,j,i));
  }

  out.chi = geo.Lchi - (1.0/6.0) * opt.chi_psi_power *
            geo.chi_guarded * state.alpha(m,k,j,i) * geo.K;

  out.vTheta = geo.LTheta + state.alpha(m,k,j,i) *
               (0.5 * geo.Ht - (2.0 + opt.damp_kappa2) * kappa1_eff *
                state.vTheta(m,k,j,i));
  if (include_matter) {
    out.vTheta -= 8.0 * M_PI * state.alpha(m,k,j,i) * tmunu.E(m,k,j,i);
  }
  out.vTheta *= opt.use_z4c;

  for (int a = 0; a < 3; ++a) {
    out.vGam_u(a) = 2.0 * state.alpha(m,k,j,i) * geo.DA_u(a) + geo.LGam_u(a);
    out.vGam_u(a) -= 2.0 * state.alpha(m,k,j,i) * kappa1_eff *
                     (state.vGam_u(m,a,k,j,i) - geo.Gamma_u(a));
    for (int b = 0; b < 3; ++b) {
      out.vGam_u(a) -= 2.0 * geo.A_uu(a,b) * geo.dalpha_d(b);
      if (include_matter) {
        out.vGam_u(a) -= 16.0 * M_PI * state.alpha(m,k,j,i) *
                         geo.g_uu(a,b) * tmunu.S_d(m,b,k,j,i);
      }
    }
  }

  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    out.g_dd(a,b) = -2.0 * state.alpha(m,k,j,i) * state.vA_dd(m,a,b,k,j,i) +
                    geo.Lg_dd(a,b);
    out.vA_dd(a,b) = geo.oopsi4 *
                     (-geo.Ddalpha_dd(a,b) +
                      state.alpha(m,k,j,i) * (geo.R_dd(a,b) + geo.Rphi_dd(a,b)));
    out.vA_dd(a,b) -= (1.0/3.0) * state.g_dd(m,a,b,k,j,i) *
                      (-geo.Ddalpha + state.alpha(m,k,j,i) * geo.R);
    out.vA_dd(a,b) += state.alpha(m,k,j,i) *
                      (geo.K * state.vA_dd(m,a,b,k,j,i) - 2.0 * geo.AA_dd(a,b));
    out.vA_dd(a,b) += geo.LA_dd(a,b);
    if (include_matter) {
      out.vA_dd(a,b) -= 8.0 * M_PI * state.alpha(m,k,j,i) *
                        (geo.oopsi4 * tmunu.S_dd(m,a,b,k,j,i) -
                         (1.0/3.0) * geo.S * state.g_dd(m,a,b,k,j,i));
    }
  }

  Real const f = opt.lapse_oplog * opt.lapse_harmonicf
               + opt.lapse_harmonic * state.alpha(m,k,j,i);
  out.alpha = opt.lapse_advect * geo.Lalpha
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
    out.alpha += W * geo.dB;
    for (int a = 0; a < 3; ++a) {
      out.vB_d(a) = opt.lapse_advect * geo.LB_d(a) +
                    (1.0/opt.telegraph_tau) *
                    (-state.vB_d(m,a,k,j,i) + opt.telegraph_kappa * geo.dalpha_d(a));
    }
  }

  for (int a = 0; a < 3; ++a) {
    out.beta_u(a) =
        (1.0 - opt.sss_damping_amp * exp(-0.5 * pow(time / opt.sss_damping_time, 2))) *
        opt.shift_ggamma * state.vGam_u(m,a,k,j,i)
        + opt.shift_advect * geo.Lbeta_u(a);
    out.beta_u(a) -= opt.shift_eta * state.beta_u(m,a,k,j,i);
  }
  for (int a = 0; a < 3; ++a) {
    out.beta_u(a) += opt.shift_alpha2ggamma *
                     SQR(state.alpha(m,k,j,i)) * state.vGam_u(m,a,k,j,i);
    for (int b = 0; b < 3; ++b) {
      out.beta_u(a) += opt.shift_hh * state.alpha(m,k,j,i) *
                       geo.chi_guarded *
                       (0.5 * state.alpha(m,k,j,i) * geo.dchi_d(b) - geo.dalpha_d(b)) *
                       geo.g_uu(a,b);
    }
  }
}

template <int NGHOST, typename ResState, typename BgState>
KOKKOS_INLINE_FUNCTION
void BuildBackgroundAdaptedResidualGaugeRHS(
    const ResState &res, const BgState &bg, const Z4c::Options &opt,
    Real time, Real dx1, Real dx2, Real dx3,
    const int m, const int k, const int j, const int i, PointRHS &out) {
  out.ZeroClear();

  Real idx[] = {1.0/dx1, 1.0/dx2, 1.0/dx3};
  Real Lalpha_res_bg = 0.0;
  for (int a = 0; a < 3; ++a) {
    Lalpha_res_bg += Lx<NGHOST>(a, idx, bg.beta_u, res.alpha, m, a, k, j, i);
  }

  const Real alpha_bg = bg.alpha(m,k,j,i);
  const Real f_bg = opt.lapse_oplog * opt.lapse_harmonicf +
                    opt.lapse_harmonic * alpha_bg;
  out.alpha = opt.lapse_advect * Lalpha_res_bg -
              opt.residual_lapse_f * f_bg * alpha_bg * res.vKhat(m,k,j,i) -
              opt.residual_lapse_damping * res.alpha(m,k,j,i);

  const Real shift_driver =
      (1.0 - opt.sss_damping_amp * exp(-0.5 * pow(time / opt.sss_damping_time, 2))) *
      opt.shift_ggamma;
  for (int a = 0; a < 3; ++a) {
    Real Lbeta_res_bg = 0.0;
    for (int b = 0; b < 3; ++b) {
      Lbeta_res_bg += Lx<NGHOST>(b, idx, bg.beta_u, res.beta_u, m, b, a, k, j, i);
    }
    out.beta_u(a) =
        (shift_driver + opt.shift_alpha2ggamma * SQR(alpha_bg)) *
            res.vGam_u(m,a,k,j,i) +
        opt.shift_advect * Lbeta_res_bg -
        (opt.shift_eta + opt.residual_shift_damping) * res.beta_u(m,a,k,j,i);
    out.vB_d(a) = 0.0;
  }
}

// ====================================================================================
// Input-gated term-by-term residual RHS diagnostics (<z4c>/rhs_term_debug).
// Decomposes the residual (full-minus-background) Z4c RHS into named term
// categories, reports domain-wide max |contribution| per category, and prints a
// full local breakdown at the argmax of |rhs Khat_res|, |Khat_res|, |Gam_res|,
// and |Theta_res|.

// Term slots for geometry-based residual RHS contributions.
enum ResTermIndex {
  T_KH_DDA,   // Khat: -(DDalpha_full - DDalpha_bg)
  T_KH_ALG,   // Khat: alpha(AA + K^2/3) full-bg
  T_KH_ADV,   // Khat: advection L_beta Khat full-bg
  T_KH_DAMP,  // Khat: kappa1 Theta damping full-bg
  T_KH_MAT,   // Khat: matter 4 pi alpha (S+E)
  T_TH_ADV,   // Theta: advection full-bg
  T_TH_HT,    // Theta: 0.5 alpha Ht full-bg
  T_TH_DAMP,  // Theta: kappa1 damping full-bg
  T_TH_MAT,   // Theta: matter -8 pi alpha E
  T_CH_ADV,   // chi: advection + div(beta) term full-bg
  T_CH_SRC,   // chi: -(1/6) chi_psi_power chi alpha K full-bg
  T_GM_DA,    // Gam: 2 alpha DA_u full-bg (max comp)
  T_GM_ADAL,  // Gam: -2 A^{ab} d_b alpha full-bg (max comp)
  T_GM_ADV,   // Gam: advection/shift-derivative terms full-bg (max comp)
  T_GM_DAMP,  // Gam: -2 kappa1 alpha (Gam - Gamma) full-bg (max comp)
  T_GM_MAT,   // Gam: matter -16 pi alpha g^{ab} S_b (max comp)
  T_G_A,      // g: -2 alpha A_ab full-bg (max comp)
  T_G_ADV,    // g: Lie/shift terms full-bg (max comp)
  T_A_RIC,    // A: oopsi4(-DDalpha_ab + alpha(R_ab+Rphi_ab)) full-bg (max comp)
  T_A_TR,     // A: -(1/3) g_ab (-DDalpha + alpha R) full-bg (max comp)
  T_A_ALG,    // A: alpha(K A_ab - 2 AA_ab) full-bg (max comp)
  T_A_ADV,    // A: Lie/shift terms full-bg (max comp)
  T_A_MAT,    // A: matter term (max comp)
  NTERM_GEO
};

static char const * const ResTermNames[NTERM_GEO] = {
  "Khat_dda", "Khat_alg", "Khat_adv", "Khat_damp", "Khat_mat",
  "Theta_adv", "Theta_Ht", "Theta_damp", "Theta_mat",
  "chi_adv", "chi_src",
  "Gam_DA", "Gam_Adal", "Gam_adv", "Gam_damp", "Gam_mat",
  "g_A", "g_adv",
  "A_ric", "A_tr", "A_alg", "A_adv", "A_mat",
};

template <typename FullState, typename BgState>
KOKKOS_INLINE_FUNCTION
void ComputeResidualTerms(const FullState &full, const BgState &bg,
                          const GeometryData &gf, const GeometryData &gb,
                          const Z4c::Options &opt,
                          const Tmunu::Tmunu_vars &tmunu, bool include_matter,
                          Real kappa1_eff,
                          const int m, const int k, const int j, const int i,
                          Real terms[NTERM_GEO]) {
  const Real af = full.alpha(m,k,j,i);
  const Real ab = bg.alpha(m,k,j,i);

  terms[T_KH_DDA] = -(gf.Ddalpha - gb.Ddalpha);
  terms[T_KH_ALG] = af * (gf.AA + (1.0/3.0) * SQR(gf.K)) -
                    ab * (gb.AA + (1.0/3.0) * SQR(gb.K));
  terms[T_KH_ADV] = gf.LKhat - gb.LKhat;
  terms[T_KH_DAMP] = kappa1_eff * (1.0 - opt.damp_kappa2) *
                     (af * full.vTheta(m,k,j,i) - ab * bg.vTheta(m,k,j,i));
  terms[T_KH_MAT] = include_matter
      ? 4.0 * M_PI * af * (gf.S + tmunu.E(m,k,j,i)) : 0.0;

  terms[T_TH_ADV] = gf.LTheta - gb.LTheta;
  terms[T_TH_HT] = 0.5 * (af * gf.Ht - ab * gb.Ht);
  terms[T_TH_DAMP] = -(2.0 + opt.damp_kappa2) * kappa1_eff *
                     (af * full.vTheta(m,k,j,i) - ab * bg.vTheta(m,k,j,i));
  terms[T_TH_MAT] = include_matter
      ? -8.0 * M_PI * af * tmunu.E(m,k,j,i) : 0.0;

  terms[T_CH_ADV] = gf.Lchi - gb.Lchi;
  terms[T_CH_SRC] = -(1.0/6.0) * opt.chi_psi_power *
                    (gf.chi_guarded * af * gf.K - gb.chi_guarded * ab * gb.K);

  terms[T_GM_DA] = 0.0;
  terms[T_GM_ADAL] = 0.0;
  terms[T_GM_ADV] = 0.0;
  terms[T_GM_DAMP] = 0.0;
  terms[T_GM_MAT] = 0.0;
  for (int a = 0; a < 3; ++a) {
    terms[T_GM_DA] = fmax(terms[T_GM_DA],
        fabs(2.0 * (af * gf.DA_u(a) - ab * gb.DA_u(a))));
    terms[T_GM_ADV] = fmax(terms[T_GM_ADV], fabs(gf.LGam_u(a) - gb.LGam_u(a)));
    terms[T_GM_DAMP] = fmax(terms[T_GM_DAMP],
        fabs(2.0 * kappa1_eff *
             (af * (full.vGam_u(m,a,k,j,i) - gf.Gamma_u(a)) -
              ab * (bg.vGam_u(m,a,k,j,i) - gb.Gamma_u(a)))));
    Real adal = 0.0;
    Real gmat = 0.0;
    for (int b = 0; b < 3; ++b) {
      adal -= 2.0 * (gf.A_uu(a,b) * gf.dalpha_d(b) - gb.A_uu(a,b) * gb.dalpha_d(b));
      if (include_matter) {
        gmat -= 16.0 * M_PI * af * gf.g_uu(a,b) * tmunu.S_d(m,b,k,j,i);
      }
    }
    terms[T_GM_ADAL] = fmax(terms[T_GM_ADAL], fabs(adal));
    terms[T_GM_MAT] = fmax(terms[T_GM_MAT], fabs(gmat));
  }

  terms[T_G_A] = 0.0;
  terms[T_G_ADV] = 0.0;
  terms[T_A_RIC] = 0.0;
  terms[T_A_TR] = 0.0;
  terms[T_A_ALG] = 0.0;
  terms[T_A_ADV] = 0.0;
  terms[T_A_MAT] = 0.0;
  for (int a = 0; a < 3; ++a)
  for (int b = a; b < 3; ++b) {
    terms[T_G_A] = fmax(terms[T_G_A],
        fabs(-2.0 * (af * full.vA_dd(m,a,b,k,j,i) - ab * bg.vA_dd(m,a,b,k,j,i))));
    terms[T_G_ADV] = fmax(terms[T_G_ADV], fabs(gf.Lg_dd(a,b) - gb.Lg_dd(a,b)));
    terms[T_A_RIC] = fmax(terms[T_A_RIC],
        fabs(gf.oopsi4 * (-gf.Ddalpha_dd(a,b) +
                          af * (gf.R_dd(a,b) + gf.Rphi_dd(a,b))) -
             gb.oopsi4 * (-gb.Ddalpha_dd(a,b) +
                          ab * (gb.R_dd(a,b) + gb.Rphi_dd(a,b)))));
    terms[T_A_TR] = fmax(terms[T_A_TR],
        fabs(-(1.0/3.0) *
             (full.g_dd(m,a,b,k,j,i) * (-gf.Ddalpha + af * gf.R) -
              bg.g_dd(m,a,b,k,j,i) * (-gb.Ddalpha + ab * gb.R))));
    terms[T_A_ALG] = fmax(terms[T_A_ALG],
        fabs(af * (gf.K * full.vA_dd(m,a,b,k,j,i) - 2.0 * gf.AA_dd(a,b)) -
             ab * (gb.K * bg.vA_dd(m,a,b,k,j,i) - 2.0 * gb.AA_dd(a,b))));
    terms[T_A_ADV] = fmax(terms[T_A_ADV], fabs(gf.LA_dd(a,b) - gb.LA_dd(a,b)));
    if (include_matter) {
      terms[T_A_MAT] = fmax(terms[T_A_MAT],
          fabs(-8.0 * M_PI * af *
               (gf.oopsi4 * tmunu.S_dd(m,a,b,k,j,i) -
                (1.0/3.0) * gf.S * full.g_dd(m,a,b,k,j,i))));
    }
  }
}

} // namespace

template <int NGHOST>
static void ResidualRHSTermDebug(MeshBlockPack *pmy_pack, Z4c *pz4c,
                                 Real kappa1_eff, Real time, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int nmkji = nmb * nx3 * nx2 * nx1;
  int nkji = nx3 * nx2 * nx1;
  int nji = nx2 * nx1;

  auto &full = pz4c->full;
  auto &bg = pz4c->bg;
  auto &res = pz4c->z4c;
  auto &rhs = pz4c->rhs;
  auto &opt = pz4c->opt;
  auto &u0 = pz4c->u0;
  auto &u_rhs = pz4c->u_rhs;
  const Real diss = pz4c->diss;
  const int ncycle = pmy_pack->pmesh->ncycle;

  bool is_vacuum = (pmy_pack->ptmunu == nullptr);
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmy_pack->ptmunu->tmunu;
  const bool include_matter = !is_vacuum;

  // -----------------------------------------------------------------------------
  // (1) Domain-wide max |contribution| per geometry-based term category.
  DvceArray1D<Real> term_max_d("z4c_res_term_max", NTERM_GEO);
  Kokkos::deep_copy(term_max_d, 0.0);
  Kokkos::parallel_for(
      "Z4cResTermMax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = idx - m * nkji - k * nji - j * nx1 + is;
        k += ks;
        j += js;
        GeometryData gf, gb;
        ComputeGeometryData<NGHOST>(full, opt, tmunu, include_matter,
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
            m, k, j, i, gf);
        ComputeGeometryData<NGHOST>(bg, opt, tmunu, false,
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
            m, k, j, i, gb);
        Real terms[NTERM_GEO];
        ComputeResidualTerms(full, bg, gf, gb, opt, tmunu, include_matter,
                             kappa1_eff, m, k, j, i, terms);
        for (int n = 0; n < NTERM_GEO; ++n) {
          Kokkos::atomic_max(&term_max_d(n), fabs(terms[n]));
        }
      });
  auto term_max_h = Kokkos::create_mirror_view(term_max_d);
  Kokkos::deep_copy(term_max_h, term_max_d);
  Real term_max[NTERM_GEO];
  for (int n = 0; n < NTERM_GEO; ++n) { term_max[n] = term_max_h(n); }

  // (2) KO dissipation max |contribution| per field group and nonfinite counts.
  Real ko_alpha = 0.0, ko_khat = 0.0, ko_theta = 0.0, ko_chi = 0.0;
  Real ko_gam = 0.0, ko_g = 0.0, ko_a = 0.0;
  Real nonfinite_rhs = 0.0, nonfinite_u0 = 0.0;
  Kokkos::parallel_reduce(
      "Z4cResKOMax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &m_alpha, Real &m_khat, Real &m_theta,
                    Real &m_chi, Real &m_gam, Real &m_g, Real &m_a,
                    Real &bad_rhs, Real &bad_u0) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = idx - m * nkji - k * nji - j * nx1 + is;
        k += ks;
        j += js;
        Real idx1[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                       1.0/size.d_view(m).dx3};
        for (int n = 0; n < Z4c::nz4c; ++n) {
          Real ko = 0.0;
          for (int a = 0; a < 3; ++a) {
            ko += Diss<NGHOST>(a, idx1, u0, m, n, k, j, i) * diss;
          }
          ko = fabs(ko);
          if (n == Z4c::I_Z4C_ALPHA) {
            m_alpha = fmax(m_alpha, ko);
          } else if (n == Z4c::I_Z4C_KHAT) {
            m_khat = fmax(m_khat, ko);
          } else if (n == Z4c::I_Z4C_THETA) {
            m_theta = fmax(m_theta, ko);
          } else if (n == Z4c::I_Z4C_CHI) {
            m_chi = fmax(m_chi, ko);
          } else if (n >= Z4c::I_Z4C_GAMX && n <= Z4c::I_Z4C_GAMZ) {
            m_gam = fmax(m_gam, ko);
          } else if (n >= Z4c::I_Z4C_GXX && n <= Z4c::I_Z4C_GZZ) {
            m_g = fmax(m_g, ko);
          } else if (n >= Z4c::I_Z4C_AXX && n <= Z4c::I_Z4C_AZZ) {
            m_a = fmax(m_a, ko);
          }
          if (!isfinite(u_rhs(m,n,k,j,i))) { bad_rhs += 1.0; }
          if (!isfinite(u0(m,n,k,j,i))) { bad_u0 += 1.0; }
        }
      },
      Kokkos::Max<Real>(ko_alpha), Kokkos::Max<Real>(ko_khat),
      Kokkos::Max<Real>(ko_theta), Kokkos::Max<Real>(ko_chi),
      Kokkos::Max<Real>(ko_gam), Kokkos::Max<Real>(ko_g),
      Kokkos::Max<Real>(ko_a), Kokkos::Sum<Real>(nonfinite_rhs),
      Kokkos::Sum<Real>(nonfinite_u0));

#if MPI_PARALLEL_ENABLED
  {
    Real buf[NTERM_GEO + 7];
    for (int n = 0; n < NTERM_GEO; ++n) { buf[n] = term_max[n]; }
    buf[NTERM_GEO + 0] = ko_alpha;
    buf[NTERM_GEO + 1] = ko_khat;
    buf[NTERM_GEO + 2] = ko_theta;
    buf[NTERM_GEO + 3] = ko_chi;
    buf[NTERM_GEO + 4] = ko_gam;
    buf[NTERM_GEO + 5] = ko_g;
    buf[NTERM_GEO + 6] = ko_a;
    MPI_Allreduce(MPI_IN_PLACE, buf, NTERM_GEO + 7, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
    Real cnt[2] = {nonfinite_rhs, nonfinite_u0};
    MPI_Allreduce(MPI_IN_PLACE, cnt, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    for (int n = 0; n < NTERM_GEO; ++n) { term_max[n] = buf[n]; }
    ko_alpha = buf[NTERM_GEO + 0];
    ko_khat = buf[NTERM_GEO + 1];
    ko_theta = buf[NTERM_GEO + 2];
    ko_chi = buf[NTERM_GEO + 3];
    ko_gam = buf[NTERM_GEO + 4];
    ko_g = buf[NTERM_GEO + 5];
    ko_a = buf[NTERM_GEO + 6];
    nonfinite_rhs = cnt[0];
    nonfinite_u0 = cnt[1];
  }
#endif

  if (global_variable::my_rank == 0) {
    std::cout << "Z4C_RHS_TERM_MAX cycle=" << ncycle << " time=" << time
              << " stage=" << stage;
    for (int n = 0; n < NTERM_GEO; ++n) {
      std::cout << " " << ResTermNames[n] << "=" << term_max[n];
    }
    std::cout << " KO_alpha=" << ko_alpha << " KO_Khat=" << ko_khat
              << " KO_Theta=" << ko_theta << " KO_chi=" << ko_chi
              << " KO_Gam=" << ko_gam << " KO_g=" << ko_g << " KO_A=" << ko_a
              << " nonfinite_rhs=" << nonfinite_rhs
              << " nonfinite_u0=" << nonfinite_u0 << std::endl;
  }

  // -----------------------------------------------------------------------------
  // (3) First-bad-location diagnostics: argmax of key residual quantities with a
  // full local term breakdown at each location.
  using maxloc_t = Kokkos::MaxLoc<Real, int>;
  constexpr int nfields = 4;
  const char *field_names[nfields] = {"rhs_Khat_res", "Khat_res", "Gam_res",
                                      "Theta_res"};
  maxloc_t::value_type field_loc[nfields];

  Kokkos::parallel_reduce(
      "Z4cResLocRhsKhat", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, maxloc_t::value_type &lmax) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji + ks;
        int j = ((idx / nx1) % nx2) + js;
        int i = (idx % nx1) + is;
        Real v = fabs(rhs.vKhat(m,k,j,i));
        if (v > lmax.val) { lmax.val = v; lmax.loc = idx; }
      }, maxloc_t(field_loc[0]));
  Kokkos::parallel_reduce(
      "Z4cResLocKhat", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, maxloc_t::value_type &lmax) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji + ks;
        int j = ((idx / nx1) % nx2) + js;
        int i = (idx % nx1) + is;
        Real v = fabs(res.vKhat(m,k,j,i));
        if (v > lmax.val) { lmax.val = v; lmax.loc = idx; }
      }, maxloc_t(field_loc[1]));
  Kokkos::parallel_reduce(
      "Z4cResLocGam", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, maxloc_t::value_type &lmax) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji + ks;
        int j = ((idx / nx1) % nx2) + js;
        int i = (idx % nx1) + is;
        Real v = 0.0;
        for (int a = 0; a < 3; ++a) { v = fmax(v, fabs(res.vGam_u(m,a,k,j,i))); }
        if (v > lmax.val) { lmax.val = v; lmax.loc = idx; }
      }, maxloc_t(field_loc[2]));
  Kokkos::parallel_reduce(
      "Z4cResLocTheta", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, maxloc_t::value_type &lmax) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji + ks;
        int j = ((idx / nx1) % nx2) + js;
        int i = (idx % nx1) + is;
        Real v = fabs(res.vTheta(m,k,j,i));
        if (v > lmax.val) { lmax.val = v; lmax.loc = idx; }
      }, maxloc_t(field_loc[3]));

  constexpr int NBUF = NTERM_GEO + 24;
  DvceArray1D<Real> dbuf("z4c_res_term_dbuf", NBUF);
  auto hbuf = Kokkos::create_mirror_view(dbuf);

  for (int f = 0; f < nfields; ++f) {
    Real my_val = (field_loc[f].loc >= 0) ? field_loc[f].val : -1.0;
    int print_rank = 0;
#if MPI_PARALLEL_ENABLED
    struct { double val; int rank; } in, out;
    in.val = my_val;
    in.rank = global_variable::my_rank;
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    print_rank = out.rank;
#endif
    if (global_variable::my_rank != print_rank || field_loc[f].loc < 0) {
      continue;
    }
    const int idx = field_loc[f].loc;
    const int m = idx / nkji;
    const int k = (idx - m * nkji) / nji + ks;
    const int j = ((idx / nx1) % nx2) + js;
    const int i = (idx % nx1) + is;

    Kokkos::parallel_for(
        "Z4cResTermPoint", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
        KOKKOS_LAMBDA(const int) {
          GeometryData gf, gb;
          ComputeGeometryData<NGHOST>(full, opt, tmunu, include_matter,
              size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
              m, k, j, i, gf);
          ComputeGeometryData<NGHOST>(bg, opt, tmunu, false,
              size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
              m, k, j, i, gb);
          Real terms[NTERM_GEO];
          ComputeResidualTerms(full, bg, gf, gb, opt, tmunu, include_matter,
                               kappa1_eff, m, k, j, i, terms);
          for (int n = 0; n < NTERM_GEO; ++n) { dbuf(n) = terms[n]; }
          int n = NTERM_GEO;
          dbuf(n++) = rhs.vKhat(m,k,j,i);
          dbuf(n++) = rhs.vTheta(m,k,j,i);
          dbuf(n++) = rhs.alpha(m,k,j,i);
          dbuf(n++) = rhs.chi(m,k,j,i);
          dbuf(n++) = full.alpha(m,k,j,i);
          dbuf(n++) = bg.alpha(m,k,j,i);
          dbuf(n++) = res.alpha(m,k,j,i);
          dbuf(n++) = full.vKhat(m,k,j,i);
          dbuf(n++) = bg.vKhat(m,k,j,i);
          dbuf(n++) = res.vKhat(m,k,j,i);
          dbuf(n++) = full.vTheta(m,k,j,i);
          dbuf(n++) = res.vTheta(m,k,j,i);
          dbuf(n++) = full.chi(m,k,j,i);
          dbuf(n++) = res.chi(m,k,j,i);
          dbuf(n++) = gf.detg;
          dbuf(n++) = gb.detg;
          Real gam_res = 0.0;
          for (int a = 0; a < 3; ++a) {
            gam_res = fmax(gam_res, fabs(res.vGam_u(m,a,k,j,i)));
          }
          dbuf(n++) = gam_res;
          dbuf(n++) = gf.AA;
          dbuf(n++) = gb.AA;
          dbuf(n++) = gf.K;
          dbuf(n++) = gb.K;
          dbuf(n++) = gf.R;
          dbuf(n++) = gb.R;
          dbuf(n++) = include_matter ? tmunu.E(m,k,j,i) : 0.0;
        });
    Kokkos::deep_copy(hbuf, dbuf);

    auto &msize = pmy_pack->pmb->mb_size.h_view(m);
    Real x = CellCenterX(i - is, nx1, msize.x1min, msize.x1max);
    Real y = CellCenterX(j - js, nx2, msize.x2min, msize.x2max);
    Real z = CellCenterX(k - ks, nx3, msize.x3min, msize.x3max);
    Real r_bh = sqrt(SQR(x - opt.history_excise_ks_x1) +
                     SQR(y - opt.history_excise_ks_x2) +
                     SQR(z - opt.history_excise_ks_x3));
    int gid = pmy_pack->pmb->mb_gid.h_view(m);
    int lev = pmy_pack->pmb->mb_lev.h_view(m);

    std::ostringstream oss;
    oss << "Z4C_RHS_TERM_LOC cycle=" << ncycle << " time=" << time
        << " stage=" << stage << " field=" << field_names[f]
        << " val=" << field_loc[f].val
        << " rank=" << global_variable::my_rank
        << " gid=" << gid << " level=" << lev
        << " x=" << x << " y=" << y << " z=" << z << " r_bh=" << r_bh
        << " mb_x1=[" << msize.x1min << "," << msize.x1max << "]"
        << " mb_x2=[" << msize.x2min << "," << msize.x2max << "]"
        << " mb_x3=[" << msize.x3min << "," << msize.x3max << "]";
    for (int n = 0; n < NTERM_GEO; ++n) {
      oss << " " << ResTermNames[n] << "=" << hbuf(n);
    }
    int n = NTERM_GEO;
    oss << " rhs_Khat=" << hbuf(n++);
    oss << " rhs_Theta=" << hbuf(n++);
    oss << " rhs_alpha=" << hbuf(n++);
    oss << " rhs_chi=" << hbuf(n++);
    oss << " alpha_full=" << hbuf(n++);
    oss << " alpha_bg=" << hbuf(n++);
    oss << " alpha_res=" << hbuf(n++);
    oss << " Khat_full=" << hbuf(n++);
    oss << " Khat_bg=" << hbuf(n++);
    oss << " Khat_res=" << hbuf(n++);
    oss << " Theta_full=" << hbuf(n++);
    oss << " Theta_res=" << hbuf(n++);
    oss << " chi_full=" << hbuf(n++);
    oss << " chi_res=" << hbuf(n++);
    oss << " detg_full=" << hbuf(n++);
    oss << " detg_bg=" << hbuf(n++);
    oss << " Gam_res=" << hbuf(n++);
    oss << " AA_full=" << hbuf(n++);
    oss << " AA_bg=" << hbuf(n++);
    oss << " K_full=" << hbuf(n++);
    oss << " K_bg=" << hbuf(n++);
    oss << " R_full=" << hbuf(n++);
    oss << " R_bg=" << hbuf(n++);
    oss << " tmunu_E=" << hbuf(n++);
    std::cout << oss.str() << std::endl;
  }
}

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
  const bool evolve_lapse_residual = pz4c->evolve_lapse_residual;
  const bool evolve_shift_residual = pz4c->evolve_shift_residual;
  const bool evolve_any_gauge_residual = evolve_lapse_residual || evolve_shift_residual;
  const bool background_adapted_residual_gauge =
      opt.residual_gauge_mode == Z4c::residual_gauge_background_adapted;
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
    GeometryData geo_full;
    ComputeGeometryData<NGHOST>(
        use_analytic_background ? full : z4c, opt, tmunu, !is_vacuum,
        size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
        m, k, j, i, geo_full);

    if (use_analytic_background) {
      GeometryData geo_bg;
      ComputeGeometryData<NGHOST>(
          bg, opt, tmunu, false,
          size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
          m, k, j, i, geo_bg);
      const Real alpha_full = full.alpha(m,k,j,i);
      const Real alpha_bg = bg.alpha(m,k,j,i);

      rhs.vKhat(m,k,j,i) =
          -(geo_full.Ddalpha - geo_bg.Ddalpha) +
          (alpha_full * (geo_full.AA + (1.0/3.0) * SQR(geo_full.K)) -
           alpha_bg * (geo_bg.AA + (1.0/3.0) * SQR(geo_bg.K))) +
          (geo_full.LKhat - geo_bg.LKhat) +
          kappa1_eff * (1.0 - opt.damp_kappa2) *
              (alpha_full * full.vTheta(m,k,j,i) -
               alpha_bg * bg.vTheta(m,k,j,i));
      if (!is_vacuum) {
        rhs.vKhat(m,k,j,i) += 4.0 * M_PI * alpha_full *
                              (geo_full.S + tmunu.E(m,k,j,i));
      }

      rhs.chi(m,k,j,i) =
          (geo_full.Lchi - geo_bg.Lchi) -
          (1.0/6.0) * opt.chi_psi_power *
              (geo_full.chi_guarded * alpha_full * geo_full.K -
               geo_bg.chi_guarded * alpha_bg * geo_bg.K);

      rhs.vTheta(m,k,j,i) =
          (geo_full.LTheta - geo_bg.LTheta) +
          0.5 * (alpha_full * geo_full.Ht - alpha_bg * geo_bg.Ht) -
          (2.0 + opt.damp_kappa2) * kappa1_eff *
              (alpha_full * full.vTheta(m,k,j,i) -
               alpha_bg * bg.vTheta(m,k,j,i));
      if (!is_vacuum) {
        rhs.vTheta(m,k,j,i) -= 8.0 * M_PI * alpha_full * tmunu.E(m,k,j,i);
      }
      rhs.vTheta(m,k,j,i) *= opt.use_z4c;

      PointRHS rhs_full_gauge;
      PointRHS rhs_bg_gauge;
      PointRHS rhs_adapted_gauge;
      if (evolve_any_gauge_residual) {
        if (background_adapted_residual_gauge) {
          BuildBackgroundAdaptedResidualGaugeRHS<NGHOST>(
              z4c, bg, opt, time,
              size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
              m, k, j, i, rhs_adapted_gauge);
        } else {
          BuildStandardPointwiseRHS(
              full, opt, tmunu, false, kappa1_eff, time,
              m, k, j, i, geo_full, rhs_full_gauge);
          BuildStandardPointwiseRHS(
              bg, opt, tmunu, false, kappa1_eff, time,
              m, k, j, i, geo_bg, rhs_bg_gauge);
        }
      }
      if (evolve_lapse_residual) {
        rhs.alpha(m,k,j,i) = background_adapted_residual_gauge
            ? rhs_adapted_gauge.alpha
            : rhs_full_gauge.alpha - rhs_bg_gauge.alpha;
      } else {
        rhs.alpha(m,k,j,i) = 0.0;
      }

      for (int a = 0; a < 3; ++a) {
        rhs.vGam_u(m,a,k,j,i) =
            2.0 * (alpha_full * geo_full.DA_u(a) -
                   alpha_bg * geo_bg.DA_u(a)) +
            (geo_full.LGam_u(a) - geo_bg.LGam_u(a));
        rhs.vGam_u(m,a,k,j,i) -=
            2.0 * kappa1_eff *
            (alpha_full * (full.vGam_u(m,a,k,j,i) - geo_full.Gamma_u(a)) -
             alpha_bg * (bg.vGam_u(m,a,k,j,i) - geo_bg.Gamma_u(a)));
        for (int b = 0; b < 3; ++b) {
          rhs.vGam_u(m,a,k,j,i) -= 2.0 *
                                   (geo_full.A_uu(a,b) * geo_full.dalpha_d(b) -
                                    geo_bg.A_uu(a,b) * geo_bg.dalpha_d(b));
          if (!is_vacuum) {
            rhs.vGam_u(m,a,k,j,i) -= 16.0 * M_PI * alpha_full *
                                     geo_full.g_uu(a,b) * tmunu.S_d(m,b,k,j,i);
          }
        }
        if (evolve_shift_residual) {
          rhs.beta_u(m,a,k,j,i) = background_adapted_residual_gauge
              ? rhs_adapted_gauge.beta_u(a)
              : rhs_full_gauge.beta_u(a) - rhs_bg_gauge.beta_u(a);
          rhs.vB_d(m,a,k,j,i) = background_adapted_residual_gauge
              ? rhs_adapted_gauge.vB_d(a)
              : rhs_full_gauge.vB_d(a) - rhs_bg_gauge.vB_d(a);
        } else {
          rhs.beta_u(m,a,k,j,i) = 0.0;
          rhs.vB_d(m,a,k,j,i) = 0.0;
        }
      }
      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        rhs.g_dd(m,a,b,k,j,i) =
            -2.0 * (alpha_full * full.vA_dd(m,a,b,k,j,i) -
                    alpha_bg * bg.vA_dd(m,a,b,k,j,i)) +
            (geo_full.Lg_dd(a,b) - geo_bg.Lg_dd(a,b));

        rhs.vA_dd(m,a,b,k,j,i) =
            geo_full.oopsi4 *
                (-geo_full.Ddalpha_dd(a,b) +
                 alpha_full * (geo_full.R_dd(a,b) + geo_full.Rphi_dd(a,b))) -
            geo_bg.oopsi4 *
                (-geo_bg.Ddalpha_dd(a,b) +
                 alpha_bg * (geo_bg.R_dd(a,b) + geo_bg.Rphi_dd(a,b)));
        rhs.vA_dd(m,a,b,k,j,i) -=
            (1.0/3.0) *
            (full.g_dd(m,a,b,k,j,i) * (-geo_full.Ddalpha + alpha_full * geo_full.R) -
             bg.g_dd(m,a,b,k,j,i) * (-geo_bg.Ddalpha + alpha_bg * geo_bg.R));
        rhs.vA_dd(m,a,b,k,j,i) +=
            alpha_full * (geo_full.K * full.vA_dd(m,a,b,k,j,i) -
                          2.0 * geo_full.AA_dd(a,b)) -
            alpha_bg * (geo_bg.K * bg.vA_dd(m,a,b,k,j,i) -
                        2.0 * geo_bg.AA_dd(a,b));
        rhs.vA_dd(m,a,b,k,j,i) += geo_full.LA_dd(a,b) - geo_bg.LA_dd(a,b);
        if (!is_vacuum) {
          rhs.vA_dd(m,a,b,k,j,i) -= 8.0 * M_PI * alpha_full *
                                    (geo_full.oopsi4 * tmunu.S_dd(m,a,b,k,j,i) -
                                     (1.0/3.0) * geo_full.S *
                                         full.g_dd(m,a,b,k,j,i));
        }
      }
    } else {
      PointRHS rhs_full;
      BuildStandardPointwiseRHS(
          z4c, opt, tmunu, !is_vacuum, kappa1_eff, time,
          m, k, j, i, geo_full, rhs_full);
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
  if (opt.rhs_term_debug && use_analytic_background && stage == 1 &&
      (pmy_pack->pmesh->ncycle %
       std::max(1, opt.rhs_term_debug_stride)) == 0) {
    ResidualRHSTermDebug<NGHOST>(pmy_pack, pz4c, kappa1_eff, time, stage);
  }
  pz4c->DebugDumpState("post_rhs", u_rhs, false, time, stage);

  return TaskStatus::complete;
}

template TaskStatus Z4c::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<4>(Driver *pdriver, int stage);
} // namespace z4c
