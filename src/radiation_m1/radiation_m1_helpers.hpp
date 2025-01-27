#ifndef RADIATION_M1_CLOSURE_HPP
#define RADIATION_M1_CLOSURE_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_helpers.hpp
//  \brief helper functions for grey M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::minmod2
//  \brief double minmod
KOKKOS_INLINE_FUNCTION
Real minmod2(Real rl, Real rp, Real th) {
  return Kokkos::min(1.0, Kokkos::min(th * rl, th * rp));
}

//----------------------------------------------------------------------------------------
//! \fn int radiationm1::CombinedIdx
//  \brief given flavor index, variable index and total no. of vars compute
//  combined idx
KOKKOS_INLINE_FUNCTION
int CombinedIdx(int nuidx, int varidx, int nvars) {
  return varidx + nuidx * nvars;
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_proj
//  \brief Fluid projector: delta^a_b + u^a u_b
KOKKOS_INLINE_FUNCTION
void calc_proj(const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
               const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
               AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      proj_ud(a, b) = (a == b) + u_u(a) * u_d(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_K_from_rT
//  \brief Project out the radiation pressure tensor (in any frame)
KOKKOS_INLINE_FUNCTION
void calc_K_from_rT(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      K_dd(a, b) = 0.;
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          K_dd(a, b) += proj_ud(c, a) * proj_ud(d, b) * rT_dd(c, d);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_Pthin
//  \brief Compute lab radiation pressure in the thin limit
KOKKOS_INLINE_FUNCTION
void calc_Pthin(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                const Real &E,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
                const RadiationM1Params &params) {

  const Real F2 = tensor_dot(g_uu, F_d, F_d);
#if (RADIATION_M1_CLS_METHOD == RADIATION_M1_CLS_SHIBATA)
  Real fac = (F2 > 0 ? E / F2 : 0);
#else
  Real fac = (E > params.rad_E_floor ? 1.0 / E : 0);
  Real lim = Kokkos::max(E * E, params.rad_E_floor);
  if (F2 > lim) {
    fac = fac * lim / F2;
  }
#endif
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      P_dd(a, b) = fac * F_d(a) * F_d(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_Pthick
//  \brief Compute radiation pressure in the thick limit
KOKKOS_INLINE_FUNCTION
void calc_Pthick(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
                 const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
                 const Real &W,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
                 const Real &E,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                 AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd) {

  const Real v_dot_F = tensor_dot(g_uu, v_d, F_d);
  const Real W2 = W * W;
  const Real coef = 1. / (2. * W2 + 1.);

  // J/3
  const Real Jo3 = coef * ((2. * W2 - 1.) * E - 2. * W2 * v_dot_F);

  // tH = gamma_ud H_d
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tH_d{};
  for (int a = 0; a < 4; ++a) {
    tH_d(a) = F_d(a) / W +
              coef * W * v_d(a) * ((4. * W2 + 1.) * v_dot_F - 4. * W2 * E);
  }

  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      P_dd(a, b) =
          Jo3 * (4. * W2 * v_d(a) * v_d(b) + g_dd(a, b) + n_d(a) * n_d(b));
      P_dd(a, b) += W * (tH_d(a) * v_d(b) + tH_d(b) * v_d(a));
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_Kthin
//  \brief Compute fluid frame pressure in the thin limit
KOKKOS_INLINE_FUNCTION
void calc_Kthin(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
                const Real &W,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
                const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
                const Real &J,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd,
                const Real &rad_E_floor) {
  // H_mu n^mu
  const Real H_dot_n = W * tensor_dot(g_uu, n_d, H_d);

  // J
  const Real E = Kokkos::max(rad_E_floor, SQ(H_dot_n - J * W) / J);

  // F_mu u^mu
  const Real F_dot_u = H_dot_n + W * (E - J);

  // Compute F_mu
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  const Real coef = 1.0 / (W - F_dot_u / E);
  for (int a = 0; a < 4; ++a) {
    F_d(a) = coef * (H_d(a) + E * W * (W * u_d(a) - n_d(a)) +
                     F_dot_u * (n_d(a) - 2 * W * u_d(a)) +
                     (1.0 / E) * SQ(F_dot_u) * u_d(a));
  }
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      T_dd(a, b) = E * n_d(a) * n_d(b) + F_d(a) * n_d(b) + n_d(a) * F_d(b) +
                   (F_d(a) * F_d(b)) / E;
    }
  }
  // Project radiation tensor to obtain radiation pressure
  calc_K_from_rT(T_dd, proj_ud, K_dd);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_Kthick
//  \brief Compute fluid frame pressure in the thick limit
KOKKOS_INLINE_FUNCTION
void calc_Kthick(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
                 const Real &J,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
                 AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      K_dd(a, b) = (1.0 / 3.0) * J * (g_dd(a, b) + u_d(a) * u_d(b));
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::flux_factor
//  \brief Compute the flux factor xi = H_a H^a / J^2
KOKKOS_INLINE_FUNCTION
Real flux_factor(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                 const Real &J,
                 AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
                 const Real &rad_E_floor) {
  const Real xi = (J > rad_E_floor ? tensor_dot(g_uu, H_d, H_d) / SQ(J) : 0);
  return Kokkos::max(0.0, Kokkos::min(xi, 1.0));
}

// Closures
KOKKOS_INLINE_FUNCTION Real eddington(const Real xi) { return 1.0 / 3.0; }
KOKKOS_INLINE_FUNCTION Real kershaw(const Real xi) {
  return 1.0 / 3.0 + 2.0 / 3.0 * xi * xi;
}
KOKKOS_INLINE_FUNCTION Real minerbo(const Real xi) {
  return 1.0 / 3.0 + xi * xi * (6.0 - 2.0 * xi + 6.0 * xi * xi) / 15.0;
}
KOKKOS_INLINE_FUNCTION Real thin(const Real xi) { return 1.0; }

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::assemble_fnu
//  \brief Assemble the unit-norm radiation number current
KOKKOS_INLINE_FUNCTION
void assemble_fnu(const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
                  const Real &J,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_u,
                  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &fnu_u,
                  const RadiationM1Params &params) {
  for (int a = 0; a < 4; ++a) {
    fnu_u(a) = u_u(a) + (J > params.rad_E_floor ? H_u(a) / J : 0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::compute_Gamma
//  \brief Compute the ratio of neutrino number densities in the lab and fluid
//  frame
KOKKOS_INLINE_FUNCTION
Real compute_Gamma(const Real &W,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
                   const Real &J, const Real &E,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                   const RadiationM1Params &params) {
  if (E > params.rad_E_floor && J > params.rad_E_floor) {
    const Real f_dot_v =
        Kokkos::min(tensor_dot(F_d, v_u) / E, 1 - params.rad_eps);
    return W * (E / J) * (1 - f_dot_v);
  }
  return 1;
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::assemble_rT
//  \brief Assemble the radiation stress tensor in any frame
KOKKOS_INLINE_FUNCTION
void assemble_rT(const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
                 const Real &J,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
                 const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd,
                 AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      rT_dd(a, b) =
          J * u_d(a) * u_d(b) + H_d(a) * u_d(b) + H_d(b) * u_d(a) + K_dd(a, b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_J_from_rT
//  \brief Project out the radiation energy (in any frame)
KOKKOS_INLINE_FUNCTION
Real calc_J_from_rT(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u) {
  return tensor_dot(rT_dd, u_u, u_u);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_H_from_rT
//  \brief Project out the radiation fluxes (in any frame)
KOKKOS_INLINE_FUNCTION
void calc_H_from_rT(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d) {
  for (int a = 0; a < 4; ++a) {
    H_d(a) = 0.;
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        H_d(a) -= proj_ud(b, a) * u_u(c) * rT_dd(b, c);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::apply_closure
//  \brief Computes the closure in the lab frame given the Eddington factor chi
KOKKOS_INLINE_FUNCTION
void apply_closure(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &proj_ud,
    const Real &E, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
    const Real &chi, AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    const RadiationM1Params &params) {

  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pthin_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pthick_dd{};

  calc_Pthin(g_uu, E, F_d, Pthin_dd, params);
  calc_Pthick(g_dd, g_uu, n_d, w_lorentz, v_d, E, F_d, Pthick_dd);

  const Real dthick = 3. * (1 - chi) / 2.;
  const Real dthin = 1. - dthick;

  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      P_dd(a, b) = dthick * Pthick_dd(a, b) + dthin * Pthin_dd(a, b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::apply_inv_closure
//  \brief Computes the closure in the fluid frame with a rootfinding procedure
KOKKOS_INLINE_FUNCTION
void apply_inv_closure(
    const Real &dthick,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d, const Real &J,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 1> &K_thick_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 1> &K_thin_dd, Real &E,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d) {}

// Computes the closure in the fluid frame with a rootfinding procedure
/*
KOKKOS_INLINE_FUNCTION
void calc_inv_closure(
        cGH const * cctkGH,
        int const i, int const j, int const k,
        int const ig,
        gsl_root_fsolver * fsolver,
        tensor::inv_metric<4> const & g_uu,
        tensor::metric<4> const & g_dd,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_u,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> gamma_ud,
        const Real w_lorentz,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud,
        const Real chi,
        const Real J,
        const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d,
        CCTK_REAL * E,
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d);
}*/

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_E_flux
//  \brief Compute the radiation energy flux
KOKKOS_INLINE_FUNCTION
Real calc_E_flux(const Real &alp,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &beta_u,
                 const Real &E,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_u,
                 const int &direction) { // 1:x 2:y 3:z
  return alp * F_u(direction) - beta_u(direction) * E;
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_F_flux
//  \brief Compute the flux of neutrino energy flux
KOKKOS_INLINE_FUNCTION
Real calc_F_flux(const Real &alp,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &beta_u,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &P_ud,
                 const int &direction,   // 1:x 2:y 3:z
                 const int &component) { // 1:x 2:y 3:z
  return alp * P_ud(direction, component) - beta_u(direction) * F_d(component);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_rad_sources
//  \brief Computes the sources S_a = [eta - k_abs J] u_a - [k_abs + k_scat] H_a
//  WARNING: be consistent with the densitization of eta, J, and H_d
KOKKOS_INLINE_FUNCTION
void calc_rad_sources(
    const Real &eta, const Real &kabs, const Real &kscat,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d, const Real &J,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &S_d) {
  for (int a = 0; a < 4; ++a) {
    S_d(a) = (eta - kabs * J) * u_d(a) - (kabs + kscat) * H_d(a);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_rE_source
//  \brief Computes the source term for E: -alp n^a S_a
KOKKOS_INLINE_FUNCTION
Real calc_rE_source(
    const Real &alp, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &S_d) {
  return -alp * tensor_dot(n_u, S_d);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_rF_source
//  \brief Computes the source term for F_a: alp gamma^b_a S_b
KOKKOS_INLINE_FUNCTION
void calc_rF_source(
    const Real &alp,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &S_d,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &tS_d) {
  for (int a = 0; a < 4; ++a) {
    tS_d(a) = 0.0;
    for (int b = 0; b < 4; ++b) {
      tS_d(a) += alp * gamma_ud(b, a) * S_d(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::apply_floor
//  \brief Enforce that E > rad_E_floor and F_a F^a < (1 - rad_eps) E^2
KOKKOS_INLINE_FUNCTION
void apply_floor(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                 Real &E, AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                 const RadiationM1Params &params) {
  E = Kokkos::max(params.rad_E_floor, E);

  const Real F2 = tensor_dot(g_uu, F_d, F_d);
  const Real lim = E * E * (1 - params.rad_eps);
  if (F2 > lim) {
    const Real fac = lim / F2;
    for (int a = 0; a < 4; ++a) {
      F_d(a) *= fac;
    }
  }
}

} // namespace radiationm1
#endif // RADIATION_M1_CLOSURE_HPP
