#ifndef RADIATION_M1_TENSORS_HPP
#define RADIATION_M1_TENSORS_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_tensors.hpp
//  \brief definitions for tensor operations & loading tensors

#include "athena.hpp"
#include "athena_tensor.hpp"

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_dot
//  \brief function to compute g^ab F_a G_b (or g_ab F^a G^b)
KOKKOS_INLINE_FUNCTION
Real tensor_dot(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &G_d) {
  Real F2 = 0.;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      F2 += g_uu(a, b) * F_d(a) * G_d(b);
    }
  }
  return F2;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_dot (special)
//  \brief function to compute g^ab F_a G_b (or g_ab F^a G^b)
KOKKOS_INLINE_FUNCTION
Real tensor_dot(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                const AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> &G_d) {
  Real G4d_d[4] = {0, G_d(0), G_d(1), G_d(2)};
  Real F2 = 0.;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      F2 += g_uu(a, b) * F_d(a) * G4d_d[b];
    }
  }
  return F2;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_dot
//  \brief function to compute P^ab K_ab
KOKKOS_INLINE_FUNCTION
Real tensor_dot(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
                const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd) {
  Real F2 = 0.;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          F2 += g_uu(a, c) * g_uu(b, d) * P_dd(c, d) * K_dd(a, b);
        }
      }
    }
  }
  return F2;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_dot
//  \brief function to Euclidean dot product
KOKKOS_INLINE_FUNCTION
Real tensor_dot(const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &G_d) {
  return F_d(0) * G_d(0) + F_d(1) * G_d(1) + F_d(2) * G_d(2) + F_d(3) * G_d(3);
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_contract
//  \brief find F_a = g_ab F^b or F^a = g^ab F_b
KOKKOS_INLINE_FUNCTION
void tensor_contract(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_u,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d) {
  for (int a = 0; a < 4; ++a) {
    F_d(a) = 0;
    for (int b = 0; b < 4; ++b) {
      F_d(a) += g_dd(a, b) * F_u(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::tensor_contract
//  \brief find P^a_b = g^ac P^cb
KOKKOS_INLINE_FUNCTION
void tensor_contract(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &P_ud) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      P_ud(a, b) = 0;
      for (int c = 0; c < 4; ++c) {
        P_ud(a, b) += g_uu(a, c) * P_dd(c, b);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::pack_n_d
//  \brief populate normal vector
KOKKOS_INLINE_FUNCTION
void pack_n_d(const Real &alpha,
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d) {
  n_d(0) = -alpha;
  n_d(1) = 0;
  n_d(2) = 0;
  n_d(3) = 0;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::pack_beta_u
//  \brief populate shift
KOKKOS_INLINE_FUNCTION
void pack_beta_u(const Real &betax_u, const Real &betay_u, const Real &betaz_u,
                 AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &beta_u) {
  beta_u(0) = 0;
  beta_u(1) = betax_u;
  beta_u(2) = betay_u;
  beta_u(3) = betaz_u;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::pack_u_u
//  \brief populate u_u
KOKKOS_INLINE_FUNCTION
void pack_u_u(const Real &u_mu_0, const Real &u_mu_1, const Real &u_mu_2,
              const Real &u_mu_3,
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u) {
  u_u(0) = u_mu_0;
  u_u(1) = u_mu_1;
  u_u(2) = u_mu_2;
  u_u(3) = u_mu_3;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::pack_v_u
//  \brief populate v_u
KOKKOS_INLINE_FUNCTION
void pack_v_u(const Real &u_mu_0, const Real &u_mu_1, const Real &u_mu_2,
              const Real &u_mu_3,
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u) {
  v_u(0) = 0;
  v_u(1) = u_mu_1 / u_mu_0;
  v_u(2) = u_mu_2 / u_mu_0;
  v_u(3) = u_mu_3 / u_mu_0;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::pack_F_d
//  \brief populate F_d
KOKKOS_INLINE_FUNCTION
void pack_F_d(const Real &betax_u, const Real &betay_u, const Real &betaz_u,
              const Real &Fx, const Real &Fy, const Real &Fz,
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d) {
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  F_d(0) = betax_u * Fx + betay_u * Fy + betaz_u * Fz;
  F_d(1) = Fx;
  F_d(2) = Fy;
  F_d(3) = Fz;
}

//----------------------------------------------------------------------------------------
//! \fn radiationm1::pack_P_dd
//  \brief populate P_dd
KOKKOS_INLINE_FUNCTION
void pack_P_dd(const Real &betax_u, const Real &betay_u, const Real &betaz_u,
               const Real &Pxx_dd, const Real &Pxy_dd, const Real &Pxz_dd,
               const Real &Pyy_dd, const Real &Pyz_dd, const Real &Pzz_dd,
               AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd) {
  const Real Pbetax = Pxx_dd * betax_u + Pxy_dd * betay_u + Pxz_dd * betaz_u;
  const Real Pbetay = Pxy_dd * betax_u + Pyy_dd * betay_u + Pyz_dd * betaz_u;
  const Real Pbetaz = Pxz_dd * betax_u + Pyz_dd * betay_u + Pzz_dd * betaz_u;

  // P_00 = g_0i g_k0 P^ik = beta^i beta^k P_ik
  P_dd(0, 0) = Pbetax * betax_u + Pbetay * betay_u + Pbetaz * betaz_u;

  // P_0i = g_0j g_ki P^jk = beta_j P_i^j = beta^j P_ij
  P_dd(0, 1) = P_dd(1, 0) = Pbetax;
  P_dd(0, 2) = P_dd(2, 0) = Pbetay;
  P_dd(0, 3) = P_dd(3, 0) = Pbetaz;

  P_dd(1, 1) = Pxx_dd;
  P_dd(1, 2) = P_dd(2, 1) = Pxy_dd;
  P_dd(1, 3) = P_dd(3, 1) = Pxz_dd;
  P_dd(2, 2) = Pyy_dd;
  P_dd(2, 3) = P_dd(3, 2) = Pyz_dd;
  P_dd(3, 3) = Pzz_dd;
}

#endif // RADIATION_M1_TENSORS_HPP
