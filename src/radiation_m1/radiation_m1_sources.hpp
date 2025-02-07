#ifndef RADIATION_M1_SOURCES_HPP
#define RADIATION_M1_SOURCES_HPP

#include <athena_tensor.hpp>

#include "athena.hpp"
#include "radiation_m1_roots_brent.hpp"
#include "radiation_m1_roots_hybridsj.hpp"

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

KOKKOS_INLINE_FUNCTION Real set_dthin(Real chi) { return 1.5 * chi - 0.5; }

KOKKOS_INLINE_FUNCTION Real set_dthick(Real chi) { return 1.5 * (1 - chi); }

// Low level kernel computing the Jacobian matrix
KOKKOS_INLINE_FUNCTION void source_jacobian(
    const Real qpre[4], AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>(&F_u),
    Real &F2, Real &chi, Real &kapa, Real &kaps,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>(&v_u),
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>(&v_d), Real &v2, double &W,
    double &alpha, double &cdt, const Real qstar[4], Real (&J)[4][4]) {
  const Real kapas = kapa + kaps;
  const Real alpW = alpha * W;

  const Real dthin = set_dthin(chi);
  const Real dthick = set_dthick(chi);

  const Real vx = v_d(1);
  const Real vy = v_d(2);
  const Real vz = v_d(3);
  const Real W2 = SQ(W);
  const Real W3 = W2 * W;

  const Real vdotF = F_u(1) * v_d(1) + F_u(2) * v_d(2) + F_u(3) * v_d(3);
  const Real normF = Kokkos::sqrt(F2);
  const Real inormF = (normF > 0 ? 1 / normF : 0);
  const Real vdothatf = vdotF * inormF;
  const Real vdothatf2 = SQ(vdothatf);
  const Real hatfx = qpre[1] * inormF;  // hatf_i
  const Real hatfy = qpre[2] * inormF;
  const Real hatfz = qpre[3] * inormF;
  const Real hatfupx = F_u(1) * inormF;  // hatf^i
  const Real hatfupy = F_u(2) * inormF;
  const Real hatfupz = F_u(3) * inormF;
  const Real e = qpre[0];
  const Real eonormF = Kokkos::min<Real>(e * inormF, 1.0);
  // with factor dthin ...

  // drvts of J
  Real JdE = W2 + dthin * vdothatf2 * W2 +
             (dthick * (3 - 2 * W2) * (-1 + W2)) / (1 + 2 * W2);

  Real JdFv = 2 * W2 *
              (-1 + (dthin * eonormF * vdothatf) +
               (2 * dthick * (-1 + W2)) / (1 + 2 * W2));
  Real JdFf = (-2 * dthin * eonormF * vdothatf2 * W2);

  Real JdFx = JdFv * v_u(1) + JdFf * hatfupx;
  Real JdFy = JdFv * v_u(2) + JdFf * hatfupy;
  Real JdFz = JdFv * v_u(3) + JdFf * hatfupz;

  // drvts of Hi
  Real HdEv =
      W3 * (-1 - dthin * vdothatf2 + (dthick * (-3 + 2 * W2)) / (1 + 2 * W2));
  Real HdEf = -(dthin * vdothatf * W);

  Real HxdE = HdEv * vx + HdEf * hatfx;
  Real HydE = HdEv * vy + HdEf * hatfy;
  Real HzdE = HdEv * vz + HdEf * hatfz;

  Real HdFdelta = (1 - dthick * v2 - (dthin * eonormF * vdothatf)) * W;
  Real HdFvv = (2 * (1 - dthin * eonormF * vdothatf) * W3) +
               dthick * W * (2 - 2 * W2 + 1 / (-1 - 2 * W2));
  Real HdFff = (2 * dthin * eonormF * vdothatf * W);
  Real HdFvf = (2 * dthin * eonormF * vdothatf2 * W3);
  Real HdFfv = -(dthin * eonormF * W);

  Real HxdFx = HdFdelta + HdFvv * vx * v_u(1) + HdFff * hatfx * hatfupx +
               HdFvf * vx * hatfupx + HdFfv * hatfx * v_u(1);
  Real HydFx = HdFvv * vy * v_u(1) + HdFff * hatfy * hatfupx +
               HdFvf * vy * hatfupx + HdFfv * hatfy * v_u(1);
  Real HzdFx = HdFvv * vz * v_u(1) + HdFff * hatfz * hatfupx +
               HdFvf * vz * hatfupx + HdFfv * hatfz * v_u(1);

  Real HxdFy = HdFvv * vx * v_u(2) + HdFff * hatfx * hatfupy +
               HdFvf * vx * hatfupy + HdFfv * hatfx * v_u(2);
  Real HydFy = HdFdelta + HdFvv * vy * v_u(2) + HdFff * hatfy * hatfupy +
               HdFvf * vy * hatfupy + HdFfv * hatfy * v_u(2);
  Real HzdFy = HdFvv * vz * v_u(2) + HdFff * hatfz * hatfupy +
               HdFvf * vz * hatfupy + HdFfv * hatfz * v_u(2);

  Real HxdFz = HdFvv * vx * v_u(3) + HdFff * hatfx * hatfupz +
               HdFvf * vx * hatfupz + HdFfv * hatfx * v_u(3);
  Real HydFz = HdFvv * vy * v_u(3) + HdFff * hatfy * hatfupz +
               HdFvf * vy * hatfupz + HdFfv * hatfy * v_u(3);
  Real HzdFz = HdFdelta + HdFvv * vz * v_u(3) + HdFff * hatfz * hatfupz +
               HdFvf * vz * hatfupz + HdFfv * hatfz * v_u(3);

  // Build the Jacobian
  Real J00 = -alpW * (kapas - kaps * JdE);

  Real J0x = +alpW * kaps * JdFx + alpW * kapas * v_u(1);
  Real J0y = +alpW * kaps * JdFy + alpW * kapas * v_u(2);
  Real J0z = +alpW * kaps * JdFz + alpW * kapas * v_u(3);

  Real Jx0 = -alpha * (kapas * HxdE + W * kapa * vx * JdE);
  Real Jy0 = -alpha * (kapas * HydE + W * kapa * vy * JdE);
  Real Jz0 = -alpha * (kapas * HzdE + W * kapa * vz * JdE);

  Real Jxx = -alpha * (kapas * HxdFx + W * kapa * vx * JdFx);
  Real Jxy = -alpha * (kapas * HxdFy + W * kapa * vx * JdFy);
  Real Jxz = -alpha * (kapas * HxdFz + W * kapa * vx * JdFz);

  Real Jyy = -alpha * (kapas * HydFx + W * kapa * vy * JdFx);
  Real Jyx = -alpha * (kapas * HydFy + W * kapa * vy * JdFy);
  Real Jyz = -alpha * (kapas * HydFz + W * kapa * vy * JdFz);

  Real Jzx = -alpha * (kapas * HzdFx + W * kapa * vz * JdFx);
  Real Jzy = -alpha * (kapas * HzdFy + W * kapa * vz * JdFy);
  Real Jzz = -alpha * (kapas * HzdFz + W * kapa * vz * JdFz);

  // Store Jacobian into J
  J[0][0] = 1 - cdt * J00;
  J[0][1] = -cdt * J0x;
  J[0][2] = -cdt * J0y;
  J[0][2] = -cdt * J0z;
  J[1][0] = -cdt * Jx0;
  J[1][1] = 1 - cdt * Jxx;
  J[1][2] = -cdt * Jxy;
  J[1][3] = -cdt * Jxz;
  J[2][0] = -cdt * Jy0;
  J[2][1] = -cdt * Jyx;
  J[2][2] = 1 - cdt * Jyy;
  J[2][3] = -cdt * Jyz;
  J[3][0] = -cdt * Jz0;
  J[3][1] = -cdt * Jzx;
  J[3][2] = -cdt * Jzy;
  J[3][3] = 1 - cdt * Jzz;
}

// Solves the implicit problem
// .  q^new = q^star + dt S[q^new]
// The source term is S^a = (eta - ka J) u^a - (ka + ks) H^a and includes
// also emission.
int source_update(
    const Real &cdt, const Real &alp,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
    const Real &W, const Real &Eold,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fold_d,
    const Real &Estar,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
    const Real &eta, const Real &kabs, const Real &kscat, Real &chi, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d) {
  return 0;
}

#endif  // RADIATION_M1_SOURCES_HPP