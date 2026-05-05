//========================================================================================
// Valencia GRMHD Stress-Energy Helpers (AthenaK style)
// Conventions:
//  - Primitive velocity: u_prim^i = W v^i (v^i is Eulerian 3-velocity)
//  - Magnetic field stored densitized: B_tilde^i = sqrt(gamma) * B^i
//  - Spatial metric: gamma_ij = g_dd (3x3 symmetric)
//  - EOS: Gamma-law, h = 1 + Gamma * P / [rho (Gamma - 1)]
//========================================================================================

#ifndef VALENCIA_STRESS_ENERGY_HPP_
#define VALENCIA_STRESS_ENERGY_HPP_

#include "athena.hpp"
#include "coordinates/adm.hpp" // for adm::SpatialDet
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"

// -----------------------------------------------------------------------------
// 1. From primitive W v^i to W, v^i, v_i (Eulerian velocity)
//    Inputs:
//      prim_Wv_u[3] : primitive velocity = W v^i
//      g_dd_1d[6]   : spatial metric gamma_ij
//    Outputs:
//      W            : Lorentz factor
//      v_u[3]       : Eulerian v^i
//      v_d[3]       : Eulerian v_i = gamma_ij v^j
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void ValenciaPrimsToEulerianVelocity(const Real prim_Wv_u[3],
                                     const Real g_dd_1d[6],
                                     Real &W,
                                     Real v_u[3],
                                     Real v_d[3]) {
  // u_sq = gamma_ij (W v^i)(W v^j) = W^2 v^2
  const Real u_sq = Primitive::SquareVector(prim_Wv_u, g_dd_1d);
  W = std::sqrt(1.0 + u_sq);
  const Real invW = (W > 0.0) ? 1.0/W : 0.0;

  // v^i = (W v^i) / W
  for (int i = 0; i < 3; ++i) {
    v_u[i] = prim_Wv_u[i] * invW;
  }

  // Build v_i = gamma_ij v^j
  const Real &g00 = g_dd_1d[0];
  const Real &g01 = g_dd_1d[1];
  const Real &g02 = g_dd_1d[2];
  const Real &g11 = g_dd_1d[3];
  const Real &g12 = g_dd_1d[4];
  const Real &g22 = g_dd_1d[5];

  v_d[0] = g00*v_u[0] + g01*v_u[1] + g02*v_u[2];
  v_d[1] = g01*v_u[0] + g11*v_u[1] + g12*v_u[2];
  v_d[2] = g02*v_u[0] + g12*v_u[1] + g22*v_u[2];
}

// -----------------------------------------------------------------------------
// 2. From densitized magnetic field \tilde{B}^i to Eulerian B^i, B_i
//    Inputs:
//      B_tilde_u[3] : densitized B^i = sqrt(gamma) * B^i
//      g_dd_1d[6]   : spatial metric gamma_ij
//    Outputs:
//      sqrt_gamma   : sqrt(det(gamma_ij))
//      B_u[3]       : Eulerian B^i
//      B_d[3]       : Eulerian B_i = gamma_ij B^j
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void ValenciaEulerianMagneticFromDensitized(const Real B_tilde_u[3],
                                            const Real g_dd_1d[6],
                                            Real &sqrt_gamma,
                                            Real B_u[3],
                                            Real B_d[3]) {
  const Real &g00 = g_dd_1d[0];
  const Real &g01 = g_dd_1d[1];
  const Real &g02 = g_dd_1d[2];
  const Real &g11 = g_dd_1d[3];
  const Real &g12 = g_dd_1d[4];
  const Real &g22 = g_dd_1d[5];

  // det(gamma_ij)
  const Real detg =
      g00*(g11*g22 - g12*g12)
    - g01*(g01*g22 - g12*g02)
    + g02*(g01*g12 - g11*g02);

  sqrt_gamma = (detg > 0.0) ? std::sqrt(detg) : 0.0;
  const Real inv_sqrt_gamma = (sqrt_gamma > 0.0) ? 1.0/sqrt_gamma : 0.0;

  // B^i = \tilde{B}^i / sqrt(gamma)
  for (int i = 0; i < 3; ++i) {
    B_u[i] = B_tilde_u[i] * inv_sqrt_gamma;
  }

  // B_i = gamma_ij B^j
  B_d[0] = g00*B_u[0] + g01*B_u[1] + g02*B_u[2];
  B_d[1] = g01*B_u[0] + g11*B_u[1] + g12*B_u[2];
  B_d[2] = g02*B_u[0] + g12*B_u[1] + g22*B_u[2];
}

// -----------------------------------------------------------------------------
// 3. Comoving magnetic field 4-vector b^mu and b^2 in Valencia conventions
//    Inputs:
//      v_u[3], v_d[3] : Eulerian v^i, v_i
//      B_u[3], B_d[3] : Eulerian B^i, B_i
//      W              : Lorentz factor
//    Outputs:
//      Bv             : B·v = B^i v_i
//      b0             : b^0
//      b_u[3]         : b^i
//      b_d[3]         : b_i
//      b2             : b^2 = b^mu b_mu
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void ValenciaComovingB(const Real v_u[3], const Real v_d[3],
                       const Real B_u[3], const Real B_d[3],
                       const Real W,
                       Real &Bv,
                       Real &b0,
                       Real b_u[3],
                       Real b_d[3],
                       Real &b2) {
  // B·v = B^i v_i = B_i v^i  (they are numerically identical)
  Bv = B_u[0]*v_d[0] + B_u[1]*v_d[1] + B_u[2]*v_d[2];

  const Real invW = (W > 0.0) ? 1.0/W : 0.0;

  // Valencia relations in Eulerian frame:
  //   b^0 = W (B·v)
  //   b_i = B_i / W + (B·v) v_i
  //   b^i = B^i / W + b^0 v^i
  //   b^2 = B^2 / W^2 + (B·v)^2
  b0 = W * Bv;

  // b_i
  for (int i = 0; i < 3; ++i) {
    b_d[i] = B_d[i] * invW + b0 * v_d[i];
  }

  // b^i
  for (int i = 0; i < 3; ++i) {
    b_u[i] = B_u[i] * invW + b0 * v_u[i];
  }

  // B^2 = B^i B_i
  const Real B2 = B_u[0]*B_d[0] + B_u[1]*B_d[1] + B_u[2]*B_d[2];

  // b^2
  b2 = B2 * invW * invW + Bv * Bv;
}

// -----------------------------------------------------------------------------
// 4. Gamma-law EOS enthalpy
//    h = 1 + Gamma * P / [rho (Gamma - 1)]
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
Real GammaLawSpecificEnthalpy(const Real rho, const Real P, const Real gamma_gas) {
  const Real gm1 = gamma_gas - 1.0;
  return 1.0 + (gamma_gas * P) / (rho * gm1);
}

// -----------------------------------------------------------------------------
// 5. Decomposed stress-energy: fluid + EM, in ADM / Eulerian split
//    Inputs:
//      rho, P, gamma_gas
//      v_u[3], v_d[3], W
//      B_u[3], B_d[3]
//      b0, b_d[3], b2
//      g_dd_1d[6] : gamma_ij
//
//    Outputs:
//      E_fluid, E_em
//      S_fluid_d[3], S_em_d[3]         : S_i
//      S_fluid_dd[3][3], S_em_dd[3][3] : S_ij
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void ValenciaStressEnergyDecomposed(const Real rho,
                                    const Real P,
                                    const Real gamma_gas,
                                    const Real v_u[3],
                                    const Real v_d[3],
                                    const Real W,
                                    const Real B_u[3],
                                    const Real B_d[3],
                                    const Real b0,
                                    const Real b_d[3],
                                    const Real b2,
                                    const Real g_dd_1d[6],
                                    Real &E_fluid,
                                    Real &E_em,
                                    Real S_fluid_d[3],
                                    Real S_em_d[3],
                                    Real S_fluid_dd[3][3],
                                    Real S_em_dd[3][3]) {
  const Real h = GammaLawSpecificEnthalpy(rho, P, gamma_gas);
  const Real W2 = W * W;

  // Fluid part (perfect fluid)
  // E_fluid = rho h W^2 - P
  E_fluid = rho * h * W2 - P;

  for (int i = 0; i < 3; ++i) {
    // S_i^fluid = rho h W^2 v_i
    S_fluid_d[i] = rho * h * W2 * v_d[i];
  }

  const Real &g00 = g_dd_1d[0];
  const Real &g01 = g_dd_1d[1];
  const Real &g02 = g_dd_1d[2];
  const Real &g11 = g_dd_1d[3];
  const Real &g12 = g_dd_1d[4];
  const Real &g22 = g_dd_1d[5];

  // S_ij^fluid = rho h W^2 v_i v_j + P gamma_ij
  S_fluid_dd[0][0] = rho*h*W2*v_d[0]*v_d[0] + P*g00;
  S_fluid_dd[0][1] = rho*h*W2*v_d[0]*v_d[1] + P*g01;
  S_fluid_dd[0][2] = rho*h*W2*v_d[0]*v_d[2] + P*g02;
  S_fluid_dd[1][0] = S_fluid_dd[0][1];
  S_fluid_dd[1][1] = rho*h*W2*v_d[1]*v_d[1] + P*g11;
  S_fluid_dd[1][2] = rho*h*W2*v_d[1]*v_d[2] + P*g12;
  S_fluid_dd[2][0] = S_fluid_dd[0][2];
  S_fluid_dd[2][1] = S_fluid_dd[1][2];
  S_fluid_dd[2][2] = rho*h*W2*v_d[2]*v_d[2] + P*g22;

  // EM part
  // E_EM = b^2 W^2 - (1/2) b^2 - (b^0)^2
  E_em = b2 * W2 - 0.5*b2 - b0*b0;

  for (int i = 0; i < 3; ++i) {
    // S_i^EM = b^2 W^2 v_i - b^0 b_i
    S_em_d[i] = b2 * W2 * v_d[i] - b0 * b_d[i];
  }

  // S_ij^EM = b^2 W^2 v_i v_j + (b^2/2) gamma_ij - b_i b_j
  S_em_dd[0][0] = b2*W2*v_d[0]*v_d[0] + 0.5*b2*g00 - b_d[0]*b_d[0];
  S_em_dd[0][1] = b2*W2*v_d[0]*v_d[1] + 0.5*b2*g01 - b_d[0]*b_d[1];
  S_em_dd[0][2] = b2*W2*v_d[0]*v_d[2] + 0.5*b2*g02 - b_d[0]*b_d[2];
  S_em_dd[1][0] = S_em_dd[0][1];
  S_em_dd[1][1] = b2*W2*v_d[1]*v_d[1] + 0.5*b2*g11 - b_d[1]*b_d[1];
  S_em_dd[1][2] = b2*W2*v_d[1]*v_d[2] + 0.5*b2*g12 - b_d[1]*b_d[2];
  S_em_dd[2][0] = S_em_dd[0][2];
  S_em_dd[2][1] = S_em_dd[1][2];
  S_em_dd[2][2] = b2*W2*v_d[2]*v_d[2] + 0.5*b2*g22 - b_d[2]*b_d[2];
}

// -----------------------------------------------------------------------------
// 6. Convenience: total E, S_i, S_ij if you want them directly
// -----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void ValenciaStressEnergyTotal(const Real rho,
                               const Real P,
                               const Real gamma_gas,
                               const Real v_u[3],
                               const Real v_d[3],
                               const Real W,
                               const Real B_u[3],
                               const Real B_d[3],
                               const Real b0,
                               const Real b_d[3],
                               const Real b2,
                               const Real g_dd_1d[6],
                               Real &E,
                               Real S_d[3],
                               Real S_dd[3][3]) {
  Real E_f, E_em;
  Real S_f[3], S_em[3];
  Real Sf_dd[3][3], Sem_dd[3][3];

  ValenciaStressEnergyDecomposed(rho, P, gamma_gas,
                                 v_u, v_d, W,
                                 B_u, B_d,
                                 b0, b_d, b2,
                                 g_dd_1d,
                                 E_f, E_em,
                                 S_f, S_em,
                                 Sf_dd, Sem_dd);

  E = E_f + E_em;
  for (int i = 0; i < 3; ++i) {
    S_d[i] = S_f[i] + S_em[i];
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      S_dd[i][j] = Sf_dd[i][j] + Sem_dd[i][j];
    }
  }
}

#endif // VALENCIA_STRESS_ENERGY_HPP_

