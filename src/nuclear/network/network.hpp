#ifndef NUCLEAR_NETWORK_NETWORK_HPP_
#define NUCLEAR_NETWORK_NETWORK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file network.hpp
//! \brief He4/C12/O16 reaction network: species data, REACLIB rates, the RHS and the
//! Jacobian of the conserved rho*X[n] system (species only; the energy release is
//! carried by the Helmholtz EOS through the binding energy in eps).

#include <cmath>
#include <string>
#include <vector>

#include "athena.hpp"
#include "units/units.hpp"

enum {
 ihe4=0,
 ic12=1,
 io16=2,
};

constexpr int NSPECIES = 3;


// Per-species constants, as functions so they are usable in both host and device code
// (nvcc does not allow namespace-scope arrays to be read inside device kernels).
KOKKOS_INLINE_FUNCTION Real aion(int n) {
  constexpr Real v[NSPECIES] = {
    4.0,  // 0 : He4
    12.0, // 1 : C12
    16.0, // 2 : O16
  };
  return v[n];
}


KOKKOS_INLINE_FUNCTION Real aion_inv(int n) {
  constexpr Real v[NSPECIES] = {
    1.0/   4.0, // 0 : He4
    1.0/  12.0, // 1 : C12
    1.0/  16.0, // 2 : O16
  };
  return v[n];
}


// Atomic mass excess [MeV] (AME2020). Must match <mhd>/dm of the Helmholtz EOS, which
// puts the binding energy into eps through these same values.
KOKKOS_INLINE_FUNCTION Real dmass_ions(int n) {
  constexpr Real v[NSPECIES] = {
    2.42491587,   // 0 : He4
    0.0,          // 1 : C12
    -4.73700141,  // 2 : O16
  };
  return v[n];
}


KOKKOS_INLINE_FUNCTION Real zion(int n) {
  constexpr Real v[NSPECIES] = {
    2.0, // 0 : He4
    6.0, // 1 : C12
    8.0, // 2 : O16
  };
  return v[n];
}


static const std::vector<std::string> spec_names = {
  "He4", // 0
  "C12", // 1
  "O16", // 2
};


KOKKOS_INLINE_FUNCTION Real mion(int n) {
  constexpr Real v[NSPECIES] = {
    6.646479071584587e-24,  // He4
    1.99264687992e-23,  // C12
    2.6560180592333686e-23,  // O16
  };
  return v[n];
}


struct tf_t {
  Real T9;
  Real T9i;
  Real T943i;
  Real T923i;
  Real T913i;
  Real T913;
  Real T923;
  Real T953;
  Real lnT9;
};


KOKKOS_INLINE_FUNCTION
tf_t evaluate_tfactors(const Real temp)
{
  tf_t tf;
  tf.T9 = temp / 1.e9;
  tf.T9i = 1.0 / tf.T9;
  tf.T913 = std::cbrt(tf.T9);
  tf.T913i = 1.0 / tf.T913;
  tf.T923i = tf.T913i * tf.T913i;
  tf.T943i = tf.T9i * tf.T913i;
  tf.T923 = tf.T913 * tf.T913;
  tf.T953 = tf.T9 * tf.T923;
  tf.lnT9 = std::log(tf.T9);
  return tf;
}


KOKKOS_INLINE_FUNCTION
void rate_C12_He4_to_O16_reaclib(const tf_t& tfactors, Real& rate, Real& drate_dT) {

  // C12 + He4 --> O16
  Real ln_set_rate{0.0};
  Real dln_set_rate_dT9{0.0};
  Real set_rate{0.0};

  // nac2 
  ln_set_rate =  254.634 + -1.84097 * tfactors.T9i + 103.411 * tfactors.T913i + -420.567 * tfactors.T913
                       + 64.0874 * tfactors.T9 + -12.4624 * tfactors.T953 + 137.303 * tfactors.lnT9;

  dln_set_rate_dT9 =  1.84097 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * 103.411 * tfactors.T943i + (1.0/3.0) * -420.567 * tfactors.T923i
                            + 64.0874 + (5.0/3.0) * -12.4624 * tfactors.T923 + 137.303 * tfactors.T9i;

  // avoid underflows by zeroing rates in [0.0, 1.e-100]
  ln_set_rate = Kokkos::fmax(ln_set_rate, -230.0);
  set_rate = std::exp(ln_set_rate);
  rate += set_rate;
  drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;

  // nac2 
  ln_set_rate =  69.6526 + -1.39254 * tfactors.T9i + 58.9128 * tfactors.T913i + -148.273 * tfactors.T913
                       + 9.08324 * tfactors.T9 + -0.541041 * tfactors.T953 + 70.3554 * tfactors.lnT9;
 
  dln_set_rate_dT9 =  1.39254 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * 58.9128 * tfactors.T943i + (1.0/3.0) * -148.273 * tfactors.T923i
                            + 9.08324 + (5.0/3.0) * -0.541041 * tfactors.T923 + 70.3554 * tfactors.T9i;

  // avoid underflows by zeroing rates in [0.0, 1.e-100]
  ln_set_rate = Kokkos::fmax(ln_set_rate, -230.0);
  set_rate = std::exp(ln_set_rate);
  rate += set_rate;
  drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;

  return;
} 


KOKKOS_INLINE_FUNCTION
void rate_He4_He4_He4_to_C12_reaclib(const tf_t& tfactors, Real& rate, Real& drate_dT) {

  // 3 He4 --> C12
  Real ln_set_rate{0.0};
  Real dln_set_rate_dT9{0.0};
  Real set_rate{0.0};

  // fy05r
  ln_set_rate =  -11.7884 + -1.02446 * tfactors.T9i + -23.57 * tfactors.T913i + 20.4886 * tfactors.T913
                       + -12.9882 * tfactors.T9 + -20.0 * tfactors.T953 + -2.16667 * tfactors.lnT9;

  dln_set_rate_dT9 =  1.02446 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -23.57 * tfactors.T943i + (1.0/3.0) * 20.4886 * tfactors.T923i
                                        + -12.9882 + (5.0/3.0) * -20.0 * tfactors.T923 + -2.16667 * tfactors.T9i;

  // avoid underflows by zeroing rates in [0.0, 1.e-100]
  ln_set_rate = Kokkos::fmax(ln_set_rate, -230.0);
  set_rate = std::exp(ln_set_rate);
  rate += set_rate;
  drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;

  // fy05n
  ln_set_rate =  -0.971052 + -37.06 * tfactors.T913i + 29.3493 * tfactors.T913
                       + -115.507 * tfactors.T9 + -10.0 * tfactors.T953 + -1.33333 * tfactors.lnT9;

  dln_set_rate_dT9 =  + -(1.0/3.0) * -37.06 * tfactors.T943i + (1.0/3.0) * 29.3493 * tfactors.T923i
                                   + -115.507 + (5.0/3.0) * -10.0 * tfactors.T923 + -1.33333 * tfactors.T9i;
  
  // avoid underflows by zeroing rates in [0.0, 1.e-100]
  ln_set_rate = Kokkos::fmax(ln_set_rate, -230.0);
  set_rate = std::exp(ln_set_rate);
  rate += set_rate;
  drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;

  // fy05r
  ln_set_rate =  -24.3505 + -4.12656 * tfactors.T9i + -13.49 * tfactors.T913i + 21.4259 * tfactors.T913
                       + -1.34769 * tfactors.T9 + 0.0879816 * tfactors.T953 + -13.1653 * tfactors.lnT9;
 

  dln_set_rate_dT9 =  4.12656 * tfactors.T9i * tfactors.T9i + -(1.0/3.0) * -13.49 * tfactors.T943i + (1.0/3.0) * 21.4259 * tfactors.T923i
                                     + -1.34769 + (5.0/3.0) * 0.0879816 * tfactors.T923 + -13.1653 * tfactors.T9i;

  // avoid underflows by zeroing rates in [0.0, 1.e-100]
  ln_set_rate = Kokkos::fmax(ln_set_rate, -230.0);
  set_rate = std::exp(ln_set_rate);
  rate += set_rate;
  drate_dT += set_rate * dln_set_rate_dT9 * 1.0e-9;
  
  return;
}

KOKKOS_INLINE_FUNCTION
void rhs_nuc(const Real &rho, const Real &temp, const Real X[], Real ydot_nuc[], Real ydot_nuc_dt[]) {

  tf_t tfactors = evaluate_tfactors(temp);

  Real k_C12_He4_to_O16_reaclib = 0.0;
  Real k_C12_He4_to_O16_reaclib_dt = 0.0;

  Real k_He4_He4_He4_to_C12_reaclib = 0.0;
  Real k_He4_He4_He4_to_C12_reaclib_dt = 0.0;

  rate_C12_He4_to_O16_reaclib(tfactors, k_C12_He4_to_O16_reaclib, k_C12_He4_to_O16_reaclib_dt);
  rate_He4_He4_He4_to_C12_reaclib(tfactors, k_He4_He4_He4_to_C12_reaclib, k_He4_He4_He4_to_C12_reaclib_dt);

  Real Y[NSPECIES];
  // Molar abundances from the mass fractions: Y[n] = X[n]/A[n].
  for (int n=0; n<NSPECIES; n++) {
    Y[n] = X[n] * aion_inv(n);
  }

  ydot_nuc[ihe4] =
      -k_C12_He4_to_O16_reaclib*Y[ic12]*Y[ihe4]*rho +
      -0.5*k_He4_He4_He4_to_C12_reaclib*std::pow(Y[ihe4], 3)*std::pow(rho, 2);

  ydot_nuc[ic12] =
      -k_C12_He4_to_O16_reaclib*Y[ic12]*Y[ihe4]*rho +
      0.16666666666666667*k_He4_He4_He4_to_C12_reaclib*std::pow(Y[ihe4], 3)*std::pow(rho, 2);

  ydot_nuc[io16] =
      k_C12_He4_to_O16_reaclib*Y[ic12]*Y[ihe4]*rho;

  ydot_nuc_dt[ihe4] =
      -k_C12_He4_to_O16_reaclib_dt*Y[ic12]*Y[ihe4]*rho +
      -0.5*k_He4_He4_He4_to_C12_reaclib_dt*std::pow(Y[ihe4], 3)*std::pow(rho, 2);

  ydot_nuc_dt[ic12] =
      -k_C12_He4_to_O16_reaclib_dt*Y[ic12]*Y[ihe4]*rho +
      0.16666666666666667*k_He4_He4_He4_to_C12_reaclib_dt*std::pow(Y[ihe4], 3)*std::pow(rho, 2);

  ydot_nuc_dt[io16] =
      k_C12_He4_to_O16_reaclib_dt*Y[ic12]*Y[ihe4]*rho;
}


KOKKOS_INLINE_FUNCTION
Real ener_rate(const Real X[], const Real dedX[], const Real ydot_nuc[]) {
  // Specific heating rate [erg/g/s] at fixed (rho, eps): cv dT/dt = -Sum_n dedX[n] dX_n/dt
  // with dedX = (d eps/dX_n) at fixed (rho, T) from the EOS (binding, ion, electron and
  // Coulomb terms). The k-independent constant in dedX drops out since Sum dX_n/dt = 0.
  Real enuc = 0.0;
  for (int n=0; n<NSPECIES; n++) {
    enuc -= dedX[n] * aion(n) * ydot_nuc[n];
  }
  return enuc;
}


KOKKOS_INLINE_FUNCTION
Real heating_timescale(const Real &rho, const Real &temp, const Real X[],
                       const Real dedX[], const Real &cv) {
  // Nuclear heating timescale cv*T/|enuc| [s]: the time for the burning to raise T by
  // about T. enuc is the specific rate [erg/g/s] and cv the specific heat [erg/g/K].
  // Without burning (enuc = 0) it returns the largest Real, so a min over cells
  // ignores the cell.
  Real ydot_nuc[NSPECIES];
  Real ydot_nuc_dt[NSPECIES];
  rhs_nuc(rho, temp, X, ydot_nuc, ydot_nuc_dt);
  Real abs_enuc = Kokkos::fabs(ener_rate(X, dedX, ydot_nuc));
  return (abs_enuc > 0.0) ? cv*temp/abs_enuc : Kokkos::Experimental::finite_max_v<Real>;
}



KOKKOS_INLINE_FUNCTION
void ActualRHS(Real rho, Real temp, Real X[], const Real dedX[], Real F[],
               Real &enuc_dot) {

  Real ydot_nuc[NSPECIES];
  Real ydot_nuc_dt[NSPECIES];
  rhs_nuc(rho, temp, X, ydot_nuc, ydot_nuc_dt);

  // Nuclear heating rate per unit volume: enuc [erg/g/s] -> rho*enuc [erg/cm^3/s].
  // Diagnostic only, not part of the Newton system: with the Helmholtz EOS, eps includes
  // the binding energy, so burning conserves rho*eps and the release shows up as a
  // higher T at fixed eps.
  enuc_dot = rho * ener_rate(X, dedX, ydot_nuc);

  // UX = rho*X[n] is per unit volume
  for (int n=0; n<NSPECIES; n++) {
    F[n] = rho* aion(n) * ydot_nuc[n];
  }

  return;
}

KOKKOS_INLINE_FUNCTION
void PopulateJac(Real rho, Real temp, Real X[], Real cv, Real ex[], Real J[][NSPECIES]) {

  tf_t tfactors = evaluate_tfactors(temp);

  Real k_C12_He4_to_O16_reaclib = 0.0;
  Real k_C12_He4_to_O16_reaclib_dt = 0.0;

  Real k_He4_He4_He4_to_C12_reaclib = 0.0;
  Real k_He4_He4_He4_to_C12_reaclib_dt = 0.0;

  rate_C12_He4_to_O16_reaclib(tfactors, k_C12_He4_to_O16_reaclib, k_C12_He4_to_O16_reaclib_dt);
  rate_He4_He4_He4_to_C12_reaclib(tfactors, k_He4_He4_He4_to_C12_reaclib, k_He4_He4_He4_to_C12_reaclib_dt);

  Real ydot_nuc[NSPECIES];
  Real ydot_nuc_dt[NSPECIES];
  Real scratch;

  Real Y[NSPECIES];

  for (int n=0; n<NSPECIES; n++) {
    Y[n] = X[n] * aion_inv(n);
  }

  rhs_nuc(rho, temp, X, ydot_nuc, ydot_nuc_dt);

  scratch = -k_C12_He4_to_O16_reaclib*Y[ic12]*rho - 1.5*k_He4_He4_He4_to_C12_reaclib*Kokkos::pow(Y[ihe4], 2)*Kokkos::pow(rho, 2);
  J[ihe4][ihe4] = scratch;

  scratch = -k_C12_He4_to_O16_reaclib*Y[ihe4]*rho;
  J[ihe4][ic12] = scratch;

  scratch = 0.0;
  J[ihe4][io16] = scratch;

  scratch = -k_C12_He4_to_O16_reaclib*Y[ic12]*rho + 0.5*k_He4_He4_He4_to_C12_reaclib*Kokkos::pow(Y[ihe4], 2)*Kokkos::pow(rho, 2);
  J[ic12][ihe4] = scratch;

  scratch = -k_C12_He4_to_O16_reaclib*Y[ihe4]*rho;
  J[ic12][ic12] = scratch;

  scratch = 0.0;
  J[ic12][io16] = scratch;

  scratch = k_C12_He4_to_O16_reaclib*Y[ic12]*rho;
  J[io16][ihe4] = scratch;

  scratch = k_C12_He4_to_O16_reaclib*Y[ihe4]*rho;
  J[io16][ic12] = scratch;

  scratch = 0.0;
  J[io16][io16] = scratch;

  //Add the weights:
  for(int m=0; m<NSPECIES; m++) {
    for(int n=0; n<NSPECIES; n++) {
      J[m][n] *= aion(m) * aion_inv(n);
    }
  }

  //Add the EOS correction: at fixed eps, dT/dX_n = -ex[n]/cv, so the rates' temperature
  //dependence enters every column through dXdot_m/dT = aion(m)*ydot_nuc_dt[m]
  for(int m=0; m<NSPECIES; m++) {
    for(int n=0; n<NSPECIES; n++) {
      J[m][n] -= aion(m)*ydot_nuc_dt[m]*ex[n] / cv;
    }
  }
}

#endif  // NUCLEAR_NETWORK_NETWORK_HPP_
