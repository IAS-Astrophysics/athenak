#ifndef NUCLEAR_NETWORK_NETWORK_HPP_
#define NUCLEAR_NETWORK_NETWORK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file network.hpp
//! \brief He4/C12/O16 reaction network: species data, REACLIB rates, the RHS and the
//! Jacobian of the conserved (rho*X[n], rho*e) system.

#include <cmath>
#include <string>
#include <vector>

#include "athena.hpp"
#include "units/units.hpp"

// IE (not lowercase `ie`): the energy-slot index into NX-sized arrays
// (UX[], F[], J[][]). Deliberately distinct from the common AthenaK
// mesh-bound-index convention `is, ie, js, je, ks, ke`, which this enum
// would otherwise silently shadow wherever both appear in the same scope.
enum {
 ihe4=0,
 ic12=1,
 io16=2,
 IE=3,
};

constexpr int NSPECIES = IE;
constexpr int NX = IE + 1;


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
void ener_rate(const Real &rho, const Real &temp, const Real X[], Real &enuc, Real &enuc_dt) {
  // Specific energy rate contribution and its temperature derivative.

  Real ydot_nuc[NSPECIES];
  Real ydot_nuc_dt[NSPECIES];

  rhs_nuc(rho, temp, X, ydot_nuc, ydot_nuc_dt);
  enuc = 0.0;
  enuc_dt = 0.0;

  for (int n=0; n<NSPECIES; n++) {
    enuc += ydot_nuc[n] * mion(n);
  }

  for (int n=0; n<NSPECIES; n++) {
    enuc_dt += ydot_nuc_dt[n] * mion(n);
  }

  Real m_u = units::Units::atomic_mass_unit_cgs;
  Real c_cgs = units::Units::speed_of_light_cgs;
  enuc *= -c_cgs*c_cgs/m_u;
  enuc_dt *= -c_cgs*c_cgs/m_u;
  return;
}



KOKKOS_INLINE_FUNCTION
void ActualRHS(Real rho, Real temp, Real X[], Real F[]) {
  
  Real enuc;
  Real enuc_dt;
  ener_rate(rho, temp, X, enuc, enuc_dt);
  Real ydot_nuc[NSPECIES];
  Real ydot_nuc_dt[NSPECIES];
  rhs_nuc(rho, temp, X, ydot_nuc, ydot_nuc_dt);

  // UX = (rho*X[n], rho*e) are per unit volume: enuc [erg/g/s] -> rho*enuc [erg/cm^3/s]
  F[IE] = rho * enuc;
  for (int n=0; n<NSPECIES; n++) {
    F[n] = rho* aion(n) * ydot_nuc[n];
  }

  return;
}

KOKKOS_INLINE_FUNCTION
void PopulateJac(Real rho, Real temp, Real X[], Real cv, Real ex[], Real J[][NX]) {
 
  Real m_u = units::Units::atomic_mass_unit_cgs;

  tf_t tfactors = evaluate_tfactors(temp);

  Real k_C12_He4_to_O16_reaclib = 0.0;
  Real k_C12_He4_to_O16_reaclib_dt = 0.0;

  Real k_He4_He4_He4_to_C12_reaclib = 0.0;
  Real k_He4_He4_He4_to_C12_reaclib_dt = 0.0;

  rate_C12_He4_to_O16_reaclib(tfactors, k_C12_He4_to_O16_reaclib, k_C12_He4_to_O16_reaclib_dt);
  rate_He4_He4_He4_to_C12_reaclib(tfactors, k_He4_He4_He4_to_C12_reaclib, k_He4_He4_He4_to_C12_reaclib_dt);

  Real ydot_nuc[NSPECIES];
  Real ydot_nuc_dt[NSPECIES];
  Real enuc;
  Real enuc_dt;
  Real scratch;

  Real Y[NSPECIES];
  
  for (int n=0; n<NSPECIES; n++) {
    Y[n] = X[n] * aion_inv(n); 
  }

  ener_rate(rho, temp, X, enuc, enuc_dt);
  rhs_nuc(rho, temp, X, ydot_nuc, ydot_nuc_dt);

  scratch = -k_C12_He4_to_O16_reaclib*Y[ic12]*rho - 0.5*k_He4_He4_He4_to_C12_reaclib*Kokkos::pow(Y[ihe4], 2)*Kokkos::pow(rho, 2);
  J[ihe4][ihe4] = scratch;

  scratch = -k_C12_He4_to_O16_reaclib*Y[ihe4]*rho;
  J[ihe4][ic12] = scratch;

  scratch = 0.0;
  J[ihe4][io16] = scratch;

  scratch = ydot_nuc_dt[ihe4] / cv;
  J[ihe4][IE] = scratch;

  scratch = -k_C12_He4_to_O16_reaclib*Y[ic12]*rho + 0.5*k_He4_He4_He4_to_C12_reaclib*Kokkos::pow(Y[ihe4], 2)*Kokkos::pow(rho, 2);
  J[ic12][ihe4] = scratch;

  scratch = -k_C12_He4_to_O16_reaclib*Y[ihe4]*rho;
  J[ic12][ic12] = scratch;

  scratch = 0.0;
  J[ic12][io16] = scratch;

  scratch = ydot_nuc_dt[ic12] / cv;
  J[ic12][IE] = scratch;

  scratch = k_C12_He4_to_O16_reaclib*Y[ic12]*rho;
  J[io16][ihe4] = scratch;

  scratch = k_C12_He4_to_O16_reaclib*Y[ihe4]*rho;
  J[io16][ic12] = aion(io16)*aion_inv(ic12)*scratch;

  scratch = 0.0;
  J[io16][io16] = scratch;

  scratch = ydot_nuc_dt[io16] / cv;
  J[io16][IE] = scratch;


  for(int m=0; m<NSPECIES; m++) {
    Real sum_enuc_dY = 0.0;
    for(int n=0; n<NSPECIES; n++) {
      sum_enuc_dY += -mion(n)*J[m][n] / m_u;
    }
    J[IE][m] = sum_enuc_dY;
  }

  J[IE][IE] = enuc_dt / cv;

  //Add the weights:
  for(int m=0; m<NSPECIES; m++) {
    for(int n=0; n<NSPECIES; n++) {
      J[m][n] *= aion(m) * aion_inv(n);
    }
  }

  for(int m=0; m<NSPECIES; m++) {
    J[m][IE] *= aion(m);
  }

  for(int n=0; n<NSPECIES; n++) {
    J[IE][n] *= aion_inv(n);
  }

  //Add the EOS correction
  for(int m=0; m<NSPECIES; m++) {
    for(int n=0; n<NSPECIES; n++) {
      J[m][n] -= aion(m)*ex[n]*J[m][IE] / cv;
    }
  }

  for(int n=0; n<NSPECIES; n++) {
    J[IE][n] -= ex[n] * enuc_dt / cv;
  }
}

#endif  // NUCLEAR_NETWORK_NETWORK_HPP_
