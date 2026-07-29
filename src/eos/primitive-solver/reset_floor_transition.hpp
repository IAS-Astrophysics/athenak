#ifndef EOS_PRIMITIVE_SOLVER_RESET_FLOOR_TRANSITION_HPP_
#define EOS_PRIMITIVE_SOLVER_RESET_FLOOR_TRANSITION_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reset_floor_transition.hpp
//  \brief Error floor for the two-table transition EOS.
//
//  Like ResetFloor, but aware of the validity edges of the low-density
//  (Helmholtz) and high-density (CompOSE) branches: PrimitiveFloor pushes
//  states out of the invalid wedge between the tables, and floors accumulate
//  instead of early-returning. FailureResponse keeps the (valid) conserved
//  density and species when only the energy/momentum inversion failed.

#include <math.h>

#include <limits>

#include "ps_types.hpp"
#include "error_policy_interface.hpp"
#include "ps_error.hpp"

namespace Primitive {

class ResetFloorTransition : public ErrorPolicyInterface {
 protected:
  /// Constructor
  ResetFloorTransition() {
    fail_conserved_floor = false;
    fail_primitive_floor = false;
    adjust_conserved = true;
    // Inert boundaries until SetTableBoundaries is called.
    ld_n_max = std::numeric_limits<Real>::max();
    hd_n_min = 0.0;
    ld_t_max = std::numeric_limits<Real>::max();
    hd_t_min = 0.0;
  }

  /// Floor for primitive variables. Adds two wedge clauses that push a state
  /// out of the region where neither table is valid (n below the compose edge
  /// while T is above the Helmholtz edge, and vice versa). Floors accumulate.
  KOKKOS_INLINE_FUNCTION bool PrimitiveFloor(Real& n, Real v[3], Real& T, Real *Y,
                                             int n_species) const {
    if (n < n_atm*n_threshold) {
      n = n_atm;
      v[0] = 0.0;
      v[1] = 0.0;
      v[2] = 0.0;
      T = T_atm;
      for (int i = 0; i < n_species; i++) {
        Y[i] = Y_atm[i];
      }
      return true;
    }
    bool adjusted = false;
    if (T < T_atm) {
      T = T_atm;
      adjusted = true;
    }
    if ((T > ld_t_max) && (n < hd_n_min)) {
      n = hd_n_min;
      adjusted = true;
    }
    if ((n > ld_n_max) && (T < hd_t_min)) {
      T = hd_t_min;
      adjusted = true;
    }
    return adjusted;
  }

  /// Floor for conserved variables
  KOKKOS_INLINE_FUNCTION bool ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y,
                                Real D_floor, Real tau_floor, Real tau_abs_floor,
                                int n_species) const {
    if (D < D_floor*n_threshold) {
      D = D_floor;
      Sd[0] = 0.0;
      Sd[1] = 0.0;
      Sd[2] = 0.0;
      tau = tau_abs_floor;
      for (int i = 0; i < n_species; i++) {
        Y[i] = Y_atm[i];
      }
      return true;
    } else if (tau < tau_floor) {
      tau = tau_floor;
      return true;
    }
    return false;
  }

  /// Response to excess magnetization
  KOKKOS_INLINE_FUNCTION Error MagnetizationResponse(Real& bsq, Real b_u[3]) const {
    if (bsq > max_bsq) {
      Real factor = sqrt(max_bsq/bsq);
      bsq = max_bsq;

      b_u[0] /= factor;
      b_u[1] /= factor;
      b_u[2] /= factor;

      return Error::CONS_ADJUSTED;
    }
    return Error::SUCCESS;
  }

  /// Policy for resetting density
  KOKKOS_INLINE_FUNCTION void DensityLimits(Real& n, Real n_min, Real n_max) const {
    n = fmax(n_min, fmin(n_max, n));
  }

  /// Policy for resetting temperature
  KOKKOS_INLINE_FUNCTION void TemperatureLimits(Real& T, Real T_min, Real T_max) const {
    T = fmax(T_min, fmin(T_max, T));
  }

  /// Policy for resetting species fractions
  KOKKOS_INLINE_FUNCTION bool SpeciesLimits(Real* Y, const Real* Y_min, const Real* Y_max,
                                            int n_species) const {
    bool adjusted = false;
    for (int i = 0; i < n_species; i++) {
      if (Y[i] < Y_min[i]) {
        adjusted = true;
        Y[i] = Y_min[i];
      } else if (Y[i] > Y_max[i]) {
        adjusted = true;
        Y[i] = Y_max[i];
      }
    }
    return adjusted;
  }

  /// Policy for resetting pressure
  KOKKOS_INLINE_FUNCTION void PressureLimits(Real& P, Real P_min, Real P_max) const {
    P = fmax(P_min, fmin(P_max, P));
  }

  /// Policy for resetting energy density
  KOKKOS_INLINE_FUNCTION void EnergyLimits(Real& e, Real e_min, Real e_max) const {
    e = fmax(e_min, fmin(e_max, e));
  }

  /// Policy for dealing with failed points. Mass-conserving: keep the seeded
  /// density and species (valid when only the energy/momentum inversion
  /// failed); a non-finite density marks an unusable state -> full reset.
  KOKKOS_INLINE_FUNCTION bool FailureResponse(Real prim[NPRIM]) const {
    if (!isfinite(prim[PRH])) {
      prim[PRH] = n_atm;
      for (int i = 0; i < MAX_SPECIES; i++) {
        prim[PYF + i] = Y_atm[i];
      }
    }
    prim[PVX] = 0.0;
    prim[PVY] = 0.0;
    prim[PVZ] = 0.0;
    prim[PTM] = T_atm;
    return true;
  }

 public:
  /// Set the failure mode for conserved flooring
  KOKKOS_INLINE_FUNCTION void SetConservedFloorFailure(bool failure) {
    fail_conserved_floor = failure;
  }

  /// Set the failure mode for primitive flooring
  KOKKOS_INLINE_FUNCTION void SetPrimitiveFloorFailure(bool failure) {
    fail_primitive_floor = failure;
  }

  /// Set whether or not it's okay to adjust the conserved variables.
  KOKKOS_INLINE_FUNCTION void SetAdjustConserved(bool adjust) {
    adjust_conserved = adjust;
  }

  /// Set the validity edges of the low- and high-density branches.
  KOKKOS_INLINE_FUNCTION void SetTableBoundaries(Real ld_n, Real hd_n,
                                                 Real ld_t, Real hd_t) {
    ld_n_max = ld_n;
    hd_n_min = hd_n;
    ld_t_max = ld_t;
    hd_t_min = hd_t;
  }

 private:
  Real ld_n_max, hd_n_min;
  Real ld_t_max, hd_t_min;
};

}  // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_RESET_FLOOR_TRANSITION_HPP_
