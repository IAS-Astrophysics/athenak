#ifndef EOS_PRIMITIVE_SOLVER_ERROR_POLICY_INTERFACE_HPP_
#define EOS_PRIMITIVE_SOLVER_ERROR_POLICY_INTERFACE_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file error_policy_interface.hpp
//  \brief Defines a class that provides all the basic members
//         needed by an ErrorPolicy.
//
//  It cannot be instantiated and, in fact, has no purpose in
//  being instantiated. It literally just provides member
//  variables for an ErrorPolicy;

#include "ps_types.hpp"

namespace Primitive {

class ErrorPolicyInterface {
 protected:
  ErrorPolicyInterface() = default;
  ~ErrorPolicyInterface() = default;

  Real n_atm;
  Real n_threshold;
  Real T_atm;
  Real Y_atm[MAX_SPECIES];
  Real v_max;
  Real max_bsq;
  bool fail_conserved_floor;
  bool fail_primitive_floor;
  bool adjust_conserved;
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_ERROR_POLICY_INTERFACE_HPP_
