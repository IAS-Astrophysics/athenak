#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.hpp
//  \brief definitions for ProblemGenerator class

#include <functional>
#include "parameter_input.hpp"

class ProblemGenerator {
 public:
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  ~ProblemGenerator() = default;

  // data


 private:
  Mesh* pmesh_;

  // function pointer for pgen name
  void (ProblemGenerator::*pgen_func_) (MeshBlock*, ParameterInput*);

  // predefined problem generator functions
  void ShockTube_(MeshBlock *pmb, ParameterInput *pin);
  void Advection_(MeshBlock *pmb, ParameterInput *pin);
  void LWImplode_(MeshBlock *pmb, ParameterInput *pin);

};

#endif // PGEN_PGEN_HPP_
