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

  // function pointer for pgen name
  void (ProblemGenerator::*pgen_func_) (MeshBlockPack*, ParameterInput*);

  // predefined problem generator functions
  void ShockTube_(MeshBlockPack *pmbp, ParameterInput *pin);
  void Advection_(MeshBlockPack *pmbp, ParameterInput *pin);
  void LinearWave_(MeshBlockPack *pmbp, ParameterInput *pin);
  void LWImplode_(MeshBlockPack *pmbp, ParameterInput *pin);
  void OrszagTang_(MeshBlockPack *pmbp, ParameterInput *pin);

  // template for user-specified problem generator
  void UserProblem(MeshBlockPack *pmbp, ParameterInput *pin);

 private:
  Mesh* pmesh_;
};

#endif // PGEN_PGEN_HPP_
