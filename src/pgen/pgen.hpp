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
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, Driver *pd);
  ~ProblemGenerator() = default;

  // data

  // function pointer for pgen name
  void (ProblemGenerator::*pgen_func_) (MeshBlockPack*, ParameterInput*);

  // predefined problem generator functions
  void Advection_(MeshBlockPack *pmbp, ParameterInput *pin);
  void LinearWave_(MeshBlockPack *pmbp, ParameterInput *pin);
  void ShockTube_(MeshBlockPack *pmbp, ParameterInput *pin);
  void LWImplode_(MeshBlockPack *pmbp, ParameterInput *pin);
  void OrszagTang_(MeshBlockPack *pmbp, ParameterInput *pin);
  void BondiAccretion_(MeshBlockPack *pmbp, ParameterInput *pin);

  // function called after main loop contianing any final problem-specific work
  // error functions in predefine problem generator
  void ProblemGeneratorFinalize(ParameterInput *pin, Mesh *pmesh);
  void LinearWaveErrors_(MeshBlockPack *pmbp, ParameterInput *pin);
  void BondiErrors_(MeshBlockPack *pmbp, ParameterInput *pin);

  // template for user-specified problem generator
  void UserProblem(MeshBlockPack *pmbp, ParameterInput *pin);

 private:
  Mesh* pmy_mesh_;
  Driver *pmy_driver_;
};

#endif // PGEN_PGEN_HPP_
