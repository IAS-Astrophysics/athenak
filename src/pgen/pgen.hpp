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

using UserBoundaryFnPtr = void (*)(Mesh* pm);
using UserErrorFnPtr = void (*)(MeshBlockPack *pmbp, ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \class ProblemGenerator

class ProblemGenerator {
 public:
  // constructor for new problems
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  // constructor for restarts
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, IOWrapper resfile);
  ~ProblemGenerator() = default;

  // data
  bool user_bcs;

  // function pointers for pgen errors / user enrolled BCs
  UserErrorFnPtr pgen_error_func=nullptr;
  UserBoundaryFnPtr user_bcs_func=nullptr;

  // predefined problem generator functions
  void Advection(MeshBlockPack *pmbp, ParameterInput *pin);
  void LinearWave(MeshBlockPack *pmbp, ParameterInput *pin);
  void LWImplode(MeshBlockPack *pmbp, ParameterInput *pin);
  void OrszagTang(MeshBlockPack *pmbp, ParameterInput *pin);
  void ShockTube(MeshBlockPack *pmbp, ParameterInput *pin);

  // function called after main loop contianing any final problem-specific work
  // error functions in predefine problem generator
  void ProblemGeneratorFinalize(ParameterInput *pin, Mesh *pmesh);
  void LinearWaveErrors_(MeshBlockPack *pmbp, ParameterInput *pin);

  // template for user-specified problem generator
  void UserProblem(MeshBlockPack *pmbp, ParameterInput *pin);

  // function and function pointer for user-defined boundary conditions
  void EnrollBoundaryFunction(UserBoundaryFnPtr my_bcfunc);

 private:
  Mesh* pmy_mesh_;
};

#endif // PGEN_PGEN_HPP_
