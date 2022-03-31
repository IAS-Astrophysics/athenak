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

using ProblemFinalizeFnPtr = void (*)(ParameterInput *pin, Mesh *pm);
using UserBoundaryFnPtr = void (*)(Mesh* pm);

//----------------------------------------------------------------------------------------
//! \class ProblemGenerator

class ProblemGenerator {
 public:
  // constructor for new problems
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  // constructor for restarts
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, IOWrapper resfile);
  ~ProblemGenerator() = default;

  // true if user BCs are specified on any face
  bool user_bcs;

  // function pointer for final work after main loop (e.g. compute errors).  Called by
  // Driver::Finalize()
  ProblemFinalizeFnPtr pgen_final_func=nullptr;
  // function pointer for user-enrolled BCs.  Called in ApplyPhysicalBCs in task list
  UserBoundaryFnPtr user_bcs_func=nullptr;

  // predefined problem generator functions (default test suite)
  void Advection(ParameterInput *pin, const bool restart);
  void BondiAccretion(ParameterInput *pin, const bool restart);
  void LinearWave(ParameterInput *pin, const bool restart);
  void LWImplode(ParameterInput *pin, const bool restart);
  void OrszagTang(ParameterInput *pin, const bool restart);
  void ShockTube(ParameterInput *pin, const bool restart);

  // template for user-specified problem generator
  void UserProblem(ParameterInput *pin, const bool restart);

 private:
  Mesh* pmy_mesh_;
};

#endif // PGEN_PGEN_HPP_
