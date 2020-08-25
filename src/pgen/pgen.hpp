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
#include "athena_arrays.hpp"
#include "parameter_input.hpp"

class ProblemGenerator {
 public:
  ProblemGenerator(std::unique_ptr<ParameterInput> &pin, Mesh *pmesh);
  ~ProblemGenerator() = default;

  // data


 private:
  Mesh* pmesh_;

  // function pointer for pgen name
  void (ProblemGenerator::*pgen_func_) (MeshBlock*, std::unique_ptr<ParameterInput>&);

  // predefined problem generator functions
  void ShockTube_(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin);
  void Advection_(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin);

};

#endif // PGEN_PGEN_HPP_
