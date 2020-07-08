#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include "athena_arrays.hpp"
#include "parameter_input.hpp"
//#include "mesh/mesh.hpp"

class Hydro {
 public:
  Hydro(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin);
  ~Hydro();

  // data
  MeshBlock* pmy_mblock;    // ptr to MeshBlock containing this Hydro

  AthenaCenterArray<Real> u, w;    // conserved and primitive variables

 private:

};
#endif // HYDRO_HYDRO_HPP_
