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

namespace hydro {

enum ConsIndex {IDN=0, IM1=1, IM2=2, IM3=3, IEN=4};
enum PrimIndex {IVX=1, IVY=2, IVZ=3, IPR=4};

class Hydro {
 public:
  Hydro(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin);
  ~Hydro();

  // data
  MeshBlock* pmy_mblock;    // ptr to MeshBlock containing this Hydro

  AthenaCenterArray<Real> u, w;    // conserved and primitive variables

 private:

};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
