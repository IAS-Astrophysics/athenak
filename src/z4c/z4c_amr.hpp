//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_Z4C_AMR_HPP_
#define Z4C_Z4C_AMR_HPP_

#include <string>

#include "athena.hpp"

class ParameterInput;
class MeshBlockPack;

namespace z4c {
class Z4c;

//! \class Z4c_AMR
//  \brief managing AMR for Z4c simulations
class Z4c_AMR {
 public:
  Z4c_AMR(Z4c *z4c, ParameterInput *pin);
  ~Z4c_AMR() noexcept = default;
  void Refine(MeshBlockPack *pmbp); // call the AMR method
 private:
  Z4c *pz4c;              // ptr to z4c
  ParameterInput *pin;    // ptr to parameter
  std::string ref_method; // method of refinement
  Real x1max, x1min;      // full grid extent
  Real half_initial_d; // half of the initial separation,e.g.,two puncture par_b
  Real chi_thresh;     // chi threshold for chi refinement method
 public:
  void LinfBoxInBox(MeshBlockPack *pmbp);     // Linf box in box method
  void L2SphereInSphere(MeshBlockPack *pmbp); // L2 Sphere in Sphere method
  void ChiMin(MeshBlockPack *pmbp);           // Refine based on min{chi}
};

} // namespace z4c
#endif // Z4C_Z4C_AMR_HPP_
