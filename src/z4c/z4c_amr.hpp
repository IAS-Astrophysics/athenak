//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_Z4C_AMR_HPP_
#define Z4C_Z4C_AMR_HPP_

#include <string>
#include <vector>

#include "athena.hpp"

class ParameterInput;
class MeshBlockPack;

namespace z4c {
class Z4c;

//! \class Z4c_AMR
//  \brief managing AMR for Z4c simulations
class Z4c_AMR {
  enum RefinementMethod { Trivial, Tracker, Chi, dChi };

 public:
  explicit Z4c_AMR(ParameterInput *pin);
  ~Z4c_AMR() noexcept = default;

  void Refine(MeshBlockPack *pmbp);             // call the AMR method
  void RefineTracker(MeshBlockPack *pmbp);      // Refine based on the trackers
  void RefineChiMin(MeshBlockPack *pmbp);       // Refine based on min{chi}
  void RefineDchiMax(MeshBlockPack *pmbp);      // Refine based on max{dchi}
  void RefineRadii(MeshBlockPack *pmbp);        // Refine based on the radii

  RefinementMethod method;

  // Optinally set the minimum refinement level inside different radial shells
  std::vector<Real> radius;
  std::vector<int> reflevel;

  Real chi_thresh;     // chi threshold for chi refinement method
  Real dchi_thresh;    // dchi threshold for dchi refinement method
};

} // namespace z4c
#endif // Z4C_Z4C_AMR_HPP_
