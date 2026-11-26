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
  void RefineTracker(MeshBlockPack *pmbp);      // Refine based on trackers
  void RefineChiMin(MeshBlockPack *pmbp);       // Refine based on min{chi}
  void RefineDchiMax(MeshBlockPack *pmbp);      // Refine based on max{dchi}
  void RefineRadii(MeshBlockPack *pmbp);        // Refine based on the radii
  void RefineDisk(MeshBlockPack *pmbp);         // Refine a cylinder around a tracker

  RefinementMethod method;

  // Optinally set the minimum refinement level inside different radial shells
  std::vector<Real> radius;
  std::vector<int> reflevel;

  // Optionally set the minimum refinement level inside a cylinder (R < disk_rmax,
  // |z| < disk_zmax) centred on compact object tracker disk_tracker (-1: origin)
  int disk_reflevel     = -1;  // finest level of cylinder
  int disk_tracker      = 0;   // around which tracker to refine
  Real disk_rmax        = 0.0;
  Real disk_zmax        = 0.0;
  bool disk_auto        = false; // automatic disk refinement
  Real disk_merger_sep  = 1.0;   // at what tracker separation to start
  Real disk_delay       = 0.0;   // delay after merger sep reached
  Real disk_merger_time = -1.0;  // record disk merger time
  ParameterInput *pin_  = nullptr; // to record disk merger time for restarts
  bool DiskActive(MeshBlockPack *pmbp);

  Real chi_thresh;     // chi threshold for chi refinement method
  Real dchi_thresh;    // dchi threshold for dchi refinement method
};

} // namespace z4c
#endif // Z4C_Z4C_AMR_HPP_
