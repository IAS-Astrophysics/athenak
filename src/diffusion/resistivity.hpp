#ifndef DIFFUSION_RESISTIVITY_HPP_
#define DIFFUSION_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.hpp
//  \brief Contains data and functions that implement various non-ideal MHD (resistive) 
//  processes, such as Ohmic diffusion.
//  TODO: add ambipolar diffusion, Hall effect

#include <map>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

// constants that enumerate Resistivity tasks
enum class ResistivityTaskName {undef=0, ohmic_emf};

//----------------------------------------------------------------------------------------
//! \class Resistivity
//  \brief data and functions that implement various resistive physics

class Resistivity
{
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  ~Resistivity();

  // data
  Real dtnew;
  Real eta_ohm;

  // map for associating ResistivityTaskName with TaskID
  std::map<ResistivityTaskName, TaskID> resist_tasks;

  // functions to add resistive EMF to MHD, and energy flux to Hydro
  void AssembleStageRunTasks(TaskList &tl, TaskID start);
  TaskStatus AddResistiveEMFs(Driver *pdrive, int stage);
  void AddOhmicEMF(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
