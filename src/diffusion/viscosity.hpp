#ifndef DIFFUSION_VISCOSITY_HPP_
#define DIFFUSION_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.hpp
//  \brief Contains data and functions that implement various formulations for viscosity.
//  Currently only Navier-Stokes (uniform, isotropic) viscosity implemented
//  TODO: add Braginskii viscosity

#include <map>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

// constants that enumerate Viscosity tasks
enum class ViscosityTaskName {undef=0, hydro_vflux, mhd_vflux};

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief data and functions that implement viscosity in Hydro and MHD

class Viscosity
{
 public:
  Viscosity(MeshBlockPack *pp, ParameterInput *pin);
  ~Viscosity();

  // data
  Real dtnew;
  Real hydro_nu_iso;
  Real mhd_nu_iso;

  // map for associating ViscosityTaskName with TaskID
  std::map<ViscosityTaskName, TaskID> visc_tasks;

  // functions to add viscous fluxes to Hydro and/or MHD fluxes  
  void AssembleStageRunTasks(TaskList &tl, TaskID start);
  TaskStatus AddViscosityHydro(Driver *pdrive, int stage);
  TaskStatus AddViscosityMHD(Driver *pdrive, int stage);
  void AddIsoViscousFlux(const DvceArray5D<Real> &w, DvceFaceFld5D<Real> &f, Real nu);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_VISCOSITY_HPP_
