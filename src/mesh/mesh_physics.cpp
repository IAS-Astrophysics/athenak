//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_physics.cpp
//  \brief 

#include <iostream>

#include "parameter_input.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// \fn Mesh::SelectPhysics()

void MeshBlockPack::AddPhysicsModules(ParameterInput *pin)
{
  // construct physics modules and tasks lists in this MeshBlockPack
  // physics modules

  // Hydro physics module
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);   // construct new Hydro object
  } else {
    phydro = nullptr;
  }

  // MHD physics module
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);   // construct new MHD object
  } else {
    pmhd = nullptr;
  }

  // build task lists
  TaskID none(0);
  std::vector<TaskID> hydro_start_tasks, hydro_run_tasks, hydro_end_tasks;

  // add Hydro tasks
  if (phydro != nullptr) {
    phydro->HydroStageStartTasks(tl_stagestart, none, hydro_start_tasks);
    phydro->HydroStageRunTasks(tl_stagerun, none, hydro_run_tasks);
    phydro->HydroStageEndTasks(tl_stageend, none, hydro_end_tasks);
  }

  return;
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim)
{
  // cycle over all MeshBlocks on this rank and find minimum dt
  // limit increase in timestep to 2x old value
  dt = 2.0*dt;
  dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->dtnew) );

#if MPI_PARALLEL_ENABLED
  // get minimum dt over all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}
