//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file interface_physics.cpp
//  \brief 

#include <iostream>

#include "parameter_input.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// \fn Mesh::SelectPhysics()

void MeshBlock::InitPhysicsModules(ParameterInput *pin)
{
  // construct physics modules and tasks lists on this MeshBlock
  // physics modules

  // Hydro physics module
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(pmesh_, pin, mb_gid); // construct new Hydro object
    pbvals->bbuf_ptr["hydro"] = &(phydro->bbuf);    // add pointer to Hydro bbufs in map
  } else {
    phydro = nullptr;
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

  // add physical boundary conditions, and make depend on hydro_recv (penultimate task)
  TaskID hydro_recv = hydro_run_tasks[hydro_run_tasks.size()-2];
  auto bvals_physical =
    tl_stagerun.InsertTask(&BoundaryValues::ApplyPhysicalBCs, pbvals, hydro_recv);

//  auto bvals_physical =
//    tl_onestage.AddTask(&BoundaryValues::ApplyPhysicalBCs, pbvals, hydro_tasks.back());

  return;
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim)
{
  // cycle over all MeshBlocks on this rank and find minimum dt
  // limit increase in timestep to 2x old value
  for (const auto &mb : mblocks) { dt = std::min(2.0*dt, (cfl_no)*(mb.phydro->dtnew) ); }

  // get minimum dt over all MPI ranks
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}
