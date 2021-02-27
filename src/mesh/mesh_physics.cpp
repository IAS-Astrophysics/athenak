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
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddPhysicsModules()
// \brief construct physics modules and tasks lists in this MeshBlockPack, based on which
// <blocks> are present in the input file.  Called from main().

void MeshBlockPack::AddPhysicsModules(ParameterInput *pin)
{
  // Cycle through available physics modules, and construct those requested in input file.
  // Also check that at least ONE is requested and initialized.

  int nphysics = 0;
  // Create Hydro physics module if <hydro> block exists in input file
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);   // construct new Hydro object
    nphysics++;
  } else {
    phydro = nullptr;
  }

  // Create MHD physics module if <mhd> block exists in input file
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);   // construct new MHD object
    nphysics++;
  } else {
    pmhd = nullptr;
  }

  // Error if there are no physics blocks in the input file.
  if (nphysics == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "At least one physics module must be specified in input file." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Create TaskLists for Start, Run, and End of each stage of integrator
  // add Hydro tasks
  TaskID none(0);
  if (phydro != nullptr) {
    phydro->HydroStageStartTasks(tl_stagestart, none);
    phydro->HydroStageRunTasks(tl_stagerun, none);
    phydro->HydroStageEndTasks(tl_stageend, none);
  }

  // add MHD tasks to end of Hydro tasks (if any have been added)
  if (pmhd != nullptr) {
    if (tl_stagestart.Empty()) {
      pmhd->MHDStageStartTasks(tl_stagestart, none);
    } else {
      TaskID last = tl_stagestart.GetIDLastTask();
      pmhd->MHDStageStartTasks(tl_stagestart, last);
    }

    if (tl_stagerun.Empty()) {
      pmhd->MHDStageRunTasks(tl_stagerun, none);
    } else {
      TaskID last = tl_stagerun.GetIDLastTask();
      pmhd->MHDStageRunTasks(tl_stagerun, last);
    }

    if (tl_stageend.Empty()) {
      pmhd->MHDStageEndTasks(tl_stageend, none);
    } else {
      TaskID last = tl_stageend.GetIDLastTask();
      pmhd->MHDStageEndTasks(tl_stageend, last);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim)
{
  // cycle over all MeshBlocks on this rank and find minimum dt
  // Requires at least ONE of the physics modules to be defined.
  // limit increase in timestep to 2x old value
  dt = 2.0*dt;
  if (pmb_pack->phydro != nullptr) {
    // Hydro timestep
    dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->dtnew) );
    if (pmb_pack->phydro->pvisc != nullptr) {
      // Hydro viscosity timestep
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->pvisc->dtnew) );
    }
  }
  if (pmb_pack->pmhd != nullptr) {
    // MHD timestep
    dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->dtnew) );
    if (pmb_pack->pmhd->pvisc != nullptr) {
      // MHD viscosity timestep
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->pvisc->dtnew) );
    }
    if (pmb_pack->pmhd->presist != nullptr) {
      // MHD resistivity timestep
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->presist->dtnew) );
    }
  }

#if MPI_PARALLEL_ENABLED
  // get minimum dt over all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}
