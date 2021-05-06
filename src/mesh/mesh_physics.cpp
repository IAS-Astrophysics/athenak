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
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "srcterms/turb_driver.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddPhysicsModules()
// \brief construct physics modules and tasks lists in this MeshBlockPack, based on which
// <blocks> are present in the input file.  Called from main().

void MeshBlockPack::AddPhysicsModules(ParameterInput *pin, Driver *pdrive)
{
  int nphysics = 0;
  TaskID none(0);

  // (1) HYDRODYNAMICS
  // Create both Hydro physics module and Tasks (TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);   // construct new Hydro object
    phydro->AssembleHydroTasks(start_tl, run_tl, end_tl);
    nphysics++;
  } else {
    phydro = nullptr;
  }

  // (2) MHD
  // Create both MHD physics module and Tasks (TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);   // construct new MHD object
    pmhd->AssembleMHDTasks(start_tl, run_tl, end_tl);
    nphysics++;
  } else {
    pmhd = nullptr;
  }

  // (3) TURBULENCE DRIVER
  // This is a special module to drive turbulence in hydro, MHD, or both. Cannot be
  // included as a source term since it requires evolving force array via O-U process.
  // Instead, TurbulenceDriver object is stored in MeshBlockPack and tasks for evolving
  // force and adding force to fluid are included in operator_split and stage_run
  // task lists respectively.
  if (pin->DoesBlockExist("turb_driving")) {
    pturb = new TurbulenceDriver(this, pin);
    pturb->IncludeInitializeModesTask(operator_split_tl, none);
    pturb->IncludeAddForcingTask(run_tl, none);
  } else {
    pturb = nullptr;
  }

  // Check that at least ONE is requested and initialized.
  // Error if there are no physics blocks in the input file.
  if (nphysics == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "At least one physics module must be specified in input file." << std::endl;
    std::exit(EXIT_FAILURE);
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

  // Hydro timestep
  if (pmb_pack->phydro != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->dtnew) );
    // viscosity timestep
    if (pmb_pack->phydro->pvisc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->pvisc->dtnew) );
    }
  }
  // MHD timestep
  if (pmb_pack->pmhd != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->dtnew) );
    // viscosity timestep
    if (pmb_pack->pmhd->pvisc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->pvisc->dtnew) );
    }
    // resistivity timestep
    if (pmb_pack->pmhd->presist != nullptr) {
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
