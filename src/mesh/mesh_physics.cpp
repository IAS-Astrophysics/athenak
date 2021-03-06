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
#include "srcterms/srcterms.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddPhysicsModules()
// \brief construct physics modules and tasks lists in this MeshBlockPack, based on which
// <blocks> are present in the input file.  Called from main().

void MeshBlockPack::AddPhysicsModules(ParameterInput *pin)
{
  int nphysics = 0;
  TaskID none(0);

  // (1) HYDRODYNAMICS
  // Create both Hydro physics module and Tasks (TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);   // construct new Hydro object
    nphysics++;
    phydro->AssembleStageStartTasks(stage_start_tl, none);
    phydro->AssembleStageRunTasks(stage_run_tl, none);
    phydro->AssembleStageEndTasks(stage_end_tl, none);
  } else {
    phydro = nullptr;
  }

  // (2) MHD
  // Create both MHD physics module and Tasks (TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);   // construct new MHD object
    nphysics++;

    if (stage_start_tl.Empty()) {
      pmhd->AssembleStageStartTasks(stage_start_tl, none);
    } else {
      TaskID last = stage_start_tl.GetIDLastTask();
      pmhd->AssembleStageStartTasks(stage_start_tl, last);
    }

    if (stage_run_tl.Empty()) {
      pmhd->AssembleStageRunTasks(stage_run_tl, none);
    } else {
      TaskID last = stage_run_tl.GetIDLastTask();
      pmhd->AssembleStageRunTasks(stage_run_tl, last);
    }

    if (stage_end_tl.Empty()) {
      pmhd->AssembleStageEndTasks(stage_end_tl, none);
    } else {
      TaskID last = stage_end_tl.GetIDLastTask();
      pmhd->AssembleStageEndTasks(stage_end_tl, last);
    }
  } else {
    pmhd = nullptr;
  }

  // (3) VISCOSITY
  if (pin->DoesBlockExist("viscosity")) {
    pvisc = new Viscosity(this, pin);
    pvisc->AssembleStageRunTasks(stage_run_tl, none);
  } else {
    pvisc = nullptr;
  }

  // (4) RESISTIVITY
  if (pin->DoesBlockExist("resistivity")) {
    presist = new Resistivity(this, pin);
    presist->AssembleStageRunTasks(stage_run_tl, none);
  } else {
    presist = nullptr;
  }

  // (5) SOURCE TERMS
  // Source terms can be specified by many different <blocks> in the input file.  Thus,
  // SourceTerms constructor parses input file to see if any source terms are enrolled.
  // If not, functions below do not do anything
  psrc = new SourceTerms(this, pin);
  psrc->IncludeSplitSrcTermTasks(operator_split_tl, none);
  psrc->IncludeUnsplitSrcTermTasks(stage_run_tl, none);

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
  }
  // MHD timestep
  if (pmb_pack->pmhd != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->dtnew) );
  }
  // viscosity timestep
  if (pmb_pack->pvisc != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pvisc->dtnew) );
  }
  // resistivity timestep
  if (pmb_pack->presist != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->presist->dtnew) );
  }

#if MPI_PARALLEL_ENABLED
  // get minimum dt over all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}
