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


using namespace hydro;
//----------------------------------------------------------------------------------------
// \fn Mesh::SelectPhysics()

void MeshBlock::SelectPhysics(ParameterInput *pin)
{
  // parse input blocks to see which physics defined
  bool hydro_defined = pin->DoesBlockExist("hydro");

  // construct physics modules and tasks lists on this MeshBlock
  // TODO: add multiple physics, store in std::vector of pointers?
  // TODO: add element of std::vector in BoundaryValues for send/recv buffers each physics
  
  // physics modules
  if (hydro_defined) {
    phydro = new hydro::Hydro(pmesh_, pin, mb_gid);
  } else {
    phydro = nullptr;
    std::cout << "Hydro block not found in input file" << std::endl;
  }

  // allocate memory for boundary buffers for each physics
  pbvals->AllocateBuffers(phydro->nhydro);

  // build task lists
  TaskID none(0);
  std::vector<TaskID> hydro_tasks;

  // add hydro tasks
  if (phydro != nullptr) phydro->HydroAddTasks(tl_onestage, none, hydro_tasks);

  // add physical boundary conditions, and make depend on hydro_recv (penultimate task)
  TaskID hydro_recv = hydro_tasks[hydro_tasks.size()-2];
  auto bvals_physical =
    tl_onestage.InsertTask(&BoundaryValues::ApplyPhysicalBCs, pbvals, hydro_recv);

//  auto bvals_physical =
//    tl_onestage.AddTask(&BoundaryValues::ApplyPhysicalBCs, pbvals, hydro_tasks.back());

  return;
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim) {

  // limit increase in timestep to 2x old value
  dt = 2.0*dt;

  // cycle over all MeshBlocks on this rank and find minimum dt
  for (const auto &mb : mblocks) { dt = std::min(dt, (cfl_no)*(mb.phydro->dtnew) ); }

  // TODO: get minimum dt over all MPI ranks

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}

  return;
}
