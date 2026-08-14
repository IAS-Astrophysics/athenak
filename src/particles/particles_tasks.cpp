//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_tasks.cpp
//! \brief functions that control Particles tasks stored in tasklists in MeshBlockPack

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::AssembleTasks
//! \brief Adds particle tasks to the appropriate time-integrator task lists.

void Particles::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  auto add_update_chain = [this](const std::shared_ptr<TaskList> &tasks) {
    TaskID first(0);
    id.push   = tasks->AddTask(&Particles::Push, this, first);
    id.purge  = tasks->AddTask(&Particles::PurgeDeleted, this, id.push);
    id.newgid = tasks->AddTask(&Particles::NewGID, this, id.purge);
    id.count  = tasks->AddTask(&Particles::SendCnt, this, id.newgid);
    id.irecv  = tasks->AddTask(&Particles::InitRecv, this, id.count);
    id.sendp  = tasks->AddTask(&Particles::SendP, this, id.irecv);
    id.recvp  = tasks->AddTask(&Particles::RecvP, this, id.sendp);
    id.crecv  = tasks->AddTask(&Particles::ClearRecv, this, id.recvp);
    id.csend  = tasks->AddTask(&Particles::ClearSend, this, id.crecv);
    return id.csend;
  };

  switch (pusher) {
    case ParticlesPusher::drift:
      // Drift before the fluid integrator, then refresh its timestep after the fluid.
      (void)add_update_chain(tl["before_timeintegrator"]);
      id.newdt = tl["after_timeintegrator"]->AddTask(
          &Particles::NewTimeStep, this, none);
      break;
    case ParticlesPusher::lagrangian_mc:
      {
        // Lagrangian MC consumes the completed fluid step's accumulated mass fluxes.
        TaskID update_done = add_update_chain(tl["after_timeintegrator"]);
        id.newdt = tl["after_timeintegrator"]->AddTask(
            &Particles::NewTimeStep, this, update_done);
        break;
      }
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher has no task schedule" << std::endl;
      std::exit(EXIT_FAILURE);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::NewGID
//! \brief Wrapper task list function to set new GID for particles that move between
//! MeshBlocks.

TaskStatus Particles::NewGID(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->SetNewPrtclGID();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SendCnt
//! \brief Wrapper task list function to set share number of particles communicated with
//! MPI between all ranks

TaskStatus Particles::SendCnt(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->CountSendsAndRecvs();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI).

TaskStatus Particles::InitRecv(Driver *pdrive, int stage) {
  // post receives for particles
  TaskStatus tstat = pbval_part->InitPrtclRecv();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::SendP()
//! \brief Wrapper task list function to pack/send particles

TaskStatus Particles::SendP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->PackAndSendPrtcls();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::RecvP
//! \brief Wrapper task list function to receive/unpack particles

TaskStatus Particles::RecvP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->RecvAndUnpackPrtcls();
  return tstat;
}


//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.

TaskStatus Particles::ClearSend(Driver *pdrive, int stage) {
  // check sends of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclSend();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Particles::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed.

TaskStatus Particles::ClearRecv(Driver *pdrive, int stage) {
  // check receives of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclRecv();
  return tstat;
}

} // namespace particles
