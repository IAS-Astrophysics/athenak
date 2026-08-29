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
//! \brief Add tasks for every particle population.

void Particles::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  for (auto *population : populations_) {
    population->AssembleTasks(tl);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void ParticlePopulation::AssembleTasks
//! \brief Adds particle tasks to the appropriate time-integrator task lists.

void ParticlePopulation::AssembleTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  auto add_update_chain = [this](const std::shared_ptr<TaskList> &tasks) {
    TaskID first(0);
    id.push = tasks->AddTask(&ParticlePopulation::Push, this, first);
    id.lifecycle = tasks->AddTask(
        &ParticlePopulation::ApplyUserLifecycle, this, id.push);
    id.purge  = tasks->AddTask(
        &ParticlePopulation::PurgeDeleted, this, id.lifecycle);
    id.newgid = tasks->AddTask(&ParticlePopulation::NewGID, this, id.purge);
    id.count  = tasks->AddTask(&ParticlePopulation::SendCnt, this, id.newgid);
    id.irecv  = tasks->AddTask(&ParticlePopulation::InitRecv, this, id.count);
    id.sendp  = tasks->AddTask(&ParticlePopulation::SendP, this, id.irecv);
    id.recvp  = tasks->AddTask(&ParticlePopulation::RecvP, this, id.sendp);
    id.crecv  = tasks->AddTask(&ParticlePopulation::ClearRecv, this, id.recvp);
    id.csend  = tasks->AddTask(&ParticlePopulation::ClearSend, this, id.crecv);
    return id.csend;
  };

  auto add_post_update = [this](const std::shared_ptr<TaskList> &tasks,
                                TaskID &dependency) {
    id.post_update = tasks->AddTask(
        &ParticlePopulation::ApplyUserPostUpdate, this, dependency);
    return id.post_update;
  };

  switch (pusher) {
    case ParticlesPusher::drift:
      // Drift before the fluid integrator, then refresh its timestep after the fluid.
      {
        TaskID update_done = add_update_chain(tl["before_timeintegrator"]);
        (void)add_post_update(tl["before_timeintegrator"], update_done);
      }
      id.newdt = tl["after_timeintegrator"]->AddTask(
          &ParticlePopulation::NewTimeStep, this, none);
      break;
    case ParticlesPusher::lagrangian_mc:
      {
        // Lagrangian MC consumes the completed fluid step's accumulated mass fluxes.
        TaskID update_done = add_update_chain(tl["after_timeintegrator"]);
        id.finalize = tl["after_timeintegrator"]->AddTask(
            &ParticlePopulation::FinalizeLagrangianMCMove, this, update_done);
        update_done = id.finalize;
        update_done = add_post_update(tl["after_timeintegrator"], update_done);
        id.newdt = tl["after_timeintegrator"]->AddTask(
            &ParticlePopulation::NewTimeStep, this, update_done);
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
//! \fn TaskList ParticlePopulation::NewGID
//! \brief Wrapper task list function to set new GID for particles that move between
//! MeshBlocks.

TaskStatus ParticlePopulation::NewGID(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->SetNewPrtclGID();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList ParticlePopulation::SendCnt
//! \brief Wrapper task list function to set share number of particles communicated with
//! MPI between all ranks

TaskStatus ParticlePopulation::SendCnt(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->CountSendsAndRecvs();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList ParticlePopulation::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI).

TaskStatus ParticlePopulation::InitRecv(Driver *pdrive, int stage) {
  // post receives for particles
  TaskStatus tstat = pbval_part->InitPrtclRecv();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList ParticlePopulation::SendP()
//! \brief Wrapper task list function to pack/send particles

TaskStatus ParticlePopulation::SendP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->PackAndSendPrtcls();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList ParticlePopulation::RecvP
//! \brief Wrapper task list function to receive/unpack particles

TaskStatus ParticlePopulation::RecvP(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_part->RecvAndUnpackPrtcls();
  return tstat;
}


//----------------------------------------------------------------------------------------
//! \fn TaskList ParticlePopulation::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.

TaskStatus ParticlePopulation::ClearSend(Driver *pdrive, int stage) {
  // check sends of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclSend();
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList ParticlePopulation::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed.

TaskStatus ParticlePopulation::ClearRecv(Driver *pdrive, int stage) {
  // check receives of particles complete
  TaskStatus tstat = pbval_part->ClearPrtclRecv();
  return tstat;
}

} // namespace particles
