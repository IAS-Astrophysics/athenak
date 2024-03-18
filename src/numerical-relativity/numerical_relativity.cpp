//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file numerical_relativity.cpp
//  \brief implementation of functions for NumericalRelativity
#include <iostream>

#include "numerical_relativity.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "dyngr/dyngr.hpp"
#include "tmunu/tmunu.hpp"

namespace numrel {

NumericalRelativity::NumericalRelativity(MeshBlockPack *ppack, ParameterInput *pin) {
  pmy_pack = ppack;
}

std::vector<QueuedTask>& NumericalRelativity::SelectQueue(TaskLocation loc) {
  switch(loc) {
    case Task_Start:
      return start_queue;
    case Task_Run:
      return run_queue;
    case Task_End:
      return end_queue;
    default:
      std::cout << "NumericalRelativity: Unknown task queue requested!\n";
      abort();
  }

  return end_queue;
}

PhysicsDependency NumericalRelativity::NeedsPhysics(TaskName task) {
  if (task < MHD_NTASKS) {
    return Phys_MHD;
  } else if (task < Z4c_NTASKS) {
    return Phys_Z4c;
  } else {
    return Phys_None;
  }
}

bool NumericalRelativity::DependencyAvailable(PhysicsDependency dep) {
  switch(dep) {
    case Phys_None:
      return true;
    case Phys_MHD:
      return pmy_pack->pdyngr != nullptr;
    case Phys_Z4c:
      return pmy_pack->pz4c != nullptr;
    default:
      std::cout << "NumericalRelativity: Unknown dependency\n";
  }
  return false;
}

bool NumericalRelativity::DependenciesMet(std::vector<TaskName>& tasks) {
  for (auto& task : tasks) {
    PhysicsDependency phys = NeedsPhysics(task);
    if (!DependencyAvailable(phys)) {
      return false;
    }
  }

  return true;
}

bool NumericalRelativity::DependenciesMet(QueuedTask& task,
                                          std::vector<QueuedTask>& queue,
                                          TaskID& dependencies) {
  for (auto& test_task : queue) {
    if (HasDependency(test_task.name, task.dependencies)) {
      if (!test_task.added) {
        return false;
      }
      dependencies = dependencies | test_task.id;
    }
  }
  return true;
}

bool NumericalRelativity::HasDependency(TaskName task,
                                        std::vector<TaskName>& dependencies) {
  for (TaskName test_task : dependencies) {
    if (task == test_task) {
      return true;
    }
  }
  return false;
}

void NumericalRelativity::AddExtraDependencies(std::vector<TaskName>& required,
                                               std::vector<TaskName>& optional) {
  for (auto& task : optional) {
    PhysicsDependency phys = NeedsPhysics(task);
    if (DependencyAvailable(phys)) {
      required.push_back(task);
    }
  }
}

bool NumericalRelativity::AssembleNumericalRelativityTasks(
    TaskList &list, std::vector<QueuedTask> &queue) {
  int added = 0;
  int size = queue.size();
  while (added < size) {
    int cycle_added = 0;
    for (auto &task : queue) {
      TaskID dep(0);
      if (DependenciesMet(task, queue, dep) && !task.added) {
        task.added = true;
        task.id = list.AddTask(task.func_, dep);
        cycle_added++;
        added++;
        //std::cout << "Successfully added " << task.name_string << " to task list!\n";
      }
    }
    if (cycle_added == 0) {
      return false;
    }
  }

  return true;
}

void NumericalRelativity::AssembleNumericalRelativityTasks(
    TaskList &start, TaskList &run, TaskList &end) {
  // Assemble the task lists for all physics modules
  if (pmy_pack->pdyngr != nullptr) {
    pmy_pack->pdyngr->QueueDynGRTasks();
  }
  if (pmy_pack->pz4c != nullptr) {
    pmy_pack->pz4c->QueueZ4cTasks();
  }

  bool success = AssembleNumericalRelativityTasks(start, start_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct start TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    abort();
  }

  success = AssembleNumericalRelativityTasks(run, run_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct run TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    abort();
  }

  success = AssembleNumericalRelativityTasks(end, end_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct end TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    abort();
  }
}

} // namespace numrel
