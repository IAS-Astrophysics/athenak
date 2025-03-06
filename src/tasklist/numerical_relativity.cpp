//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file numerical_relativity.cpp
//  \brief implementation of functions for NumericalRelativity
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "numerical_relativity.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

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
  // Loop through all the dependencies, then compare them to each task in the queue.
  // If a dependency hasn't been added to the task list, return false. Otherwise, add its
  // id to the dependency id. If the dependency simply doesn't exist in the list, always
  // return false.
  for (auto& dep : task.dependencies) {
    bool found = false;
    for (auto &test_task : queue) {
      if (test_task.name == dep) {
        found = true;
        if (!test_task.added) {
          return false;
        }
        dependencies = dependencies | test_task.id;
      }
    }
    // This handles the corner case where the dependency isn't in the queue.
    if (!found) {
      return false;
    }
  }

  return true;
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
    std::shared_ptr<TaskList>& list, std::vector<QueuedTask> &queue) {
  int added = 0;
  int size = queue.size();
  while (added < size) {
    int cycle_added = 0;
    for (auto &task : queue) {
      TaskID dep(0);
      if (DependenciesMet(task, queue, dep) && !task.added) {
        task.added = true;
        task.id = list->AddTask(task.func_, dep);
        cycle_added++;
        added++;
        /*std::cout << "Successfully added " << task.name_string << " to task list!\n"
                  << "  ID: ";
        task.id.PrintID();
        std::cout << "  Dependencies: ";
        dep.PrintID();*/
      }
    }
    if (cycle_added == 0) {
      return false;
    }
  }

  return true;
}

void NumericalRelativity::PrintMissingTasks(std::vector<QueuedTask> &queue) {
  std::cout << "Successfully added the following tasks:\n";
  for (auto& task : queue) {
    if (task.added) {
      std::cout << "  " << task.name_string << "\n";
    }
  }
  std::cout << "Could not add the following tasks:\n";
  for (auto& task : queue) {
    if (!task.added) {
      std::cout << "  " << task.name_string << "\n";
    }
  }
}

void NumericalRelativity::AssembleNumericalRelativityTasks(
       std::map<std::string, std::shared_ptr<TaskList>>& tl) {
  // Assemble the task lists for all physics modules
  if (pmy_pack->pdyngr != nullptr) {
    pmy_pack->pdyngr->QueueDynGRMHDTasks();
  }
  if (pmy_pack->pz4c != nullptr) {
    pmy_pack->pz4c->QueueZ4cTasks();
  }

  bool success = AssembleNumericalRelativityTasks(tl["before_stagen"], start_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct start TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    PrintMissingTasks(start_queue);
    abort();
  }

  success = AssembleNumericalRelativityTasks(tl["stagen"], run_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct run TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    PrintMissingTasks(run_queue);
    abort();
  }

  success = AssembleNumericalRelativityTasks(tl["after_stagen"], end_queue);
  if (!success) {
    std::cout << "NumericalRelativity: Failed to construct end TaskList!\n"
              << "  Check that there are no cyclical dependencies or missing tasks.\n";
    PrintMissingTasks(end_queue);
    abort();
  }
}

} // namespace numrel
