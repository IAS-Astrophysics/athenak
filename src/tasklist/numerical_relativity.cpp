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
#include "hydro/hydro.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "radiation/radiation.hpp"

namespace numrel {

NumericalRelativity::NumericalRelativity(MeshBlockPack *ppack, ParameterInput *pin) {
  pmy_pack = ppack;
  for (const auto& [key, value] : pmy_pack->tl_map) {
    queue_map[key] = std::vector<QueuedTask>();
  }
}

bool NumericalRelativity::DependencyAvailable(std::string& task) {
  if (task.compare(0,2, "Hyd") == 0) {
    return pmy_pack->phydro != nullptr;
  } else if (task.compare(0,3,"MHD") == 0) {
    return pmy_pack->pmhd != nullptr;
  } else if (task.compare(0,3,"Z4c") == 0) {
    return pmy_pack->pz4c != nullptr;
  } else if (task.compare(0,3,"Rad") == 0) {
    return pmy_pack->prad != nullptr;
  }

  return true;
}

bool NumericalRelativity::DependenciesMet(std::vector<std::string>& tasks) {
  for (auto& task : tasks) {
    if (!DependencyAvailable(task)) {
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
      if (test_task.name_string == dep) {
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

void NumericalRelativity::AddExtraDependencies(std::vector<std::string>& required,
                                               std::vector<std::string>& optional) {
  for (auto& task : optional) {
    if (DependencyAvailable(task)) {
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
        std::cout << task.name_string << "\n";
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
  if (pmy_pack->phydro != nullptr) {
    pmy_pack->phydro->QueueHydroTasks();
  }
  if (pmy_pack->pdyngr != nullptr) {
    pmy_pack->pdyngr->QueueDynGRMHDTasks();
  }
  if (pmy_pack->pz4c != nullptr) {
    pmy_pack->pz4c->QueueZ4cTasks();
  }
  if (pmy_pack->prad != nullptr) {
    pmy_pack->prad->QueueRadTasks();
  }

  for (auto& [key, queue] : queue_map) {
    std::cout << key << "\n";
    bool success = AssembleNumericalRelativityTasks(tl[key], queue);
    if (!success) {
      std::cout << "NumericalRelativity: Failed to construct " << key << "TaskList!\n"
                << "  Check that there are no cyclical dependencies or missing tasks.\n";
      PrintMissingTasks(queue);
      abort();
    }
    std::cout << "\n";
  }
}

} // namespace numrel
