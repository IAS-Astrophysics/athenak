#ifndef TASKLIST_NUMERICAL_RELATIVITY_HPP_
#define TASKLIST_NUMERICAL_RELATIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file numerical_relativity.hpp
//  \brief NumericalRelativity handles the creation of a TaskList for NR modules.

#include <vector>
#include <map>
#include <functional>
#include <string>
#include <memory>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"

// Forward declarations
namespace z4c {class Z4c;}
namespace dyngr {class DynGRMHD;}
class ADM;
class Tmunu;
class MeshBlockPack;

namespace numrel {

enum PhysicsDependency {
  Phys_None,
  Phys_MHD,
  Phys_Z4c
};

struct QueuedTask {
  QueuedTask(const std::string s, bool a, TaskID i, std::vector<std::string>& d,
             std::function<TaskStatus(Driver *, int)> func) :
      name_string(s), added(a), id(i), dependencies(d), func_(func) {}
  const std::string name_string;
  bool added;
  TaskID id;
  std::vector<std::string> dependencies;

  std::function<TaskStatus(Driver*, int)> func_;
};

class NumericalRelativity {
 public:
  NumericalRelativity(MeshBlockPack *ppack, ParameterInput *pin);

  // Queue a task to be added to the task list. Filter for dependencies.
  // Task function must have arguments (Driver*, int).
  template <class F>
  void QueueTask(F func, const std::string name_string,
                 const std::string queue, std::vector<std::string> dependencies = {},
                 std::vector<std::string> optional = {}) {
    // Don't add this task if its physics dependencies aren't met, e.g,
    // don't add Z4c matter source terms if Z4c is disabled.
    if (!DependenciesMet(dependencies)) {
      return;
    }
    // Otherwise, check for additional physics and add necessary dependencies.
    AddExtraDependencies(dependencies, optional);
    // Add a new task to the queue.
    //std::cout << "Queuing " << name_string << "...\n";
    queue_map[queue].push_back(QueuedTask(name_string, false, TaskID(),
      dependencies, [=](Driver *d, int s) mutable -> TaskStatus {return func(d,s);}));
  }

  // Queue a task to be added to the task list. Filter for dependencies.
  // Task function must have arguments (Driver*, int) and must be a member
  // function of class T.
  template <class F, class T>
  void QueueTask(F func, T *obj, const std::string name_string,
                 const std::string queue, std::vector<std::string> dependencies = {},
                 std::vector<std::string> optional = {}) {
    // Don't add this task if its physics dependencies aren't met, e.g.,
    // don't add Z4c matter source terms if Z4c is disabled.
    if (!DependenciesMet(dependencies)) {
      return;
    }
    // Otherwise, check for additional physics and add necessary dependencies.
    AddExtraDependencies(dependencies, optional);
    // Add a new task to the queue.
    //std::cout << "Queuing " << name_string << "...\n";
    queue_map[queue].push_back(QueuedTask(name_string, false, TaskID(),
      dependencies,
      [=](Driver *d, int s) mutable -> TaskStatus {return (obj->*func)(d,s);}));
  }

  void AssembleNumericalRelativityTasks(
         std::map<std::string, std::shared_ptr<TaskList>>& tl);

 private:
  MeshBlockPack *pmy_pack;
  std::map<std::string, std::vector<QueuedTask>> queue_map;

  PhysicsDependency NeedsPhysics(std::string& task);
  bool DependencyAvailable(PhysicsDependency dep);

  bool DependenciesMet(std::vector<std::string>& tasks);
  bool DependenciesMet(QueuedTask& task, std::vector<QueuedTask>& queue,
                       TaskID& dependencies);
  bool HasDependency(std::string& task, std::vector<std::string>& dependencies);
  void AddExtraDependencies(std::vector<std::string>& required,
                            std::vector<std::string>& optional);

  void PrintMissingTasks(std::vector<QueuedTask> &queue);

  bool AssembleNumericalRelativityTasks(std::shared_ptr<TaskList>& list,
         std::vector<QueuedTask> &queue);
};

} // namespace numrel

#endif  // TASKLIST_NUMERICAL_RELATIVITY_HPP_
