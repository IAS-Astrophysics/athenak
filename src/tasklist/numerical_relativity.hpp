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

enum TaskName {
  MHD_Recv,
  MHD_CopyU,
  MHD_Flux,
  MHD_SetTmunu,
  MHD_SendFlux,
  MHD_RecvFlux,
  MHD_ExplRK,
  MHD_AddSrc,
  MHD_RestU,
  MHD_SendU,
  MHD_RecvU,
  MHD_EField,
  MHD_SendE,
  MHD_RecvE,
  MHD_CT,
  MHD_RestB,
  MHD_SendB,
  MHD_RecvB,
  MHD_BCS,
  MHD_Prolong,
  MHD_SetADM,
  MHD_Excise,
  MHD_C2P,
  MHD_Newdt,
  MHD_ClearS,
  MHD_ClearR,
  MHD_NTASKS,

  Z4c_Recv,
  Z4c_IRecvW,
  Z4c_CopyU,
  Z4c_CalcRHS,
  Z4c_SomBC,
  Z4c_ExplRK,
  Z4c_SendU,
  Z4c_RestU,
  Z4c_RecvU,
  Z4c_Newdt,
  Z4c_BCS,
  Z4c_Prolong,
  Z4c_AlgC,
  Z4c_Z4c2ADM,
  Z4c_Excise,
  Z4c_ADMC,
  Z4c_ClearS,
  Z4c_ClearR,
  Z4c_Weyl,
  Z4c_RestW,
  Z4c_SendW,
  Z4c_RecvW,
  Z4c_ProlW,
  Z4c_ClearSW,
  Z4c_ClearRW,
  Z4c_Wave,
  Z4c_PT,
  Z4c_CCE,
  Z4c_DumpHorizon,
  Z4c_NTASKS
};

enum PhysicsDependency {
  Phys_None,
  Phys_MHD,
  Phys_Z4c
};

enum TaskLocation {
  Task_Start,
  Task_Run,
  Task_End
};

struct QueuedTask {
  QueuedTask(TaskName n, const std::string s, bool a, TaskID i, std::vector<TaskName>& d,
             std::function<TaskStatus(Driver *, int)> func) :
      name(n), name_string(s), added(a), id(i), dependencies(d), func_(func) {}
  TaskName name;
  const std::string name_string;
  bool added;
  TaskID id;
  std::vector<TaskName> dependencies;

  std::function<TaskStatus(Driver*, int)> func_;
};

class NumericalRelativity {
 public:
  NumericalRelativity(MeshBlockPack *ppack, ParameterInput *pin);

  // Queue a task to be added to the task list. Filter for dependencies.
  // Task function must have arguments (Driver*, int).
  template <class F>
  void QueueTask(F func, TaskName name, const std::string name_string,
                 TaskLocation loc, std::vector<TaskName> dependencies = {},
                 std::vector<TaskName> optional = {}) {
    // Don't add this task if its physics dependencies aren't met, e.g,
    // don't add Z4c matter source terms if Z4c is disabled.
    if (!DependenciesMet(dependencies)) {
      return;
    }
    // Otherwise, check for additional physics and add necessary dependencies.
    AddExtraDependencies(dependencies, optional);
    // Add a new task to the queue.
    //std::cout << "Queuing " << name_string << "...\n";
    SelectQueue(loc).push_back(QueuedTask(name, name_string, false, TaskID(),
      dependencies, [=](Driver *d, int s) mutable -> TaskStatus {return func(d,s);}));
  }

  // Queue a task to be added to the task list. Filter for dependencies.
  // Task function must have arguments (Driver*, int) and must be a member
  // function of class T.
  template <class F, class T>
  void QueueTask(F func, T *obj, TaskName name, const std::string name_string,
                 TaskLocation loc, std::vector<TaskName> dependencies = {},
                 std::vector<TaskName> optional = {}) {
    // Don't add this task if its physics dependencies aren't met, e.g.,
    // don't add Z4c matter source terms if Z4c is disabled.
    if (!DependenciesMet(dependencies)) {
      return;
    }
    // Otherwise, check for additional physics and add necessary dependencies.
    AddExtraDependencies(dependencies, optional);
    // Add a new task to the queue.
    //std::cout << "Queuing " << name_string << "...\n";
    auto& queue = SelectQueue(loc);
    SelectQueue(loc).push_back(QueuedTask(name, name_string, false, TaskID(),
      dependencies,
      [=](Driver *d, int s) mutable -> TaskStatus {return (obj->*func)(d,s);}));
  }

  void AssembleNumericalRelativityTasks(
         std::map<std::string, std::shared_ptr<TaskList>>& tl);

 private:
  MeshBlockPack *pmy_pack;
  std::vector<QueuedTask> start_queue;
  std::vector<QueuedTask> run_queue;
  std::vector<QueuedTask> end_queue;

  std::vector<QueuedTask>& SelectQueue(TaskLocation loc);
  PhysicsDependency NeedsPhysics(TaskName task);
  bool DependencyAvailable(PhysicsDependency dep);

  bool DependenciesMet(std::vector<TaskName>& tasks);
  bool DependenciesMet(QueuedTask& task, std::vector<QueuedTask>& queue,
                       TaskID& dependencies);
  bool HasDependency(TaskName task, std::vector<TaskName>& dependencies);
  void AddExtraDependencies(std::vector<TaskName>& required,
                            std::vector<TaskName>& optional);

  void PrintMissingTasks(std::vector<QueuedTask> &queue);

  bool AssembleNumericalRelativityTasks(std::shared_ptr<TaskList>& list,
         std::vector<QueuedTask> &queue);
};

} // namespace numrel

#endif  // TASKLIST_NUMERICAL_RELATIVITY_HPP_
