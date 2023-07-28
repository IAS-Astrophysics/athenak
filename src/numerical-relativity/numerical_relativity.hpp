#ifndef NUMERICAL_RELATIVITY_HPP_
#define NUMERICAL_RELATIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file numerical_relativity.hpp
//  \brief NumericalRelativity handles the creation of a TaskList for NR modules.

#include <vector>
#include <functional>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"

// Forward declarations
namespace z4c {class Z4c;}
namespace dyngr {class DynGR;}
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
  MHD_RestU,
  MHD_RecvU,
  MHD_EField,
  MHD_SendE,
  MHD_RecvE,
  MHD_CT,
  MHD_RestB,
  MHD_SendB,
  MHD_RecvB,
  MHD_BCS,
  MHD_C2P,
  MHD_Newdt,
  MHD_Clear,
  MHD_NTASKS,

  Z4c_Recv,
  Z4c_CopyU,
  Z4c_MatterSrc,
  Z4c_CalcRHS,
  Z4c_SomBC,
  Z4c_ExplRK,
  Z4c_SendU,
  Z4c_RecvU,
  Z4c_Newdt,
  Z4c_BCS,
  Z4c_AlgC,
  Z4c_Z4c2ADM,
  Z4c_ADMC,
  Z4c_Clear,
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
  QueuedTask(TaskName n, bool a, TaskID i, std::vector<TaskName> d) : 
      name(n), added(a), id(i), dependencies(d) {}
  TaskName name;
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
    void QueueTask(F func, TaskName name, TaskLocation loc, std::vector<TaskName>& dependencies, 
                   std::vector<TaskName>& optional) {
      // Don't add this task if its physics dependencies aren't met, e.g,
      // don't add Z4c matter source terms if Z4c is disabled.
      if (!DependenciesMet(dependencies)) {
        return;
      }
      // Otherwise, check for additional physics and add necessary dependencies.
      AddExtraDependencies(dependencies, optional);
      // Add a new task to the queue.
      SelectQueue(loc).push_back(QueuedTask(name, false, TaskID(), dependencies,
        [=](Driver *d, int s) mutable -> TaskStatus {return func(d,s);}));
    }
    
    // Queue a task to be added to the task list. Filter for dependencies.
    // Task function must have arguments (Driver*, int) and must be a member
    // function of class T.
    template <class F, class T>
    void QueueTask(F func, T *obj, TaskName name, TaskLocation loc,
                   std::vector<TaskName>& dependencies, std::vector<TaskName>& optional) {
      // Don't add this task if its physics dependencies aren't met, e.g.,
      // don't add Z4c matter source terms if Z4c is disabled.
      if (!DependenciesMet(dependencies)) {
        return;
      }
      // Otherwise, check for additional physics and add necessary dependencies.
      AddExtraDependencies(dependencies, optional);
      // Add a new task to the queue.
      SelectQueue(loc).push_back(QueuedTask(name, false, TaskID(), dependencies,
        [=](Driver *d, int s) mutable -> TaskStatus {return (obj->*func(d,s));}));
    }

    void AssembleNumericalRelativityTasks(TaskList &start, TaskList &run, TaskList &end);
  private:
    MeshBlockPack *pmy_pack;
    std::vector<QueuedTask> start_queue;
    std::vector<QueuedTask> run_queue;
    std::vector<QueuedTask> end_queue;

    std::vector<QueuedTask>& SelectQueue(TaskLocation loc);
    PhysicsDependency NeedsPhysics(TaskName task);
    bool DependencyAvailable(PhysicsDependency dep);

    bool DependenciesMet(std::vector<TaskName>& tasks);
    bool DependenciesMet(QueuedTask& task, std::vector<QueuedTask>& queue, TaskID& dependencies);
    bool HasDependency(TaskName task, std::vector<TaskName>& dependencies);
    void AddExtraDependencies(std::vector<TaskName>& required, std::vector<TaskName>& optional);

    bool AssembleNumericalRelativityTasks(TaskList &list, std::vector<QueuedTask> &queue);
};

}

#endif
