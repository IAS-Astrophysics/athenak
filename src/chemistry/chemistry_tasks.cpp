//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry_tasks.cpp
//! \brief functions that control Chemistry tasks stored in tasklists in
//! MeshBlockPack

#include <map>
#include <memory>
#include <string>

#include "chemistry/chemistry.hpp"
#include "tasklist/task_list.hpp"

namespace chemistry {

void Chemistry::AssembleChemistryTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);  // indicator of no dependency for a given task

  // assemble "after_timeintegrator" task list
  id.test_kernel =
      tl["after_timeintegrator"]->AddTask(&Chemistry::TestKernel, this, none);
}

}  // namespace chemistry
