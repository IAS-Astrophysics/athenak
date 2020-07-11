#ifndef TASKLIST_TASK_LIST_HPP_
#define TASKLIST_TASK_LIST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//!   \file task_list.hpp
//    \brief provides functionality to control dynamic execution using tasks
//           currently everything is implemented in this single header file
//
// The original idea and implementation of TaskLists was by Kengo Tomida
// This version includes ideas due to Josh Dolence and the Parthenon development team

//#include <cstdint>      // std::uint64_t
//#include <string>       // std::string
#include <bitset>
#include <functional>
#include <vector>
#include <list>

//#include "athena.hpp"

#define NUMBER_TASKID_BITS 64

// constants = return codes for functions working on individual Tasks and TaskList
enum class TaskStatus {fail, complete, incomplete};
enum class TaskListStatus {running, stuck, complete, nothing_to_do};

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief generalization of bit fields for Task IDs, status, and dependencies.

class TaskID {
 public:
  TaskID() = default;
  // ctor, default id = 0.
  explicit TaskID(unsigned int id) {
    if (id == 0) {
      bitfld_.reset();      // set all bits to zero
    } else {
      bitfld_.set((id--));  // set [id-1] bit to one
    }
  }

  // functions (all implemented here)
  void Clear() { bitfld_.reset(); }  // set all bits to zero

  // return true if all input dependencies are clear
  bool CheckDependencies(const TaskID &dep) const {
    return ((bitfld_ & dep.bitfld_) == dep.bitfld_);
  }

  // mark task with input TaskID as finished
  void SetFinished(const TaskID &rhs) { bitfld_ |= rhs.bitfld_; }

  // overload some operators
  bool operator== (const TaskID &rhs) const {return (bitfld_ == rhs.bitfld_); }
  TaskID operator| (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ | rhs.bitfld_);
    return ret;
  }

 private:
  std::bitset<NUMBER_TASKID_BITS> bitfld_;
};

//----------------------------------------------------------------------------------------
//! \class Task
//  \brief data and function pointer for an individual Task

class Task {
 public:
  Task(TaskID id, TaskID dep, std::function<TaskStatus()> func)
      : myid_(id), dep_(dep), func_(func) {}
  TaskStatus operator()() { return func_(); }  // operator() calls task function
  TaskID GetID() { return myid_; }
  TaskID GetDependency() { return dep_; }
  void SetComplete() { complete_ = true; }
  bool IsComplete() { return complete_; }

 private:
  TaskID myid_;   // encodes task ID in bitfld_
  TaskID dep_;     // encodes dependencies to other tasks in bitfld_
  bool lb_time_;   // flag to include this task in timing for automatic load balancing
  bool complete_ = false;
  std::function<TaskStatus()> func_;  // ptr to Task function

};

//----------------------------------------------------------------------------------------
//! \class TaskList
//  \brief data and function definitions for task list base class

class TaskList {
 public:
  TaskList() = default;
  ~TaskList() = default;

  // functions (all implemented here)
  bool IsComplete() { return task_list_.empty(); }
  int Size() { return task_list_.size(); }
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetFinished(id); }

  //
  void Reset() {
    ntasks_added_ = 0;
    task_list_.clear();        // std::vect clear() fn
    dependencies_.clear();     // std::vect clear() fn
    tasks_completed_.Clear();  // TaskID Clear() fn
  }

  //
  bool IsReady() {
    for (auto &l : dependencies_) {
      if (!l->IsComplete()) { return false; }
    }
    return true;
  }

  //
  void ClearComplete() {
    for (auto it = task_list_.begin(); it != task_list_.end(); ++it) {
      if (it->IsComplete()) { it = task_list_.erase(it);}
    }
  }

  //
  TaskListStatus DoAvailable() {
    for (auto &task : task_list_) {
      auto dep = task.GetDependency();
      if (tasks_completed_.CheckDependencies(dep)) {
        TaskStatus status = task();  // calls Task function using overloaded operator()
        if (status == TaskStatus::complete) {
          task.SetComplete();
          MarkTaskComplete(task.GetID());
        }
      }
    }
    ClearComplete();
    if (IsComplete()) return TaskListStatus::complete;
    return TaskListStatus::running;
  }

  //
  template <class F, class... Args>
  TaskID AddTask(F func, TaskID &dep, Args &&... args) {
    TaskID id(ntasks_added_ + 1);
    task_list_.push_back(
        Task(id, dep, [=]() mutable -> TaskStatus { return func(args...); }));
    ntasks_added_++;
    return id;
  }

 protected:
  std::list<Task> task_list_;
  int ntasks_added_ = 0;
  std::vector<TaskList *> dependencies_;
  TaskID tasks_completed_;
};

#endif  // TASKLIST_TASK_LIST_HPP_
