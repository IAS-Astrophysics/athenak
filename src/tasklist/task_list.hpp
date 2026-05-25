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
// This version includes improvements due to Josh Dolence and the Parthenon dev team, and
// extensions by J.M.Stone.

#include <iostream>
#include <bitset>
#include <functional>
#include <vector>
#include <list>
#include <iterator>

class Driver;

// Maximum size of TL
#define NUMBER_TASKID_BITS 64

// constants = return codes for functions working on individual Tasks and TaskList
enum class TaskStatus {fail, complete, incomplete};
enum class TaskListStatus {running, stuck, complete, nothing_to_do};

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief container class for bit fields (used to encode Task IDs) and access functions

class TaskID {
 public:
  TaskID() = default;
  // ctor, default id = 0.
  explicit TaskID(unsigned int id) {
    if (id == 0) {
      bitfld_.reset();      // set all bits to zero
    } else {
      bitfld_.set((--id));  // set [id-1] bit to one
    }
  }

  // functions (all implemented here)
  void Clear() { bitfld_.reset(); }  // set all bits to zero
  // return true if input dependencies are clear
  bool CheckDependencies(const TaskID &dep) const {
    return ((bitfld_ & dep.bitfld_) == dep.bitfld_);
  }
  // output ID (useful for debugging)
  void PrintID() {std::cout << "TaskID = " << bitfld_.to_string() << std::endl;}
  // mark task with input TaskID as complete
  void SetComplete(const TaskID &rhs) { bitfld_ |= rhs.bitfld_; }

  // overload some operators
  bool operator== (const TaskID &rhs) const {return (bitfld_ == rhs.bitfld_); }
  bool operator!= (const TaskID &rhs) const {return (bitfld_ != rhs.bitfld_); }
  TaskID operator| (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ | rhs.bitfld_);
    return ret;
  }
  TaskID operator^ (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ ^ rhs.bitfld_);
    return ret;
  }
  TaskID operator& (const TaskID &rhs) const {
    TaskID ret;
    ret.bitfld_ = (bitfld_ & rhs.bitfld_);
    return ret;
  }

 private:
  std::bitset<NUMBER_TASKID_BITS> bitfld_;
};

//----------------------------------------------------------------------------------------
//! \class Task
//  \brief data and function pointer for an individual Task
//  NOTE: Task function must take arguments (Driver*, int)

class Task {
 public:
  Task(TaskID id, TaskID dep, std::function<TaskStatus(Driver*, int)> func) :
  myid_(id), dep_(dep), func_(func) {}
  // overloaded operator() calls task function
  TaskStatus operator()(Driver *d, int s) {return func_(d,s);}
  TaskID GetID() {return myid_;}
  TaskID GetDependency() {return dep_;}
  void SetComplete() {complete_ = true;}
  void SetIncomplete() {complete_ = false;}
  bool IsComplete() {return complete_;}
  // If this Task depends on id, change that dependency to 'newdep'
  void ChangeDependency(TaskID id, TaskID newdep) {
    if ((dep_ & id) == id) {dep_ = ((dep_ ^ id) | newdep);}
  }

 private:
  TaskID myid_;    // encodes task ID in bitfld_
  TaskID dep_;     // encodes dependencies to other tasks in bitfld_
  // bool lb_time_;   // flag to include this task in timing for automatic load balancing
  bool complete_ = false;
  std::function<TaskStatus(Driver*, int)> func_;  // ptr to Task function
};

//----------------------------------------------------------------------------------------
//! \class TaskList
//  \brief data and function definitions for task list class

class TaskList {
 public:
  TaskList() = default;
  ~TaskList() = default;

  // functions (all implemented here)
  bool IsComplete() {
    // cycle through task list and check if each task completed
    for (auto &it : task_list_) {
      auto id = it.GetID();
      if (!(tasks_completed_.CheckDependencies(id))) return false;
    }
    // everything is done
    return true;
  }
  int Size() {return task_list_.size();}
  bool Empty() {return task_list_.empty();}
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetComplete(id); }
  TaskID GetIDLastTask() {return task_list_.back().GetID();}
  // output diagnostics (useful for debugging)
  void PrintIDs() { for (auto &it : task_list_) {it.GetID().PrintID();} }
  void PrintDependencies() { for (auto &it : task_list_) {it.GetDependency().PrintID();} }

  //
  void Reset() {
    tasks_completed_.Clear();  // TaskID Clear() fn
    for (auto &it : task_list_) { it.SetIncomplete(); }
  }

  // cycle through task list once, do any tasks whose dependencies are clear
  TaskListStatus DoAvailable(Driver *d, int s) {
    for (auto &task : task_list_) {
      auto dep = task.GetDependency();
      if ( tasks_completed_.CheckDependencies(dep) && !(task.IsComplete()) ) {
        TaskStatus status = task(d,s);  // calls Task function using overloaded operator()
        if (status == TaskStatus::complete) {
          task.SetComplete();              // set bool flag in task
          MarkTaskComplete(task.GetID());  // add TaskID to tasks_completed_
        }
      }
    }
    if (IsComplete()) return TaskListStatus::complete;
    return TaskListStatus::running;
  }

  // ADD new Task with ID, given dependency, and a pointer to a static or non-member
  // function to the end of task list.  Returns ID of new task. Task function must have
  // arguments (Driver*, int). Usage:
  //     taskid = tl.AddTask(DoSomething, dependency, name);
  template <class F>
  TaskID AddTask(F func, TaskID &dep) {
    auto size = task_list_.size();
    TaskID id(size+1);
    task_list_.push_back(
      Task(id, dep, [=](Driver *d, int s) mutable -> TaskStatus {return func(d,s);}));
    return id;
  }

  // ADD new Task with ID, given dependency, and a pointer to a member function of
  // class T to the end of task list.  Returns ID of new task. Task function must have
  // arguments (Driver*, int).  Usage:
  //     taskid = tl.AddTask(&T::DoSomething, T, dependency);
  template <class F, class T>
  TaskID AddTask(F func, T *obj, TaskID &dep) {
    auto size = task_list_.size();
    TaskID id(size+1);
    task_list_.push_back( Task(id, dep,
       [=](Driver *d, int s) mutable -> TaskStatus {return (obj->*func)(d,s);}) );
    return id;
  }

  // ADD new Task with ID, given dependency, and a std::function to the end of task
  // list. Returns ID of new task. Task function must have arguments (Driver*, int).
  // Usage:
  //      taskid = tl.AddTask(DoSomething, dependency);
  TaskID AddTask(std::function<TaskStatus(Driver*, int)> func, TaskID &dep) {
    auto size = task_list_.size();
    TaskID id(size+1);
    task_list_.push_back(Task(id, dep, func));
    return id;
  }

  // INSERT new Task with ID, given dependency, and a pointer to a member function of
  // class T in a position BEFORE the task with ID 'location'.  Returns ID of new task,
  // or taskID(0) if location not found. Usage:
  //     taskid = tl.InsertTask(&T::DoSomething, T, dependency, location);
  template <class F, class T>
  TaskID InsertTask(F func, T *obj, TaskID &dep, TaskID &loc) {
    std::list<Task>::iterator it;
    for (it=task_list_.begin(); it!=task_list_.end(); ++it) {
      if (it->GetID() == loc) {
        auto size = task_list_.size();
        TaskID id(size+1);
        auto old_dep = it->GetDependency();
        task_list_.insert(it, Task(id, dep,
           [=](Driver *d, int s) mutable -> TaskStatus {return (obj->*func)(d,s); }));
        // now change dependencies for all but this newly added Task
        for (auto it2=task_list_.begin(); it2!=task_list_.end(); ++it2) {
          if (it2->GetID() != id) {
            it2->ChangeDependency(old_dep, id);
          }
        }
        return id;
      }
    }
    return TaskID(0);
  }

 protected:
  std::list<Task> task_list_;
  TaskID tasks_completed_;
};

#endif  // TASKLIST_TASK_LIST_HPP_
