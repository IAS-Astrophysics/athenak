#ifndef CHEMISTRY_CHEMISTRY_HPP_
#define CHEMISTRY_CHEMISTRY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry.hpp
//  \brief definitions for Chemistry class

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

namespace chemistry {

//! \struct ChemistryTaskIDs
//  \brief container to hold TaskIDs of all chemistry tasks
struct ChemistryTaskIDs {
  TaskID test_kernel;
};

//! \class Chemistry
class Chemistry {
 public:
  // ==========================
  // Constructor and Destructor
  // ==========================
  Chemistry(MeshBlockPack* ppack, ParameterInput* pin, int& nscalars,
            int const nconserved);
  ~Chemistry();

  // ================
  // Member Variables
  // ================
  // Container to hold names of TaskIDs
  ChemistryTaskIDs id;

  // ================
  // Member Functions
  // ================
  void AssembleChemistryTasks(
      std::map<std::string, std::shared_ptr<TaskList>> tl);

  TaskStatus HelloWorld(Driver* d, int stage);
  TaskStatus TestKernel(Driver* d, int stage);

  // ===================
  // Getters and Setters
  // ===================
  int get_nscalars_chemistry() const { return nscalars_chemistry; }
  int get_chemistry_scalars_start_idx() const {
    return chemistry_scalars_start_idx;
  }
  // The index of the final chemistry scalar
  int get_chemistry_scalars_stop_idx() const {
    return chemistry_scalars_start_idx + nscalars_chemistry - 1;
  }

 private:
  // ptr to MeshBlockPack containing this chemistry. note that this is a const
  // pointers, the contents can be changed but the pointer address can't
  MeshBlockPack* const pmy_pack;

  // These indicate if hydro or MHD is in use
  bool const is_hydro_enabled;
  bool const is_mhd_enabled;

  // The number of passive scalars used in the chemistry module
  int nscalars_chemistry = 3;

  // The beginning index of passive scalars reserved for chemisty
  int const chemistry_scalars_start_idx;

  // Get the correct u0 array
  DvceArray5D<Real> GetU0();
};
}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_HPP_
