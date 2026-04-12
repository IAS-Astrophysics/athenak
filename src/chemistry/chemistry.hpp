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
  Chemistry(MeshBlockPack* ppack, ParameterInput* pin);
  ~Chemistry();

  // ================
  // Member Variables
  // ================
  // Container to hold names of TaskIDs
  ChemistryTaskIDs id;

  // The number of passive scalars used in the chemistry module
  int const nscalars_chemistry;

  // ================
  // Member Functions
  // ================
  void AssembleChemistryTasks(
      std::map<std::string, std::shared_ptr<TaskList>> tl);

  TaskStatus TestKernel(Driver* d, int stage);

  /*!
   * \brief Compute the number of required passive scalars that the chemistry
   * module will use
   *
   * \param ppack The MeshBlockPack instance in use
   * \param pin The ParameterInput instance in use
   * \param nscalars The current number of nscalars
   * \param chemistry_constructor Whether or not this is being called within the
   * Chemistry constructor. This should be left to its default value otherwise.
   * \return int The number of passive scalars that the chemistry module
   * requires
   */
  static int SetupGetNumChemistryScalars(
      MeshBlockPack* ppack, ParameterInput* pin, int const& nscalars,
      bool const not_in_chemistry_constructor = true) {
    // Capture the pre-chemistry number of passive scalars so that the Chemistry
    // constructor later knows where to start the indexing for Chemistry passive
    // scalars
    if (not_in_chemistry_constructor) {
      nscalars_pre_chemistry = nscalars;
    }

    // This is temporary, eventually it will return a value dependent on the
    // chemistry module specified
    return 3;
  }

  /*!
   * \brief Return the name of the chemical species at the grid field index provided
   *
   * \param scalar_idx The index in the grid to the passive scalar in question
   * \return std::string The name of the chemical species
   */
  std::string GetSpeciesNames(int const &scalar_idx);

  // ===================
  // Getters and Setters
  // ===================
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

  // The number of regular passive scalars created before the chemistry scalars
  int inline static nscalars_pre_chemistry;

  // The beginning index of passive scalars reserved for chemisty
  int const chemistry_scalars_start_idx;

  // Get the correct u0 array
  DvceArray5D<Real> GetU0();

  // Functions for setting up
  int ComputeChemistryScalarsStartIndex();
};
}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_HPP_
