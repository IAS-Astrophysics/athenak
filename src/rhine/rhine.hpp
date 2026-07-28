#ifndef RHINE_RHINE_HPP_
#define RHINE_RHINE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rhine.hpp
//! \brief Definitions for the RHINE class (neural-network r-process heating source term).

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "rhine_net.hpp"

// Forward declarations.
class MeshBlockPack;
class Driver;

namespace rhine {

//----------------------------------------------------------------------------------------
//! Passive-scalar layout in the MHD conserved/primitive arrays.
enum RhineScalar {
  I_YE = IYF,       // electron fraction
  I_YN = IYF + 1,   // neutron abundance
  I_YA = IYF + 2,   // alpha abundance
  I_YH = IYF + 3,   // heavy abundance
  I_AH = IYF + 4,   // average mass number of heavy nuclei
  I_MA = IYF + 5    // average mass excess per baryon
};

//----------------------------------------------------------------------------------------
//! \class RHINE
//! \brief Operator-split r-process heating and composition source term.
class RHINE {
 public:
  RHINE(MeshBlockPack *ppack, ParameterInput *pin);
  ~RHINE();

  //! Apply RHINE source terms to the MHD conserved variables for one RK substep.
  TaskStatus AddSources(Driver *pdrive, int stage);

  RhineNets nets;
  int pmode;

 private:
  MeshBlockPack *pmy_pack;
};

}  // namespace rhine

#endif  // RHINE_RHINE_HPP_
