#ifndef NUCLEAR_HPP
#define NUCLEAR_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.hpp
//  \brief definitions for Nuclear module class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

namespace nuclear {

//----------------------------------------------------------------------------------------
//! \struct NuclearParams
//  \brief Parameters of the network integration methods, read from <nuclear>. Fields
//  shared by all methods have no prefix; method-specific ones carry the method's prefix
//  (be_: backward Euler, nuclear/integrate/backward_euler.hpp). Plain data, so it can be
//  captured by value into device kernels.
struct NuclearParams {
  // shared by all integration methods
  Real y_atol;        // absolute tolerance on rho*X[n] [g/cm^3]
  Real y_rtol;        // relative tolerance on rho*X[n]
  int max_iter;       // max nonlinear iterations per network substep
  // backward Euler
  int be_n_refactor;  // refactor the Jacobian at least every be_n_refactor iterations
};

//----------------------------------------------------------------------------------------
//! \struct NuclearTaskIDs
//  \brief container to hold TaskIDs of all Nuclear tasks
struct NuclearTaskIDs {
  TaskID Nuclear_integrate;
};

//----------------------------------------------------------------------------------------
//! \class RadiationM1
//  \brief class for grey M1
class Nuclear {
 public:
  Nuclear (MeshBlockPack* ppack, ParameterInput* pin);
  ~Nuclear();

  bool ismhd;                              // flag to check if <mhd> present
  bool ishydro;                            // flag to check if <hydro> present
  bool isunits;                            // flag to check if <units> present
                                           //
  //DvceArray5D<Real> Ydot;

  NuclearTaskIDs id;          // container to hold names of TaskIDs
  Real dt_nuc{};              // network substep size, set by the pgen
  int max_substeps{};         // cap on the network substeps per hydro step
  Real dt_nuc_factor{};       // dt_nuc = dt_nuc_factor*min(cv*T/|edot|), set by the pgen
  NuclearParams params{};     // parameters of the network integration methods

  void AssembleNuclearTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);

  TaskStatus Integrate(Driver* pdrive, int stage);
  // Hydro ghost-zone refresh after the burn (see nuclear_task.cpp)
  TaskStatus InitRecvU(Driver* pdrive, int stage);
  TaskStatus SendU(Driver* pdrive, int stage);
  TaskStatus RecvU(Driver* pdrive, int stage);
  TaskStatus SetGhostZones(Driver* pdrive, int stage);
  TaskStatus ClearSendU(Driver* pdrive, int stage);
  TaskStatus ClearRecvU(Driver* pdrive, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack
};
}

#endif  // NUCLEAR_HPP
