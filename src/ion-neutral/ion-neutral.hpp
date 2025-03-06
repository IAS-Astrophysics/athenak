#ifndef ION_NEUTRAL_ION_NEUTRAL_HPP_
#define ION_NEUTRAL_ION_NEUTRAL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion-neutral.hpp
//  \brief definitions for IonNeutral class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"

//----------------------------------------------------------------------------------------
//! \struct IonNeutralTaskIDs
//  \brief container to hold TaskIDs of all ion-neutral tasks

struct IonNeutralTaskIDs {
  TaskID i_irecv;
  TaskID n_irecv;
  TaskID impl_2x;
  TaskID i_flux;
  TaskID i_sendf;
  TaskID i_recvf;
  TaskID i_rkupdt;
  TaskID i_restu;
  TaskID n_flux;
  TaskID n_sendf;
  TaskID n_recvf;
  TaskID n_rkupdt;
  TaskID n_restu;
  TaskID impl;
  TaskID i_sendu;
  TaskID i_recvu;
  TaskID n_sendu;
  TaskID n_recvu;
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID i_bcs;
  TaskID n_bcs;
  TaskID i_prol;
  TaskID n_prol;
  TaskID i_c2p;
  TaskID n_c2p;
  TaskID i_newdt;
  TaskID n_newdt;
  TaskID i_clear;
  TaskID n_clear;
  TaskID i_srctrms;
  TaskID n_srctrms;
};

namespace ion_neutral {

//----------------------------------------------------------------------------------------
//! \class IonNeutral

class IonNeutral {
 public:
  IonNeutral(MeshBlockPack *ppack, ParameterInput *pin);
  ~IonNeutral();

  // F = - gamma rho_i rho_n (u_i - u_n) + xi rho_n u_n - alpha rho_i^2 u_i
  // G = xi rho_n - alpha rho_i^2
  Real drag_coeff;       // ion-neutral coupling coefficient, gamma
  Real ionization_coeff;         // ionization rate, xi
  Real recombination_coeff;      // recombination rate, alpha

  // container to hold names of TaskIDs
  IonNeutralTaskIDs id;

  // functions
  void AssembleIonNeutralTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus FirstTwoImpRK(Driver* pdrive, int stage);
  TaskStatus ImpRKUpdate(Driver* pdrive, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace ion_neutral
#endif // ION_NEUTRAL_ION_NEUTRAL_HPP_
