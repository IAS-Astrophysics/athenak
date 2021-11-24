#ifndef ION_NEUTRAL_ION_NEUTRAL_HPP_
#define ION_NEUTRAL_ION_NEUTRAL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion-neutral.hpp
//  \brief definitions for IonNeutral class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"

//----------------------------------------------------------------------------------------
//! \struct IonNeutralTaskIDs
//  \brief container to hold TaskIDs of all ion-neutral tasks
  
struct IonNeutralTaskIDs
{   
  TaskID i_irecv;
  TaskID n_irecv;
  TaskID impl_2x;
  TaskID i_flux;
  TaskID n_flux;
  TaskID i_expl;
  TaskID n_expl;
  TaskID impl;
  TaskID i_sendu;
  TaskID n_sendu;
  TaskID i_recvu;
  TaskID n_recvu;
  TaskID efld;
  TaskID ct;
  TaskID sendb;
  TaskID recvb;
  TaskID i_bcs;
  TaskID n_bcs;
  TaskID i_c2p;
  TaskID n_c2p;
  TaskID i_newdt;
  TaskID n_newdt;
  TaskID i_clear;
  TaskID n_clear;
};

namespace ion_neutral {

//----------------------------------------------------------------------------------------
//! \class IonNeutral

class IonNeutral
{
 public:
  IonNeutral(MeshBlockPack *ppack, ParameterInput *pin);
  ~IonNeutral();

  Real drag_coeff;       // ion-neutral coupling coefficient
  DvceArray6D<Real> ru;  // drag term in each dirn evaluated at different time levels

  // container to hold names of TaskIDs
  IonNeutralTaskIDs id;

  // functions
  void AssembleIonNeutralTasks(TaskList &start, TaskList &run, TaskList &end);
  TaskStatus FirstTwoImpRK(Driver* pdrive, int stage);
  TaskStatus ImpRKUpdate(Driver* pdrive, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace ion_neutral
#endif // ION_NEUTRAL_ION_NEUTRAL_HPP_
