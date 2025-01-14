//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_newdt.cpp
//! \brief function to compute radiation timestep across all MeshBlock(s) in a
// MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_m1/radiation_m1.hpp"


namespace radiationm1 {

//----------------------------------------------------------------------------------------
// \!fn void RadiationM1::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for radiation problems

TaskStatus RadiationM1::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  // insert code here

  return TaskStatus::complete;
}
} // namespace radiationm1
