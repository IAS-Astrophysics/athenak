//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_beams.cpp
//  \brief set up beam sources

#include <string>
#include <cmath>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::BeamsSourcesFEMN(Driver *pdriver, int stage) {

  std::cout << "Hello! I am beam sources!" << std::endl;
  return TaskStatus::complete;
}
}