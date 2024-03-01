//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//! \brief implementation of ShearingBox class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"

namespace shearing_box {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ShearingBox::ShearingBox(MeshBlockPack *ppack, ParameterInput *pin) :
    qshear(0.0),
    omega0(0.0),
    jshift(1),
    pmy_pack(ppack) {
  // read shear parameters
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");

  // estimate maximum integer shift in x2-direction for orbital advection
  Real xmin = fabs(ppack->pmesh->mesh_size.x1min);
  Real xmax = fabs(ppack->pmesh->mesh_size.x1max);
  jshift = static_cast<int>((ppack->pmesh->cfl_no)*std::max(xmin,xmax)) + 2;
}

//----------------------------------------------------------------------------------------
// destructor

ShearingBox::~ShearingBox() {
}

} // namespace shearing_box
