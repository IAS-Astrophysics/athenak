//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // read number of particles per cell, and calculate number of particles this pack
  int ppc = pin->GetOrAddInteger("particles","ppc",1);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  nparticles_thispack = ppc*(pmy_pack->nmb_thispack)*indcs.nx1*indcs.nx2*indcs.nx3;

  // select pusher algorithm
  std::string ppush = pin->GetString("particles","pusher");
  if (ppush.compare("drift") == 0) {
    pusher = ParticlesPusher::drift;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle pusher must be specified in <particles> block" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  int ndim=1;
  if (pmy_pack->pmesh->multi_d) {ndim++;}
  if (pmy_pack->pmesh->three_d) {ndim++;}
  Kokkos::realloc(prtcl_pos, nparticles_thispack, ndim);
  Kokkos::realloc(prtcl_vel, nparticles_thispack, ndim);
  Kokkos::realloc(prtcl_gid, nparticles_thispack, ndim);

  // allocate boundary buffers for conserved (cell-centered) variables

}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
}

} // namespace particles