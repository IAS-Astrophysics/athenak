//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tmunu.hpp
//! \brief implementation of Tmunu class
#include <algorithm>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"
#include "z4c/tmunu.hpp"
#include "mesh/mesh.hpp"

char const * const Tmunu::Tmunu_names[Tmunu::N_Tmunu] = {
  "tmunu_Sxx", "tmunu_Sxy", "tmunu_Sxz", "tmunu_Syy", "tmunu_Syz", "tmunu_Szz",
  "tmunu_E", "tmunu_Sx", "tmunu_Sy", "tmunu_Sz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
Tmunu::Tmunu(MeshBlockPack *ppack, ParameterInput *pin):
  pmy_pack(ppack),
  u_tmunu("u_tmunu",1,1,1,1,1) {
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(u_tmunu, nmb, N_Tmunu, ncells3, ncells2, ncells1);
  tmunu.S_dd.InitWithShallowSlice(u_tmunu, I_Tmunu_Sxx, I_Tmunu_Szz);
  tmunu.E.InitWithShallowSlice(u_tmunu, I_Tmunu_E);
  tmunu.S_d.InitWithShallowSlice(u_tmunu, I_Tmunu_Sx, I_Tmunu_Sz);
}

Tmunu::~Tmunu() {}
