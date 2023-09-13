//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.cpp
//  \brief implementation of ADM class
#include <algorithm>

#include "adm/adm.hpp"
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "z4c/z4c.hpp"

namespace adm {
char const * const ADM::ADM_names[ADM::nadm] = {
  "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
  "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
  "adm_psi4",
  "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
ADM::ADM(MeshBlockPack *ppack, ParameterInput *pin):
  pmy_pack(ppack),
  u_adm("u_adm",1,1,1,1,1) {
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  if (pmy_pack->pz4c == nullptr) {
    Kokkos::realloc(u_adm, nmb, nadm, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_ALPHA);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_BETAX, I_ADM_BETAZ);
  } else {
    // Lapse and shift are stored in the Z4c class
    z4c::Z4c * pz4c = pmy_pack->pz4c;
    Kokkos::realloc(u_adm, nmb, nadm - 4, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_ALPHA);
    adm.beta_u.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_BETAX, pz4c->I_Z4C_BETAZ);
  }
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_PSI4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_GXX, I_ADM_GZZ);
  adm.vK_dd.InitWithShallowSlice(u_adm, I_ADM_KXX, I_ADM_KZZ);
}

//----------------------------------------------------------------------------------------
// destructor
ADM::~ADM() {}

} // namespace adm
