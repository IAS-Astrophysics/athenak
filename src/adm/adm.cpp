//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.cpp
//  \brief implementation of ADM class

#include "adm/adm.hpp"
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "z4c/z4c.hpp"

namespace adm {
char const * const ADM::ADM_names[ADM::N_ADM] = {
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
  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  if (nullptr == pmy_pack->pz4c) {
    Kokkos::realloc(u_adm, nmb, N_ADM, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_alpha);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_betax, I_ADM_betaz);
  } else {
    // Lapse and shift are stored in the Z4c class
    z4c::Z4c * pz4c = pmy_pack->pz4c;
    Kokkos::realloc(u_adm, nmb, N_ADM - 4, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(pz4c->u0, pz4c->IZ4CALPHA);
    adm.beta_u.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_BETAX, pz4c->I_Z4C_BETAZ);
  }
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_psi4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_gxx, I_ADM_gzz);
  adm.K_dd.InitWithShallowSlice(u_adm, I_ADM_Kxx, I_ADM_Kzz);
}

//----------------------------------------------------------------------------------------
// destructor
ADM::~ADM() {}

} // namespace adm
