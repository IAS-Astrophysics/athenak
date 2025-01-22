//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.cpp
//  \brief implementation for Grey M1 radiation class

#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1.hpp"

namespace radiationm1 {

RadiationM1::RadiationM1(MeshBlockPack *ppack, ParameterInput *pin)
    : pmy_pack(ppack), u0("cons", 1, 1, 1, 1, 1),
      coarse_u0("ccons", 1, 1, 1, 1, 1), u1("cons1", 1, 1, 1, 1, 1),
      uflx("uflx", 1, 1, 1, 1, 1), beam_source_vals("beam_vals", 1) {

  // parameters
  nspecies = pin->GetOrAddInteger("radiation_m1", "num_species", 1);
  source_limiter = pin->GetOrAddInteger("radiation_m1", "source_limiter", 0.5);
  params.rad_E_floor = pin->GetOrAddReal("radiation_m1", "rad_E_floor", 1e-14);
  params.rad_eps = pin->GetOrAddReal("radiation_m1", "rad_eps", 1e-14);
  std::string closure_fun =
      pin->GetOrAddString("radiation_m1", "closure_fun", "minerbo");
  if (closure_fun == "minerbo") {
    params.closure_fun = Minerbo;
  } else if (closure_fun == "eddington") {
    params.closure_fun = Eddington;
  } else {
    params.closure_fun = Thin;
  }

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

  // Evolved variables: E, F_x, F_y, F_z, N (if nspecies > 1)
  nvars = (nspecies > 1) ? 5 : 4;
  nvarstot = nvars * nspecies;

  // allocate memory for evolved variables
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;
  Kokkos::realloc(u0, nmb, nvarstot, ncells3, ncells2, ncells1);

  // allocate memory for evolved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    int n_ccells1 = indcs.cnx1 + 2 * (indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2 * (indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2 * (indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, nvarstot, n_ccells3, n_ccells2, n_ccells1);
  }

  // allocate second registers, fluxes
  Kokkos::realloc(u1, nmb, nvarstot, ncells3, ncells2, ncells1);
  Kokkos::realloc(uflx.x1f, nmb, nvarstot, ncells3, ncells2, ncells1);
  Kokkos::realloc(uflx.x2f, nmb, nvarstot, ncells3, ncells2, ncells1);
  Kokkos::realloc(uflx.x3f, nmb, nvarstot, ncells3, ncells2, ncells1);

  // allocate velocity and lab radiation pressure
  Kokkos::realloc(u_mu_data, nmb, 4, ncells3, ncells2, ncells1);
  u_mu.InitWithShallowSlice(u_mu_data, 0, 3);
  Kokkos::realloc(P_dd, nmb, 6 * nspecies, ncells3, ncells2,
                  ncells1); // stores Pxx, Pxy, Pxz, Pyy, Pyz, Pzz
  // radiation mask
  Kokkos::realloc(radiation_mask, nmb, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(radiation_mask, false);

  // allocate boundary buffers for evolved (cell-centered) variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers(nvars);
}

//----------------------------------------------------------------------------------------
// destructor
RadiationM1::~RadiationM1() { delete pbval_u; }
} // namespace radiationm1