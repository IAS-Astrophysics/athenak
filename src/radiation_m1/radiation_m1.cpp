//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.cpp
//  \brief implementation of Grey M1 radiation class

#include "radiation_m1/radiation_m1.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include "Kokkos_SIMD_Common_Math.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace radiationm1 {
RadiationM1::RadiationM1(MeshBlockPack *ppack, ParameterInput *pin)
    : pmy_pack(ppack),
      u0("cons", 1, 1, 1, 1, 1),
      coarse_u0("ccons", 1, 1, 1, 1, 1),
      u1("cons1", 1, 1, 1, 1, 1),
      eta_0("eta_0", 1, 1, 1, 1, 1),
      abs_0("abs_0", 1, 1, 1, 1, 1),
      eta_1("eta_1", 1, 1, 1, 1, 1),
      abs_1("abs_1", 1, 1, 1, 1, 1),
      scat_1("scat_1", 1, 1, 1, 1, 1),
      chi("chi", 1, 1, 1, 1, 1),
      uflx("uflx", 1, 1, 1, 1, 1) {
  // parameters
  params.gr_sources = pin->GetOrAddBoolean("radiation_m1", "gr_sources", true);
  params.matter_sources = pin->GetOrAddBoolean("radiation_m1", "matter_sources", false);
  params.theta_limiter = pin->GetOrAddBoolean("radiation_m1", "theta_limiter", true);
  params.closure_epsilon = pin->GetOrAddReal("radiation_m1", "closure_epsilon", 1e-5);
  params.closure_maxiter = pin->GetOrAddInteger("radiation_m1", "closure_maxiter", 64);
  params.inv_closure_epsilon =
      pin->GetOrAddReal("radiation_m1", "inv_closure_epsilon", 1e-15);
  params.inv_closure_maxiter =
      pin->GetOrAddInteger("radiation_m1", "inv_closure_maxiter", 64);
  params.rad_N_floor = pin->GetOrAddReal("radiation_m1", "rad_E_floor", 1e-20);
  params.rad_E_floor = pin->GetOrAddReal("radiation_m1", "rad_E_floor", 1e-30);
  params.rad_eps = pin->GetOrAddReal("radiation_m1", "rad_eps", 1e-5);
  params.source_epsabs = pin->GetOrAddReal("radiation_m1", "source_epsabs", 1e-15);
  params.source_epsrel = pin->GetOrAddReal("radiation_m1", "source_epsrel", 1e-5);
  params.source_maxiter = pin->GetOrAddInteger("radiation_m1", "source_maxiter", 64);
  params.source_Ye_min = pin->GetOrAddReal("radiation_m1", "source_Ye_min", 0);
  params.source_Ye_max = pin->GetOrAddReal("radiation_m1", "source_Ye_max", 0.6);
  params.source_thick_limit =
      pin->GetOrAddReal("radiation_m1", "source_thick_limit", 20.);
  params.source_therm_limit =
      pin->GetOrAddReal("radiation_m1", "source_therm_limit", -1.);
  params.source_scat_limit = pin->GetOrAddReal("radiation_m1", "source_scat_limit", -1.);

  nspecies = pin->GetOrAddInteger("radiation_m1", "num_species", 1);
  params.source_limiter = pin->GetOrAddReal("radiation_m1", "source_limiter", 0.5);
  params.beam_sources = pin->GetOrAddBoolean("radiation_m1", "beam_sources", false);

  std::string closure_fun = pin->GetOrAddString("radiation_m1", "closure_fun", "minerbo");
  if (closure_fun == "minerbo") {
    params.closure_type = Minerbo;
  } else if (closure_fun == "eddington") {
    params.closure_type = Eddington;
  } else if (closure_fun == "thin") {
    params.closure_type = Thin;
  } else {
    std::cerr << "Error: Unknown choice for closure: " << closure_fun << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string src_update = pin->GetOrAddString("radiation_m1", "src_update", "explicit");
  if (src_update == "explicit") {
    params.src_update = Explicit;
  } else if (src_update == "implicit") {
    params.src_update = Implicit;
  } else {
    std::cerr << "Error: Unknown src_update: " << src_update << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string opacity_type = pin->GetOrAddString("radiation_m1", "opacity_type", "toy");
  if (opacity_type == "toy") {
    params.opacity_type = Toy;
  } else if (opacity_type == "bns-nurates") {
    params.opacity_type = BnsNurates;
  } else if (opacity_type == "none") {
    params.opacity_type = None;
  } else {
    std::cerr << "Error: Unknown opacity_type: " << opacity_type << std::endl;
    exit(EXIT_FAILURE);
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

  // allocate velocity and Eddington factor
  Kokkos::realloc(u_mu_data, nmb, 4, ncells3, ncells2, ncells1);
  u_mu.InitWithShallowSlice(u_mu_data, 0, 3);
  Kokkos::realloc(chi, nmb, nspecies, ncells3, ncells2, ncells1);

  // allocate opacities
  Kokkos::realloc(eta_0, nmb, nspecies, ncells3, ncells2, ncells1);
  Kokkos::realloc(abs_0, nmb, nspecies, ncells3, ncells2, ncells1);
  Kokkos::realloc(eta_1, nmb, nspecies, ncells3, ncells2, ncells1);
  Kokkos::realloc(abs_1, nmb, nspecies, ncells3, ncells2, ncells1);
  Kokkos::realloc(scat_1, nmb, nspecies, ncells3, ncells2, ncells1);

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
}  // namespace radiationm1