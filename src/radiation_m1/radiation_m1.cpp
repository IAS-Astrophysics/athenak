//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.cpp
//  \brief implementation of grey M1 radiation class

#include "radiation_m1/radiation_m1.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

#ifdef ENABLE_NURATES
#include "bns_nurates/include/integration.hpp"
#endif

namespace radiationm1 {
RadiationM1::RadiationM1(MeshBlockPack *ppack, ParameterInput *pin)
    : pmy_pack(ppack),
      u0("rad_cons", 1, 1, 1, 1, 1),
      coarse_u0("rad_ccons", 1, 1, 1, 1, 1),
      u1("rad_cons1", 1, 1, 1, 1, 1),
      eta_0("eta_0", 1, 1, 1, 1, 1),
      abs_0("abs_0", 1, 1, 1, 1, 1),
      eta_1("eta_1", 1, 1, 1, 1, 1),
      abs_1("abs_1", 1, 1, 1, 1, 1),
      scat_1("scat_1", 1, 1, 1, 1, 1),
      chi("chi", 1, 1, 1, 1, 1),
      uflx("rad_uflx", 1, 1, 1, 1, 1) {
  // set up parameters and flags
  ismhd = pin->DoesBlockExist("mhd");
  nspecies = M1_TOTAL_NUM_SPECIES;

  params.gr_sources = pin->GetOrAddBoolean("radiation_m1", "gr_sources", true);
  params.matter_sources = pin->GetOrAddBoolean("radiation_m1", "matter_sources", false);
  params.backreact = pin->GetOrAddBoolean("radiation_m1", "backreact", true);
  params.theta_limiter = pin->GetOrAddBoolean("radiation_m1", "theta_limiter", false);
  params.closure_epsilon = pin->GetOrAddReal("radiation_m1", "closure_epsilon", 1e-14);
  params.closure_maxiter = pin->GetOrAddInteger("radiation_m1", "closure_maxiter", 164);
  params.inv_closure_epsilon =
      pin->GetOrAddReal("radiation_m1", "inv_closure_epsilon", 1e-15);
  params.inv_closure_maxiter =
      pin->GetOrAddInteger("radiation_m1", "inv_closure_maxiter", 64);
  params.rad_N_floor = pin->GetOrAddReal("radiation_m1", "rad_N_floor", 1e-14);
  params.rad_E_floor = pin->GetOrAddReal("radiation_m1", "rad_E_floor", 1e-14);
  params.rad_eps = pin->GetOrAddReal("radiation_m1", "rad_eps", 1e-14);
  params.source_epsabs = pin->GetOrAddReal("radiation_m1", "source_epsabs", 1e-15);
  params.source_epsrel = pin->GetOrAddReal("radiation_m1", "source_epsrel", 1e-15);
  params.source_maxiter = pin->GetOrAddInteger("radiation_m1", "source_maxiter", 164);
  params.source_Ye_min = pin->GetOrAddReal("radiation_m1", "source_Ye_min", 0);
  params.source_Ye_max = pin->GetOrAddReal("radiation_m1", "source_Ye_max", 0.6);
  params.source_thick_limit =
      pin->GetOrAddReal("radiation_m1", "source_thick_limit", 20.);
  params.source_therm_limit =
      pin->GetOrAddReal("radiation_m1", "source_therm_limit", -1.);
  params.source_scat_limit = pin->GetOrAddReal("radiation_m1", "source_scat_limit", -1.);
  params.minmod_theta = pin->GetOrAddReal("radiation_m1", "minmod_theta", 1);
  params.source_limiter = pin->GetOrAddReal("radiation_m1", "source_limiter", 0.5);
  params.beam_sources = pin->GetOrAddBoolean("radiation_m1", "beam_sources", false);

  // set closure (default: minerbo)
  std::string closure_fun = pin->GetOrAddString("radiation_m1", "closure_fun", "minerbo");
  if (closure_fun == "minerbo") {
    params.closure_type = Minerbo;
  } else if (closure_fun == "eddington") {
    params.closure_type = Eddington;
  } else if (closure_fun == "thin") {
    params.closure_type = Thin;
  } else if (closure_fun == "kershaw") {
    params.closure_type = Kershaw;
  } else {
    std::cerr << "Error: Unknown choice for closure: " << closure_fun << std::endl;
    exit(EXIT_FAILURE);
  }

  // set source update strategy (default: implicit)
  std::string src_update = pin->GetOrAddString("radiation_m1", "src_update", "implicit");
  if (src_update == "explicit") {
    params.src_update = Explicit;
  } else if (src_update == "implicit") {
    params.src_update = Implicit;
  } else {
    std::cerr << "Error: Unknown src_update: " << src_update << std::endl;
    exit(EXIT_FAILURE);
  }

  // select opacities (default: none)
  std::string opacity_type = pin->GetOrAddString("radiation_m1", "opacity_type", "none");
  if (opacity_type == "toy") {
    params.opacity_type = Toy;
  } else if (opacity_type == "bns-nurates") {
#ifdef ENABLE_NURATES
    params.opacity_type = BnsNurates;

    nurates_params.quad_nx = pin->GetOrAddInteger("bns_nurates", "nurates_quad_nx", 10);
    nurates_params.opacity_tau_trap =
        pin->GetOrAddReal("bns_nurates", "opacity_tau_trap", 1.0);
    nurates_params.opacity_tau_delta =
        pin->GetOrAddReal("bns_nurate", "opacity_tau_delta", 1.0);
    nurates_params.opacity_corr_fac_max =
        pin->GetOrAddReal("bns_nurates", "opacity_corr_fac_max", 3.0);
    nurates_params.nb_min_cgs = pin->GetOrAddReal("bns_nurates", "rho_min_cgs", 0.);
    nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates", "temp_min_mev", 0.);

    nurates_params.use_abs_em = pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true);
    nurates_params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true);
    nurates_params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true);
    nurates_params.use_iso = pin->GetOrAddBoolean("bns_nurates", "use_iso", true);
    nurates_params.use_inelastic_scatt =
        pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", true);
    nurates_params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", true);
    nurates_params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", true);
    nurates_params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", true);
    nurates_params.use_dm_eff = pin->GetOrAddBoolean("bns_nurates", "use_dm_eff", true);
    nurates_params.use_equilibrium_distribution =
        pin->GetOrAddBoolean("bns_nurates", "use_equilibrium_distribution", false);
    nurates_params.use_kirchhoff_law =
        pin->GetOrAddBoolean("bns_nurates", "use_kirchoff_law", false);
    nurates_params.use_NN_medium_corr =
        pin->GetOrAddBoolean("bns_nurates", "use_NN_medium_corr", true);
    nurates_params.neglect_blocking =
        pin->GetOrAddBoolean("bns_nurates", "neglect_blocking", false);
    nurates_params.use_decay = pin->GetOrAddBoolean("bns_nurates", "use_decay", false);
    nurates_params.use_BRT_brem =
        pin->GetOrAddBoolean("bns_nurates", "use_BRT_brem", false);

    nurates_params.quadrature.nx = nurates_params.quad_nx;
    nurates_params.quadrature.dim = 1;
    nurates_params.quadrature.type = kGauleg;
    nurates_params.quadrature.x1 = 0.;
    nurates_params.quadrature.x2 = 1.;
    GaussLegendre(&nurates_params.quadrature);
#else
    std::cerr << "Error: To use BNS_NURATES, executable must be compiled with "
                 "-DAthena_ENABLE_NURATES=ON"
              << std::endl;
    exit(EXIT_FAILURE);
#endif
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

  // allocate Eddington factor
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

  // allocate 4-velocity if not using mhd
  if (!ismhd) {
    Kokkos::realloc(u_mu_data, nmb, 4, ncells3, ncells2, ncells1);
    u_mu.InitWithShallowSlice(u_mu_data, 0, 3);
  }

  // allocate boundary buffers for evolved (cell-centered) variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers(nvarstot);
}

//----------------------------------------------------------------------------------------
// destructor
RadiationM1::~RadiationM1() { delete pbval_u; }
}  // namespace radiationm1