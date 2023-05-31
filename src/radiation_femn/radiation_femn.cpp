//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.cpp
//  \brief implementation of the radiation FEM_N class constructor and other functions

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"

namespace radiationfemn {

//----------------------------------------------------------------------------------------------
// class constructor, initialize parameters and data structures
RadiationFEMN::RadiationFEMN(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    g_dd("spatial_metric", 1, 1, 1, 1, 1, 1),
    sqrt_det_g("square_root_det_g", 1, 1, 1, 1),
    u_mu("fluid_vel_lab", 1, 1, 1, 1, 1),
    L_mu_muhat0("L^mu_muhat0", 1, 1, 1, 1, 1, 1),
    L_mu_muhat1("L^mu_muhat1", 1, 1, 1, 1, 1, 1),
    scheme_points("scheme_points", 1, 1),
    scheme_weights("scheme_weights", 1),
    f0("f0", 1, 1, 1, 1, 1),
    coarse_f0("ci0", 1, 1, 1, 1, 1),
    f1("f1", 1, 1, 1, 1, 1),
    iflx("iflx", 1, 1, 1, 1, 1),
    ftemp("ftemp", 1, 1, 1, 1, 1, 1),
    etemp0("etemp0", 1, 1, 1, 1),
    etemp1("etemp1", 1, 1, 1, 1),
    energy_grid("energy_grid", 1),
    angular_grid("angular_grid", 1, 1),
    mass_matrix("mm", 1, 1),
    stiffness_matrix_x("sx", 1, 1),
    stiffness_matrix_y("sy", 1, 1),
    stiffness_matrix_z("sz", 1, 1),
    P_matrix("PmuAB", 1, 1, 1),
    G_matrix("GnumuiAB", 1, 1, 1, 1, 1),
    F_matrix("FnumuiAB", 1, 1, 1, 1, 1),
    e_source("e_source", 1),
    S_source("S_source", 1, 1),
    W_matrix("W_matrix", 1, 1),
    eta("eta", 1, 1, 1, 1, 1),
    kappa_a("kappa_a", 1, 1, 1, 1, 1),
    kappa_s("kappa_s", 1, 1, 1, 1, 1),
    beam_mask("beam_mask", 1, 1, 1, 1, 1, 1) {

  // ---------------------------------------------------------------------------
  // set up from parfile parameters

  // (1) Set limiters for DG and FP_N
  limiter_dg = pin->GetOrAddString("radiation-femn", "limiter_dg", "minmod2");
  fpn = pin->GetOrAddInteger("radiation-femn", "fpn", 0) == 1;

  // (2) Set up the energy grid from [0, energy_max] with num_energy_bins bins.
  num_energy_bins = pin->GetOrAddInteger("radiation_femn", "num_energy_bins", 1);
  energy_max = pin->GetReal("radiation-femn", "energy_max");
  Kokkos::realloc(energy_grid, num_energy_bins + 1);
  for (size_t i = 0; i < num_energy_bins + 1; i++) {
    energy_grid(i) = i * energy_max / Real(num_energy_bins);
  }

  // (3) Set up FEM_N/FP_N specific parameters (redundant values set to -42)
  if (!fpn) {
    lmax = -42;
    refinement_level = pin->GetOrAddInteger("radiation-femn", "num_refinement", 0);
    num_ref = refinement_level;
    num_points = 12 * pow(4, refinement_level);
    if (refinement_level != 0) {
      for (size_t i = 0; i < refinement_level; i++) {
        num_points -= 6 * pow(4, i);
      }
    }
    num_edges = 3 * (num_points - 2);
    num_triangles = 2 * (num_points - 2);
    basis = pin->GetOrAddInteger("radiation-femn", "basis", 1);
    filter_sigma_eff = -42;
    limiter_fem = pin->GetOrAddString("radiation-femn", "limiter_fem", "clp");
  } else {
    lmax = pin->GetInteger("radiation-femn", "lmax");
    refinement_level = -42;
    num_ref = refinement_level;
    num_points = (lmax + 1) * (lmax + 1);
    num_edges = -42;
    num_triangles = -42;
    basis = -42;
    filter_sigma_eff = pin->GetOrAddInteger("radiation-femn", "filter_opacity", 0);
    limiter_fem = "-42";
  }

  num_points_total = num_energy_bins * num_points;

  // (3) set up source specific parameters
  rad_source = pin->GetOrAddInteger("radiation-femn", "sources", 0) == 1;
  beam_source = pin->GetOrAddInteger("radiation-femn", "beam_sources", 0) == 1;

  // ---------------------------------------------------------------------------
  // allocate memory and populate matrices for the angular variables

  Kokkos::realloc(mass_matrix, num_points, num_points);
  Kokkos::realloc(stiffness_matrix_x, num_points, num_points);
  Kokkos::realloc(stiffness_matrix_y, num_points, num_points);
  Kokkos::realloc(stiffness_matrix_z, num_points, num_points);

  Kokkos::realloc(P_matrix, 4, num_points, num_points);
  Kokkos::realloc(G_matrix, 4, 4, 3, num_points, num_points);
  Kokkos::realloc(F_matrix, 4, 4, 3, num_points, num_points);

  Kokkos::realloc(angular_grid, num_points, 2);

  if (!fpn) {
    scheme_num_points = pin->GetOrAddInteger("radiation-femn", "quad_scheme_num_points", 453);
    scheme_name = pin->GetOrAddString("radiation-femn", "quad_scheme_name", "xiao_gimbutas");

    if (!(scheme_name == "xiao_gimbutas" || scheme_name == "vioreanu_rokhlin")) {
      std::cout << "Quadrature scheme cannot be " + scheme_name + " for FEM_N" << std::endl;
      std::cout << "Use xiao_gimbutas or vioreanu_rokhlin instead!" << std::endl;
      exit(EXIT_FAILURE);
    }

    radiationfemn::LoadQuadrature(scheme_name, scheme_num_points, scheme_weights, scheme_points);
    this->LoadFEMNMatrices();

  } else {
    scheme_num_points = pin->GetOrAddInteger("radiation-femn", "quad_scheme_num_points", 2702);
    scheme_name = pin->GetOrAddString("radiation-femn", "quad_scheme_name", "lebedev");

    if (scheme_name != "lebedev") {
      std::cout << "Quadrature scheme cannot be " + scheme_name + " for FP_N" << std::endl;
      std::cout << "Use lebedev instead!" << std::endl;
      exit(EXIT_FAILURE);
    }

    radiationfemn::LoadQuadrature(scheme_name, scheme_num_points, scheme_weights, scheme_points);
    this->LoadFPNMatrices();
  }

  // --------------------------------------------------------------------------------------------------------------------------
  // allocate memory for evolved variables
  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;

  // tetrad quantities
  Kokkos::realloc(g_dd, nmb, 4, 4, ncells3, ncells2, ncells1);
  Kokkos::realloc(sqrt_det_g, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_mu, nmb, 4, ncells3, ncells2, ncells1);
  Kokkos::realloc(L_mu_muhat0, nmb, 4, 4, ncells3, ncells2, ncells1);
  Kokkos::realloc(L_mu_muhat1, nmb, 4, 4, ncells3, ncells2, ncells1);

  // initialize tetrad
  this->TetradInitialize();

  // state vector and fluxes
  Kokkos::realloc(f0, nmb, num_points_total, ncells3, ncells2, ncells1);
  Kokkos::realloc(f1, nmb, num_points_total, ncells3, ncells2, ncells1);
  Kokkos::realloc(iflx.x1f, nmb, num_points_total, ncells3, ncells2, ncells1);
  Kokkos::realloc(iflx.x2f, nmb, num_points_total, ncells3, ncells2, ncells1);
  Kokkos::realloc(iflx.x3f, nmb, num_points_total, ncells3, ncells2, ncells1);
  Kokkos::realloc(ftemp, nmb, num_points_total, ncells3, ncells2, ncells1);

  // reallocate memory for the temporary intensity matrices if the clipping limiter is on
  if (limiter_fem == "clp") {
    Kokkos::realloc(etemp0, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(etemp1, nmb, ncells3, ncells2, ncells1);
  }

  // reallocate allocate memory for evolved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2 * (indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2 * (indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2 * (indcs.ng)) : 1;
    Kokkos::realloc(coarse_f0, nmb, num_points_total, nccells3, nccells2, nccells1);
  }

  // only do if sources are present
  if (rad_source) {
    //Kokkos::realloc(int_psi, num_points);
    Kokkos::realloc(e_source, num_points);
    Kokkos::realloc(S_source, num_points, num_points);
    Kokkos::realloc(W_matrix, num_points, num_points);
    //  this->CalcIntPsi(); @TODO: fix during sources

    Kokkos::realloc(eta, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(kappa_a, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(kappa_s, nmb, ncells3, ncells2, ncells1);
  }

  if (beam_source) {
    Kokkos::realloc(beam_mask, nmb, num_points, ncells3, ncells2, ncells1);
  }

  // allocate boundary buffers for cell-centered variables
  pbval_f = new BoundaryValuesCC(ppack, pin);
  pbval_f->InitializeBuffers(num_points);
}

//----------------------------------------------------------------------------------------------
// class constructor, initialize parameters and data structures

RadiationFEMN::~RadiationFEMN() {
  delete pbval_f;
}

} // namespace radiationfemn
