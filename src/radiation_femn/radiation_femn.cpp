//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.cpp
//  \brief implementation of the radiation FEM_N class constructor and other functions

#include <string>
#include <cmath>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "units/units.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_linalg.hpp"
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
    ftemp("ftemp", 1, 1, 1, 1, 1),
    energy_grid("energy_grid", 1),
    angular_grid("angular_grid", 1, 1),
    angular_grid_cartesian("angular_grid_cartesian", 1, 1),
    triangle_information("triangle_information", 1, 1),
    mass_matrix("mm", 1, 1),
    stiffness_matrix_x("sx", 1, 1),
    stiffness_matrix_y("sy", 1, 1),
    stiffness_matrix_z("sz", 1, 1),
    P_matrix("PmuAB", 1, 1, 1),
    Pmod_matrix("PmodmuAB", 1, 1, 1),
    G_matrix("GnumuiAB", 1, 1, 1, 1, 1),
    F_matrix("FnumuiAB", 1, 1, 1, 1, 1),
    Q_matrix("QmuhatA", 1, 1),
    beam_source_1_vals("beam_source_1_vals", 1),
    beam_source_2_vals("beam_source_2_vals", 1),
    e_source("e_source", 1),
    e_source_nominv("e_source_nominv", 1),
    S_source("S_source", 1, 1),
    W_matrix("W_matrix", 1, 1),
    eta("eta", 1, 1, 1, 1),
    kappa_a("kappa_a", 1, 1, 1, 1),
    kappa_s("kappa_s", 1, 1, 1, 1) {

  // -----------------------------------------------------------------------------------
  // set essential parameters from par file and allocate memory to energy, angular grids

  mass_lumping = pin->GetOrAddInteger("radiation-femn", "mass_lumping", 1) == 1;

  limiter_dg = pin->GetOrAddString("radiation-femn", "limiter_dg", "minmod2");      // limiter for DG (default:sawtooth-free minmod2)
  fpn = pin->GetOrAddInteger("radiation-femn", "fpn", 0) == 1;                      // fpn switch (0: use FEM_N, 1: use FP_N) (default: 0)

  num_energy_bins = pin->GetOrAddInteger("radiation-femn", "num_energy_bins", 1);   // number of energy bins (default: 1)
  energy_max = pin->GetOrAddReal("radiation-femn", "energy_max", 1);                // maximum value of energy (default: 1)

  num_species = pin->GetOrAddInteger("radiation-femn", "num_species", 1);       // number of neutrino species (default: 1)

  // set up energy grid
  HostArray1D<Real> temp_array;
  Kokkos::realloc(energy_grid, num_energy_bins + 1);
  Kokkos::realloc(temp_array, num_energy_bins + 1);
  for (int i = 0; i < num_energy_bins + 1; i++) {
    temp_array(i) = i * energy_max / Real(num_energy_bins);
  }
  Kokkos::deep_copy(energy_grid, temp_array);

  if (!fpn) {   // parameters for FEM_N
    lmax = -42;                                                                     // maximum value of l in spherical harmonics expansion (redundant: set to -42)
    refinement_level = pin->GetOrAddInteger("radiation-femn", "num_refinement", 0); // refinement level for geodesic grid (default: 0)

    // compute number of points, edges and triangles from refinement level (Note: num_ref can change but never refinement_level)
    num_ref = refinement_level;
    num_points = 12 * pow(4, refinement_level);
    if (refinement_level != 0) {
      for (int i = 0; i < refinement_level; i++) {
        num_points -= 6 * pow(4, i);
      }
    }
    num_edges = 3 * (num_points - 2);
    num_triangles = 2 * (num_points - 2);

    basis = pin->GetOrAddInteger("radiation-femn", "basis", 1);                     // choice of FEM_N basis (default: 1, that is overlapping tent)
    filter_sigma_eff = -42;                                                         // redundant: set to -42
    limiter_fem = pin->GetOrAddString("radiation-femn", "limiter_fem", "clp");      // limiter for angle (default: clipping limiter)
  } else {  // parameters for FP_N
    lmax = pin->GetOrAddInteger("radiation-femn", "lmax", 3);                       // maximum value of l in spherical harmonic expansion (default: FP3)
    refinement_level = -42;                                                         // redundant: set to -42
    num_ref = refinement_level;                                                     // redundant: set to -42
    num_points = (lmax + 1) * (lmax + 1);                                           // total number of (l,m) modes
    num_edges = -42;                                                                // redundant: set to -42
    num_triangles = -42;                                                            // redundant: set to -42
    basis = -42;                                                                    // redundant: set to -42
    filter_sigma_eff = pin->GetOrAddInteger("radiation-femn", "filter_opacity", 0); // filter opacity for FP_N (default: no filter)
    limiter_fem = "-42";                                                            // redundant: set to -42
  }

  num_points_total = num_species * num_energy_bins * num_points;

  m1_flag = pin->GetOrAddBoolean("radiation-femn", "m1", false);
  rad_source = pin->GetOrAddBoolean("radiation-femn", "source_terms", false);           // switch for sources (default: false)

  beam_source = pin->GetOrAddBoolean("radiation-femn", "beam_sources", false);     // switch for beam sources (default: false)
  num_beams = pin->GetOrAddInteger("radiation-femn", "num_beam_sources", 0);
  beam_source_1_y1 = pin->GetOrAddReal("radiation-femn", "beam_source_1_y1", -42.);
  beam_source_1_y2 = pin->GetOrAddReal("radiation-femn", "beam_source_1_y2", -42.);
  beam_source_1_phi = pin->GetOrAddReal("radiation-femn", "beam_source_1_phi", -42.);
  beam_source_1_theta = pin->GetOrAddReal("radiation-femn", "beam_source_1_theta", -42.);
  beam_source_2_y1 = pin->GetOrAddReal("radiation-femn", "beam_source_2_y1", -42.);
  beam_source_2_y2 = pin->GetOrAddReal("radiation-femn", "beam_source_2_y2", -42.);
  beam_source_2_phi = pin->GetOrAddReal("radiation-femn", "beam_source_2_phi", -42.);
  beam_source_2_theta = pin->GetOrAddReal("radiation-femn", "beam_source_2_theta", -42.);

  Kokkos::realloc(mass_matrix, num_points, num_points);
  Kokkos::realloc(mass_matrix_inv, num_points, num_points);
  Kokkos::realloc(stiffness_matrix_x, num_points, num_points);
  Kokkos::realloc(stiffness_matrix_y, num_points, num_points);
  Kokkos::realloc(stiffness_matrix_z, num_points, num_points);
  Kokkos::realloc(P_matrix, 4, num_points, num_points);
  Kokkos::realloc(Pmod_matrix, 4, num_points, num_points);
  Kokkos::realloc(G_matrix, 4, 4, 3, num_points, num_points);
  Kokkos::realloc(F_matrix, 4, 4, 3, num_points, num_points);
  Kokkos::realloc(Q_matrix, 4, num_points);
  Kokkos::realloc(e_source, num_points);
  Kokkos::realloc(e_source_nominv, num_points);
  Kokkos::realloc(S_source, num_points, num_points);
  Kokkos::realloc(angular_grid, num_points, 2);

  if (!fpn) {   // populate arrays for FEM_N
    Kokkos::realloc(angular_grid_cartesian, num_points, 3);
    Kokkos::realloc(triangle_information, num_triangles, 3);
    scheme_num_points = pin->GetOrAddInteger("radiation-femn", "quad_scheme_num_points", 453);  // number of points in numerical integration scheme (default: 453)
    scheme_name = pin->GetOrAddString("radiation-femn", "quad_scheme_name", "xiao_gimbutas");   // type of quadrature (xioa_gimbutas: default or vioreanu_rokhlin)

    // quadrature check from par file
    if (!(scheme_name == "xiao_gimbutas" || scheme_name == "vioreanu_rokhlin")) {
      std::cout << "Quadrature scheme cannot be " + scheme_name + " for FEM_N" << std::endl;
      std::cout << "Use xiao_gimbutas or vioreanu_rokhlin instead!" << std::endl;
      exit(EXIT_FAILURE);
    }

    radiationfemn::LoadQuadrature(scheme_name, scheme_num_points, scheme_weights, scheme_points); // populate quadrature from disk
    this->LoadFEMNMatrices(); // populate all matrices with FEM_N data

  } else {    // populate arrays for FP_N
    scheme_num_points = pin->GetOrAddInteger("radiation-femn", "quad_scheme_num_points", 2702);   // number of points in numerical integration scheme (default: 2702)
    scheme_name = pin->GetOrAddString("radiation-femn", "quad_scheme_name", "lebedev");           // type of quadrature (lebedev: default)

    // quadrature check from par file
    if (!(scheme_name == "lebedev" || scheme_name == "gauss_legendre")) {
      std::cout << "Quadrature scheme cannot be " + scheme_name + " for FP_N" << std::endl;
      std::cout << "Use lebedev or gauss_legendre instead!" << std::endl;
      exit(EXIT_FAILURE);
    }

    radiationfemn::LoadQuadrature(scheme_name, scheme_num_points, scheme_weights, scheme_points); // populate quadrature from disk
    this->LoadFPNMatrices();                                                                      // populate all matrices with FP_N data
  }

  // compute lumped mass matrix
  if (mass_lumping) {
    std::cout << "Mass Lumping switched on ..." << std::endl;
    radiationfemn::MatLumping(mass_matrix);
  }

  this->ComputeMassInverse();

  if (m1_flag) {
    if (lmax != 2) {
      std::cout << " To run M1 you must have FP_N on with lmax = 2!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // compute P, Pmod matrices, source matrices
  this->ComputePMatrices();
  this->ComputeSourceMatrices();

  // --------------------------------------------------------------------------------------------------------------------------
  // allocate memory for all other variables

  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;

  // tetrad and fluid quantities
  Kokkos::realloc(g_dd, nmb, 4, 4, ncells3, ncells2, ncells1);        // 4-metric from GR
  Kokkos::realloc(sqrt_det_g, nmb, ncells3, ncells2, ncells1);        // sqrt(-det(4-metric))
  Kokkos::realloc(u_mu, nmb, 4, ncells3, ncells2, ncells1);           // u^mu: fluid velocity in lab frame
  Kokkos::realloc(L_mu_muhat0, nmb, 4, 4, ncells3, ncells2, ncells1); // tetrad L^mu_muhat
  Kokkos::realloc(L_mu_muhat1, nmb, 4, 4, ncells3, ncells2, ncells1); // tetrad L^mu_muhat

  // Hardcode metric fluid quantities @TODO: change this later
  this->InitializeMetricFluid();

  // state vector and fluxes
  Kokkos::realloc(f0, nmb, num_points_total, ncells3, ncells2, ncells1);        // distribution function
  Kokkos::realloc(f1, nmb, num_points_total, ncells3, ncells2, ncells1);        // distribution function
  Kokkos::realloc(iflx.x1f, nmb, num_points_total, ncells3, ncells2, ncells1);  // spatial flux (x)
  Kokkos::realloc(iflx.x2f, nmb, num_points_total, ncells3, ncells2, ncells1);  // spatial flux (y)
  Kokkos::realloc(iflx.x3f, nmb, num_points_total, ncells3, ncells2, ncells1);  // spatial flux (z)
  Kokkos::realloc(ftemp, nmb, num_points_total, ncells3, ncells2, ncells1);     // distribution function (temp storage)

  // reallocate allocate memory for evolved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2 * (indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2 * (indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2 * (indcs.ng)) : 1;
    Kokkos::realloc(coarse_f0, nmb, num_points_total, nccells3, nccells2, nccells1);
  }

  // beam sources
  if (beam_source && fpn) {
    Kokkos::realloc(beam_source_1_vals, num_points);
    Kokkos::realloc(beam_source_2_vals, num_points);
    if (m1_flag) {
      this->InitializeBeamsSourcesM1();
    } else {
      this->InitializeBeamsSourcesFPN();
    }
  }

  if (beam_source && !fpn) {
    Kokkos::realloc(beam_source_1_vals, num_points);
    Kokkos::realloc(beam_source_2_vals, num_points);
    this->InitializeBeamsSourcesFEMN();
  }

  // sources
  Kokkos::realloc(eta, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(kappa_a, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(kappa_s, nmb, ncells3, ncells2, ncells1);

  // allocate boundary buffers for cell-centered variables
  pbval_f = new BoundaryValuesCC(ppack, pin, false);
  pbval_f->InitializeBuffers(num_points);

}

//----------------------------------------------------------------------------------------------
// class constructor, initialize parameters and data structures

RadiationFEMN::~RadiationFEMN() {
  delete pbval_f;
}

} // namespace radiationfemn
