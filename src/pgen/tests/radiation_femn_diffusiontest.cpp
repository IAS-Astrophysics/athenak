//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_diffusiontest.cpp
//! \brief the diffusion test problem in a moving medium

// C++ headers
#include <iostream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"

void ProblemGenerator::RadiationFEMNDiffusiontest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The radiation FEM_N grid test can only be run with radiation-femn, but no " << std::endl
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pmesh->multi_d || pmbp->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The radiation FEM_N grid test can only be run in 1d!" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  int npts1 = pmbp->pradfemn->num_points_total - 1;

  Real omega = 0.03; //  Eqn. (58) of Garrett & Hauck 2013 (DOI: 10.1080/00411450.2014.910226)
  auto &f0_ = pmbp->pradfemn->f0;

  std::cout << "Setting up initial data for the distribution function for the diffusion test ..." << std::endl;
  if (!pmbp->pradfemn->fpn) {
    par_for("pgen_diffusiontest_radiation_femn", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, npts1, ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int A, int k, int j, int i) {

              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              f0_(m, A, k, j, i) = exp(-9. * x1 * x1);

            });
  } else {
    par_for("pgen_diffusiontest_radiation_fpn", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {

              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              f0_(m, 0, k, j, i) = 2. * sqrt(M_PI) * exp(-9. * x1 * x1);

            });
  }

  auto &u_mu_ = pmbp->pradfemn->u_mu;
  Real fluid_vel = pin->GetOrAddReal("radiation-femn", "fluid_velocity", 0.5);

  std::cout << "Setting up velocities for the diffusion test ... " << std::endl;
  par_for("pgen_diffusiontest_radiation_femn_velocity", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {

            u_mu_(m, 1, k, j, i) = sqrt(1. + fluid_vel * fluid_vel);
            u_mu_(m, 1, k, j, i) = fluid_vel;
            u_mu_(m, 2, k, j, i) = 0;
            u_mu_(m, 3, k, j, i) = 0;
          });
  return;
}
