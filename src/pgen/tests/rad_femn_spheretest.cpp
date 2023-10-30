//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_spheretest.cpp
//! \brief the 3d problem with FEM_N

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

void ProblemGenerator::RadiationFEMNSpheretest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 3d sphere test problem generator can only be run with radiation-femn, but no "
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pmesh->three_d == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 3d sphere test problem generator can only be run with three dimensions, but parfile"
              << "grid setup is not in 2d" << std::endl;
    exit(EXIT_FAILURE);
  }


  // capture var pmy_mesh_->mb_indcs;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  if (pmbp->pradfemn->num_energy_bins != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 3d sphere test problem generator can only be run with one energy bin!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradfemn->rad_source == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 3d sphere test problem generator needs sources!" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &eta_ = pmbp->pradfemn->eta;
  auto &kappa_a_ = pmbp->pradfemn->kappa_a;
  auto &kappa_s_ = pmbp->pradfemn->kappa_s;

    par_for("pgen_linetest_radiation_femn", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {

              Real &x1min = size.d_view(m).x1min;
              Real &x1max = size.d_view(m).x1max;
              int nx1 = indcs.nx1;
              Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

              Real &x2min = size.d_view(m).x2min;
              Real &x2max = size.d_view(m).x2max;
              int nx2 = indcs.nx2;
              Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

              Real &x3min = size.d_view(m).x3min;
              Real &x3max = size.d_view(m).x3max;
              int nx3 = indcs.nx3;
              Real x3 = CellCenterX(k - ks, nx3, x3min, x3max);

              if (x1 * x1 + x2 * x2 + x3 * x3 <= 1.0) { // Inside the sphere
                  kappa_a_(m, k, j, i) = 10.0;  
                  eta_(m, k, j, i) = 10.0;
              } else {
                  kappa_a_(m, k, j, i) = 0;  // Outside the sphere
              }
            });

  return;
}