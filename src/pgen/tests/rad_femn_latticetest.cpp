//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_linetest.cpp
//! \brief the 2d linetest problem with FEM_N/FP_N/M1

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

void ProblemGenerator::RadiationFEMNLatticetest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 2d line source problem generator can only be run with radiation-femn, but no "
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pmesh->two_d == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 2d line source problem generator can only be run with two dimensions, but parfile"
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
              << "The 2d lattice source problem generator can only be run with one energy bin!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (pmbp->pradfemn->rad_source == false) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The 2d lattice source problem generator needs sources!" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &eta_ = pmbp->pradfemn->eta;
  auto &kappa_a_ = pmbp->pradfemn->kappa_a;
  auto &kappa_s_ = pmbp->pradfemn->kappa_s;
  auto Ven = (1. / 3.) * (pow(pmbp->pradfemn->energy_grid(1), 3) - pow(pmbp->pradfemn->energy_grid(0), 3));

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

            if (x1 >= 3 && x1 <= 4 && x2 >= 3 && x2 <= 4) {
              eta_(m, k, j, i) = Ven * 1. / (4. * M_PI);
            }

            if ((x1 >= 1 && x1 <= 2 && x2 >= 1 && x2 <= 2) || (x1 >= 3 && x1 <= 4 && x2 >= 1 && x2 <= 2) || (x1 >= 5 && x1 <= 6 && x2 >= 1 && x2 <= 2) ||
                (x1 >= 2 && x1 <= 3 && x2 >= 2 && x2 <= 3) || (x1 >= 4 && x1 <= 5 && x2 >= 2 && x2 <= 3) || (x1 >= 1 && x1 <= 2 && x2 >= 3 && x2 <= 4) ||
                (x1 >= 5 && x1 <= 6 && x2 >= 3 && x2 <= 4) || (x1 >= 2 && x1 <= 3 && x2 >= 4 && x2 <= 5) || (x1 >= 4 && x1 <= 5 && x2 >= 4 && x2 <= 5) ||
                (x1 >= 1 && x1 <= 2 && x2 >= 5 && x2 <= 6) || (x1 >= 5 && x1 <= 6 && x2 >= 5 && x2 <= 6)) {

              kappa_a_(m, k, j, i) = Ven * 10.;
            } else {
              kappa_s_(m, k, j, i) = Ven * 1.;
            }
          });

  return;
}