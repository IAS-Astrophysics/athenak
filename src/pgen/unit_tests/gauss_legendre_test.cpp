//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>
#include <cctype>
#include <random>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "utils/spherical_harm.hpp"

using u32    = uint_least32_t;
using engine = std::mt19937;

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  // ADMOnePuncture(pmbp, pin);
  int ntheta = pin->GetOrAddInteger("problem", "ntheta", 16);

  GaussLegendreGrid grid(pmbp, ntheta, 1);

  // test that the cross integral of spherical harmonics are delta functions.
  // First initialize 10 random pairs of l and m, with 0 <= l <=ntheta.

  std::random_device os_seed;
  const u32 seed = os_seed();

  engine generator( seed );
  std::uniform_int_distribution< u32 > distribute_l( 1, ntheta-1);

  std::vector<int> ls;
  std::vector<int> ms;

  for (int repetition = 0; repetition < 10; ++repetition) {
    int l = distribute_l(generator);
    std::uniform_int_distribution< u32 > distribute_m( -l, l);
    int m = distribute_m(generator);
    ls.push_back(l);
    ms.push_back(m);
  }

  double ylmR1,ylmI1,ylmR2,ylmI2;
  double int_r, int_i;
  bool failed = false;
  double max_err = 0;

  // outer loop over pairs of spherical harmonics
  for (int n1 = 0; n1 < 10; ++n1)
  for (int n2 = n1; n2 < 10; ++n2) {
    // reset doubles to store integration value
    int_r = 0;
    int_i = 0;

    // iterate over the angles
    for (int ip = 0; ip < grid.nangles; ++ip) {
      Real theta = grid.polar_pos.h_view(ip,0);
      Real phi = grid.polar_pos.h_view(ip,1);
      Real weight = grid.int_weights.h_view(ip);
      SWSphericalHarm(&ylmR1,&ylmI1, ls[n1], ms[n1], 0, theta, phi);
      SWSphericalHarm(&ylmR2,&ylmI2, ls[n2], ms[n2], 0, theta, phi);
      // complex conjugate
      ylmI2 *= -1;
      int_r += weight*(ylmR1*ylmR2 - ylmI1*ylmI2);
      int_i += weight*(ylmR1*ylmI2 + ylmR2*ylmI1);
    }

    if (ls[n1] == ls[n2] && ms[n1] == ms[n2]) {
      max_err = (abs(int_r-1)> max_err) ? abs(int_r-1) : max_err;
      max_err = (abs(int_i)> max_err) ? abs(int_i) : max_err;

      if (abs(int_r-1) >= 1e-10 || abs(int_i) >= 1e-10) {
        failed = true;
      }
    } else {
      max_err = (abs(int_r)> max_err) ? abs(int_r) : max_err;
      max_err = (abs(int_i)> max_err) ? abs(int_i) : max_err;

      if (abs(int_r) >= 1e-10 || abs(int_i) >= 1e-10) {
        failed = true;
      }
    }
    if (failed == true) {
      std::cout << "Gauss Legendre Integral Test Failed"<< std::endl;
      std::cout << "l1=" << ls[n1] << '\t' << "m1=" << ms[n1]<< std::endl;
      std::cout << "l2=" << ls[n2] << '\t' << "m2=" << ms[n2]<< std::endl;
      std::cout << "Maximum Error is " << max_err << std::endl;
      std::cout << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Test Passed with Maximum Error is " << max_err << std::endl;

  return;
}
