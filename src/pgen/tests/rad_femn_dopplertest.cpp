//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_dopplertest.cpp
//! \brief doppler test in 1d for radiation FEM_N

// C++ headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max
#include <iomanip>
#include "radiation_femn/radiation_femn_geodesic_grid.hpp"

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "adm/adm.hpp"
#include "coordinates/cell_locations.hpp"

KOKKOS_INLINE_FUNCTION Real B(Real en, Real T) {
  Real h = 1.;
  Real cspeed = 1.;
  Real kb = 1.;

  return 1;
}

void ProblemGenerator::RadiationFEMNDopplertest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pradfemn == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "The radiation FEM_N Doppler test can only be run with radiation-femn, but no " << std::endl
              << "<radiation-femn> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &num_energy_bins_ = pmbp->pradfemn->num_energy_bins;
  auto &energy_grid_ = pmbp->pradfemn->energy_grid;
  auto &energy_max_ = pmbp->pradfemn->energy_max;
  auto &num_points = pmbp->pradfemn->num_points;
  auto &num_energy_bins = pmbp->pradfemn->num_energy_bins;
  auto &rad_E_floor = pmbp->pradfemn->rad_E_floor;
  auto &rad_eps = pmbp->pradfemn->rad_eps;
  auto &beam_source_1_vals_ = pmbp->pradfemn->beam_source_1_vals;

  HostArray1D<Real> temp_array;
  Kokkos::realloc(temp_array, num_energy_bins_ + 1);
  Kokkos::deep_copy(temp_array, energy_grid_);

  std::cout << "Number of energy bins: " << num_energy_bins_ << std::endl;
  std::cout << "Number of points on energy grid: " << num_energy_bins_ + 1 << std::endl;
  std::cout << "Energy grid values: " << std::endl;
  for (int i = 0; i < num_energy_bins_ + 1; i++) {
    std::cout << temp_array(i) << std::endl;
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

  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;
  int nmb = pmbp->nmb_thispack;
  auto &u_mu_ = pmbp->pradfemn->u_mu;
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  Real x0 = 2.0;
  Real x1 = 3.5;
  Real x2 = 6.5;
  Real x3 = 8.0;
  Real A = 5.*1e7/(3.*1e8);
  Real l = 6.0;

  // set metric to minkowski, initialize velocity
  par_for("pgen_dopplertest_metric_initialize", DevExeSpace(),
          0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            for (int a = 0; a < 3; ++a)
              for (int b = a; b < 3; ++b) {
                adm.g_dd(m, a, b, k, j, i) = (a == b ? 1. : 0.);
              }
            adm.psi4(m, k, j, i) = 1.; // adm.psi4
            adm.alpha(m, k, j, i) = 1.;

            Real& x1min = size.d_view(m).x1min;
            Real& x1max = size.d_view(m).x1max;
            int nx1 = indcs.nx1;
            Real xval = CellCenterX(i - is, nx1, x1min, x1max);

            Real vx = 0.0;
            if(xval < x0) {
              vx = 0.0;
            } else if (xval >= x0 && xval < x1) {
              vx = A * Kokkos::sin(2.*M_PI*(xval - x0)/Real(l))
                  * Kokkos::sin(2.*M_PI*(xval - x0)/Real(l));
            } else if (xval >= x1 && xval < x2) {
              vx = A;
            } else if (xval >= x2 && xval < x3) {
              vx = A * Kokkos::sin(2.*M_PI*(xval - x0)/Real(l))
                  * Kokkos::sin(2.*M_PI*(xval - x0)/Real(l));
            } else {
              vx = 0.0;
            }
            auto lorentz_w = 1. / sqrt(1 - vx * vx);

            u_mu_(m, 0, k, j, i) = lorentz_w;
            u_mu_(m, 1, k, j, i) = vx * lorentz_w;
            u_mu_(m, 2, k, j, i) = 0.;
            u_mu_(m, 3, k, j, i) = 0.;
          });

  // initialize values of the beam sources
  HostArray1D<Real> beam_source_1_vals_h;
  Kokkos::realloc(beam_source_1_vals_h, num_points * num_energy_bins);
  Kokkos::realloc(beam_source_1_vals_, num_points * num_energy_bins);

  user_bcs = true;
  user_bcs_func = radiationfemn::ApplyBeamSourcesFEMN1D;

  Real temp = 1000; //[K]

  for (int en = 0; en < num_energy_bins; en++) {
    int idx = radiationfemn::IndicesUnited(0, en, 0, 1, num_energy_bins, num_points);
    Real enval = (temp_array(en)+temp_array(en+1))/2.;
    Real fnorm = B(enval, temp);
    Real en_dens = fnorm;
    Real fx = fnorm;
    Real fy = 0;
    Real fz = 0;
    Real f2 = fx * fx + fy * fy + fz * fz;

    en_dens = Kokkos::fmax(en_dens, rad_E_floor);
    Real lim = en_dens * en_dens * (1. - rad_eps);
    if (f2 > lim) {
      Real fac = lim / f2;
      fx = fac * fx;
      fy = fac * fy;
      fz = fac * fz;
  }
  beam_source_1_vals_h(idx) = (1. / Kokkos::sqrt(4. * M_PI)) * en_dens;
  beam_source_1_vals_h(idx+1) = -Kokkos::sqrt(3. / (4. * M_PI)) * fy;
  beam_source_1_vals_h(idx+2) = Kokkos::sqrt(3. / (4. * M_PI)) * fz;
  beam_source_1_vals_h(idx+3) = -Kokkos::sqrt(3. / (4. * M_PI)) * fx;
  }
  Kokkos::deep_copy(beam_source_1_vals_, beam_source_1_vals_h);

  return;
}