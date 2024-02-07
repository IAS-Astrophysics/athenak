//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_initialize.cpp
//  \brief this is a temporary file -- it hardcodes metric and fluid velocities

#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

void RadiationFEMN::InitializeMetricFluid() {

  auto &indices = pmy_pack->pmesh->mb_indcs;
  int &is = indices.is, &ie = indices.ie;
  int &js = indices.js, &je = indices.je;
  int &ks = indices.ks, &ke = indices.ke;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * (indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * (indcs.ng)) : 1;

  DvceArray6D<Real> g_dd_host;
  DvceArray4D<Real> sqrt_det_g_host;
  DvceArray5D<Real> u_mu_host;

  Kokkos::realloc(g_dd_host, nmb, 4, 4, ncells3, ncells2, ncells1);        // 4-metric from GR
  Kokkos::realloc(u_mu_host, nmb, 4, ncells3, ncells2, ncells1);

  Kokkos::deep_copy(g_dd_host, 0.);
  Kokkos::deep_copy(u_mu_host, 0.);

  par_for("radiation_femn_dummy_initialize_1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
            g_dd_host(m, 0, 0, k, j, i) = -1.;
            g_dd_host(m, 1, 1, k, j, i) = 1.;
            g_dd_host(m, 2, 2, k, j, i) = 1.;
            g_dd_host(m, 3, 3, k, j, i) = 1.;
            u_mu_host(m, 0, k, j, i) = 1;
          });

  Kokkos::deep_copy(g_dd, g_dd_host);
  Kokkos::deep_copy(u_mu, u_mu_host);
  
}
}