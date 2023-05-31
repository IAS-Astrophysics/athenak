//========================================================================================
// Radiation FEM_N code for Athena
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_update.cpp
//  \brief Performs update of Radiation conserved variables (f0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.
//  Explicit (not implicit) radiation source terms are included in this update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {

TaskStatus RadiationFEMN::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int npts1 = num_points_total - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage - 1];
  Real &gam1 = pdriver->gam1[stage - 1];
  Real beta_dt = (pdriver->beta[stage - 1]) * (pmy_pack->pmesh->dt);

  auto f0_ = f0;
  auto f1_ = f1;
  auto &flx1 = iflx.x1f;
  auto &flx2 = iflx.x2f;
  auto &flx3 = iflx.x3f;
  auto &L_mu_muhat0_ = L_mu_muhat0;
  auto &L_mu_muhat1_ = L_mu_muhat1;
  auto &u_mu_ = u_mu;
  auto &Gamma_ = Gamma;

  // update the distribution function for radiation
  par_for("radiation_femn_update", DevExeSpace(), 0, nmb1, 0, npts1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int enang, int k, int j, int i) {

            // Compute Christoeffel in fluid frame
            double Gamma_fluid = 0;
            //RadiationFEMNPhaseIndices idcs = this->Indices(enang);
            //int en = idcs.eindex;
            //int n = idcs.angindex;

            Real divf_s = (flx1(m, enang, k, j, i + 1) - flx1(m, enang, k, j, i)) / mbsize.d_view(m).dx1;
            if (multi_d) {
              divf_s += (flx2(m, enang, k, j + 1, i) - flx2(m, enang, k, j, i)) / mbsize.d_view(m).dx2;
            }
            if (three_d) {
              divf_s += (flx3(m, enang, k + 1, j, i) - flx3(m, enang, k, j, i)) / mbsize.d_view(m).dx3;
            }
            f0_(m, enang, k, j, i) = gam0 * f0_(m, enang, k, j, i) + gam1 * f1_(m, enang, k, j, i) - beta_dt * divf_s;
          });

  // update the tetrad quantities
  par_for("radiation_femn_tetrad_update", DevExeSpace(), 0, nmb1, 0, 4, 0, 4, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int mu, int muhat, int k, int j, int i) {
            Real tetr_rhs = (u_mu_(m, 1, k, j, i) / u_mu_(m, 0, k, j, i))
                * (L_mu_muhat1_(m, mu, muhat, k, j, i + 1) - L_mu_muhat1_(m, mu, muhat, k, j, i))
                / mbsize.d_view(m).dx1;
            if (multi_d) {
              tetr_rhs += (u_mu_(m, 2, k, j, i) / u_mu_(m, 0, k, j, i))
                  * (L_mu_muhat1_(m, mu, muhat, k, j + 1, i) - L_mu_muhat1_(m, mu, muhat, k, j, i))
                  / mbsize.d_view(m).dx2;
            }
            if (three_d) {
              tetr_rhs += (u_mu_(m, 3, k, j, i) / u_mu_(m, 0, k, j, i))
                  * (L_mu_muhat1_(m, mu, muhat, k + 1, j, i) - L_mu_muhat1_(m, mu, muhat, k, j, i))
                  / mbsize.d_view(m).dx3;
            }
            L_mu_muhat0_(m, mu, muhat, k, j, i) =
                gam0 * L_mu_muhat0_(m, mu, muhat, k, j, i) + gam1 * L_mu_muhat1_(m, mu, muhat, k, j, i)
                    - beta_dt * tetr_rhs;
          });

  // Add explicit source terms
  if (beam_source) {
    // @TODO: Add beam source support
    //AddBeamSource(f0_);
  }

  return TaskStatus::complete;
}
} // namespace radiationfemn