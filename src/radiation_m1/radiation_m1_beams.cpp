//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_beams.cpp
//! \brief beam initial data for grey M1

#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"

namespace radiationm1 {

// Beams from left wall of domain (1d, single species)
void ApplyBeamSources1D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;

  int &ng = indcs.ng;
  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_source_vals;


  par_for(
      "radiation_m1_beams_populate_1d", DevExeSpace(), 0, nmb1, 0, nvars_-1,
      KOKKOS_LAMBDA(int m, int n) {
        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            for (int i = 0; i < ng; ++i) {
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, n, nvars_), 0, 0, is - i - 1) = beam_source_1_vals_(n);
                }
            }
            break;
          default:
            break;
        }
      });
}

// Beams from left wall of domain (2d, single species)
void ApplyBeamSources2D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;

  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;

  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;

  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_source_vals;
  auto &beam_source_1_y1_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymin;
  auto &beam_source_1_y2_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymax;


  par_for(
      "radiation_m1_beams_populate_2d", DevExeSpace(), 0, nmb1, 0, nvars_-1, 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            if (beam_source_1_y1_ <= x2 && x2 <= beam_source_1_y2_) {
              for (int i = 0; i < ng; ++i) {
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, n, nvars_), k, j, is - i - 1) = beam_source_1_vals_(n);
                }
              }
            }
            break;
          default:
            break;
        }
      });
}

// Beam for the M1 beam test around black hole (2d, single species)
void ApplyBeamSourcesBlackHole(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;

  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  auto &params_ = pmesh->pmb_pack->pradm1->params;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;

  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  adm::ADM::ADM_vars &adm = pmesh->pmb_pack->padm->adm;

  auto &beam_source_1_y1_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymin;
  auto &beam_source_1_y2_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymax;


  par_for(
      "radiation_m1_beam_populate_black_hole", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0,
      (n2 - 1), KOKKOS_LAMBDA(int m, int k, int j) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            if (beam_source_1_y1_ <= x2 && x2 <= beam_source_1_y2_) {
              for (int i = 0; i < ng; ++i) {
                // Calculate inverse 4-metric and sqrt(det(3-metric))
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
                Real garr_uu[16];
                adm::SpacetimeUpperMetric(
                    adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                    adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                    adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                    adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                    adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_uu);
                for (int a = 0; a < 4; ++a) {
                  for (int b = 0; b < 4; ++b) {
                    g_uu(a, b) = garr_uu[a + b * 4];
                  }
                }
                const Real gam = adm::SpatialDet(
                    adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                    adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                    adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
                const Real volform = Kokkos::sqrt(gam);

                // Solve for beam id
                const Real g_xx = adm.g_dd(m, 0, 0, k, j, i);
                Real beta_x = 0;
                Real beta2 = 0;
                for (int idx = 0; idx < 3; idx++) {
                  beta_x += adm.g_dd(m, 0, idx, k, j, i) * adm.beta_u(m, idx, k, j, i);
                  for (int idx2 = 0; idx2 < 3; idx2++) {
                    beta2 += adm.g_dd(m, idx, idx2, k, j, i) *
                             adm.beta_u(m, idx, k, j, i) * adm.beta_u(m, idx2, k, j, i);
                  }
                }
                const Real a =
                    (-beta_x + sqrt(beta_x * beta_x - beta2 +
                                    adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i) *
                                        (1 - params_.rad_eps))) /
                    g_xx;

                Real E = volform * 1.;
                Real Fx = a * E / adm.alpha(m, k, j, i) +
                          adm.beta_u(m, 0, k, j, i) * E / adm.alpha(m, k, j, i);
                Real Fy = adm.beta_u(m, 1, k, j, i) * E / adm.alpha(m, k, j, i);
                Real Fz = adm.beta_u(m, 2, k, j, i) * E / adm.alpha(m, k, j, i);
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
                pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                         adm.beta_u(m, 2, k, j, i), volform * Fx, volform * Fy,
                         volform * Fz, F_d);
                apply_floor(g_uu, E, F_d, params_);
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, is - i - 1) = E;
                  u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, is - i - 1) = F_d(1);
                  u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, is - i - 1) = F_d(2);
                  u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, is - i - 1) = F_d(3);
                }
              }
            }
            break;
          default:
            break;
        }
      });
}

}  // namespace radiationm1