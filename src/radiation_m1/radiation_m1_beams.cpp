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

// Beams from left wall of domain (1d only)
void ApplyBeamSources1D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvarstotm1 = pmesh->pmb_pack->pradm1->nvarstot - 1;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;

  int &ng = indcs.ng;
  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->beam_source_vals;

  par_for(
      "radiation_femn_beams_populate_1d", DevExeSpace(), 0, nmb1, 0, nvarstotm1,
      KOKKOS_LAMBDA(int m, int n) {
        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            for (int i = 0; i < ng; ++i) {
              u0_(m, n, 0, 0, is - i - 1) = beam_source_1_vals_(n);
            }
            break;
          default:
            break;
        }
      });
}

// Beams from left wall of domain (2d only)
void ApplyBeamSources2D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;

  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvarstotm1 = pmesh->pmb_pack->pradm1->nvarstot - 1;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;

  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;

  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->beam_source_vals;
  //auto &beam_source_1_y1_ = pmesh->pmb_pack->pradm1->beam_source_1_y1;
  //auto &beam_source_1_y2_ = pmesh->pmb_pack->pradm1->beam_source_1_y2;
  /*
  par_for(
      "radiation_femn_beams_populate_2d", DevExeSpace(), 0, nmb1, 0, nvarstotm1, 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            if (beam_source_1_y1_ <= x2 && x2 <= beam_source_1_y2_) {
              for (int i = 0; i < ng; ++i) {
                u0_(m, n, k, j, is - i - 1) = beam_source_1_vals_(n);
              }
            }
            break;
          default:
            break;
        }
      }); */
}
}  // namespace radiationm1