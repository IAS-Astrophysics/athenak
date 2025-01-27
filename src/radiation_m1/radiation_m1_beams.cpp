//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_beams.cpp
//! \brief beam initial data for grey M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_helpers.hpp"

namespace radiationm1 {

// Beams from left wall of domain for FEM (1d only)
void ApplyBeamSources1D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvarstotm1 = pmesh->pmb_pack->pradm1->nvarstot - 1;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;

  int &ng = indcs.ng;
  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->beam_source_vals;

  par_for("radiation_femn_beams_populate_1d", DevExeSpace(), 0, nmb1, 0, nvarstotm1,
          KOKKOS_LAMBDA(int m, int n) {
            switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
              case BoundaryFlag::outflow:
                for (int i = 0; i < ng; ++i) {
                  u0_(m, n, 0, 0, is - i - 1) = beam_source_1_vals_(n);
                }
                break;
              default:break;
            }
          });
}

}