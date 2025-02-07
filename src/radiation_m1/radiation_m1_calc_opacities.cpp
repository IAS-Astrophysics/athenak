//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacity.cpp
//! \brief calculate opacities for grey M1

#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_opacities.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::CalcOpacityToy(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int &ng = indcs.ng;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto nvarstotm1 = pmy_pack->pradm1->nvarstot - 1;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &params_ = pmy_pack->pradm1->params;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &u_mu_ = pmy_pack->pradm1->u_mu;
  auto &u0_ = pmy_pack->pradm1->u0;

  auto &eta_0_ = pmy_pack->pradm1->eta_0;
  auto &abs_0_ = pmy_pack->pradm1->abs_0;
  auto &eta_1_ = pmy_pack->pradm1->eta_1;
  auto &abs_1_ = pmy_pack->pradm1->abs_1;
  auto &scat_1_ = pmy_pack->pradm1->scat_1;

  auto &chi_ = pmy_pack->pradm1->chi;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  par_for(
      "radiation_m1_calc_toy_opacity", DevExeSpace(), 0, nmb1, ks, ke, js, je,
      is, ie, 0, nspecies_ - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    const int nuidx) {
        Real &x1min = mbsize.d_view(m).x1min;
        Real &x1max = mbsize.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

        Real &x2min = mbsize.d_view(m).x2min;
        Real &x2max = mbsize.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        Real &x3min = mbsize.d_view(m).x3min;
        Real &x3max = mbsize.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real x3 = CellCenterX(k - ks, nx3, x3min, x3max);

        ComputeToyOpacities(
            x1, x2, x3, nuidx, eta_0_(m, nuidx, k, j, i),
            abs_0_(m, nuidx, k, j, i), eta_1_(m, nuidx, k, j, i),
            abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i));
      });

  return TaskStatus::complete;
}

TaskStatus RadiationM1::CalcOpacityNurates(Driver *pdrive, int stage) {
  return TaskStatus::complete;
}
}  // namespace radiationm1