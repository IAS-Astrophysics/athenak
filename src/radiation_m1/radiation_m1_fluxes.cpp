//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_fluxes.cpp
//! \brief calculate 3D fluxes for M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"

namespace radiationm1 {
TaskStatus RadiationM1::Fluxes(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto nvarstotm1 = pmy_pack->pradm1->nvarstot - 1;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = 2;
  int scr_level = 0;
  auto &flx1_ = uflx.x1f;

  int il = is, iu = ie + 1, jl = js, ju = je, kl = ks, ku = ke;

  par_for_outer(
      "radiation_m1_flux_x1", DevExeSpace(), scr_size, scr_level, 0, nmb1, kl,
      ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j,
                    const int i) {
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 1> gamma_uu;
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 1> g_uu;
        /*
        geom.get_inv_metric(ijk, &gamma_uu);
        geom.get_inv_metric(ijk, &g_uu);
        geom.get_shift_vec(ijk, &beta_u);

        tensor::generic<CCTK_REAL, 4, 1> u_u;
        fidu.get(ijk, &u_u);

        tensor::generic<CCTK_REAL, 4, 1> v_u;
        pack_v_u(fidu_velx[ijk], fidu_vely[ijk], fidu_velz[ijk], &v_u);

        tensor::generic<CCTK_REAL, 4, 1> H_d;
        tensor::generic<CCTK_REAL, 4, 1> H_u;
        tensor::generic<CCTK_REAL, 4, 1> F_d;
        tensor::generic<CCTK_REAL, 4, 1> F_u;
        tensor::symmetric2<CCTK_REAL, 4, 2> P_dd;
        tensor::generic<CCTK_REAL, 4, 2> P_ud;

        tensor::generic<CCTK_REAL, 4, 1> fnu_u;

        // inner loop over species
        // par_for_inner(member, 0, nspecies - 1, KOKKOS_LAMBDA[=](int
        // nuidx) {
        //}); */
      });
  return TaskStatus::complete;
}
} // namespace radiationm1
