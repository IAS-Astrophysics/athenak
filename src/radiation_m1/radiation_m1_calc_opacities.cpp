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
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_nurates.hpp"

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
  auto &toy_opacity_fn_ = pmy_pack->pradm1->toy_opacity_fn;
  auto &chi_ = pmy_pack->pradm1->chi;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  par_for(
      "radiation_m1_calc_toy_opacity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie, 0,
      nspecies_ - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int nuidx) {
        Real &x1min = mbsize.d_view(m).x1min;
        Real &x1max = mbsize.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real dx = (x1max - x1min) / static_cast<Real>(nx1);
        Real x1 = CellCenterX(i - is, nx1, x1min, x1max);

        Real &x2min = mbsize.d_view(m).x2min;
        Real &x2max = mbsize.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real dy = (x2max - x2min) / static_cast<Real>(nx2);
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        Real &x3min = mbsize.d_view(m).x3min;
        Real &x3max = mbsize.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real dz = (x3max - x3min) / static_cast<Real>(nx3);
        Real x3 = CellCenterX(k - ks, nx3, x3min, x3max);
        toy_opacity_fn_(x1, x2, x3, dx, dy, dz, nuidx, eta_0_(m, nuidx, k, j, i),
                        abs_0_(m, nuidx, k, j, i), eta_1_(m, nuidx, k, j, i),
                        abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i));
      });

  return TaskStatus::complete;
}

TaskStatus RadiationM1::CalcOpacityNurates(Driver *pdrive, int stage) {
  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return CalcOpacityNurates_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                               Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

template<class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcOpacityNurates_(Driver *pdrive, int stage) {
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

  auto &m1_ = pmy_pack->pradm1->u0;
  auto &w0_ = pmy_pack->pmhd->w0;

  auto &eta_0_ = pmy_pack->pradm1->eta_0;
  auto &abs_0_ = pmy_pack->pradm1->abs_0;
  auto &eta_1_ = pmy_pack->pradm1->eta_1;
  auto &abs_1_ = pmy_pack->pradm1->abs_1;
  auto &scat_1_ = pmy_pack->pradm1->scat_1;
  auto &toy_opacity_fn_ = pmy_pack->pradm1->toy_opacity_fn;
  auto &chi_ = pmy_pack->pradm1->chi;
  auto &nurates_params_ = pmy_pack->pradm1->nurates_params;
  auto &radiation_mask_ = pmy_pack->pradm1->radiation_mask;
  auto &adm = pmy_pack->padm->adm;
  auto &dyngr = pmy_pack->pdyngr;

  // This is a ugly hack stolen from eos_compose_test.cpp
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pdyngr)
          ->eos.ps.GetEOSMutable();
  const Real mb = eos.ps.GetEOS().GetBaryonMass();

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  par_for(
      "radiation_m1_calc_nurates_opacity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if (radiation_mask(m, k, j, i)) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            if (nspecies_ != 1) {
              abs_0_(m, nuidx, k, j, i) = 0;
              eta_0_(m, nuidx, k, j, i) = 0;
            }
            abs_1_(m, nuidx, k, j, i) = 0;
            eta_1_(m, nuidx, k, j, i) = 0;
            scat_1_(m, nuidx, k, j, i) = 0;
          }
        } else {
          Real gam =
              adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                              adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                              adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
          Real volform = Kokkos::sqrt(gam);

          // compute from state vector
          Real rnnu[4]{};
          Real J[4]{};

          // fluid quantities
          Real nb = w0_(m, IDN, k, j, i)/mb;
          Real p = w0_(m, IPR, k, j, i);
          Real Y = w0_(m, PYF, k, j, i);
          Real T = eos.GetTemperatureFromP(nb, p, &Y);
          Real mu_b = eos.GetBaryonChemicalPotential(nb, T, &Y);
          Real mu_q = eos.GetChargeChemicalPotential(nb, T, &Y);
          Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, T, &Y);

          Real mu_n = mu_b;
          Real mu_p = mu_b + mu_q;
          Real mu_e = mu_le - mu_q;

          // Local neutrino quantities (undesitized)
          Real nudens_0[4], nudens_1[4], chi_loc[4];
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            nudens_0[nuidx] = rnnu[nuidx] / volform;
            nudens_1[nuidx] = J[nuidx] / volform;
            chi_loc[nuidx] = chi_(m, nuidx, k, j, i);
          }

          // Get emissivities and opacities
          Real eta_0_loc[4]{}, eta_1_loc[4]{};
          Real abs_0_loc[4]{}, abs_1_loc[4]{};
          Real scat_0_loc[4]{}, scat_1_loc[4]{};

          bns_nurates(nb, T, Y, mu_n, mu_p, mu_e,
            nudens_0[0], nudens_1[0], chi_loc[0],
            nudens_0[1], nudens_1[1], chi_loc[1],
            nudens_0[2], nudens_1[2], chi_loc[2],
            nudens_0[3], nudens_1[3], chi_loc[3],
            eta_0_loc[0], eta_0_loc[1], eta_0_loc[2], eta_0_loc[3],
            eta_1_loc[0], eta_1_loc[1], eta_1_loc[2], eta_1_loc[3],
            abs_0_loc[0], abs_0_loc[1], abs_0_loc[2], abs_0_loc[3],
            abs_1_loc[0], abs_1_loc[1], abs_1_loc[2], abs_1_loc[3],
            scat_0_loc[0], scat_0_loc[1], scat_0_loc[2], scat_0_loc[3],
            scat_1_loc[0], scat_1_loc[1], scat_1_loc[2], scat_1_loc[3],
            nurates_params);

          Real tau;
          Real nudens_0_trap[4], nudens_1_trap[4];
          Real nudens_0_thin[4], nudens_1_thin[4];

          if (nurates_params_.use_kirchhoff_law) {
            // An effective optical depth used to decide whether to compute
            // the black body function for neutrinos assuming neutrino trapping
            // or at a fixed temperature and Ye
            tau =
                Kokkos::min(Kokkos::sqrt(abs_1_loc[0] * (abs_1_loc[0] + scat_1_loc[0])),
                            Kokkos::sqrt(abs_1_loc[1] * (abs_1_loc[1] + scat_1_loc[1]))) *
                beta_dt;

            // Compute the neutrino black body functions assuming trapped neutrinos
            if (nurates_params_.opacity_tau_trap >= 0 &&
                tau > nurates_params_.opacity_tau_trap) {
              Real temperature_trap{}, Y_trap{};

              //auto ierr = Primitive::GetBetaEquilibriumTrapped(
              //    rho, temp, [ 1, 1, 1, 1 ], &temperature_trap, Y_trap, Real T_guess,
              //    Real * Y_guess);
              /*
              auto ierr = WeakEquilibrium(
                  rho, temp, ye, nudens_0[0], nudens_0[1], nudens_0[2], nudens_1[0],
                  nudens_1[1], nudens_1[2], &temperature_trap, &Y_e_trap,
                  &nudens_0_trap[0], &nudens_0_trap[1], &nudens_0_trap[2],
                  &nudens_1_trap[0], &nudens_1_trap[1], &nudens_1_trap[2]);
                  */
              //if (ierr) {
                // Try to recompute the weak equilibrium using neglecting
                // current neutrino data
                //ierr = WeakEquilibrium(rho[ijk], temperature[ijk], Y_e[ijk], 0.0, 0.0,
                //                       0.0, 0.0, 0.0, 0.0, &temperature_trap, &Y_e_trap,
                //                       &nudens_0_trap[0], &nudens_0_trap[1],
                //                       &nudens_0_trap[2], &nudens_1_trap[0],
                //                       &nudens_1_trap[1], &nudens_1_trap[2]);
              //}
              assert(Kokkos::isfinite(nudens_0_trap[0]));
              assert(Kokkos::isfinite(nudens_0_trap[1]));
              assert(Kokkos::isfinite(nudens_0_trap[2]));
              assert(Kokkos::isfinite(nudens_1_trap[0]));
              assert(Kokkos::isfinite(nudens_1_trap[1]));
              assert(Kokkos::isfinite(nudens_1_trap[2]));

              nudens_0_trap[2] *= 0.5;
              nudens_1_trap[2] *= 0.5;
              nudens_0_trap[3] = nudens_0_trap[2];
              nudens_1_trap[3] = nudens_1_trap[2];
            }
            // Compute the neutrino black body function assuming fixed temperature and Y_e
            auto ierr = NeutrinoDensity(rho, temp, ye, nudens_0_thin[0], nudens_0_thin[1],
                                        nudens_0_thin[2], nudens_1_thin[0],
                                        nudens_1_thin[1], nudens_1_thin[2]);
            assert(!ierr);
            nudens_0_thin[2] *= 0.5;
            nudens_1_thin[2] *= 0.5;
            nudens_0_thin[3] = nudens_0_thin[2];
            nudens_1_thin[3] = nudens_1_thin[2];
          }
        }
      });

  return TaskStatus::complete;
}
}  // namespace radiationm1