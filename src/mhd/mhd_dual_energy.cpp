//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_dual_energy.cpp
//! \brief Dual-energy formalism for Newtonian ideal-gas MHD.

#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real MHDInternalEnergyFloor(const EOS_Data &eos, const Real dens) {
  Real eint_floor = eos.pfloor/(eos.gamma - 1.0);
  if (eos.tfloor > 0.0) {
    eint_floor = fmax(eint_floor, dens*eos.tfloor/(eos.gamma - 1.0));
  }
  if (eos.sfloor > 0.0) {
    eint_floor = fmax(eint_floor, dens*eos.sfloor*pow(dens, eos.gamma - 1.0)/
                                  (eos.gamma - 1.0));
  }
  return eint_floor;
}

KOKKOS_INLINE_FUNCTION
bool DualEnergySyncEligible(const Real eint_cons, const Real local_etot_max,
                            const Real eta2) {
  if (eint_cons <= 0.0) return false;
  if (eta2 <= 0.0) return true;
  return (eint_cons > eta2*fmax(local_etot_max, 1.0e-18));
}

} // namespace

namespace mhd {

TaskStatus MHD::DualEnergyStep(Driver *pdrive, int stage) {
  if (use_dual_energy) {
    const Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
    ApplyDualEnergyFormalism(beta_dt);
  }
  return TaskStatus::complete;
}

void MHD::InitializeDualEnergyFieldFromTotal() {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto b0_ = b0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  par_for("mhd_dual_energy_init", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    const Real bx = 0.5*(b0_.x1f(m, k, j, i) + b0_.x1f(m, k, j, i+1));
    const Real by = 0.5*(b0_.x2f(m, k, j, i) + b0_.x2f(m, k, j+1, i));
    const Real bz = 0.5*(b0_.x3f(m, k, j, i) + b0_.x3f(m, k+1, j, i));
    const Real e_m = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
    const Real eint_floor = MHDInternalEnergyFloor(eos, dens);
    const Real eint = fmax(u0_(m, IEN, k, j, i) - e_k - e_m, eint_floor);
    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void MHD::ApplyDualEnergyFormalism(const Real dt) {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto u0_ = u0;
  auto w0_ = w0;
  auto vf1_ = dual_vf.x1f;
  auto vf2_ = dual_vf.x2f;
  auto vf3_ = dual_vf.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  par_for("mhd_dual_energy_compress", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real divv = (vf1_(m, 0, k, j, i+1) - vf1_(m, 0, k, j, i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      divv += (vf2_(m, 0, k, j+1, i) - vf2_(m, 0, k, j, i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divv += (vf3_(m, 0, k+1, j, i) - vf3_(m, 0, k, j, i))/mbsize.d_view(m).dx3;
    }

    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real eint_floor = MHDInternalEnergyFloor(eos, dens);
    Real eint = fmax(u0_(m, de_idx, k, j, i), eint_floor);
    eint *= exp(-(eos.gamma - 1.0)*divv*dt);
    eint = fmax(eint, eint_floor);
    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void MHD::SynchronizeDualEnergyFieldFromTotal() {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int gis = is - ng;
  const int gie = ie + ng;
  const int gjs = multi_d ? (js - ng) : js;
  const int gje = multi_d ? (je + ng) : je;
  const int gks = three_d ? (ks - ng) : ks;
  const int gke = three_d ? (ke + ng) : ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto b0_ = b0;
  auto etot_max_ = dual_etot_max;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  const Real eta2 = dual_energy_eta2;

  if (eta2 > 0.0) {
    par_for("mhd_dual_energy_etot_max", DevExeSpace(), 0, nmb1, gks, gke, gjs, gje, gis, gie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real emax = 0.0;
      const int kmin = three_d ? ((k - 1 > gks) ? (k - 1) : gks) : k;
      const int kmax = three_d ? ((k + 1 < gke) ? (k + 1) : gke) : k;
      const int jmin = multi_d ? ((j - 1 > gjs) ? (j - 1) : gjs) : j;
      const int jmax = multi_d ? ((j + 1 < gje) ? (j + 1) : gje) : j;
      const int imin = (i - 1 > gis) ? (i - 1) : gis;
      const int imax = (i + 1 < gie) ? (i + 1) : gie;
      for (int kk = kmin; kk <= kmax; ++kk) {
        for (int jj = jmin; jj <= jmax; ++jj) {
          for (int ii = imin; ii <= imax; ++ii) {
            emax = fmax(emax, u0_(m, IEN, kk, jj, ii));
          }
        }
      }
      etot_max_(m, k, j, i) = emax;
    });
  }

  par_for("mhd_dual_energy_sync", DevExeSpace(), 0, nmb1, gks, gke, gjs, gje, gis, gie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    const Real bx = 0.5*(b0_.x1f(m, k, j, i) + b0_.x1f(m, k, j, i+1));
    const Real by = 0.5*(b0_.x2f(m, k, j, i) + b0_.x2f(m, k, j+1, i));
    const Real bz = 0.5*(b0_.x3f(m, k, j, i) + b0_.x3f(m, k+1, j, i));
    const Real e_m = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
    const Real eint_cons = u0_(m, IEN, k, j, i) - e_k - e_m;
    const Real local_etot_max = (eta2 > 0.0) ? etot_max_(m, k, j, i) : 0.0;
    Real eint_aux = u0_(m, de_idx, k, j, i);
    if (DualEnergySyncEligible(eint_cons, local_etot_max, eta2)) {
      eint_aux = eint_cons;
    }
    eint_aux = fmax(eint_aux, MHDInternalEnergyFloor(eos, dens));
    u0_(m, de_idx, k, j, i) = eint_aux;
    w0_(m, de_idx, k, j, i) = eint_aux;
  });
}

void MHD::SynchronizeRestrictedDualEnergyField() {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.cis, ie = indcs.cie;
  const int js = indcs.cjs, je = indcs.cje;
  const int ks = indcs.cks, ke = indcs.cke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto cu = coarse_u0;
  auto cw = coarse_w0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  par_for("mhd_dual_energy_sync_restricted", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(cu(m, IDN, k, j, i), eos.dfloor);
    const Real eint_aux = fmax(cu(m, de_idx, k, j, i),
                               MHDInternalEnergyFloor(eos, dens));
    cu(m, de_idx, k, j, i) = eint_aux;
    cw(m, de_idx, k, j, i) = eint_aux;
  });
}

void MHD::RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                       const int new_gids, const int new_nmb_local) {
  if (!use_dual_energy || new_nmb_local <= 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = new_nmb_local - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  par_for("mhd_dual_energy_repair_refined", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag.d_view(n2o.d_view(m + new_gids)) <= 0) return;
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real eint_aux = fmax(u0_(m, de_idx, k, j, i),
                               MHDInternalEnergyFloor(eos, dens));
    u0_(m, de_idx, k, j, i) = eint_aux;
    w0_(m, de_idx, k, j, i) = eint_aux;
  });
}

} // namespace mhd
