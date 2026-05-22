//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_neutrino_shock.cpp
//  \brief pgen for neutrino-mediated shock calculations

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <random>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#ifdef ENABLE_NURATES
#include "radiation_m1/radiation_m1_nurates.hpp"
#endif

// Forward declaration
template <class EOSPolicy, class ErrorPolicy>
void NeutrinoDominatedShock(Mesh *pmesh, ParameterInput* pin);

// Prototype for user-defined history
void NeutrinoShockHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for a shock reflection
void ProblemGenerator::UserProblem(ParameterInput* pin, const bool restart) {
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout
        << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << "GRMHD shock reflection problem can only be run when <adm> block is present"
        << std::endl;
    std::exit(EXIT_FAILURE);
  }

  user_hist_func = &NeutrinoShockHistory;

  if (restart) {
    return;
  }

  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto* ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor>*>(pmbp->pdyngr);
  if (ptest_nqt != nullptr) {
    return NeutrinoDominatedShock<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor>(pmy_mesh_, pin);
  }

  auto* ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor>*>(
      pmbp->pdyngr);
  if (ptest_nlog != nullptr) {
    return NeutrinoDominatedShock<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor>(pmy_mesh_, pin);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

// Actual initial data solver
template <class EOSPolicy, class ErrorPolicy>
void NeutrinoDominatedShock(Mesh *pmesh, ParameterInput* pin) {
  // Get the EOS and set units to CGS
  MeshBlockPack* pmbp = pmesh->pmb_pack;
  const bool has_m1 = (pmbp->pradm1 != nullptr);

  if (has_m1) {
#ifndef ENABLE_NURATES
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "dyngr_neutrino_shock pgen requires ENABLE_NURATES=ON to "
                 "initialize neutrinos in LTE when M1 radiation is enabled."
              << std::endl;
    std::exit(EXIT_FAILURE);
#endif  // ENABLE_NURATES
  }

  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();

  MHDPrim1D wpt;
  wpt.d = pin->GetReal("problem", "rho");
  Real nb = wpt.d/mb;
  // 3-velocity components (v^i in the Eulerian frame)
  Real v3x = pin->GetOrAddReal("problem", "vx", 0.0);
  Real v3y = pin->GetOrAddReal("problem", "vy", 0.0);
  Real v3z = pin->GetOrAddReal("problem", "vz", 0.0);
  Real temp = pin->GetReal("problem", "temp");
  wpt.bx = pin->GetOrAddReal("problem", "bx", 0.0);
  wpt.by = pin->GetOrAddReal("problem", "by", 0.0);
  wpt.bz = pin->GetOrAddReal("problem", "bz", 0.0);
  Real yq = pin->GetReal("problem", "yq");
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);

  // Spectral density perturbation: delta_rho/rho = N * sum_n |k_n|^(-alpha/2)
  // * cos(k_n . x + phi_n), with random phases and isotropic directions in the
  // active dimensions. N is chosen so the analytic RMS equals pert_amp.
  // pert_nmodes = 0 (default) disables the perturbation.
  int pert_nmodes = pin->GetOrAddInteger("problem", "pert_nmodes", 0);
  Real pert_kmin = pin->GetOrAddReal("problem", "pert_kmin", 1.0);
  Real pert_kmax = pin->GetOrAddReal("problem", "pert_kmax", 64.0);
  Real pert_alpha = pin->GetOrAddReal("problem", "pert_alpha", 5.0/3.0);
  int pert_seed = pin->GetOrAddInteger("problem", "pert_seed", 12345);

  int nmodes_alloc = std::max(pert_nmodes, 1);
  DualArray1D<Real> pert_kx("pert_kx", nmodes_alloc);
  DualArray1D<Real> pert_ky("pert_ky", nmodes_alloc);
  DualArray1D<Real> pert_kz("pert_kz", nmodes_alloc);
  DualArray1D<Real> pert_an("pert_an", nmodes_alloc);
  DualArray1D<Real> pert_ph("pert_ph", nmodes_alloc);

  if (pert_nmodes > 0) {
    Real Lx = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
    Real two_pi = 2.0*acos(-1.0);
    std::mt19937 rng(static_cast<unsigned>(pert_seed));
    std::uniform_real_distribution<Real> uni01(0.0, 1.0);

    Real sumsq = 0.0;
    for (int n = 0; n < pert_nmodes; ++n) {
      Real frac = (pert_nmodes > 1) ?
                  static_cast<Real>(n)/static_cast<Real>(pert_nmodes - 1) : 0.0;
      Real k_units = pert_kmin * pow(pert_kmax/pert_kmin, frac);
      Real kmag = two_pi * k_units / Lx;

      Real kxn, kyn, kzn;
      if (pmesh->three_d) {
        Real cos_th = 2.0*uni01(rng) - 1.0;
        Real sin_th = sqrt(fmax(0.0, 1.0 - cos_th*cos_th));
        Real phi = two_pi*uni01(rng);
        kxn = kmag * sin_th * cos(phi);
        kyn = kmag * sin_th * sin(phi);
        kzn = kmag * cos_th;
      } else if (pmesh->two_d) {
        Real phi = two_pi*uni01(rng);
        kxn = kmag * cos(phi);
        kyn = kmag * sin(phi);
        kzn = 0.0;
      } else {
        kxn = kmag;
        kyn = 0.0;
        kzn = 0.0;
      }

      pert_kx.h_view(n) = kxn;
      pert_ky.h_view(n) = kyn;
      pert_kz.h_view(n) = kzn;
      Real raw_amp = pow(k_units, -0.5*pert_alpha);
      pert_an.h_view(n) = raw_amp;
      pert_ph.h_view(n) = two_pi*uni01(rng);
      sumsq += raw_amp*raw_amp;
    }
    Real norm = (sumsq > 0.0) ? pert_amp*sqrt(2.0/sumsq) : 0.0;
    for (int n = 0; n < pert_nmodes; ++n) {
      pert_an.h_view(n) *= norm;
    }
    pert_kx.template modify<HostMemSpace>();
    pert_ky.template modify<HostMemSpace>();
    pert_kz.template modify<HostMemSpace>();
    pert_an.template modify<HostMemSpace>();
    pert_ph.template modify<HostMemSpace>();
    pert_kx.template sync<DevExeSpace>();
    pert_ky.template sync<DevExeSpace>();
    pert_kz.template sync<DevExeSpace>();
    pert_an.template sync<DevExeSpace>();
    pert_ph.template sync<DevExeSpace>();
  }

  // compute Lorentz factor
  Real w_lorentz = 1.0 / sqrt(1.0 - (SQR(v3x) + SQR(v3y) + SQR(v3z)));
  wpt.vx = w_lorentz*v3x;
  wpt.vy = w_lorentz*v3y;
  wpt.vz = w_lorentz*v3z;

  // capture variables for the kernel
  auto& indcs = pmesh->mb_indcs;
  int& is = indcs.is;
  int& ie = indcs.ie;
  int& js = indcs.js;
  int& je = indcs.je;
  int& ks = indcs.ks;
  int& ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // capture grid arrays
  auto &w0_ = pmbp->pmhd->w0;

  // Initialize ADM variables now so the metric is available later for the
  // Lorentz transformation of the radiation fields (and for the prim-to-cons).
  pmbp->padm->SetADMVariables(pmbp);

  // setup MHD primitives (always)
  par_for("pgen_shock1_mhd", DevExeSpace(),
          0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real Y[2] = {yq, 0.0};

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx1 = indcs.nx1;
    int nx2 = indcs.nx2;
    int nx3 = indcs.nx3;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);
    Real y = CellCenterX(j-js, nx2, x2min, x2max);
    Real z = CellCenterX(k-ks, nx3, x3min, x3max);

    // spectral (turbulence-like) density perturbation
    Real delta = 0.0;
    for (int n = 0; n < pert_nmodes; ++n) {
      delta += pert_an.d_view(n) * cos(pert_kx.d_view(n)*x
                                     + pert_ky.d_view(n)*y
                                     + pert_kz.d_view(n)*z
                                     + pert_ph.d_view(n));
    }
    w0_(m,IDN,k,j,i) = wpt.d * (1.0 + delta);

    w0_(m,IVX,k,j,i) = wpt.vx * (x > 0 ? -1 : 1);
    w0_(m,IVY,k,j,i) = wpt.vy;
    w0_(m,IVZ,k,j,i) = wpt.vz;
    w0_(m,IPR,k,j,i) = eos.GetPressure(nb, temp, &Y[0]);
    w0_(m,IYF,k,j,i) = Y[0];
    w0_(m,IYF+1,k,j,i) = Y[1];
  });

  // Initialize face-centered and cell-centered magnetic fields (uniform)
  {
    auto &b0_ = pmbp->pmhd->b0;
    auto &bcc0_ = pmbp->pmhd->bcc0;
    Real bx0 = wpt.bx;
    Real by0 = wpt.by;
    Real bz0 = wpt.bz;
    par_for("pgen_shock_bfield", DevExeSpace(),
            0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0_.x1f(m,k,j,i) = bx0;
      b0_.x2f(m,k,j,i) = by0;
      b0_.x3f(m,k,j,i) = bz0;
      if (i==ie) b0_.x1f(m,k,j,i+1) = bx0;
      if (j==je) b0_.x2f(m,k,j+1,i) = by0;
      if (k==ke) b0_.x3f(m,k+1,j,i) = bz0;
      bcc0_(m,IBX,k,j,i) = bx0;
      bcc0_(m,IBY,k,j,i) = by0;
      bcc0_(m,IBZ,k,j,i) = bz0;
    });
  }

  // setup M1 radiation fields (only if M1 is enabled)
  if (has_m1) {
#ifdef ENABLE_NURATES
    // Optically-thick (Eddington) closure: in the fluid frame H^a = 0 and
    // K^{ab} = (1/3) J h^{ab}, so T^{ab} = (4J/3) u^a u^b + (J/3) g^{ab}.
    // Projecting onto the Eulerian observer gives, for a uniform 3-velocity,
    //   E    = (4 W^2 - 1) J / 3            (lab energy density)
    //   F^i  = (4/3) J W^2 v^i              (lab energy flux, upper index)
    //   N    = W n                          (lab number density)
    Real W2 = w_lorentz * w_lorentz;
    Real cE = (4.0 * W2 - 1.0) / 3.0;
    Real cF = (4.0 / 3.0) * W2;

    // Unit systems and nurates parameters (host-only, no GPU table access)
    auto code_units    = eos.GetCodeUnitSystem();
    auto eos_units     = eos.GetEOSUnitSystem();
    auto nurates_units = Primitive::MakeNGS();
    auto &nurates_params_ = pmbp->pradm1->nurates_params;

    auto &uradm1_ = pmbp->pradm1->u0;
    auto &nspecies_ = pmbp->pradm1->nspecies;
    if (nspecies_ == 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl;
      std::cout << "Neutrino radiation transport is required for this pgen!\n";
      abort();
    }

    auto &m1_params_ = pmbp->pradm1->params;
    auto &m1_nvars_ = pmbp->pradm1->nvars;

    par_for("pgen_shock1_m1", DevExeSpace(),
            0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real Y[2] = {yq, 0.0};

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x = CellCenterX(i-is, nx1, x1min, x1max);

      // LTE neutrino fluid-frame number and energy densities.
      // Use the uniform (unperturbed) matter state to match the uniform pressure
      // init above. NeutrinoDens returns the total contribution from all four
      // heavy-flavor species in n_nux/e_nux; we split that evenly between
      // species 2 (nux) and species 3 (anti-nux), matching the convention used
      // in radiation_m1_calc_opacities_nurates.cpp.
      Real mu_b  = eos.GetBaryonChemicalPotential(nb, temp, &Y[0]);
      Real mu_q  = eos.GetChargeChemicalPotential(nb, temp, &Y[0]);
      Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, temp, &Y[0]);
      Real mu_n = mu_b;
      Real mu_p = mu_b + mu_q;
      Real mu_e = mu_le - mu_q;

      Real n_nue, n_anue, n_nux, e_nue, e_anue, e_nux;
      radiationm1::NeutrinoDens(mu_n, mu_p, mu_e, temp,
                                n_nue, n_anue, n_nux,
                                e_nue, e_anue, e_nux,
                                nurates_params_, code_units, eos_units,
                                nurates_units);
      Real nuN[4] = {n_nue, n_anue, 0.5*n_nux, 0.5*n_nux};
      Real nuJ[4] = {e_nue, e_anue, 0.5*e_nux, 0.5*e_nux};

      for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
        Real J = nuJ[nuidx];
        Real n_fluid = nuN[nuidx];

        Real E   = cE * J;
        Real Fxd = cF * J * v3x * (x > 0.0 ? -1.0 : 1.0);
        Real Fyd = cF * J * v3y;
        Real Fzd = cF * J * v3z;
        Real N   = w_lorentz * n_fluid;

        E = fmax(E, m1_params_.rad_E_floor);
        N = fmax(N, m1_params_.rad_N_floor);

        uradm1_(m, radiationm1::CombinedIdx(nuidx,M1_E_IDX, m1_nvars_),k,j,i) = E;
        uradm1_(m, radiationm1::CombinedIdx(nuidx,M1_FX_IDX,m1_nvars_),k,j,i) = Fxd;
        uradm1_(m, radiationm1::CombinedIdx(nuidx,M1_FY_IDX,m1_nvars_),k,j,i) = Fyd;
        uradm1_(m, radiationm1::CombinedIdx(nuidx,M1_FZ_IDX,m1_nvars_),k,j,i) = Fzd;
        uradm1_(m, radiationm1::CombinedIdx(nuidx,M1_N_IDX, m1_nvars_),k,j,i) = N;
      }
    });
#endif  // ENABLE_NURATES
  }

  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  return;
}

// History function: maximum |B^i| (each Cartesian component) over the domain
void NeutrinoShockHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 4;
  pdata->label[0] = "bx-max";
  pdata->label[1] = "by-max";
  pdata->label[2] = "bz-max";
  pdata->label[3] = "rhoy5-max";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real bx_max = 0.0;
  Real by_max = 0.0;
  Real bz_max = 0.0;
  Real rhoy5_max = 0.0;
  Kokkos::parallel_reduce("NeutrinoShockHistorySums",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_bx_max, Real &mb_by_max, Real &mb_bz_max,
                Real &mb_rhoy5_max) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // bcc0 stores the densitized B-field; divide by sqrt(det g) to get B^i
    Real gamma = sqrt(
        adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                        adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                        adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)));
    Real bx = fabs(bcc0_(m,IBX,k,j,i)/gamma);
    Real by = fabs(bcc0_(m,IBY,k,j,i)/gamma);
    Real bz = fabs(bcc0_(m,IBZ,k,j,i)/gamma);
    Real rhoy5 = fabs(w0_(m,IDN,k,j,i) * w0_(m,IYF+1,k,j,i));

    mb_bx_max = fmax(bx, mb_bx_max);
    mb_by_max = fmax(by, mb_by_max);
    mb_bz_max = fmax(bz, mb_bz_max);
    mb_rhoy5_max = fmax(rhoy5, mb_rhoy5_max);
  }, Kokkos::Max<Real>(bx_max), Kokkos::Max<Real>(by_max), Kokkos::Max<Real>(bz_max),
     Kokkos::Max<Real>(rhoy5_max));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need
  // MPI_MAX here. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &bx_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &by_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &bz_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &rhoy5_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&bx_max, &bx_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&by_max, &by_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bz_max, &bz_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhoy5_max, &rhoy5_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    bx_max = 0.0;
    by_max = 0.0;
    bz_max = 0.0;
    rhoy5_max = 0.0;
  }
#endif

  pdata->hdata[0] = bx_max;
  pdata->hdata[1] = by_max;
  pdata->hdata[2] = bz_max;
  pdata->hdata[3] = rhoy5_max;
}
