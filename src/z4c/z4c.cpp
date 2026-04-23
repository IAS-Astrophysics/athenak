//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c.cpp
//! \brief implementation of Z4c class constructor and assorted other functions

#include <math.h>
#include <sys/stat.h>  // mkdir

#include <iostream>
#include <string>
#include <algorithm>
#include <memory>    // make_unique, unique_ptr
#include <vector>    // vector
#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/horizon_dump.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "utils/cart_grid.hpp"

namespace z4c {

char const * const Z4c::Z4c_names[Z4c::nz4c] = {
  "z4c_chi",
  "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz",
  "z4c_Khat",
  "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz",
  "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
  "z4c_Theta",
  "z4c_alpha",
  "z4c_betax", "z4c_betay", "z4c_betaz",
  "z4c_Bx", "z4c_By", "z4c_Bz",
};

char const * const Z4c::Constraint_names[Z4c::ncon] = {
  "con_C",
  "con_H",
  "con_M",
  "con_Z",
  "con_Mx", "con_My", "con_Mz",
};

/*char const * const Z4c::Matter_names[Z4c::nmat] = {
  "mat_rho",
  "mat_Sx", "mat_Sy", "mat_Sz",
  "mat_Sxx", "mat_Sxy", "mat_Sxz", "mat_Syy", "mat_Syz", "mat_Szz",
};*/

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Z4c::Z4c(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u_con("u_con",1,1,1,1,1),
  //u_mat("u_mat",1,1,1,1,1),
  u0("u0 z4c",1,1,1,1,1),
  u_bg("u_bg z4c",1,1,1,1,1),
  u_full("u_full z4c",1,1,1,1,1),
  u_adm_bg("u_adm_bg",1,1,1,1,1),
  coarse_u0("coarse u0 z4c",1,1,1,1,1),
  u1("u1 z4c",1,1,1,1,1),
  u_rhs("u_rhs z4c",1,1,1,1,1),
  u_weyl("u_weyl",1,1,1,1,1),
  coarse_u_weyl("coarse_u_weyl",1,1,1,1,1),
  pamr(new Z4c_AMR(pin)) {
  // (1) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  // int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::Profiling::pushRegion("Tensor fields");
  Kokkos::realloc(u_con, nmb, (ncon), ncells3, ncells2, ncells1);
  // Matter commented out
  // kokkos::realloc(u_mat, nmb, (N_MAT), ncells3, ncells2, ncells1);
  Kokkos::realloc(u0,    nmb, (nz4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u_bg,  nmb, (nz4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u_full,nmb, (nz4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u1,    nmb, (nz4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u_rhs, nmb, (nz4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u_adm_bg, nmb, 17, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_weyl,    nmb, (2), ncells3, ncells2, ncells1);

  con.C.InitWithShallowSlice(u_con, I_CON_C);
  con.H.InitWithShallowSlice(u_con, I_CON_H);
  con.M.InitWithShallowSlice(u_con, I_CON_M);
  con.Z.InitWithShallowSlice(u_con, I_CON_Z);
  con.M_d.InitWithShallowSlice(u_con, I_CON_MX, I_CON_MZ);

  z4c.alpha.InitWithShallowSlice (u0, I_Z4C_ALPHA);
  z4c.beta_u.InitWithShallowSlice(u0, I_Z4C_BETAX, I_Z4C_BETAZ);
  z4c.vB_d.InitWithShallowSlice(u0, I_Z4C_BX, I_Z4C_BZ);
  z4c.chi.InitWithShallowSlice   (u0, I_Z4C_CHI);
  z4c.vKhat.InitWithShallowSlice  (u0, I_Z4C_KHAT);
  z4c.vTheta.InitWithShallowSlice (u0, I_Z4C_THETA);
  z4c.vGam_u.InitWithShallowSlice (u0, I_Z4C_GAMX, I_Z4C_GAMZ);
  z4c.g_dd.InitWithShallowSlice  (u0, I_Z4C_GXX, I_Z4C_GZZ);
  z4c.vA_dd.InitWithShallowSlice  (u0, I_Z4C_AXX, I_Z4C_AZZ);

  bg.alpha.InitWithShallowSlice (u_bg, I_Z4C_ALPHA);
  bg.beta_u.InitWithShallowSlice(u_bg, I_Z4C_BETAX, I_Z4C_BETAZ);
  bg.vB_d.InitWithShallowSlice(u_bg, I_Z4C_BX, I_Z4C_BZ);
  bg.chi.InitWithShallowSlice   (u_bg, I_Z4C_CHI);
  bg.vKhat.InitWithShallowSlice  (u_bg, I_Z4C_KHAT);
  bg.vTheta.InitWithShallowSlice (u_bg, I_Z4C_THETA);
  bg.vGam_u.InitWithShallowSlice (u_bg, I_Z4C_GAMX, I_Z4C_GAMZ);
  bg.g_dd.InitWithShallowSlice  (u_bg, I_Z4C_GXX, I_Z4C_GZZ);
  bg.vA_dd.InitWithShallowSlice  (u_bg, I_Z4C_AXX, I_Z4C_AZZ);

  full.alpha.InitWithShallowSlice (u_full, I_Z4C_ALPHA);
  full.beta_u.InitWithShallowSlice(u_full, I_Z4C_BETAX, I_Z4C_BETAZ);
  full.vB_d.InitWithShallowSlice(u_full, I_Z4C_BX, I_Z4C_BZ);
  full.chi.InitWithShallowSlice   (u_full, I_Z4C_CHI);
  full.vKhat.InitWithShallowSlice  (u_full, I_Z4C_KHAT);
  full.vTheta.InitWithShallowSlice (u_full, I_Z4C_THETA);
  full.vGam_u.InitWithShallowSlice (u_full, I_Z4C_GAMX, I_Z4C_GAMZ);
  full.g_dd.InitWithShallowSlice  (u_full, I_Z4C_GXX, I_Z4C_GZZ);
  full.vA_dd.InitWithShallowSlice  (u_full, I_Z4C_AXX, I_Z4C_AZZ);

  rhs.alpha.InitWithShallowSlice (u_rhs, I_Z4C_ALPHA);
  rhs.beta_u.InitWithShallowSlice(u_rhs, I_Z4C_BETAX, I_Z4C_BETAZ);
  rhs.vB_d.InitWithShallowSlice  (u_rhs, I_Z4C_BX, I_Z4C_BZ);
  rhs.chi.InitWithShallowSlice   (u_rhs, I_Z4C_CHI);
  rhs.vKhat.InitWithShallowSlice  (u_rhs, I_Z4C_KHAT);
  rhs.vTheta.InitWithShallowSlice (u_rhs, I_Z4C_THETA);
  rhs.vGam_u.InitWithShallowSlice (u_rhs, I_Z4C_GAMX, I_Z4C_GAMZ);
  rhs.g_dd.InitWithShallowSlice  (u_rhs, I_Z4C_GXX, I_Z4C_GZZ);
  rhs.vA_dd.InitWithShallowSlice  (u_rhs, I_Z4C_AXX, I_Z4C_AZZ);

  adm_bg.g_dd.InitWithShallowSlice(u_adm_bg, 0, 5);
  adm_bg.vK_dd.InitWithShallowSlice(u_adm_bg, 6, 11);
  adm_bg.psi4.InitWithShallowSlice(u_adm_bg, 12);
  adm_bg.alpha.InitWithShallowSlice(u_adm_bg, 13);
  adm_bg.beta_u.InitWithShallowSlice(u_adm_bg, 14, 16);

  weyl.rpsi4.InitWithShallowSlice (u_weyl, 0);
  weyl.ipsi4.InitWithShallowSlice (u_weyl, 1);

  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opt.chi_min_floor = pin->GetOrAddReal("z4c", "chi_min_floor", 1e-12);
  opt.diss = pin->GetOrAddReal("z4c", "diss", 0.0);
  opt.eps_floor = pin->GetOrAddReal("z4c", "eps_floor", 1e-12);
  opt.damp_kappa1 = pin->GetOrAddReal("z4c", "damp_kappa1", 0.0);
  opt.damp_kappa2 = pin->GetOrAddReal("z4c", "damp_kappa2", 0.0);
  // Gauge conditions (default to moving puncture gauge)
  opt.lapse_harmonicf = pin->GetOrAddReal("z4c", "lapse_harmonicf", 1.0);
  opt.lapse_harmonic = pin->GetOrAddReal("z4c", "lapse_harmonic", 0.0);
  opt.lapse_oplog = pin->GetOrAddReal("z4c", "lapse_oplog", 2.0);
  opt.lapse_advect = pin->GetOrAddReal("z4c", "lapse_advect", 1.0);
  opt.slow_start_lapse = pin->GetOrAddBoolean("z4c", "slow_start_lapse", false);
  opt.ssl_damping_amp = pin->GetOrAddReal("z4c", "ssl_damping_amp", 0.6);
  opt.ssl_damping_time = pin->GetOrAddReal("z4c", "ssl_damping_time", 20.0);
  opt.ssl_damping_index = pin->GetOrAddInteger("z4c", "ssl_damping_index", 1);
  opt.sss_damping_amp = pin->GetOrAddReal("z4c", "sss_damping_amp", 0.);
  opt.sss_damping_time = pin->GetOrAddReal("z4c", "sss_damping_time", 10.0);
  opt.telegraph_lapse = pin->GetOrAddBoolean("z4c", "telegraph_lapse", false);
  opt.telegraph_tau = pin->GetOrAddReal("z4c", "telegraph_tau", 0.1);
  opt.telegraph_kappa = pin->GetOrAddReal("z4c", "telegraph_kappa", 0.1);

  opt.shift_ggamma = pin->GetOrAddReal("z4c", "shift_Gamma", 1.0);
  opt.shift_advect = pin->GetOrAddReal("z4c", "shift_advect", 1.0);
  opt.shift_alpha2ggamma = pin->GetOrAddReal("z4c", "shift_alpha2Gamma", 0.0);
  opt.shift_hh = pin->GetOrAddReal("z4c", "shift_H", 0.0);
  opt.shift_eta = pin->GetOrAddReal("z4c", "shift_eta", 2.0);

  opt.use_z4c = pin->GetOrAddBoolean("z4c", "use_z4c", true);
  use_analytic_background = pin->GetOrAddBoolean("z4c", "use_analytic_background",
                                                 false);

  opt.user_Sbc = pin->GetOrAddBoolean("z4c", "user_Sbc", false);

  opt.excise_chi = pin->GetOrAddReal("z4c", "excise_chi", 0.0625);
  opt.history_excise_ks_horizon =
      pin->GetOrAddBoolean("z4c", "history_excise_ks_horizon", false);
  Real default_ks_spin = 0.0;
  if (ppack->pcoord != nullptr) {
    default_ks_spin = ppack->pcoord->coord_data.bh_spin;
  }
  opt.history_excise_ks_spin =
      pin->GetOrAddReal("z4c", "history_excise_ks_spin", default_ks_spin);
  Real horizon_spin = fmin(1.0, fabs(opt.history_excise_ks_spin));
  Real default_ks_radius = 1.0 + sqrt(fmax(0.0, 1.0 - SQR(horizon_spin)));
  opt.history_excise_ks_radius =
      pin->GetOrAddReal("z4c", "history_excise_ks_radius", default_ks_radius);
  Real default_ks_x1 = 0.0;
  Real default_ks_x2 = 0.0;
  Real default_ks_x3 = 0.0;
  if (pin->DoesParameterExist("problem", "bh_center_x1")) {
    default_ks_x1 = pin->GetReal("problem", "bh_center_x1");
  }
  if (pin->DoesParameterExist("problem", "bh_center_x2")) {
    default_ks_x2 = pin->GetReal("problem", "bh_center_x2");
  }
  if (pin->DoesParameterExist("problem", "bh_center_x3")) {
    default_ks_x3 = pin->GetReal("problem", "bh_center_x3");
  }
  opt.history_excise_ks_x1 =
      pin->GetOrAddReal("z4c", "history_excise_ks_x1", default_ks_x1);
  opt.history_excise_ks_x2 =
      pin->GetOrAddReal("z4c", "history_excise_ks_x2", default_ks_x2);
  opt.history_excise_ks_x3 =
      pin->GetOrAddReal("z4c", "history_excise_ks_x3", default_ks_x3);

  opt.extrap_order = fmax(2,fmin(indcs.ng,fmin(4,
      pin->GetOrAddInteger("z4c", "extrap_order", 2))));

  opt.roll_kappa = pin->GetOrAddBoolean("z4c", "roll_kappa", false);
  opt.kappa_roll_start_time = pin->GetOrAddReal("z4c", "kappa_roll_start_time", 0.0);
  opt.roll_window = pin->GetOrAddReal("z4c", "roll_window", 20.0);
  opt.target_kappa1 = pin->GetOrAddReal("z4c", "target_kappa1", 0.0);

  diss = opt.diss*pow(2., -2.*indcs.ng)*(indcs.ng % 2 == 0 ? -1. : 1.);
  Kokkos::deep_copy(DevExeSpace(), u_bg, 0.0);
  Kokkos::deep_copy(DevExeSpace(), u_full, 0.0);
  Kokkos::deep_copy(DevExeSpace(), u_adm_bg, 0.0);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nz4c), nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_u_weyl, nmb, (2), nccells3, nccells2, nccells1);
  }
  Kokkos::Profiling::popRegion();

  // allocate boundary buffers for conserved (cell-centered) variables
  Kokkos::Profiling::pushRegion("Buffers");
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers((nz4c));
  pbval_weyl = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_weyl->InitializeBuffers((2));
  Kokkos::Profiling::popRegion();

  // wave extraction spheres
  // TODO(@hzhu): Read radii from input file
  auto &grids = spherical_grids;
  // set nrad_wave_extraction = 0 to turn off wave extraction
  nrad = pin->GetOrAddReal("z4c", "nrad_wave_extraction", 0);
  int nlev = pin->GetOrAddReal("z4c", "extraction_nlev", 10);
  for (int i=0; i<nrad; i++) {
    Real rad = pin->GetOrAddReal("z4c", "extraction_radius_"+std::to_string(i), 10);
    grids.push_back(std::make_unique<SphericalGrid>(ppack, nlev, rad));
  }
  // TODO(@dur566): Why is the size of psi_out hardcoded?
  psi_out = new Real[nrad*77*2];
  if (nrad > 0) {
    mkdir("waveforms",0775);
  }
  waveform_dt = pin->GetOrAddReal("z4c", "waveform_dt", 1);
  last_output_time = 0;
  // CCE
  cce_dump_dt = pin->GetOrAddReal("cce", "cce_dt", 1);
  int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
  if (ncce > 0) {
    mkdir("cce",0775);
  }
  cce_dump_last_output_time = -100;

  // Construct the compact object trackers
  int n = 0;
  while (true) {
    if (pin->DoesParameterExist("z4c", "co_" + std::to_string(n) + "_type")) {
      ptracker.push_back(std::make_unique<CompactObjectTracker>(pmy_pack->pmesh, pin, n));
      n++;
    } else {
      break;
    }
  }
  // Construct the Cartesian data grid for dumping horizon data
  n = 0;
  while (true) {
    if (pin->GetOrAddBoolean("z4c", "dump_horizon_" + std::to_string(n),false)) {
      // phorizon_dump.emplace_back(pmy_pack, pin, n,false);
      phorizon_dump.push_back(std::make_unique<HorizonDump>(pmy_pack, pin, n, 0));
      std::string foldername = "horizon_"+std::to_string(n);
      mkdir(foldername.c_str(),0775);
      n++;
    } else {
      break;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::AlgConstr(AthenaArray<Real> & u)
//! \brief algebraic constraints projection
//
// This function operates on all grid points of the MeshBlock
void Z4c::EnforceAlgConstrOn(Z4c_vars &state) {
  // capture variables for the kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;

  int nmb = pmy_pack->nmb_thispack;
  par_for("Alg constr loop",DevExeSpace(),
  0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real detg = adm::SpatialDet(state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
                              state.g_dd(m,0,2,k,j,i),state.g_dd(m,1,1,k,j,i),
                              state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i));
    detg = detg > 0. ? detg : 1.;
    // Real eps = detg - 1.;
    // Real oopsi4 = (eps < opt.eps_floor) ? (1. - opt.eps_floor/3.) :
    //             (std::pow(1./detg, 1./3.));
    Real oopsi4 = std::cbrt(1./detg);

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      state.g_dd(m,a,b,k,j,i) *= oopsi4;
    }

    // compute trace of A
    // note: here we are assuming that det g = 1, which we enforced above
    Real A = adm::Trace(1.0,
              state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i), state.g_dd(m,0,2,k,j,i),
              state.g_dd(m,1,1,k,j,i), state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i),
              state.vA_dd(m,0,0,k,j,i), state.vA_dd(m,0,1,k,j,i), state.vA_dd(m,0,2,k,j,i),
              state.vA_dd(m,1,1,k,j,i), state.vA_dd(m,1,2,k,j,i), state.vA_dd(m,2,2,k,j,i));

    // enforce trace of A to be zero
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      state.vA_dd(m,a,b,k,j,i) -= (1.0/3.0) * A * state.g_dd(m,a,b,k,j,i);
    }
  });
}

void Z4c::AlgConstr(MeshBlockPack *pmbp) {
  EnforceAlgConstrOn(z4c);
}

void Z4c::RefreshBackground(Real time) {
  if (!(use_analytic_background) || SetADMBackground == nullptr) {
    Kokkos::deep_copy(DevExeSpace(), u_bg, 0.0);
    return;
  }
  SetADMBackground(pmy_pack, time);
}

void Z4c::ReconstructFullState() {
  if (!(use_analytic_background) || SetADMBackground == nullptr) {
    Kokkos::deep_copy(DevExeSpace(), u_full, u0);
    return;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int isg = indcs.is-indcs.ng; int ieg = indcs.ie+indcs.ng;
  int jsg = indcs.js-indcs.ng; int jeg = indcs.je+indcs.ng;
  int ksg = indcs.ks-indcs.ng; int keg = indcs.ke+indcs.ng;

  auto &res = z4c;
  auto &bg_ = bg;
  auto &full_ = full;
  par_for("ReconstructFullState", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    full_.chi(m,k,j,i) = bg_.chi(m,k,j,i) + res.chi(m,k,j,i);
    full_.vKhat(m,k,j,i) = bg_.vKhat(m,k,j,i) + res.vKhat(m,k,j,i);
    full_.vTheta(m,k,j,i) = bg_.vTheta(m,k,j,i) + res.vTheta(m,k,j,i);
    full_.alpha(m,k,j,i) = bg_.alpha(m,k,j,i);
    for (int a = 0; a < 3; ++a) {
      full_.vGam_u(m,a,k,j,i) = bg_.vGam_u(m,a,k,j,i) + res.vGam_u(m,a,k,j,i);
      full_.beta_u(m,a,k,j,i) = bg_.beta_u(m,a,k,j,i);
      full_.vB_d(m,a,k,j,i) = bg_.vB_d(m,a,k,j,i);
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        full_.g_dd(m,a,b,k,j,i) = bg_.g_dd(m,a,b,k,j,i) + res.g_dd(m,a,b,k,j,i);
        full_.vA_dd(m,a,b,k,j,i) = bg_.vA_dd(m,a,b,k,j,i) + res.vA_dd(m,a,b,k,j,i);
      }
    }
  });
}

void Z4c::RecastResidualState() {
  if (!(use_analytic_background) || SetADMBackground == nullptr) {
    Kokkos::deep_copy(DevExeSpace(), u0, u_full);
    return;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int isg = indcs.is-indcs.ng; int ieg = indcs.ie+indcs.ng;
  int jsg = indcs.js-indcs.ng; int jeg = indcs.je+indcs.ng;
  int ksg = indcs.ks-indcs.ng; int keg = indcs.ke+indcs.ng;

  auto &res = z4c;
  auto &bg_ = bg;
  auto &full_ = full;
  par_for("RecastResidualState", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    res.chi(m,k,j,i) = full_.chi(m,k,j,i) - bg_.chi(m,k,j,i);
    res.vKhat(m,k,j,i) = full_.vKhat(m,k,j,i) - bg_.vKhat(m,k,j,i);
    res.vTheta(m,k,j,i) = full_.vTheta(m,k,j,i) - bg_.vTheta(m,k,j,i);
    res.alpha(m,k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      res.vGam_u(m,a,k,j,i) = full_.vGam_u(m,a,k,j,i) - bg_.vGam_u(m,a,k,j,i);
      res.beta_u(m,a,k,j,i) = 0.0;
      res.vB_d(m,a,k,j,i) = 0.0;
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        res.g_dd(m,a,b,k,j,i) = full_.g_dd(m,a,b,k,j,i) - bg_.g_dd(m,a,b,k,j,i);
        res.vA_dd(m,a,b,k,j,i) = full_.vA_dd(m,a,b,k,j,i) - bg_.vA_dd(m,a,b,k,j,i);
      }
    }
  });
}

void Z4c::PrescribeGaugeResidual() {
  if (!(use_analytic_background) || SetADMBackground == nullptr) {
    return;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  auto &res = z4c;
  par_for("PrescribeGaugeResidual", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    res.alpha(m,k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      res.beta_u(m,a,k,j,i) = 0.0;
      res.vB_d(m,a,k,j,i) = 0.0;
    }
  });
}

//----------------------------------------------------------------------------------------
// destructor
Z4c::~Z4c() {
  delete[] psi_out;
  delete pbval_u;
  delete pbval_weyl;
  delete pamr;
}

} // namespace z4c
