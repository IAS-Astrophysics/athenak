//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chevalier_clegg.cpp
//! \brief Simple Chevalier & Clegg (1985) wind test with uniform injection region.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

namespace {
Real rho0, pres0, v10, v20, v30;
Real r_inj, mdot, edot;
Real q_m, q_e;
Real x0, y0, z0;

void CC85Source(Mesh *pm, const Real bdt);
} // namespace

//----------------------------------------------------------------------------------------
//  \brief Problem Generator for a CC85-style injection test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Chevalier-Clegg test requires Hydro or MHD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  pres0 = pin->GetOrAddReal("problem", "pres0", 1.0);
  v10 = pin->GetOrAddReal("problem", "v1", 0.0);
  v20 = pin->GetOrAddReal("problem", "v2", 0.0);
  v30 = pin->GetOrAddReal("problem", "v3", 0.0);

  r_inj = pin->GetOrAddReal("problem", "r_inj", 0.1);
  mdot = pin->GetOrAddReal("problem", "mdot", 1.0);
  edot = pin->GetOrAddReal("problem", "edot", 1.0);
  x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  z0 = pin->GetOrAddReal("problem", "z0", 0.0);

  if (r_inj <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "r_inj must be > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real vol_inj = (4.0/3.0) * M_PI * std::pow(r_inj, 3);
  q_m = mdot / vol_inj;
  q_e = edot / vol_inj;

  user_srcs_func = CC85Source;

  if (restart) return;

  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    if (!eos.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Chevalier-Clegg test assumes ideal EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real gm1 = eos.gamma - 1.0;

    par_for("cc85_init_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = rho0 * v10;
      u0(m, IM2, k, j, i) = rho0 * v20;
      u0(m, IM3, k, j, i) = rho0 * v30;
      u0(m, IEN, k, j, i) = pres0 / gm1
                            + 0.5 * rho0 *
                            (SQR(v10) + SQR(v20) + SQR(v30));
    });
  } else if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    auto &bcc = pmbp->pmhd->bcc0;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    if (!eos.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Chevalier-Clegg test assumes ideal EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real gm1 = eos.gamma - 1.0;

    par_for("cc85_init_mhd", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m, IDN, k, j, i) = rho0;
      u0(m, IM1, k, j, i) = rho0 * v10;
      u0(m, IM2, k, j, i) = rho0 * v20;
      u0(m, IM3, k, j, i) = rho0 * v30;
      u0(m, IEN, k, j, i) = pres0 / gm1
                            + 0.5 * rho0 *
                            (SQR(v10) + SQR(v20) + SQR(v30));
      bcc(m, IBX, k, j, i) = 0.0;
      bcc(m, IBY, k, j, i) = 0.0;
      bcc(m, IBZ, k, j, i) = 0.0;
    });
  }
}

namespace {

void CC85Source(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;

  DvceArray5D<Real> u0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
  } else {
    u0_ = pmbp->pmhd->u0;
  }

  Real q_m_ = q_m;
  Real q_e_ = q_e;
  Real r_inj_ = r_inj;
  Real x0_ = x0;
  Real y0_ = y0;
  Real z0_ = z0;

  par_for("cc85_injection", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real y = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real z = CellCenterX(k-ks, nx3, x3min, x3max);

    Real dx = x - x0_;
    Real dy = y - y0_;
    Real dz = z - z0_;
    Real r = sqrt(dx*dx + dy*dy + dz*dz);

    if (r <= r_inj_) {
      Real dm = q_m_ * bdt;
      Real dE = q_e_ * bdt;

      u0_(m, IDN, k, j, i) += dm;

      u0_(m, IEN, k, j, i) += dE;
    }
  });
}

} // namespace
