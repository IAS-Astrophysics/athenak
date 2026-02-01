//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chevalier_clegg.cpp
//! \brief Simple Chevalier & Clegg (1985) wind test with uniform injection region.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

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
Real x0_cc, y0_cc, z0_cc;

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
  x0_cc = pin->GetOrAddReal("problem", "x0", 0.0);
  y0_cc = pin->GetOrAddReal("problem", "y0", 0.0);
  z0_cc = pin->GetOrAddReal("problem", "z0", 0.0);
  const int n_spheres = pin->GetOrAddInteger("problem", "n_spheres", 0);
  const Real sphere_radius = pin->GetOrAddReal("problem", "sphere_radius", 0.0);
  Real sphere_rho = pin->GetOrAddReal("problem", "sphere_rho", rho0);
  const Real sphere_rho_factor = pin->GetOrAddReal("problem", "sphere_rho_factor", -1.0);
  const int sphere_seed = pin->GetOrAddInteger("problem", "sphere_seed", 12345);

  if (sphere_rho_factor > 0.0) {
    sphere_rho = rho0 * sphere_rho_factor;
  }
  if (n_spheres < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "n_spheres must be >= 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (n_spheres > 0) {
    if (sphere_radius <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "sphere_radius must be > 0 when n_spheres > 0"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (sphere_rho <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "sphere_rho must be > 0 when n_spheres > 0"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const auto &mesh_size = pmy_mesh_->mesh_size;
    if ((mesh_size.x1max - mesh_size.x1min) < 2.0 * sphere_radius ||
        (mesh_size.x2max - mesh_size.x2min) < 2.0 * sphere_radius ||
        (mesh_size.x3max - mesh_size.x3min) < 2.0 * sphere_radius) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "sphere_radius is too large for the mesh domain"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

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
  auto &size = pmbp->pmb->mb_size;

  Kokkos::View<Real*[3]> sphere_centers("cc85_sphere_centers", n_spheres);
  if (n_spheres > 0) {
    const auto &mesh_size = pmy_mesh_->mesh_size;
    const Real x1min = mesh_size.x1min + sphere_radius;
    const Real x1max = mesh_size.x1max - sphere_radius;
    const Real x2min = mesh_size.x2min + sphere_radius;
    const Real x2max = mesh_size.x2max - sphere_radius;
    const Real x3min = mesh_size.x3min + sphere_radius;
    const Real x3max = mesh_size.x3max - sphere_radius;

    std::mt19937 rng(static_cast<uint32_t>(sphere_seed));
    std::uniform_real_distribution<Real> dist_x1(x1min, x1max);
    std::uniform_real_distribution<Real> dist_x2(x2min, x2max);
    std::uniform_real_distribution<Real> dist_x3(x3min, x3max);

    auto h_centers = Kokkos::create_mirror_view(sphere_centers);
    for (int s = 0; s < n_spheres; ++s) {
      h_centers(s, 0) = dist_x1(rng);
      h_centers(s, 1) = dist_x2(rng);
      h_centers(s, 2) = dist_x3(rng);
    }
    Kokkos::deep_copy(sphere_centers, h_centers);
  }

  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    if (!eos.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Chevalier-Clegg test assumes ideal EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real gm1 = eos.gamma - 1.0;
    const Real rho0_ = rho0;
    const Real pres0_ = pres0;
    const Real v10_ = v10;
    const Real v20_ = v20;
    const Real v30_ = v30;
    const Real gm1_ = gm1;
    const int n_spheres_ = n_spheres;
    const Real sphere_r2_ = sphere_radius * sphere_radius;
    const Real sphere_rho_ = sphere_rho;
    auto sphere_centers_ = sphere_centers;

    par_for("cc85_init_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real rho = rho0_;
      if (n_spheres_ > 0) {
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

        for (int s = 0; s < n_spheres_; ++s) {
          Real dx = x - sphere_centers_(s, 0);
          Real dy = y - sphere_centers_(s, 1);
          Real dz = z - sphere_centers_(s, 2);
          if ((dx*dx + dy*dy + dz*dz) <= sphere_r2_) {
            rho = sphere_rho_;
            break;
          }
        }
      }

      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * v10_;
      u0(m, IM2, k, j, i) = rho * v20_;
      u0(m, IM3, k, j, i) = rho * v30_;
      u0(m, IEN, k, j, i) = pres0_ / gm1_
                            + 0.5 * rho *
                            (SQR(v10_) + SQR(v20_) + SQR(v30_));
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
    const Real rho0_ = rho0;
    const Real pres0_ = pres0;
    const Real v10_ = v10;
    const Real v20_ = v20;
    const Real v30_ = v30;
    const Real gm1_ = gm1;
    const int n_spheres_ = n_spheres;
    const Real sphere_r2_ = sphere_radius * sphere_radius;
    const Real sphere_rho_ = sphere_rho;
    auto sphere_centers_ = sphere_centers;

    par_for("cc85_init_mhd", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real rho = rho0_;
      if (n_spheres_ > 0) {
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

        for (int s = 0; s < n_spheres_; ++s) {
          Real dx = x - sphere_centers_(s, 0);
          Real dy = y - sphere_centers_(s, 1);
          Real dz = z - sphere_centers_(s, 2);
          if ((dx*dx + dy*dy + dz*dz) <= sphere_r2_) {
            rho = sphere_rho_;
            break;
          }
        }
      }

      u0(m, IDN, k, j, i) = rho;
      u0(m, IM1, k, j, i) = rho * v10_;
      u0(m, IM2, k, j, i) = rho * v20_;
      u0(m, IM3, k, j, i) = rho * v30_;
      u0(m, IEN, k, j, i) = pres0_ / gm1_
                            + 0.5 * rho *
                            (SQR(v10_) + SQR(v20_) + SQR(v30_));
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
  Real x0_ = x0_cc;
  Real y0_ = y0_cc;
  Real z0_ = z0_cc;

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
