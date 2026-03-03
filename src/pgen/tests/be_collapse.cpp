//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file be_collapse.cpp
//! \brief Problem generator for collapse of a Bonnor-Ebert-like sphere with AMR.
//!
//! Port of the Athena++ collapse.cpp problem generator (hydro variant).
//! Sets up an enhanced Bonnor-Ebert density profile and lets it collapse under
//! self-gravity solved by the multigrid Poisson solver.  Adaptive mesh refinement
//! is driven by a Jeans-length criterion registered through user_ref_func.
//!
//! Reference: Tomida (2011) PhD Thesis (BE profile approximation)

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

namespace {

// AMR parameter (set from input in pgen, read in refinement function)
Real njeans_threshold;
Real iso_cs_global;

// Approximated Bonnor-Ebert density profile (Tomida 2011)
KOKKOS_INLINE_FUNCTION
Real BEProfile(Real r, Real rcsq) {
  return Kokkos::pow(1.0 + r * r / rcsq, -1.5);
}

}  // namespace

// Forward declaration
void JeansRefinement(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BECollapse()
//! \brief Sets up a Bonnor-Ebert sphere for gravitational collapse with Jeans AMR.

void ProblemGenerator::BECollapse(ParameterInput *pin, const bool restart) {
  // --- AMR Jeans criterion (must be set on both fresh start and restart) ---
  njeans_threshold = pin->GetOrAddReal("problem", "njeans", 16.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro != nullptr) {
    iso_cs_global = pmbp->phydro->peos->eos_data.iso_cs;
  } else {
    iso_cs_global = 1.0;
  }

  user_ref_func = JeansRefinement;

  if (restart) return;

  // --- gravity coupling ---
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (pmbp->pgrav != nullptr) {
    pmbp->pgrav->four_pi_G = four_pi_G;
    if (pmbp->pgrav->pmgd != nullptr) {
      pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
    }
  }

  // --- problem parameters ---
  // Bonnor-Ebert profile constants (dimensionless)
  Real rc   = 6.45;           // cloud radius in normalized units  default value
  Real f   = pin->GetOrAddReal("problem", "f", 1.2);    // density enhancement
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);  // m=2 perturbation amplitude
  Real x_center = pin->GetOrAddReal("problem", "x_center", 0.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  rc = pin->GetOrAddReal("problem", "cloud_radius", rc);
  Real rcsq = SQR(rc) / 3.0; // parameter of the BE profile

  // Solid-body rotation: omega = omegatff / tff, where tff = pi*sqrt(3/(8f))
  Real tff = std::sqrt(3.0 / (8.0 * f)) * M_PI;
  Real omegatff = pin->GetOrAddReal("problem", "omegatff", 0.0);
  Real omega = omegatff / tff;

  // --- initialize density ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  if (pmbp->phydro == nullptr) return;
  auto &u0 = pmbp->phydro->u0;
  int nmb = pmbp->nmb_thispack;

  par_for("be_collapse_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    Real r = Kokkos::sqrt(SQR(x - x_center) + SQR(y - y_center) + SQR(z - z_center));
    // Clamp to rc for pressure confinement (constant density beyond cloud)
    Real r_clamped = Kokkos::fmin(r, rc);

    Real rho = f * BEProfile(r_clamped, rcsq);
    if (amp > 0.0 && r < rc) {
      rho *= (1.0 + amp * (r * r) / (rc * rc)
              * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
    }

    u0(m, IDN, k, j, i) = rho;
    if (r < rc) {
      u0(m, IM1, k, j, i) =  rho * omega * (y - y_center);
      u0(m, IM2, k, j, i) = -rho * omega * (x - x_center);
    } else {
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
    }
    u0(m, IM3, k, j, i) = 0.0;
  });

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- Bonnor-Ebert Collapse ---" << std::endl
      << "Density enhancement f   = " << f << std::endl
      << "Cloud radius rc         = " << rc << std::endl
      << "Free-fall time tff      = " << tff << std::endl
      << "Omega * tff             = " << omegatff << std::endl
      << "Angular velocity omega  = " << omega << std::endl
      << "Perturbation amplitude  = " << amp << std::endl
      << "Jeans AMR threshold     = " << njeans_threshold << std::endl
      << "Sound speed cs          = " << iso_cs_global << std::endl
      << "four_pi_G               = " << four_pi_G << std::endl
      << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void JeansRefinement()
//! \brief Jeans-length AMR criterion for isothermal self-gravitating gas.
//!
//! For each meshblock, computes the minimum Jeans number:
//!   nJ = cs / sqrt(rho_max) * (2*pi / dx)
//! and sets the refinement flag accordingly.

void JeansRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  int ng = indcs.ng;
  const int nkji = (nx3 + 2 * ng) * (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int nji  = (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int ni   = (nx1 + 2 * ng);
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real cs = iso_cs_global;
  Real njeans = njeans_threshold;

  par_for_outer("JeansAMR", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_rhomax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &rhomax) {
        int k = idx / nji;
        int j = (idx - k * nji) / ni;
        int i = (idx - k * nji - j * ni);
        rhomax = Kokkos::fmax(u0(m, IDN, k, j, i), rhomax);
      },
      Kokkos::Max<Real>(team_rhomax));

    Real dx = size.d_view(m).dx1;
    Real nj_min = cs / Kokkos::sqrt(team_rhomax) * (2.0 * M_PI / dx);

    if (nj_min < njeans) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (nj_min > njeans * 2.5) {
      refine_flag.d_view(m + mbs) = -1;
    } else {
      refine_flag.d_view(m + mbs) = 0;
    }
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
