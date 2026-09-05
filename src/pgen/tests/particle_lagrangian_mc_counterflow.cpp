//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_lagrangian_mc_counterflow.cpp
//! \brief Prescribed-flux regression for Lagrangian MC coarse/fine counterflow.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "particles/lagrangian_mc.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace lagrangian_mc = particles::lagrangian_mc;

namespace {

constexpr int kFineParticles = 256;
constexpr Real kInwardFraction = 0.25;
Real outward_fraction = 0.25;

Real CounterflowTimestep(MeshBlockPack*) { return 0.125; }

// Group 0 is the coarse cell; groups 1 and 2 touch opposite corners of its x1 face.
void InitialPosition(int group, bool three_d, Real &x, Real &y, Real &z) {
  x = (group == 0) ? 0.5 : -0.25;
  y = (group == 0) ? -1.75 : ((group == 1) ? -1.875 : -1.625);
  z = three_d ? y : 0.0;
}

void InitializeCounterflowParticles(MeshBlockPack *pmbp,
                                   particles::ParticleCreation *creation) {
  const bool three_d = pmbp->pmesh->three_d;
  const int volume_ratio = three_d ? 8 : 4;
  const int counts[3] = {volume_ratio*kFineParticles, kFineParticles, kFineParticles};
  int owners[3] = {-1, -1, -1};
  int nnew = 0;
  for (int group=0; group<3; ++group) {
    Real x, y, z;
    InitialPosition(group, three_d, x, y, z);
    for (int m=0; m<pmbp->nmb_thispack; ++m) {
      auto size = pmbp->pmb->mb_size.h_view(m);
      if (x >= size.x1min && x < size.x1max &&
          y >= size.x2min && y < size.x2max &&
          z >= size.x3min && z < size.x3max) {
        owners[group] = m;
        nnew += counts[group];
      }
    }
  }
  creation->Resize(nnew);
  auto pr = creation->prtcl_rdata;
  auto pi = creation->prtcl_idata;
  int offset = 0;
  for (int group=0; group<3; ++group) {
    const int m = owners[group];
    if (m < 0) continue;
    Real x, y, z;
    InitialPosition(group, three_d, x, y, z);
    auto size = pmbp->pmb->mb_size.h_view(m);
    const int gid = pmbp->gids + m;
    const int source_cell = lagrangian_mc::EncodeSourceCell(
        pmbp->pmb->mb_lev.h_view(m), static_cast<int>((x-size.x1min)/size.dx1),
        static_cast<int>((y-size.x2min)/size.dx2),
        three_d ? static_cast<int>((z-size.x3min)/size.dx3) : 0);
    par_for("particle_counterflow_init", DevExeSpace(), offset, offset+counts[group]-1,
    KOKKOS_LAMBDA(const int p) {
      pr(lagrangian_mc::IPX,p) = x;
      pr(lagrangian_mc::IPY,p) = y;
      pr(lagrangian_mc::IPZ,p) = z;
      pr(lagrangian_mc::IPXMIN,p) = x;
      pr(lagrangian_mc::IPYMIN,p) = y;
      pr(lagrangian_mc::IPZMIN,p) = z;
      pr(lagrangian_mc::IPTMIN,p) = 0.0;
      pi(lagrangian_mc::PGID,p) = gid;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_NONE;
      pi(lagrangian_mc::PSOURCECELL,p) = source_cell;
    });
    offset += counts[group];
  }
}

void PrescribeCounterflowFlux(Mesh *pm, const Real) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &mbsize = pmbp->pmb->mb_size;
  auto &mblev = pmbp->pmb->mb_lev;
  auto flux1 = pmbp->phydro->density_flux_integral.x1f;
  auto flux2 = pmbp->phydro->density_flux_integral.x2f;
  auto flux3 = pmbp->phydro->density_flux_integral.x3f;
  const bool three_d = pm->three_d;
  const int root_level = pm->root_level;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const bool second_step = pm->ncycle > 0;
  const Real outward = second_step ? 0.5*outward_fraction : outward_fraction;

  // This is a particle-operator test, not a fluid-evolution test. Supply integrated
  // fluxes after each RK stage, leaving the uniform background density at one.
  // On the second step swap the fine donors and change the outgoing amount, so stale
  // communication buffers cannot satisfy the same expected particle moves.
  Kokkos::deep_copy(flux1, 0.0);
  Kokkos::deep_copy(flux2, 0.0);
  Kokkos::deep_copy(flux3, 0.0);
  par_for("particle_counterflow_flux", DevExeSpace(), 0, pmbp->nmb_thispack-1,
          ks, indcs.ke, js, indcs.je, is, indcs.ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    auto size = mbsize.d_view(m);
    const Real x = LeftEdgeX(i-is, nx1, size.x1min, size.x1max);
    if (x != 0.0) return;
    const Real y = CellCenterX(j-js, nx2, size.x2min, size.x2max);
    const Real z = three_d ? CellCenterX(k-ks, nx3, size.x3min, size.x3max) : 0.0;
    if (mblev.d_view(m) == root_level) {
      if (y == -1.75 && (!three_d || z == -1.75)) {
        // Correct signed restriction includes fine-face area and coarse-cell volume.
        flux1(m,k,j,i) = (kInwardFraction-outward)/(three_d ? 8.0 : 4.0);
      }
    } else {
      const Real to_fine = second_step ? -1.625 : -1.875;
      const Real to_coarse = second_step ? -1.875 : -1.625;
      if (y == to_fine && (!three_d || z == to_fine)) flux1(m,k,j,i) = -outward;
      if (y == to_coarse && (!three_d || z == to_coarse)) {
        flux1(m,k,j,i) = kInwardFraction;
      }
    }
  });
}

void CounterflowOwnerRank(particles::ParticleOutputData *output) {
  auto pr = output->prtcl_rdata;
  auto pi = output->prtcl_idata;
  auto values = output->output_data;
  auto &mbsize = output->pmbp->pmb->mb_size;
  const int gids = output->pmbp->gids;
  const int nmb = output->pmbp->nmb_thispack;
  const int field = output->field_index;
  const int rank = global_variable::my_rank;
  const int npart = output->nprtcl;
  if (npart == 0) return;
  par_for("particle_counterflow_owner", DevExeSpace(), 0, npart-1,
  KOKKOS_LAMBDA(const int p) {
    // A negative rank flags invalid ownership; Python also checks actual rank changes.
    values(field,p) = -1.0;
    const int m = pi(lagrangian_mc::PGID,p) - gids;
    if (m < 0 || m >= nmb) return;
    auto size = mbsize.d_view(m);
    if (pr(lagrangian_mc::IPX,p) >= size.x1min &&
        pr(lagrangian_mc::IPX,p) < size.x1max &&
        pr(lagrangian_mc::IPY,p) >= size.x2min &&
        pr(lagrangian_mc::IPY,p) < size.x2max &&
        pr(lagrangian_mc::IPZ,p) >= size.x3min &&
        pr(lagrangian_mc::IPZ,p) < size.x3max) {
      values(field,p) = rank;
    }
  });
}

} // namespace

void ProblemGenerator::ParticleLagrangianMCCounterflow(ParameterInput *pin,
                                                     const bool restart) {
  auto *pmbp = pmy_mesh_->pmb_pack;
  if (restart || pmbp->phydro == nullptr || pmbp->ppart == nullptr ||
      !pmy_mesh_->multilevel || !user_srcs || pmy_mesh_->nprtcl_total != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Counterflow test requires a new Hydro SMR run with "
              << "problem/user_srcs=true and particles/ppc=0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->ppart->FindPopulation("particles")->particle_type !=
      ParticleType::lagrangian_mc) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Counterflow test requires Lagrangian MC particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  outward_fraction = pin->GetOrAddReal("problem", "outward_fraction", 0.25);
  if (!(outward_fraction > 0.0 && outward_fraction <= 0.5)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Counterflow test requires 0 < outward_fraction <= 0.5"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_srcs_func = PrescribeCounterflowFlux;
  user_particle_dt_func = CounterflowTimestep;
  user_initial_particle_injection_func = InitializeCounterflowParticles;
  EnrollParticleVTKOutputVariable("owner_rank", CounterflowOwnerRank);

  auto u0 = pmbp->phydro->u0;
  auto &indcs = pmy_mesh_->mb_indcs;
  Kokkos::deep_copy(u0, 0.0);
  par_for("particle_counterflow_fluid_init", DevExeSpace(), 0, pmbp->nmb_thispack-1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u0(m,IDN,k,j,i) = 1.0;
  });
}
