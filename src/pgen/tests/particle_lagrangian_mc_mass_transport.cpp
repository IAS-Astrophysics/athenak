//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file particle_lagrangian_mc_mass_transport.cpp
//! \brief End-to-end scientific test problem for mass-flux tracer particles.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "particles/lagrangian_mc.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace lagrangian_mc = particles::lagrangian_mc;

namespace {

struct MassTransportParameters {
  int target_count;
  int radial_bins;
  std::uint64_t random_seed;
  Real sample_radius;
  // Match the spherical Cartesian cutoff used by the production implementation.
  Real horizon_radius;
  Real density_contrast;
  Real inflow_speed;
};

struct MassTransportDiagnostics {
  Real initial_mass = 0.0;
  Real coarse_mass = 0.0;
  Real fine_mass = 0.0;
  Real cell_mass_min = 0.0;
  Real cell_mass_max = 0.0;
  Real initialization_zmax = 0.0;
  Real fluid_accreted = 0.0;
  int initial_count = 0;
  int coarse_count = 0;
  int fine_count = 0;
  int coarse_to_fine = 0;
  int retired = 0;
  int last_history_cycle = -1;
};

MassTransportParameters mass_transport;
MassTransportDiagnostics diagnostics;

KOKKOS_INLINE_FUNCTION
std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

KOKKOS_INLINE_FUNCTION
Real CellRandom(const std::uint64_t seed, const int gid, const int cell) {
  std::uint64_t key = seed;
  key ^= static_cast<std::uint64_t>(gid) * 0xd2b74407b1ce6e93ULL;
  key ^= static_cast<std::uint64_t>(cell) * 0x9e3779b97f4a7c15ULL;
  const std::uint64_t bits = SplitMix64(key);
#if SINGLE_PRECISION_ENABLED
  return static_cast<Real>(bits >> 40) * (1.0f/16777216.0f);
#else
  return static_cast<Real>(bits >> 11) * (1.0/9007199254740992.0);
#endif
}

void InitializeFlow(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  const Real log_contrast = std::log(mass_transport.density_contrast);
  const Real sample_radius = mass_transport.sample_radius;
  const Real horizon_radius = mass_transport.horizon_radius;
  const Real inflow_speed = mass_transport.inflow_speed;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  par_for("particle_mass_transport_fluid_init", DevExeSpace(), 0, nmb-1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    auto block = size.d_view(m);
    const Real x = CellCenterX(i-is, nx1, block.x1min, block.x1max);
    const Real y = CellCenterX(j-js, nx2, block.x2min, block.x2max);
    const Real z = CellCenterX(k-ks, nx3, block.x3min, block.x3max);
    const Real radius = sqrt(x*x + y*y + z*z);
    const Real profile = fmax(0.0, 1.0 - radius/sample_radius);
    const Real density = exp(log_contrast*profile);
    const Real pressure = 0.01*density;
    Real ur = -inflow_speed;
    if (radius <= 0.5*horizon_radius) ur = 0.0;

    w0(m,IDN,k,j,i) = density;
    w0(m,IEN,k,j,i) = pressure/gm1;
    w0(m,IVX,k,j,i) = (radius > 0.0) ? ur*x/radius : 0.0;
    w0(m,IVY,k,j,i) = (radius > 0.0) ? ur*y/radius : 0.0;
    w0(m,IVZ,k,j,i) = (radius > 0.0) ? ur*z/radius : 0.0;
    bcc0(m,IBX,k,j,i) = 0.0;
    bcc0(m,IBY,k,j,i) = 0.0;
    bcc0(m,IBZ,k,j,i) = 0.0;
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = 0.0;
    if (i == ie) b0.x1f(m,k,j,i+1) = 0.0;
    if (j == je) b0.x2f(m,k,j+1,i) = 0.0;
    if (k == ke) b0.x3f(m,k+1,j,i) = 0.0;
  });
  pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
}

void InitializeParticles(MeshBlockPack *pmbp,
                         particles::ParticleCreation *creation) {
  auto *population = creation->GetPopulation();
  if (population->particle_type != ParticleType::lagrangian_mc) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test requires Lagrangian MC particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pmbp->nmb_thispack*nkji;
  const int gids = pmbp->gids;
  const int root_level = pmbp->pmesh->root_level;
  const Real sample_radius = mass_transport.sample_radius;
  const Real horizon_radius = mass_transport.horizon_radius;
  auto &size = pmbp->pmb->mb_size;
  auto &levels = pmbp->pmb->mb_lev;
  auto &u0 = pmbp->pmhd->u0;

  DualArray1D<Real> cell_mass("particle_mass_transport_cell_mass", ncell);
  DualArray1D<Real> cell_radius("particle_mass_transport_cell_radius", ncell);
  DualArray1D<int> cell_level("particle_mass_transport_cell_level", ncell);
  par_for("particle_mass_transport_measure_cells", DevExeSpace(), 0, ncell-1,
  KOKKOS_LAMBDA(const int idx) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = idx - m*nkji - k*nji - j*nx1 + is;
    j += js;
    k += ks;
    auto block = size.d_view(m);
    const Real x = CellCenterX(i-is, nx1, block.x1min, block.x1max);
    const Real y = CellCenterX(j-js, nx2, block.x2min, block.x2max);
    const Real z = CellCenterX(k-ks, nx3, block.x3min, block.x3max);
    const Real radius = sqrt(x*x + y*y + z*z);
    const Real volume = block.dx1*block.dx2*block.dx3;
    cell_radius.d_view(idx) = radius;
    cell_level.d_view(idx) = levels.d_view(m);
    cell_mass.d_view(idx) = (radius > horizon_radius && radius <= sample_radius)
        ? u0(m,IDN,k,j,i)*volume : 0.0;
  });
  cell_mass.template modify<DevExeSpace>();
  cell_radius.template modify<DevExeSpace>();
  cell_level.template modify<DevExeSpace>();
  cell_mass.template sync<HostMemSpace>();
  cell_radius.template sync<HostMemSpace>();
  cell_level.template sync<HostMemSpace>();

  // Keep the flux-based accretion reference independent of SMR face subdivision. Every
  // cell that can enter its one-cell horizon stencil must be on the same refined level.
  int local_horizon_level_min = std::numeric_limits<int>::max();
  int local_horizon_level_max = std::numeric_limits<int>::min();
  for (int idx=0; idx<ncell; ++idx) {
    const int m = idx/nkji;
    auto block = size.h_view(m);
    const Real stencil_width = std::max(block.dx1, std::max(block.dx2, block.dx3));
    if (cell_radius.h_view(idx) <= horizon_radius + stencil_width) {
      local_horizon_level_min = std::min(
          local_horizon_level_min, cell_level.h_view(idx));
      local_horizon_level_max = std::max(
          local_horizon_level_max, cell_level.h_view(idx));
    }
  }
  int horizon_level_min = local_horizon_level_min;
  int horizon_level_max = local_horizon_level_max;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &horizon_level_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &horizon_level_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (horizon_level_max == std::numeric_limits<int>::min() ||
      horizon_level_min != horizon_level_max || horizon_level_min <= root_level) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test requires the horizon "
              << "flux stencil to lie on one refined level" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real local_mass = 0.0;
  for (int idx=0; idx<ncell; ++idx) local_mass += cell_mass.h_view(idx);
  Real global_mass = local_mass;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_mass, &global_mass, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (!(global_mass > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test sampled no positive "
              << "coordinate mass"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real tracer_mass = global_mass/static_cast<Real>(mass_transport.target_count);

  DualArray1D<int> cell_count("particle_mass_transport_cell_count", ncell);
  const std::uint64_t random_seed = mass_transport.random_seed;
  par_for("particle_mass_transport_count_particles", DevExeSpace(), 0, ncell-1,
  KOKKOS_LAMBDA(const int idx) {
    const Real expected = cell_mass.d_view(idx)/tracer_mass;
    int count = static_cast<int>(expected);
    const Real fraction = expected - static_cast<Real>(count);
    const int m = idx/nkji;
    if (CellRandom(random_seed, gids + m, idx - m*nkji) < fraction) ++count;
    cell_count.d_view(idx) = count;
  });
  cell_count.template modify<DevExeSpace>();
  cell_count.template sync<HostMemSpace>();

  DualArray1D<int> cell_offset("particle_mass_transport_cell_offset", ncell);
  int local_count = 0;
  for (int idx=0; idx<ncell; ++idx) {
    cell_offset.h_view(idx) = local_count;
    local_count += cell_count.h_view(idx);
  }
  cell_offset.template modify<HostMemSpace>();
  cell_offset.template sync<DevMemSpace>();

  const int nbins = 2*mass_transport.radial_bins;
  std::vector<Real> bin_mass(nbins, 0.0);
  std::vector<int> bin_count(nbins, 0);
  Real local_min = std::numeric_limits<Real>::max();
  Real local_max = 0.0;
  Real local_coarse_mass = 0.0;
  Real local_fine_mass = 0.0;
  int local_coarse_count = 0;
  int local_fine_count = 0;
  for (int idx=0; idx<ncell; ++idx) {
    const Real mass = cell_mass.h_view(idx);
    if (!(mass > 0.0)) continue;
    local_min = std::min(local_min, mass);
    local_max = std::max(local_max, mass);
    const bool fine = cell_level.h_view(idx) > root_level;
    if (fine) {
      local_fine_mass += mass;
      local_fine_count += cell_count.h_view(idx);
    } else {
      local_coarse_mass += mass;
      local_coarse_count += cell_count.h_view(idx);
    }
    int radial_bin = static_cast<int>(
        cell_radius.h_view(idx)/sample_radius*static_cast<Real>(mass_transport.radial_bins));
    radial_bin = std::min(radial_bin, mass_transport.radial_bins - 1);
    const int bin = radial_bin + (fine ? mass_transport.radial_bins : 0);
    bin_mass[bin] += mass;
    bin_count[bin] += cell_count.h_view(idx);
  }

  Real global_min = local_min;
  Real global_max = local_max;
  int global_count = local_count;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_min, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, bin_mass.data(), nbins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, bin_count.data(), nbins, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  Real initialization_zmax = 0.0;
  for (int bin=0; bin<nbins; ++bin) {
    if (!(bin_mass[bin] > 0.0)) continue;
    const Real probability = bin_mass[bin]/global_mass;
    const Real expected = static_cast<Real>(global_count)*probability;
    const Real sigma = sqrt(static_cast<Real>(global_count)*probability*
                            (1.0 - probability));
    const Real zscore = fabs(static_cast<Real>(bin_count[bin]) - expected)/
                        fmax(sigma, 1.0);
    initialization_zmax = fmax(initialization_zmax, zscore);
  }

  diagnostics.initial_mass = local_mass;
  diagnostics.coarse_mass = local_coarse_mass;
  diagnostics.fine_mass = local_fine_mass;
  diagnostics.initial_count = local_count;
  diagnostics.coarse_count = local_coarse_count;
  diagnostics.fine_count = local_fine_count;
  if (global_variable::my_rank == 0) {
    diagnostics.cell_mass_min = global_min;
    diagnostics.cell_mass_max = global_max;
    diagnostics.initialization_zmax = initialization_zmax;
  }

  creation->Resize(local_count);
  if (local_count == 0) return;

  auto pr = creation->prtcl_rdata;
  auto pi = creation->prtcl_idata;
  const Real initial_time = pmbp->pmesh->time;
  par_for("particle_mass_transport_initialize_particles", DevExeSpace(), 0, ncell-1,
  KOKKOS_LAMBDA(const int idx) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = idx - m*nkji - k*nji - j*nx1 + is;
    j += js;
    k += ks;
    auto block = size.d_view(m);
    const Real x = CellCenterX(i-is, nx1, block.x1min, block.x1max);
    const Real y = CellCenterX(j-js, nx2, block.x2min, block.x2max);
    const Real z = CellCenterX(k-ks, nx3, block.x3min, block.x3max);
    const int offset = cell_offset.d_view(idx);
    for (int q=0; q<cell_count.d_view(idx); ++q) {
      const int p = offset + q;
      pi(lagrangian_mc::PGID,p) = gids + m;
      pi(lagrangian_mc::PLASTMOVE,p) = lagrangian_mc::PMOVE_NONE;
      pi(lagrangian_mc::PSOURCECELL,p) = lagrangian_mc::EncodeSourceCell(
          levels.d_view(m), i-is, j-js, k-ks);
      pr(lagrangian_mc::IPX,p) = x;
      pr(lagrangian_mc::IPY,p) = y;
      pr(lagrangian_mc::IPZ,p) = z;
      pr(lagrangian_mc::IPXMIN,p) = x;
      pr(lagrangian_mc::IPYMIN,p) = y;
      pr(lagrangian_mc::IPZMIN,p) = z;
      pr(lagrangian_mc::IPTMIN,p) = initial_time;
    }
  });
}

void ApplyHorizonPostUpdate(particles::ParticleLifecycleData *lifecycle) {
  auto pr = lifecycle->prtcl_rdata;
  auto pi = lifecycle->prtcl_idata;
  const int npart = lifecycle->nprtcl;
  const Real horizon_sq = mass_transport.horizon_radius*mass_transport.horizon_radius;

  int newly_retired = 0;
  Kokkos::parallel_reduce(
      "particle_mass_transport_horizon", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &count) {
    if (lagrangian_mc::SourceCorrectionPending(
            pi(lagrangian_mc::PSOURCECELL,p))) {
      Kokkos::abort("Post-update particle callback ran before SMR correction");
    }
    if (pi(lagrangian_mc::PSTATUS,p) != PACTIVE) return;
    const Real x = pr(lagrangian_mc::IPX,p);
    const Real y = pr(lagrangian_mc::IPY,p);
    const Real z = pr(lagrangian_mc::IPZ,p);
    const Real radius_sq = x*x + y*y + z*z;
    if (radius_sq <= horizon_sq) {
      pi(lagrangian_mc::PSTATUS,p) = PDELETE_AFTER_SNAPSHOT;
      ++count;
    }
  }, newly_retired);
  diagnostics.retired += newly_retired;
}

Real AccretedFluidMass(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int ncell = pmbp->nmb_thispack*nkji;
  const Real horizon_sq = mass_transport.horizon_radius*mass_transport.horizon_radius;
  auto &size = pmbp->pmb->mb_size;
  auto &flux = pmbp->pmhd->density_flux_integral;
  Real accreted = 0.0;
  Kokkos::parallel_reduce(
      "particle_mass_transport_accreted_fluid", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(const int idx, Real &sum) {
        const int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        const int i = idx - m*nkji - k*nji - j*nx1 + is;
        j += js;
        k += ks;
        auto block = size.d_view(m);
        const Real x = CellCenterX(i-is, nx1, block.x1min, block.x1max);
        const Real y = CellCenterX(j-js, nx2, block.x2min, block.x2max);
        const Real z = CellCenterX(k-ks, nx3, block.x3min, block.x3max);
        if (x*x + y*y + z*z <= horizon_sq) return;

        Real inward = 0.0;
        if ((x-block.dx1)*(x-block.dx1) + y*y + z*z <= horizon_sq) {
          inward += fmax(-flux.x1f(m,k,j,i), 0.0);
        }
        if ((x+block.dx1)*(x+block.dx1) + y*y + z*z <= horizon_sq) {
          inward += fmax(flux.x1f(m,k,j,i+1), 0.0);
        }
        if (x*x + (y-block.dx2)*(y-block.dx2) + z*z <= horizon_sq) {
          inward += fmax(-flux.x2f(m,k,j,i), 0.0);
        }
        if (x*x + (y+block.dx2)*(y+block.dx2) + z*z <= horizon_sq) {
          inward += fmax(flux.x2f(m,k,j+1,i), 0.0);
        }
        if (x*x + y*y + (z-block.dx3)*(z-block.dx3) <= horizon_sq) {
          inward += fmax(-flux.x3f(m,k,j,i), 0.0);
        }
        if (x*x + y*y + (z+block.dx3)*(z+block.dx3) <= horizon_sq) {
          inward += fmax(flux.x3f(m,k+1,j,i), 0.0);
        }
        sum += inward*block.dx1*block.dx2*block.dx3;
      }, accreted);
  return accreted;
}

int CoarseToFineThisCycle(Mesh *pm) {
  auto *population = pm->pmb_pack->ppart->FindPopulation("particles");
  auto pi = population->prtcl_idata;
  auto &levels = pm->pmb_pack->pmb->mb_lev;
  const int gids = pm->pmb_pack->gids;
  const int gide = pm->pmb_pack->gide;
  const int npart = population->nprtcl_thispack;
  int count = 0;
  Kokkos::parallel_reduce(
      "particle_mass_transport_coarse_to_fine", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &sum) {
        const int gid = pi(lagrangian_mc::PGID,p);
        if (gid >= gids && gid <= gide &&
            levels.d_view(gid-gids) > lagrangian_mc::SourceLevel(
                pi(lagrangian_mc::PSOURCECELL,p))) {
          ++sum;
        }
      }, count);
  return count;
}

void MassTransportHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 16;
  pdata->label[0] = "init_mass";
  pdata->label[1] = "fluid_acc";
  pdata->label[2] = "tracer_ini";
  pdata->label[3] = "tracer_acc";
  pdata->label[4] = "cell_mmin";
  pdata->label[5] = "cell_mmax";
  pdata->label[6] = "crs_mass";
  pdata->label[7] = "fin_mass";
  pdata->label[8] = "crs_n";
  pdata->label[9] = "fin_n";
  pdata->label[10] = "init_zmax";
  pdata->label[11] = "c2f_count";
  pdata->label[12] = "owner_err";
  pdata->label[13] = "status_err";
  pdata->label[14] = "count_err";
  pdata->label[15] = "min_err";

  auto *population = pm->pmb_pack->ppart->FindPopulation("particles");
  if (pm->ncycle > diagnostics.last_history_cycle) {
    if (pm->ncycle > 0) {
      diagnostics.fluid_accreted += AccretedFluidMass(pm);
      diagnostics.coarse_to_fine += CoarseToFineThisCycle(pm);
    }
    diagnostics.last_history_cycle = pm->ncycle;
  }

  const int retired = diagnostics.retired;

  auto pr = population->prtcl_rdata;
  auto pi = population->prtcl_idata;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const int gids = pm->pmb_pack->gids;
  const int gide = pm->pmb_pack->gide;
  const int npart = population->nprtcl_thispack;
  const Real output_time = pm->time;

  Real owner_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_mass_transport_owner_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &sum) {
        const int gid = pi(lagrangian_mc::PGID,p);
        if (gid < gids || gid > gide) {
          sum += 1.0;
          return;
        }
        auto block = size.d_view(gid-gids);
        if (!(pr(lagrangian_mc::IPX,p) >= block.x1min &&
              pr(lagrangian_mc::IPX,p) < block.x1max &&
              pr(lagrangian_mc::IPY,p) >= block.x2min &&
              pr(lagrangian_mc::IPY,p) < block.x2max &&
              pr(lagrangian_mc::IPZ,p) >= block.x3min &&
              pr(lagrangian_mc::IPZ,p) < block.x3max)) {
          sum += 1.0;
        }
      }, owner_error);

  int active = 0;
  Real status_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_mass_transport_status", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &sum) {
        const int status = pi(lagrangian_mc::PSTATUS,p);
        if (status == PACTIVE) {
          sum += 0.0;
        } else if (status != PDELETE_AFTER_SNAPSHOT && status != PDELETE_PENDING) {
          sum += 1.0;
        }
      }, status_error);
  Kokkos::parallel_reduce(
      "particle_mass_transport_active", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &sum) {
        if (pi(lagrangian_mc::PSTATUS,p) == PACTIVE) ++sum;
      }, active);

  Real minimum_error = 0.0;
  Kokkos::parallel_reduce(
      "particle_mass_transport_minimum_error", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &sum) {
        if (!Kokkos::isfinite(pr(lagrangian_mc::IPXMIN,p)) ||
            !Kokkos::isfinite(pr(lagrangian_mc::IPYMIN,p)) ||
            !Kokkos::isfinite(pr(lagrangian_mc::IPZMIN,p)) ||
            pr(lagrangian_mc::IPTMIN,p) < 0.0 ||
            !Kokkos::isfinite(pr(lagrangian_mc::IPTMIN,p)) ||
            pr(lagrangian_mc::IPTMIN,p) > output_time + 1.0e-12) {
          sum += 1.0;
        }
      }, minimum_error);

  pdata->hdata[0] = diagnostics.initial_mass;
  pdata->hdata[1] = diagnostics.fluid_accreted;
  pdata->hdata[2] = static_cast<Real>(diagnostics.initial_count);
  pdata->hdata[3] = static_cast<Real>(retired);
  pdata->hdata[4] = diagnostics.cell_mass_min;
  pdata->hdata[5] = diagnostics.cell_mass_max;
  pdata->hdata[6] = diagnostics.coarse_mass;
  pdata->hdata[7] = diagnostics.fine_mass;
  pdata->hdata[8] = static_cast<Real>(diagnostics.coarse_count);
  pdata->hdata[9] = static_cast<Real>(diagnostics.fine_count);
  pdata->hdata[10] = diagnostics.initialization_zmax;
  pdata->hdata[11] = static_cast<Real>(diagnostics.coarse_to_fine);
  pdata->hdata[12] = owner_error;
  pdata->hdata[13] = status_error;
  pdata->hdata[14] = static_cast<Real>(active + retired -
                                       diagnostics.initial_count);
  pdata->hdata[15] = minimum_error;
}

} // namespace

void ProblemGenerator::ParticleLagrangianMCMassTransport(
    ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr || pmbp->pmhd == nullptr || pmbp->phydro != nullptr ||
      pmbp->pdyngr != nullptr || !pmbp->pcoord->is_general_relativistic ||
      pmbp->pcoord->coord_data.is_minkowski ||
      !pmbp->pcoord->coord_data.bh_excise || !pmy_mesh_->three_d ||
      !pmy_mesh_->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test requires fixed-spacetime, 3D MHD "
              << "with static refinement and particles" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *population = pmbp->ppart->FindPopulation("particles");
  if (population->particle_type != ParticleType::lagrangian_mc) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test requires "
              << "particle_type=lagrangian_mc"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  mass_transport.target_count = pin->GetInteger("problem", "particle_target_count");
  mass_transport.radial_bins = pin->GetInteger("problem", "particle_radial_bins");
  mass_transport.random_seed = static_cast<std::uint64_t>(
      pin->GetInteger("particles", "random_seed"));
  mass_transport.sample_radius = pin->GetReal("problem", "particle_sample_radius");
  mass_transport.horizon_radius = pin->GetReal("problem", "particle_horizon_radius");
  mass_transport.density_contrast = pin->GetReal("problem", "density_contrast");
  mass_transport.inflow_speed = pin->GetOrAddReal("problem", "inflow_speed", 0.25);
  const Real spin = pmbp->pcoord->coord_data.bh_spin;
  const Real event_horizon = 1.0 + sqrt(1.0 - spin*spin);
  const Real horizon_tolerance =
      64.0*std::numeric_limits<Real>::epsilon()*event_horizon;
  if (mass_transport.target_count <= 0 || mass_transport.radial_bins <= 1 ||
      !(mass_transport.sample_radius > mass_transport.horizon_radius) ||
      !(mass_transport.horizon_radius > 0.0) ||
      fabs(mass_transport.horizon_radius - event_horizon) > horizon_tolerance ||
      !(mass_transport.density_contrast >= 16.0) || !(mass_transport.inflow_speed > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test parameters are invalid"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  user_hist_func = MassTransportHistory;
  user_initial_particle_injection_func = InitializeParticles;
  user_particle_post_update_func = ApplyHorizonPostUpdate;
  diagnostics = MassTransportDiagnostics{};

  if (restart) return;
  if (pin->GetReal("particles", "ppc") != 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Lagrangian MC mass-transport test requires particles/ppc=0"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  InitializeFlow(pmy_mesh_);
}
