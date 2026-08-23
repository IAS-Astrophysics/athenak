//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_creation.cpp
//! \brief particle creation and injection implementation

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace particles {
namespace {

[[noreturn]] void CreationError(const char *file, int line, const char *message) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

struct MaxParticleTag {
  DvceArray2D<int> idata;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int p, int &max_tag) const {
    if (idata(PTAG,p) > max_tag) max_tag = idata(PTAG,p);
  }
};

} // namespace

//----------------------------------------------------------------------------------------
//! \brief Construct an empty creation batch for one particle population.

ParticleCreation::ParticleCreation(ParticlePopulation *population) :
    population_(population), nprtcl_(0) {
}

//----------------------------------------------------------------------------------------
//! \brief Allocate and zero temporary device storage for newly created particles.

void ParticleCreation::Resize(int count) {
  if (count < 0) {
    CreationError(__FILE__, __LINE__, "Particle creation count cannot be negative");
  }
  nprtcl_ = count;
  Kokkos::realloc(prtcl_rdata, population_->nrdata, nprtcl_);
  Kokkos::realloc(prtcl_idata, population_->nidata, nprtcl_);
  if (nprtcl_ > 0) {
    Kokkos::deep_copy(prtcl_rdata, 0.0);
    Kokkos::deep_copy(prtcl_idata, 0);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Invoke the user-enrolled particle creation function for every population.

std::int64_t Particles::RunParticleInjection(bool initial) {
  auto *pgen = pmy_pack_->pmesh->pgen.get();
  if (pgen == nullptr) return 0;

  UserParticleInjectionFnPtr injection_func = initial
      ? pgen->user_initial_particle_injection_func
      : pgen->user_particle_injection_func;
  if (injection_func == nullptr) return 0;

  std::int64_t ncreated = 0;
  for (auto *population : populations_) {
    ParticleCreation creation(population);
    injection_func(pmy_pack_, &creation);
    ncreated += AppendParticles(creation);
  }
  return ncreated;
}

//----------------------------------------------------------------------------------------
//! \brief Create passive particles after initial fluid primitive variables are available.

void Particles::InitialParticleInjection() {
  // A problem generator may deliberately replace the initial deterministic tags.
  next_tag_ = std::max(next_tag_, NextTagFromParticles());
  (void)RunParticleInjection(true);
}

//----------------------------------------------------------------------------------------
//! \brief Create passive particles after a completed step and refresh their timestep.

void Particles::InjectParticles(Driver *pdriver) {
  if (RunParticleInjection(false) > 0) {
    (void)NewTimeStep(pdriver, 1);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Append a collective batch of locally owned particles to one population.

std::int64_t Particles::AppendParticles(ParticleCreation &creation) {
  ParticlePopulation *population = creation.GetPopulation();
  bool known_population = false;
  for (const auto *candidate : populations_) {
    if (candidate == population) known_population = true;
  }
  if (!known_population) {
    CreationError(__FILE__, __LINE__,
                  "Cannot append particles to an unknown population");
  }

  const int nnew = creation.GetCount();
  const int gids = pmy_pack_->gids;
  const int gide = pmy_pack_->gide;
  if (nnew > 0) {
    Kokkos::fence();
    auto pr = creation.prtcl_rdata;
    auto pi = creation.prtcl_idata;
    auto &mbsize = pmy_pack_->pmb->mb_size;
    int invalid = 0;
    Kokkos::parallel_reduce(
        "validate_created_particles", Kokkos::RangePolicy<>(DevExeSpace(), 0, nnew),
        KOKKOS_LAMBDA(const int p, int &count) {
          const int gid = pi(PGID,p);
          if (gid < gids || gid > gide) {
            ++count;
          } else {
            const int m = gid - gids;
            const Real x1 = pr(IPX,p);
            const Real x2 = pr(IPY,p);
            const Real x3 = pr(IPZ,p);
            if (!(x1 >= mbsize.d_view(m).x1min && x1 < mbsize.d_view(m).x1max &&
                  x2 >= mbsize.d_view(m).x2min && x2 < mbsize.d_view(m).x2max &&
                  x3 >= mbsize.d_view(m).x3min && x3 < mbsize.d_view(m).x3max)) {
              ++count;
            }
          }
        }, invalid);
    if (invalid != 0) {
      CreationError(__FILE__, __LINE__,
                    "Created particles must belong to a local MeshBlock and lie inside "
                    "the assigned block");
    }
  }

  std::int64_t local_new = nnew;
  std::int64_t global_new = local_new;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_new, &global_new, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_new == 0) return 0;

  std::int64_t rank_offset = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Exscan(&local_new, &rank_offset, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  if (global_variable::my_rank == 0) rank_offset = 0;
#endif

  const std::int64_t max_next_tag =
      static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
  if (next_tag_ < 0 || next_tag_ > max_next_tag ||
      global_new > max_next_tag - next_tag_) {
    CreationError(__FILE__, __LINE__, "Particle tag range is exhausted");
  }

  if (nnew > std::numeric_limits<int>::max() - population->nprtcl_thispack) {
    CreationError(__FILE__, __LINE__,
                  "Local particle count exceeds the in-memory integer limit");
  }

  if (nnew > 0) {
    const int nold = population->nprtcl_thispack;
    const int ntotal = nold + nnew;
    const int nrdata = population->nrdata;
    const int nidata = population->nidata;
    const std::int64_t tag_start = next_tag_ + rank_offset;
    auto old_pr = population->prtcl_rdata;
    auto old_pi = population->prtcl_idata;
    auto add_pr = creation.prtcl_rdata;
    auto add_pi = creation.prtcl_idata;
    DvceArray2D<Real> new_pr("particle_rdata_append", nrdata, ntotal);
    DvceArray2D<int> new_pi("particle_idata_append", nidata, ntotal);
    par_for("particle_append",DevExeSpace(),0,(ntotal-1),
    KOKKOS_LAMBDA(const int p) {
      if (p < nold) {
        for (int n=0; n<nrdata; ++n) new_pr(n,p) = old_pr(n,p);
        for (int n=0; n<nidata; ++n) new_pi(n,p) = old_pi(n,p);
      } else {
        const int q = p - nold;
        for (int n=0; n<nrdata; ++n) new_pr(n,p) = add_pr(n,q);
        for (int n=0; n<nidata; ++n) new_pi(n,p) = add_pi(n,q);
        new_pi(PTAG,p) = static_cast<int>(tag_start + q);
        new_pi(PSTATUS,p) = PACTIVE;
      }
    });
    Kokkos::fence();
    population->prtcl_rdata = new_pr;
    population->prtcl_idata = new_pi;
    population->nprtcl_thispack = ntotal;
  }

  next_tag_ += global_new;
  pmy_pack_->pmesh->UpdateParticleCounts();
  return global_new;
}

//----------------------------------------------------------------------------------------
//! \brief Return one greater than the largest tag currently stored across all ranks.

std::int64_t Particles::NextTagFromParticles() const {
  int local_max = -1;
  for (const auto *population : populations_) {
    const int npart = population->nprtcl_thispack;
    if (npart == 0) continue;
    auto pi = population->prtcl_idata;
    int population_max = -1;
    Kokkos::parallel_reduce(
        "particle_max_tag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        MaxParticleTag{pi}, Kokkos::Max<int>(population_max));
    local_max = std::max(local_max, population_max);
  }

  int global_max = local_max;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  return static_cast<std::int64_t>(global_max) + 1;
}

} // namespace particles
