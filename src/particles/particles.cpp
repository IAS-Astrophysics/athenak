//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of the particle manager and population lifecycle

#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "cosmic_ray.hpp"
#include "lagrangian_mc.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// Particles constructor

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin, bool is_restart) :
    next_tag_(0),
    tag_assignment_(pin->GetOrAddString("particles","assign_tag","index_order")),
    restart_sort_by_tag_(
        pin->GetOrAddBoolean("particles","restart_sort_by_tag",false)),
    pmy_pack_(ppack) {
  populations_.emplace_back(
      new ParticlePopulation("particles", "particles", ppack, pin, is_restart));
}

//----------------------------------------------------------------------------------------
// Particles destructor

Particles::~Particles() {
  for (auto *population : populations_) {
    delete population;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Return a population by its stable name, or nullptr when it is not present.

ParticlePopulation* Particles::FindPopulation(const std::string &name) {
  for (auto *population : populations_) {
    if (population->name == name) return population;
  }
  return nullptr;
}

const ParticlePopulation* Particles::FindPopulation(const std::string &name) const {
  for (const auto *population : populations_) {
    if (population->name == name) return population;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \brief Return the aggregate local particle count over all populations.

int Particles::GetLocalCount() const {
  int count = 0;
  for (const auto *population : populations_) {
    count += population->nprtcl_thispack;
  }
  return count;
}

//----------------------------------------------------------------------------------------
//! \brief Return the tightest timestep estimate over all populations.

Real Particles::GetTimestep() const {
  Real dt = std::numeric_limits<float>::max();
  for (const auto *population : populations_) {
    dt = std::min(dt, population->dtnew);
  }
  return dt;
}

//----------------------------------------------------------------------------------------
// ParticlePopulation constructor, initializes data structures and parameters

ParticlePopulation::ParticlePopulation(const std::string &population_name,
                                       const std::string &input_block,
                                       MeshBlockPack *ppack, ParameterInput *pin,
                                       bool is_restart) :
    name(population_name),
    dtnew(std::numeric_limits<float>::max()),
    input_block_(input_block),
    lmc_random_seed(0),
    pmy_pack(ppack) {
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  nprtcl_thispack = 0;
  // read number of particles per cell on both fresh starts and restarts
  Real ppc = pin->GetOrAddReal(input_block_,"ppc",1.0);
  if (!is_restart) {
    // calculate number of particles in this pack
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
    Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
    // then cast to integer
    nprtcl_thispack = static_cast<int>(r_npart);
  }

  // select particle type
  {
    std::string ptype = pin->GetString(input_block_,"particle_type");
    type_name = ptype;
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
    } else if (ptype.compare("lagrangian_mc") == 0) {
      particle_type = ParticleType::lagrangian_mc;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString(input_block_,"pusher");
    if (ppush.compare("drift") == 0) {
      if (particle_type != ParticleType::cosmic_ray) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Particle pusher 'drift' requires particle type "
                  << "'cosmic_ray'" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("lagrangian_mc") == 0) {
      if (particle_type != ParticleType::lagrangian_mc) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Particle pusher 'lagrangian_mc' requires particle type "
                  << "'lagrangian_mc'" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const bool has_hydro = pmy_pack->phydro != nullptr;
      const bool has_mhd = pmy_pack->pmhd != nullptr;
      if ((!has_hydro && !has_mhd) || (has_hydro && has_mhd) ||
          pmy_pack->pionn != nullptr || pmy_pack->prad != nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles currently require "
                  << "single-fluid Hydro or MHD" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (pmy_pack->pdyngr != nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles currently do not support "
                  << "dynamical spacetime" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      bool orbital_advection = false;
      if (has_hydro) {
        orbital_advection = pmy_pack->phydro->porb_u != nullptr;
      } else {
        orbital_advection = pmy_pack->pmhd->porb_u != nullptr;
      }
      if (pmy_pack->pmesh->multilevel || orbital_advection) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles currently require a "
                  << "uniform mesh without orbital advection" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::string evolution = pin->GetString("time", "evolution");
      if (evolution.compare("dynamic") != 0 && evolution.compare("kinematic") != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles require a time-evolving fluid"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::string integrator = pin->GetOrAddString("time", "integrator", "rk2");
      if (integrator.compare("rk2") != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles currently require RK2"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // The fluid solver limits each directional Courant number; one-hop transport requires
      // their sum to be no larger than one.
      const int ndim = pmy_pack->pmesh->three_d ? 3 : 2;
      const Real cfl_number = pin->GetReal("time", "cfl_number");
      const Real max_cfl_number = 1.0/static_cast<Real>(ndim);
      if (!(cfl_number <= max_cfl_number)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles require time/cfl_number <= 1/"
                  << ndim << " in " << ndim << "D, but received " << cfl_number
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      int random_seed = pin->GetOrAddInteger(input_block_, "random_seed", 0);
      if (random_seed < 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "particles/random_seed must be non-negative"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      lmc_random_seed = static_cast<std::uint64_t>(random_seed);
      pusher = ParticlesPusher::lagrangian_mc;
      if (has_hydro) {
        pmy_pack->phydro->EnableDensityFluxIntegral();
      } else {
        pmy_pack->pmhd->EnableDensityFluxIntegral();
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher = '" << ppush << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      {
        nrdata = cosmic_ray::NREAL;
        nidata = cosmic_ray::NINT;
        real_names = cosmic_ray::real_names;
        int_names = cosmic_ray::int_names;
        real_output = cosmic_ray::real_output;
        int_output = cosmic_ray::int_output;
        break;
      }
    case ParticleType::lagrangian_mc:
      {
        nrdata = lagrangian_mc::NREAL;
        nidata = lagrangian_mc::NINT;
        real_names = lagrangian_mc::real_names;
        int_names = lagrangian_mc::int_names;
        real_output = lagrangian_mc::real_output;
        int_output = lagrangian_mc::int_output;
        break;
      }
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type has no storage definition" << std::endl;
      std::exit(EXIT_FAILURE);
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
  auto status = Kokkos::subview(prtcl_idata, static_cast<int>(PSTATUS), Kokkos::ALL);
  Kokkos::deep_copy(status, static_cast<int>(PACTIVE));
  if (particle_type == ParticleType::lagrangian_mc) {
    auto last_move = Kokkos::subview(
        prtcl_idata, static_cast<int>(lagrangian_mc::PLASTMOVE), Kokkos::ALL);
    Kokkos::deep_copy(last_move, 0);
  }

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

ParticlePopulation::~ParticlePopulation() {
  delete pbval_part;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlePopulation::ApplyUserLifecycle
//! \brief Let the problem generator update particle state after the pusher.

TaskStatus ParticlePopulation::ApplyUserLifecycle(Driver*, int) {
  auto *pgen = pmy_pack->pmesh->pgen.get();
  if (pgen != nullptr && pgen->user_particle_lifecycle_func != nullptr) {
    ParticleLifecycleData lifecycle(
        pmy_pack, this, prtcl_rdata, prtcl_idata, nprtcl_thispack);
    pgen->user_particle_lifecycle_func(&lifecycle);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ParticlePopulation::PurgeDeleted
//! \brief Remove particles marked for deletion and compact all particle data arrays.

TaskStatus ParticlePopulation::PurgeDeleted(Driver*, int) {
  const int npart = nprtcl_thispack;
  if (npart == 0) return TaskStatus::complete;

  auto pi = prtcl_idata;
  int ndelete = 0;
  Kokkos::parallel_reduce(
      "particle_count_deleted", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &count) {
        if (pi(PSTATUS,p) == PDELETE_PENDING) ++count;
      }, ndelete);
  if (ndelete == 0) return TaskStatus::complete;

  const int new_npart = npart - ndelete;
  auto pr = prtcl_rdata;
  DvceArray2D<Real> new_pr("particle_rdata_compact", nrdata, new_npart);
  DvceArray2D<int> new_pi("particle_idata_compact", nidata, new_npart);

  const int nr = nrdata;
  const int ni = nidata;
  int ncopy = 0;
  Kokkos::parallel_scan(
      "particle_compact", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &offset, const bool final) {
        if (pi(PSTATUS,p) != PDELETE_PENDING) {
          if (final) {
            for (int n=0; n<nr; ++n) new_pr(n,offset) = pr(n,p);
            for (int n=0; n<ni; ++n) new_pi(n,offset) = pi(n,p);
          }
          ++offset;
        }
      }, ncopy);

  if (ncopy != new_npart) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle compaction copied an unexpected number of particles"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  prtcl_rdata = new_pr;
  prtcl_idata = new_pi;
  nprtcl_thispack = new_npart;

#if !MPI_PARALLEL_ENABLED
  pmy_pack->pmesh->UpdateParticleCounts();
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlePopulation::MarkSnapshotComplete
//! \brief Queue deferred particles for deletion by the next purge task.

void ParticlePopulation::MarkSnapshotComplete() {
  auto pi = prtcl_idata;
  const int npart = nprtcl_thispack;
  if (npart > 0) {
    par_for("particle_mark_snapshot_complete", DevExeSpace(), 0, npart-1,
    KOKKOS_LAMBDA(const int p) {
      if (pi(PSTATUS,p) == PDELETE_AFTER_SNAPSHOT) {
        pi(PSTATUS,p) = PDELETE_PENDING;
      }
    });
  }
}

//----------------------------------------------------------------------------------------
// Particles::CreateParticleTags()
// Assigns unique integer tags to particles and initializes the persistent high-water mark.

void Particles::CreateParticleTags() {
  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (tag_assignment_.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack_->pmesh->nprtcl_eachrank[n-1];
    }

    int population_offset = 0;
    for (auto *population : populations_) {
      auto &pi = population->prtcl_idata;
      const int offset = population_offset;
      const int npart = population->nprtcl_thispack;
      par_for("ptags",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        pi(PTAG,p) = tagstart + offset + p;
      });
      population_offset += npart;
    }

  // tags are assigned sequentially across ranks
  } else if (tag_assignment_.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    int population_offset = 0;
    for (auto *population : populations_) {
      auto &pi = population->prtcl_idata;
      const int offset = population_offset;
      const int npart = population->nprtcl_thispack;
      par_for("ptags",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        pi(PTAG,p) = myrank + nranks*(offset + p);
      });
      population_offset += npart;
    }

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << tag_assignment_ << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  next_tag_ = NextTagFromParticles();
}

} // namespace particles
