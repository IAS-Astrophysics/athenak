//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

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
#include "cosmic_ray.hpp"
#include "lagrangian_mc.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    dtnew(std::numeric_limits<float>::max()),
    lmc_random_seed(0),
    pmy_pack(ppack) {
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // read number of particles per cell, and calculate number of particles this pack
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
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
    std::string ppush = pin->GetString("particles","pusher");
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
      if (pmy_pack->phydro == nullptr || pmy_pack->pmhd != nullptr ||
          pmy_pack->pionn != nullptr || pmy_pack->prad != nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles currently require "
                  << "single-fluid Hydro" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (pmy_pack->pmesh->multilevel || pmy_pack->phydro->porb_u != nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles currently require a "
                  << "uniform mesh without orbital advection" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::string evolution = pin->GetString("time", "evolution");
      if (evolution.compare("dynamic") != 0 && evolution.compare("kinematic") != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Lagrangian MC particles require time-evolving Hydro"
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
      int random_seed = pin->GetOrAddInteger("particles", "random_seed", 0);
      if (random_seed < 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "particles/random_seed must be non-negative"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      lmc_random_seed = static_cast<std::uint64_t>(random_seed);
      pusher = ParticlesPusher::lagrangian_mc;
      pmy_pack->phydro->EnableDensityFluxIntegral();
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
        break;
      }
    case ParticleType::lagrangian_mc:
      {
        nrdata = lagrangian_mc::NREAL;
        nidata = lagrangian_mc::NINT;
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

Particles::~Particles() {
  delete pbval_part;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::PurgeDeleted
//! \brief Remove particles marked for deletion and compact all particle data arrays.

TaskStatus Particles::PurgeDeleted(Driver*, int) {
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

  Mesh *pm = pmy_pack->pmesh;
  pm->nprtcl_thisrank = new_npart;
  pm->nprtcl_eachrank[global_variable::my_rank] = new_npart;
  if (global_variable::nranks == 1) pm->nprtcl_total = new_npart;

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack->pmesh->nprtcl_eachrank[n-1];
    }

    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = tagstart + p;
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = myrank + nranks*p;
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
