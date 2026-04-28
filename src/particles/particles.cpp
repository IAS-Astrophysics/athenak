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
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
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
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("null_geodesic") == 0) {
      pusher = ParticlesPusher::null_geodesic;
      if (pmy_pack->padm == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "null_geodesic particle pusher requires <adm> "
                  << "or <z4c> so it can use ADM metric fields" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
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
        int ndim=4;
        if (pmy_pack->pmesh->three_d) {ndim+=2;}
        nrdata = ndim;
        nidata = 2;
        break;
      }
    default:
      break;
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  particle_cfl = pin->GetOrAddReal("particles", "cfl", 0.4);
  dtnew = std::numeric_limits<Real>::max();
  for (int m=0; m<pmy_pack->nmb_thispack; ++m) {
    dtnew = std::min(dtnew, pmy_pack->pmb->mb_size.h_view(m).dx1);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, pmy_pack->pmb->mb_size.h_view(m).dx2);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, pmy_pack->pmb->mb_size.h_view(m).dx3);
    }
  }
  dtnew *= particle_cfl;

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::NewTimeStep()
//! \brief Compute a particle timestep cap.  Null-geodesic particles move with ADM
//! coordinate light speeds, not necessarily with unit coordinate speed.

TaskStatus Particles::NewTimeStep(Driver *pdriver, int stage) {
  (void) pdriver;
  (void) stage;

  const Real max_dt = std::numeric_limits<Real>::max();
  const Real tiny = std::numeric_limits<Real>::min();

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  Real dt_min = max_dt;
  if (pusher == ParticlesPusher::null_geodesic) {
    auto &adm_vars = pmy_pack->padm->adm;
    Kokkos::parallel_reduce("part_null_newdt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      const Real gxx = adm_vars.g_dd(m,0,0,k,j,i);
      const Real gxy = adm_vars.g_dd(m,0,1,k,j,i);
      const Real gxz = adm_vars.g_dd(m,0,2,k,j,i);
      const Real gyy = adm_vars.g_dd(m,1,1,k,j,i);
      const Real gyz = adm_vars.g_dd(m,1,2,k,j,i);
      const Real gzz = adm_vars.g_dd(m,2,2,k,j,i);
      const Real det = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
      const Real detinv = (fabs(det) > tiny) ? 1.0/det : 1.0/tiny;
      Real uxx, uxy, uxz, uyy, uyz, uzz;
      adm::SpatialInv(detinv, gxx, gxy, gxz, gyy, gyz, gzz,
                      &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);

      const Real alpha = adm_vars.alpha(m,k,j,i);
      const Real c1 = fabs(adm_vars.beta_u(m,0,k,j,i)) +
                      alpha*sqrt(fmax(uxx, 0.0));
      min_dt = fmin(min_dt, mbsize.d_view(m).dx1/fmax(c1, tiny));

      if (multi_d) {
        const Real c2 = fabs(adm_vars.beta_u(m,1,k,j,i)) +
                        alpha*sqrt(fmax(uyy, 0.0));
        min_dt = fmin(min_dt, mbsize.d_view(m).dx2/fmax(c2, tiny));
      }
      if (three_d) {
        const Real c3 = fabs(adm_vars.beta_u(m,2,k,j,i)) +
                        alpha*sqrt(fmax(uzz, 0.0));
        min_dt = fmin(min_dt, mbsize.d_view(m).dx3/fmax(c3, tiny));
      }
    }, Kokkos::Min<Real>(dt_min));
  } else {
    Kokkos::parallel_reduce("part_newdt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, pmy_pack->nmb_thispack),
    KOKKOS_LAMBDA(const int &m, Real &min_dt) {
      min_dt = fmin(min_dt, mbsize.d_view(m).dx1);
      if (multi_d) { min_dt = fmin(min_dt, mbsize.d_view(m).dx2); }
      if (three_d) { min_dt = fmin(min_dt, mbsize.d_view(m).dx3); }
    }, Kokkos::Min<Real>(dt_min));
  }

  dtnew = particle_cfl*dt_min;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// CreatePaticleTags()
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
              << "Particle tag assinment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
