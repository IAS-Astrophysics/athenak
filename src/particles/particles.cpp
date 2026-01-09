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
#include <fstream>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"
#include "units/units.hpp"
#include "utils/sn_scheduler.hpp"

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

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {

      particle_type = ParticleType::cosmic_ray;

      // read number of particles per cell, and calculate number of particles this pack
      Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

      // compute number of particles as real number, since ppc can be < 1
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
      Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
      nprtcl_thispack = static_cast<int>(r_npart); // then cast to integer
						   
    } else if (ptype.compare("star") == 0) {

      particle_type = ParticleType::star;
      nprtcl_thispack = 0; // initialize to zero
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString("particles","pusher");
    if (ppush.compare("drift") == 0) {
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("rk4_gravity") == 0) {
      pusher = ParticlesPusher::rk4_gravity;
      // load gravity constants
      r_scale   = pin->GetReal("potential", "r_scale");
      rho_scale = pin->GetReal("potential", "rho_scale");
      m_gal     = pin->GetReal("potential", "mass_gal");
      a_gal     = pin->GetReal("potential", "scale_gal");
      z_gal     = pin->GetReal("potential", "z_gal");
      r_200     = pin->GetReal("potential", "r_200");
      rho_mean  = pin->GetReal("potential", "rho_mean");
      par_grav_dx = pin->GetOrAddReal("particles", "grav_dx", 1e-6);
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

  // stars must be 3D
  if (particle_type == ParticleType::star and not pmy_pack->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Star particles only work in 3D" <<std::endl;
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
    case ParticleType::star:
      {
        nrdata = 9;
        nidata = 3;
        break;
      }
    default:
      break;
  }
  
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);

  // Initialize Star particles
  if (particle_type == ParticleType::star) {
    // Check if we should load from particle restart file
    int prtcl_rst_flag = pin->GetInteger("particles","star_prtcl_rst_flag");
    bool loaded_from_restart = false;
 
    if (prtcl_rst_flag) {
      LoadFromRestart(pin, pmy_pack, this);
    } else {
      InitializeStars(pin);
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
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

//----------------------------------------------------------------------------------------
// InitializeStars()
// Initializes star particles by reading from file

void Particles::InitializeStars(ParameterInput *pin) {
  // Load particles from file
  std::string particle_file = pin->GetString("particles","star_particle_file");
  std::ifstream infile(particle_file);
  if (!infile) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unable to open particle file: " << particle_file
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Read particle positions from file
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;

  // Skip header comments
  while (infile.peek() == '#') {
    infile.ignore(1000, '\n');
  }

  // Build particle list for particles in this pack's meshblocks
  std::vector<std::array<Real, 9>> particle_list;
  std::array<Real, 8> p;
  while (infile >> p[0] >> p[1] >> p[2] >> p[3] >>
                   p[4] >> p[5] >> p[6] >> p[7]) {
    for (int m=0; m<nmb; ++m) {
      if (p[0] > size.h_view(m).x1min && p[0] <= size.h_view(m).x1max &&
          p[1] > size.h_view(m).x2min && p[1] <= size.h_view(m).x2max &&
          p[2] > size.h_view(m).x3min && p[2] <= size.h_view(m).x3max) {
        std::array<Real, 9> new_particle = {p[0], p[1], p[2],
                                            p[3], p[4], p[5],
                                            p[6], p[7], static_cast<Real>(m)};
        particle_list.push_back(new_particle);
      }
    }
  }
  infile.close();

  // Set particle count and allocate arrays
  nprtcl_thispack = particle_list.size();
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  //std::cout << "Loaded " << nprtcl_thispack
  //          << " star particles from file " << particle_file << std::endl;

  // Copy to device-accessible arrays
  HostArray2D<Real> host_pos("host_positions", 9, nprtcl_thispack);
  for (size_t i = 0; i < nprtcl_thispack; ++i) {
    for (size_t j = 0; j < 9; ++j) {
      host_pos(j, i) = particle_list[i][j];
    }
  }
  auto pos_data = Kokkos::create_mirror_view_and_copy(DevExeSpace(), host_pos);

  const int &gids = pmy_pack->gids;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  Real unit_time = pmy_pack->punit->time_cgs();

  // Initialize particles on device
  par_for("star_par", DevExeSpace(), 0, nprtcl_thispack-1,
  KOKKOS_LAMBDA(const int p) {
    int m = static_cast<int>(pos_data(8, p));
    pi(PGID,p) = gids + m;
    pi(NSN,p) = 0;
    pr(IPX,p)  = pos_data(0, p);
    pr(IPY,p)  = pos_data(1, p);
    pr(IPZ,p)  = pos_data(2, p);
    pr(IPVX,p) = pos_data(3, p);
    pr(IPVY,p) = pos_data(4, p);
    pr(IPVZ,p) = pos_data(5, p);
    pr(IPT_CREATE, p) = pos_data(6, p);
    pr(IPMASS, p)     = pos_data(7, p);
    pr(IPT_NEXT_SN,p) = GetNthSNTime(pr(IPMASS,p), pr(IPT_CREATE,p), unit_time, 0);
  });

  dtnew = std::min(size.h_view(0).dx1, size.h_view(0).dx2);
  dtnew = std::min(dtnew, size.h_view(0).dx3);
}

//----------------------------------------------------------------------------------------
// LoadFromRestart()
// Load particle data from restart file

void Particles::LoadFromRestart(ParameterInput *pin, MeshBlockPack *ppack,
                                Particles* ppart) {
  // Step 1: All ranks read the file header
  std::string restart_file = pin->GetString("particles","star_particle_file");
  std::ifstream prtcl_file(restart_file, std::ios::binary);
  if (!prtcl_file.is_open()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unable to open particle file: " << restart_file
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real restart_time;
  int restart_ncycle, total_particles;
  int nrdata_global = 0, nidata_global = 0;

  prtcl_file.read(reinterpret_cast<char*>(&restart_time), sizeof(Real));
  prtcl_file.read(reinterpret_cast<char*>(&restart_ncycle), sizeof(int));
  prtcl_file.read(reinterpret_cast<char*>(&total_particles), sizeof(int));

  if (total_particles == 0) {
    prtcl_file.close();
    ppart->nprtcl_thispack = 0;
    return;
  }  

  prtcl_file.read(reinterpret_cast<char*>(&nrdata_global), sizeof(int));
  prtcl_file.read(reinterpret_cast<char*>(&nidata_global), sizeof(int));

  // Step 2: All ranks read all particle data
  std::vector<Real> all_particle_rdata(total_particles * nrdata_global);
  std::vector<int> all_particle_idata(total_particles * nidata_global);

  prtcl_file.read(reinterpret_cast<char*>(all_particle_rdata.data()),
                  total_particles * nrdata_global * sizeof(Real));
  prtcl_file.read(reinterpret_cast<char*>(all_particle_idata.data()),
                  total_particles * nidata_global * sizeof(int));

  prtcl_file.close();
  
  if (global_variable::my_rank == 0) {
    std::cout << "Read " << total_particles << " particles from restart file: "
              << restart_file << std::endl;
  }
  
  // Step 3: Each rank filters particles that belong to its domain
  std::vector<Real> my_particle_rdata;
  std::vector<int> my_particle_idata;
  
  auto &size = ppack->pmb->mb_size;
  int nmb = ppack->nmb_thispack;
  const int &gids = ppack->gids;
  
  // Reserve space to avoid repeated allocations
  my_particle_rdata.reserve(total_particles * nrdata_global / global_variable::nranks);
  my_particle_idata.reserve(total_particles * nidata_global / global_variable::nranks);
  
  for (int p = 0; p < total_particles; ++p) {
    Real px = all_particle_rdata[p * nrdata_global + IPX];
    Real py = all_particle_rdata[p * nrdata_global + IPY];
    Real pz = all_particle_rdata[p * nrdata_global + IPZ];
    
    // Check if particle is in any of this rank's mesh blocks
    bool particle_belongs_here = false;
    int local_mb = -1;
    
    for (int m = 0; m < nmb; ++m) {
      if (px > size.h_view(m).x1min && px <= size.h_view(m).x1max &&
          py > size.h_view(m).x2min && py <= size.h_view(m).x2max &&
          pz > size.h_view(m).x3min && pz <= size.h_view(m).x3max) {
        particle_belongs_here = true;
        local_mb = m;
        break;
      }
    }
    
    if (particle_belongs_here) {
      // Copy particle data
      for (int i = 0; i < nrdata_global; ++i) {
        my_particle_rdata.push_back(all_particle_rdata[p * nrdata_global + i]);
      }
      for (int i = 0; i < nidata_global; ++i) {
        my_particle_idata.push_back(all_particle_idata[p * nidata_global + i]);
      }
      
      // Update particle's global ID to reflect new rank assignment
      int gid_idx = (my_particle_idata.size() / nidata_global - 1) * nidata_global + PGID;
      my_particle_idata[gid_idx] = gids + local_mb;
    }
  }
  
  // Step 4: Set up this rank's particle data
  ppart->nprtcl_thispack = my_particle_rdata.size() / nrdata_global;
  ppart->nrdata = nrdata_global;
  ppart->nidata = nidata_global;
  
  if (ppart->nprtcl_thispack > 0) {
    // Allocate and populate particle arrays
    Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, ppart->nprtcl_thispack);
    Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, ppart->nprtcl_thispack);
    
    auto rdata_host = Kokkos::create_mirror_view(ppart->prtcl_rdata);
    auto idata_host = Kokkos::create_mirror_view(ppart->prtcl_idata);
    
    // Copy data
    for (int p = 0; p < ppart->nprtcl_thispack; ++p) {
      for (int i = 0; i < ppart->nrdata; ++i) {
        rdata_host(i, p) = my_particle_rdata[p * ppart->nrdata + i];
      }
      for (int i = 0; i < ppart->nidata; ++i) {
        idata_host(i, p) = my_particle_idata[p * ppart->nidata + i];
      }
    }
    
    // Copy to device
    Kokkos::deep_copy(ppart->prtcl_rdata, rdata_host);
    Kokkos::deep_copy(ppart->prtcl_idata, idata_host);
  }

  if (ppart->nprtcl_thispack > 0) {
    auto rdata_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          ppart->prtcl_rdata);
    auto idata_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          ppart->prtcl_idata);
  }

  dtnew = std::min(size.h_view(0).dx1, size.h_view(0).dx2);
  dtnew = std::min(dtnew, size.h_view(0).dx3);
  
  //std::cout << "Rank " << global_variable::my_rank << " loaded "
  //          << ppart->nprtcl_thispack << " particles after redistribution" << std::endl;
  
  return;
}

} // namespace particles
