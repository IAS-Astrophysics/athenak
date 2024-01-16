#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations

// constants that enumerate ParticlesPusher options
enum class ParticlesPusher {drift, leap_frog, lagrangian_tracer, lagrangian_mc};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID crecv;
};

namespace particles {

//----------------------------------------------------------------------------------------
//! \class Particles

class Particles {
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin);
  ~Particles();

  // data
  ParticlesPusher pusher;

  int nparticles_total;                           // total number of particles all ranks
  int nparticles_thispack;                        // number of particles in this pack
  DualArray1D<int> prtcl_gid;                     // GID of MeshBlock containing each par
  DvceArray1D<Real> prtcl_x,  prtcl_y,  prtcl_z;  // positions
  DvceArray1D<Real> prtcl_vx, prtcl_vy, prtcl_vz; // velocities
  DvceArray2D<Real> prtcl_rprop;                  // real number properties each particle
  DvceArray2D<int> prtcl_iprop;                   // integer properties each particle

  // Boundary communication buffers and functions for particles
//  ParticlesBoundaryValues *pbval;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  TaskStatus Push(Driver *pdriver, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
