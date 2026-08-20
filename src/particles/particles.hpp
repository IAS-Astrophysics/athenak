#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for the particle manager and particle populations

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations

// constants that enumerate ParticlesPusher options
enum class ParticlesPusher {drift, leap_frog, lagrangian_tracer, lagrangian_mc};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray, lagrangian_mc};

// common array indices used by type-independent particle machinery
enum ParticleRealIndex {IPX=0, IPY=1, IPZ=2};
enum ParticleIntIndex {PGID=0, PTAG=1, PSTATUS=2};

// common particle lifecycle states
enum ParticleStatus {PACTIVE=0, PFROZEN=1, PDELETE_PENDING=2};

//----------------------------------------------------------------------------------------
//! \struct ParticleTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticleTaskIDs {
  TaskID push;
  TaskID purge;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID csend;
  TaskID crecv;
  TaskID newdt;
};

namespace particles {

//----------------------------------------------------------------------------------------
//! \class ParticlePopulation
//! \brief Homogeneous particle storage, physics, and communication state.

class ParticlePopulation {
  friend class ParticlesBoundaryValues;
 public:
  ParticlePopulation(const std::string &population_name,
                     const std::string &input_block,
                     MeshBlockPack *ppack, ParameterInput *pin, bool is_restart);
  ~ParticlePopulation();

  // data
  std::string name;
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
  const char *const *real_names;   // output names in prtcl_rdata index order
  const char *const *int_names;    // output names in prtcl_idata index order
  const bool *real_output;         // fields included in particle snapshots
  const bool *int_output;
//  DvceArray1D<int>  prtcl_gid;     // GID of MeshBlock containing each par
//  DvceArray2D<Real> prtcl_pos;     // positions
//  DvceArray2D<Real> prtcl_vel;     // velocities
  DvceArray2D<Real> prtcl_rdata;   // positions followed by type-specific real properties
  DvceArray2D<int>  prtcl_idata;   // gid, tag, status, then type-specific properties
  Real dtnew;

  ParticlesPusher pusher;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticleTaskIDs id;

  // functions...
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus Push(Driver *pdriver, int stage);
  TaskStatus PurgeDeleted(Driver *pdriver, int stage);
  TaskStatus NewGID(Driver *pdriver, int stage);
  TaskStatus SendCnt(Driver *pdriver, int stage);
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus SendP(Driver *pdriver, int stage);
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);

  // particle pusher implementations
  TaskStatus PushDrift(Driver *pdriver, int stage);
  TaskStatus PushLagrangianMC(Driver *pdriver, int stage);
  Real EstimateTimestepDrift();

  int RestartLayoutVersion() const;
  std::vector<char> RestartMetadata() const;
  void ValidateRestartMetadata(const std::vector<char> &metadata) const;

 private:
  std::string input_block_;
  std::uint64_t lmc_random_seed;
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this population
};

//----------------------------------------------------------------------------------------
//! \class Particles
//! \brief Owns and coordinates all particle populations on a MeshBlockPack.

class Particles {
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin, bool is_restart);
  ~Particles();

  ParticlePopulation* FindPopulation(const std::string &name);
  const ParticlePopulation* FindPopulation(const std::string &name) const;

  int GetLocalCount() const;
  Real GetTimestep() const;
  void CreateParticleTags();
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);
  void WriteRestart(const std::string &filename) const;
  void LoadRestart(const std::string &filename);

 private:
  std::vector<ParticlePopulation*> populations_;
  std::string tag_assignment_;
  bool restart_sort_by_tag_;
  MeshBlockPack *pmy_pack_;
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
