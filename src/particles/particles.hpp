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
enum ParticleStatus {
  PACTIVE=0, PFROZEN=1, PDELETE_PENDING=2, PDELETE_AFTER_SNAPSHOT=3
};

//----------------------------------------------------------------------------------------
//! \struct ParticleTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticleTaskIDs {
  TaskID push;
  TaskID lifecycle;
  TaskID purge;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID csend;
  TaskID crecv;
  TaskID finalize;
  TaskID post_update;
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
  std::string type_name;
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
  const char *const *real_names;   // output names in prtcl_rdata index order
  const char *const *int_names;    // output names in prtcl_idata index order
  const bool *real_output;         // fields included in particle analysis outputs
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
  TaskStatus ApplyUserLifecycle(Driver *pdriver, int stage);
  TaskStatus ApplyUserPostUpdate(Driver *pdriver, int stage);
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
  TaskStatus FinalizeLagrangianMCMove(Driver *pdriver, int stage);
  Real EstimateTimestepDrift();

  void MarkSnapshotComplete();

  int RestartLayoutVersion() const;
  std::vector<char> RestartMetadata() const;
  void ValidateRestartMetadata(const std::vector<char> &metadata) const;

 private:
  std::string input_block_;
  std::uint64_t lmc_random_seed;
  bool lmc_check_flux_probabilities;
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this population
};

//----------------------------------------------------------------------------------------
//! \struct ParticleLifecycleData
//! \brief Live particle data supplied to a problem-generator lifecycle callback.

struct ParticleLifecycleData {
  ParticleLifecycleData(MeshBlockPack *ppack, ParticlePopulation *ppopulation,
                        DvceArray2D<Real> rdata, DvceArray2D<int> idata, int count) :
      pmbp(ppack), population(ppopulation), prtcl_rdata(rdata), prtcl_idata(idata),
      nprtcl(count) {}

  MeshBlockPack *pmbp;
  ParticlePopulation *population;
  // The callback may update PSTATUS and type-owned fields for PACTIVE particles. It must
  // not resize/reorder these views or change common positions, identifiers, or ownership.
  // In the pre-routing hook, PGID still identifies the pre-push owner. In the post-update
  // hook, PGID and position identify the final owner and corrected position. Use the normal
  // execution space or fence work launched on another execution instance before returning.
  DvceArray2D<Real> prtcl_rdata;
  DvceArray2D<int> prtcl_idata;
  int nprtcl;  // may be zero; the callback is still invoked on every rank
};

//----------------------------------------------------------------------------------------
//! \struct ParticleOutputData
//! \brief Device-resident particle data supplied to one pgen output callback.

struct ParticleOutputData {
  ParticleOutputData(MeshBlockPack *ppack, ParticlePopulation *ppopulation,
                     DvceArray2D<Real> rdata, DvceArray2D<int> idata,
                     DvceArray2D<Real> output_data, int output_field, int count) :
      pmbp(ppack), population(ppopulation), prtcl_rdata(rdata), prtcl_idata(idata),
      output_data(output_data), field_index(output_field), nprtcl(count) {}

  MeshBlockPack *pmbp;
  ParticlePopulation *population;
  // Callbacks must treat particle views as read-only and copy the views/indices they need
  // before launching device work; this host descriptor is temporary.
  DvceArray2D<Real> prtcl_rdata;
  DvceArray2D<int> prtcl_idata;
  DvceArray2D<Real> output_data;
  int field_index;
  int nprtcl;  // may be zero; the callback is still invoked on every rank
};

//----------------------------------------------------------------------------------------
//! \class ParticleCreation
//! \brief Temporary device storage filled by a particle creation function.

class ParticleCreation {
 public:
  explicit ParticleCreation(ParticlePopulation *population);

  void Resize(int count);
  int GetCount() const {return nprtcl_;}
  ParticlePopulation* GetPopulation() const {return population_;}

  DvceArray2D<Real> prtcl_rdata;
  DvceArray2D<int> prtcl_idata;

 private:
  ParticlePopulation *population_;
  int nprtcl_;
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
  void InitialParticleInjection();
  void InjectParticles(Driver *pdriver);
  std::int64_t AppendParticles(ParticleCreation &creation);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);
  void WriteRestart(const std::string &filename) const;
  void LoadRestart(const std::string &filename);

 private:
  std::int64_t RunParticleInjection(bool initial);
  std::int64_t NextTagFromParticles() const;
  std::vector<ParticlePopulation*> populations_;
  std::int64_t next_tag_;
  std::string tag_assignment_;
  bool restart_sort_by_tag_;
  MeshBlockPack *pmy_pack_;
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
