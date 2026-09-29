#ifndef NR_RADIATION_NR_RADIATION_HPP_
#define NR_RADIATION_NR_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_radiation.hpp
//! \brief definitions for the SC class: gray LTE short-characteristics radiation solver
//! following Davis, Stone & Jiang (2012). Quasi-static formal solution on discrete
//! ordinates, iterated only for the meshblock boundary exchange. SMR/AMR via CC
//! Restrict/Prolong of ir and srad (source S).
//!
//! Opacity/coupling (no scattering):
//!   sigma_a   absorption coefficient per unit volume;  chi = sigma_a
//!   brad      the LTE source function (emissivity / sigma_a);  S = brad
//!   Q        = crat * prat * sigma_a * (J - brad)
//! sigma_a and brad are recomputed at the start of every solve, each by one writer:
//!   an enrolled hook (EnrollOpacityFunction / EnrollEmissionFunction, sc/sc_hooks.hpp),
//!   else the fluid: sigma_a = kappa_a * rho (kappa_a required), brad = T^4.
//! Without a fluid both hooks are required. The only state carried between steps is ir
//! (plus srad, packed by AMR for the future scattering iterate).

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "nr_radiation/angular_grid.hpp"
#include "nr_radiation/sc/sc_hooks.hpp"

// forward declarations
class Driver;

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \struct SCTaskIDs
//! \brief container to hold TaskIDs of all nr_radiation tasks

struct SCTaskIDs {
  TaskID sc_solve;  // formal solution + iteration + J (+ H, K if requested) + Q_rad
  TaskID sc_qrad;   // apply Q_rad onto hydro/mhd conserved energy for this stage
  TaskID sc_newdt;  // radiation-relaxation timestep (stagen, after hydro/mhd newdt)
  // boundary-exchange tasks for the "sc_bvals" task list (driven by ExecuteTaskList)
  TaskID ir_irecv, ir_rest, ir_send, ir_recv, ir_bcs, ir_prol, ir_csend, ir_crecv;
  TaskID srad_irecv, srad_rest, srad_send, srad_recv,
         srad_bcs, srad_prol, srad_csend, srad_crecv;
};

//----------------------------------------------------------------------------------------
//! \struct HyperplaneOrder
//! \brief Upwind hyperplane ordering of the items (cells or tiles) of a box.
//! items[start[h] ..
//! start[h+1]) are the items on hyperplane h (l1+l2+l3 == h, l = distance from the upwind
//! corner), packed as lin = (l3*n2+l2)*n1+l1. A pure function of the box dims, so each
//! instance is built once. The host offsets drive launch loops; the device copy lets a
//! kernel slice items per plane inside a team loop.

struct HyperplaneOrder {
  DvceArray1D<int> items;
  DvceArray1D<int> start_dev;
  std::vector<int> start;
  int nplanes = 0;
};

//----------------------------------------------------------------------------------------
//! \enum SweepKernel
//! \brief how the sweep is packaged into Kokkos kernels (<nr_radiation>/sweep_kernel);
//! same sweep and the same result whichever is chosen

enum class SweepKernel {tiled, plane};

//----------------------------------------------------------------------------------------
//! \class SC
//! \brief Short-characteristics (SC) radiative transfer in the gray LTE limit: a
//! quasi-static formal solution on discrete ordinates (Davis, Stone & Jiang 2012),
//! iterated only for the meshblock boundary exchange, coupled to the gas energy through
//! Q = crat prat sigma_a (J - B).

class SC {
 public:
  SC(MeshBlockPack *ppack, ParameterInput *pin);
  ~SC();

  // angular quadrature (Carlson S_N / Bruls et al. 1999 type-A grid)
  SCAngularGrid *pang = nullptr;
  // = noct*nang, total number of discrete rays (nang is the per-octant count)
  int nang_tot;

  // opacity/coupling parameters
  Real kappa_a;   // convenience hook for a global constant absorption opacity
                  // per unit mass: sigma_a = kappa_a * rho;
  Real prat;      // radiation-to-gas pressure ratio P_rad/P_gas
  Real crat;      // speed-of-light to sound-speed ratio c/a_gas
  // <nr_radiation>/kappa_a present in input; required with a fluid
  // unless an opacity hook is enrolled (checked post-pgen in pgen.cpp)
  bool kappa_a_specified;

  // user-enrollable opacity / emission hooks (contract in sc/sc_hooks.hpp);
  // dispatch is by null-check, no input flags
  SCOpacityFnPtr user_opacity_func = nullptr;
  SCEmissionFnPtr user_emission_func = nullptr;
  void EnrollOpacityFunction(SCOpacityFnPtr myfunc);
  void EnrollEmissionFunction(SCEmissionFnPtr myfunc);

  // flags controlling coupling to (M)HD
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  // radiation -> gas: apply Q_rad to the fluid energy and the radiation
  // dt limit. The gas always emits when present, regardless of this flag.
  bool affect_fluid;

  // how the sweep is packaged into Kokkos kernels (same sweep, same result either way)
  //   plane:     one launch per plane along each ray's own dominant axis, a flat
  //              par_for over the transverse cells x rays; no tiles or teams
  //   tiled:     one launch per tile hyperplane, one team per tile x ray, each team
  //              walking its tile's cell hyperplanes with a team barrier between them.
  //              Tunable through tile_size and team_size.
  SweepKernel sweep_kernel;
  std::string sweep_kernel_name;   // as given in the deck, for messages
  int tile_size;   // tiled only: cells per tile edge; 0 = the whole meshblock is one tile
  int team_size;   // tiled only: threads per team; 0 = Kokkos::AUTO

  // iteration control
  int iter_max;
  // minimum sweeps before early exit (default 1: sweep 1 is compared to the
  // previous step's converged J, so one sweep suffices for a static field)
  int iter_min;
  // residual: max over cells of |Jn-Jo| / max(Jn,Jo). Bounds the change per
  // sweep, not the error; a small perturbation on a bright background converges
  // only to iter_tol / (relative amplitude).
  Real iter_tol;
  int niter_last;    // diagnostic: number of sweeps used in the most recent solve
  Real resid_last;   // diagnostic: final residual of the most recent solve
  bool converged;    // true if the last SolveTransfer met iter_tol

  // intensity: (nmb, nang_tot, nx3, nx2, nx1) with ghost zones; persists across steps as
  // the warm start of the next solve
  DvceArray5D<Real> ir;
  DvceArray5D<Real> coarse_ir;

  // source S the sweep reads: (nmb, 1, nx3, nx2, nx1), nvar=1 so the ghost exchange and
  // AMR treat it as a CC field. S = brad in LTE; with scattering, the solve's iterate
  DvceArray5D<Real> srad;
  DvceArray5D<Real> coarse_srad;

  // per-zone radiation quantities (nmb,nx3,nx2,nx1)
  // total extinction the sweep reads; = sigma_a until scattering adds sigma_s
  DvceArray4D<Real> chi;
  // LTE source function per cell: emissivity / sigma_a. Written every solve by the
  // emission hook, else T^4 of the gas; the sweep reads the copy srad
  DvceArray4D<Real> brad;
  // Angular moments, Athena++ rad_mom order (radiation.hpp: IER, IFR1-3, IPR...):
  //   0        J        the zeroth moment; also the residual's quantity and Q_int's J
  //   1-3      H_1, H_2, H_3
  //   4-9      K_11, K_22, K_33, K_12, K_13, K_23   (symmetric: six, not nine)
  // Slot 0 is refreshed every sweep for the residual; 1-9 once per solve. Allocated at
  // full width in the constructor and never reallocated: a restart rebuilds J from the
  // restored ir BEFORE the outputs are constructed, so a later realloc would wipe it.
  // J is derived from ir, so MeshRefinement calls ComputeJ() after a remesh (ir moves
  // with the blocks, J does not).
  DvceArray5D<Real> moments;
  DvceArray5D<Real> j_prev; // J from the previous sweep (residual scratch)
  DvceArray4D<Real> qrad;       // radiative heating/cooling rate of the gas
  // absorption coefficient per unit volume (nmb, 1, nx3, nx2, nx1); nvar=1 5D so it is
  // directly registrable as a stored output variable. Written every solve by the opacity
  // hook, else kappa_a * rho; the sweep reads the copy chi
  DvceArray5D<Real> sigma_a;

  // hyperplane orderings (sc/formal_solution.cpp BuildIndices; built once in the
  // constructor)
  HyperplaneOrder tile_cells;    // cells of one tile: tiled, inner loop
  HyperplaneOrder block_tiles;   // tiles of the meshblock: tiled, launch loop
  // plane on a non-cubic block: rays sorted by the axis they march along, so each
  // (axis, plane) launch is sized to that axis's own transverse extent. Group g holds
  // ray_by_axis[ray_axis_start[g] .. ray_axis_start[g+1]). Empty on a cubic block, which
  // needs no grouping (see SC::FormalSolutionPlane).
  DvceArray1D<int> ray_by_axis;
  int ray_axis_start[4] = {0, 0, 0, 0};
  int tile_nx1 = 0, tile_nx2 = 0, tile_nx3 = 0;   // tile dims in cells
  int ntile1 = 0, ntile2 = 0, ntile3 = 0;         // tiles per axis


  // boundary communication: separate MeshBoundaryValuesCC for ir and srad
  MeshBoundaryValuesCC *pbval_ir = nullptr;
  MeshBoundaryValuesCC *pbval_srad = nullptr;

  Real dtnew;

  SCTaskIDs id;

  void AssembleSCTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SolveTransfer(Driver *pdrive, int stage);
  TaskStatus AddQrad(Driver *pdrive, int stage);

  // boundary-exchange wrapper tasks for the "sc_bvals" task list
  TaskStatus InitRecvIr(Driver *pdrive, int stage);
  TaskStatus RestrictIr(Driver *pdrive, int stage);
  TaskStatus SendIr(Driver *pdrive, int stage);
  TaskStatus RecvIr(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCsIr(Driver *pdrive, int stage);
  TaskStatus ProlongateIr(Driver *pdrive, int stage);
  TaskStatus ClearSendIr(Driver *pdrive, int stage);
  TaskStatus ClearRecvIr(Driver *pdrive, int stage);

  TaskStatus InitRecvSrad(Driver *pdrive, int stage);
  TaskStatus RestrictSrad(Driver *pdrive, int stage);
  TaskStatus SendSrad(Driver *pdrive, int stage);
  TaskStatus RecvSrad(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCsSrad(Driver *pdrive, int stage);
  TaskStatus ProlongateSrad(Driver *pdrive, int stage);
  TaskStatus ClearSendSrad(Driver *pdrive, int stage);
  TaskStatus ClearRecvSrad(Driver *pdrive, int stage);

  // stagen task: radiation-relaxation timestep
  TaskStatus NewTimeStep(Driver *pdrive, int stage);

  // helpers called from SolveTransfer. UpdateOpacity also runs wherever the relaxation
  // timestep is evaluated (NewTimeStep, driver initialization, after a remesh)
  void UpdateOpacity();             // sigma_a (hook or kappa_a*rho) and chi = sigma_a
  void UpdateEmission();            // brad (hook or T^4)
  void UpdateSource();
  void ApplyPhysicalBCs();
  void ApplyReflectBCs();   // intensity mirror at reflecting faces
  void ApplyPhysicalBCsSource();
  void ComputeJ();       // J only, into moments slot 0 (every sweep; after a remesh)
  Real ResidualJ();      // max symmetric relative change J vs j_prev, allreduced
  void ComputeHK();      // H_i and K_ij into moments slots 1-9 (once per solve)
  void ComputeQrad();

  // formal solution (sc/formal_solution.cpp): dispatch on sweep_kernel. Public because
  // Kokkos CUDA device lambdas cannot be defined inside private/protected members.
  void FormalSolution();
  void FormalSolutionTiled();
  void FormalSolutionPlane();
  void PlaneSweepAllRays();
  void PlaneSweepRaysByAxis();
  void BuildRayGroups();
  void BuildIndices();

 private:
  MeshBlockPack* pmy_pack;
};

}  // namespace nr_radiation
#endif  // NR_RADIATION_NR_RADIATION_HPP_
