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
//! ordinates, iterated only for the meshblock boundary exchange; SMR/AMR via CC
//! Restrict/Prolong of ir and srad.
//!   sigma_a   absorption coefficient per unit volume (chi = sigma_a, no scattering)
//!   brad      LTE source function;  Q = crat * prat * sigma_a * (J - brad)
//! Both are recomputed each solve by an enrolled hook (sc/sc_hooks.hpp) or from the fluid
//! (sigma_a = kappa_a * rho, brad = T^4). Only ir (and srad) is carried between steps.

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
  // "sc_bvals_srad": the source-only refresh at solve start. ir ghosts survive from
  // the previous solve's post-sweep exchange, but srad follows the fluid and does not.
  TaskID sronly_irecv, sronly_rest, sronly_send, sronly_recv,
         sronly_bcs, sronly_prol, sronly_csend, sronly_crecv;
  // "sc_flxcor": coarse-fine correction of hflx. The matching InitFluxRecv is posted
  // in SolveTransfer, not here, so it is not a task.
  TaskID hflx_send, hflx_recv, hflx_csend, hflx_crecv;
  TaskID sc_solve;  // SolveTransfer: sweeps to convergence, J, H, K, Q_rad, dt limit
  TaskID sc_qrad;   // apply Q_rad onto hydro/mhd conserved energy for this stage
  TaskID sc_newdt;  // radiation-relaxation timestep (stagen, after hydro/mhd newdt)
  // boundary-exchange tasks for the "sc_bvals" task list (driven by ExecuteTaskList)
  TaskID ir_irecv, ir_rest, ir_send, ir_recv, ir_bcs, ir_prol, ir_csend, ir_crecv;
  TaskID srad_irecv, srad_rest, srad_send, srad_recv,
         srad_bcs, srad_prol, srad_csend, srad_crecv;
};

//----------------------------------------------------------------------------------------
//! \struct HyperplaneOrder
//! \brief Upwind hyperplane ordering of the tiles of a meshblock, for the tiled kernel's
//! launch loop. items[start[h] .. start[h+1]) are the tiles on hyperplane h (l1+l2+l3 ==
//! h, l = distance from the upwind corner), packed as lin = (l3*n2+l2)*n1+l1. Built once
//! in the constructor.

struct HyperplaneOrder {
  DvceArray1D<int> items;
  std::vector<int> start;
  int nplanes = 0;
};

//----------------------------------------------------------------------------------------
//! \enum SweepKernel
//! \brief how the sweep is packaged into Kokkos kernels (<nr_radiation>/sweep_kernel);
//! same sweep and the same result whichever is chosen

enum class SweepKernel {tiled, plane};

//----------------------------------------------------------------------------------------
//! \enum QradForm
//! \brief Form of the gas-energy coupling (<nr_radiation>/qrad_form), Davis 2012:
//!   integral  Eq. 27, Q = crat prat sigma_a (J - brad): local; loses precision where
//!             the zone is thick and near equilibrium (J -> brad with sigma_a large).
//!   divh      Eq. 28, Q = -crat prat div H from face fluxes: accurate thick, exactly
//!             conservative (telescopes), degrades thin where sigma_a -> 0.
//!   hybrid    integral where tau = chi dx <= kTauLo, divh where tau >= kTauHi, a smooth
//!             weighted sum between. Conservative only where tau >= kTauHi.
//! Quadrature weights sum to one, so J is the mean intensity, H = <mu I>, and no 4 pi
//! appears.

enum class QradForm {integral, divh, hybrid};

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
  // timestep. The gas always emits when present, regardless of this flag.
  bool affect_fluid;

  // how the sweep is packaged into Kokkos kernels (same sweep, same result either way)
  //   plane:     one launch per plane along each ray's own marching axis, a flat
  //              par_for over the transverse cells x rays; no tiles or teams
  //   tiled:     one launch per tile hyperplane, one team per tile x ray, each team
  //              walking its tile in planes along that ray's marching axis, with a team
  //              barrier between them. Tunable through tile_size and team_size.
  SweepKernel sweep_kernel;
  std::string sweep_kernel_name;   // as given in the input file, for messages
  int tile_size;   // tiled only: cells per tile edge; 0 = the whole meshblock is one tile
  int team_size;   // tiled only: threads per team; 0 = Kokkos::AUTO

  // which form of the gas-energy coupling to evaluate (see QradForm above)
  QradForm qrad_form;
  std::string qrad_form_name;      // as given in the input file, for messages

  // Fraction of a cell's internal energy Q_rad may remove per step, imposed as
  // dt <= cfl_qrad * e_int/|Q| (sc/coupling.cpp). 0 disables; the default depends on
  // qrad_form.
  Real cfl_qrad;
  bool cfl_qrad_specified;         // <nr_radiation>/cfl_qrad present in the input

  // iteration control
  int iter_max; // maximum number of sweeps per solve
  int iter_min; // minimum number of sweeps per solve (>=1)
  // residual: max over cells of |Jn-Jo| / max(Jn,Jo). Bounds the change per
  // sweep, not the error; a small perturbation on a bright background converges
  // only to iter_tol / (relative amplitude).
  Real iter_tol;
  int niter_last;    // diagnostic: number of sweeps used in the most recent solve
  Real resid_last;   // diagnostic: final residual of the most recent solve
  bool converged;    // true if the last SolveTransfer met iter_tol

  // intensity: (nmb, nang_tot, nx3, nx2, nx1) with ghost zones
  DvceArray5D<Real> ir;
  DvceArray5D<Real> coarse_ir;

  // source function S: (nmb, 1, nx3, nx2, nx1), nvar=1 so the ghost exchange and
  // AMR treat it as a CC field. S set to brad without scattering.
  DvceArray5D<Real> srad;
  DvceArray5D<Real> coarse_srad;

  // per-zone radiation quantities (nmb,nx3,nx2,nx1)
  // total extinction the sweep reads; = sigma_a without scattering.
  DvceArray4D<Real> chi;
  // LTE source function per cell: emissivity / sigma_a. Written every solve by the
  // emission hook, else T^4 of the gas; the sweep reads the copy srad
  DvceArray4D<Real> brad;
  // Angular moments in Athena++ rad_mom order: 0 J; 1-3 H_i; 4-9 K_11, K_22, K_33,
  // K_12, K_13, K_23. Slot 0 is refreshed every sweep, 1-9 once per solve. Allocated at
  // full width in the constructor and never reallocated (a restart rebuilds J before the
  // outputs exist); MeshRefinement calls ComputeJ() after a remesh.
  DvceArray5D<Real> moments;
  DvceArray5D<Real> j_prev; // J from the previous sweep (for evaluating residual)
  DvceArray4D<Real> qrad;       // radiative heating/cooling rate of the gas
  // Radiative flux normal to each cell face for the divh/hybrid forms. DvceFaceFld5D
  // with nvar = 1.
  DvceFaceFld5D<Real> hflx;
  // absorption coefficient per unit volume (nmb, 1, nx3, nx2, nx1). Written every solve
  // by the opacity hook, else kappa_a * rho; the sweep reads the copy chi
  DvceArray5D<Real> sigma_a;

  // hyperplane ordering of the meshblock's TILES, driving the tiled kernel's launch loop
  // (sc/formal_solution.cpp BuildIndices; built once in the constructor)
  HyperplaneOrder block_tiles;
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
  // coarse-fine flux correction of hflx; its own object because flux buffers and requests
  // are per-object. nullptr unless a differential form runs on a multilevel mesh.
  MeshBoundaryValuesCC *pbval_hflx = nullptr;

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

  // Q_rad energy-depletion timestep limit. No-op when cfl_qrad == 0 (the default for
  // qrad_form = integral) or affect_fluid is false.
  void LimitDtByQrad();

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
  void ComputeHK();      // H_i and K_ij into slots 1-9, active + 1 ghost layer
  void FillHflx();       // face-centred H from the cell-centred slots 1-3
  void BoundaryHflx();   // physical faces: per-ray upwind face flux
  void BuildFaceFlux(Driver *pdrive);   // the two above + the coarse-fine correction
  TaskStatus SendHflx(Driver *pdrive, int stage);
  TaskStatus RecvHflx(Driver *pdrive, int stage);
  TaskStatus ClearSendHflx(Driver *pdrive, int stage);
  TaskStatus ClearRecvHflx(Driver *pdrive, int stage);
  bool ir_ghosts_fresh_ = false;   // ghost ir still current from the last solve
  int ir_ghosts_seq_ = -1;         // AMR/load-balance sequence those ghosts belong to
  static constexpr Real kTauLo = 0.3, kTauHi = 3.0;   // hybrid weighting interval
  void ComputeQrad();    // Q = (1-w) Q_int + w Q_dif, w fixed by qrad_form

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
