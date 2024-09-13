#ifndef Z4C_Z4C_HPP_
#define Z4C_Z4C_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c.hpp
//! \brief definitions for Z4c class

#include <map>
#include <memory>    // make_unique, unique_ptr
#include <string>
#include <vector>
#include "athena.hpp"
#include "utils/finite_diff.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "athena_tensor.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"

// forward declarations
class Coordinates;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct Z4cTaskIDs
//  \brief container to hold TaskIDs of all z4c tasks

struct Z4cTaskIDs {
  TaskID irecv;
  TaskID irecvweyl;
  TaskID copyu;
  TaskID crhs;
  TaskID sombc;
  TaskID expl;
  TaskID sendu;
  TaskID recvu;
  TaskID newdt;
  TaskID bcs;
  TaskID prol;
  TaskID algc;
  TaskID z4tad;
  TaskID admc;
  TaskID csend;
  TaskID crecv;
  TaskID restu;
  TaskID ptrck;
  TaskID weyl_scalar;
  TaskID wave_extr;
  TaskID weyl_rest;
  TaskID weyl_send;
  TaskID weyl_prol;
  TaskID weyl_recv;
  TaskID csendweyl;
  TaskID crecvweyl;
};

namespace z4c {
class Z4c_AMR;

// Shift needed for derivatives
//----------------------------------------------------------------------------------------
//! \class Z4c

class Z4c {
 public:
  Z4c(MeshBlockPack *ppack, ParameterInput *pin);
  ~Z4c();

  // Indices of evolved variables
  enum {
    I_Z4C_CHI,
    I_Z4C_GXX, I_Z4C_GXY, I_Z4C_GXZ, I_Z4C_GYY, I_Z4C_GYZ, I_Z4C_GZZ,
    I_Z4C_KHAT,
    I_Z4C_AXX, I_Z4C_AXY, I_Z4C_AXZ, I_Z4C_AYY, I_Z4C_AYZ, I_Z4C_AZZ,
    I_Z4C_GAMX, I_Z4C_GAMY, I_Z4C_GAMZ,
    I_Z4C_THETA,
    I_Z4C_ALPHA,
    I_Z4C_BETAX, I_Z4C_BETAY, I_Z4C_BETAZ,
    nz4c
  };
  // Names of Z4c variables
  static char const * const Z4c_names[nz4c];
  // Indices of Constraint variables
  enum {
    I_CON_C,
    I_CON_H,
    I_CON_M,
    I_CON_Z,
    I_CON_MX, I_CON_MY, I_CON_MZ,
    ncon,
  };
  // Names of costraint variables
  static char const * const Constraint_names[ncon];
  // Indices of matter fields
  /*enum {
    I_MAT_RHO,
    I_MAT_SX, I_MAT_SY, I_MAT_SZ,
    I_MAT_SXX, I_MAT_SXY, I_MAT_SXZ, I_MAT_SYY, I_MAT_SYZ, I_MAT_SZZ,
    nmat
  };
  // Names of matter variables
  static char const * const Matter_names[nmat];*/

  // data
  // flags to denote relativistic dynamics
  DvceArray5D<Real> u_con;     // constraints fields
  DvceArray5D<Real> u_mat;
  DvceArray5D<Real> u0;        // z4c solution
  DvceArray5D<Real> u1;        // z4c solution at intermediate timestep
  DvceArray5D<Real> u_rhs;     // z4c rhs storage
  DvceArray5D<Real> coarse_u0; // coarse representation of z4c solution
  DvceArray5D<Real> u_weyl; // weyl scalars
  DvceArray5D<Real> coarse_u_weyl; // coarse representation of weyl scalars

  // puncture location
  Real ppos[3] = {0.,0.,0.}; // later on initiate from input file
#if TWO_PUNCTURES
  // second puncture location
  Real ppos2[3] = {0.,0.,0.}; // later on initiate from input file
#endif
  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };
  ADM_vars adm;

  struct ADMhost_vars {
    AthenaHostTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };

  struct Wave_Extr_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> rpsi4;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> ipsi4;
  };
  Wave_Extr_vars weyl;

  struct Z4c_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;     // conf. factor
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vKhat;   // trace extr. curvature
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vTheta;  // Theta var in Z4c
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;   // lapse
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vGam_u;  // Gamma functions (BSSN)
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;  // shift
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;    // conf. 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vA_dd;   // conf. traceless extr. curvature
  };
  Z4c_vars z4c;
  Z4c_vars rhs;

  // aliases for the constraints
  struct Constraint_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> C;         // Z constraint monitor
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> H;         // hamiltonian constraint
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> M;         // norm squared of M_d
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> Z;         // Z constraint violation
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> M_d;       // momentum constraint
  };
  Constraint_vars con;

  // aliases for the matter variables
  /*struct Matter_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> rho;       // matter energy density
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vS_d;       // matter momentum density
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vS_dd;      // matter stress tensor
  };
  Matter_vars mat;*/

  struct Options {
    Real chi_psi_power;   // chi = psi^N, N = chi_psi_power
    // puncture's floor value for chi, use max(chi, chi_div_floor)
    // in non-differentiated chi
    Real chi_div_floor;
    Real diss;            // amount of numerical dissipation
    Real eps_floor;       // a small number O(10^-12)
    // Constraint damping parameters
    Real damp_kappa1;
    Real damp_kappa2;
    // Gauge conditions for the lapse
    Real lapse_oplog;
    Real lapse_harmonicf;
    Real lapse_harmonic;
    Real lapse_advect;
    // Gauge condition for the shift
    Real shift_ggamma;
    Real shift_alpha2ggamma;
    Real shift_hh;
    Real shift_advect;
    Real shift_eta;
    // Enable BSSN if false (disable theta)
    bool use_z4c;
    // Apply the Sommerfeld condition for user BCs.
    bool user_Sbc;
    // Boundary extrapolation order
    int extrap_order;
  };
  Options opt;
  Real diss;              // Dissipation parameter

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Boundary communication buffers for the weyl scalar
  MeshBoundaryValuesCC *pbval_weyl;

  // following only used for time-evolving flow
  Real dtnew;
  // container to hold names of TaskIDs
  Z4cTaskIDs id;

  // geodesic grid for wave extr
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;
  // array storing waveform at each radii
  HostArray3D<Real> psi_out;
  Real waveform_dt;
  Real last_output_time;
  int nrad; // number of radii to perform wave extraction

  // functions
  void QueueZ4cTasks();
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus InitRecvWeyl(Driver *d, int stage);
  TaskStatus ClearRecvWeyl(Driver *d, int stage);
  TaskStatus ClearSendWeyl(Driver *d, int stage);
  TaskStatus CopyU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendWeyl(Driver *d, int stage);
  TaskStatus RecvWeyl(Driver *d, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);
  TaskStatus ProlongateWeyl(Driver *pdrive, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  TaskStatus EnforceAlgConstr(Driver *d, int stage);

  TaskStatus Z4cToADM_(Driver *d, int stage);
  TaskStatus UpdateExcisionMasks(Driver *d, int stage);
  TaskStatus ADMConstraints_(Driver *d, int stage);
  TaskStatus Z4cBoundaryRHS(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus RestrictWeyl(Driver *d, int stage);
  TaskStatus PunctureTracker(Driver *d, int stage);
  TaskStatus CalcWeylScalar(Driver *d, int stage);
  TaskStatus CalcWaveForm(Driver *d, int stage);

  template <int NGHOST>
  TaskStatus CalcRHS(Driver *d, int stage);
  template <int NGHOST>
  void ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin);
  void GaugePreCollapsedLapse(MeshBlockPack *pmbp, ParameterInput *pin);
  void Z4cToADM(MeshBlockPack *pmbp);
  template <int NGHOST>
  void ADMConstraints(MeshBlockPack *pmbp);
  template <int NGHOST>
  void Z4cWeyl(MeshBlockPack *pmbp);
  void WaveExtr(MeshBlockPack *pmbp);
  void AlgConstr(MeshBlockPack *pmbp);

  // amr criteria
  Z4c_AMR *pz4c_amr{nullptr};

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
};



} // namespace z4c
#endif //Z4C_Z4C_HPP_
