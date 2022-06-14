#ifndef Z4C_Z4C_HPP_
#define Z4C_Z4C_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// This file has been generated with py/write_FD.py, please do modifications there.
//! \file z4c.hpp
//  \brief definitions for Z4c class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "athena_tensor.hpp"
#if TWO_PUNCTURES
  #include "TwoPunctures.h"
#endif

#define POW3(x) ( (x)*(x)*(x) )
#define POW4(x) ( (x)*(x)*(x)*(x))
#define PI 3.1415926535897932
#define TWO_PI 6.2831853071795862
#define SQRT2 1.4142135623730951
#define SQRT3 1.7320508075688773
#define SQRT_PI 1.77245385090551603

// forward declarations
class Coordinates;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct Z4cTaskIDs
//  \brief container to hold TaskIDs of all z4c tasks

struct Z4cTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID crhs;
  TaskID sombc;
  TaskID expl;
  TaskID sendu;
  TaskID recvu;
  TaskID newdt;
  TaskID bcs;
  TaskID algc;
  TaskID z4tad;
  TaskID admc;
  TaskID clear;
  TaskID restu;
};

namespace z4c {

// Shift needed for derivatives
//----------------------------------------------------------------------------------------
//! \class Z4c

class Z4c {
 public:
  Z4c(MeshBlockPack *ppack, ParameterInput *pin);
  ~Z4c();

    // Indexes of evolved variables
  enum {
    I_Z4c_chi,
    I_Z4c_gxx, I_Z4c_gxy, I_Z4c_gxz, I_Z4c_gyy, I_Z4c_gyz, I_Z4c_gzz,
    I_Z4c_Khat,
    I_Z4c_Axx, I_Z4c_Axy, I_Z4c_Axz, I_Z4c_Ayy, I_Z4c_Ayz, I_Z4c_Azz,
    I_Z4c_Gamx, I_Z4c_Gamy, I_Z4c_Gamz,
    I_Z4c_Theta,
    I_Z4c_alpha,
    I_Z4c_betax, I_Z4c_betay, I_Z4c_betaz,
    N_Z4c
  };
  // Names of Z4c variables
  static char const * const Z4c_names[N_Z4c];
  // Indexes of ADM variables
  enum {
    I_ADM_gxx, I_ADM_gxy, I_ADM_gxz, I_ADM_gyy, I_ADM_gyz, I_ADM_gzz,
    I_ADM_Kxx, I_ADM_Kxy, I_ADM_Kxz, I_ADM_Kyy, I_ADM_Kyz, I_ADM_Kzz,
    I_ADM_psi4,
    N_ADM
  };
  // Names of ADM variables
  static char const * const ADM_names[N_ADM];
  // Indexes of Constraint variables
  enum {
    I_CON_C,
    I_CON_H,
    I_CON_M,
    I_CON_Z,
    I_CON_Mx, I_CON_My, I_CON_Mz,
    N_CON,
  };
  // Names of costraint variables
  static char const * const Constraint_names[N_CON];
  // Indexes of matter fields
  enum {
    I_MAT_rho,
    I_MAT_Sx, I_MAT_Sy, I_MAT_Sz,
    I_MAT_Sxx, I_MAT_Sxy, I_MAT_Sxz, I_MAT_Syy, I_MAT_Syz, I_MAT_Szz,
    N_MAT
  };
  // Names of matter variables
  static char const * const Matter_names[N_MAT];

  // data
  // flags to denote relativistic dynamics
  DvceArray5D<Real> u_adm;     // adm fields
  DvceArray5D<Real> u_con;     // constraints fields
  DvceArray5D<Real> u_mat;    
  DvceArray5D<Real> u0;        // z4c solution
  DvceArray5D<Real> u1;        // z4c solution at intermediate timestep
  DvceArray5D<Real> u_rhs;     // z4c rhs storage
  DvceArray5D<Real> coarse_u0; // coarse representation of z4c solution
  
  struct ADM_vars {
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> K_dd;
  }; 
  ADM_vars adm;
  
  struct Z4c_vars {
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> chi;       // conf. factor
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> Khat;      // trace extr. curvature
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> Theta;     // Theta var in Z4c
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> alpha;     // lapse
    AthenaTensorField<Real, TensorSymm::NONE, 3, 1> Gam_u;     // Gamma functions (BSSN)
    AthenaTensorField<Real, TensorSymm::NONE, 3, 1> beta_u;    // shift
    AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> g_dd;      // conf. 3-metric
    AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> A_dd;      // conf. traceless extr. curvature
  };
  Z4c_vars z4c;
  Z4c_vars rhs;

  // aliases for the constraints
  struct Constraint_vars {
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> C;         // Z constraint monitor
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> H;         // hamiltonian constraint
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> M;         // norm squared of M_d
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> Z;         // Z constraint violation
    AthenaTensorField<Real, TensorSymm::NONE, 3, 1> M_d;       // momentum constraint
  };
  Constraint_vars con;

  // aliases for the matter variables
  struct Matter_vars {
    AthenaTensorField<Real, TensorSymm::NONE, 3, 0> rho;       // matter energy density
    AthenaTensorField<Real, TensorSymm::NONE, 3, 1> S_d;       // matter momentum density
    AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> S_dd;      // matter stress tensor
  };
  Matter_vars mat;

  struct Options {
    Real chi_psi_power;   // chi = psi^N, N = chi_psi_power
    Real chi_div_floor;   // puncture's floor value for chi, use max(chi, chi_div_floor) in non-differentiated chi
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
    Real shift_Gamma;
    Real shift_alpha2Gamma;
    Real shift_H;
    Real shift_advect;
    Real shift_eta;
  };
  Options opt;
  Real diss;
  
  // Dissipation parameter
 
  // Boundary communication buffers and functions for u
  BoundaryValuesCC *pbval_u;

  // following only used for time-evolving flow
  Real dtnew;
  int  NDIM;
  // container to hold names of TaskIDs
  Z4cTaskIDs id;

  // functions
  void AssembleZ4cTasks(TaskList &start, TaskList &run, TaskList &end);
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  TaskStatus EnforceAlgConstr(Driver *d, int stage);
  
  TaskStatus Z4cToADM_(Driver *d, int stage);
  TaskStatus ADMConstraints_(Driver *d, int stage);
  TaskStatus Z4cBoundaryRHS(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  
  template <int NGHOST>
  TaskStatus CalcRHS(Driver *d, int stage);
  
  void ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin);
  void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin);
  void GaugePreCollapsedLapse(MeshBlockPack *pmbp, ParameterInput *pin);
  void Z4cToADM(MeshBlockPack *pmbp);
  template <int NGHOST>
  void ADMConstraints(MeshBlockPack *pmbp);
  void AlgConstr(MeshBlockPack *pmbp);
#if TWO_PUNCTURES
  void ADMTwoPunctures(MeshBlockPack *pmbp, ini_data *data);
#endif
  // Sommerfeld boundary conditions
KOKKOS_FUNCTION
  void Z4cSommerfeld(int const m,
                     int const is, int const ie,
                     int const js, int const j,
                     int const ks, int const k,
                     int const parity,
                     int const scr_size,
                     int const scr_level,
                     TeamMember_t member);

 
KOKKOS_INLINE_FUNCTION
  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                  Real const gyy, Real const gyz, Real const gzz)
{
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz 
         - SQR(gyz)*gxx 
         - SQR(gxy)*gzz +   gxx*gyy*gzz;
}
 
KOKKOS_INLINE_FUNCTION
  Real Trace(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real const Axx, Real const Axy, Real const Axz,
                Real const Ayy, Real const Ayz, Real const Azz)
{
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
       ));
}
KOKKOS_INLINE_FUNCTION
  // compute inverse of a 3x3 matrix
  void SpatialInv(Real const detginv,
                  Real const gxx, Real const gxy, Real const gxz,
                  Real const gyy, Real const gyz, Real const gzz,
                  Real * uxx, Real * uxy, Real * uxz,
                  Real * uyy, Real * uyz, Real * uzz)
  {
    *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
    *uxy = (gxz*gyz  - gxy*gzz)*detginv;
    *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
    *uxz = (-gxz*gyy + gxy*gyz)*detginv;
    *uyz = (gxy*gxz  - gxx*gyz)*detginv;
    *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
    return;
  }

#include "bits/derivatives.inc"

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
};

} // namespace z4c
#endif //Z4C_Z4C_HPP_
