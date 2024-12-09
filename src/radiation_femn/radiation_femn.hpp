#ifndef ATHENA_RADIATION_FEMN_HPP
#define ATHENA_RADIATION_FEMN_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn.hpp
//  \brief definitions for Radiation FEMN class

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Driver;

//! \struct RadiationFEMNTaskIDs
//  \brief container to hold TaskIDs of all radiation FEM_N tasks
struct RadiationFEMNTaskIDs {
  TaskID rad_irecv;
  TaskID rad_beams;
  TaskID copycons;
  TaskID rad_tetrad;
  TaskID rad_flux;
  TaskID rad_sendf;
  TaskID rad_recvf;
  TaskID rad_expl;
  TaskID rad_limfem;
  TaskID rad_limdg;
  TaskID rad_limtheta;
  TaskID rad_filterfpn;
  TaskID rad_resti;
  TaskID rad_sendi;
  TaskID rad_recvi;
  TaskID bcs;
  TaskID rad_csend;
  TaskID rad_crecv;

};

//! \struct RadiationFEMNPhaseIndices
// hold indices in momentum space
struct RadiationFEMNPhaseIndices {
  int combinedidx;
  int nuidx;
  int enidx;
  int angidx;
};

//! \enum LimiterDG
// choice of DG limiter
enum LimiterDG {
  none,
  minmod,
  minmod2
};

enum M1Closure {
  Charon,
  Shibata,
  Simple
};

enum ClosureFunc {
  Eddington,
  Kershaw,
  Minerbo,
  Thin
};



namespace radiationfemn {

class RadiationFEMN {
 public:
  RadiationFEMN(MeshBlockPack *ppack, ParameterInput *pin);
  ~RadiationFEMN();

  // params
  int refinement_level;             // level of angular refinement (does not change)
  int num_points;                   // no. of angular DOF
  int num_species;                  // no. of nu species
  int num_energy_bins;              // no. of energy bins
  int num_points_total;             // no. of momentum space DOF
  Real energy_max;                  // maximum value of energy
  int num_ref;                      // geodesic grid info: current refinement level
  int num_edges;                    // geodesic grid info: number of unique edges
  int num_triangles;                // geodesic grid info: number of unique triangular elements
  int basis;                        // choice of basis functions on the geodesic grid (1: tent - FEMN)
  bool mass_lumping;                // flag for mass lumping
  bool multiply_massinv;            // flag to multiply everything by inverse of mass matrix
  bool m1_flag;                     // flag for M1
  std::string limiter_dg;           // choice of limiter for DG, default: "minmod2"
  std::string limiter_fem;          // choice of limiter for FEM, default: "clp"
  bool limiter_theta;               // flag for theta limiter (M1), default: false
  bool fpn;                         // flag for FP_N, default: false
  int lmax;                         // maximum value of l (FPN), default: 0
  int filter_sigma_eff;             // effective opacity of the filter (FPN), default: 0
  LimiterDG limiter_dg_minmod_type; // type of DG limiter, default:: LimiterDG::minmod2
  bool rad_source;                  // flag for source terms, default: false
  M1Closure m1_closure;             // choice of form of pressure (M1)
  ClosureFunc closure_fun;          // choice of closure function (M1)
  Real rad_E_floor;                 // floor params for M1
  Real rad_eps;                     // floor params for M1

  // @TODO: cleanup
  int num_beams;
  bool beam_source;
  Real beam_source_1_y1;
  Real beam_source_1_y2;
  Real beam_source_2_y1;
  Real beam_source_2_y2;
  Real beam_source_1_phi;
  Real beam_source_1_theta;
  Real beam_source_2_phi;
  Real beam_source_2_theta;
  DvceArray1D<Real> beam_source_1_vals;
  DvceArray1D<Real> beam_source_2_vals;

  // @TODO: delete
  Real energy_par = 1.;

  // quadratures on unit sphere
  HostArray2D<Real> scheme_points;
  HostArray1D<Real> scheme_weights;
  int scheme_num_points;
  std::string scheme_name;

  // quadratures on energy grid
  HostArray1D<Real> energy_scheme_points;
  HostArray1D<Real> energy_scheme_weights;
  int energy_scheme_num_points;

  // angular grid matrices
  DvceArray2D<Real> angular_grid;           // store (l,m) for FPN or (phi,theta) for FEMN
  HostArray2D<Real> angular_grid_cartesian; // cartesian coordinates of angular grid (FENM)
  HostArray2D<int> triangle_information;    // vertex info. of triangular elements (FEMN)

  DvceArray2D<Real> mass_matrix;             // mass matrix [Eqn. 12 arXiv:2212.01409]
  DvceArray2D<Real> mass_matrix_inv;         // mass matrix inverse
  DvceArray2D<Real> stiffness_matrix_x;      // x component of the stiffness matrix
  DvceArray2D<Real> stiffness_matrix_y;      // y component of the stiffness matrix
  DvceArray2D<Real> stiffness_matrix_z;      // z component of the stiffness matrix
  DvceArray3D<Real> P_matrix;                // store mass and stiffness in P^muhat_A^B
  DvceArray3D<Real> Pmod_matrix;             // Zero speed mode corrected P_matrix
  HostArray5D<Real> G_mat_host;              // G^nuhat^muhat_ihat_A^B (contains angular basis derviatives)
  DvceArray5D<Real> G_matrix;                // Gmat on device
  HostArray5D<Real> F_mat_host;              // F^nuhat^muhat_ihat_A^B
  DvceArray5D<Real> F_matrix;                // Fmat on device
  DvceArray2D<Real> Q_matrix;                // Q^muhat_A

  // energy grid matrices
  DvceArray1D<Real> energy_grid;             // energy grid array
  DvceArray2D<Real> Ven_matrix;              // V_m^n matrix
  DvceArray2D<Real> Veninv_matrix;           // Invese of V
  DvceArray2D<Real> Wen_matrix;              // W_m^m matrix (containing energy basis derivatives)


  // distribution function, flux and other arrays
  DvceArray5D<Real> f0;             // distribution function
  DvceArray5D<Real> f1;             // distribution at intermediate step
  DvceArray5D<Real> ftemp;          // intermediate arrays needed for limiting
  DvceArray4D<bool> radiation_mask; // mask for radiation
  DvceArray5D<Real> coarse_f0;      // distribution function on 2x coarser grid (for SMR/AMR)
  DvceFaceFld5D<Real> iflx;         // spatial fluxes

  // tetrad quantities
  DvceArray5D<Real> L_mu_muhat0_data;
  AthenaTensor4d<Real, TensorSymm::NONE, 4, 2> L_mu_muhat0;

  // fluid velocity
  DvceArray5D<Real> u_mu_data;
  AthenaTensor4d<Real, TensorSymm::NONE, 4, 1> u_mu;

  // source terms
  DvceArray4D<Real> eta;              // emissivity [assume isotropic]
  DvceArray4D<Real> kappa_s;          // scattering coefficient [assume isotropic]
  DvceArray4D<Real> kappa_a;          // absorption coefficient [assume isotropic]
  DvceArray1D<Real> e_source;         // [Eq. (19) of Radice et. al. arXiv:1209.1634v3]
  DvceArray1D<Real> e_source_nominv;  // without mass inv multiplication
  DvceArray2D<Real> S_source;         // [Eq. (19) of Radice et. al. arXiv:1209.1634v3]
  DvceArray2D<Real> W_matrix;         // holds the inverse of (delta^A_B - k * S^A_B) where k = dt or dt/2


  BoundaryValuesCC *pbval_f;      // Boundary communication buffers and functions for f
  Real dtnew;                     // timestep
  RadiationFEMNTaskIDs id;        // container to hold TaskIDs

  // Tasklist functions
  TaskStatus InitRecv(Driver *d, int stage);

  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalculateFluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus ApplyLimiterDG(Driver *pdriver, int stage);
  TaskStatus ApplyLimiterTheta(Driver *pdriver, int stage);
  TaskStatus ApplyLimiterFEM(Driver *pdrive, int stage);
  TaskStatus ApplyFilterLanczos(Driver *pdriver, int stage);
  TaskStatus AddRadiationSourceTerm(Driver *d, int stage);
  TaskStatus TetradOrthogonalize(Driver *pdriver, int stage);
  TaskStatus RestrictI(Driver *d, int stage);
  TaskStatus SendI(Driver *d, int stage);
  TaskStatus RecvI(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);

  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  void AssembleRadiationFEMNTasks(TaskList &start, TaskList &run, TaskList &end);

  // other functions
  void LoadFEMNMatrices();
  void LoadFPNMatrices();
  void LoadEnergyGridMatricesGLag();
  void ComputeMassInverse();
  void ComputePMatrices();
  void ComputeSourceMatrices();
  void InitializeBeamsSourcesFEMN();
  void InitializeBeamsSourcesFPN();
  void InitializeBeamsSourcesM1();

 private:
  MeshBlockPack *pmy_pack;

};

void ApplyBeamSourcesFEMN(Mesh *pmesh);
void ApplyBeamSourcesFEMN1D(Mesh *pmesh);
void ApplyBeamSourcesBlackHoleM1(Mesh *pmesh);
void RadiationFEMNBCs(MeshBlockPack *ppack, DualArray2D<Real> i_in,
                                     DvceArray5D<Real> i0);
KOKKOS_INLINE_FUNCTION
Real Sgn(Real x) { return (x >= 0) ? +1. : -1.; }

KOKKOS_INLINE_FUNCTION
RadiationFEMNPhaseIndices IndicesComponent(int n, int num_points, int num_energy_bins = 1, int num_species = 1) {
  RadiationFEMNPhaseIndices idcs = {.combinedidx = n,
      .nuidx = int(n / (num_energy_bins * num_points)),
      .enidx = int((n - int(n / (num_energy_bins * num_points)) * num_energy_bins * num_points) / num_points),
      .angidx = n - int(n / (num_energy_bins * num_points)) * num_energy_bins * num_points
          - int((n - int(n / (num_energy_bins * num_points)) * num_energy_bins * num_points) / num_points) * num_points};
  return idcs;
}

KOKKOS_INLINE_FUNCTION
int IndicesUnited(int nuidx, int enidx, int angidx, int num_species, int num_energy_bins, int num_points) {
  int combinedidx = angidx + nuidx * num_energy_bins * num_points + enidx * num_points;
  return combinedidx;
}

KOKKOS_INLINE_FUNCTION
RadiationFEMNPhaseIndices NuEnIndicesComponent(int n, int num_species, int num_energy_bins) {
  RadiationFEMNPhaseIndices idcs = {.combinedidx = n,
      .nuidx = int(n / num_energy_bins),
      .enidx = n - int(n / num_energy_bins) * num_energy_bins,
      .angidx = -42};
  return idcs;
}

} // namespace radiationfemn

#endif //ATHENA_RADIATION_FEMN_HPP
