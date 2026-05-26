#ifndef Z4C_ID_SOLVE_HPP_
#define Z4C_ID_SOLVE_HPP_

#include <cstdio>
#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "tasklist/task_list.hpp"

class Driver;
class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace z4c {

enum IDRelaxVar {
  ID_RELAX_U,
  ID_RELAX_V,
  ID_RELAX_NVAR
};

enum IDRelaxFreeVar {
  ID_RELAX_PSI_SINGULAR,
  ID_RELAX_AHAT2,
  ID_RELAX_RESIDUAL,
  ID_RELAX_WAVESPEED,
  ID_RELAX_NFREE
};

class IDConformalThinSandwich {
 public:
  struct RelaxVars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> u;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> v;
  };

  struct FreeVars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> psi_singular;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> ahat2;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> residual;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> wavespeed;
  };

  struct Diagnostics {
    Real residual_l2;
    Real residual_rel_l2;
    Real residual_max;
    Real residual_excised_l2;
    Real residual_excised_rel_l2;
    Real residual_excised_max;
    Real u_l2;
    Real v_l2;
    Real v_max;
    Real volume;
    Real excised_volume;
    Real ncell;
    Real excised_ncell;
    Real finite;
  };

  IDConformalThinSandwich(MeshBlockPack *pmbp, ParameterInput *pin);
  ~IDConformalThinSandwich();

  TaskStatus SolveTask(Driver *pdriver, int stage);
  void PrepareForRestart();
  bool StopAfterSolveRequested() const { return stop_after_solve_ && solved_; }
  bool SkipInitialOutput() const { return skip_initial_output_; }

  DvceArray5D<Real> u_relax;
  DvceArray5D<Real> u_relax_tmp;
  DvceArray5D<Real> u_relax_best;
  DvceArray5D<Real> u_rhs;
  DvceArray5D<Real> coarse_u_relax;
  DvceArray5D<Real> u_free;

 private:
  MeshBlockPack *pmy_pack_;
  MeshBoundaryValuesCC *pbval_relax_;
  RelaxVars relax_;
  RelaxVars rhs_;
  FreeVars free_;
  bool enabled_;
  bool solved_;
  bool solve_once_;
  bool run_on_restart_;
  bool stop_after_solve_;
  bool skip_initial_output_;
  bool reject_worse_;
  bool stop_on_growth_;
  int growth_window_;
  int growth_start_iter_;
  int max_steps_;
  int history_every_;
  Real tolerance_;
  Real growth_tolerance_;
  Real relax_cfl_;
  Real eta_;
  Real diss_;
  Real residual_excision_radius_;
  Real wavespeed_scale_;
  Real wavespeed_center_[3];
  std::string wavespeed_mode_;
  // Pseudo-time step used by ExpRKUpdate(); set by SolveRelaxation() each
  // iteration before the per-stage loop runs.  We use the same gam0/gam1/
  // beta/delta low-storage coefficients as the Driver-level explicit
  // integrator (set via <time>/integrator), so e.g. <time>integrator=rk4
  // gives the AthenaK 4-stage 2-register RK4()4[2S] of Ketcheson (2010);
  // dtau replaces pmesh->dt in the standard ExpRKUpdate formula.
  Real dtau_;
  DualArray1D<Real> wavespeed_radii_;
  Real bare_mass_[2];
  Real pos_[2][3];
  Real mom_[2][3];
  Real spin_[2][3];
  FILE *history_file_;
  std::string history_name_;

  void BuildCTTFreeData();
  void BuildWaveSpeedProfile(Real dx_min);
  void ApplySolution();
  void SolveRelaxation(Driver *pdriver);
  void RefreshZ4cBoundariesAfterSolve(Driver *pdriver);
  void RecomputeConstraintsAfterSolve();
  void OpenHistory();
  void RecordHistory(int iter, Real tau, const Diagnostics &diag);
  Diagnostics ReduceDiagnostics(Real initial_residual_l2,
                                Real initial_residual_excised_l2);

  // ----- Z4c-style per-stage task methods ----------------------------------
  // These mirror the public task methods on z4c::Z4c one-for-one, in both
  // signature ((Driver *, int stage) -> TaskStatus) and behaviour, so that
  // the relaxation pseudo-time loop drives the same low-storage RK update
  // (CopyU -> CalcRHS -> ExpRKUpdate -> RestrictU -> SendU -> RecvU ->
  // ApplyPhysicalBCs -> Prolongate) and the same exchange/refinement
  // infrastructure that the Z4c evolution uses.  Physical boundaries are
  // ID-specific Sommerfeld/radiation boundaries, not Z4cBCs.
  TaskStatus InitRecv(Driver *pdrive, int stage);
  TaskStatus ClearRecv(Driver *pdrive, int stage);
  TaskStatus ClearSend(Driver *pdrive, int stage);
  TaskStatus CopyU(Driver *pdrive, int stage);
  template <int NGHOST>
  TaskStatus CalcRHS(Driver *pdrive, int stage);
  template <int NGHOST>
  void ApplyKODissipation();
  void ApplySommerfeldRHS();
  TaskStatus ExpRKUpdate(Driver *pdrive, int stage);
  TaskStatus RestrictU(Driver *pdrive, int stage);
  TaskStatus SendU(Driver *pdrive, int stage);
  TaskStatus RecvU(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *pdrive, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);

  // Helper used by both the pseudo-time loop's convergence reduction and
  // CalcRHS<NGHOST> below.  Writes the Hamiltonian-constraint residual
  // (Eq. 18 of NRPyElliptic, arXiv:2111.02424) into free_.residual at every
  // interior cell.
  template <int NGHOST>
  void ComputeResidual();
};

} // namespace z4c

#endif // Z4C_ID_SOLVE_HPP_
