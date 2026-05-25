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

  IDConformalThinSandwich(MeshBlockPack *pmbp, ParameterInput *pin);
  ~IDConformalThinSandwich();

  TaskStatus SolveTask(Driver *pdriver, int stage);
  void PrepareForRestart();
  bool StopAfterSolveRequested() const { return stop_after_solve_ && solved_; }
  bool SkipInitialOutput() const { return skip_initial_output_; }

  DvceArray5D<Real> u_relax;
  DvceArray5D<Real> u_relax_tmp;
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
  int max_steps_;
  int history_every_;
  Real tolerance_;
  Real relax_cfl_;
  Real eta_;
  Real wavespeed_scale_;
  Real wavespeed_center_[3];
  std::string wavespeed_mode_;
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
  void RecordHistory(int iter, Real residual_l2, Real max_update);

  TaskStatus InitRecv();
  TaskStatus Send();
  TaskStatus Recv();
  TaskStatus ClearSend();
  TaskStatus ClearRecv();
  void Restrict();
  void Prolongate();
  void ApplyPhysicalBCs();

  template <int NGHOST>
  void ComputeResidual();
  template <int NGHOST>
  void RKStep(Real dtau, Real dx_min);
};

} // namespace z4c

#endif // Z4C_ID_SOLVE_HPP_
