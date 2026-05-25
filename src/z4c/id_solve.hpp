#ifndef Z4C_ID_SOLVE_HPP_
#define Z4C_ID_SOLVE_HPP_

#include <cstdio>
#include <string>
#include <vector>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "multigrid/multigrid.hpp"
#include "tasklist/task_list.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;

namespace z4c {

class IDConformalThinSandwich;

enum IDCTSUVar {
  ID_CTS_PSI,
  ID_CTS_BETAX,
  ID_CTS_BETAY,
  ID_CTS_BETAZ,
  ID_CTS_NVAR
};

enum IDCTSSmootherStat {
  ID_CTS_SMOOTH_ACCEPTED,
  ID_CTS_SMOOTH_LIMITED,
  ID_CTS_SMOOTH_BACKTRACKED,
  ID_CTS_SMOOTH_FALLBACK,
  ID_CTS_SMOOTH_SINGULAR,
  ID_CTS_SMOOTH_NONFINITE,
  ID_CTS_SMOOTH_REJECTED,
  ID_CTS_SMOOTH_PSI_FLOOR,
  ID_CTS_SMOOTH_MAX_UPDATE,
  ID_CTS_SMOOTH_NSTAT
};

// CTS free data are stored with respect to the conformal background metric.
// ID_FREE_GXX...ID_FREE_GZZ are gamma_bar_ij.  ID_FREE_GDOTXX...
// ID_FREE_GDOTZZ store the contravariant trace-free tensor u_bar^ij_TF,
// despite the historical "GDOT" component names.
enum IDCTSFreeVar {
  ID_FREE_GXX, ID_FREE_GXY, ID_FREE_GXZ, ID_FREE_GYY, ID_FREE_GYZ, ID_FREE_GZZ,
  ID_FREE_GDOTXX, ID_FREE_GDOTXY, ID_FREE_GDOTXZ,
  ID_FREE_GDOTYY, ID_FREE_GDOTYZ, ID_FREE_GDOTZZ,
  ID_FREE_K,
  ID_FREE_DKX, ID_FREE_DKY, ID_FREE_DKZ,
  ID_FREE_ALPHA,
  ID_FREE_E,
  ID_FREE_PX, ID_FREE_PY, ID_FREE_PZ,
  ID_FREE_SOURCE,
  ID_FREE_MASK,
  ID_FREE_BASE_PSI,
  ID_FREE_BASE_BETAX, ID_FREE_BASE_BETAY, ID_FREE_BASE_BETAZ,
  ID_FREE_NVAR
};

class IDCTSMultigrid : public Multigrid {
 public:
  IDCTSMultigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                 bool on_host = false);
  ~IDCTSMultigrid();

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;
  void DiagnosticRestrictPack() final;

 private:
  void PrepareFrozenView();

  DualArray5D<Real> frozen_u_;
};

class IDCTSMultigridDriver : public MultigridDriver {
 public:
  IDCTSMultigridDriver(IDConformalThinSandwich *owner, MeshBlockPack *pmbp,
                       ParameterInput *pin);
  ~IDCTSMultigridDriver();

  void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;
  void SmoothOctet(MGOctet &oct, int rlev, int color) final;
  void CalculateDefectOctet(MGOctet &oct, int rlev) final;
  void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;
  void DiagnosticRestrictOctets(int lev) final;
  bool SolutionApplied() const { return solution_applied_; }

 private:
  void TransferCoefficientsFromBlocksToRoot();
  void ResetSmootherStats();
  void AccumulateSmootherStats(const Real stats[ID_CTS_SMOOTH_NSTAT]);
  void PrintSmootherStats(int iter) const;
  void ValidateCompositeFASOptions() const;
  void BuildCompositeMasks();
  void BuildCompositeMeshBlockMasks();
  void BuildCompositeRootAndOctetMasks();
  void PrintCompositeMaskDiagnostics() const;
  Real CalculateCompositeDefectNorm(MGNormType nrm, int n);

  IDConformalThinSandwich *owner_;
  Real omega_;
  Real defect_increase_tol_;
  Real ngs_jacobian_eps_;
  Real ngs_max_update_;
  Real smoother_max_update_fraction_;
  Real ngs_line_search_min_;
  Real smoother_stats_[ID_CTS_SMOOTH_NSTAT];
  int max_iter_;
  int octet_fd_stencil_;
  int mg_coarse_fd_stencil_;
  int smoother_type_;
  int ngs_iterations_;
  int ngs_line_search_steps_;
  bool reject_worse_;
  bool keep_best_solution_;
  bool stop_on_defect_increase_;
  bool allow_incomplete_amr_;
  bool solution_applied_;
  bool show_smoother_stats_;
  bool check_octet_coefficients_;
  bool composite_fas_;
  bool composite_second_order_only_;
  bool debug_composite_masks_;
  bool debug_composite_residual_;
  bool composite_masks_ready_;
  int composite_restriction_;
  bool debug_composite_restriction_;
  bool composite_restriction_self_check_done_;

  friend class IDCTSMultigrid;
};

class IDConformalThinSandwich {
 public:
  IDConformalThinSandwich(MeshBlockPack *pmbp, ParameterInput *pin);
  ~IDConformalThinSandwich();

  TaskStatus SolveTask(Driver *pdriver, int stage);
  void PrepareForRestart();
  void BuildFreeData();
  void ApplySolution();
  void RecordConstraintHistory(int iter, const Real defects[ID_CTS_NVAR]);
  bool StopAfterSolveRequested() const { return stop_after_solve_ && solved_; }
  bool SkipInitialOutput() const { return skip_initial_output_; }

  DvceArray5D<Real> u_cts;
  DvceArray5D<Real> u_free;
  DvceArray5D<Real> u_defect;

 private:
  MeshBlockPack *pmy_pack_;
  IDCTSMultigridDriver *pmgd_;
  bool enabled_;
  bool solved_;
  bool solve_once_;
  bool run_on_restart_;
  bool stop_after_solve_;
  bool skip_initial_output_;
  bool full_multigrid_;
  bool fill_horizon_junk_;
  bool mask_horizon_defect_;
  bool dump_constraint_diagnostics_;
  int history_every_;
  Real horizon_radius_;
  Real horizon_mask_radius_;
  Real horizon_center_[3];
  Real diagnostic_slice_z_;
  FILE *history_file_;
  std::string history_name_;

  template <int NGHOST>
  void BuildGammaDotAndDK();
  template <int NGHOST>
  void FillHorizonJunk();
  void RefreshZ4cBoundariesAfterSolve(Driver *pdriver);
  void RecomputeConstraintsAfterSolve();
  void WriteConstraintDiagnostics(const char *stage);

  friend class IDCTSMultigridDriver;
  friend class IDCTSMultigrid;
};

} // namespace z4c

#endif // Z4C_ID_SOLVE_HPP_
