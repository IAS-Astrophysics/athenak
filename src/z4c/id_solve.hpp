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

 private:
  void TransferCoefficientsFromBlocksToRoot();

  IDConformalThinSandwich *owner_;
  Real omega_;
  bool reject_worse_;

  friend class IDCTSMultigrid;
};

class IDConformalThinSandwich {
 public:
  IDConformalThinSandwich(MeshBlockPack *pmbp, ParameterInput *pin);
  ~IDConformalThinSandwich();

  TaskStatus SolveTask(Driver *pdriver, int stage);
  void BuildFreeData();
  void ApplySolution();
  void RecordConstraintHistory(int iter, Real defect);

  DvceArray5D<Real> u_cts;
  DvceArray5D<Real> u_free;
  DvceArray5D<Real> u_defect;

 private:
  MeshBlockPack *pmy_pack_;
  IDCTSMultigridDriver *pmgd_;
  bool enabled_;
  bool solved_;
  bool solve_once_;
  bool full_multigrid_;
  bool fill_horizon_junk_;
  int history_every_;
  Real horizon_radius_;
  Real horizon_center_[3];
  FILE *history_file_;
  std::string history_name_;

  template <int NGHOST>
  void BuildGammaDotAndDK();

  friend class IDCTSMultigridDriver;
  friend class IDCTSMultigrid;
};

} // namespace z4c

#endif // Z4C_ID_SOLVE_HPP_
