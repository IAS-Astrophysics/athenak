#ifndef MULTIGRID_MULTIGRID_HPP_
#define MULTIGRID_MULTIGRID_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid.hpp
//  \brief defines the Multigrid base class

// C headers

// C++ headers
#include <cstdint>  // std::int64_t
#include <cstdio> // std::size_t
#include <iostream>

// AthenaK headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../tasklist/task_list.hpp"
#include "../bvals/bvals.hpp"

class Mesh;
class MeshBlockPack;
class ParameterInput;
class Coordinates;
class MultigridDriver;
class MultigridBoundaryValues;

enum class MGVariable {src, u, coeff};
enum class MGNormType {max, l1, l2};

struct MultigridTaskIDs {
      TaskID send0;
      TaskID recv0;
      TaskID physb0;
      TaskID send1;
      TaskID recv1;
      TaskID physb1;
      TaskID sendR;
      TaskID recvR;
      TaskID physbR;
      TaskID smoothR;
      TaskID clearR;
      TaskID sendB;
      TaskID recvB;
      TaskID physbB;
      TaskID smoothB;
      TaskID clearB;
      TaskID restrict;
      TaskID prolongate;
      TaskID calc_rhs;
      TaskID apply_mask;
      TaskID zero_clear;
      TaskID store_old;
      TaskID smooth;
};

//! \class Multigrid
//  \brief Multigrid object containing each MeshBlock and/or the root block

class Multigrid {
 public:
  Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost);
  virtual ~Multigrid();

  // KGF: Both btype=BoundaryQuantity::mg and btypef=BoundaryQuantity::mg_faceonly (face
  // neighbors only) are passed to comm function calls in mg_task_list.cpp Only
  // BoundaryQuantity::mg is handled in a case in InitBoundaryData(). Passed directly
  // (not through btype) in MGBoundaryValues() ctor
  MultigridBoundaryValues *pbval = nullptr;
  void LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh);
  void LoadSource(const DvceArray5D<Real> &src, int ns, int ngh, Real fac);
  void LoadCoefficients(const DvceArray5D<Real> &coeff, int ngh);
  void ApplyMask();
  void RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh);
  void RetrieveDefect(DvceArray5D<Real> &dst, int ns, int ngh);
  void ZeroClearData();
  void RestrictPack();
  void RestrictCoefficients();
  void ProlongateAndCorrectPack();
  void SmoothPack(int color);
  void CalculateDefectPack();
  void CalculateFASRHSPack();
  void ComputeCorrection();
  void CalculateMatrixPack(Real dt);
  void SetFromRootGrid(bool folddata);
  Real CalculateDefectNorm(MGNormType nrm, int n);
  Real CalculateTotal(MGVariable type, int n);
  void SubtractAverage(MGVariable type, int n, Real ave);
  void StoreOldData();
  Real GetCoarsestData(MGVariable type, int m, int n);
  void SetData(MGVariable type, int n, int k, int j, int i, Real v);
  void PrintActiveRegion(const DvceArray5D<Real> &data);
  void PrintAll(const DvceArray5D<Real> &data);

  // small functions
  int GetCurrentNumberOfCells() { return 1<<current_level_; }
  int GetNumberOfLevels() { return nlevel_; }
  int GetCurrentLevel() { return current_level_; }
  int GetLevelShift() { return nlevel_ - 1 - current_level_; }
  int GetSize() { return indcs_.nx1; }
  DvceArray5D<Real>& GetCurrentData() { return u_[current_level_]; }
  DvceArray5D<Real>& GetCurrentSource() { return src_[current_level_]; }
  DvceArray5D<Real>& GetCurrentOldData() { return uold_[current_level_]; }
  DvceArray5D<Real>& GetCurrentCoefficient() { return coeff_[current_level_]; }

  // actual implementations of Multigrid operations
  void Restrict(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
                int nvar, int il, int iu, int jl, int ju, int kl, int ku, bool th);
  void ProlongateAndCorrect(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th);

  // physics-dependent virtual functions
  virtual void Smooth(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
                      const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrx,
                      int rlev, int il, int iu, int jl, int ju, int kl, int ku,
                      int color, bool th) = 0;
  virtual void CalculateDefect(DvceArray5D<Real> &def, const DvceArray5D<Real> &u,
               const DvceArray5D<Real> &src, const DvceArray5D<Real> &coeff,
               const DvceArray5D<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
               int kl, int ku, bool th) = 0;
  virtual void CalculateFASRHS(DvceArray5D<Real> &def, const DvceArray5D<Real> &src,
                 const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrix,
                 int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) = 0;
  
  friend class MultigridDriver;

 protected:
  MultigridDriver *pmy_driver_;
  MeshBlock *pmy_block_;
  MeshBlockPack *pmy_pack_;
  Mesh *pmy_mesh_;
  LogicalLocation loc_;
  RegionSize size_;
  RegionIndcs indcs_;
  BoundaryFlag mg_block_bcs_[6];
  int nlevel_, ngh_, nvar_, ncoeff_, nmatrix_, current_level_;
  int nmmbx1_, nmmbx2_, nmmbx3_;
  int nmmb_;
  Real rdx_, rdy_, rdz_;
  Real defscale_;
  DvceArray5D<Real> *u_, *def_, *src_, *uold_, *coeff_, *matrix_;
  Coordinates *coord_, *ccoord_;
};


//! \class MultigridDriver
//  \brief Multigrid driver

class MultigridDriver {
 public:
  MultigridDriver(MeshBlockPack *pmbp, int invar);
  virtual ~MultigridDriver();

  // pure virtual function
  virtual void Solve(Driver *pdriver, int step, Real dt = 0.0) = 0;
  int GetCoffset() const { return coffset_; }

  friend class Multigrid;

 protected:
  void CheckBoundaryFunctions();
  void SubtractAverage(MGVariable type);
  void SetupMultigrid(Real dt, bool ftrivial);
  void TransferFromBlocksToRoot(bool initflag);
  void TransferFromRootToBlocks(bool folddata);
  void OneStepToFiner(Driver *pdriver,int nsmooth);
  void OneStepToCoarser(Driver *pdriver,int nsmooth);
  void SolveVCycle(Driver *pdriver,int npresmooth, int npostsmooth);
  void SolveIterative(Driver *pdriver);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SendBoundary(Driver *pdrive, int stag);
  TaskStatus RecvBoundary(Driver *pdrive, int stag);
  TaskStatus StartReceive(Driver *pdrive, int stag);
  TaskStatus StartReceiveFluxCons(Driver *pdrive, int stag);
  TaskStatus StartReceiveForProlongation(Driver *pdrive, int stag);
  TaskStatus ClearBoundary(Driver *pdrive, int stag);
  TaskStatus ClearBoundaryFluxCons(Driver *pdrive, int stag);
  TaskStatus SendBoundaryFluxCons(Driver *pdrive, int stag);
  TaskStatus SendBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus ReceiveBoundaryFluxCons(Driver *pdrive, int stag);
  TaskStatus ReceiveBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus SmoothRed(Driver *pdrive, int stag);
  TaskStatus SmoothBlack(Driver *pdrive, int stag);
  TaskStatus Smooth(Driver *pdrive, int stag);
  TaskStatus PhysicalBoundary(Driver *pdrive, int stag);
  TaskStatus Restrict(Driver *pdrive, int stag);
  TaskStatus Prolongate(Driver *pdrive, int stag);
  TaskStatus ProlongateBoundary(Driver *pdrive, int stag);
  TaskStatus ProlongateBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus CalculateFASRHS(Driver *pdrive, int stag);
  TaskStatus StoreOldData(Driver *pdrive, int stag);
  void SetMGTaskListToFiner(int nsmooth, int ngh);
  void SetMGTaskListToCoarser(int nsmooth, int ngh);
  void SetMGTaskListBoundaryCommunication();
  void DoTaskListOneStage();
  void MGRootBoundary(const DvceArray5D<Real> &u);

  virtual void SolveCoarsestGrid();
  Real CalculateDefectNorm(MGNormType nrm, int n);
  void CalculateMatrix(Real dt);
  Multigrid* FindMultigrid(int tgid);
  // container to hold names of TaskIDs
  MultigridTaskIDs id;

  // small functions
  int GetNumMultigrids() { return nblist_[global_variable::my_rank]; }

  int nranks_, nthreads_, nbtotal_, nvar_, ncoeff_, nmatrix_, mode_;
  int locrootlevel_, nrootlevel_, nmblevel_, ntotallevel_, nreflevel_, maxreflevel_;
  int current_level_, fmglevel_;
  int *nslist_, *nblist_, *nvlist_, *nvslist_, *nvlisti_, *nvslisti_,
                          *nclist_, *ncslist_, *ranklist_;
  int nrbx1_, nrbx2_, nrbx3_;
  BoundaryFlag mg_mesh_bcs_[6];
  //MGBoundaryFunc MGBoundaryFunction_[6];
  //MGBoundaryFunc MGCoeffBoundaryFunction_[6];
  //MGMaskFunc srcmask_, coeffmask_;
  Mesh *pmy_mesh_;
  MeshBlockPack *pmy_pack_;
  Multigrid *mgroot_;
  Multigrid *mglevels_;
  Multigrid *pmg;
  bool fsubtract_average_, needinit_;
  Real last_ave_;
  Real eps_;
  int niter_, npresmooth_, npostsmooth_;
  int os_, oe_;
  int coffset_;
  int fprolongation_;
  int fshowdef_;

  DvceArray5D<Real> *cbuf_, *cbufold_;
  DvceArray3D<bool> *ncoarse_;

 private:
  Real *rootbuf_;
  int nb_rank_;
};

class MultigridBoundaryValues : public MeshBoundaryValuesCC {
 public:
  MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse, Multigrid *pmg);
  ~MultigridBoundaryValues();

  // pack/restrict fluxes at fine/coarse boundaries into boundary buffers and send
  TaskStatus PackAndSendMG(const DvceArray5D<Real> &u);
  // receive/unpack fluxes at fine/coarse boundaries from boundary buffers and
  TaskStatus RecvAndUnpackMG(DvceArray5D<Real> &u);

 private:
  // data
  Multigrid *pmy_mg;
  // functions
};

//KOKKOS_INLINE_FUNCTION
//Real RestrictOne(const DvceArray5D<Real> &src, int m, int v, int fi, int fj, int fk) {
//  return 0.125*(src(m, v, fk,   fj,   fi)+src(m, v, fk,   fj,   fi+1)
//               +src(m, v, fk,   fj+1, fi)+src(m, v, fk,   fj+1, fi+1)
//               +src(m, v, fk+1, fj,   fi)+src(m, v, fk+1, fj,   fi+1)
//               +src(m, v, fk+1, fj+1, fi)+src(m, v, fk+1, fj+1, fi+1));
//}



#endif // MULTIGRID_MULTIGRID_HPP_
