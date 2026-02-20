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
#include <cstring> // memcpy
#include <iostream>
#include <unordered_map>
#include <vector>

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

//----------------------------------------------------------------------------------------
// LogicalLocation hash and equality for std::unordered_map

inline bool operator==(const LogicalLocation &l1, const LogicalLocation &l2) {
  return (l1.level == l2.level) && (l1.lx1 == l2.lx1)
      && (l1.lx2 == l2.lx2) && (l1.lx3 == l2.lx3);
}

inline std::int64_t rotl64(std::int64_t i, int s) {
  return (i << s) | (i >> (64 - s));
}

struct LogicalLocationHash {
  std::size_t operator()(const LogicalLocation &l) const {
    return static_cast<std::size_t>(l.lx1 ^ rotl64(l.lx2, 21) ^ rotl64(l.lx3, 42));
  }
};

//----------------------------------------------------------------------------------------
//! \class MGOctet
//  \brief structure containing 2x2x2 interior cells (+ ghost) for mesh refinement
//  Each octet represents a "parent cell" that has children at a finer level.
//  Arrays are stored as flat std::vector<Real> with 4D indexing (v, k, j, i).

class MGOctet {
 public:
  LogicalLocation loc;
  bool fleaf;
  int nc, nvar;  // nc = 2 + 2*ngh

  std::vector<Real> u, def, src, uold;

  void Allocate(int nv, int ngh) {
    nc = 2 + 2*ngh;
    nvar = nv;
    int sz = nv * nc * nc * nc;
    u.assign(sz, 0.0);
    def.assign(sz, 0.0);
    src.assign(sz, 0.0);
    uold.assign(sz, 0.0);
  }

  void ZeroClearU() { std::fill(u.begin(), u.end(), 0.0); }
  void ZeroClearSrc() { std::fill(src.begin(), src.end(), 0.0); }
  void StoreOld() { uold = u; }

  inline Real& U(int v, int k, int j, int i) {
    return u.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline Real& Def(int v, int k, int j, int i) {
    return def.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline Real& Src(int v, int k, int j, int i) {
    return src.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline Real& Uold(int v, int k, int j, int i) {
    return uold.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline const Real& U(int v, int k, int j, int i) const {
    return u.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline const Real& Def(int v, int k, int j, int i) const {
    return def.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline const Real& Src(int v, int k, int j, int i) const {
    return src.at(((v*nc + k)*nc + j)*nc + i);
  }
  inline const Real& Uold(int v, int k, int j, int i) const {
    return uold.at(((v*nc + k)*nc + j)*nc + i);
  }
};

struct MultigridTaskIDs {
      TaskID send0;
      TaskID ircv0;
      TaskID recv0;
      TaskID physb0;
      TaskID send1;
      TaskID ircv1;
      TaskID recv1;
      TaskID physb1;
      TaskID sendR;
      TaskID ircvR;
      TaskID recvR;
      TaskID physbR;
      TaskID smoothR;
      TaskID clearR;
      TaskID sendB;
      TaskID ircvB;
      TaskID recvB;
      TaskID physbB;
      TaskID smoothB;
      TaskID sendR2;
      TaskID ircvR2;
      TaskID recvR2;
      TaskID physbR2;
      TaskID smoothR2;
      TaskID sendB2;
      TaskID ircvB2;
      TaskID recvB2;
      TaskID physbB2;
      TaskID smoothB2;
      TaskID restrict_;
      TaskID prolongate;
      TaskID fmg_prolongate;
      TaskID calc_rhs;
      TaskID apply_mask;
      TaskID zero_clear;
      TaskID store_old;
      TaskID smooth;
      TaskID clear_recv0;
      TaskID clear_send0;
      TaskID clear_recv1;
      TaskID clear_send1;
      TaskID clear_recvR;
      TaskID clear_sendR;
      TaskID clear_recvB;
      TaskID clear_sendB;
      TaskID clear_recvR2;
      TaskID clear_sendR2;
      TaskID clear_recvB2;
      TaskID clear_sendB2;
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
  void CopySourceToData();
  void RestrictPack();
  void RestrictSourcePack();
  void RestrictCoefficients();
  void FMGProlongatePack();
  void ProlongateAndCorrectPack();
  void SmoothPack(int color);
  void CalculateDefectPack();
  void CalculateFASRHSPack();
  void ComputeCorrection();
  void CalculateMatrixPack(Real dt);
  void SetFromRootGrid(bool folddata);
  Real CalculateDefectNorm(MGNormType nrm, int n);
  Real CalculateAverage(MGVariable type);
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
  int GetGhostCells() { return ngh_; }
  Real GetRootDx() { return rdx_; }
  DvceArray5D<Real>& GetCurrentData() { return u_[current_level_]; }
  DvceArray5D<Real>& GetCurrentSource() { return src_[current_level_]; }
  DvceArray5D<Real>& GetCurrentOldData() { return uold_[current_level_]; }
  DvceArray5D<Real>& GetCurrentCoefficient() { return coeff_[current_level_]; }

  // actual implementations of Multigrid operations
  void Restrict(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
                int nvar, int il, int iu, int jl, int ju, int kl, int ku, bool th);
  void ProlongateAndCorrect(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th);
  void FMGProlongate(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl);

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
  friend class MultigridBoundaryValues;

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
  DvceArray1D<Real> block_rdx_;
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
  void MGRootBoundary(const DvceArray5D<Real> &u);
  void TransferFromBlocksToRoot(bool initflag);
  void TransferFromRootToBlocks(bool folddata);

  // per-cell octet operations (Athena++ style)
  void InitializeOctets();
  void SmoothOctets(int color);
  void RestrictOctets();
  void ProlongateAndCorrectOctets();
  void FMGProlongateOctets();
  void SetBoundariesOctets(bool fprolong, bool folddata);
  void ProlongateOctetBoundaries(MGOctet &oct,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, const std::vector<bool> &ncoarse, bool folddata);
  void StoreOldDataOctets();
  void CalculateFASRHSOctets();
  void ZeroClearOctets();
  void RestrictFMGSourceOctets();
  void RestrictOctetsBeforeTransfer();
  void SetOctetBoundariesBeforeTransfer(bool folddata);
  void SetOctetBoundarySameLevel(MGOctet &dst, const MGOctet &src,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, int ox1, int ox2, int ox3, bool folddata);
  void SetOctetBoundaryFromCoarser(const std::vector<Real> &un,
       const std::vector<Real> &unold,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, int un_nc, const LogicalLocation &loc,
       int ox1, int ox2, int ox3, bool folddata);

  // physics-dependent octet operations (virtual, overridden in derived drivers)
  virtual void SmoothOctet(MGOctet &oct, int rlev, int color) = 0;
  virtual void CalculateDefectOctet(MGOctet &oct, int rlev) = 0;
  virtual void CalculateFASRHSOctet(MGOctet &oct, int rlev) = 0;

  DualArray2D<Real> rootbuf_;

  friend class Multigrid;

 protected:
  void CheckBoundaryFunctions();
  void SubtractAverage(MGVariable type);
  void SetupMultigrid(Real dt, bool ftrivial);
  void FMGProlongate(Driver *pdriver);
  void OneStepToFiner(Driver *pdriver,int nsmooth);
  void OneStepToCoarser(Driver *pdriver,int nsmooth);
  void SolveVCycle(Driver *pdriver,int npresmooth, int npostsmooth);
  void SolveFMG(Driver *pdriver);
  void SolveMG(Driver *pdriver);
  void SolveFMGCoarser();
  void SolveIterative(Driver *pdriver);
  void SolveIterativeFixedTimes(Driver *pdriver);
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
  TaskStatus FMGProlongateTask(Driver *pdrive, int stag);
  TaskStatus ProlongateBoundary(Driver *pdrive, int stag);
  TaskStatus ProlongateBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus CalculateFASRHS(Driver *pdrive, int stag);
  TaskStatus StoreOldData(Driver *pdrive, int stag);
  TaskStatus ClearRecv(Driver *pdrive, int stag);
  TaskStatus ClearSend(Driver *pdrive, int stag);
  void SetMGTaskListToFiner(int nsmooth, int ngh);
  void SetMGTaskListFMGProlongate(int ngh);
  void SetMGTaskListToCoarser(int nsmooth, int ngh);
  void SetMGTaskListBoundaryCommunication();
  void DoTaskListOneStage();

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
  bool full_multigrid_;
  int fmg_ncycle_;

  // per-cell octets (Athena++ style)
  std::vector<MGOctet> *octets_;
  std::unordered_map<LogicalLocation, int, LogicalLocationHash> *octetmap_;
  std::vector<bool> *octetbflag_;
  int *noctets_;
  std::vector<Real> cbuf_, cbufold_;  // scratch buffers for boundary exchange
  std::vector<bool> ncoarse_;         // 3x3x3 flags for coarser neighbors

  // helper to read root grid data on host
  inline Real RootU(const Kokkos::View<Real*****, Kokkos::HostSpace> &h,
                    int v, int k, int j, int i) const { return h(0,v,k,j,i); }

 private:
  int nb_rank_;
};

class MultigridBoundaryValues : public MeshBoundaryValuesCC {
 public:
  MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse, Multigrid *pmg);
  ~MultigridBoundaryValues();

  // pack/restrict fluxes at fine/coarse boundaries into boundary buffers and send
  TaskStatus PackAndSendMG(const DvceArray5D<Real> &u);
  TaskStatus RecvAndUnpackMG(DvceArray5D<Real> &u);
  TaskStatus InitRecvMG(const int nvars);

 private:
  // data
  Multigrid *pmy_mg;
  // functions
};

inline Real RestrictOne(const MGOctet &oct, int v, int fi, int fj, int fk) {
  return 0.125*(oct.U(v, fk,   fj,   fi)   + oct.U(v, fk,   fj,   fi+1)
               +oct.U(v, fk,   fj+1, fi)   + oct.U(v, fk,   fj+1, fi+1)
               +oct.U(v, fk+1, fj,   fi)   + oct.U(v, fk+1, fj,   fi+1)
               +oct.U(v, fk+1, fj+1, fi)   + oct.U(v, fk+1, fj+1, fi+1));
}

inline Real RestrictOneSrc(const MGOctet &oct, int v, int fi, int fj, int fk) {
  return 0.125*(oct.Src(v, fk,   fj,   fi)   + oct.Src(v, fk,   fj,   fi+1)
               +oct.Src(v, fk,   fj+1, fi)   + oct.Src(v, fk,   fj+1, fi+1)
               +oct.Src(v, fk+1, fj,   fi)   + oct.Src(v, fk+1, fj,   fi+1)
               +oct.Src(v, fk+1, fj+1, fi)   + oct.Src(v, fk+1, fj+1, fi+1));
}

inline Real RestrictOneDef(const MGOctet &oct, int v, int fi, int fj, int fk) {
  return 0.125*(oct.Def(v, fk,   fj,   fi)   + oct.Def(v, fk,   fj,   fi+1)
               +oct.Def(v, fk,   fj+1, fi)   + oct.Def(v, fk,   fj+1, fi+1)
               +oct.Def(v, fk+1, fj,   fi)   + oct.Def(v, fk+1, fj,   fi+1)
               +oct.Def(v, fk+1, fj+1, fi)   + oct.Def(v, fk+1, fj+1, fi+1));
}

// access flat buffer of size (nvar, nc, nc, nc)
inline Real& BufRef(std::vector<Real> &buf, int nc, int v, int k, int j, int i) {
  return buf.at(((v*nc + k)*nc + j)*nc + i);
}
inline const Real& BufRef(const std::vector<Real> &buf, int nc,
                          int v, int k, int j, int i) {
  return buf.at(((v*nc + k)*nc + j)*nc + i);
}


#endif // MULTIGRID_MULTIGRID_HPP_
