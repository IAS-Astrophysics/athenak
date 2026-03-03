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
#include <type_traits>
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
// Precomputed neighbor info for each of 27 directions (center unused).
// Indexed by (ox3+1)*9 + (ox2+1)*3 + (ox1+1).

struct OctetNeighborInfo {
  int same_id;    // same-level neighbor octet ID, or -1 if coarser/absent
  int coarse_id;  // coarser-level neighbor octet ID, or -1 if from root
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
  OctetNeighborInfo neighbors[27];

  // Raw pointers into contiguous per-level buffers managed by MultigridDriver.
  Real *u, *def, *src, *uold;

  void Init(int nv, int ngh) {
    nc = 2 + 2*ngh;
    nvar = nv;
    u = def = src = uold = nullptr;
  }

  int size() const { return nvar * nc * nc * nc; }
  void ZeroClearU() { std::memset(u, 0, size() * sizeof(Real)); }
  void ZeroClearSrc() { std::memset(src, 0, size() * sizeof(Real)); }
  void StoreOld() { std::memcpy(uold, u, size() * sizeof(Real)); }

  inline Real& U(int v, int k, int j, int i) {
    return u[((v*nc + k)*nc + j)*nc + i];
  }
  inline Real& Def(int v, int k, int j, int i) {
    return def[((v*nc + k)*nc + j)*nc + i];
  }
  inline Real& Src(int v, int k, int j, int i) {
    return src[((v*nc + k)*nc + j)*nc + i];
  }
  inline Real& Uold(int v, int k, int j, int i) {
    return uold[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& U(int v, int k, int j, int i) const {
    return u[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& Def(int v, int k, int j, int i) const {
    return def[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& Src(int v, int k, int j, int i) const {
    return src[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& Uold(int v, int k, int j, int i) const {
    return uold[((v*nc + k)*nc + j)*nc + i];
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
      TaskID fc_ghosts0;
      TaskID fc_ghostsR;
      TaskID fc_ghostsB;
      TaskID fc_ghostsR2;
      TaskID fc_ghostsB2;
      TaskID fc_ghosts_prol;
};

//! \class Multigrid
//  \brief Multigrid object containing each MeshBlock and/or the root block

class Multigrid {
 public:
  Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
            bool on_host = false);
  virtual ~Multigrid();

  // KGF: Both btype=BoundaryQuantity::mg and btypef=BoundaryQuantity::mg_faceonly (face
  // neighbors only) are passed to comm function calls in mg_task_list.cpp Only
  // BoundaryQuantity::mg is handled in a case in InitBoundaryData(). Passed directly
  // (not through btype) in MGBoundaryValues() ctor
  MultigridBoundaryValues *pbval = nullptr;
  void ReallocateForAMR();
  void UpdateBlockDx();
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
  virtual void SmoothPack(int color) = 0;
  virtual void CalculateDefectPack() = 0;
  virtual void CalculateFASRHSPack() = 0;
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
  auto GetCurrentData() { return u_[current_level_].d_view; }
  auto GetCurrentSource() { return src_[current_level_].d_view; }
  auto GetCurrentOldData() { return uold_[current_level_].d_view; }
  auto GetCurrentCoefficient() { return coeff_[current_level_].d_view; }
  auto GetCurrentData_h() { return u_[current_level_].h_view; }
  auto GetCurrentSource_h() { return src_[current_level_].h_view; }
  auto GetCurrentOldData_h() { return uold_[current_level_].h_view; }
  bool OnHost() const { return on_host_; }


  // actual implementations of Multigrid operations (templated on view type)
  template <typename ViewType>
  void Restrict(ViewType &dst, const ViewType &src,
                int nvar, int il, int iu, int jl, int ju, int kl, int ku, bool th);
  template <typename ViewType>
  void ProlongateAndCorrect(ViewType &dst, const ViewType &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th);
  template <typename ViewType>
  void FMGProlongate(ViewType &dst, const ViewType &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl);

  // Physics-dependent operations templated on view type and stencil.
  // The stencil functor must provide:
  //   Real Apply(const ViewType&, const ViewType&, int m, int v, int k, int j, int i)
  //   Real omega_over_diag
  template <typename ViewType, typename StencilOp>
  void Smooth(ViewType &u, const ViewType &src, const ViewType &coeff,
              const ViewType &matrix, const StencilOp &stencil, int rlev,
              int il, int iu, int jl, int ju, int kl, int ku, int color, bool th) {
    using ExeSpace = typename ViewType::execution_space;
    auto brdx = [this]() {
      if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
        return block_rdx_.h_view;
      else
        return block_rdx_.d_view;
    }();
    int rlev_l = rlev;
    Real odiag = stencil.omega_over_diag;
    color ^= pmy_driver_->GetCoffset();
    par_for("Multigrid::Smooth", ExeSpace(), 0, nmmb_-1, kl, ku, jl, ju,
    KOKKOS_LAMBDA(const int m, const int k, const int j) {
      Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev_l))
                              : brdx(m) / static_cast<Real>(1<<rlev_l);
      Real dx2 = dx * dx;
      const int c = (color + k + j) & 1;
      for (int i = il + c; i <= iu; i += 2) {
        Real lap = stencil.Apply(u, coeff, m, 0, k, j, i);
        u(m,0,k,j,i) -= (lap - src(m,0,k,j,i)*dx2) * odiag;
      }
    });
  }

  template <typename ViewType, typename StencilOp>
  void CalculateDefect(ViewType &def, const ViewType &u, const ViewType &src,
                       const ViewType &coeff, const ViewType &matrix,
                       const StencilOp &stencil, int rlev,
                       int il, int iu, int jl, int ju, int kl, int ku, bool th) {
    using ExeSpace = typename ViewType::execution_space;
    auto brdx = [this]() {
      if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
        return block_rdx_.h_view;
      else
        return block_rdx_.d_view;
    }();
    int rlev_l = rlev;
    par_for("Multigrid::CalculateDefect", ExeSpace(), 0, nmmb_-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev_l))
                              : brdx(m) / static_cast<Real>(1<<rlev_l);
      Real idx2 = 1.0 / (dx * dx);
      def(m,0,k,j,i) = src(m,0,k,j,i) - stencil.Apply(u, coeff, m, 0, k, j, i) * idx2;
    });
  }

  template <typename ViewType, typename StencilOp>
  void CalculateFASRHS(ViewType &src, const ViewType &u, const ViewType &coeff,
                       const ViewType &matrix, const StencilOp &stencil, int rlev,
                       int il, int iu, int jl, int ju, int kl, int ku, bool th) {
    using ExeSpace = typename ViewType::execution_space;
    auto brdx = [this]() {
      if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
        return block_rdx_.h_view;
      else
        return block_rdx_.d_view;
    }();
    int rlev_l = rlev;
    par_for("Multigrid::CalculateFASRHS", ExeSpace(), 0, nmmb_-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev_l))
                              : brdx(m) / static_cast<Real>(1<<rlev_l);
      Real idx2 = 1.0 / (dx * dx);
      src(m,0,k,j,i) += stencil.Apply(u, coeff, m, 0, k, j, i) * idx2;
    });
  }
  
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
  bool on_host_;

  Real rdx_, rdy_, rdz_;
  Real defscale_;
  DualArray1D<Real> block_rdx_;
  DvceArray1D<int> fc_childx_, fc_childy_, fc_childz_;
  DualArray5D<Real> *u_, *def_, *src_, *uold_, *coeff_, *matrix_;
  Coordinates *coord_, *ccoord_;
};


//! \class MultigridDriver
//  \brief Multigrid driver

class MultigridDriver {
 public:
  MultigridDriver(MeshBlockPack *pmbp, int invar);
  virtual ~MultigridDriver();

  auto GetRootData_h() { return mgroot_->GetCurrentData_h(); }
  auto GetRootOldData_h() { return mgroot_->GetCurrentOldData_h(); }
  auto GetRootSource_h() { return mgroot_->GetCurrentSource_h(); }
  // pure virtual function
  virtual void Solve(Driver *pdriver, int step, Real dt = 0.0) = 0;
  void PrepareForAMR();
  int GetCoffset() const { return coffset_; }
  void MGRootBoundary();
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
  void PreRestrictOctetU();
  void RestrictOctetsBeforeTransfer();
  void SetOctetBoundariesBeforeTransfer(bool folddata);
  void SetOctetBoundarySameLevel(MGOctet &dst, const MGOctet &src,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, int ox1, int ox2, int ox3, bool folddata);
  void SetOctetBoundaryFromCoarser(const Real *un,
       const Real *unold,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, int un_nc, const LogicalLocation &loc,
       int ox1, int ox2, int ox3, bool folddata);
  void ApplyPhysicalBoundariesOctet(MGOctet &oct, bool fcbuf);

  // physics-dependent octet operations (virtual, overridden in derived drivers)
  virtual void SmoothOctet(MGOctet &oct, int rlev, int color) = 0;
  virtual void CalculateDefectOctet(MGOctet &oct, int rlev) = 0;
  virtual void CalculateFASRHSOctet(MGOctet &oct, int rlev) = 0;
  virtual void ProlongateOctetBoundariesFluxCons(MGOctet &oct,
       std::vector<Real> &cbuf, const std::vector<bool> &ncoarse);

  DualArray2D<Real> rootbuf_;

  TaskStatus PhysicalBoundary(Driver *pdrive, int stag);

  void AllocateMultipoleCoefficients();
  void CalculateMultipoleCoefficients();
  void ScaleMultipoleCoefficients();
  void CalculateCenterOfMass();

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
  TaskStatus Restrict(Driver *pdrive, int stag);
  TaskStatus Prolongate(Driver *pdrive, int stag);
  TaskStatus FMGProlongateTask(Driver *pdrive, int stag);
  TaskStatus ProlongateBoundary(Driver *pdrive, int stag);
  TaskStatus ProlongateBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus FillFCBoundary(Driver *pdrive, int stag);
  TaskStatus CalculateFASRHS(Driver *pdrive, int stag);
  TaskStatus StoreOldData(Driver *pdrive, int stag);
  TaskStatus ClearRecv(Driver *pdrive, int stag);
  TaskStatus ClearSend(Driver *pdrive, int stag);
  void SetMGTaskListToFiner(int nsmooth, int ngh, int flag=0);
  void SetMGTaskListFMGProlongate(int ngh);
  void SetMGTaskListToCoarser(int nsmooth, int ngh);
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
  int amr_seq_;  // tracks cumulative AMR events to detect mesh changes
  Real last_ave_;
  Real eps_;
  int niter_, npresmooth_, npostsmooth_;
  int os_, oe_;
  int coffset_;
  int fprolongation_;
  int fshowdef_;
  bool full_multigrid_;
  int fmg_ncycle_;

  // Source masking (zero source outside mask_radius_)
  Real mask_radius_;
  Real mask_origin_[3];

  // Multipole expansion boundary conditions
  int mporder_;     // -1: disabled, 0: detected but not yet configured, 2 or 4: active
  int nmpcoeff_;    // number of multipole coefficients
  Real mpcoeff_[25]; // multipole coefficients (max 25 for l=4)
  DvceArray1D<Real> d_mpcoeff_; // device copy of multipole coefficients
  Real mpo_[3];     // multipole expansion origin
  bool autompo_;    // automatically compute center of mass as origin
  bool nodipole_;   // suppress dipole moment (assume origin = center of mass)
  void SyncMultipoleToDevice();

  // per-cell octets (Athena++ style)
  std::vector<MGOctet> *octets_;
  std::unordered_map<LogicalLocation, int, LogicalLocationHash> *octetmap_;
  std::vector<bool> *octetbflag_;
  int *noctets_;
  std::vector<Real> cbuf_, cbufold_;  // scratch buffers for boundary exchange
  std::vector<bool> ncoarse_;         // 3x3x3 flags for coarser neighbors

  // Contiguous per-level buffers backing MGOctet raw pointers.
  // Layout: octet_stride_ consecutive Reals per octet (nvar*nc*nc*nc).
  std::vector<Real> *oct_u_buf_, *oct_def_buf_, *oct_src_buf_, *oct_uold_buf_;
  int octet_stride_;  // elements per octet = nvar * nc^3

  std::vector<Real> root_u_buf_, root_uold_buf_;
  int root_buf_nc_;
  bool root_flat_buf_stale_;
  void BuildRootFlatBuffers();
  void SyncRootToHost();
  void SyncRootToDevice();

  enum class RootSyncState { SYNCED, HOST_MODIFIED, DEVICE_MODIFIED };
  RootSyncState root_sync_state_;
  void MarkRootDeviceModified();

 private:
  int nb_rank_;
};

class MultigridBoundaryValues : public MeshBoundaryValuesCC {
 public:
  MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse, Multigrid *pmg);
  ~MultigridBoundaryValues();

  void RemapIndicesForMG();

  // pack/restrict fluxes at fine/coarse boundaries into boundary buffers and send
  TaskStatus PackAndSendMG(const DvceArray5D<Real> &u);
  TaskStatus RecvAndUnpackMG(DvceArray5D<Real> &u);
  TaskStatus InitRecvMG(const int nvars);
  TaskStatus FillFineCoarseMGGhosts(DvceArray5D<Real> &u);

 private:
  Multigrid *pmy_mg;
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
  return buf[((v*nc + k)*nc + j)*nc + i];
}
inline const Real& BufRef(const std::vector<Real> &buf, int nc,
                          int v, int k, int j, int i) {
  return buf[((v*nc + k)*nc + j)*nc + i];
}

KOKKOS_INLINE_FUNCTION
Real EvalMultipolePhi(Real x, Real y, Real z,
                      const Real *mpc, int order) {
  Real x2 = x*x, y2 = y*y, z2 = z*z;
  Real xy = x*y, yz = y*z, zx = z*x;
  Real r2 = x2 + y2 + z2;
  Real ir2 = 1.0/r2, ir1 = Kokkos::sqrt(ir2);
  Real ir3 = ir2*ir1, ir5 = ir3*ir2;
  Real hx2my2 = 0.5*(x2-y2);
  Real phis = ir1*mpc[0]
    + ir3*(mpc[1]*y + mpc[2]*z + mpc[3]*x)
    + ir5*(mpc[4]*xy + mpc[5]*yz + (3.0*z2-r2)*mpc[6]
         + mpc[7]*zx + mpc[8]*hx2my2);
  if (order == 4) {
    Real ir7 = ir5*ir2, ir9 = ir7*ir2;
    Real x2mty2 = x2-3.0*y2;
    Real tx2my2 = 3.0*x2-y2;
    phis += ir7*(y*tx2my2*mpc[9] + x*x2mty2*mpc[15]
               + xy*z*mpc[10] + z*hx2my2*mpc[14]
               + (5.0*z2-r2)*(y*mpc[11] + x*mpc[13])
               + z*(z2-3.0*r2)*mpc[12])
         + ir9*(xy*hx2my2*mpc[16]
               + 0.125*(x2*x2mty2-y2*tx2my2)*mpc[24]
               + yz*tx2my2*mpc[17] + zx*x2mty2*mpc[23]
               + (7.0*z2-r2)*(xy*mpc[18] + hx2my2*mpc[22])
               + (7.0*z2-3.0*r2)*(yz*mpc[19] + zx*mpc[21])
               + (35.0*z2*z2-30.0*z2*r2+3.0*r2*r2)*mpc[20]);
  }
  return phis;
}


#endif // MULTIGRID_MULTIGRID_HPP_
