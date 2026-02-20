//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid_driver.cpp
//! \brief implementation of functions in class MultigridDriver

// C headers

// C++ headers
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>    // abs
#include <iomanip>    // setprecision
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

// constructor, initializes data structures and parameters

MultigridDriver::MultigridDriver(MeshBlockPack *pmbp, int invar):
    nranks_(global_variable::nranks), nthreads_(1), nbtotal_(pmbp->pmesh->nmb_total),
    nvar_(invar),
    maxreflevel_(pmbp->pmesh->multilevel?pmbp->pmesh->max_level-pmbp->pmesh->root_level:0),
    nrbx1_(pmbp->pmesh->nmb_rootx1), nrbx2_(pmbp->pmesh->nmb_rootx2), nrbx3_(pmbp->pmesh->nmb_rootx3),
    pmy_pack_(pmbp),
    pmy_mesh_(pmbp->pmesh),
    needinit_(true), eps_(-1.0),
    niter_(-1), npresmooth_(1), npostsmooth_(1), coffset_(0), fprolongation_(0),
    nb_rank_(0), ncoeff_(0),
    octets_(nullptr), octetmap_(nullptr), octetbflag_(nullptr), noctets_(nullptr) {
  if (pmy_mesh_->mb_indcs.nx2==1 || pmy_mesh_->mb_indcs.nx3==1) {
    std::cout << "### FATAL ERROR in MultigridDriver::MultigridDriver" << std::endl
        << "Currently the Multigrid solver works only in 3D." << std::endl;
    exit(EXIT_FAILURE);
    return;
  }
  ranklist_  = new int[nbtotal_];
  int nv = nvar_*2;
  Kokkos::realloc(rootbuf_, nv, nbtotal_);
  for (int n = 0; n < nbtotal_; ++n)
    ranklist_[n] = pmy_mesh_->rank_eachmb[n];
  nslist_  = new int[nranks_];
  nblist_  = new int[nranks_];
  nvlist_  = new int[nranks_];
  nvslist_ = new int[nranks_];
  nvlisti_  = new int[nranks_];
  nvslisti_ = new int[nranks_];
  if (ncoeff_ > 0) {
    nclist_  = new int[nranks_];
    ncslist_ = new int[nranks_];
  }

  // Allocate octet arrays for max possible refinement levels
  if (maxreflevel_ > 0) {
    octets_ = new std::vector<MGOctet>[maxreflevel_];
    octetmap_ = new std::unordered_map<LogicalLocation, int, LogicalLocationHash>[maxreflevel_];
    octetbflag_ = new std::vector<bool>[maxreflevel_];
    noctets_ = new int[maxreflevel_]();
  }
}

//! destructor

MultigridDriver::~MultigridDriver() {
  delete [] ranklist_;
  delete [] nslist_;
  delete [] nblist_;
  delete [] nvlist_;
  delete [] nvslist_;
  delete [] nvlisti_;
  delete [] nvslisti_;
  delete [] octets_;
  delete [] octetmap_;
  delete [] octetbflag_;
  delete [] noctets_;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SubtractAverage(MGVariable type)
//  \brief Calculate the global average and subtract it

void MultigridDriver::SubtractAverage(MGVariable type) {
  pmg->SubtractAverage(type,0,pmg->CalculateAverage(type));
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial)
//  \brief initialize the source assuming that the source terms are already loaded

void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial) {
  locrootlevel_ = pmy_mesh_->root_level;
  nrootlevel_ = mgroot_->GetNumberOfLevels();
  nmblevel_ = mglevels_->GetNumberOfLevels();
  
  // Calculate number of refinement levels present in mesh
  nreflevel_ = 0;
  if (pmy_mesh_->multilevel) {
    for (int n = 0; n < nbtotal_; ++n) {
      int lev = pmy_mesh_->lloc_eachmb[n].level - locrootlevel_;
      nreflevel_ = std::max(nreflevel_, lev);
    }
    if (nreflevel_ > 0) {
      std::cout << "MultigridDriver::SetupMultigrid: Number of refinement levels = "
                << nreflevel_ << std::endl;
    }
  }
  
  // Include refinement levels in total (octets are V-cycle participants)
  ntotallevel_ = nrootlevel_ + nmblevel_ + nreflevel_ - 1;
  os_ = mgroot_->ngh_;
  oe_ = os_+1;
  
  if (needinit_) {
    for (int n = 0; n < nbtotal_; ++n)
      ranklist_[n] = pmy_mesh_->gids_eachrank[n];
    for (int n = 0; n < nranks_; ++n) {
      nslist_[n]  = pmy_mesh_->gids_eachrank[n];
      nblist_[n]  = pmy_mesh_->nmb_eachrank[n];
      nvslist_[n] = nslist_[n]*nvar_*2;
      nvlist_[n]  = nblist_[n]*nvar_*2;
      nvslisti_[n] = nslist_[n]*nvar_;
      nvlisti_[n]  = nblist_[n]*nvar_;
    }
    if (nreflevel_ > 0) {
      InitializeOctets();
    }
  }
  needinit_ = false;
  if (fsubtract_average_) {
    pmg = mglevels_;
    SubtractAverage(MGVariable::src);
  }
  current_level_ = ntotallevel_ - 1;
  fmglevel_ = current_level_;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::InitializeOctets()
//! \brief Create per-cell octets for each refinement level

void MultigridDriver::InitializeOctets() {
  int ngh = mgroot_->ngh_;
  const auto &loc = pmy_mesh_->lloc_eachmb;

  // Clear and rebuild
  for (int l = 0; l < nreflevel_; ++l) {
    octetmap_[l].clear();
    noctets_[l] = 0;
  }

  // Scan all blocks: for each block at level > root, find its parent cell at each
  // octet level. An octet at level l has LogicalLocation at level (locrootlevel_ + l).
  // It represents a 2x2x2 group of cells from level (locrootlevel_ + l + 1).
  for (int n = 0; n < nbtotal_; ++n) {
    int blevel = loc[n].level;
    int blev = blevel - locrootlevel_;
    if (blev <= 0) continue;

    // For each refinement level from 0 to blev-1, this block implies an octet exists
    for (int l = blev - 1; l >= 0; --l) {
      LogicalLocation oloc;
      int shift = blevel - (locrootlevel_ + l + 1);
      oloc.lx1 = static_cast<int>(loc[n].lx1) >> (shift + 1);
      oloc.lx2 = static_cast<int>(loc[n].lx2) >> (shift + 1);
      oloc.lx3 = static_cast<int>(loc[n].lx3) >> (shift + 1);
      oloc.level = locrootlevel_ + l;

      if (octetmap_[l].count(oloc) == 0) {
        int oid = noctets_[l];
        octetmap_[l][oloc] = oid;
        noctets_[l]++;

        if (static_cast<int>(octets_[l].size()) <= oid) {
          octets_[l].resize(oid + 1);
          octets_[l][oid].Allocate(nvar_, ngh);
        }
        octets_[l][oid].loc = oloc;
        octets_[l][oid].fleaf = false;
      }
    }
  }

  // Set fleaf for each octet: an octet is a leaf if none of its 8 children have octets
  for (int l = 0; l < nreflevel_; ++l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &oct = octets_[l][o];
      oct.fleaf = true;
      if (l < nreflevel_ - 1) {
        // Check if any finer octet has this one as parent
        for (int fo = 0; fo < noctets_[l+1]; ++fo) {
          LogicalLocation &floc = octets_[l+1][fo].loc;
          if ((floc.lx1>>1) == oct.loc.lx1 && (floc.lx2>>1) == oct.loc.lx2 &&
              (floc.lx3>>1) == oct.loc.lx3 && floc.level-1 == oct.loc.level) {
            oct.fleaf = false;
            break;
          }
        }
      }
    }
  }

  // Resize boundary flags
  for (int l = 0; l < nreflevel_; ++l) {
    octetbflag_[l].resize(noctets_[l], false);
  }

  // Allocate scratch buffers for boundary exchange
  int nv = std::max(nvar_, std::max(ncoeff_, 1));
  int cbnc = 3;  // coarse buffer is 3x3x3
  cbuf_.assign(nv * cbnc * cbnc * cbnc, 0.0);
  cbufold_.assign(nv * cbnc * cbnc * cbnc, 0.0);
  ncoarse_.assign(3 * 3 * 3, false);

  for (int l = 0; l < nreflevel_; ++l) {
    std::cout << "  Octet level " << l << ": " << noctets_[l] << " octets" << std::endl;
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromBlocksToRoot(bool initflag)
//! \brief collect the coarsest data and transfer to the root grid
//! Following Athena++: each block sends its coarsest cell data to either
//! the root grid (if at root level) or the appropriate octet.

void MultigridDriver::TransferFromBlocksToRoot(bool initflag) {
  const int nv = nvar_;
  auto rootbuf = rootbuf_;
  const auto &src_ = mglevels_->src_[0];
  const auto &u_ = mglevels_->u_[0];
  const int ngh_mb = mglevels_->ngh_;
  int nmmb = mglevels_->nmmb_ - 1;
  int padding = nslist_[global_variable::my_rank];
  par_for("Multigrid:SaveToRoot", DevExeSpace(), 0, nmmb, KOKKOS_LAMBDA(const int m) {
    for (int v = 0; v < nv; ++v) {
      rootbuf.d_view(v,    m+padding) = src_(m, v, ngh_mb, ngh_mb, ngh_mb);
      rootbuf.d_view(v+nv, m+padding) = u_(m, v, ngh_mb, ngh_mb, ngh_mb);
    }
  });
  rootbuf.template modify<DevExeSpace>();
  rootbuf.template sync<HostExeSpace>();
#if MPI_PARALLEL_ENABLED
  for (int v = 0; v < 2*nv; ++v) {
    MPI_Allgatherv(MPI_IN_PLACE, nblist_[global_variable::my_rank], MPI_ATHENA_REAL,
                   &rootbuf.h_view(v,0), nblist_, nslist_, MPI_ATHENA_REAL, MPI_COMM_WORLD);
  }
#endif

  const auto loc = pmy_mesh_->lloc_eachmb;
  int rootlevel = locrootlevel_;
  int ngh = mgroot_->ngh_;

  // Scatter block data into root grid or octets
  const auto &src_r = mgroot_->GetCurrentSource();
  const auto &u_r = mgroot_->GetCurrentData();
  auto src_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), src_r);
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_r);

  for (int n = 0; n < nbtotal_; ++n) {
    int i = static_cast<int>(loc[n].lx1);
    int j = static_cast<int>(loc[n].lx2);
    int k = static_cast<int>(loc[n].lx3);
    if (loc[n].level == rootlevel) {
      // Root-level block -> root grid
      for (int v = 0; v < nv; ++v) {
        src_h(0, v, k+ngh, j+ngh, i+ngh) = rootbuf.h_view(v, n);
        if (!initflag)
          u_h(0, v, k+ngh, j+ngh, i+ngh) = rootbuf.h_view(v+nv, n);
      }
    } else {
      // Refined block -> appropriate octet
      LogicalLocation oloc;
      oloc.lx1 = (loc[n].lx1 >> 1);
      oloc.lx2 = (loc[n].lx2 >> 1);
      oloc.lx3 = (loc[n].lx3 >> 1);
      oloc.level = loc[n].level - 1;
      int olev = oloc.level - rootlevel;
      int oid = octetmap_[olev][oloc];
      int oi = (i & 1) + ngh;
      int oj = (j & 1) + ngh;
      int ok = (k & 1) + ngh;
      MGOctet &oct = octets_[olev][oid];
      for (int v = 0; v < nv; ++v) {
        oct.Src(v, ok, oj, oi) = rootbuf.h_view(v, n);
        if (!initflag)
          oct.U(v, ok, oj, oi) = rootbuf.h_view(v+nv, n);
      }
    }
  }

  Kokkos::deep_copy(src_r, src_h);
  if (!initflag)
    Kokkos::deep_copy(u_r, u_h);

  mgroot_->current_level_ = nrootlevel_ - 1;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromRootToBlocks(bool folddata)
//! \brief Transfer data from root/octets to block coarsest levels

void MultigridDriver::TransferFromRootToBlocks(bool folddata) {
  if (nreflevel_ > 0) {
    RestrictOctetsBeforeTransfer();
    SetOctetBoundariesBeforeTransfer(folddata);
  }
  mglevels_->SetFromRootGrid(folddata);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FMGProlongate(Driver *pdriver)
//! \brief FMG prolongation one level

void MultigridDriver::FMGProlongate(Driver *pdriver) {
  int ngh = mgroot_->ngh_;
  if (current_level_ == nrootlevel_ + nreflevel_ - 1) {
    MGRootBoundary(mgroot_->GetCurrentData());
    TransferFromRootToBlocks(false);
  }
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListFMGProlongate(ngh);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_fmg_prolongate", 0);
    current_level_++;
  } else if (current_level_ >= nrootlevel_ - 1) { // octets
    if (current_level_ == nrootlevel_ - 1)
      MGRootBoundary(mgroot_->GetCurrentData());
    else
      SetBoundariesOctets(true, false);
    FMGProlongateOctets();
    current_level_++;
  } else { // root grid
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->FMGProlongatePack();
    current_level_++;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::OneStepToFiner(int nsmooth)
//! \brief prolongation and smoothing one level

void MultigridDriver::OneStepToFiner(Driver *pdriver, int nsmooth) {
  int ngh = mgroot_->ngh_;
  if (current_level_ == nrootlevel_ + nreflevel_ - 1) {
    MGRootBoundary(mgroot_->GetCurrentData());
    TransferFromRootToBlocks(true);
  }
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListToFiner(nsmooth, ngh);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_to_finer", 0);
    current_level_++;
  } else if (current_level_ >= nrootlevel_ - 1) { // octets
    if (current_level_ == nrootlevel_ - 1)
      MGRootBoundary(mgroot_->GetCurrentData());
    else
      SetBoundariesOctets(true, true);
    ProlongateAndCorrectOctets();
    current_level_++;
    for (int n = 0; n < nsmooth; ++n) {
      SetBoundariesOctets(false, false);
      SmoothOctets(coffset_);
      SetBoundariesOctets(false, false);
      SmoothOctets(1 - coffset_);
    }
  } else { // root grid
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->ProlongateAndCorrectPack();
    current_level_++;
    for (int n = 0; n < nsmooth; ++n) {
      MGRootBoundary(mgroot_->GetCurrentData());
      mgroot_->SmoothPack(coffset_);
      MGRootBoundary(mgroot_->GetCurrentData());
      mgroot_->SmoothPack(1-coffset_);
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::OneStepToCoarser(int nsmooth)
//! \brief smoothing and restriction one level

void MultigridDriver::OneStepToCoarser(Driver *pdriver, int nsmooth) {
  int ngh = mgroot_->ngh_;
  if (current_level_ >= nrootlevel_ + nreflevel_) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListToCoarser(nsmooth, ngh);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_to_coarser", 0);
    if (current_level_ == nrootlevel_ + nreflevel_) {
      TransferFromBlocksToRoot(false);
    }
  } else if (current_level_ > nrootlevel_ - 1) { // octets
    SetBoundariesOctets(false, false);
    if (current_level_ < fmglevel_) {
      StoreOldDataOctets();
      CalculateFASRHSOctets();
    }
    for (int n = 0; n < nsmooth; ++n) {
      SmoothOctets(coffset_);
      SetBoundariesOctets(false, false);
      SmoothOctets(1 - coffset_);
      SetBoundariesOctets(false, false);
    }
    RestrictOctets();
  } else { // root grid
    MGRootBoundary(mgroot_->GetCurrentData());
    if (current_level_ < fmglevel_) {
      mgroot_->StoreOldData();
      mgroot_->CalculateFASRHSPack();
    }
    for (int n = 0; n < nsmooth; ++n) {
      mgroot_->SmoothPack(coffset_);
      MGRootBoundary(mgroot_->GetCurrentData());
      mgroot_->SmoothPack(1-coffset_);
      MGRootBoundary(mgroot_->GetCurrentData());
    }
    mgroot_->RestrictPack();
  }
  current_level_--;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveVCycle(int npresmooth, int npostsmooth)
//! \brief Solve the V-cycle starting from the current level

void MultigridDriver::SolveVCycle(Driver *pdriver, int npresmooth, int npostsmooth) {
  int startlevel=current_level_;
  coffset_ ^= 1;
  while (current_level_ > 0){
    OneStepToCoarser(pdriver, npresmooth);
  }
  SolveCoarsestGrid();
  while (current_level_ < startlevel)
  {
    OneStepToFiner(pdriver, npostsmooth);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveMG(Driver *pdriver)
//! \brief Multigrid (MG) solve using V-cycles

void MultigridDriver::SolveMG(Driver *pdriver) {
  if (eps_ >= 0.0) {
    SolveIterative(pdriver);
  } else {
    SolveIterativeFixedTimes(pdriver);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveFMG(Driver *pdriver)
//! \brief Full multigrid (FMG) solve using FAS and V-cycles

void MultigridDriver::SolveFMG(Driver *pdriver) {
  int cycles = std::max(1, fmg_ncycle_);
  SolveFMGCoarser();
  while (current_level_ < ntotallevel_ - 1) {
    fmglevel_ = current_level_ + 1;
    FMGProlongate(pdriver);
    fmglevel_ = current_level_;
    for (int n = 0; n < cycles; ++n) {
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    }
  }
  fmglevel_ = ntotallevel_ - 1;
  if (fsubtract_average_) {
    pmg = mglevels_;
    SubtractAverage(MGVariable::u);
  }
  SolveMG(pdriver);
  return;
}


void MultigridDriver::SolveFMGCoarser() {
  while (current_level_ >= nrootlevel_ + nreflevel_) {
    pmg = mglevels_;
    pmg->RestrictSourcePack();
    if (current_level_ == nrootlevel_ + nreflevel_) {
      TransferFromBlocksToRoot(true);
    }
    current_level_--;
  }
  if (nreflevel_ > 0) {
    RestrictFMGSourceOctets();
  }
  current_level_ = nrootlevel_ - 1;
  while (current_level_ > 0) {
    pmg = mgroot_;
    pmg->RestrictSourcePack();
    current_level_--;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveIterative(Driver *pdriver)
//! \brief Solve iteratively until defect norm drops below eps_

void MultigridDriver::SolveIterative(Driver *pdriver) {
  Real def = 0.0;
  for (int v = 0; v < nvar_; ++v) {
    def += CalculateDefectNorm(MGNormType::l2, v);
  }
  int n = 0;
  while (def > eps_) {
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    Real olddef = def;
    def = 0.0;
    for (int v = 0; v < nvar_; ++v) {
      def += CalculateDefectNorm(MGNormType::l2, v);
    }
    if (fshowdef_) {
      std::cout << "  MG iteration " << n << ": defect = " << def << std::endl; 
    }
    if (def/olddef > 0.9) {
      if (eps_ == 0.0) break;
      if (fshowdef_) {
        std::cout << "### WARNING in MultigridDriver::SolveIterative" << std::endl
                  << "Slow convergence: defect ratio = " << def/olddef << std::endl;
      }
    }
    ++n;
  }
  Kokkos::fence();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveIterativeFixedTimes(Driver *pdriver)
//! \brief Solve iteratively niter_ times (fixed count)

void MultigridDriver::SolveIterativeFixedTimes(Driver *pdriver) {
  for (int n = 0; n < niter_; ++n) {
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
  }
  Kokkos::fence();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveCoarsestGrid()
//! \brief Solve the coarsest root grid

void MultigridDriver::SolveCoarsestGrid() {
  pmg = mgroot_;
  int ni = (std::max(nrbx1_, std::max(nrbx2_, nrbx3_))
            >> (nrootlevel_-1));
  if (fsubtract_average_ && ni == 1) {
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->StoreOldData();
    mgroot_->ZeroClearData();
    return;
  }
  if (fsubtract_average_)
    SubtractAverage(MGVariable::src);
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  MGRootBoundary(mgroot_->GetCurrentData());
  mgroot_->StoreOldData();
  mgroot_->CalculateFASRHSPack();
  for (int i = 0; i < ni; ++i) {
    mgroot_->SmoothPack(coffset_);
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->SmoothPack(1-coffset_);
    MGRootBoundary(mgroot_->GetCurrentData());
  }
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real MultigridDriver::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the defect norm

Real MultigridDriver::CalculateDefectNorm(MGNormType nrm, int n) {
  Real norm = 0.0;
  if (mglevels_ != nullptr) {
    Real mg_norm = mglevels_->CalculateDefectNorm(nrm, n);
    if (nrm == MGNormType::max) {
      norm = std::max(norm, mg_norm);
    } else {
      norm += mg_norm;
    }
  }
  #if MPI_PARALLEL_ENABLED
  Real global_norm = 0.0;
  if (nrm == MGNormType::max) {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  } else {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
  norm = global_norm;
  #endif
  if (nrm != MGNormType::max) {
    Real vol = (mgroot_->size_.x1max - mgroot_->size_.x1min)
             * (mgroot_->size_.x2max - mgroot_->size_.x2min)
             * (mgroot_->size_.x3max - mgroot_->size_.x3min);
    norm /= vol;
  }
  if (nrm == MGNormType::l2) {
    norm = std::sqrt(norm);
  }
  return norm;
}


//========================================================================================
// Octet operations
//========================================================================================

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SmoothOctets(int color)
//! \brief Apply smoothing on all octets at current level

void MultigridDriver::SmoothOctets(int color) {
  int lev = current_level_ - nrootlevel_;
  for (int o = 0; o < noctets_[lev]; ++o) {
    SmoothOctet(octets_[lev][o], lev + 1, color);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::StoreOldDataOctets()
//! \brief Store u -> uold for all octets at current level

void MultigridDriver::StoreOldDataOctets() {
  int lev = current_level_ - nrootlevel_;
  for (int o = 0; o < noctets_[lev]; ++o) {
    octets_[lev][o].StoreOld();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CalculateFASRHSOctets()
//! \brief Calculate FAS RHS for all octets at current level

void MultigridDriver::CalculateFASRHSOctets() {
  int lev = current_level_ - nrootlevel_;
  for (int o = 0; o < noctets_[lev]; ++o) {
    CalculateFASRHSOctet(octets_[lev][o], lev + 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ZeroClearOctets()
//! \brief Zero clear u data in all octets up to current level

void MultigridDriver::ZeroClearOctets() {
  int maxlev = current_level_ - 1 - nrootlevel_;
  for (int l = 0; l <= maxlev && l < nreflevel_; ++l) {
    for (int o = 0; o < noctets_[l]; ++o)
      octets_[l][o].ZeroClearU();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RestrictFMGSourceOctets()
//! \brief Restrict the source through the octet hierarchy for FMG initialization

void MultigridDriver::RestrictFMGSourceOctets() {
  int ngh = mgroot_->ngh_;
  // Fine octets to coarser octets
  for (int l = nreflevel_ - 1; l >= 1; --l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &foct = octets_[l][o];
      const LogicalLocation &floc = foct.loc;
      LogicalLocation cloc;
      cloc.lx1 = (floc.lx1 >> 1);
      cloc.lx2 = (floc.lx2 >> 1);
      cloc.lx3 = (floc.lx3 >> 1);
      cloc.level = floc.level - 1;
      int oid = octetmap_[l-1][cloc];
      int oi = (static_cast<int>(floc.lx1) & 1) + ngh;
      int oj = (static_cast<int>(floc.lx2) & 1) + ngh;
      int ok = (static_cast<int>(floc.lx3) & 1) + ngh;
      MGOctet &coct = octets_[l-1][oid];
      for (int v = 0; v < nvar_; ++v)
        coct.Src(v, ok, oj, oi) = RestrictOneSrc(foct, v, ngh, ngh, ngh);
    }
  }
  // Octets at level 0 to root grid
  const auto &rsrc = mgroot_->GetCurrentSource();
  auto rsrc_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rsrc);
  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    const LogicalLocation &oloc = oct.loc;
    for (int v = 0; v < nvar_; ++v)
      rsrc_h(0, v, static_cast<int>(oloc.lx3)+ngh,
                    static_cast<int>(oloc.lx2)+ngh,
                    static_cast<int>(oloc.lx1)+ngh) =
        RestrictOneSrc(oct, v, ngh, ngh, ngh);
  }
  Kokkos::deep_copy(rsrc, rsrc_h);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RestrictOctets()
//! \brief Compute defect and restrict octets at current level to coarser level

void MultigridDriver::RestrictOctets() {
  int lev = current_level_ - nrootlevel_;
  int ngh = mgroot_->ngh_;

  if (lev >= 1) { // fine octets to coarser octets
    for (int o = 0; o < noctets_[lev]; ++o) {
      MGOctet &foct = octets_[lev][o];
      const LogicalLocation &floc = foct.loc;
      LogicalLocation cloc;
      cloc.lx1 = (floc.lx1 >> 1);
      cloc.lx2 = (floc.lx2 >> 1);
      cloc.lx3 = (floc.lx3 >> 1);
      cloc.level = floc.level - 1;
      int oid = octetmap_[lev-1][cloc];
      int oi = (static_cast<int>(floc.lx1) & 1) + ngh;
      int oj = (static_cast<int>(floc.lx2) & 1) + ngh;
      int ok = (static_cast<int>(floc.lx3) & 1) + ngh;
      MGOctet &coct = octets_[lev-1][oid];
      CalculateDefectOctet(foct, lev + 1);
      for (int v = 0; v < nvar_; ++v) {
        coct.Src(v, ok, oj, oi) = RestrictOneDef(foct, v, ngh, ngh, ngh);
        coct.U(v, ok, oj, oi) = RestrictOne(foct, v, ngh, ngh, ngh);
      }
    }
  } else { // octets to root grid
    const auto &rsrc = mgroot_->GetCurrentSource();
    const auto &ru = mgroot_->GetCurrentData();
    auto rsrc_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rsrc);
    auto ru_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ru);

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      const LogicalLocation &oloc = oct.loc;
      int ri = static_cast<int>(oloc.lx1);
      int rj = static_cast<int>(oloc.lx2);
      int rk = static_cast<int>(oloc.lx3);
      CalculateDefectOctet(oct, 1);
      for (int v = 0; v < nvar_; ++v) {
        rsrc_h(0, v, rk+ngh, rj+ngh, ri+ngh) = RestrictOneDef(oct, v, ngh, ngh, ngh);
        ru_h(0, v, rk+ngh, rj+ngh, ri+ngh) = RestrictOne(oct, v, ngh, ngh, ngh);
      }
    }
    Kokkos::deep_copy(rsrc, rsrc_h);
    Kokkos::deep_copy(ru, ru_h);
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ProlongateAndCorrectOctets()
//! \brief Prolongate and correct the potential in octets using tricubic interpolation
//! of the FAS correction (u - uold) from a 3x3x3 coarse neighborhood.

void MultigridDriver::ProlongateAndCorrectOctets() {
  int clev = current_level_ - nrootlevel_;
  int flev = clev + 1;
  int ngh = mgroot_->ngh_;
  constexpr Real w0[3] = {5.0, 30.0, -3.0};
  constexpr Real w1[3] = {-3.0, 30.0, 5.0};
  constexpr Real inv = 1.0 / 32768.0;

  if (flev == 0) { // from root to octets
    const auto &ru = mgroot_->GetCurrentData();
    const auto &ruold = mgroot_->GetCurrentOldData();
    auto ru_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ru);
    auto ruold_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ruold);

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      const LogicalLocation &oloc = oct.loc;
      int ri = static_cast<int>(oloc.lx1) + ngh;
      int rj = static_cast<int>(oloc.lx2) + ngh;
      int rk = static_cast<int>(oloc.lx3) + ngh;

      Real cbuf[3][3][3];
      for (int v = 0; v < nvar_; ++v) {
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              cbuf[kk+1][jj+1][ii+1] = ru_h(0, v, rk+kk, rj+jj, ri+ii)
                                      - ruold_h(0, v, rk+kk, rj+jj, ri+ii);
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                  for (int c = 0; c < 3; ++c)
                    sum += wk[a]*wj[b]*wi[c] * cbuf[a][b][c];
              oct.U(v, ngh+dk, ngh+dj, ngh+di) += sum * inv;
            }
          }
        }
      }
    }
  } else { // from coarser octets to finer octets
    for (int o = 0; o < noctets_[flev]; ++o) {
      MGOctet &foct = octets_[flev][o];
      const LogicalLocation &floc = foct.loc;
      LogicalLocation cloc;
      cloc.lx1 = (floc.lx1 >> 1);
      cloc.lx2 = (floc.lx2 >> 1);
      cloc.lx3 = (floc.lx3 >> 1);
      cloc.level = floc.level - 1;
      int cid = octetmap_[clev][cloc];
      MGOctet &coct = octets_[clev][cid];
      int ci = (static_cast<int>(floc.lx1) & 1) + ngh;
      int cj = (static_cast<int>(floc.lx2) & 1) + ngh;
      int ck = (static_cast<int>(floc.lx3) & 1) + ngh;

      Real cbuf[3][3][3];
      for (int v = 0; v < nvar_; ++v) {
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              cbuf[kk+1][jj+1][ii+1] = coct.U(v, ck+kk, cj+jj, ci+ii)
                                      - coct.Uold(v, ck+kk, cj+jj, ci+ii);
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                  for (int c = 0; c < 3; ++c)
                    sum += wk[a]*wj[b]*wi[c] * cbuf[a][b][c];
              foct.U(v, ngh+dk, ngh+dj, ngh+di) += sum * inv;
            }
          }
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FMGProlongateOctets()
//! \brief FMG prolongation for octets (tricubic interpolation from coarser level)

void MultigridDriver::FMGProlongateOctets() {
  int clev = current_level_ - nrootlevel_;
  int flev = clev + 1;
  int ngh = mgroot_->ngh_;
  constexpr Real w0[3] = {5.0, 30.0, -3.0};
  constexpr Real w1[3] = {-3.0, 30.0, 5.0};
  constexpr Real inv = 1.0 / 32768.0;

  if (flev == 0) { // from root to octets
    const auto &ru = mgroot_->GetCurrentData();
    auto ru_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ru);

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      const LogicalLocation &oloc = oct.loc;
      int ri = static_cast<int>(oloc.lx1) + ngh;
      int rj = static_cast<int>(oloc.lx2) + ngh;
      int rk = static_cast<int>(oloc.lx3) + ngh;
      for (int v = 0; v < nvar_; ++v) {
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int kk = -1; kk <= 1; ++kk)
                for (int jj = -1; jj <= 1; ++jj)
                  for (int ii = -1; ii <= 1; ++ii)
                    sum += wk[kk+1]*wj[jj+1]*wi[ii+1]
                           * ru_h(0, v, rk+kk, rj+jj, ri+ii);
              oct.U(v, ngh+dk, ngh+dj, ngh+di) = sum * inv;
            }
          }
        }
      }
    }
  } else { // from coarser octets to finer octets
    for (int o = 0; o < noctets_[flev]; ++o) {
      MGOctet &foct = octets_[flev][o];
      const LogicalLocation &floc = foct.loc;
      LogicalLocation cloc;
      cloc.lx1 = (floc.lx1 >> 1);
      cloc.lx2 = (floc.lx2 >> 1);
      cloc.lx3 = (floc.lx3 >> 1);
      cloc.level = floc.level - 1;
      int cid = octetmap_[clev][cloc];
      MGOctet &coct = octets_[clev][cid];
      int ci = (static_cast<int>(floc.lx1) & 1) + ngh;
      int cj = (static_cast<int>(floc.lx2) & 1) + ngh;
      int ck = (static_cast<int>(floc.lx3) & 1) + ngh;
      for (int v = 0; v < nvar_; ++v) {
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int kk = -1; kk <= 1; ++kk)
                for (int jj = -1; jj <= 1; ++jj)
                  for (int ii = -1; ii <= 1; ++ii)
                    sum += wk[kk+1]*wj[jj+1]*wi[ii+1]
                           * coct.U(v, ck+kk, cj+jj, ci+ii);
              foct.U(v, ngh+dk, ngh+dj, ngh+di) = sum * inv;
            }
          }
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetBoundariesOctets(bool fprolong, bool folddata)
//! \brief Apply boundary conditions for octets at current level
//! fprolong=true: skip leaf octets (used before prolongation)
//! folddata=true: also fill uold ghost cells

void MultigridDriver::SetBoundariesOctets(bool fprolong, bool folddata) {
  int lev = current_level_ - nrootlevel_;

  for (int o = 0; o < noctets_[lev]; ++o) {
    MGOctet &oct = octets_[lev][o];
    if (fprolong && oct.fleaf) continue;

    std::fill(ncoarse_.begin(), ncoarse_.end(), false);
    std::fill(cbuf_.begin(), cbuf_.end(), 0.0);
    std::fill(cbufold_.begin(), cbufold_.end(), 0.0);

    const LogicalLocation &loc = oct.loc;
    LogicalLocation nloc = loc;

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      nloc.lx3 = loc.lx3 + ox3;
      if (nloc.lx3 < 0) {
        nloc.lx3 = (nrbx3_ << lev) - 1;
      }
      if (nloc.lx3 >= (nrbx3_ << lev)) {
        nloc.lx3 = 0;
      }
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        nloc.lx2 = loc.lx2 + ox2;
        if (nloc.lx2 < 0) {
          nloc.lx2 = (nrbx2_ << lev) - 1;
        }
        if (nloc.lx2 >= (nrbx2_ << lev)) {
          nloc.lx2 = 0;
        }
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          nloc.lx1 = loc.lx1 + ox1;
          if (nloc.lx1 < 0) {
            nloc.lx1 = (nrbx1_ << lev) - 1;
          }
          if (nloc.lx1 >= (nrbx1_ << lev)) {
            nloc.lx1 = 0;
          }
          if (octetmap_[lev].count(nloc) == 1) { // same level neighbor
            int nid = octetmap_[lev][nloc];
            MGOctet &noct = octets_[lev][nid];
            SetOctetBoundarySameLevel(oct, noct, cbuf_, cbufold_,
                                      nvar_, ox1, ox2, ox3, folddata);
          } else if (!fprolong) { // coarser level
            ncoarse_[(ox3+1)*9 + (ox2+1)*3 + (ox1+1)] = true;
            if (lev > 0) { // from octet
              LogicalLocation cloc;
              cloc.lx1 = nloc.lx1 >> 1;
              cloc.lx2 = nloc.lx2 >> 1;
              cloc.lx3 = nloc.lx3 >> 1;
              cloc.level = nloc.level - 1;
              int cid = octetmap_[lev-1][cloc];
              MGOctet &coct = octets_[lev-1][cid];
              SetOctetBoundaryFromCoarser(coct.u, coct.uold, cbuf_, cbufold_,
                                          nvar_, coct.nc, loc, ox1, ox2, ox3, folddata);
            } else { // from root
              // Get root data on host
              const auto &ru = mgroot_->GetCurrentData();
              const auto &ruold = mgroot_->GetCurrentOldData();
              auto ru_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ru);
              auto ruold_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ruold);
              // Convert to flat vector for SetOctetBoundaryFromCoarser
              int rnx = ru_h.extent_int(4);
              int rny = ru_h.extent_int(3);
              int rnz = ru_h.extent_int(2);
              int rnv = ru_h.extent_int(1);
              std::vector<Real> rbuf(rnv*rnz*rny*rnx);
              std::vector<Real> rbufo(rnv*rnz*rny*rnx);
              for (int v = 0; v < rnv; ++v)
                for (int k = 0; k < rnz; ++k)
                  for (int j = 0; j < rny; ++j)
                    for (int i = 0; i < rnx; ++i) {
                      rbuf[((v*rnz+k)*rny+j)*rnx+i] = ru_h(0,v,k,j,i);
                      rbufo[((v*rnz+k)*rny+j)*rnx+i] = ruold_h(0,v,k,j,i);
                    }
              SetOctetBoundaryFromCoarser(rbuf, rbufo, cbuf_, cbufold_,
                                          nvar_, rnx, nloc, ox1, ox2, ox3, folddata);
            }
          }
        }
      }
    }

    // Prolongate coarse buffer to fill ghost cells at coarser boundaries
    if (!fprolong) {
      ProlongateOctetBoundaries(oct, cbuf_, cbufold_, nvar_, ncoarse_, folddata);
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetBoundarySameLevel(...)
//! \brief Set octet boundary from a neighbor on the same level

void MultigridDriver::SetOctetBoundarySameLevel(MGOctet &dst, const MGOctet &src_oct,
     std::vector<Real> &cbuf, std::vector<Real> &cbufold,
     int nvar, int ox1, int ox2, int ox3, bool folddata) {
  const int ngh = mgroot_->ngh_;
  constexpr Real fac = 0.125;
  const int l = ngh, r = ngh + 1;
  int is, ie, js, je, ks, ke, nis, njs, nks;
  if (ox1 == 0)     is = ngh,   ie = ngh+1, nis = ngh;
  else if (ox1 < 0) is = 0,     ie = ngh-1, nis = ngh+1;
  else              is = ngh+2, ie = ngh+2, nis = ngh;
  if (ox2 == 0)     js = ngh,   je = ngh+1, njs = ngh;
  else if (ox2 < 0) js = 0,     je = ngh-1, njs = ngh+1;
  else              js = ngh+2, je = ngh+2, njs = ngh;
  if (ox3 == 0)     ks = ngh,   ke = ngh+1, nks = ngh;
  else if (ox3 < 0) ks = 0,     ke = ngh-1, nks = ngh+1;
  else              ks = ngh+2, ke = ngh+2, nks = ngh;
  int ci = ox1 + 1, cj = ox2 + 1, ck = ox3 + 1;

  for (int v = 0; v < nvar; ++v) {
    for (int k = ks, nk = nks; k <= ke; ++k, ++nk) {
      for (int j = js, nj = njs; j <= je; ++j, ++nj) {
        for (int i = is, ni = nis; i <= ie; ++i, ++ni)
          dst.U(v, k, j, i) = src_oct.U(v, nk, nj, ni);
      }
    }
  }
  for (int v = 0; v < nvar; ++v)
    BufRef(cbuf, 3, v, ck, cj, ci) = fac*(src_oct.U(v,l,l,l) + src_oct.U(v,l,l,r)
        + src_oct.U(v,l,r,l) + src_oct.U(v,r,l,l)
        + src_oct.U(v,r,r,l) + src_oct.U(v,r,l,r)
        + src_oct.U(v,l,r,r) + src_oct.U(v,r,r,r));

  if (folddata) {
    for (int v = 0; v < nvar; ++v) {
      for (int k = ks, nk = nks; k <= ke; ++k, ++nk) {
        for (int j = js, nj = njs; j <= je; ++j, ++nj) {
          for (int i = is, ni = nis; i <= ie; ++i, ++ni)
            dst.Uold(v, k, j, i) = src_oct.Uold(v, nk, nj, ni);
        }
      }
    }
    for (int v = 0; v < nvar; ++v)
      BufRef(cbufold, 3, v, ck, cj, ci) = fac*
        (src_oct.Uold(v,l,l,l)+src_oct.Uold(v,l,l,r) + src_oct.Uold(v,l,r,l)
        +src_oct.Uold(v,r,l,l)+src_oct.Uold(v,r,r,l) + src_oct.Uold(v,r,l,r)
        +src_oct.Uold(v,l,r,r)+src_oct.Uold(v,r,r,r));
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetBoundaryFromCoarser(...)
//! \brief Fill coarse buffer entry from a coarser neighbor (octet or root)

void MultigridDriver::SetOctetBoundaryFromCoarser(const std::vector<Real> &un,
     const std::vector<Real> &unold,
     std::vector<Real> &cbuf, std::vector<Real> &cbufold,
     int nvar, int un_nc, const LogicalLocation &loc,
     int ox1, int ox2, int ox3, bool folddata) {
  int ngh = mgroot_->ngh_;
  int ci, cj, ck;
  if (loc.level == locrootlevel_) { // from root
    ci = static_cast<int>(loc.lx1) + ngh;
    cj = static_cast<int>(loc.lx2) + ngh;
    ck = static_cast<int>(loc.lx3) + ngh;
  } else { // from a neighbor octet (given loc is MY location)
    int ix1 = (static_cast<int>(loc.lx1) & 1);
    int ix2 = (static_cast<int>(loc.lx2) & 1);
    int ix3 = (static_cast<int>(loc.lx3) & 1);
    if (ox1 == 0) ci = ix1 + ngh;
    else          ci = (ix1^1) + ngh;
    if (ox2 == 0) cj = ix2 + ngh;
    else          cj = (ix2^1) + ngh;
    if (ox3 == 0) ck = ix3 + ngh;
    else          ck = (ix3^1) + ngh;
  }
  int i = 1 + ox1, j = 1 + ox2, k = 1 + ox3;
  for (int v = 0; v < nvar; ++v)
    BufRef(cbuf, 3, v, k, j, i) = un[((v*un_nc + ck)*un_nc + cj)*un_nc + ci];
  if (folddata) {
    for (int v = 0; v < nvar; ++v)
      BufRef(cbufold, 3, v, k, j, i) = unold[((v*un_nc + ck)*un_nc + cj)*un_nc + ci];
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ProlongateOctetBoundaries(...)
//! \brief Prolongate coarse buffer to fill ghost cells at coarser boundaries

void MultigridDriver::ProlongateOctetBoundaries(MGOctet &oct,
     std::vector<Real> &cbuf, std::vector<Real> &cbufold,
     int nvar, const std::vector<bool> &ncoarse, bool folddata) {
  const int ngh = mgroot_->ngh_;
  const int nc = oct.nc;
  const int flim = nc;
  constexpr Real fac = 0.125;
  const int l = ngh, r = ngh + 1;

  // Fill center of coarse buffer from this octet's own data (center = 1,1,1)
  for (int v = 0; v < nvar; ++v)
    BufRef(cbuf, 3, v, 1, 1, 1) = fac*(oct.U(v,l,l,l)+oct.U(v,l,l,r)
        +oct.U(v,l,r,l)+oct.U(v,r,l,l)
        +oct.U(v,r,r,l)+oct.U(v,r,l,r)+oct.U(v,l,r,r)+oct.U(v,r,r,r));
  if (folddata) {
    for (int v = 0; v < nvar; ++v)
      BufRef(cbufold, 3, v, 1, 1, 1) = fac*(oct.Uold(v,l,l,l)+oct.Uold(v,l,l,r)
          +oct.Uold(v,l,r,l)+oct.Uold(v,r,l,l)
          +oct.Uold(v,r,r,l)+oct.Uold(v,r,l,r)+oct.Uold(v,l,r,r)+oct.Uold(v,r,r,r));
  }

  // Prolongate from coarse buffer to fine ghost cells where neighbor is coarser
  for (int ox3 = -1; ox3 <= 1; ++ox3) {
    for (int ox2 = -1; ox2 <= 1; ++ox2) {
      for (int ox1 = -1; ox1 <= 1; ++ox1) {
        if (ncoarse[(ox3+1)*9 + (ox2+1)*3 + (ox1+1)]) {
          int ci = ox1 + 1, cj = ox2 + 1, ck = ox3 + 1;
          int fi = ox1*2 + ngh, fj = ox2*2 + ngh, fk = ox3*2 + ngh;
          for (int v = 0; v < nvar; ++v) {
            auto cb = [&](int vv, int kk, int jj, int ii) -> Real {
              kk = std::max(0, std::min(2, kk));
              jj = std::max(0, std::min(2, jj));
              ii = std::max(0, std::min(2, ii));
              return BufRef(cbuf, 3, vv, kk, jj, ii);
            };
            if (fk >= 0 && fj >= 0 && fi >= 0)
              oct.U(v,fk,fj,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj-1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj-1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj-1,ci)+cb(v,ck-1,cj,ci-1)+cb(v,ck,cj-1,ci-1)));
            if (fk >= 0 && fj >= 0 && fi+1 < flim)
              oct.U(v,fk,fj,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj-1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj-1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj-1,ci)+cb(v,ck-1,cj,ci+1)+cb(v,ck,cj-1,ci+1)));
            if (fk >= 0 && fj+1 < flim && fi >= 0)
              oct.U(v,fk,fj+1,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj+1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj+1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj+1,ci)+cb(v,ck-1,cj,ci-1)+cb(v,ck,cj+1,ci-1)));
            if (fk+1 < flim && fj >= 0 && fi >= 0)
              oct.U(v,fk+1,fj,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj-1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj-1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj-1,ci)+cb(v,ck+1,cj,ci-1)+cb(v,ck,cj-1,ci-1)));
            if (fk+1 < flim && fj+1 < flim && fi >= 0)
              oct.U(v,fk+1,fj+1,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj+1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj+1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj+1,ci)+cb(v,ck+1,cj,ci-1)+cb(v,ck,cj+1,ci-1)));
            if (fk+1 < flim && fj >= 0 && fi+1 < flim)
              oct.U(v,fk+1,fj,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj-1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj-1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj-1,ci)+cb(v,ck+1,cj,ci+1)+cb(v,ck,cj-1,ci+1)));
            if (fk >= 0 && fj+1 < flim && fi+1 < flim)
              oct.U(v,fk,fj+1,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj+1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj+1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj+1,ci)+cb(v,ck-1,cj,ci+1)+cb(v,ck,cj+1,ci+1)));
            if (fk+1 < flim && fj+1 < flim && fi+1 < flim)
              oct.U(v,fk+1,fj+1,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj+1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj+1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj+1,ci)+cb(v,ck+1,cj,ci+1)+cb(v,ck,cj+1,ci+1)));
          }
          if (folddata) {
            for (int v = 0; v < nvar; ++v) {
              auto co = [&](int vv, int kk, int jj, int ii) -> Real {
                kk = std::max(0, std::min(2, kk));
                jj = std::max(0, std::min(2, jj));
                ii = std::max(0, std::min(2, ii));
                return BufRef(cbufold, 3, vv, kk, jj, ii);
              };
              if (fk >= 0 && fj >= 0 && fi >= 0)
                oct.Uold(v,fk,fj,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj-1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj-1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj-1,ci)+co(v,ck-1,cj,ci-1)+co(v,ck,cj-1,ci-1)));
              if (fk >= 0 && fj >= 0 && fi+1 < flim)
                oct.Uold(v,fk,fj,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj-1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj-1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj-1,ci)+co(v,ck-1,cj,ci+1)+co(v,ck,cj-1,ci+1)));
              // remaining 6 corners for uold (abbreviated for brevity - same pattern)
              if (fk >= 0 && fj+1 < flim && fi >= 0)
                oct.Uold(v,fk,fj+1,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj+1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj+1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj+1,ci)+co(v,ck-1,cj,ci-1)+co(v,ck,cj+1,ci-1)));
              if (fk+1 < flim && fj >= 0 && fi >= 0)
                oct.Uold(v,fk+1,fj,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj-1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj-1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj-1,ci)+co(v,ck+1,cj,ci-1)+co(v,ck,cj-1,ci-1)));
              if (fk+1 < flim && fj+1 < flim && fi >= 0)
                oct.Uold(v,fk+1,fj+1,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj+1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj+1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj+1,ci)+co(v,ck+1,cj,ci-1)+co(v,ck,cj+1,ci-1)));
              if (fk+1 < flim && fj >= 0 && fi+1 < flim)
                oct.Uold(v,fk+1,fj,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj-1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj-1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj-1,ci)+co(v,ck+1,cj,ci+1)+co(v,ck,cj-1,ci+1)));
              if (fk >= 0 && fj+1 < flim && fi+1 < flim)
                oct.Uold(v,fk,fj+1,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj+1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj+1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj+1,ci)+co(v,ck-1,cj,ci+1)+co(v,ck,cj+1,ci+1)));
              if (fk+1 < flim && fj+1 < flim && fi+1 < flim)
                oct.Uold(v,fk+1,fj+1,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj+1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj+1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj+1,ci)+co(v,ck+1,cj,ci+1)+co(v,ck,cj+1,ci+1)));
            }
          }
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RestrictOctetsBeforeTransfer()
//! \brief Restrict all octets to prepare for root-to-blocks transfer

void MultigridDriver::RestrictOctetsBeforeTransfer() {
  const int ngh = mgroot_->ngh_;
  for (int l = nreflevel_ - 1; l >= 1; --l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &foct = octets_[l][o];
      const LogicalLocation &floc = foct.loc;
      LogicalLocation cloc;
      cloc.lx1 = (floc.lx1 >> 1);
      cloc.lx2 = (floc.lx2 >> 1);
      cloc.lx3 = (floc.lx3 >> 1);
      cloc.level = floc.level - 1;
      int oid = octetmap_[l-1][cloc];
      MGOctet &coct = octets_[l-1][oid];
      int oi = (static_cast<int>(floc.lx1) & 1) + ngh;
      int oj = (static_cast<int>(floc.lx2) & 1) + ngh;
      int ok = (static_cast<int>(floc.lx3) & 1) + ngh;
      for (int v = 0; v < nvar_; ++v)
        coct.U(v, ok, oj, oi) = RestrictOne(foct, v, ngh, ngh, ngh);
    }
  }
  // Octets to root grid
  const auto &ru = mgroot_->GetCurrentData();
  auto ru_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ru);
  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    const LogicalLocation &oloc = oct.loc;
    for (int v = 0; v < nvar_; ++v)
      ru_h(0, v, static_cast<int>(oloc.lx3)+ngh,
                 static_cast<int>(oloc.lx2)+ngh,
                 static_cast<int>(oloc.lx1)+ngh) = RestrictOne(oct, v, ngh, ngh, ngh);
  }
  Kokkos::deep_copy(ru, ru_h);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetBoundariesBeforeTransfer(bool folddata)
//! \brief Set octet boundaries before transfer from root to blocks

void MultigridDriver::SetOctetBoundariesBeforeTransfer(bool folddata) {
  // Clear octet boundary flags
  for (int l = 0; l < nreflevel_; ++l) {
    for (int o = 0; o < noctets_[l]; ++o)
      octetbflag_[l][o] = false;
  }

  int padding = nslist_[global_variable::my_rank];

  // For each refined block, find its parent octet and set boundaries
  for (int m = 0; m < mglevels_->nmmb_; ++m) {
    LogicalLocation loc = pmy_mesh_->lloc_eachmb[m + padding];
    if (loc.level == locrootlevel_) continue;

    // Parent octet
    LogicalLocation oloc;
    oloc.lx1 = loc.lx1 >> 1;
    oloc.lx2 = loc.lx2 >> 1;
    oloc.lx3 = loc.lx3 >> 1;
    oloc.level = loc.level - 1;
    int lev = oloc.level - locrootlevel_;
    int oid = octetmap_[lev][oloc];
    if (octetbflag_[lev][oid]) continue;
    octetbflag_[lev][oid] = true;

    MGOctet &oct = octets_[lev][oid];
    std::fill(ncoarse_.begin(), ncoarse_.end(), false);
    std::fill(cbuf_.begin(), cbuf_.end(), 0.0);
    std::fill(cbufold_.begin(), cbufold_.end(), 0.0);

    LogicalLocation nloc = oloc;
    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      nloc.lx3 = oloc.lx3 + ox3;
      if (nloc.lx3 < 0) nloc.lx3 = (nrbx3_ << lev) - 1;
      if (nloc.lx3 >= (nrbx3_ << lev)) nloc.lx3 = 0;
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        nloc.lx2 = oloc.lx2 + ox2;
        if (nloc.lx2 < 0) nloc.lx2 = (nrbx2_ << lev) - 1;
        if (nloc.lx2 >= (nrbx2_ << lev)) nloc.lx2 = 0;
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          nloc.lx1 = oloc.lx1 + ox1;
          if (nloc.lx1 < 0) nloc.lx1 = (nrbx1_ << lev) - 1;
          if (nloc.lx1 >= (nrbx1_ << lev)) nloc.lx1 = 0;
          if (octetmap_[lev].count(nloc) == 1) {
            int nid = octetmap_[lev][nloc];
            MGOctet &noct = octets_[lev][nid];
            SetOctetBoundarySameLevel(oct, noct, cbuf_, cbufold_,
                                      nvar_, ox1, ox2, ox3, folddata);
          } else {
            ncoarse_[(ox3+1)*9 + (ox2+1)*3 + (ox1+1)] = true;
            if (lev > 0) {
              LogicalLocation cloc;
              cloc.lx1 = nloc.lx1 >> 1;
              cloc.lx2 = nloc.lx2 >> 1;
              cloc.lx3 = nloc.lx3 >> 1;
              cloc.level = nloc.level - 1;
              int cid = octetmap_[lev-1][cloc];
              MGOctet &coct = octets_[lev-1][cid];
              SetOctetBoundaryFromCoarser(coct.u, coct.uold, cbuf_, cbufold_,
                                          nvar_, coct.nc, oloc, ox1, ox2, ox3, folddata);
            } else {
              const auto &ru = mgroot_->GetCurrentData();
              const auto &ruold = mgroot_->GetCurrentOldData();
              auto ru_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ru);
              auto ruold_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ruold);
              int rnx = ru_h.extent_int(4);
              int rny = ru_h.extent_int(3);
              int rnz = ru_h.extent_int(2);
              int rnv = ru_h.extent_int(1);
              std::vector<Real> rbuf(rnv*rnz*rny*rnx);
              std::vector<Real> rbufo(rnv*rnz*rny*rnx);
              for (int v = 0; v < rnv; ++v)
                for (int k = 0; k < rnz; ++k)
                  for (int j = 0; j < rny; ++j)
                    for (int i = 0; i < rnx; ++i) {
                      rbuf[((v*rnz+k)*rny+j)*rnx+i] = ru_h(0,v,k,j,i);
                      rbufo[((v*rnz+k)*rny+j)*rnx+i] = ruold_h(0,v,k,j,i);
                    }
              SetOctetBoundaryFromCoarser(rbuf, rbufo, cbuf_, cbufold_,
                                          nvar_, rnx, nloc, ox1, ox2, ox3, folddata);
            }
          }
        }
      }
    }

    if (folddata)
      ProlongateOctetBoundaries(oct, cbuf_, cbufold_, nvar_, ncoarse_, folddata);
    else
      ProlongateOctetBoundaries(oct, cbuf_, cbufold_, nvar_, ncoarse_, false);
  }
}


void MultigridDriver::MGRootBoundary(const DvceArray5D<Real> &u) {
  int nvar = u.extent_int(1);
  int current_level = mgroot_->GetCurrentLevel();
  int nlevels = mgroot_->GetNumberOfLevels();
  int ngh = mgroot_->ngh_;
  
  int ll = nlevels - 1 - current_level;
  int nx = (mgroot_->indcs_.nx1 >> ll) + 2*ngh;
  int ny = (mgroot_->indcs_.nx2 >> ll) + 2*ngh;
  int nz = (mgroot_->indcs_.nx3 >> ll) + 2*ngh;
  
  // Root grid is single meshblock (m=0)
  int m = 0;
  
  // Apply periodic boundary conditions directly
  par_for("MG::PackAndSendMGRoot_x", DevExeSpace(),
          0, nvar-1, 0, nz-1, 0, ny-1,
  KOKKOS_LAMBDA(const int v, const int k, const int j) {
    for (int n = 0; n < ngh; ++n) {
      u(m, v, k, j, n) = u(m, v, k, j, nx - 2*ngh + n);
      u(m, v, k, j, nx - ngh + n) = u(m, v, k, j, ngh + n);
    }
  });
  
  par_for("MG::PackAndSendMGRoot_y", DevExeSpace(),
          0, nvar-1, 0, nz-1, 0, nx-1,
  KOKKOS_LAMBDA(const int v, const int k, const int i) {
    for (int n = 0; n < ngh; ++n) {
      u(m, v, k, n, i) = u(m, v, k, ny - 2*ngh + n, i);
      u(m, v, k, ny - ngh + n, i) = u(m, v, k, ngh + n, i);
    }
  });
  
  par_for("MG::PackAndSendMGRoot_z", DevExeSpace(),
          0, nvar-1, 0, ny-1, 0, nx-1,
  KOKKOS_LAMBDA(const int v, const int j, const int i) {
    for (int n = 0; n < ngh; ++n) {
      u(m, v, n, j, i) = u(m, v, nz - 2*ngh + n, j, i);
      u(m, v, nz - ngh + n, j, i) = u(m, v, ngh + n, j, i);
    }
  });
}
