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
#include <cmath>
#include <cstdlib>    // abs
#include <iomanip>    // setprecision
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
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
    nb_rank_(0), ncoeff_(0) {
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
  
  // Initialize octet arrays for SMR
  noct_ = maxreflevel_;
  if (noct_ > 0) {
    mgoct_ = new Multigrid*[noct_];
    for (int i = 0; i < noct_; ++i) {
      mgoct_[i] = nullptr;
    }
  } else {
    mgoct_ = nullptr;
  }

//#if MPI_PARALLEL_ENABLED
//  MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_MULTIGRID);
//  mg_phys_id_ = pmy_mesh_->ReserveTagPhysIDs(1);
//#endif

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
  //delete [] rootbuf_;
  
  // Clean up octets
  if (mgoct_ != nullptr) {
    for (int i = 0; i < noct_; ++i) {
      if (mgoct_[i] != nullptr) delete mgoct_[i];
    }
    delete [] mgoct_;
  }
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
    std::cout << "MultigridDriver::SetupMultigrid: Number of refinement levels = "
              << nreflevel_ << std::endl;
  }
  
  // For uniform meshes (nreflevel_=0): ntotallevel = nrootlevel + nmblevel - 1
  // For SMR (nreflevel_>0): add refinement levels to the total
  ntotallevel_ = nrootlevel_ + nmblevel_ - 1 + nreflevel_;
  os_ = mgroot_->ngh_;
  oe_ = os_+1;
  
  // Initialize octet levels for SMR (one per refinement level)
  // Note: the level of an Octet is one level lower than the data stored there
  if (nreflevel_ > 0 && mgoct_ != nullptr) {
    // Create octet multigrids for intermediate refinement levels
    // Each octet handles the transition between root grid and refined blocks
    // For now, we'll create them on demand in TransferFromRootToBlocks
  }
  
  if (needinit_) {
  // assume the same parallelization as hydro
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
  }
  needinit_ = false;
  //TODO: Apply mask to source if needed
  if (fsubtract_average_) {
    pmg = mglevels_;
    SubtractAverage(MGVariable::src);
  }
  current_level_ = ntotallevel_ - 1;
  fmglevel_ = current_level_;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromBlocksToRoot(bool initflag)
//! \brief collect the coarsest data and transfer to the root grid

void MultigridDriver::TransferFromBlocksToRoot(bool initflag) {
  const int nv = nvar_;
  auto rootbuf = rootbuf_;
  const auto &src_ =  mglevels_->src_[0];
  const auto &u_ =  mglevels_->u_[0];
  const int ngh_ = mglevels_->ngh_;
  // Gather data from meshblock-level multigrids into rootbuf_
  // mglevels_ is a pack-aware multigrid covering nmmb_ meshblocks
  int nmmb = mglevels_->nmmb_-1;
  int padding = nslist_[global_variable::my_rank];
  par_for("Multigrid:SaveToRoot",DevExeSpace(), 0, nmmb, KOKKOS_LAMBDA(const int m) {  
    // Transfer source and solution data (always)
      for (int v = 0; v < nv; ++v) {
        rootbuf.d_view(v   , m+padding) = src_(m, v, ngh_, ngh_, ngh_);
        rootbuf.d_view(v+nv, m+padding) = u_(m, v, ngh_, ngh_, ngh_);
      }
    });
  rootbuf.template modify<DevExeSpace>();
  rootbuf.template sync<HostExeSpace>();
  #if MPI_PARALLEL_ENABLED
  //TODO: Optimize MPI communication (make rootbuf_ contiguous in memory)
  for (int v = 0; v < 2*nv; ++v) {
    MPI_Allgatherv(MPI_IN_PLACE, nblist_[global_variable::my_rank], MPI_ATHENA_REAL,
                     &rootbuf.h_view(v,0), nblist_, nslist_, MPI_ATHENA_REAL, MPI_COMM_WORLD);
                    }
  #endif
  const auto loc = pmy_mesh_->lloc_eachmb;
  int rootlevel = locrootlevel_;
  const auto &src_r = mgroot_->GetCurrentSource();
  const auto &u_r = mgroot_->GetCurrentData();
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_r);
  auto src_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), src_r);
  
  // For SMR: handle blocks at different refinement levels
  for (int m = 0; m < nbtotal_; ++m) {
    int lev = loc[m].level - rootlevel;
    int i = static_cast<int>(loc[m].lx1);
    int j = static_cast<int>(loc[m].lx2);
    int k = static_cast<int>(loc[m].lx3);
    
    if (lev == 0) {
      // Direct copy for root level blocks
      for (int v = 0; v < nv; ++v){
        src_h(0, v, k+ngh_, j+ngh_, i+ngh_) = rootbuf.h_view(v, m);
        u_h(0, v, k+ngh_, j+ngh_, i+ngh_) = rootbuf.h_view(v+nv, m);
      }
    } else if (lev > 0) {
      // Refined block: restrict to root level
      // For now, just use the block's coarsest value
      // TODO: implement proper restriction with averaging
      Real rfac = 1.0 / (1 << lev);
      int ri = static_cast<int>(i * rfac);
      int rj = static_cast<int>(j * rfac);
      int rk = static_cast<int>(k * rfac);
      for (int v = 0; v < nv; ++v){
        src_h(0, v, rk+ngh_, rj+ngh_, ri+ngh_) = rootbuf.h_view(v, m);
        u_h(0, v, rk+ngh_, rj+ngh_, ri+ngh_) = rootbuf.h_view(v+nv, m);
      }
    }
  }
  Kokkos::deep_copy(src_r, src_h);
  Kokkos::deep_copy(u_r, u_h);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromRootToBlocks(bool folddata)
//! \brief Transfer the data from the root grid to the coarsest level of each MeshBlock

void MultigridDriver::TransferFromRootToBlocks(bool folddata) {
  // For SMR: first transfer from root grid to octets, then octets to blocks
  if (nreflevel_ > 0 && mgoct_ != nullptr) {
    // Create octets on demand and populate them from root grid
    // For now, we skip explicit octet creation and let SetFromRootGrid
    // handle the transfer directly, with octet support built-in
    // Future: implement explicit TransferFromRootToOctets() here
  }
  
  // Transfer to meshblock levels (handles octets internally if available)
  mglevels_->SetFromRootGrid(folddata);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FMGProlongate(Driver *pdriver)
//! \brief FMG prolongation one level (direct overwrite, no smoothing)
//! Following Athena++, this is a standalone function that handles
//! root grid, octets, and MeshBlock levels for FMG prolongation.

void MultigridDriver::FMGProlongate(Driver *pdriver) {
  int ngh = mgroot_->ngh_;
  if (current_level_ == nrootlevel_ - 1) {
    // Transition from root grid to MeshBlocks
    MGRootBoundary(mgroot_->GetCurrentData());
    TransferFromRootToBlocks(false);
  }
  if (current_level_ >= nrootlevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListFMGProlongate(ngh);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_fmg_prolongate", 0);
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
  int ngh=mgroot_->ngh_;
  if (current_level_ == nrootlevel_ - 1) {
    MGRootBoundary(mgroot_->GetCurrentData());
    TransferFromRootToBlocks(true);
  }
  if (current_level_ >= nrootlevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListToFiner(nsmooth, ngh);
    pdriver->ExecuteTaskList(pmy_mesh_,"mg_to_finer",0);
    current_level_++;
  } 
  else { // root grid
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->ProlongateAndCorrectPack();
    current_level_++;
    for (int n = 0; n < nsmooth; ++n) {
      //RED
      MGRootBoundary(mgroot_->GetCurrentData());
      mgroot_->SmoothPack(coffset_);
      //BLACK
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
  int ngh=mgroot_->ngh_;
  if (current_level_ >= nrootlevel_) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListToCoarser(nsmooth, ngh);
    pdriver->ExecuteTaskList(pmy_mesh_,"mg_to_coarser",0);
    if (current_level_ == nrootlevel_ + nreflevel_) {
      TransferFromBlocksToRoot(false);
    }
  }
  else { // uniform root grid
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->StoreOldData();
    mgroot_->CalculateFASRHSPack();
    for (int n = 0; n < nsmooth; ++n) {
      //RED
      mgroot_->SmoothPack(coffset_);
      MGRootBoundary(mgroot_->GetCurrentData());
      //BLACK
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
    //std::cout << "SolveVCycle(coarser):" << current_level_ <<"->" << current_level_-1 << std::endl;
    OneStepToCoarser(pdriver, npresmooth);
  }
  SolveCoarsestGrid();
  while (current_level_ < startlevel)
  {
    //std::cout << "SolveVCycle(finer):" << current_level_ <<"->" << current_level_+1 << std::endl;
    OneStepToFiner(pdriver, npostsmooth);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveMG(Driver *pdriver)
//! \brief Multigrid (MG) solve using V-cycles
//! \brief Solve the V-cycle starting from the current level

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
//! Performs the FMG sweep (coarsest to finest) with fmg_ncycle_ V-cycles per level,
//! then polishes with iterative V-cycles (convergence-based or fixed-count).

void MultigridDriver::SolveFMG(Driver *pdriver) {
  int cycles = std::max(1, fmg_ncycle_);
  SolveFMGCoarser();
  // Prolongate one level at a time and apply V-cycles
  while (current_level_ < ntotallevel_ - 1) {
    fmglevel_ = current_level_ + 1;
    // Prolongate solution.
    FMGProlongate(pdriver);
    fmglevel_ = current_level_;
    // Solve V-cycles at this FMG level
    for (int n = 0; n < cycles; ++n) {
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    }
  }
  fmglevel_ = ntotallevel_ - 1;
  if (fsubtract_average_) {
    pmg = mglevels_;
    SubtractAverage(MGVariable::u);
  }
  // Polish with additional V-cycles after the FMG sweep
  SolveMG(pdriver);
  return;
}

void MultigridDriver::SolveFMGCoarser() {
  while (current_level_ > 0){
    //std::cout << "SolveFMG(coarser):" << current_level_ <<"->" << current_level_-1 << std::endl;
    if (current_level_ >= nrootlevel_) { // MeshBlocks
      pmg = mglevels_;
      pmg->RestrictSourcePack(); // restrict the source to the coarsest level
      if (current_level_ == nrootlevel_ + nreflevel_) {
        TransferFromBlocksToRoot(true); // transfer the restricted source to the root grid  
      }
    }
    else { // root grid
      pmg = mgroot_;
      pmg->RestrictSourcePack(); // restrict the source to the root grid
    }
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
  if (fsubtract_average_ && ni == 1) { // trivial case - all zero
    MGRootBoundary(mgroot_->GetCurrentData());
    mgroot_->StoreOldData();
    mgroot_->ZeroClearData();
    return;
  }
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  MGRootBoundary(mgroot_->GetCurrentData());
  mgroot_->StoreOldData();
  mgroot_->CalculateFASRHSPack();
  for (int i = 0; i < ni; ++i) { // iterate ni times
    //RED
    mgroot_->SmoothPack(coffset_);
    MGRootBoundary(mgroot_->GetCurrentData());
    //BLACK
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

  // Compute defect norm from meshblock-level multigrids (finest levels)
  if (mglevels_ != nullptr) {
    Real mg_norm = mglevels_->CalculateDefectNorm(nrm, n);
    if (nrm == MGNormType::max) {
      norm = std::max(norm, mg_norm);
    } else {
      norm += mg_norm;
    }
  }

  #if MPI_PARALLEL_ENABLED
  // Reduce over all MPI ranks
  Real global_norm = 0.0;
  if (nrm == MGNormType::max) {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  } else {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
  norm = global_norm;
  #endif
  // Normalize by volume for L2 norm
  if (nrm != MGNormType::max) {
    Real vol = (mgroot_->size_.x1max - mgroot_->size_.x1min)
             * (mgroot_->size_.x2max - mgroot_->size_.x2min)
             * (mgroot_->size_.x3max - mgroot_->size_.x3min);
    norm /= vol;
  }
  // Take square root for L2 norm
  if (nrm == MGNormType::l2) {
    norm = std::sqrt(norm);
  }
  return norm;
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
    // Periodic BC in x-direction
    for (int n = 0; n < ngh; ++n) {
      u(m, v, k, j, n) = u(m, v, k, j, nx - 2*ngh + n);  // left ghost <- right interior
      u(m, v, k, j, nx - ngh + n) = u(m, v, k, j, ngh + n);  // right ghost <- left interior
    }
  });
  
  par_for("MG::PackAndSendMGRoot_y", DevExeSpace(),
          0, nvar-1, 0, nz-1, 0, nx-1,
  KOKKOS_LAMBDA(const int v, const int k, const int i) {
    // Periodic BC in y-direction
    for (int n = 0; n < ngh; ++n) {
      u(m, v, k, n, i) = u(m, v, k, ny - 2*ngh + n, i);  // front ghost <- back interior
      u(m, v, k, ny - ngh + n, i) = u(m, v, k, ngh + n, i);  // back ghost <- front interior
    }
  });
  
  par_for("MG::PackAndSendMGRoot_z", DevExeSpace(),
          0, nvar-1, 0, ny-1, 0, nx-1,
  KOKKOS_LAMBDA(const int v, const int j, const int i) {
    // Periodic BC in z-direction
    for (int n = 0; n < ngh; ++n) {
      u(m, v, n, j, i) = u(m, v, nz - 2*ngh + n, j, i);  // bottom ghost <- top interior
      u(m, v, nz - ngh + n, j, i) = u(m, v, ngh + n, j, i);  // top ghost <- bottom interior
    }
  });
}