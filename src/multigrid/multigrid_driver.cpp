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
    pmy_mesh_(pmbp->pmesh), fsubtract_average_(false),
    needinit_(true), eps_(-1.0),
    niter_(-1), npresmooth_(1), npostsmooth_(1), coffset_(0), fprolongation_(0),
    nb_rank_(0) {
  std::cout << std::scientific << std::setprecision(15);
  if (pmy_mesh_->mb_indcs.nx2==1 || pmy_mesh_->mb_indcs.nx3==1) {
    std::cout << "### FATAL ERROR in MultigridDriver::MultigridDriver" << std::endl
        << "Currently the Multigrid solver works only in 3D." << std::endl;
    exit(EXIT_FAILURE);
    return;
  }

  //for (int i=0; i<6; i++) {
  //  MGBoundaryFunction_[i] = MGBoundary[i];
  //  MGCoeffBoundaryFunction_[i] = MGCoeffBoundary[i];
  //}

  ranklist_  = new int[nbtotal_];
  int nv = std::max(nvar_*2, ncoeff_);
  rootbuf_ = new Real[nbtotal_*nv];
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

#ifdef MPI_PARALLEL
  MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_MULTIGRID);
  mg_phys_id_ = pmy_mesh_->ReserveTagPhysIDs(1);
#endif

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
  delete [] rootbuf_;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SubtractAverage(MGVariable type)
//  \brief Calculate the global average and subtract it

void MultigridDriver::SubtractAverage(MGVariable type) {
//#pragma omp parallel for num_threads(nthreads_)
//  for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
//    Multigrid *pmg = *itr;
//    for (int v=0; v<nvar_; ++v)
//      rootbuf_[pmg->pmy_block_-> mb_gid[0]*nvar_+v] = pmg->CalculateTotal(type, v);
//  }
//#ifdef MPI_PARALLEL
//  if (nb_rank_ > 0)  // every rank has the same number of MeshBlocks
//    MPI_Allgather(MPI_IN_PLACE, nb_rank_*nvar_, MPI_ATHENA_REAL,
//                  rootbuf_, nb_rank_*nvar_, MPI_ATHENA_REAL, MPI_COMM_MULTIGRID);
//  else
//    MPI_Allgatherv(MPI_IN_PLACE, nblist_[Globals::my_rank]*nvar_, MPI_ATHENA_REAL,
//                   rootbuf_, nvlisti_, nvslisti_, MPI_ATHENA_REAL, MPI_COMM_MULTIGRID);
//#endif
//  Real vol = (pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min)
//           * (pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min)
//           * (pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);
//  for (int v=0; v<nvar_; ++v) {
//    Real total = 0.0;
//    for (int n = 0; n < nbtotal_; ++n)
//      total += rootbuf_[n*nvar_+v];
//    last_ave_ = total/vol;
//#pragma omp parallel for num_threads(nthreads_)
//    for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
//      Multigrid *pmg = *itr;
//      pmg->SubtractAverage(type, v, last_ave_);
//    }
//  }
//
//  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial)
//  \brief initialize the source assuming that the source terms are already loaded

void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial) {
  locrootlevel_ = pmy_mesh_->root_level;
  nrootlevel_ = mgroot_->GetNumberOfLevels();
  nmblevel_ = mglevels_->GetNumberOfLevels();
  //nreflevel_ = current_level_ - locrootlevel_;
  ntotallevel_ = nrootlevel_ + nmblevel_ - 1;
  std::cout<< "Multigrid total levels: " << ntotallevel_ << std::endl;
  std::cout<< "Multigrid root levels: " << nrootlevel_ << std::endl;
  std::cout<< "Multigrid meshblock levels: " << nmblevel_ << std::endl;
  std::cout<< "Multigrid refinement levels: " << nreflevel_ << std::endl;

  os_ = mgroot_->ngh_;
  oe_ = os_+1;
  // note: the level of an Octet is one level lower than the data stored there
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
  if (fsubtract_average_)SubtractAverage(MGVariable::src);
  current_level_ = ntotallevel_ - 1;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromBlocksToRoot(bool initflag)
//! \brief collect the coarsest data and transfer to the root grid

void MultigridDriver::TransferFromBlocksToRoot(bool initflag) {
  int nv = nvar_;
  if (!initflag) nv *= 2; // store both src and u when !initflag
  
  // Gather data from meshblock-level multigrids into rootbuf_
  if (mglevels_ != nullptr) {
    // mglevels_ is a pack-aware multigrid covering nmmb_ meshblocks
    int nmmb = mglevels_->nmmb_;
    
    for (int m = 0; m < nmmb; ++m) {
      // Get the global meshblock ID (or use m as a simple index)
      // For pack-based structures, you may need to map pack index to global gid
      int gid = m; // placeholder; adjust if global IDs are available elsewhere

      // Transfer source data (always)
      for (int v = 0; v < nvar_; ++v) {
        rootbuf_[gid * nv + v] = mglevels_->GetCoarsestData(MGVariable::src, v, m);
      }

      // Transfer solution data when running FAS (full approximation storage) and not initializing
      if (!initflag) {
        for (int v = 0; v < nvar_; ++v) {
          rootbuf_[gid * nv + nvar_ + v] = mglevels_->GetCoarsestData(MGVariable::u, v, m);
        }
      }
    }
  }
  // TODO: Add MPI communication here if running in parallel to gather data to root grid

  for (int m = 0; m < nbtotal_; ++m) {
    const LogicalLocation &loc=pmy_mesh_->lloc_eachmb[m];
    int i = static_cast<int>(loc.lx1);
    int j = static_cast<int>(loc.lx2);
    int k = static_cast<int>(loc.lx3);
    if (loc.level == locrootlevel_) {
        for (int v = 0; v < nvar_; ++v)
        mgroot_->SetData(MGVariable::src, v, k, j, i, rootbuf_[m*nv+v]);
      if (!initflag) {
        for (int v = 0; v < nvar_; ++v)
          mgroot_->SetData(MGVariable::u, v, k, j, i, rootbuf_[m*nv+nvar_+v]);
      }
    } 
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromRootToBlocks(bool folddata)
//! \brief Transfer the data from the root grid to the coarsest level of each MeshBlock

void MultigridDriver::TransferFromRootToBlocks(bool folddata) {
  //TODO: Add Octet-based data transfer if needed
  mglevels_->SetFromRootGrid(folddata);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::OneStepToFiner(int nsmooth)
//! \brief prolongation and smoothing one level

void MultigridDriver::OneStepToFiner(Driver *pdriver, int nsmooth) {
  int ngh=mgroot_->ngh_;
  int flag=0;
  if (current_level_ == nrootlevel_ - 1) {
    MGRootBoundary(mgroot_->GetCurrentData());
    TransferFromRootToBlocks(true);
    flag=1;
  }
  if (current_level_ >= nrootlevel_ - 1) { // MeshBlocks
    std::cout << "Meshblocks at level " << current_level_ << std::endl;
    mglevels_->PrintActiveRegion(mglevels_->GetCurrentData());
    pmg = mglevels_;
    if (current_level_ == ntotallevel_ - 2) flag=2;
    SetMGTaskListToFiner(nsmooth, ngh);
    std::cout << "Prolongate and correct to level " << current_level_+1 << std::endl; 
    pdriver->ExecuteTaskList(pmy_mesh_,"mg_to_finer",0);
    current_level_++;
  } 
  else { // root grid
    std::cout << "Root grid at level " << current_level_ << std::endl;
    mgroot_->PrintActiveRegion(mgroot_->GetCurrentData());
    MGRootBoundary(mgroot_->GetCurrentData());
    std::cout << "Prolongate and correct to level " << current_level_+1 << std::endl; 
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
    std::cout << "Meshblocks at level " << current_level_ << std::endl;
    pmg = mglevels_;
    SetMGTaskListToCoarser(nsmooth, ngh);
    pdriver->ExecuteTaskList(pmy_mesh_,"mg_to_coarser",0);
    std::cout << "Current level: " << current_level_ << ", nrootlevel_: " << nrootlevel_ << ", nreflevel_: " << nreflevel_ << std::endl;
    if (current_level_ == nrootlevel_ + nreflevel_) {
      std::cout << "Transfer from blocks to root grid at level " << current_level_ << std::endl;
      TransferFromBlocksToRoot(false);
    }
  }
  else { // uniform root grid
    std::cout << "Root grid at level " << current_level_ << std::endl;
    //mgroot_->pmgbval->ApplyPhysicalBoundaries(0, false);
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
  std::cout << "Starting V-Cycle at level " << current_level_ << std::endl;
  while (current_level_ > 0)
    OneStepToCoarser(pdriver, npresmooth);
  SolveCoarsestGrid();
  while (current_level_ < startlevel)
    OneStepToFiner(pdriver, npostsmooth);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveIterativeFixedTimes()
//  \brief Solve iteratively niter_ times

void MultigridDriver::SolveIterative(Driver *pdriver) {
  niter_ = 1; // for testing
  std::cout << "Starting Multigrid SolveIterative with " << niter_ << " V-cycles." << std::endl;
  for (int n = 0; n < niter_; ++n)
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  Real def = 0.0;
  for (int v = 0; v < nvar_; ++v)
    def += CalculateDefectNorm(MGNormType::l2, v);
  //if (fshowdef_ && global_variable::my_rank == 0)
    std::cout << "Multigrid defect L2-norm : " << def << std::endl;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveCoarsestGrid()
//! \brief Solve the coarsest root grid

void MultigridDriver::SolveCoarsestGrid() {
  int ni = (std::max(nrbx1_, std::max(nrbx2_, nrbx3_))
            >> (nrootlevel_-1));
  if (fsubtract_average_ && ni == 1) { // trivial case - all zero
      MGRootBoundary(mgroot_->GetCurrentData());
      mgroot_->StoreOldData();
      mgroot_->ZeroClearData();
  } else {
    if (fsubtract_average_) {
      Real vol=(mgroot_->size_.x1max-mgroot_->size_.x1min)
          *(mgroot_->size_.x2max-mgroot_->size_.x2min)
          *(mgroot_->size_.x3max-mgroot_->size_.x3min);
      for (int v=0; v<nvar_; ++v) {
        Real ave=mgroot_->CalculateTotal(MGVariable::u, v)/vol;
        mgroot_->SubtractAverage(MGVariable::u, v, ave);
      }
    }
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
    std::cout << "Solved coarsest grid with " << ni << " smoothing iterations." << std::endl; 
    std::cout << "Solution at coarsest grid:" << std::endl;
    mgroot_->PrintActiveRegion(mgroot_->GetCurrentData());
  }
  if (fsubtract_average_) {
    Real vol=(mgroot_->size_.x1max-mgroot_->size_.x1min)
            *(mgroot_->size_.x2max-mgroot_->size_.x2min)
            *(mgroot_->size_.x3max-mgroot_->size_.x3min);
    for (int v = 0; v < nvar_; ++v) {
      Real ave=mgroot_->CalculateTotal(MGVariable::u, v)/vol;
      mgroot_->SubtractAverage(MGVariable::u, v, ave);
    }
  }
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

  // Also include root-grid defect if applicable (coarsest level)
  else if (mgroot_ != nullptr) {
    Real root_norm = mgroot_->CalculateDefectNorm(nrm, n);
    if (nrm == MGNormType::max) {
      norm = std::max(norm, root_norm);
    } else {
      norm += root_norm;
    }
  }

  //TODO: Add MPI reduction here if running in parallel

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

  std::cout << " MG::PackAndSendMGRoot at level " << current_level << std::endl;
}