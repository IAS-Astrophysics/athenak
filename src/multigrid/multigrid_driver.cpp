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
#include "../driver/driver.hpp"
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
    needinit_(true), amr_seq_(0), nreflevel_(0), eps_(-1.0),
    niter_(-1), npresmooth_(1), npostsmooth_(1), coffset_(0), fprolongation_(0),
    nb_rank_(0), ncoeff_(0),
    octets_(nullptr), octetmap_(nullptr), octetbflag_(nullptr), noctets_(nullptr),
    oct_u_buf_(nullptr), oct_def_buf_(nullptr),
    oct_src_buf_(nullptr), oct_uold_buf_(nullptr), octet_stride_(0),
    root_buf_nc_(0), root_flat_buf_stale_(true),
    root_sync_state_(RootSyncState::SYNCED),
    mask_radius_(-1.0), autompo_(false), nodipole_(false),
    mporder_(-1), nmpcoeff_(0) {
  mask_origin_[0] = mask_origin_[1] = mask_origin_[2] = 0.0;
  mpo_[0] = mpo_[1] = mpo_[2] = 0.0;
  std::memset(mpcoeff_, 0, sizeof(mpcoeff_));
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

  // Initialize MG mesh boundary conditions from mesh BCs.
  // Periodic stays periodic; all other types default to mg_zerofixed (Dirichlet zero).
  for (int f = 0; f < 6; ++f) {
    BoundaryFlag mbc = pmy_mesh_->mesh_bcs[f];
    if (mbc == BoundaryFlag::periodic) {
      mg_mesh_bcs_[f] = BoundaryFlag::periodic;
    } else {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerofixed;
    }
  }

  // Allocate octet arrays for max possible refinement levels
  if (maxreflevel_ > 0) {
    octets_ = new std::vector<MGOctet>[maxreflevel_];
    octetmap_ = new std::unordered_map<LogicalLocation, int, LogicalLocationHash>[maxreflevel_];
    octetbflag_ = new std::vector<bool>[maxreflevel_];
    noctets_ = new int[maxreflevel_]();
    oct_u_buf_    = new std::vector<Real>[maxreflevel_];
    oct_def_buf_  = new std::vector<Real>[maxreflevel_];
    oct_src_buf_  = new std::vector<Real>[maxreflevel_];
    oct_uold_buf_ = new std::vector<Real>[maxreflevel_];
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
  delete [] oct_u_buf_;
  delete [] oct_def_buf_;
  delete [] oct_src_buf_;
  delete [] oct_uold_buf_;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootToHost()
//! \brief Sync root grid arrays from device to host (no-op if root runs on host)

void MultigridDriver::SyncRootToHost() {
  if (mgroot_->on_host_) return;
  if (root_sync_state_ != RootSyncState::DEVICE_MODIFIED) return;
  int lev = mgroot_->current_level_;
  Kokkos::deep_copy(mgroot_->u_[lev].h_view, mgroot_->u_[lev].d_view);
  Kokkos::deep_copy(mgroot_->uold_[lev].h_view, mgroot_->uold_[lev].d_view);
  Kokkos::deep_copy(mgroot_->src_[lev].h_view, mgroot_->src_[lev].d_view);
  root_sync_state_ = RootSyncState::SYNCED;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootToDevice()
//! \brief Sync root grid arrays from host to device (no-op if root runs on host)

void MultigridDriver::SyncRootToDevice() {
  if (mgroot_->on_host_) return;
  if (root_sync_state_ != RootSyncState::HOST_MODIFIED) return;
  int lev = mgroot_->current_level_;
  Kokkos::deep_copy(mgroot_->u_[lev].d_view, mgroot_->u_[lev].h_view);
  Kokkos::deep_copy(mgroot_->src_[lev].d_view, mgroot_->src_[lev].h_view);
  root_sync_state_ = RootSyncState::SYNCED;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::MarkRootDeviceModified()
//! \brief Mark root device data as modified (for sync tracking)

void MultigridDriver::MarkRootDeviceModified() {
  if (!mgroot_->on_host_)
    root_sync_state_ = RootSyncState::DEVICE_MODIFIED;
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

void MultigridDriver::PrepareForAMR() {
  locrootlevel_ = pmy_mesh_->root_level;

  // Detect if mesh has changed (AMR or load balancing).
  // Athena++ uses pmy_mesh_->amr_updated; AthenaK tracks cumulative AMR events instead.
  int new_nbtotal = pmy_mesh_->nmb_total;
  if (new_nbtotal != nbtotal_) {
    nbtotal_ = new_nbtotal;
    delete[] ranklist_;
    ranklist_ = new int[nbtotal_];
    int nv = nvar_*2;
    Kokkos::realloc(rootbuf_, nv, nbtotal_);
    needinit_ = true;
  }

  if (pmy_mesh_->pmr != nullptr) {
    int new_seq = pmy_mesh_->pmr->nmb_created + pmy_mesh_->pmr->nmb_deleted;
    if (new_seq != amr_seq_) {
      amr_seq_ = new_seq;
      needinit_ = true;
    }
  }

  // Calculate number of refinement levels present in mesh
  int old_nreflevel = nreflevel_;
  nreflevel_ = 0;
  if (pmy_mesh_->multilevel) {
    for (int n = 0; n < nbtotal_; ++n) {
      int lev = pmy_mesh_->lloc_eachmb[n].level - locrootlevel_;
      nreflevel_ = std::max(nreflevel_, lev);
    }
    if (nreflevel_ != old_nreflevel) {
      std::cout << "MultigridDriver::SetupMultigrid: Number of refinement levels = "
                << nreflevel_ << std::endl;
    }
  }

  if (needinit_) {
    mglevels_->ReallocateForAMR();
    for (int n = 0; n < nbtotal_; ++n)
      ranklist_[n] = pmy_mesh_->rank_eachmb[n];
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
    root_flat_buf_stale_ = true;
    if (pmy_mesh_->multilevel) {
      mglevels_->UpdateBlockDx();
    }
  }
  needinit_ = false;
}


void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial) {
  locrootlevel_ = pmy_mesh_->root_level;
  nrootlevel_ = mgroot_->GetNumberOfLevels();
  nmblevel_ = mglevels_->GetNumberOfLevels();

  // Include refinement levels in total (octets are V-cycle participants)
  ntotallevel_ = nrootlevel_ + nmblevel_ + nreflevel_ - 1;
  os_ = mgroot_->ngh_;
  oe_ = os_+1;

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
          octets_[l][oid].Init(nvar_, ngh);
        }
        octets_[l][oid].loc = oloc;
        octets_[l][oid].fleaf = false;
      }
    }
  }

  // Allocate contiguous buffers and wire octet pointers
  {
    int nc = 2 + 2*ngh;
    octet_stride_ = nvar_ * nc * nc * nc;
    for (int l = 0; l < nreflevel_; ++l) {
      int noct = noctets_[l];
      std::size_t total = static_cast<std::size_t>(noct) * octet_stride_;
      oct_u_buf_[l].assign(total, 0.0);
      oct_def_buf_[l].assign(total, 0.0);
      oct_src_buf_[l].assign(total, 0.0);
      oct_uold_buf_[l].assign(total, 0.0);
      for (int o = 0; o < noct; ++o) {
        std::size_t off = static_cast<std::size_t>(o) * octet_stride_;
        octets_[l][o].u    = oct_u_buf_[l].data()    + off;
        octets_[l][o].def  = oct_def_buf_[l].data()  + off;
        octets_[l][o].src  = oct_src_buf_[l].data()   + off;
        octets_[l][o].uold = oct_uold_buf_[l].data() + off;
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

  // Precompute neighbor tables (eliminates hash lookups in SetBoundariesOctets)
  for (int l = 0; l < nreflevel_; ++l) {
    int maxlx1 = nrbx1_ << l;
    int maxlx2 = nrbx2_ << l;
    int maxlx3 = nrbx3_ << l;
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &oct = octets_[l][o];
      const LogicalLocation &oloc = oct.loc;
      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            int dir = (ox3+1)*9 + (ox2+1)*3 + (ox1+1);
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) {
              oct.neighbors[dir] = {-1, -1};
              continue;
            }
            LogicalLocation nloc;
            nloc.level = oloc.level;
            bool outside = false;
            nloc.lx1 = oloc.lx1 + ox1;
            if (nloc.lx1 < 0) {
              if (mg_mesh_bcs_[BoundaryFace::inner_x1] == BoundaryFlag::periodic)
                nloc.lx1 = maxlx1 - 1;
              else outside = true;
            }
            if (nloc.lx1 >= maxlx1) {
              if (mg_mesh_bcs_[BoundaryFace::outer_x1] == BoundaryFlag::periodic)
                nloc.lx1 = 0;
              else outside = true;
            }
            nloc.lx2 = oloc.lx2 + ox2;
            if (nloc.lx2 < 0) {
              if (mg_mesh_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::periodic)
                nloc.lx2 = maxlx2 - 1;
              else outside = true;
            }
            if (nloc.lx2 >= maxlx2) {
              if (mg_mesh_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::periodic)
                nloc.lx2 = 0;
              else outside = true;
            }
            nloc.lx3 = oloc.lx3 + ox3;
            if (nloc.lx3 < 0) {
              if (mg_mesh_bcs_[BoundaryFace::inner_x3] == BoundaryFlag::periodic)
                nloc.lx3 = maxlx3 - 1;
              else outside = true;
            }
            if (nloc.lx3 >= maxlx3) {
              if (mg_mesh_bcs_[BoundaryFace::outer_x3] == BoundaryFlag::periodic)
                nloc.lx3 = 0;
              else outside = true;
            }
            if (outside) {
              oct.neighbors[dir] = {-2, -2};
              continue;
            }
            if (octetmap_[l].count(nloc) == 1) {
              oct.neighbors[dir] = {octetmap_[l][nloc], -1};
            } else {
              int cid = -1;
              if (l > 0) {
                LogicalLocation cloc;
                cloc.lx1 = nloc.lx1 >> 1;
                cloc.lx2 = nloc.lx2 >> 1;
                cloc.lx3 = nloc.lx3 >> 1;
                cloc.level = nloc.level - 1;
                cid = octetmap_[l-1][cloc];
              }
              oct.neighbors[dir] = {-1, cid};
            }
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


void MultigridDriver::BuildRootFlatBuffers() {
  if (!root_flat_buf_stale_) return;
  SyncRootToHost();
  auto root_u_h = GetRootData_h();
  auto root_uold_h = GetRootOldData_h();
  int rnx = root_u_h.extent_int(4);
  int rny = root_u_h.extent_int(3);
  int rnz = root_u_h.extent_int(2);
  int rnv = root_u_h.extent_int(1);
  int nc = std::max({rnx, rny, rnz});
  root_buf_nc_ = nc;
  int total = rnv*nc*nc*nc;
  root_u_buf_.assign(total, 0.0);
  root_uold_buf_.assign(total, 0.0);
  for (int v = 0; v < rnv; ++v)
    for (int k = 0; k < rnz; ++k)
      for (int j = 0; j < rny; ++j)
        for (int i = 0; i < rnx; ++i) {
          int idx = ((v*nc+k)*nc+j)*nc+i;
          root_u_buf_[idx] = root_u_h(0,v,k,j,i);
          root_uold_buf_[idx] = root_uold_h(0,v,k,j,i);
        }
  root_flat_buf_stale_ = false;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromBlocksToRoot(bool initflag)
//! \brief collect the coarsest data and transfer to the root grid
//! Following Athena++: each block sends its coarsest cell data to either
//! the root grid (if at root level) or the appropriate octet.

void MultigridDriver::TransferFromBlocksToRoot(bool initflag) {
  const int nv = nvar_;
  auto rootbuf = rootbuf_;
  const auto &src = mglevels_->src_[0].d_view;
  const auto &u = mglevels_->u_[0].d_view;
  const int ngh_mb = mglevels_->ngh_;
  int nmmb = mglevels_->nmmb_ - 1;
  int padding = nslist_[global_variable::my_rank];
  par_for("Multigrid:SaveToRoot", DevExeSpace(), 0, nmmb, KOKKOS_LAMBDA(const int m) {
    for (int v = 0; v < nv; ++v) {
      rootbuf.d_view(v,    m+padding) = src(m, v, ngh_mb, ngh_mb, ngh_mb);
      if (!initflag)
        rootbuf.d_view(v+nv, m+padding) = u(m, v, ngh_mb, ngh_mb, ngh_mb);
    }
  });
  rootbuf.template modify<DevExeSpace>();
  rootbuf.template sync<HostExeSpace>();
#if MPI_PARALLEL_ENABLED
  int ncomm = initflag ? nv : 2*nv;
  for (int v = 0; v < ncomm; ++v) {
    MPI_Allgatherv(MPI_IN_PLACE, nblist_[global_variable::my_rank], MPI_ATHENA_REAL,
                   &rootbuf.h_view(v,0), nblist_, nslist_, MPI_ATHENA_REAL, MPI_COMM_WORLD);
  }
#endif

  const auto loc = pmy_mesh_->lloc_eachmb;
  int rootlevel = locrootlevel_;
  int ngh = mgroot_->ngh_;

  auto root_src_h = GetRootSource_h();
  auto root_u_h = GetRootData_h();

  for (int n = 0; n < nbtotal_; ++n) {
    int i = static_cast<int>(loc[n].lx1);
    int j = static_cast<int>(loc[n].lx2);
    int k = static_cast<int>(loc[n].lx3);
    if (loc[n].level == rootlevel) {
      for (int v = 0; v < nv; ++v) {
        root_src_h(0, v, k+ngh, j+ngh, i+ngh) = rootbuf.h_view(v, n);
        if (!initflag)
          root_u_h(0, v, k+ngh, j+ngh, i+ngh) = rootbuf.h_view(v+nv, n);
      }
    } else {
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
  root_flat_buf_stale_ = true;
  mgroot_->current_level_ = nrootlevel_ - 1;
  root_sync_state_ = RootSyncState::HOST_MODIFIED;
  if (nreflevel_ == 0) SyncRootToDevice();
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
    MGRootBoundary();
    TransferFromRootToBlocks(false);
  }
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListFMGProlongate(ngh);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_fmg_prolongate", 0);
    current_level_++;
  } else if (current_level_ >= nrootlevel_ - 1) { // octets
    if (current_level_ == nrootlevel_ - 1)
      MGRootBoundary();
    else
      SetBoundariesOctets(true, false);
    FMGProlongateOctets();
    current_level_++;
  } else { // root grid
    MGRootBoundary();
    mgroot_->FMGProlongatePack();
    MarkRootDeviceModified();
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
    MGRootBoundary();
    TransferFromRootToBlocks(true);
  }
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    int flag = 0;
    // flag = 1: first time on meshblock levels
    if (current_level_ == nrootlevel_ + nreflevel_ - 1) flag = 1;
    
    if (current_level_ == ntotallevel_ - 2) flag = 2;
    SetMGTaskListToFiner(nsmooth, ngh, flag);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_to_finer", 0);
    current_level_++;
  } else if (current_level_ >= nrootlevel_ - 1) { // octets
    if (current_level_ == nrootlevel_ - 1) {
      MGRootBoundary();
    } else {
      SetBoundariesOctets(true, true);
    }
    ProlongateAndCorrectOctets();
    current_level_++;
    for (int n = 0; n < nsmooth; ++n) {
      SetBoundariesOctets(false, false);
      SmoothOctets(coffset_);
      SetBoundariesOctets(false, false);
      SmoothOctets(1 - coffset_);
    }
  } else { // root grid
    MGRootBoundary();
    mgroot_->ProlongateAndCorrectPack();
    MarkRootDeviceModified();
    current_level_++;
    for (int n = 0; n < nsmooth; ++n) {
      MGRootBoundary();
      mgroot_->SmoothPack(coffset_);
      MarkRootDeviceModified();
      MGRootBoundary();
      mgroot_->SmoothPack(1-coffset_);
      MarkRootDeviceModified();
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
      if (nreflevel_ > 0) {
        PreRestrictOctetU();
      }
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
    MGRootBoundary();
    if (current_level_ < fmglevel_) {
      mgroot_->StoreOldData();
      mgroot_->CalculateFASRHSPack();
      MarkRootDeviceModified();
    }
    for (int n = 0; n < nsmooth; ++n) {
      mgroot_->SmoothPack(coffset_);
      MarkRootDeviceModified();
      MGRootBoundary();
      mgroot_->SmoothPack(1-coffset_);
      MarkRootDeviceModified();
      MGRootBoundary();
    }
    mgroot_->RestrictPack();
    MarkRootDeviceModified();
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
    MarkRootDeviceModified();
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
  if (fshowdef_) {
    std::cout << "MG initial defect = " << def << std::endl;
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
    if (n >= 80) {
      std::cout << "### FATAL ERROR in MultigridDriver::SolveIterative" << std::endl
                << "Failed to converge after " << n << " iterations (defect = "
                << def << ", threshold = " << eps_ << ")" << std::endl;
      pdriver->nlim = pmy_mesh_->ncycle;
      break;
    }
  }
  Kokkos::fence();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveIterativeFixedTimes(Driver *pdriver)
//! \brief Solve iteratively niter_ times (fixed count)

void MultigridDriver::SolveIterativeFixedTimes(Driver *pdriver) {
  if (fshowdef_) {
    Real norm = CalculateDefectNorm(MGNormType::l2, 0);
    std::cout << "MG initial defect = " << norm << std::endl;
  }
  for (int n = 0; n < niter_; ++n) {
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    if (fshowdef_) {
      Real norm = CalculateDefectNorm(MGNormType::l2, 0);
      std::cout << "MG iteration " << n << ": defect = " << norm << std::endl;
    }
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
    MGRootBoundary();
    mgroot_->StoreOldData();
    mgroot_->ZeroClearData();
    MarkRootDeviceModified();
    return;
  }
  if (fsubtract_average_) {
    SubtractAverage(MGVariable::src);
    MarkRootDeviceModified();
    SubtractAverage(MGVariable::u);
    MarkRootDeviceModified();
  }
  MGRootBoundary();
  mgroot_->StoreOldData();
  mgroot_->CalculateFASRHSPack();
  MarkRootDeviceModified();
  for (int i = 0; i < ni; ++i) {
    mgroot_->SmoothPack(coffset_);
    MarkRootDeviceModified();
    MGRootBoundary();
    mgroot_->SmoothPack(1-coffset_);
    MarkRootDeviceModified();
    MGRootBoundary();
  }
  if (fsubtract_average_) {
    SubtractAverage(MGVariable::u);
    MarkRootDeviceModified();
  }
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
  auto root_src_h = GetRootSource_h();
  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    const LogicalLocation &oloc = oct.loc;
    for (int v = 0; v < nvar_; ++v)
      root_src_h(0, v, static_cast<int>(oloc.lx3)+ngh,
                       static_cast<int>(oloc.lx2)+ngh,
                       static_cast<int>(oloc.lx1)+ngh) =
        RestrictOneSrc(oct, v, ngh, ngh, ngh);
  }
  root_flat_buf_stale_ = true;
  root_sync_state_ = RootSyncState::HOST_MODIFIED;
  SyncRootToDevice();
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
    auto root_src_h = GetRootSource_h();
    auto root_u_h = GetRootData_h();

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      const LogicalLocation &oloc = oct.loc;
      int ri = static_cast<int>(oloc.lx1);
      int rj = static_cast<int>(oloc.lx2);
      int rk = static_cast<int>(oloc.lx3);
      CalculateDefectOctet(oct, 1);
      for (int v = 0; v < nvar_; ++v) {
        root_src_h(0, v, rk+ngh, rj+ngh, ri+ngh) =
            RestrictOneDef(oct, v, ngh, ngh, ngh);
        root_u_h(0, v, rk+ngh, rj+ngh, ri+ngh) =
            RestrictOne(oct, v, ngh, ngh, ngh);
      }
    }
    root_flat_buf_stale_ = true;
    root_sync_state_ = RootSyncState::HOST_MODIFIED;
    SyncRootToDevice();
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
    auto root_u_h = GetRootData_h();
    auto root_uold_h = GetRootOldData_h();

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
              cbuf[kk+1][jj+1][ii+1] =
                  root_u_h(0, v, rk+kk, rj+jj, ri+ii)
                  - root_uold_h(0, v, rk+kk, rj+jj, ri+ii);
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
              cbuf[kk+1][jj+1][ii+1] =
                  coct.U(v, ck+kk, cj+jj, ci+ii)
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
    auto root_u_h = GetRootData_h();

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
                           * root_u_h(0, v, rk+kk, rj+jj, ri+ii);
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

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          int dir = (ox3+1)*9 + (ox2+1)*3 + (ox1+1);
          const OctetNeighborInfo &nb = oct.neighbors[dir];
          if (nb.same_id == -2) {
            continue;  // physical boundary — handled by ApplyPhysicalBoundariesOctet
          } else if (nb.same_id >= 0) {
            MGOctet &noct = octets_[lev][nb.same_id];
            SetOctetBoundarySameLevel(oct, noct, cbuf_, cbufold_,
                                      nvar_, ox1, ox2, ox3, folddata);
          } else if (!fprolong) {
            ncoarse_[dir] = true;
            if (lev > 0 && nb.coarse_id >= 0) {
              MGOctet &coct = octets_[lev-1][nb.coarse_id];
              SetOctetBoundaryFromCoarser(coct.u, coct.uold, cbuf_, cbufold_,
                                          nvar_, coct.nc, loc, ox1, ox2, ox3, folddata);
            } else if (lev == 0) {
              BuildRootFlatBuffers();
              LogicalLocation nloc;
              nloc.level = loc.level;
              nloc.lx1 = loc.lx1 + ox1;
              nloc.lx2 = loc.lx2 + ox2;
              nloc.lx3 = loc.lx3 + ox3;
              // periodic wrapping (only reached for periodic BCs)
              if (nloc.lx1 < 0)       nloc.lx1 = nrbx1_ - 1;
              if (nloc.lx1 >= nrbx1_)  nloc.lx1 = 0;
              if (nloc.lx2 < 0)       nloc.lx2 = nrbx2_ - 1;
              if (nloc.lx2 >= nrbx2_)  nloc.lx2 = 0;
              if (nloc.lx3 < 0)       nloc.lx3 = nrbx3_ - 1;
              if (nloc.lx3 >= nrbx3_)  nloc.lx3 = 0;
              SetOctetBoundaryFromCoarser(root_u_buf_.data(), root_uold_buf_.data(),
                                          cbuf_, cbufold_,
                                          nvar_, root_buf_nc_, nloc,
                                          ox1, ox2, ox3, folddata);
            }
          }
        }
      }
    }

    if (!fprolong) {
      ApplyPhysicalBoundariesOctet(oct, true);
      ProlongateOctetBoundariesFluxCons(oct, cbuf_, ncoarse_);
    }
    ApplyPhysicalBoundariesOctet(oct, false);
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

void MultigridDriver::SetOctetBoundaryFromCoarser(const Real *un,
     const Real *unold,
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
//! \fn void MultigridDriver::ProlongateOctetBoundariesFluxCons(...)
//! \brief Base class default: delegates to normal trilinear prolongation.
//!        Physics-specific drivers (e.g. gravity) override this with the
//!        conservative formulation from Tomida & Stone (2023) Section 3.3.2.

void MultigridDriver::ProlongateOctetBoundariesFluxCons(MGOctet &oct,
     std::vector<Real> &cbuf, const std::vector<bool> &ncoarse) {
  ProlongateOctetBoundaries(oct, cbuf, cbufold_, nvar_, ncoarse, false);
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
//! \fn void MultigridDriver::PreRestrictOctetU()
//! \brief After TransferFromBlocksToRoot, restrict u from finer octets to fill parent
//!        cells at coarser levels. This ensures that within mixed octets (containing both
//!        parent and leaf cells), the u values are consistent — preventing corrupt
//!        transverse gradients in ProlongateOctetBoundariesFluxCons during the descent.

void MultigridDriver::PreRestrictOctetU() {
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
  auto root_u_h = GetRootData_h();
  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    const LogicalLocation &oloc = oct.loc;
    for (int v = 0; v < nvar_; ++v)
      root_u_h(0, v, static_cast<int>(oloc.lx3)+ngh,
                     static_cast<int>(oloc.lx2)+ngh,
                     static_cast<int>(oloc.lx1)+ngh) =
          RestrictOne(oct, v, ngh, ngh, ngh);
  }
  root_sync_state_ = RootSyncState::HOST_MODIFIED;
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

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          int dir = (ox3+1)*9 + (ox2+1)*3 + (ox1+1);
          const OctetNeighborInfo &nb = oct.neighbors[dir];
          if (nb.same_id == -2) {
            continue;  // physical boundary — handled by ApplyPhysicalBoundariesOctet
          } else if (nb.same_id >= 0) {
            MGOctet &noct = octets_[lev][nb.same_id];
            SetOctetBoundarySameLevel(oct, noct, cbuf_, cbufold_,
                                      nvar_, ox1, ox2, ox3, folddata);
          } else {
            ncoarse_[dir] = true;
            if (lev > 0 && nb.coarse_id >= 0) {
              MGOctet &coct = octets_[lev-1][nb.coarse_id];
              SetOctetBoundaryFromCoarser(coct.u, coct.uold, cbuf_, cbufold_,
                                          nvar_, coct.nc, oloc, ox1, ox2, ox3, folddata);
            } else if (lev == 0) {
              BuildRootFlatBuffers();
              LogicalLocation nloc;
              nloc.level = oloc.level;
              nloc.lx1 = oloc.lx1 + ox1;
              nloc.lx2 = oloc.lx2 + ox2;
              nloc.lx3 = oloc.lx3 + ox3;
              // periodic wrapping (only reached for periodic BCs)
              if (nloc.lx1 < 0)       nloc.lx1 = nrbx1_ - 1;
              if (nloc.lx1 >= nrbx1_)  nloc.lx1 = 0;
              if (nloc.lx2 < 0)       nloc.lx2 = nrbx2_ - 1;
              if (nloc.lx2 >= nrbx2_)  nloc.lx2 = 0;
              if (nloc.lx3 < 0)       nloc.lx3 = nrbx3_ - 1;
              if (nloc.lx3 >= nrbx3_)  nloc.lx3 = 0;
              SetOctetBoundaryFromCoarser(root_u_buf_.data(), root_uold_buf_.data(),
                                          cbuf_, cbufold_,
                                          nvar_, root_buf_nc_, nloc,
                                          ox1, ox2, ox3, folddata);
            }
          }
        }
      }
    }

    ApplyPhysicalBoundariesOctet(oct, true);
    if (folddata)
      ProlongateOctetBoundaries(oct, cbuf_, cbufold_, nvar_, ncoarse_, folddata);
    else
      ProlongateOctetBoundaries(oct, cbuf_, cbufold_, nvar_, ncoarse_, false);
    ApplyPhysicalBoundariesOctet(oct, false);
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ApplyPhysicalBoundariesOctet(...)
//! \brief Apply physical boundary conditions to an octet at the domain boundary.
//! fcbuf=true: apply to the coarse buffer (ngh x ngh), fcbuf=false: apply to oct.u

void MultigridDriver::ApplyPhysicalBoundariesOctet(MGOctet &oct, bool fcbuf) {
  if (pmy_mesh_->strictly_periodic) return;
  const LogicalLocation &loc = oct.loc;
  int lev = loc.level - locrootlevel_;
  int ngh = mgroot_->ngh_;
  int l = ngh, r = ngh + 1;
  if (fcbuf) r = ngh;  // coarse buffer has only ngh cells

  // For zerofixed: ghost = -interior (antisymmetric reflection)
  // For zerograd:  ghost = +interior (symmetric reflection)
  // For multipole: ghost = 2*phi_mp - interior (linear extrapolation)
  // We implement zerofixed and zerograd here. Multipole at octet level
  // uses the same reflection approach (the root-level multipole already
  // sets the correct values; octets near the boundary inherit from the
  // root through the coarser-level boundary exchange).

  auto apply_bc = [&](Real *data, int nc) {
    auto ref = [&](int v, int k, int j, int i) -> Real& {
      return data[((v*nc + k)*nc + j)*nc + i];
    };

    // inner x1
    if (loc.lx1 == 0 && mg_mesh_bcs_[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
      Real sign = (mg_mesh_bcs_[BoundaryFace::inner_x1] == BoundaryFlag::mg_zerofixed)
                  ? -1.0 : 1.0;
      for (int v = 0; v < nvar_; ++v)
        for (int k = 0; k < nc; ++k)
          for (int j = 0; j < nc; ++j)
            for (int n = 0; n < ngh; ++n)
              ref(v, k, j, ngh-1-n) = sign * ref(v, k, j, ngh+n);
    }
    // outer x1
    int maxlx1 = nrbx1_ << lev;
    if (loc.lx1 == maxlx1-1
        && mg_mesh_bcs_[BoundaryFace::outer_x1] != BoundaryFlag::periodic) {
      Real sign = (mg_mesh_bcs_[BoundaryFace::outer_x1] == BoundaryFlag::mg_zerofixed)
                  ? -1.0 : 1.0;
      int ie = fcbuf ? ngh : ngh + 1;
      for (int v = 0; v < nvar_; ++v)
        for (int k = 0; k < nc; ++k)
          for (int j = 0; j < nc; ++j)
            for (int n = 0; n < ngh; ++n)
              ref(v, k, j, ie+n+1) = sign * ref(v, k, j, ie-n);
    }
    // inner x2
    if (loc.lx2 == 0 && mg_mesh_bcs_[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
      Real sign = (mg_mesh_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::mg_zerofixed)
                  ? -1.0 : 1.0;
      for (int v = 0; v < nvar_; ++v)
        for (int k = 0; k < nc; ++k)
          for (int i = 0; i < nc; ++i)
            for (int n = 0; n < ngh; ++n)
              ref(v, k, ngh-1-n, i) = sign * ref(v, k, ngh+n, i);
    }
    // outer x2
    int maxlx2 = nrbx2_ << lev;
    if (loc.lx2 == maxlx2-1
        && mg_mesh_bcs_[BoundaryFace::outer_x2] != BoundaryFlag::periodic) {
      Real sign = (mg_mesh_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::mg_zerofixed)
                  ? -1.0 : 1.0;
      int je = fcbuf ? ngh : ngh + 1;
      for (int v = 0; v < nvar_; ++v)
        for (int k = 0; k < nc; ++k)
          for (int i = 0; i < nc; ++i)
            for (int n = 0; n < ngh; ++n)
              ref(v, k, je+n+1, i) = sign * ref(v, k, je-n, i);
    }
    // inner x3
    if (loc.lx3 == 0 && mg_mesh_bcs_[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
      Real sign = (mg_mesh_bcs_[BoundaryFace::inner_x3] == BoundaryFlag::mg_zerofixed)
                  ? -1.0 : 1.0;
      for (int v = 0; v < nvar_; ++v)
        for (int j = 0; j < nc; ++j)
          for (int i = 0; i < nc; ++i)
            for (int n = 0; n < ngh; ++n)
              ref(v, ngh-1-n, j, i) = sign * ref(v, ngh+n, j, i);
    }
    // outer x3
    int maxlx3 = nrbx3_ << lev;
    if (loc.lx3 == maxlx3-1
        && mg_mesh_bcs_[BoundaryFace::outer_x3] != BoundaryFlag::periodic) {
      Real sign = (mg_mesh_bcs_[BoundaryFace::outer_x3] == BoundaryFlag::mg_zerofixed)
                  ? -1.0 : 1.0;
      int ke = fcbuf ? ngh : ngh + 1;
      for (int v = 0; v < nvar_; ++v)
        for (int j = 0; j < nc; ++j)
          for (int i = 0; i < nc; ++i)
            for (int n = 0; n < ngh; ++n)
              ref(v, ke+n+1, j, i) = sign * ref(v, ke-n, j, i);
    }
  };

  if (fcbuf) {
    int nc_cbuf = 1 + 2*ngh;
    apply_bc(cbuf_.data(), nc_cbuf);
  } else {
    apply_bc(oct.u, oct.nc);
  }
}


void MultigridDriver::MGRootBoundary() {
  SyncRootToHost();
  auto u = mgroot_->GetCurrentData_h();
  int nvar = u.extent_int(1);
  int current_level = mgroot_->GetCurrentLevel();
  int nlevels = mgroot_->GetNumberOfLevels();
  int ngh = mgroot_->ngh_;

  int ll = nlevels - 1 - current_level;
  int nx = (mgroot_->indcs_.nx1 >> ll) + 2*ngh;
  int ny = (mgroot_->indcs_.nx2 >> ll) + 2*ngh;
  int nz = (mgroot_->indcs_.nx3 >> ll) + 2*ngh;

  BoundaryFlag bc_ix1 = mg_mesh_bcs_[BoundaryFace::inner_x1];
  BoundaryFlag bc_ox1 = mg_mesh_bcs_[BoundaryFace::outer_x1];
  BoundaryFlag bc_ix2 = mg_mesh_bcs_[BoundaryFace::inner_x2];
  BoundaryFlag bc_ox2 = mg_mesh_bcs_[BoundaryFace::outer_x2];
  BoundaryFlag bc_ix3 = mg_mesh_bcs_[BoundaryFace::inner_x3];
  BoundaryFlag bc_ox3 = mg_mesh_bcs_[BoundaryFace::outer_x3];

  for (int v = 0; v < nvar; ++v) {
    // x1 boundaries
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int n = 0; n < ngh; ++n) {
          if (bc_ix1 == BoundaryFlag::periodic) {
            u(0, v, k, j, n) = u(0, v, k, j, nx - 2*ngh + n);
          } else if (bc_ix1 == BoundaryFlag::mg_zerofixed) {
            u(0, v, k, j, ngh - 1 - n) = -u(0, v, k, j, ngh + n);
          } else if (bc_ix1 == BoundaryFlag::mg_zerograd) {
            u(0, v, k, j, ngh - 1 - n) = u(0, v, k, j, ngh + n);
          }
          if (bc_ox1 == BoundaryFlag::periodic) {
            u(0, v, k, j, nx - ngh + n) = u(0, v, k, j, ngh + n);
          } else if (bc_ox1 == BoundaryFlag::mg_zerofixed) {
            u(0, v, k, j, nx - ngh + n) = -u(0, v, k, j, nx - ngh - 1 - n);
          } else if (bc_ox1 == BoundaryFlag::mg_zerograd) {
            u(0, v, k, j, nx - ngh + n) = u(0, v, k, j, nx - ngh - 1 - n);
          }
        }
      }
    }
    // x2 boundaries
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i < nx; ++i) {
        for (int n = 0; n < ngh; ++n) {
          if (bc_ix2 == BoundaryFlag::periodic) {
            u(0, v, k, n, i) = u(0, v, k, ny - 2*ngh + n, i);
          } else if (bc_ix2 == BoundaryFlag::mg_zerofixed) {
            u(0, v, k, ngh - 1 - n, i) = -u(0, v, k, ngh + n, i);
          } else if (bc_ix2 == BoundaryFlag::mg_zerograd) {
            u(0, v, k, ngh - 1 - n, i) = u(0, v, k, ngh + n, i);
          }
          if (bc_ox2 == BoundaryFlag::periodic) {
            u(0, v, k, ny - ngh + n, i) = u(0, v, k, ngh + n, i);
          } else if (bc_ox2 == BoundaryFlag::mg_zerofixed) {
            u(0, v, k, ny - ngh + n, i) = -u(0, v, k, ny - ngh - 1 - n, i);
          } else if (bc_ox2 == BoundaryFlag::mg_zerograd) {
            u(0, v, k, ny - ngh + n, i) = u(0, v, k, ny - ngh - 1 - n, i);
          }
        }
      }
    }
    // x3 boundaries
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        for (int n = 0; n < ngh; ++n) {
          if (bc_ix3 == BoundaryFlag::periodic) {
            u(0, v, n, j, i) = u(0, v, nz - 2*ngh + n, j, i);
          } else if (bc_ix3 == BoundaryFlag::mg_zerofixed) {
            u(0, v, ngh - 1 - n, j, i) = -u(0, v, ngh + n, j, i);
          } else if (bc_ix3 == BoundaryFlag::mg_zerograd) {
            u(0, v, ngh - 1 - n, j, i) = u(0, v, ngh + n, j, i);
          }
          if (bc_ox3 == BoundaryFlag::periodic) {
            u(0, v, nz - ngh + n, j, i) = u(0, v, ngh + n, j, i);
          } else if (bc_ox3 == BoundaryFlag::mg_zerofixed) {
            u(0, v, nz - ngh + n, j, i) = -u(0, v, nz - ngh - 1 - n, j, i);
          } else if (bc_ox3 == BoundaryFlag::mg_zerograd) {
            u(0, v, nz - ngh + n, j, i) = u(0, v, nz - ngh - 1 - n, j, i);
          }
        }
      }
    }
  }

  // Multipole expansion boundaries (applied after periodic/zerofixed/zerograd)
  if (mporder_ > 0) {
    int ncx = (mgroot_->indcs_.nx1 >> ll);
    int ncy = (mgroot_->indcs_.nx2 >> ll);
    int ncz = (mgroot_->indcs_.nx3 >> ll);
    Real x1min = pmy_mesh_->mesh_size.x1min;
    Real x1max = pmy_mesh_->mesh_size.x1max;
    Real x2min = pmy_mesh_->mesh_size.x2min;
    Real x2max = pmy_mesh_->mesh_size.x2max;
    Real x3min = pmy_mesh_->mesh_size.x3min;
    Real x3max = pmy_mesh_->mesh_size.x3max;
    Real dx1 = (x1max - x1min) / static_cast<Real>(ncx);
    Real dx2 = (x2max - x2min) / static_cast<Real>(ncy);
    Real dx3 = (x3max - x3min) / static_cast<Real>(ncz);
    Real xo = mpo_[0], yo = mpo_[1], zo = mpo_[2];
    int order = mporder_;

    auto eval_phi = [&](Real x, Real y, Real z) -> Real {
      Real x2 = x*x, y2 = y*y, z2 = z*z;
      Real xy = x*y, yz = y*z, zx = z*x;
      Real r2 = x2 + y2 + z2;
      Real ir2 = 1.0/r2, ir1 = std::sqrt(ir2);
      Real ir3 = ir2*ir1, ir5 = ir3*ir2;
      Real hx2my2 = 0.5*(x2-y2);
      Real phis = ir1*mpcoeff_[0]
        + ir3*(mpcoeff_[1]*y + mpcoeff_[2]*z + mpcoeff_[3]*x)
        + ir5*(mpcoeff_[4]*xy + mpcoeff_[5]*yz + (3.0*z2-r2)*mpcoeff_[6]
             + mpcoeff_[7]*zx + mpcoeff_[8]*hx2my2);
      if (order == 4) {
        Real ir7 = ir5*ir2, ir9 = ir7*ir2;
        Real x2mty2 = x2-3.0*y2;
        Real tx2my2 = 3.0*x2-y2;
        phis += ir7*(y*tx2my2*mpcoeff_[9] + x*x2mty2*mpcoeff_[15]
                   + xy*z*mpcoeff_[10] + z*hx2my2*mpcoeff_[14]
                   + (5.0*z2-r2)*(y*mpcoeff_[11] + x*mpcoeff_[13])
                   + z*(z2-3.0*r2)*mpcoeff_[12])
             + ir9*(xy*hx2my2*mpcoeff_[16]
                   + 0.125*(x2*x2mty2-y2*tx2my2)*mpcoeff_[24]
                   + yz*tx2my2*mpcoeff_[17] + zx*x2mty2*mpcoeff_[23]
                   + (7.0*z2-r2)*(xy*mpcoeff_[18] + hx2my2*mpcoeff_[22])
                   + (7.0*z2-3.0*r2)*(yz*mpcoeff_[19] + zx*mpcoeff_[21])
                   + (35.0*z2*z2-30.0*z2*r2+3.0*r2*r2)*mpcoeff_[20]);
      }
      return phis;
    };

    // Inner x1
    if (bc_ix1 == BoundaryFlag::mg_multipole) {
      Real x = x1min - xo;
      for (int k = ngh; k < ngh + ncz; ++k) {
        Real z = x3min + (k - ngh + 0.5)*dx3 - zo;
        for (int j = ngh; j < ngh + ncy; ++j) {
          Real y = x2min + (j - ngh + 0.5)*dx2 - yo;
          Real phis = eval_phi(x, y, z);
          for (int n = 0; n < ngh; ++n)
            u(0, 0, k, j, ngh - 1 - n) = 2.0*phis - u(0, 0, k, j, ngh + n);
        }
      }
    }
    // Outer x1
    if (bc_ox1 == BoundaryFlag::mg_multipole) {
      Real x = x1max - xo;
      for (int k = ngh; k < ngh + ncz; ++k) {
        Real z = x3min + (k - ngh + 0.5)*dx3 - zo;
        for (int j = ngh; j < ngh + ncy; ++j) {
          Real y = x2min + (j - ngh + 0.5)*dx2 - yo;
          Real phis = eval_phi(x, y, z);
          for (int n = 0; n < ngh; ++n)
            u(0, 0, k, j, ngh + ncx + n) = 2.0*phis - u(0, 0, k, j, ngh + ncx - 1 - n);
        }
      }
    }
    // Inner x2
    if (bc_ix2 == BoundaryFlag::mg_multipole) {
      Real y = x2min - yo;
      for (int k = ngh; k < ngh + ncz; ++k) {
        Real z = x3min + (k - ngh + 0.5)*dx3 - zo;
        for (int i = ngh; i < ngh + ncx; ++i) {
          Real x = x1min + (i - ngh + 0.5)*dx1 - xo;
          Real phis = eval_phi(x, y, z);
          for (int n = 0; n < ngh; ++n)
            u(0, 0, k, ngh - 1 - n, i) = 2.0*phis - u(0, 0, k, ngh + n, i);
        }
      }
    }
    // Outer x2
    if (bc_ox2 == BoundaryFlag::mg_multipole) {
      Real y = x2max - yo;
      for (int k = ngh; k < ngh + ncz; ++k) {
        Real z = x3min + (k - ngh + 0.5)*dx3 - zo;
        for (int i = ngh; i < ngh + ncx; ++i) {
          Real x = x1min + (i - ngh + 0.5)*dx1 - xo;
          Real phis = eval_phi(x, y, z);
          for (int n = 0; n < ngh; ++n)
            u(0, 0, k, ngh + ncy + n, i) = 2.0*phis - u(0, 0, k, ngh + ncy - 1 - n, i);
        }
      }
    }
    // Inner x3
    if (bc_ix3 == BoundaryFlag::mg_multipole) {
      Real z = x3min - zo;
      for (int j = ngh; j < ngh + ncy; ++j) {
        Real y = x2min + (j - ngh + 0.5)*dx2 - yo;
        for (int i = ngh; i < ngh + ncx; ++i) {
          Real x = x1min + (i - ngh + 0.5)*dx1 - xo;
          Real phis = eval_phi(x, y, z);
          for (int n = 0; n < ngh; ++n)
            u(0, 0, ngh - 1 - n, j, i) = 2.0*phis - u(0, 0, ngh + n, j, i);
        }
      }
    }
    // Outer x3
    if (bc_ox3 == BoundaryFlag::mg_multipole) {
      Real z = x3max - zo;
      for (int j = ngh; j < ngh + ncy; ++j) {
        Real y = x2min + (j - ngh + 0.5)*dx2 - yo;
        for (int i = ngh; i < ngh + ncx; ++i) {
          Real x = x1min + (i - ngh + 0.5)*dx1 - xo;
          Real phis = eval_phi(x, y, z);
          for (int n = 0; n < ngh; ++n)
            u(0, 0, ngh + ncz + n, j, i) = 2.0*phis - u(0, 0, ngh + ncz - 1 - n, j, i);
        }
      }
    }
  }

  root_flat_buf_stale_ = true;
  root_sync_state_ = RootSyncState::HOST_MODIFIED;
  SyncRootToDevice();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::AllocateMultipoleCoefficients()
//! \brief Set nmpcoeff_ based on mporder_

void MultigridDriver::AllocateMultipoleCoefficients() {
  nmpcoeff_ = 0;
  if (mporder_ <= 0) return;
  for (int l = 0; l <= mporder_; ++l)
    nmpcoeff_ += 2 * l + 1;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CalculateMultipoleCoefficients()
//! \brief Compute multipole moments by integrating source * solid harmonics over all MBs

void MultigridDriver::CalculateMultipoleCoefficients() {
  if (mporder_ <= 0 || nmpcoeff_ == 0) return;
  std::memset(mpcoeff_, 0, sizeof(Real) * nmpcoeff_);

  auto src = mglevels_->src_[mglevels_->nlevel_-1].d_view;
  int nmb = mglevels_->nmmb_;
  int ngh = mglevels_->ngh_;
  int nx1 = mglevels_->indcs_.nx1;
  int nx2 = mglevels_->indcs_.nx2;
  int nx3 = mglevels_->indcs_.nx3;
  auto &mb_size = pmy_pack_->pmb->mb_size;

  Real xorigin = mpo_[0], yorigin = mpo_[1], zorigin = mpo_[2];
  int order = mporder_;
  bool skip_dipole = nodipole_;
  int ncoeff = nmpcoeff_;

  DvceArray2D<Real> partial("mp_partial", nmb, nmpcoeff_);

  par_for("MGMultipoleCoeffs", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(const int m) {
    Real dx1 = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
               / static_cast<Real>(nx1);
    Real dx2 = (mb_size.d_view(m).x2max - mb_size.d_view(m).x2min)
               / static_cast<Real>(nx2);
    Real dx3 = (mb_size.d_view(m).x3max - mb_size.d_view(m).x3min)
               / static_cast<Real>(nx3);
    Real vol = dx1 * dx2 * dx3;

    Real mp[25] = {};

    for (int k = ngh; k < ngh + nx3; ++k) {
      Real z = mb_size.d_view(m).x3min + (k - ngh + 0.5) * dx3 - zorigin;
      Real z2 = z * z;
      for (int j = ngh; j < ngh + nx2; ++j) {
        Real y = mb_size.d_view(m).x2min + (j - ngh + 0.5) * dx2 - yorigin;
        Real y2 = y * y;
        Real yz = y * z;
        for (int i = ngh; i < ngh + nx1; ++i) {
          Real x = mb_size.d_view(m).x1min + (i - ngh + 0.5) * dx1 - xorigin;
          Real x2 = x * x;
          Real xy = x * y;
          Real zx = z * x;
          Real r2 = x2 + y2 + z2;
          Real s = src(m, 0, k, j, i) * vol;

          mp[0] += s;
          if (!skip_dipole) {
            mp[1] += s * y;
            mp[2] += s * z;
            mp[3] += s * x;
          }
          Real hx2my2 = 0.5 * (x2 - y2);
          mp[4] += s * xy;
          mp[5] += s * yz;
          mp[6] += s * (3.0 * z2 - r2);
          mp[7] += s * zx;
          mp[8] += s * hx2my2;

          if (order == 4) {
            Real tx2my2 = 3.0 * x2 - y2;
            Real x2mty2 = x2 - 3.0 * y2;
            Real fz2mr2 = 5.0 * z2 - r2;
            mp[9]  += s * y * tx2my2;
            mp[10] += s * xy * z;
            mp[11] += s * y * fz2mr2;
            mp[12] += s * z * (z2 - 3.0 * r2);
            mp[13] += s * x * fz2mr2;
            mp[14] += s * z * hx2my2;
            mp[15] += s * x * x2mty2;
            Real sz2mr2 = 7.0 * z2 - r2;
            Real sz2mtr2 = 7.0 * z2 - 3.0 * r2;
            mp[16] += s * xy * hx2my2;
            mp[17] += s * yz * tx2my2;
            mp[18] += s * xy * sz2mr2;
            mp[19] += s * yz * sz2mtr2;
            mp[20] += s * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r2 * r2);
            mp[21] += s * zx * sz2mtr2;
            mp[22] += s * hx2my2 * sz2mr2;
            mp[23] += s * zx * x2mty2;
            mp[24] += s * 0.125 * (x2 * x2mty2 - y2 * tx2my2);
          }
        }
      }
    }
    for (int c = 0; c < ncoeff; ++c)
      partial(m, c) = mp[c];
  });

  auto partial_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), partial);
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < nmpcoeff_; ++c) {
      mpcoeff_[c] += partial_h(m, c);
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, mpcoeff_, nmpcoeff_, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif

  ScaleMultipoleCoefficients();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ScaleMultipoleCoefficients()
//! \brief Apply normalization to raw multipole moments

void MultigridDriver::ScaleMultipoleCoefficients() {
  constexpr Real c0  = -0.25 / M_PI;
  constexpr Real c1  = -0.25 / M_PI;
  constexpr Real c2  = -0.0625 / M_PI;
  constexpr Real c2a = -0.75 / M_PI;
  constexpr Real c30 = -0.0625 / M_PI;
  constexpr Real c31 = -0.0625 * 1.5 / M_PI;
  constexpr Real c32 = -0.25 * 15.0 / M_PI;
  constexpr Real c33 = -0.0625 * 2.5 / M_PI;
  constexpr Real c40 = -0.0625 * 0.0625 / M_PI;
  constexpr Real c41 = -0.0625 * 2.5 / M_PI;
  constexpr Real c42 = -0.0625 * 5.0 / M_PI;
  constexpr Real c43 = -0.0625 * 17.5 / M_PI;
  constexpr Real c44 = -0.25 * 35.0 / M_PI;

  mpcoeff_[0] *= c0;
  mpcoeff_[1] *= c1;
  mpcoeff_[2] *= c1;
  mpcoeff_[3] *= c1;
  mpcoeff_[4] *= c2a;
  mpcoeff_[5] *= c2a;
  mpcoeff_[6] *= c2;
  mpcoeff_[7] *= c2a;
  mpcoeff_[8] *= c2a;
  if (mporder_ == 4) {
    mpcoeff_[9]  *= c33;
    mpcoeff_[10] *= c32;
    mpcoeff_[11] *= c31;
    mpcoeff_[12] *= c30;
    mpcoeff_[13] *= c31;
    mpcoeff_[14] *= c32;
    mpcoeff_[15] *= c33;
    mpcoeff_[16] *= c44;
    mpcoeff_[17] *= c43;
    mpcoeff_[18] *= c42;
    mpcoeff_[19] *= c41;
    mpcoeff_[20] *= c40;
    mpcoeff_[21] *= c41;
    mpcoeff_[22] *= c42;
    mpcoeff_[23] *= c43;
    mpcoeff_[24] *= c44;
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CalculateCenterOfMass()
//! \brief Compute center of mass from monopole and dipole moments

void MultigridDriver::CalculateCenterOfMass() {
  if (mporder_ <= 0) return;

  auto src = mglevels_->src_[mglevels_->nlevel_-1].d_view;
  int nmb = mglevels_->nmmb_;
  int ngh = mglevels_->ngh_;
  int nx1 = mglevels_->indcs_.nx1;
  int nx2 = mglevels_->indcs_.nx2;
  int nx3 = mglevels_->indcs_.nx3;
  auto &mb_size = pmy_pack_->pmb->mb_size;

  DvceArray2D<Real> partial("com_partial", nmb, 4);

  par_for("MGCenterOfMass", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(const int m) {
    Real dx1 = (mb_size.d_view(m).x1max - mb_size.d_view(m).x1min)
               / static_cast<Real>(nx1);
    Real dx2 = (mb_size.d_view(m).x2max - mb_size.d_view(m).x2min)
               / static_cast<Real>(nx2);
    Real dx3 = (mb_size.d_view(m).x3max - mb_size.d_view(m).x3min)
               / static_cast<Real>(nx3);
    Real vol = dx1 * dx2 * dx3;
    Real m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0;
    for (int k = ngh; k < ngh + nx3; ++k) {
      Real z = mb_size.d_view(m).x3min + (k - ngh + 0.5) * dx3;
      for (int j = ngh; j < ngh + nx2; ++j) {
        Real y = mb_size.d_view(m).x2min + (j - ngh + 0.5) * dx2;
        for (int i = ngh; i < ngh + nx1; ++i) {
          Real x = mb_size.d_view(m).x1min + (i - ngh + 0.5) * dx1;
          Real s = src(m, 0, k, j, i) * vol;
          m0 += s;
          m1 += s * y;
          m2 += s * z;
          m3 += s * x;
        }
      }
    }
    partial(m, 0) = m0;
    partial(m, 1) = m1;
    partial(m, 2) = m2;
    partial(m, 3) = m3;
  });

  auto partial_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), partial);
  Real totals[4] = {};
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < 4; ++c) {
      totals[c] += partial_h(m, c);
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, totals, 4, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif

  Real im = 1.0 / totals[0];
  mpo_[0] = im * totals[3];  // x
  mpo_[1] = im * totals[1];  // y
  mpo_[2] = im * totals[2];  // z
}
