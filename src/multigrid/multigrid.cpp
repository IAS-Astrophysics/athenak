//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid.cpp
//! \brief implementation of the functions commonly used in Multigrid

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdlib>    // getenv
#include <cstring>    // memset, memcpy
#include <iostream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>    // setprecision 

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/nghbr_index.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

namespace {

bool MGDebugEnabled() {
  static bool enabled = (std::getenv("ATHENA_MG_DEBUG") != nullptr);
  return enabled;
}

void ReduceCorrectionStats(Real local_sum2, Real local_max, long long local_count,
                           Real &rms, Real &maxabs) {
  Real global_sum2 = local_sum2;
  Real global_max = local_max;
  long long global_count = local_count;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_sum2, &global_sum2, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  rms = (global_count > 0) ? std::sqrt(global_sum2/static_cast<Real>(global_count)) : 0.0;
  maxabs = global_max;
}

KOKKOS_INLINE_FUNCTION
void DecodeNeighborIndexMG(const int idx, int &ox1, int &ox2, int &ox3,
                           int &f1, int &f2) {
  ox1 = ox2 = ox3 = 0;
  f1 = f2 = 0;
  for (int iz = -1; iz <= 1; ++iz) {
    for (int iy = -1; iy <= 1; ++iy) {
      for (int ix = -1; ix <= 1; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) continue;
        for (int sf2 = 0; sf2 <= 1; ++sf2) {
          for (int sf1 = 0; sf1 <= 1; ++sf1) {
            if (NeighborIndex(ix, iy, iz, sf1, sf2) == idx) {
              ox1 = ix;
              ox2 = iy;
              ox3 = iz;
              f1 = sf1;
              f2 = sf2;
              return;
            }
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void AdjustSendRangeForMG(int &il, int &iu, int &jl, int &ju, int &kl, int &ku,
                          const MeshBoundaryBuffer &buf, int shift, int nx,
                          int ngh) {
  while (shift > 0) {
    if (buf.faces.d_view(0) && (il == nx)) {
      int d = iu - il; il = il >> 1; iu = il + d;
    } else if (!buf.faces.d_view(0)) {
      il = ngh + ((il - ngh) >> 1);
      iu = ngh + ((iu - ngh) >> 1);
    }
    if (buf.faces.d_view(1) && (jl == nx)) {
      int d = ju - jl; jl = jl >> 1; ju = jl + d;
    } else if (!buf.faces.d_view(1)) {
      jl = ngh + ((jl - ngh) >> 1);
      ju = ngh + ((ju - ngh) >> 1);
    }
    if (buf.faces.d_view(2) && (kl == nx)) {
      int d = ku - kl; kl = kl >> 1; ku = kl + d;
    } else if (!buf.faces.d_view(2)) {
      kl = ngh + ((kl - ngh) >> 1);
      ku = ngh + ((ku - ngh) >> 1);
    }
    --shift;
    nx = nx >> 1;
  }
}

KOKKOS_INLINE_FUNCTION
void AdjustRecvRangeForMG(int &il, int &iu, int &jl, int &ju, int &kl, int &ku,
                          const MeshBoundaryBuffer &buf, int shift, int ngh) {
  while (shift > 0) {
    if (buf.faces.d_view(0) && il > 1) {
      int d = iu - il; il = (il + ngh) >> 1; iu = il + d;
    } else if (!buf.faces.d_view(0)) {
      il = ngh + ((il - ngh) >> 1);
      iu = ngh + ((iu - ngh) >> 1);
    }
    if (buf.faces.d_view(1) && jl > 1) {
      int d = ju - jl; jl = (jl + ngh) >> 1; ju = jl + d;
    } else if (!buf.faces.d_view(1)) {
      jl = ngh + ((jl - ngh) >> 1);
      ju = ngh + ((ju - ngh) >> 1);
    }
    if (buf.faces.d_view(2) && kl > 1) {
      int d = ku - kl; kl = (kl + ngh) >> 1; ku = kl + d;
    } else if (!buf.faces.d_view(2)) {
      kl = ngh + ((kl - ngh) >> 1);
      ku = ngh + ((ku - ngh) >> 1);
    }
    --shift;
  }
}

inline int AdjustMGBufferSizeHost(const MeshBoundaryBuffer &buf, int ndat,
                                  int shift) {
  int size = ndat;
  if (!buf.faces.h_view(0)) size >>= shift;
  if (!buf.faces.h_view(1)) size >>= shift;
  if (!buf.faces.h_view(2)) size >>= shift;
  return size;
}

KOKKOS_INLINE_FUNCTION
Real ReadMGStage(const MeshBoundaryBuffer &buf, int m, int v,
                 int k, int j, int i, int il, int iu,
                 int jl, int ju, int kl, int ku) {
  if (i < il) i = il;
  if (i > iu) i = iu;
  if (j < jl) j = jl;
  if (j > ju) j = ju;
  if (k < kl) k = kl;
  if (k > ku) k = ku;
  int ni = iu - il + 1;
  int nj = ju - jl + 1;
  int nk = ku - kl + 1;
  return buf.vars(m, i - il + ni*(j - jl + nj*(k - kl + nk*v)));
}

} // namespace

//namespace multigrid{ // NOLINT (build/namespace)
//----------------------------------------------------------------------------------------
//! \fn Multigrid::Multigrid(MultigridDriver *pmd, MeshBlock *pmb, int nghost)
//  \brief Multigrid constructor

Multigrid::Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host):
  pmy_driver_(pmd), pmy_pack_(pmbp), pmy_mesh_(pmd->pmy_mesh_), ngh_(nghost),
  nvar_(pmd->nvar_), ncoeff_(pmd->ncoeff_), nmatrix_(pmd->nmatrix_),
  defscale_(1.0), on_host_(on_host)  {
  if(pmy_pack_ != nullptr) {
    //Meshblock levels
    indcs_ = pmy_mesh_->mb_indcs;
    nmmb_  = pmy_pack_->nmb_thispack;
    nmmbx1_ = pmy_mesh_->nmb_rootx1;
    nmmbx2_ = pmy_mesh_->nmb_rootx2;
    nmmbx3_ = pmy_mesh_->nmb_rootx3;
    std::cout<< "Number of MeshBlocks in the pack: " << nmmb_ << std::endl;
    std::cout<< "MeshBlock size: "
             << indcs_.nx1 << " x " << indcs_.nx2 << " x " << indcs_.nx3 << std::endl;
    if (indcs_.nx1 != indcs_.nx2 || indcs_.nx1 != indcs_.nx3) {
      std::cout << "### FATAL ERROR in Multigrid::Multigrid" << std::endl
         << "The Multigrid solver requires logically cubic MeshBlock." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
     }
    
     // initialize loc/size from the first meshblock in the pack (needs to be addpated for AMR)
    loc_ = pmy_pack_->pmesh->lloc_eachmb[0];
    size_ = pmy_pack_->pmb->mb_size.h_view(0);
  } else {
    //Root levels
    indcs_.nx1 = pmy_mesh_->nmb_rootx1;
    indcs_.nx2 = pmy_mesh_->nmb_rootx2;
    indcs_.nx3 = pmy_mesh_->nmb_rootx3;
    size_ = pmy_mesh_->mesh_size;
    nmmbx1_ = 1;
    nmmbx2_ = 1;
    nmmbx3_ = 1;
    // Root grid should be a single meshblock
    nmmb_ = 1;
    loc_  = pmy_mesh_->lloc_eachmb[0];
  }

  rdx_ = (size_.x1max-size_.x1min)/static_cast<Real>(indcs_.nx1);
  rdy_ = (size_.x2max-size_.x2min)/static_cast<Real>(indcs_.nx2);
  rdz_ = (size_.x3max-size_.x3min)/static_cast<Real>(indcs_.nx3);

  block_rdx_ = DualArray1D<Real>("block_rdx", nmmb_);
  Kokkos::realloc(fc_childx_, nmmb_);
  Kokkos::realloc(fc_childy_, nmmb_);
  Kokkos::realloc(fc_childz_, nmmb_);
  {
    auto brdx_h = block_rdx_.h_view;
    if (pmy_pack_ != nullptr) {
      auto &mb_size = pmy_pack_->pmb->mb_size;
      Real rnx1 = static_cast<Real>(indcs_.nx1);
      for (int m = 0; m < nmmb_; ++m) {
        brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
      }
    } else {
      brdx_h(0) = rdx_;
    }
    Kokkos::deep_copy(block_rdx_.d_view, brdx_h);
  }

  nlevel_ = 0;
  if (pmy_pack_ == nullptr) { 
    // Root grid levels
    int nbx = 0, nby = 0, nbz = 0;
    for (int l = 0; l < 20; l++) {
      if (indcs_.nx1%(1<<l) == 0 && indcs_.nx2%(1<<l) == 0 && indcs_.nx3%(1<<l) == 0) {
        nbx = indcs_.nx1/(1<<l), nby = indcs_.nx2/(1<<l), nbz = indcs_.nx3/(1<<l);
        nlevel_ = l+1;
      }
    }
    int nmaxr = std::max(nbx, std::max(nby, nbz));
    std::cout<< "Multigrid root grid levels: " << nlevel_ << std::endl;
    // int nminr=std::min(nbx, std::min(nby, nbz)); // unused variable
    if (nmaxr != 1 && global_variable::my_rank == 0) {
      std::cout
          << "### Warning in Multigrid::Multigrid" << std::endl
          << "The root grid can not be reduced to a single cell." << std::endl
          << "Multigrid should still work, but this is not the"
          << " most efficient configuration"
          << " as the coarsest level is not solved exactly but iteratively." << std::endl;
    }
    if (nbx*nby*nbz>100 && global_variable::my_rank==0) {
      std::cout << "### Warning in Multigrid::Multigrid" << std::endl
                << "The degrees of freedom on the coarsest level is very large: "
                << nbx << " x " << nby << " x " << nbz << " = " << nbx*nby*nbz<< std::endl
                << "Multigrid should still work, but this is not efficient configuration "
                << "as the coarsest level solver costs considerably." << std::endl
                << "We recommend to reconsider grid configuration." << std::endl;
    }
  } else {
    // MeshBlock levels
    for (int l = 0; l < 20; l++) {
      if ((1<<l) == indcs_.nx1) {
        nlevel_=l+1;
        break;
      }
    }
    if (nlevel_ == 0) {
      std::cout << "### FATAL ERROR in Multigrid::Multigrid" << std::endl
          << "The MeshBlock size must be power of two." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
    }
  }

  current_level_ = nlevel_-1;

  // allocate arrays
  u_ = new DualArray5D<Real>[nlevel_];
  src_ = new DualArray5D<Real>[nlevel_];
  def_ = new DualArray5D<Real>[nlevel_];
  coeff_ = new DualArray5D<Real>[nlevel_];
  matrix_ = new DualArray5D<Real>[nlevel_];
  uold_ = new DualArray5D<Real>[nlevel_];
  comp_mask_ = new DualArray5D<int>[nlevel_];

  for (int l = 0; l < nlevel_; l++) {
    int ll=nlevel_-1-l;
    int ncx=(indcs_.nx1>>ll)+2*ngh_;
    int ncy=(indcs_.nx2>>ll)+2*ngh_;
    int ncz=(indcs_.nx3>>ll)+2*ngh_;
    Kokkos::realloc(u_[l]  , nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(src_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(def_[l], nmmb_, nvar_, ncz, ncy, ncx);
    if (ncoeff_ > 0) {
      Kokkos::realloc(coeff_[l], nmmb_, ncoeff_, ncz, ncy, ncx);
    }
    if (nmatrix_ > 0) {
      Kokkos::realloc(matrix_[l], nmmb_, nmatrix_, ncz, ncy, ncx);
    }

    if (!((pmy_pack_ != nullptr) && (l == nlevel_-1)))
      Kokkos::realloc(uold_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(comp_mask_[l], nmmb_, COMP_NMASK, ncz, ncy, ncx);

    ncx=(indcs_.nx1>>(ll+1))+2*ngh_;
    ncy=(indcs_.nx2>>(ll+1))+2*ngh_;
    ncz=(indcs_.nx3>>(ll+1))+2*ngh_;

  }

}


//----------------------------------------------------------------------------------------
//! \fn Multigrid::~Multigrid
//! \brief Multigrid destroctor

Multigrid::~Multigrid() {
  delete [] u_;
  delete [] src_;
  delete [] def_;
  delete [] uold_;
  delete [] coeff_;
  delete [] matrix_;
  delete [] comp_mask_;
  delete [] coord_;
  delete [] ccoord_;
}



//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ReallocateForAMR()
//! \brief Reallocate MG arrays when the number of MeshBlocks changes (AMR)

void Multigrid::UpdateBlockDx() {
  if (pmy_pack_ == nullptr) return;
  auto &mb_size = pmy_pack_->pmb->mb_size;
  Real rnx1 = static_cast<Real>(indcs_.nx1);

  // Refresh reference block size and cell widths from the first MeshBlock
  size_ = mb_size.h_view(0);
  rdx_ = (size_.x1max - size_.x1min) / static_cast<Real>(indcs_.nx1);
  rdy_ = (size_.x2max - size_.x2min) / static_cast<Real>(indcs_.nx2);
  rdz_ = (size_.x3max - size_.x3min) / static_cast<Real>(indcs_.nx3);

  auto brdx_h = block_rdx_.h_view;
  for (int m = 0; m < nmmb_; ++m) {
    brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
  }
  Kokkos::deep_copy(block_rdx_.d_view, brdx_h);

  // Compute per-block child octant positions for FC ghost fills
  auto &mbgid = pmy_pack_->pmb->mb_gid;
  auto *lloc = pmy_mesh_->lloc_eachmb;
  int root_level = pmy_mesh_->root_level;
  auto cx_h = Kokkos::create_mirror_view(fc_childx_);
  auto cy_h = Kokkos::create_mirror_view(fc_childy_);
  auto cz_h = Kokkos::create_mirror_view(fc_childz_);
  for (int m = 0; m < nmmb_; ++m) {
    int gid = mbgid.h_view(m);
    LogicalLocation &loc = lloc[gid];
    cx_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx1 & 1) : 0;
    cy_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx2 & 1) : 0;
    cz_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx3 & 1) : 0;
  }
  Kokkos::deep_copy(fc_childx_, cx_h);
  Kokkos::deep_copy(fc_childy_, cy_h);
  Kokkos::deep_copy(fc_childz_, cz_h);
}

void Multigrid::ReallocateForAMR() {
  if (pmy_pack_ == nullptr) return;
  int new_nmmb = pmy_pack_->nmb_thispack;
  if (new_nmmb == nmmb_) return;
  nmmb_ = new_nmmb;

  Kokkos::realloc(block_rdx_, nmmb_);
  Kokkos::realloc(fc_childx_, nmmb_);
  Kokkos::realloc(fc_childy_, nmmb_);
  Kokkos::realloc(fc_childz_, nmmb_);
  UpdateBlockDx();

  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_ - 1 - l;
    int ncx = (indcs_.nx1 >> ll) + 2 * ngh_;
    int ncy = (indcs_.nx2 >> ll) + 2 * ngh_;
    int ncz = (indcs_.nx3 >> ll) + 2 * ngh_;
    Kokkos::realloc(u_[l],   nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(src_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(def_[l], nmmb_, nvar_, ncz, ncy, ncx);
    if (ncoeff_ > 0) {
      Kokkos::realloc(coeff_[l], nmmb_, ncoeff_, ncz, ncy, ncx);
    }
    if (nmatrix_ > 0) {
      Kokkos::realloc(matrix_[l], nmmb_, nmatrix_, ncz, ncy, ncx);
    }
    if (l != nlevel_ - 1)
      Kokkos::realloc(uold_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(comp_mask_[l], nmmb_, COMP_NMASK, ncz, ncy, ncx);
  }

}

void Multigrid::ClearCompositeMasks() {
  for (int l = 0; l < nlevel_; ++l) {
    Kokkos::deep_copy(comp_mask_[l].h_view, 0);
    comp_mask_[l].template modify<HostExeSpace>();
    comp_mask_[l].template sync<DevExeSpace>();
  }
}

CompositeMaskCounts Multigrid::CountCompositeMasks(int level, bool active_only) const {
  CompositeMaskCounts counts;
  if (level < 0 || level >= nlevel_) return counts;
  const auto mask_h = comp_mask_[level].h_view;
  int il = 0, iu = mask_h.extent_int(4) - 1;
  int jl = 0, ju = mask_h.extent_int(3) - 1;
  int kl = 0, ku = mask_h.extent_int(2) - 1;
  if (active_only) {
    const int ncells = GetLevelActiveCells(level);
    il = jl = kl = ngh_;
    iu = il + ncells - 1;
    ju = jl + ncells - 1;
    ku = kl + ncells - 1;
  }
  for (int m = 0; m < nmmb_; ++m) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          counts.valid += mask_h(m, COMP_VALID, k, j, i);
          counts.relax += mask_h(m, COMP_RELAX, k, j, i);
          counts.covered += mask_h(m, COMP_COVERED, k, j, i);
          counts.interface += mask_h(m, COMP_INTERFACE, k, j, i);
        }
      }
    }
  }
  return counts;
}


//! \fn void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh)
//! \brief Fill the inital guess in the active zone of the finest level

void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh) {
  auto &dst = u_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1 - 1; je = js + indcs_.nx2 - 1; ke = ks + indcs_.nx3 - 1;

  const int lns = ns;
  const int lks = ks, ljs = js, lis = is, lngh = ngh;

  par_for("Multigrid::LoadFinestData", DevExeSpace(),0, nmmb_-1,
          0, nvar_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = lns + v;
    const int k = mk - lks + lngh;
    const int j = mj - ljs + lngh;
    const int i = mi - lis + lngh;
    dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i);
  });

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::LoadSource(const DvceArray5D<Real> &src, int ns, int ngh,
//!                                Real fac)
//! \brief Fill the source in the active zone of the finest level

void Multigrid::LoadSource(const DvceArray5D<Real> &src, int ns, int ngh, Real fac) {
  auto &dst = src_[nlevel_-1].d_view;
  int sngh = std::min(ngh_, ngh);
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_ - sngh;
  ie = is + indcs_.nx1 + 2*sngh - 1;
  je = js + indcs_.nx2 + 2*sngh - 1;
  ke = ks + indcs_.nx3 + 2*sngh - 1;

  // local copies for device lambda capture
  const Real lfac = fac;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int src_off = ngh - ngh_;

  par_for("Multigrid::LoadSource", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = ns + v;
    const int k = mk + src_off;
    const int j = mj + src_off;
    const int i = mi + src_off;
    if (lfac == (Real)1.0) {
      dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i);
    } else {
      dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i) * lfac;
    }
  });

  current_level_ = nlevel_-1;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::LoadCoefficients(const DvceArray5D<Real> &coeff, int ngh)
//! \brief Load coefficients of the diffusion and source terms

void Multigrid::LoadCoefficients(const DvceArray5D<Real> &coeff, int ngh) {
  auto &cm = coeff_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = 0;
  ie = indcs_.nx1 + 2*ngh_ - 1; je = indcs_.nx2 + 2*ngh_ - 1; ke = indcs_.nx3 + 2*ngh_ - 1;

  // copy locals for device lambda capture
  const int coeff_off = ngh - ngh_;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = ncoeff_ - 1;

  auto cm_ = cm;
  auto coeff_ = coeff;

  par_for("Multigrid::LoadCoefficients", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int k = mk + coeff_off;
    const int j = mj + coeff_off;
    const int i = mi + coeff_off;
    cm_(m, v, mk, mj, mi) = coeff_(m, v, k, j, i);
  });

  FillCoefficientBoundaries(nlevel_ - 1);
  return;
}

void Multigrid::FillCoefficientBoundaries(int level) {
  if (ncoeff_ <= 0) return;
  int saved_level = current_level_;
  current_level_ = level;
  int ll = nlevel_ - 1 - current_level_;
  int ncells = indcs_.nx1 >> ll;

  if (pbval != nullptr) {
    DvceArray5D<Real> coeff = coeff_[current_level_].d_view;
    (void)pbval->InitRecvMG(ncoeff_);
    (void)pbval->PackAndSendMG(coeff);
    Kokkos::fence();
    while (pbval->RecvAndUnpackMG(coeff) == TaskStatus::incomplete) {}
    Kokkos::fence();
    while (pbval->ClearSend() == TaskStatus::incomplete) {}
    while (pbval->ClearRecv() == TaskStatus::incomplete) {}
  }

  if (ncells < 1) {
    current_level_ = saved_level;
    return;
  }
  int nx = ncells + 2*ngh_;
  int ny = (indcs_.nx2 >> ll) + 2*ngh_;
  int nz = (indcs_.nx3 >> ll) + 2*ngh_;
  int ngh = ngh_;

  BoundaryFlag bc_ix1 = pmy_driver_->mg_mesh_bcs_[BoundaryFace::inner_x1];
  BoundaryFlag bc_ox1 = pmy_driver_->mg_mesh_bcs_[BoundaryFace::outer_x1];
  BoundaryFlag bc_ix2 = pmy_driver_->mg_mesh_bcs_[BoundaryFace::inner_x2];
  BoundaryFlag bc_ox2 = pmy_driver_->mg_mesh_bcs_[BoundaryFace::outer_x2];
  BoundaryFlag bc_ix3 = pmy_driver_->mg_mesh_bcs_[BoundaryFace::inner_x3];
  BoundaryFlag bc_ox3 = pmy_driver_->mg_mesh_bcs_[BoundaryFace::outer_x3];

  if (pmy_pack_ == nullptr) {
    auto fill_root = [&](auto coeff) {
      for (int v = 0; v < ncoeff_; ++v) {
        for (int k = 0; k < nz; ++k) {
          for (int j = 0; j < ny; ++j) {
            for (int n = 0; n < ngh; ++n) {
              coeff(0, v, k, j, ngh - 1 - n) =
                  (bc_ix1 == BoundaryFlag::periodic)
                  ? coeff(0, v, k, j, nx - 2*ngh + n)
                  : coeff(0, v, k, j, ngh + n);
              coeff(0, v, k, j, nx - ngh + n) =
                  (bc_ox1 == BoundaryFlag::periodic)
                  ? coeff(0, v, k, j, ngh + n)
                  : coeff(0, v, k, j, nx - ngh - 1 - n);
            }
          }
        }
        for (int k = 0; k < nz; ++k) {
          for (int i = 0; i < nx; ++i) {
            for (int n = 0; n < ngh; ++n) {
              coeff(0, v, k, ngh - 1 - n, i) =
                  (bc_ix2 == BoundaryFlag::periodic)
                  ? coeff(0, v, k, ny - 2*ngh + n, i)
                  : coeff(0, v, k, ngh + n, i);
              coeff(0, v, k, ny - ngh + n, i) =
                  (bc_ox2 == BoundaryFlag::periodic)
                  ? coeff(0, v, k, ngh + n, i)
                  : coeff(0, v, k, ny - ngh - 1 - n, i);
            }
          }
        }
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            for (int n = 0; n < ngh; ++n) {
              coeff(0, v, ngh - 1 - n, j, i) =
                  (bc_ix3 == BoundaryFlag::periodic)
                  ? coeff(0, v, nz - 2*ngh + n, j, i)
                  : coeff(0, v, ngh + n, j, i);
              coeff(0, v, nz - ngh + n, j, i) =
                  (bc_ox3 == BoundaryFlag::periodic)
                  ? coeff(0, v, ngh + n, j, i)
                  : coeff(0, v, nz - ngh - 1 - n, j, i);
            }
          }
        }
      }
    };
    if (on_host_) {
      fill_root(coeff_[current_level_].h_view);
      coeff_[current_level_].template modify<HostExeSpace>();
      coeff_[current_level_].template sync<DevExeSpace>();
    } else {
      auto coeff_d = coeff_[current_level_].d_view;
      Kokkos::parallel_for("MGCoeffRootBnd",
        Kokkos::RangePolicy<DevExeSpace>(0, 1), KOKKOS_LAMBDA(const int) {
          for (int v = 0; v < ncoeff_; ++v) {
            for (int k = 0; k < nz; ++k) {
              for (int j = 0; j < ny; ++j) {
                for (int n = 0; n < ngh; ++n) {
                  coeff_d(0, v, k, j, ngh - 1 - n) =
                      (bc_ix1 == BoundaryFlag::periodic)
                      ? coeff_d(0, v, k, j, nx - 2*ngh + n)
                      : coeff_d(0, v, k, j, ngh + n);
                  coeff_d(0, v, k, j, nx - ngh + n) =
                      (bc_ox1 == BoundaryFlag::periodic)
                      ? coeff_d(0, v, k, j, ngh + n)
                      : coeff_d(0, v, k, j, nx - ngh - 1 - n);
                }
              }
            }
            for (int k = 0; k < nz; ++k) {
              for (int i = 0; i < nx; ++i) {
                for (int n = 0; n < ngh; ++n) {
                  coeff_d(0, v, k, ngh - 1 - n, i) =
                      (bc_ix2 == BoundaryFlag::periodic)
                      ? coeff_d(0, v, k, ny - 2*ngh + n, i)
                      : coeff_d(0, v, k, ngh + n, i);
                  coeff_d(0, v, k, ny - ngh + n, i) =
                      (bc_ox2 == BoundaryFlag::periodic)
                      ? coeff_d(0, v, k, ngh + n, i)
                      : coeff_d(0, v, k, ny - ngh - 1 - n, i);
                }
              }
            }
            for (int j = 0; j < ny; ++j) {
              for (int i = 0; i < nx; ++i) {
                for (int n = 0; n < ngh; ++n) {
                  coeff_d(0, v, ngh - 1 - n, j, i) =
                      (bc_ix3 == BoundaryFlag::periodic)
                      ? coeff_d(0, v, nz - 2*ngh + n, j, i)
                      : coeff_d(0, v, ngh + n, j, i);
                  coeff_d(0, v, nz - ngh + n, j, i) =
                      (bc_ox3 == BoundaryFlag::periodic)
                      ? coeff_d(0, v, ngh + n, j, i)
                      : coeff_d(0, v, nz - ngh - 1 - n, j, i);
                }
              }
            }
          }
        });
    }
    current_level_ = saved_level;
    return;
  }

  if (!pmy_mesh_->strictly_periodic) {
    auto coeff = coeff_[current_level_].d_view;
    auto &mb_bcs = pmy_pack_->pmb->mb_bcs;
    int nmb = pmy_pack_->nmb_thispack;
    Kokkos::parallel_for("MGCoeffPhysicalBoundary",
      Kokkos::RangePolicy<DevExeSpace>(0, nmb), KOKKOS_LAMBDA(const int m) {
        for (int v = 0; v < ncoeff_; ++v) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) != BoundaryFlag::block &&
              mb_bcs.d_view(m, BoundaryFace::inner_x1) != BoundaryFlag::periodic) {
            for (int k = 0; k < nz; ++k)
              for (int j = 0; j < ny; ++j)
                for (int n = 0; n < ngh; ++n)
                  coeff(m, v, k, j, ngh - 1 - n) = coeff(m, v, k, j, ngh + n);
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x1) != BoundaryFlag::block &&
              mb_bcs.d_view(m, BoundaryFace::outer_x1) != BoundaryFlag::periodic) {
            for (int k = 0; k < nz; ++k)
              for (int j = 0; j < ny; ++j)
                for (int n = 0; n < ngh; ++n)
                  coeff(m, v, k, j, nx - ngh + n) =
                      coeff(m, v, k, j, nx - ngh - 1 - n);
          }
          if (mb_bcs.d_view(m, BoundaryFace::inner_x2) != BoundaryFlag::block &&
              mb_bcs.d_view(m, BoundaryFace::inner_x2) != BoundaryFlag::periodic) {
            for (int k = 0; k < nz; ++k)
              for (int i = 0; i < nx; ++i)
                for (int n = 0; n < ngh; ++n)
                  coeff(m, v, k, ngh - 1 - n, i) = coeff(m, v, k, ngh + n, i);
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x2) != BoundaryFlag::block &&
              mb_bcs.d_view(m, BoundaryFace::outer_x2) != BoundaryFlag::periodic) {
            for (int k = 0; k < nz; ++k)
              for (int i = 0; i < nx; ++i)
                for (int n = 0; n < ngh; ++n)
                  coeff(m, v, k, ny - ngh + n, i) =
                      coeff(m, v, k, ny - ngh - 1 - n, i);
          }
          if (mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::block &&
              mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::periodic) {
            for (int j = 0; j < ny; ++j)
              for (int i = 0; i < nx; ++i)
                for (int n = 0; n < ngh; ++n)
                  coeff(m, v, ngh - 1 - n, j, i) = coeff(m, v, ngh + n, j, i);
          }
          if (mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::block &&
              mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::periodic) {
            for (int j = 0; j < ny; ++j)
              for (int i = 0; i < nx; ++i)
                for (int n = 0; n < ngh; ++n)
                  coeff(m, v, nz - ngh + n, j, i) =
                      coeff(m, v, nz - ngh - 1 - n, j, i);
          }
        }
      });
    Kokkos::fence();
  }

  if (pbval != nullptr && pmy_driver_->nreflevel_ > 0 && CanFillFineCoarseGhosts()) {
    DvceArray5D<Real> coeff = coeff_[current_level_].d_view;
    (void)pbval->FillFineCoarseMGGhosts(coeff);
    Kokkos::fence();
  } else if (pbval != nullptr && pmy_driver_->nreflevel_ > 0 && MGDebugEnabled()) {
    std::cerr << "[rank " << global_variable::my_rank
              << "] skipping coefficient fine/coarse MG ghosts at level "
              << current_level_ << " with " << ncells << " active cell(s)"
              << std::endl;
  }
  current_level_ = saved_level;
}



//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ApplyMask()
//  \brief Apply the user-defined source mask function on the finest level

void Multigrid::ApplyMask() {
  Real mask_r = pmy_driver_->mask_radius_;
  if (mask_r <= 0.0) return;

  int ngh = ngh_;
  int nx1 = indcs_.nx1, nx2 = indcs_.nx2, nx3 = indcs_.nx3;
  int nmb = nmmb_;
  Real mask_r2 = mask_r * mask_r;
  Real ox = pmy_driver_->mask_origin_[0];
  Real oy = pmy_driver_->mask_origin_[1];
  Real oz = pmy_driver_->mask_origin_[2];

  auto src = src_[nlevel_-1].d_view;
  auto &mb_size = pmy_pack_->pmb->mb_size;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;

  par_for("Multigrid::ApplyMask", DevExeSpace(), 0, nmb-1,
          ngh, ngh+nx3-1, ngh, ngh+nx2-1, ngh, ngh+nx1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x = CellCenterX(i - ngh, nx1, mb_size.d_view(m).x1min, mb_size.d_view(m).x1max);
    Real y = CellCenterX(j - ngh, nx2, mb_size.d_view(m).x2min, mb_size.d_view(m).x2max);
    Real z = CellCenterX(k - ngh, nx3, mb_size.d_view(m).x3min, mb_size.d_view(m).x3max);
    Real r2 = (x-ox)*(x-ox) + (y-oy)*(y-oy) + (z-oz)*(z-oz);
    if (r2 > mask_r2) {
      src(m, 0, k, j, i) = 0.0;
    }
  });
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictCoefficients()
//! \brief restrict coefficients within a Multigrid object

void Multigrid::RestrictCoefficients() {
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  int saved_level = current_level_;
  if (on_host_) {
    for (int lev = nlevel_ - 1; lev > 0; lev--) {
      int ll = nlevel_ - lev;
      ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
      Restrict(coeff_[lev-1].h_view, coeff_[lev].h_view, ncoeff_,
               is, ie, js, je, ks, ke, false);
      coeff_[lev-1].template modify<HostExeSpace>();
      FillCoefficientBoundaries(lev - 1);
    }
  } else {
    for (int lev = nlevel_ - 1; lev > 0; lev--) {
      int ll = nlevel_ - lev;
      ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
      Restrict(coeff_[lev-1].d_view, coeff_[lev].d_view, ncoeff_,
               is, ie, js, je, ks, ke, false);
      FillCoefficientBoundaries(lev - 1);
    }
  }
  current_level_ = saved_level;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh)
//! \brief Set the result, including the ghost zone

void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh) {
  const auto &src = u_[nlevel_-1].d_view;
  int sngh = std::min(ngh_,ngh);

  if (ns == 0 && ngh_ == ngh && nvar_ == 1
      && src.extent(0) == dst.extent(0)
      && src.extent(2) == dst.extent(2)
      && src.extent(3) == dst.extent(3)
      && src.extent(4) == dst.extent(4)) {
    Kokkos::deep_copy(dst, src);
  } else {
    int is, ie, js, je, ks, ke;
    is = js = ks = ngh_ - sngh;
    ie = indcs_.nx1 + ngh_ + sngh - 1;
    je = indcs_.nx2 + ngh_ + sngh - 1;
    ke = indcs_.nx3 + ngh_ + sngh - 1;

    const int dst_off = ngh - ngh_;

    par_for("Multigrid::RetrieveResult", DevExeSpace(),
            0, nmmb_-1, 0, nvar_-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
      const int ndst = ns + v;
      const int k = mk + dst_off;
      const int j = mj + dst_off;
      const int i = mi + dst_off;
      dst(m, ndst, k, j, i) = src(m, v, mk, mj, mi);
    });
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RetrieveDefect(DvceArray5D<Real> &dst, int ns, int ngh)
//! \brief Set the defect, including the ghost zone

void Multigrid::RetrieveDefect(DvceArray5D<Real> &dst, int ns, int ngh) {
  const auto &src = def_[nlevel_-1].d_view;
  int sngh = std::min(ngh_,ngh);
  int ie = indcs_.nx1 + ngh_ + sngh - 1;
  int je = indcs_.nx2 + ngh_ + sngh - 1;
  int ke = indcs_.nx3 + ngh_ + sngh - 1;

  // local copies for device lambda capture
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int mk0 = ngh_ - sngh, mk1 = ke;
  const int mj0 = ngh_ - sngh, mj1 = je;
  const int mi0 = ngh_ - sngh, mi1 = ie;
  const Real scale = defscale_;
  const int dst_off = ngh - ngh_;

  auto dst_ = dst;
  auto src_ = src;

  par_for("Multigrid::RetrieveDefect", DevExeSpace(),
          m0, m1, v0, v1, mk0, mk1, mj0, mj1, mi0, mi1,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int ndst = ns + v;
    const int k = mk + dst_off;
    const int j = mj + dst_off;
    const int i = mi + dst_off;
    dst_(m, ndst, k, j, i) = src_(m, v, mk, mj, mi) * scale;
  });

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ZeroClearData()
//! \brief Clear the data array with zero

void Multigrid::ZeroClearData() {
  if (on_host_) {
    Kokkos::deep_copy(u_[current_level_].h_view, 0.0);
  } else {
    Kokkos::deep_copy(u_[current_level_].d_view, 0.0);
  }
}

void Multigrid::CopySourceToData() {
  if (on_host_) {
    Kokkos::deep_copy(u_[nlevel_-1].h_view, src_[nlevel_-1].h_view);
  } else {
    Kokkos::deep_copy(u_[nlevel_-1].d_view, src_[nlevel_-1].d_view);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictPack()
//! \brief Restrict the defect to the source

void Multigrid::RestrictPack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  CalculateDefectPack();
  DiagnosticRestrictPack();
  if (CompositeRestrictPack()) {
    current_level_--;
    return;
  }
  is=js=ks= ngh_;
  ie = is + (indcs_.nx1>>ll) - 1;
  je = js + (indcs_.nx2>>ll) - 1;
  ke = ks + (indcs_.nx3>>ll) - 1;
  if (on_host_) {
    Restrict(src_[current_level_-1].h_view, def_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
    Restrict(u_[current_level_-1].h_view, u_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
  } else {
    Restrict(src_[current_level_-1].d_view, def_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
    Restrict(u_[current_level_-1].d_view, u_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
  }
  current_level_--;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictSourcePack()
//! \brief Restrict the source (and solution) without forming defect

void Multigrid::RestrictSourcePack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks= ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;
  if (on_host_) {
    Restrict(src_[current_level_-1].h_view, src_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
  } else {
    Restrict(src_[current_level_-1].d_view, src_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
  }
  current_level_--;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrectPack()
//! \brief Prolongate the potential using tri-linear interpolation

void Multigrid::ProlongateAndCorrectPack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  const bool debug_meshblock =
      pmy_pack_ != nullptr && pmy_driver_->nreflevel_ > 0 && current_level_ == 0
      && pmy_driver_->debug_meshblock_correction_
      && pmy_driver_->meshblock_correction_debug_pending_;
  Real level1_defect_before = 0.0;
  std::vector<Real> level1_before;
  const int fine_level = current_level_ + 1;
  const int fine_ncells = (fine_level < nlevel_) ? GetLevelActiveCells(fine_level) : 0;
  if (debug_meshblock) {
    level1_defect_before = CalculateDiagnosticDefectRMS(fine_level);
    if (!on_host_) Kokkos::deep_copy(u_[fine_level].h_view, u_[fine_level].d_view);
    auto fine_h = u_[fine_level].h_view;
    level1_before.reserve(static_cast<std::size_t>(nmmb_) * nvar_ * fine_ncells
                          * fine_ncells * fine_ncells);
    for (int m = 0; m < nmmb_; ++m) {
      for (int v = 0; v < nvar_; ++v) {
        for (int k = ngh_; k < ngh_ + fine_ncells; ++k) {
          for (int j = ngh_; j < ngh_ + fine_ncells; ++j) {
            for (int i = ngh_; i < ngh_ + fine_ncells; ++i) {
              level1_before.push_back(fine_h(m, v, k, j, i));
            }
          }
        }
      }
    }
  }

  ComputeCorrection();

  if (pmy_pack_ != nullptr && pmy_driver_->nreflevel_ > 0 && current_level_ == 0
      && pmy_driver_->meshblock_correction_mode_ == 3) {
    ClampCurrentCorrectionGhostsToActive();
  }

  if (debug_meshblock) {
    if (!on_host_) Kokkos::deep_copy(u_[current_level_].h_view, u_[current_level_].d_view);
    auto coarse_h = u_[current_level_].h_view;
    const int nall = GetCurrentLevelActiveCells() + 2*ngh_;
    Real active_sum2 = 0.0, active_max = 0.0;
    Real ghost_sum2 = 0.0, ghost_max = 0.0;
    long long active_count = 0, ghost_count = 0;
    for (int m = 0; m < nmmb_; ++m) {
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k < nall; ++k) {
          for (int j = 0; j < nall; ++j) {
            for (int i = 0; i < nall; ++i) {
              Real val = coarse_h(m, v, k, j, i);
              bool active = (i == ngh_ && j == ngh_ && k == ngh_);
              if (active) {
                active_sum2 += val*val;
                active_max = std::max(active_max, std::abs(val));
                ++active_count;
              } else {
                ghost_sum2 += val*val;
                ghost_max = std::max(ghost_max, std::abs(val));
                ++ghost_count;
              }
            }
          }
        }
      }
    }
    Real active_rms = 0.0, active_global_max = 0.0;
    Real ghost_rms = 0.0, ghost_global_max = 0.0;
    ReduceCorrectionStats(active_sum2, active_max, active_count,
                          active_rms, active_global_max);
    ReduceCorrectionStats(ghost_sum2, ghost_max, ghost_count,
                          ghost_rms, ghost_global_max);
    if (global_variable::my_rank == 0) {
      std::cout << "CTS MeshBlock correction debug: level0 active corr max="
                << active_global_max << " rms=" << active_rms
                << " ghost max=" << ghost_global_max << " rms=" << ghost_rms
                << " level1 defect before=" << level1_defect_before
                << " active/defect_rms="
                << ((level1_defect_before > 0.0) ? active_rms/level1_defect_before : 0.0)
                << std::endl;
    }
  }

  if (on_host_) {
    ProlongateAndCorrect(u_[current_level_+1].h_view, u_[current_level_].h_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  } else {
    ProlongateAndCorrect(u_[current_level_+1].d_view, u_[current_level_].d_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  }

  current_level_++;
  pmy_driver_->PostProlongationCorrection(this);
  if (debug_meshblock) {
    if (!on_host_) Kokkos::deep_copy(u_[current_level_].h_view, u_[current_level_].d_view);
    auto fine_h = u_[current_level_].h_view;
    Real applied_sum2 = 0.0, applied_max = 0.0;
    long long applied_count = 0;
    std::size_t idx = 0;
    for (int m = 0; m < nmmb_; ++m) {
      for (int v = 0; v < nvar_; ++v) {
        for (int k = ngh_; k < ngh_ + fine_ncells; ++k) {
          for (int j = ngh_; j < ngh_ + fine_ncells; ++j) {
            for (int i = ngh_; i < ngh_ + fine_ncells; ++i) {
              Real diff = fine_h(m, v, k, j, i) - level1_before[idx++];
              applied_sum2 += diff*diff;
              applied_max = std::max(applied_max, std::abs(diff));
              ++applied_count;
            }
          }
        }
      }
    }
    Real applied_rms = 0.0, applied_global_max = 0.0;
    ReduceCorrectionStats(applied_sum2, applied_max, applied_count,
                          applied_rms, applied_global_max);
    Real level1_defect_after = CalculateDiagnosticDefectRMS(current_level_);
    if (global_variable::my_rank == 0) {
      std::cout << "CTS MeshBlock correction debug: applied level1 corr max="
                << applied_global_max << " rms=" << applied_rms
                << " level1 defect after=" << level1_defect_after
                << " defect_ratio="
                << ((level1_defect_before > 0.0)
                    ? level1_defect_after/level1_defect_before : 0.0)
                << std::endl;
    }
    pmy_driver_->meshblock_correction_debug_pending_ = false;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongatePack()
//! \brief Prolongate the solution for FMG (direct overwrite, linear/cubic policy)

void Multigrid::FMGProlongatePack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  if (on_host_) {
    FMGProlongate(u_[current_level_+1].h_view, u_[current_level_].h_view,
                  is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);
  } else {
    FMGProlongate(u_[current_level_+1].d_view, u_[current_level_].d_view,
                  is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);
  }

  current_level_++;
}


//----------------------------------------------------------------------------------------
//! \fn  void Multigrid::SmoothPack(int color)
//! \brief Apply Smoother on the Pack


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::SetFromRootGrid(bool folddata)
//! \brief Load the data from the root grid or octets (Athena++ style per-cell octets)

void Multigrid::SetFromRootGrid(bool folddata) {
  current_level_ = 0;
  auto dst_h = u_[current_level_].h_view;
  auto odst_h = uold_[current_level_].h_view;

  auto src_h = pmy_driver_->GetRootData_h();
  auto osrc_h = pmy_driver_->GetRootOldData_h();
  int padding = pmy_mesh_->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmmb_; ++m) {
    auto loc = pmy_mesh_->lloc_eachmb[m + padding];
    int lev = loc.level - pmy_driver_->locrootlevel_;
    if (lev == 0) {
      int ci = static_cast<int>(loc.lx1);
      int cj = static_cast<int>(loc.lx2);
      int ck = static_cast<int>(loc.lx3);
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2*ngh_; ++k) {
          for (int j = 0; j <= 2*ngh_; ++j) {
            for (int i = 0; i <= 2*ngh_; ++i) {
              dst_h(m, v, k, j, i) = src_h(0, v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = osrc_h(0, v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    } else {
      LogicalLocation oloc;
      oloc.lx1 = (loc.lx1 >> 1);
      oloc.lx2 = (loc.lx2 >> 1);
      oloc.lx3 = (loc.lx3 >> 1);
      oloc.level = loc.level - 1;
      int olev = oloc.level - pmy_driver_->locrootlevel_;
      int oid = pmy_driver_->FindOctetIdOrDie(olev, oloc,
                                              "Multigrid::RetrieveResult");
      int ci = (static_cast<int>(loc.lx1) & 1);
      int cj = (static_cast<int>(loc.lx2) & 1);
      int ck = (static_cast<int>(loc.lx3) & 1);
      const MGOctet &oct = pmy_driver_->octets_[olev][oid];
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2*ngh_; ++k) {
          for (int j = 0; j <= 2*ngh_; ++j) {
            for (int i = 0; i <= 2*ngh_; ++i) {
              dst_h(m, v, k, j, i) = oct.U(v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = oct.Uold(v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    }
  }
  u_[current_level_].template modify<HostExeSpace>();
  u_[current_level_].template sync<DevExeSpace>();
  if (folddata) {
    uold_[current_level_].template modify<HostExeSpace>();
    uold_[current_level_].template sync<DevExeSpace>();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the residual norm

Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n) {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
  int ll_l = ll;
  CalculateDefectPack();

  Real norm = 0.0;

  if (on_host_) {
    auto &def = def_[current_level_].h_view;
    auto brdx = block_rdx_.h_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::DefectNorm_Linf",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_max) {
          local_max = std::max(local_max, std::abs(def(m, v, k, j, i)));
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::DefectNorm_L1",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dx = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dV = dx * dx * dx;
          local_sum += dV * std::abs(def(m, v, k, j, i));
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dx = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dV = dx * dx * dx;
          Real val = def(m, v, k, j, i);
          local_sum += dV * val * val;
        }, Kokkos::Sum<Real>(norm));
    }
  } else {
    auto &def = def_[current_level_].d_view;
    auto brdx = block_rdx_.d_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::DefectNorm_Linf",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_max) {
          local_max = std::max(local_max, std::abs(def(m, v, k, j, i)));
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::DefectNorm_L1",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dx = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dV = dx * dx * dx;
          local_sum += dV * std::abs(def(m, v, k, j, i));
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dx = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dV = dx * dx * dx;
          Real val = def(m, v, k, j, i);
          local_sum += dV * val * val;
        }, Kokkos::Sum<Real>(norm));
    }
  }
  norm *= defscale_;
  return norm;
}

//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateAverage(MGVariable type)
//! \brief Calculate volume-weighted average of variable 0 on current level

Real Multigrid::CalculateAverage(MGVariable type) {
  int ll = nlevel_ - 1 - current_level_;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + (indcs_.nx1 >> ll) - 1;
  je = js + (indcs_.nx2 >> ll) - 1;
  ke = ks + (indcs_.nx3 >> ll) - 1;
  int ll_l = ll;

  Real sum = 0.0;
  if (on_host_) {
    auto data = (type == MGVariable::src) ? src_[current_level_].h_view
                                          : u_[current_level_].h_view;
    auto brdx = block_rdx_.h_view;
    Kokkos::parallel_reduce("MG::Average",
      Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                            {nmmb_, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
        Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
        Real dV_m = dx_m * dx_m * dx_m;
        local_sum += data(m, 0, k, j, i) * dV_m;
      }, Kokkos::Sum<Real>(sum));
  } else {
    auto data = (type == MGVariable::src) ? src_[current_level_].d_view
                                          : u_[current_level_].d_view;
    auto brdx = block_rdx_.d_view;
    Kokkos::parallel_reduce("MG::Average",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                           {nmmb_, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
        Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
        Real dV_m = dx_m * dx_m * dx_m;
        local_sum += data(m, 0, k, j, i) * dV_m;
      }, Kokkos::Sum<Real>(sum));
  }

  Real volume = 0.0;
  {
    Real nx = static_cast<Real>(indcs_.nx1);
    for (int m = 0; m < nmmb_; ++m) {
      Real len = block_rdx_.h_view(m) * nx;
      volume += len * len * len;
    }
  }

  #if MPI_PARALLEL_ENABLED
  Real global_sum = 0.0;
  Real global_volume = 0.0;
  MPI_Allreduce(&sum, &global_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&volume, &global_volume, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sum = global_sum;
  volume = global_volume;
  #endif

  return (volume > 0.0) ? (sum / volume) : 0.0;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateTotal(MGVariable type, int n)
//! \brief calculate the sum of the array (type: 0=src, 1=u)

Real Multigrid::CalculateTotal(MGVariable type, int n) {
  //DvceArray5D<Real> &src =
  //                  (type == MGVariable::src) ? src_[current_level_] : u_[current_level_];
  //int ll = nlevel_ - 1 - current_level_;
  //Real s=0.0;
  //int is, ie, js, je, ks, ke;
  //is=js=ks=ngh_;
  //ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
  //Real dx=rdx_*static_cast<Real>(1<<ll), dy=rdy_*static_cast<Real>(1<<ll),
  //     dz=rdz_*static_cast<Real>(1<<ll);
  //for (int k=ks; k<=ke; ++k) {
  //  for (int j=js; j<=je; ++j) {
  //    for (int i=is; i<=ie; ++i)
  //      s+=src(n,k,j,i);
  //  }
  //}
  //return s*dx*dy*dz;
  return 0.0;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::SubtractAverage(MGVariable type, int v, Real ave)
//! \brief subtract the average value (type: 0=src, 1=u)

void Multigrid::SubtractAverage(MGVariable type, int n, Real ave) {
  int is, ie, js, je, ks, ke;
  is = js = ks = 0;
  ie = is + indcs_.nx1 + 2*ngh_ - 1;
  je = js + indcs_.nx2 + 2*ngh_ - 1;
  ke = ks + indcs_.nx3 + 2*ngh_ - 1;

  const int m0 = 0, m1 = nmmb_ - 1;
  const int vn = n;
  const Real lave = ave;

  if (on_host_) {
    auto dst = (type == MGVariable::src) ? src_[nlevel_-1].h_view
                                         : u_[nlevel_-1].h_view;
    par_for("Multigrid::SubtractAverage", HostExeSpace(),
            m0, m1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
      dst(m, vn, mk, mj, mi) -= lave;
    });
  } else {
    auto dst = (type == MGVariable::src) ? src_[nlevel_-1].d_view
                                         : u_[nlevel_-1].d_view;
    par_for("Multigrid::SubtractAverage", DevExeSpace(),
            m0, m1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
      dst(m, vn, mk, mj, mi) -= lave;
    });
  }
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::StoreOldData()
//! \brief store the old u data in the uold array

void Multigrid::StoreOldData() {
  if (on_host_) {
    Kokkos::deep_copy(HostExeSpace(), uold_[current_level_].h_view,
                      u_[current_level_].h_view);
  } else {
    Kokkos::deep_copy(DevExeSpace(), uold_[current_level_].d_view,
                      u_[current_level_].d_view);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::Restrict(...)
//  \brief Actual implementation of restriction (templated on view type)

template <typename ViewType>
void Multigrid::Restrict(ViewType &dst, const ViewType &src,
                int nvar, int i0, int i1, int j0, int j1, int k0, int k1, bool th) {

  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar - 1;
  const int ngh = ngh_;
                
  par_for("Multigrid::Restrict", ExeSpace(),
          m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fk = 2*k - ngh;
    const int fj = 2*j - ngh;
    const int fi = 2*i - ngh;
    dst(m, v, k, j, i) = 0.125 * (
        src(m, v, fk,   fj,   fi)   + src(m, v, fk,   fj,   fi+1)
      + src(m, v, fk,   fj+1, fi)   + src(m, v, fk,   fj+1, fi+1)
      + src(m, v, fk+1, fj,   fi)   + src(m, v, fk+1, fj,   fi+1)
      + src(m, v, fk+1, fj+1, fi)   + src(m, v, fk+1, fj+1, fi+1));
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ComputeCorrection(DvceArray5D<Real> &correction, int level)
//! \brief Compute the correction as u_[level] - uold_[level]

void Multigrid::ComputeCorrection() {
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  int ll = nlevel_ - 1 - current_level_;
  int is = 0, ie = is + (indcs_.nx1 >> ll) + 2*ngh_ -1;
  int js = 0, je = js + (indcs_.nx2 >> ll) + 2*ngh_ -1;
  int ks = 0, ke = ks + (indcs_.nx3 >> ll) + 2*ngh_ -1;

  if (on_host_) {
    auto u = u_[current_level_].h_view;
    auto uold = uold_[current_level_].h_view;
    par_for("Multigrid::ComputeCorrection", HostExeSpace(),
            m0, m1, v0, v1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) -= uold(m, v, k, j, i);
    });
  } else {
    auto u = u_[current_level_].d_view;
    auto uold = uold_[current_level_].d_view;
    par_for("Multigrid::ComputeCorrection", DevExeSpace(),
            m0, m1, v0, v1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) -= uold(m, v, k, j, i);
    });
  }
}

void Multigrid::ClampCurrentCorrectionGhostsToActive() {
  const int ncells = GetCurrentLevelActiveCells();
  const int nall = ncells + 2*ngh_;
  if (on_host_) {
    auto u = u_[current_level_].h_view;
    par_for("Multigrid::ClampCorrectionGhostsToActive", HostExeSpace(),
            0, nmmb_-1, 0, nvar_-1, 0, nall-1, 0, nall-1, 0, nall-1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) = u(m, v, ngh_, ngh_, ngh_);
    });
  } else {
    auto u = u_[current_level_].d_view;
    par_for("Multigrid::ClampCorrectionGhostsToActive", DevExeSpace(),
            0, nmmb_-1, 0, nvar_-1, 0, nall-1, 0, nall-1, 0, nall-1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) = u(m, v, ngh_, ngh_, ngh_);
    });
  }
}

Real Multigrid::CalculateDiagnosticDefectRMS(int level) {
  const int saved_level = current_level_;
  current_level_ = level;
  CalculateDefectPack();
  if (!on_host_) {
    Kokkos::deep_copy(def_[level].h_view, def_[level].d_view);
  }
  auto def_h = def_[level].h_view;
  const int ncells = GetLevelActiveCells(level);
  Real local_sum2 = 0.0;
  Real local_max = 0.0;
  long long local_count = 0;
  for (int m = 0; m < nmmb_; ++m) {
    for (int v = 0; v < nvar_; ++v) {
      for (int k = ngh_; k < ngh_ + ncells; ++k) {
        for (int j = ngh_; j < ngh_ + ncells; ++j) {
          for (int i = ngh_; i < ngh_ + ncells; ++i) {
            Real val = def_h(m, v, k, j, i);
            local_sum2 += val*val;
            local_max = std::max(local_max, std::abs(val));
            ++local_count;
          }
        }
      }
    }
  }
  Real rms = 0.0, maxabs = 0.0;
  ReduceCorrectionStats(local_sum2, local_max, local_count, rms, maxabs);
  current_level_ = saved_level;
  return rms;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrect(...)
//! \brief Actual implementation of prolongation and correction (templated on view type)

template <typename ViewType>
void Multigrid::ProlongateAndCorrect(ViewType &dst, const ViewType &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th) {

  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  const bool meshblock_transfer_correction =
      pmy_pack_ != nullptr && pmy_driver_->nreflevel_ > 0 && current_level_ == 0;
  const int mode = meshblock_transfer_correction
                   ? pmy_driver_->meshblock_correction_mode_ : 2;
  if (meshblock_transfer_correction && mode == 0) return;

  const int ll = pmy_driver_->fprolongation_; // copy host flag for capture
  Real corr_omega = (pmy_pack_ == nullptr)
                    ? pmy_driver_->root_correction_omega_
                    : pmy_driver_->meshblock_correction_omega_;
  if (meshblock_transfer_correction) {
    corr_omega *= static_cast<Real>(pmy_driver_->meshblock_correction_sign_);
  }
  const Real cubic_fac = corr_omega / 32768.0;
  const Real linear_fac = corr_omega * 0.015625;

  auto dst_ = dst;
  auto src_ = src;

  if (meshblock_transfer_correction && mode == 1) {
    par_for("Multigrid::ProlongateAndCorrect_injection", ExeSpace(),
            m0, m1, v0, v1,
    KOKKOS_LAMBDA(const int m, const int v) {
      const Real corr = corr_omega * src_(m, v, kl, jl, il);
      const int fk = fkl;
      const int fj = fjl;
      const int fi = fil;
      dst_(m, v, fk,   fj,   fi)   += corr;
      dst_(m, v, fk,   fj,   fi+1) += corr;
      dst_(m, v, fk,   fj+1, fi)   += corr;
      dst_(m, v, fk,   fj+1, fi+1) += corr;
      dst_(m, v, fk+1, fj,   fi)   += corr;
      dst_(m, v, fk+1, fj,   fi+1) += corr;
      dst_(m, v, fk+1, fj+1, fi)   += corr;
      dst_(m, v, fk+1, fj+1, fi+1) += corr;
    });
    return;
  }

  if (ll == 1) { // tricubic
    par_for("Multigrid::ProlongateAndCorrect_tricubic", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      // For brevity: local references to src entries
      // compute and add to 8 target cells as in original implementation
      dst_(m,v,fk  ,fj  ,fi  ) += (
        + 125.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
        + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
        + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
        -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
        +  45.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )-  27.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk,  fj,  fi+1) += (
        -  75.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )+ 125.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
        +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
        + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        -  27.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk  ,fj+1,fi  ) += (
        -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
        + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        + 125.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        +  45.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )-  27.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
        -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk,  fj+1,fi+1) += (
        +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
        -  75.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )+ 125.*src_(m,v,k-1,j+1,i+1)
        + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
        -  27.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
        + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk+1,fj,  fi  ) += (
        -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
        +  45.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )-  27.*src_(m,v,k-1,j+1,i+1)
        + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
        + 125.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
        + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk+1,fj,  fi+1) += (
        +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
        + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        -  27.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        -  75.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )+ 125.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
        +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk+1,fj+1,fi  ) += (
        +  45.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )-  27.*src_(m,v,k-1,j-1,i+1)
        - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
        -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
        - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
        +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
        + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
        -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
        + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
        + 125.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;

      dst_(m,v,fk+1,fj+1,fi+1) += (
        -  27.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
        + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
        +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
        + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
        -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
        - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
        +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
        - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
        -  75.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )+ 125.*src_(m,v,k+1,j+1,i+1)
      ) * cubic_fac;
    });
  } else { // trilinear
    par_for("Multigrid::ProlongateAndCorrect_trilinear", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      dst_(m,v,fk  ,fj  ,fi  ) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk  ,fj  ,fi+1) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi  ) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi  ) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk+1,fj+1,fi  ) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi+1) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi+1) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j+1,i+1)));
      dst_(m,v,fk+1,fj+1,fi+1) +=
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j+1,i+1)));
    });
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongate(...)
//! \brief FMG prolongation: direct overwrite (=) using the selected interpolation policy.
//! Unlike ProlongateAndCorrect (+=), this overwrites the destination array.

template <typename ViewType>
void Multigrid::FMGProlongate(ViewType &dst, const ViewType &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl) {

  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  const int ll = pmy_driver_->fprolongation_;
  constexpr Real linear_fac = 0.015625;

  auto dst_ = dst;
  auto src_ = src;

  if (ll != 1) { // trilinear direct overwrite for FMG initial guesses
    par_for("Multigrid::FMGProlongate_trilinear", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      dst_(m,v,fk  ,fj  ,fi  ) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk  ,fj  ,fi+1) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi  ) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi  ) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk+1,fj+1,fi  ) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi+1) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi+1) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j+1,i+1)));
      dst_(m,v,fk+1,fj+1,fi+1) =
          linear_fac*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j+1,i+1)));
    });
    return;
  }

  par_for("Multigrid::FMGProlongate", ExeSpace(),
          m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fk = 2*(k-kl) + fkl;
    const int fj = 2*(j-jl) + fjl;
    const int fi = 2*(i-il) + fil;

    dst_(m,v,fk  ,fj  ,fi  ) = (
      + 125.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
      + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
      + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )-  27.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk,  fj,  fi+1) = (
      -  75.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )+ 125.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
      + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      -  27.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk  ,fj+1,fi  ) = (
      -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
      + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      + 125.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )-  27.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,  i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk,  fj+1,fi+1) = (
      +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,  i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )+ 125.*src_(m,v,k-1,j+1,i+1)
      + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
      -  27.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
      + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj,  fi  ) = (
      -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )-  27.*src_(m,v,k-1,j+1,i+1)
      + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,i+1)
      + 125.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
      + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj,  fi+1) = (
      +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,i+1)
      + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      -  27.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )+ 125.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj+1,fi  ) = (
      +  45.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )-  27.*src_(m,v,k-1,j-1,i+1)
      - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,  i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,i+1)
      - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,  i+1)
      + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,i+1)
      + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,  i+1)
      + 125.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj+1,fi+1) = (
      -  27.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,i+1)
      + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,  i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,i+1)
      + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,  i+1)
      - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,i+1)
      - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,  i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )+ 125.*src_(m,v,k+1,j+1,i+1)
    ) / 32768.0;
  });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::MultigridBoundaryValues()
//! \brief Constructor for multigrid boundary values object
//----------------------------------------------------------------------------------------

MultigridBoundaryValues::MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse, Multigrid *pmg) 
  :
   MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg) {
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::RemapIndicesForMG()
//! \brief Remap isame indices from hydro coordinates (ng ghost cells) to MG coordinates
//! (ngh_ ghost cells). Must be called AFTER InitializeBuffers.

void MultigridBoundaryValues::RemapIndicesForMG() {
  int ng  = pmy_pack->pmesh->mb_indcs.ng;
  int ngh = pmy_mg->GetGhostCells();
  if (ng != ngh) {
    int nx1 = pmy_pack->pmesh->mb_indcs.nx1;
    int nx2 = pmy_pack->pmesh->mb_indcs.nx2;
    int nx3 = pmy_pack->pmesh->mb_indcs.nx3;
    int is_h = ng, ie_h = ng + nx1 - 1;
    int js_h = ng, je_h = ng + nx2 - 1;
    int ks_h = ng, ke_h = ng + nx3 - 1;
    int is_m = ngh, ie_m = ngh + nx1 - 1;
    int js_m = ngh, je_m = ngh + nx2 - 1;
    int ks_m = ngh, ke_m = ngh + nx3 - 1;
    int ng1_m = ngh - 1;
    int nnghbr = pmy_pack->pmb->nnghbr;

    auto remap_send = [](int &lo, int &hi,
                         int s_h, int e_h, int s_m, int e_m, int ng1) {
      if (lo == s_h && hi == e_h) { lo = s_m; hi = e_m; }
      else if (lo > s_h)          { lo = e_m - ng1; hi = e_m; }
      else                        { lo = s_m; hi = s_m + ng1; }
    };
    auto remap_recv = [](int &lo, int &hi,
                         int s_h, int e_h, int s_m, int e_m, int ng_m) {
      if (lo >= s_h && hi <= e_h) { lo = s_m; hi = e_m; }
      else if (lo > e_h)          { lo = e_m + 1; hi = e_m + ng_m; }
      else                        { lo = s_m - ng_m; hi = s_m - 1; }
    };

    for (int n = 0; n < nnghbr; ++n) {
      auto &si = sendbuf[n].isame[0];
      remap_send(si.bis, si.bie, is_h, ie_h, is_m, ie_m, ng1_m);
      remap_send(si.bjs, si.bje, js_h, je_h, js_m, je_m, ng1_m);
      remap_send(si.bks, si.bke, ks_h, ke_h, ks_m, ke_m, ng1_m);
      sendbuf[n].isame_ndat = (si.bie-si.bis+1)*(si.bje-si.bjs+1)*(si.bke-si.bks+1);

      auto &ri = recvbuf[n].isame[0];
      remap_recv(ri.bis, ri.bie, is_h, ie_h, is_m, ie_m, ngh);
      remap_recv(ri.bjs, ri.bje, js_h, je_h, js_m, je_m, ngh);
      remap_recv(ri.bks, ri.bke, ks_h, ke_h, ks_m, ke_m, ngh);
      recvbuf[n].isame_ndat = (ri.bie-ri.bis+1)*(ri.bje-ri.bjs+1)*(ri.bke-ri.bks+1);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts()
//! \brief Fill ghost cells at fine-coarse boundaries.
//! Faces use flux-conserving prolongation/restriction matching Athena++ formulas.
//! Edges and corners use simple injection/restriction. Same-rank only.

TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;

  if (ncells < 1) return TaskStatus::complete;
  if (ncells < 2) {
    if (MGDebugEnabled()) {
      std::cerr << "[rank " << global_variable::my_rank
                << "] skipping fine/coarse MG ghosts at level "
                << pmy_mg->GetCurrentLevel() << " with " << ncells
                << " active cell(s)" << std::endl;
    }
    return TaskStatus::complete;
  }

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;

  auto &nghbr_h = pmy_pack->pmb->nghbr.h_view;
  auto &mblev_h = pmy_pack->pmb->mb_lev.h_view;
  auto &mbgid_h = pmy_pack->pmb->mb_gid.h_view;
  for (int m = 0; m < nmb; ++m) {
    int m_lev = mblev_h(m);
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr_h(m, n).gid < 0 || nghbr_h(m, n).lev == m_lev) continue;
      if (nghbr_h(m, n).rank == my_rank) continue;
      if (fc_stage_valid_) continue;
      std::cout << "### FATAL ERROR in MultigridBoundaryValues::FillFineCoarseMGGhosts"
                << std::endl
                << "Cross-rank fine/coarse MG ghost fill requested before staged "
                << "payloads were received. "
                << "Local block gid=" << mbgid_h(m) << " rank=" << my_rank
                << " level=" << m_lev << " has neighbor gid=" << nghbr_h(m, n).gid
                << " rank=" << nghbr_h(m, n).rank
                << " level=" << nghbr_h(m, n).lev << "."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  auto nghbr_d = pmy_pack->pmb->nghbr.d_view;
  auto mblev_d = pmy_pack->pmb->mb_lev.d_view;
  auto mbgid_d = pmy_pack->pmb->mb_gid.d_view;
  auto fc_cx = pmy_mg->fc_childx_;
  auto fc_cy = pmy_mg->fc_childy_;
  auto fc_cz = pmy_mg->fc_childz_;
  auto &rbuf = recvbuf;

  int nmb_l = nmb;
  int nnghbr_l = nnghbr;
  int my_rank_l = my_rank;
  int nvar_l = nvar;
  int ngh_l = ngh;
  int ncells_l = ncells;
  int shift_l = shift;
  constexpr Real ot = 1.0/3.0;

  Kokkos::parallel_for("FillFCMGGhosts",
    Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m) {
      int m_lev = mblev_d(m);
      int child_x = fc_cx(m);
      int child_y = fc_cy(m);
      int child_z = fc_cz(m);
      int half = ncells_l / 2;

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);

            for (int f2 = 0; f2 <= 1; ++f2) {
              for (int f1 = 0; f1 <= 1; ++f1) {
                int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (n < 0 || n >= nnghbr_l) continue;
                if (nghbr_d(m, n).gid < 0) continue;

                int nlev = nghbr_d(m, n).lev;
                if (nlev == m_lev) continue;
                bool remote_fc = (nghbr_d(m, n).rank != my_rank_l);

                int dm = nghbr_d(m, n).gid - mbgid_d(0);
                if (!remote_fc && (dm < 0 || dm >= nmb_l)) continue;

                MeshBufferIndcs stage_idx = (nlev < m_lev) ? rbuf[n].icoar[0]
                                           : rbuf[n].ifine[0];
                int sil = stage_idx.bis, siu = stage_idx.bie;
                int sjl = stage_idx.bjs, sju = stage_idx.bje;
                int skl = stage_idx.bks, sku = stage_idx.bke;
                AdjustRecvRangeForMG(sil, siu, sjl, sju, skl, sku,
                                     rbuf[n], shift_l, ngh_l);

                if (nlev < m_lev && nface == 1) {
                  // Coarser neighbor, face: flux-conserving prolongation
                  if (ox1 != 0) {
                    int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int si  = (ox1 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int sj0 = ngh_l + child_y * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int sj = sj0; sj < sj0 + half; ++sj) {
                          int fj = ngh_l + 2*(sj - sj0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,si);
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real ym = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sjm, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sjm,si);
                          Real yp = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sjp, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sjp,si);
                          Real zm = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, skm, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,skm,sj,si);
                          Real zp = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, skp, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,skp,sj,si);
                          Real gy = 0.125*(yp - ym);
                          Real gz = 0.125*(zp - zm);
                          u(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u(m,v,fk  ,fj  ,fi));
                          u(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u(m,v,fk  ,fj+1,fi));
                          u(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u(m,v,fk+1,fj  ,fi));
                          u(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u(m,v,fk+1,fj+1,fi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sj  = (ox2 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sk0 = ngh_l + child_z * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sk = sk0; sk < sk0 + half; ++sk) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fk = ngh_l + 2*(sk - sk0);
                          Real cc = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int skm = (sk > ngh_l) ? sk-1 : sk;
                          int skp = (sk < ngh_l+ncells_l-1) ? sk+1 : sk;
                          Real xm = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, sim,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,sim);
                          Real xp = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, sip,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,sip);
                          Real zm = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, skm, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,skm,sj,si);
                          Real zp = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, skp, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,skp,sj,si);
                          Real gx = 0.125*(xp - xm);
                          Real gz = 0.125*(zp - zm);
                          u(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u(m,v,fk  ,fj,fi  ));
                          u(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u(m,v,fk  ,fj,fi+1));
                          u(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u(m,v,fk+1,fj,fi  ));
                          u(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u(m,v,fk+1,fj,fi+1));
                        }
                      }
                    }
                  } else {
                    int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
                    int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
                    int sk  = (ox3 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
                    int si0 = ngh_l + child_x * half;
                    int sj0 = ngh_l + child_y * half;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        for (int si = si0; si < si0 + half; ++si) {
                          int fi = ngh_l + 2*(si - si0);
                          int fj = ngh_l + 2*(sj - sj0);
                          Real cc = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,si);
                          int sim = (si > ngh_l) ? si-1 : si;
                          int sip = (si < ngh_l+ncells_l-1) ? si+1 : si;
                          int sjm = (sj > ngh_l) ? sj-1 : sj;
                          int sjp = (sj < ngh_l+ncells_l-1) ? sj+1 : sj;
                          Real xm = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, sim,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,sim);
                          Real xp = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, sip,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sj,sip);
                          Real ym = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sjm, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sjm,si);
                          Real yp = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sjp, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm,v,sk,sjp,si);
                          Real gx = 0.125*(xp - xm);
                          Real gy = 0.125*(yp - ym);
                          u(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u(m,v,fk,fj  ,fi  ));
                          u(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u(m,v,fk,fj  ,fi+1));
                          u(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u(m,v,fk,fj+1,fi  ));
                          u(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u(m,v,fk,fj+1,fi+1));
                        }
                      }
                    }
                  }

                } else if (nlev > m_lev && nface == 1) {
                  // Finer neighbor, face: flux-conserving restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (ox1 != 0) { sub_y = f1; sub_z = f2; }
                  if (ox2 != 0) { sub_x = f1; sub_z = f2; }
                  if (ox3 != 0) { sub_x = f1; sub_y = f2; }

                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else { gis = ngh_l+sub_x*half; gie = ngh_l+sub_x*half+half-1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else { gjs = ngh_l+sub_y*half; gje = ngh_l+sub_y*half+half-1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else { gks = ngh_l+sub_z*half; gke = ngh_l+sub_z*half+half-1; }

                  int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                  int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                  int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                  if (ox1 != 0) {
                    int fi = (ox1 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gj = gjs; gj <= gje; ++gj) {
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, gk, gj, gis,
                                            sil, siu, sjl, sju, skl, sku)
                              : 0.25*(u(dm,v,fk0,fj0,fi)+u(dm,v,fk0,fj0+1,fi)
                                     +u(dm,v,fk0+1,fj0,fi)+u(dm,v,fk0+1,fj0+1,fi));
                          u(m,v,gk,gj,gis) = ot*(4.0*favg - u(m,v,gk+ok,gj+oj,gis+oi));
                        }
                      }
                    }
                  } else if (ox2 != 0) {
                    int fj = (ox2 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gk = gks; gk <= gke; ++gk) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
                          Real favg = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, gk, gjs, gi,
                                            sil, siu, sjl, sju, skl, sku)
                              : 0.25*(u(dm,v,fk0,fj,fi0)+u(dm,v,fk0,fj,fi0+1)
                                     +u(dm,v,fk0+1,fj,fi0)+u(dm,v,fk0+1,fj,fi0+1));
                          u(m,v,gk,gjs,gi) = ot*(4.0*favg - u(m,v,gk+ok,gjs+oj,gi+oi));
                        }
                      }
                    }
                  } else {
                    int fk = (ox3 > 0) ? ngh_l : ngh_l + ncells_l - 1;
                    for (int v = 0; v < nvar_l; ++v) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
                          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
                          Real favg = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, gks, gj, gi,
                                            sil, siu, sjl, sju, skl, sku)
                              : 0.25*(u(dm,v,fk,fj0,fi0)+u(dm,v,fk,fj0,fi0+1)
                                     +u(dm,v,fk,fj0+1,fi0)+u(dm,v,fk,fj0+1,fi0+1));
                          u(m,v,gks,gj,gi) = ot*(4.0*favg - u(m,v,gks+ok,gj+oj,gi+oi));
                        }
                      }
                    }
                  }

                } else if (nlev < m_lev) {
                  // Coarser neighbor, edge/corner: simple injection
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else              { gis = ngh_l;           gie = ngh_l + ncells_l - 1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else              { gjs = ngh_l;           gje = ngh_l + ncells_l - 1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else              { gks = ngh_l;           gke = ngh_l + ncells_l - 1; }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int si, sj, sk;
                          if (ox1 < 0)      si = ngh_l + ncells_l - 1;
                          else if (ox1 > 0) si = ngh_l;
                          else si = ngh_l + child_x*half + (gi - ngh_l)/2;
                          if (ox2 < 0)      sj = ngh_l + ncells_l - 1;
                          else if (ox2 > 0) sj = ngh_l;
                          else sj = ngh_l + child_y*half + (gj - ngh_l)/2;
                          if (ox3 < 0)      sk = ngh_l + ncells_l - 1;
                          else if (ox3 > 0) sk = ngh_l;
                          else sk = ngh_l + child_z*half + (gk - ngh_l)/2;

                          u(m, v, gk, gj, gi) = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, sk, sj, si,
                                            sil, siu, sjl, sju, skl, sku)
                              : u(dm, v, sk, sj, si);
                        }
                      }
                    }
                  }

                } else {
                  // Finer neighbor, edge/corner: simple restriction
                  int sub_x = 0, sub_y = 0, sub_z = 0;
                  if (nface == 2) {
                    if (ox1 == 0) sub_x = f1;
                    if (ox2 == 0) sub_y = f1;
                    if (ox3 == 0) sub_z = f1;
                  }
                  int gis, gie, gjs, gje, gks, gke;
                  if (ox1 < 0)      { gis = 0;             gie = ngh_l - 1; }
                  else if (ox1 > 0) { gis = ngh_l+ncells_l; gie = ngh_l+ncells_l+ngh_l-1; }
                  else { gis = ngh_l+sub_x*half; gie = ngh_l+sub_x*half+half-1; }
                  if (ox2 < 0)      { gjs = 0;             gje = ngh_l - 1; }
                  else if (ox2 > 0) { gjs = ngh_l+ncells_l; gje = ngh_l+ncells_l+ngh_l-1; }
                  else { gjs = ngh_l+sub_y*half; gje = ngh_l+sub_y*half+half-1; }
                  if (ox3 < 0)      { gks = 0;             gke = ngh_l - 1; }
                  else if (ox3 > 0) { gks = ngh_l+ncells_l; gke = ngh_l+ncells_l+ngh_l-1; }
                  else { gks = ngh_l+sub_z*half; gke = ngh_l+sub_z*half+half-1; }

                  for (int v = 0; v < nvar_l; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        for (int gi = gis; gi <= gie; ++gi) {
                          int fi0, fi1, fj0, fj1, fk0, fk1;
                          if (ox1 < 0) {
                            fi0 = ngh_l+ncells_l-2; fi1 = ngh_l+ncells_l-1;
                          } else if (ox1 > 0) {
                            fi0 = ngh_l; fi1 = ngh_l + 1;
                          } else {
                            fi0 = ngh_l+2*(gi-(ngh_l+sub_x*half)); fi1 = fi0+1;
                          }
                          if (ox2 < 0) {
                            fj0 = ngh_l+ncells_l-2; fj1 = ngh_l+ncells_l-1;
                          } else if (ox2 > 0) {
                            fj0 = ngh_l; fj1 = ngh_l + 1;
                          } else {
                            fj0 = ngh_l+2*(gj-(ngh_l+sub_y*half)); fj1 = fj0+1;
                          }
                          if (ox3 < 0) {
                            fk0 = ngh_l+ncells_l-2; fk1 = ngh_l+ncells_l-1;
                          } else if (ox3 > 0) {
                            fk0 = ngh_l; fk1 = ngh_l + 1;
                          } else {
                            fk0 = ngh_l+2*(gk-(ngh_l+sub_z*half)); fk1 = fk0+1;
                          }
                          u(m, v, gk, gj, gi) = remote_fc
                              ? ReadMGStage(rbuf[n], m, v, gk, gj, gi,
                                            sil, siu, sjl, sju, skl, sku)
                              : 0.125 * (
                                  u(dm,v,fk0,fj0,fi0) + u(dm,v,fk0,fj0,fi1) +
                                  u(dm,v,fk0,fj1,fi0) + u(dm,v,fk0,fj1,fi1) +
                                  u(dm,v,fk1,fj0,fi0) + u(dm,v,fk1,fj0,fi1) +
                                  u(dm,v,fk1,fj1,fi0) + u(dm,v,fk1,fj1,fi1));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
  });
  Kokkos::fence();

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PackAndSend()
//! \brief Pack restricted fluxes of multigrid variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors. Adapts to different block sizes per level.

TaskStatus MultigridBoundaryValues::PackAndSendMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = u.extent_int(1);

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  int shift_ = pmy_mg->GetLevelShift();
  int nx1_ = pmy_mg->GetSize();
  int ngh_ = pmy_mg->GetGhostCells();
  int ncells_ = nx1_ >> shift_;
  bool can_fc_ = pmy_mg->CanFillFineCoarseGhosts();

  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(m, n).gid < 0) continue;
      int mlev = mblev.h_view(m);
      int nlev = nghbr.h_view(m, n).lev;
      if (!can_fc_ && nlev != mlev) continue;

      MeshBufferIndcs bi = (nlev < mlev) ? sendbuf[n].icoar[0]
                         : (nlev == mlev) ? sendbuf[n].isame[0]
                                          : sendbuf[n].ifine[0];
      int il = bi.bis, iu = bi.bie;
      int jl = bi.bjs, ju = bi.bje;
      int kl = bi.bks, ku = bi.bke;
      int raw_il = il, raw_iu = iu;
      int raw_jl = jl, raw_ju = ju;
      int raw_kl = kl, raw_ku = ku;
      AdjustSendRangeForMG(il, iu, jl, ju, kl, ku, sendbuf[n], shift_, nx1_, ngh_);
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int data_size = nvar * ni * nj * nk;
      bool invalid = (ni < 0 || nj < 0 || nk < 0 ||
                      il < 0 || iu >= u.extent_int(4) ||
                      jl < 0 || ju >= u.extent_int(3) ||
                      kl < 0 || ku >= u.extent_int(2) ||
                      data_size > sendbuf[n].vars.extent_int(1));
      if (nghbr.h_view(m, n).rank == my_rank) {
        int dn = nghbr.h_view(m, n).dest;
        int dm = nghbr.h_view(m, n).gid - mbgid.h_view(0);
        invalid = invalid || dm < 0 || dm >= nmb ||
                  data_size > recvbuf[dn].vars.extent_int(1);
      }
      if (invalid) {
        std::cout << "### FATAL ERROR in MultigridBoundaryValues::PackAndSendMG"
                  << std::endl
                  << "MG boundary payload exceeds buffer capacity or local block range."
                  << std::endl
                  << "m=" << m << " n=" << n
                  << " mlev=" << mlev << " nlev=" << nlev
                  << " ncells=" << ncells_ << " shift=" << shift_
                  << " il=" << il << " iu=" << iu
                  << " jl=" << jl << " ju=" << ju
                  << " kl=" << kl << " ku=" << ku
                  << " raw_il=" << raw_il << " raw_iu=" << raw_iu
                  << " raw_jl=" << raw_jl << " raw_ju=" << raw_ju
                  << " raw_kl=" << raw_kl << " raw_ku=" << raw_ku
                  << " u_extents=(" << u.extent_int(2) << ","
                  << u.extent_int(3) << "," << u.extent_int(4) << ")"
                  << " data_size=" << data_size
                  << " send_capacity=" << sendbuf[n].vars.extent_int(1)
                  << " rank=" << nghbr.h_view(m, n).rank
                  << " local_rank=" << my_rank
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("PackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid >= 0) {
      int mlev = mblev.d_view(m);
      int nlev = nghbr.d_view(m, n).lev;
      if (!can_fc_ && nlev != mlev) return;
      MeshBufferIndcs bi = (nlev < mlev) ? sbuf[n].icoar[0]
                         : (nlev == mlev) ? sbuf[n].isame[0]
                                          : sbuf[n].ifine[0];
      int il = bi.bis, iu = bi.bie;
      int jl = bi.bjs, ju = bi.bje;
      int kl = bi.bks, ku = bi.bke;
      AdjustSendRangeForMG(il, iu, jl, ju, kl, ku, sbuf[n], shift_, nx1_, ngh_);

      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj = nk * nj;

      int dm = nghbr.d_view(m, n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m, n).dest;
      if (nghbr.d_view(m, n).rank == my_rank &&
          (dm < 0 || dm >= nmb || dn < 0 || dn >= nnghbr)) {
        return;
      }

      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        if (nghbr.d_view(m, n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            Real val = u(m, v, k, j, i);
            if (nlev < mlev) {
              int ox1, ox2, ox3, f1, f2;
              DecodeNeighborIndexMG(n, ox1, ox2, ox3, f1, f2);
              int nface = (ox1 != 0 ? 1 : 0) + (ox2 != 0 ? 1 : 0) + (ox3 != 0 ? 1 : 0);
              int i0 = ngh_ + 2*(i - ngh_);
              int j0 = ngh_ + 2*(j - ngh_);
              int k0 = ngh_ + 2*(k - ngh_);
              if (i0 < ngh_) i0 = ngh_;
              if (j0 < ngh_) j0 = ngh_;
              if (k0 < ngh_) k0 = ngh_;
              if (i0 > ngh_ + ncells_ - 2) i0 = ngh_ + ncells_ - 2;
              if (j0 > ngh_ + ncells_ - 2) j0 = ngh_ + ncells_ - 2;
              if (k0 > ngh_ + ncells_ - 2) k0 = ngh_ + ncells_ - 2;
              if (nface == 1 && ox1 != 0) {
                int fi = (ox1 < 0) ? ngh_ : ngh_ + ncells_ - 1;
                val = 0.25*(u(m,v,k0,j0,fi) + u(m,v,k0,j0+1,fi)
                           +u(m,v,k0+1,j0,fi) + u(m,v,k0+1,j0+1,fi));
              } else if (nface == 1 && ox2 != 0) {
                int fj = (ox2 < 0) ? ngh_ : ngh_ + ncells_ - 1;
                val = 0.25*(u(m,v,k0,fj,i0) + u(m,v,k0,fj,i0+1)
                           +u(m,v,k0+1,fj,i0) + u(m,v,k0+1,fj,i0+1));
              } else if (nface == 1 && ox3 != 0) {
                int fk = (ox3 < 0) ? ngh_ : ngh_ + ncells_ - 1;
                val = 0.25*(u(m,v,fk,j0,i0) + u(m,v,fk,j0,i0+1)
                           +u(m,v,fk,j0+1,i0) + u(m,v,fk,j0+1,i0+1));
              } else {
                val = 0.125*(u(m,v,k0,j0,i0) + u(m,v,k0,j0,i0+1)
                            +u(m,v,k0,j0+1,i0) + u(m,v,k0,j0+1,i0+1)
                            +u(m,v,k0+1,j0,i0) + u(m,v,k0+1,j0,i0+1)
                            +u(m,v,k0+1,j0+1,i0) + u(m,v,k0+1,j0+1,i0+1));
              }
            }
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = val;
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            Real val = u(m, v, k, j, i);
            if (nlev < mlev) {
              int ox1, ox2, ox3, f1, f2;
              DecodeNeighborIndexMG(n, ox1, ox2, ox3, f1, f2);
              int nface = (ox1 != 0 ? 1 : 0) + (ox2 != 0 ? 1 : 0) + (ox3 != 0 ? 1 : 0);
              int i0 = ngh_ + 2*(i - ngh_);
              int j0 = ngh_ + 2*(j - ngh_);
              int k0 = ngh_ + 2*(k - ngh_);
              if (i0 < ngh_) i0 = ngh_;
              if (j0 < ngh_) j0 = ngh_;
              if (k0 < ngh_) k0 = ngh_;
              if (i0 > ngh_ + ncells_ - 2) i0 = ngh_ + ncells_ - 2;
              if (j0 > ngh_ + ncells_ - 2) j0 = ngh_ + ncells_ - 2;
              if (k0 > ngh_ + ncells_ - 2) k0 = ngh_ + ncells_ - 2;
              if (nface == 1 && ox1 != 0) {
                int fi = (ox1 < 0) ? ngh_ : ngh_ + ncells_ - 1;
                val = 0.25*(u(m,v,k0,j0,fi) + u(m,v,k0,j0+1,fi)
                           +u(m,v,k0+1,j0,fi) + u(m,v,k0+1,j0+1,fi));
              } else if (nface == 1 && ox2 != 0) {
                int fj = (ox2 < 0) ? ngh_ : ngh_ + ncells_ - 1;
                val = 0.25*(u(m,v,k0,fj,i0) + u(m,v,k0,fj,i0+1)
                           +u(m,v,k0+1,fj,i0) + u(m,v,k0+1,fj,i0+1));
              } else if (nface == 1 && ox3 != 0) {
                int fk = (ox3 < 0) ? ngh_ : ngh_ + ncells_ - 1;
                val = 0.25*(u(m,v,fk,j0,i0) + u(m,v,fk,j0,i0+1)
                           +u(m,v,fk,j0+1,i0) + u(m,v,fk,j0+1,i0+1));
              } else {
                val = 0.125*(u(m,v,k0,j0,i0) + u(m,v,k0,j0,i0+1)
                            +u(m,v,k0,j0+1,i0) + u(m,v,k0,j0+1,i0+1)
                            +u(m,v,k0+1,j0,i0) + u(m,v,k0+1,j0,i0+1)
                            +u(m,v,k0+1,j0+1,i0) + u(m,v,k0+1,j0+1,i0+1));
              }
            }
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = val;
          });
        }
      });
    }
    tmember.team_barrier();
  });
  }

  #if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        int mlev = pmy_pack->pmb->mb_lev.h_view(m);
        int nlev = nghbr.h_view(m,n).lev;
        if (!can_fc_ && nlev != mlev) continue;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          int data_size = nvar;
          int ndat = (nlev < mlev) ? sendbuf[n].icoar_ndat
                   : (nlev == mlev) ? sendbuf[n].isame_ndat
                                    : sendbuf[n].ifine_ndat;
          data_size *= AdjustMGBufferSizeHost(sendbuf[n], ndat, shift_);
          if (data_size > sendbuf[n].vars.extent_int(1)) {
            std::cout << "### FATAL ERROR in MultigridBoundaryValues::PackAndSendMG"
                      << std::endl
                      << "MG send buffer is too small for fine/coarse payload."
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }

          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);
          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValuesCC::RecvAndUnpackMG()
//! \brief Receive and unpack cell-centered multigrid variables.
//! Handles ghost-cell filling at each multigrid level independently.

TaskStatus MultigridBoundaryValues::RecvAndUnpackMG(DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recvbuf;
  int shift_ = pmy_mg->GetLevelShift();
  bool can_fc_ = pmy_mg->CanFillFineCoarseGhosts();
  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          if (!can_fc_ && nghbr.h_view(m,n).lev != mblev.h_view(m)) continue;
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack
  int nvar = u.extent_int(1);
  int ngh = pmy_mg->GetGhostCells();
  fc_stage_nvars_ = nvar;
  fc_stage_level_ = pmy_mg->GetCurrentLevel();
  fc_stage_role_ = (pmy_mg->ncoeff_ > 0 && nvar == pmy_mg->ncoeff_) ? 1 : 0;
  fc_stage_valid_ = can_fc_;

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("UnpackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid >= 0 &&
        nghbr.d_view(m, n).lev == mblev.d_view(m)) {
      int il = rbuf[n].isame[0].bis;
      int iu = rbuf[n].isame[0].bie;
      int jl = rbuf[n].isame[0].bjs;
      int ju = rbuf[n].isame[0].bje;
      int kl = rbuf[n].isame[0].bks;
      int ku = rbuf[n].isame[0].bke;

      AdjustRecvRangeForMG(il, iu, jl, ju, kl, ku, rbuf[n], shift_, ngh);

      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj = nk * nj;

      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        [&](const int i) {
          u(m, v, k, j, i) = rbuf[n].vars(m,
              (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
        });
      });
    }
    tmember.team_barrier();
  });
  }
  Kokkos::fence();

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MultigridBoundaryValues::InitRecvMG(const int nvars) {
#if MPI_PARALLEL_ENABLED
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int shift_ = pmy_mg->GetLevelShift();
  bool can_fc_ = pmy_mg->CanFillFineCoarseGhosts();
  fc_stage_valid_ = false;

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars;
          int mlev = mblev.h_view(m);
          int nlev = nghbr.h_view(m,n).lev;
          if (!can_fc_ && nlev != mlev) continue;
          int ndat = (nlev < mlev) ? recvbuf[n].icoar_ndat
                   : (nlev == mlev) ? recvbuf[n].isame_ndat
                                    : recvbuf[n].ifine_ndat;
          data_size *= AdjustMGBufferSizeHost(recvbuf[n], ndat, shift_);
          if (data_size > recvbuf[n].vars.extent_int(1)) {
            std::cout << "### FATAL ERROR in MultigridBoundaryValues::InitRecvMG"
                      << std::endl
                      << "MG receive buffer is too small for fine/coarse payload."
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }

          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, Kokkos::ALL);

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(recvbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

void Multigrid::PrintActiveRegion(const DvceArray5D<Real> &u_in) {
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_in);
  int ll = nlevel_ - 1 - current_level_;
  int ngh = ngh_;  // number of ghost cells
  
  int is = ngh, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh, ke = ks + (indcs_.nx3 >> ll) - 1;
  std::cout<<"nrbx1="<<nmmbx1_<<", nrbx2="<<nmmbx2_<<", nrbx3="<<nmmbx3_<<std::endl;  
  std::cout << "Active region at level " << current_level_ << " (nx=" << (indcs_.nx1 >> ll) << ")\n";
  std::cout << "Range: i=[" << is << "," << ie << "], j=[" << js << "," << je 
            << "], k=[" << ks << "," << ke << "]\n";
  std::cout << "[";
  for (int mz = 0; mz < nmmbx3_/global_variable::nranks; ++mz) {
  for (int k = ks; k <= ks+((ke-ks)/(3-global_variable::nranks)); ++k) {
        std::cout << "[";
        for (int my=0; my < nmmbx2_; ++my) {
          for (int j = js; j <= je; ++j) {
            std::cout << "[";
            for (int mx= 0; mx < nmmbx1_; ++mx) {
              for (int i = is; i <= ie; ++i){
                std::cout << std::setprecision(3) << u_h(mx+my*2+mz*4, 0, k, j, i) << ", ";
              }
            }
            std::cout << "],";
            std::cout << "\n";
          }
        }
        std::cout << "],";
        std::cout << "\n";
    }
  }
  std::cout << "]";
  return;
}

void Multigrid::PrintAll(const DvceArray5D<Real> &u_in) {
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_in);
  int ll = nlevel_ - 1 - current_level_;
  int ngh = ngh_;  // number of ghost cells
  
  int is = 2, ie = is + (indcs_.nx1 >> ll) +2 *ngh - 5;
  int js = 2, je = js + (indcs_.nx2 >> ll) +2 *ngh - 5;
  int ks = 2, ke = ks + (indcs_.nx3 >> ll) +2 *ngh - 5;
  //std::cout<<"nrbx1="<<nmmbx1_<<", nrbx2="<<nmmbx2_<<", nrbx3="<<nmmbx3_<<std::endl;  
  //std::cout << "Whole domain at level " << current_level_ << " (nx=" << (indcs_.nx1 >> ll) << ")\n";
  //std::cout << "Range: i=[" << is << "," << ie << "], j=[" << js << "," << je 
  //          << "], k=[" << ks << "," << ke << "]\n";
  for (int mz = 0; mz < nmmbx3_; ++mz) {
  for (int k = ks+mz; k <= ke+(1-nmmbx3_)+mz; ++k) {
        for (int my=0; my < nmmbx2_; ++my) {
          for (int j = js+my; j <= je+(1-nmmbx2_)+my; ++j) {
            for (int mx= 0; mx < nmmbx1_; ++mx) {
              for (int i = is+mx; i <= ie+(1-nmmbx1_)+mx; ++i){
                std::cout << std::setprecision(3) << u_h(mx+my*2+mz*4, 0, k, j, i) << ", ";
              }
            }
            std::cout << "],";
            std::cout << "\n";
          }
        }
        std::cout << "],";
        std::cout << "\n";
    }
  }
  return;
}
