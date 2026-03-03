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

//namespace multigrid{ // NOLINT (build/namespace)
//----------------------------------------------------------------------------------------
//! \fn Multigrid::Multigrid(MultigridDriver *pmd, MeshBlock *pmb, int nghost)
//  \brief Multigrid constructor

Multigrid::Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host):
  pmy_driver_(pmd), pmy_pack_(pmbp), pmy_mesh_(pmd->pmy_mesh_), ngh_(nghost),
  nvar_(pmd->nvar_), defscale_(1.0), on_host_(on_host)  {
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

  for (int l = 0; l < nlevel_; l++) {
    int ll=nlevel_-1-l;
    int ncx=(indcs_.nx1>>ll)+2*ngh_;
    int ncy=(indcs_.nx2>>ll)+2*ngh_;
    int ncz=(indcs_.nx3>>ll)+2*ngh_;
    Kokkos::realloc(u_[l]  , nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(src_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(def_[l], nmmb_, nvar_, ncz, ncy, ncx);

    if (!((pmy_pack_ != nullptr) && (l == nlevel_-1)))
      Kokkos::realloc(uold_[l], nmmb_, nvar_, ncz, ncy, ncx);

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
}

void Multigrid::ReallocateForAMR() {
  if (pmy_pack_ == nullptr) return;
  int new_nmmb = pmy_pack_->nmb_thispack;
  if (new_nmmb == nmmb_) return;
  nmmb_ = new_nmmb;

  Kokkos::realloc(block_rdx_, nmmb_);
  UpdateBlockDx();

  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_ - 1 - l;
    int ncx = (indcs_.nx1 >> ll) + 2 * ngh_;
    int ncy = (indcs_.nx2 >> ll) + 2 * ngh_;
    int ncz = (indcs_.nx3 >> ll) + 2 * ngh_;
    Kokkos::realloc(u_[l],   nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(src_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(def_[l], nmmb_, nvar_, ncz, ncy, ncx);
    if (l != nlevel_ - 1)
      Kokkos::realloc(uold_[l], nmmb_, nvar_, ncz, ncy, ncx);
  }

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

  return;
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
  if (on_host_) {
    for (int lev = nlevel_ - 1; lev > 0; lev--) {
      int ll = nlevel_ - lev;
      ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
      Restrict(coeff_[lev-1].h_view, coeff_[lev].h_view, ncoeff_,
               is, ie, js, je, ks, ke, false);
    }
  } else {
    for (int lev = nlevel_ - 1; lev > 0; lev--) {
      int ll = nlevel_ - lev;
      ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
      Restrict(coeff_[lev-1].d_view, coeff_[lev].d_view, ncoeff_,
               is, ie, js, je, ks, ke, false);
    }
  }
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

  ComputeCorrection();

  if (on_host_) {
    ProlongateAndCorrect(u_[current_level_+1].h_view, u_[current_level_].h_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  } else {
    ProlongateAndCorrect(u_[current_level_+1].d_view, u_[current_level_].d_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  }

  current_level_++;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongatePack()
//! \brief Prolongate the solution for FMG (direct overwrite, always tricubic)

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
      int oid = pmy_driver_->octetmap_[olev][oloc];
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
  Real dx = rdx_ * static_cast<Real>(1 << ll);
  Real dy = rdy_ * static_cast<Real>(1 << ll);
  Real dz = rdz_ * static_cast<Real>(1 << ll);
  Real dV = dx * dy * dz;
  CalculateDefectPack();

  Real norm = 0.0;

  if (on_host_) {
    auto &def = def_[current_level_].h_view;
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
          local_sum += std::abs(def(m, v, k, j, i));
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real val = def(m, v, k, j, i);
          local_sum += val * val;
        }, Kokkos::Sum<Real>(norm));
    }
  } else {
    auto &def = def_[current_level_].d_view;
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
          local_sum += std::abs(def(m, v, k, j, i));
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real val = def(m, v, k, j, i);
          local_sum += val * val;
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

  const int ll = pmy_driver_->fprolongation_; // copy host flag for capture

  auto dst_ = dst;
  auto src_ = src;

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
      ) / 32768.0;

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
      ) / 32768.0;

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
      ) / 32768.0;

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
      ) / 32768.0;

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
      ) / 32768.0;

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
      ) / 32768.0;

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
      ) / 32768.0;

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
      ) / 32768.0;  
    });
  } else { // trilinear
    par_for("Multigrid::ProlongateAndCorrect_trilinear", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      dst_(m,v,fk  ,fj  ,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk  ,fj  ,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk+1,fj+1,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j+1,i+1)));
      dst_(m,v,fk+1,fj+1,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j+1,i+1)));
    });
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongate(...)
//! \brief FMG prolongation: direct overwrite (=) with tricubic interpolation.
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

  auto dst_ = dst;
  auto src_ = src;

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

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &lloc = pmy_pack->pmesh->lloc_eachmb;

  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u);
  bool modified = false;
  constexpr Real ot = 1.0/3.0;

  for (int m = 0; m < nmb; ++m) {
    int m_lev = mblev.h_view(m);
    int m_gid = mbgid.h_view(m);
    LogicalLocation m_loc = lloc[m_gid];

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);

          for (int f2 = 0; f2 <= 1; ++f2) {
            for (int f1 = 0; f1 <= 1; ++f1) {
              int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
              if (n < 0 || n >= nnghbr) continue;
              if (nghbr.h_view(m, n).gid < 0) continue;

              int nlev = nghbr.h_view(m, n).lev;
              if (nlev == m_lev) continue;
              if (nghbr.h_view(m, n).rank != my_rank) continue;

              int dm = nghbr.h_view(m, n).gid - mbgid.h_view(0);
              if (dm < 0 || dm >= nmb) continue;

              int child_x = m_loc.lx1 & 1;
              int child_y = m_loc.lx2 & 1;
              int child_z = m_loc.lx3 & 1;
              int half = ncells / 2;

              if (nlev < m_lev && nface == 1) {
                if (ox1 != 0) {
                  int fig = (ox1 < 0) ? ngh - 1 : ngh + ncells;
                  int fi  = (ox1 < 0) ? ngh : ngh + ncells - 1;
                  int si  = (ox1 < 0) ? ngh + ncells - 1 : ngh;
                  int sj0 = ngh + child_y * half;
                  int sk0 = ngh + child_z * half;
                  for (int v = 0; v < nvar; ++v) {
                    for (int sk = sk0; sk < sk0 + half; ++sk) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        int fj = ngh + 2*(sj - sj0);
                        int fk = ngh + 2*(sk - sk0);
                        Real cc = u_h(dm,v,sk,sj,si);
                        int sjm = (sj > ngh) ? sj-1 : sj;
                        int sjp = (sj < ngh+ncells-1) ? sj+1 : sj;
                        int skm = (sk > ngh) ? sk-1 : sk;
                        int skp = (sk < ngh+ncells-1) ? sk+1 : sk;
                        Real gy = 0.125*(u_h(dm,v,sk,sjp,si)-u_h(dm,v,sk,sjm,si));
                        Real gz = 0.125*(u_h(dm,v,skp,sj,si)-u_h(dm,v,skm,sj,si));
                        u_h(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u_h(m,v,fk  ,fj  ,fi));
                        u_h(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u_h(m,v,fk  ,fj+1,fi));
                        u_h(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u_h(m,v,fk+1,fj  ,fi));
                        u_h(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u_h(m,v,fk+1,fj+1,fi));
                      }
                    }
                  }
                } else if (ox2 != 0) {
                  int fjg = (ox2 < 0) ? ngh - 1 : ngh + ncells;
                  int fj  = (ox2 < 0) ? ngh : ngh + ncells - 1;
                  int sj  = (ox2 < 0) ? ngh + ncells - 1 : ngh;
                  int si0 = ngh + child_x * half;
                  int sk0 = ngh + child_z * half;
                  for (int v = 0; v < nvar; ++v) {
                    for (int sk = sk0; sk < sk0 + half; ++sk) {
                      for (int si = si0; si < si0 + half; ++si) {
                        int fi = ngh + 2*(si - si0);
                        int fk = ngh + 2*(sk - sk0);
                        Real cc = u_h(dm,v,sk,sj,si);
                        int sim = (si > ngh) ? si-1 : si;
                        int sip = (si < ngh+ncells-1) ? si+1 : si;
                        int skm = (sk > ngh) ? sk-1 : sk;
                        int skp = (sk < ngh+ncells-1) ? sk+1 : sk;
                        Real gx = 0.125*(u_h(dm,v,sk,sj,sip)-u_h(dm,v,sk,sj,sim));
                        Real gz = 0.125*(u_h(dm,v,skp,sj,si)-u_h(dm,v,skm,sj,si));
                        u_h(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u_h(m,v,fk  ,fj,fi  ));
                        u_h(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u_h(m,v,fk  ,fj,fi+1));
                        u_h(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u_h(m,v,fk+1,fj,fi  ));
                        u_h(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u_h(m,v,fk+1,fj,fi+1));
                      }
                    }
                  }
                } else {
                  int fkg = (ox3 < 0) ? ngh - 1 : ngh + ncells;
                  int fk  = (ox3 < 0) ? ngh : ngh + ncells - 1;
                  int sk  = (ox3 < 0) ? ngh + ncells - 1 : ngh;
                  int si0 = ngh + child_x * half;
                  int sj0 = ngh + child_y * half;
                  for (int v = 0; v < nvar; ++v) {
                    for (int sj = sj0; sj < sj0 + half; ++sj) {
                      for (int si = si0; si < si0 + half; ++si) {
                        int fi = ngh + 2*(si - si0);
                        int fj = ngh + 2*(sj - sj0);
                        Real cc = u_h(dm,v,sk,sj,si);
                        int sim = (si > ngh) ? si-1 : si;
                        int sip = (si < ngh+ncells-1) ? si+1 : si;
                        int sjm = (sj > ngh) ? sj-1 : sj;
                        int sjp = (sj < ngh+ncells-1) ? sj+1 : sj;
                        Real gx = 0.125*(u_h(dm,v,sk,sj,sip)-u_h(dm,v,sk,sj,sim));
                        Real gy = 0.125*(u_h(dm,v,sk,sjp,si)-u_h(dm,v,sk,sjm,si));
                        u_h(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u_h(m,v,fk,fj  ,fi  ));
                        u_h(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u_h(m,v,fk,fj  ,fi+1));
                        u_h(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u_h(m,v,fk,fj+1,fi  ));
                        u_h(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u_h(m,v,fk,fj+1,fi+1));
                      }
                    }
                  }
                }
                modified = true;

              } else if (nlev > m_lev && nface == 1) {
                // ==== FINER neighbor, FACE: flux-conserving restriction ====
                // face_avg = area-average of 4 fine cells on the shared face
                // coarse_ghost = (1/3)*(4*face_avg - coarse_interior)
                int sub_x = 0, sub_y = 0, sub_z = 0;
                if (ox1 != 0) { sub_y = f1; sub_z = f2; }
                if (ox2 != 0) { sub_x = f1; sub_z = f2; }
                if (ox3 != 0) { sub_x = f1; sub_y = f2; }

                int gis, gie, gjs, gje, gks, gke;
                if (ox1 < 0)      { gis = 0;            gie = ngh - 1; }
                else if (ox1 > 0) { gis = ngh + ncells;  gie = ngh + ncells + ngh - 1; }
                else { gis = ngh + sub_x*half; gie = ngh + sub_x*half + half - 1; }
                if (ox2 < 0)      { gjs = 0;            gje = ngh - 1; }
                else if (ox2 > 0) { gjs = ngh + ncells;  gje = ngh + ncells + ngh - 1; }
                else { gjs = ngh + sub_y*half; gje = ngh + sub_y*half + half - 1; }
                if (ox3 < 0)      { gks = 0;            gke = ngh - 1; }
                else if (ox3 > 0) { gks = ngh + ncells;  gke = ngh + ncells + ngh - 1; }
                else { gks = ngh + sub_z*half; gke = ngh + sub_z*half + half - 1; }

                int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                if (ox1 != 0) {
                  int fi = (ox1 > 0) ? ngh : ngh + ncells - 1;
                  for (int v = 0; v < nvar; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        int fj0 = ngh + 2*(gj - (ngh + sub_y*half));
                        int fk0 = ngh + 2*(gk - (ngh + sub_z*half));
                        Real favg = 0.25*(u_h(dm,v,fk0,fj0,fi)+u_h(dm,v,fk0,fj0+1,fi)
                                         +u_h(dm,v,fk0+1,fj0,fi)+u_h(dm,v,fk0+1,fj0+1,fi));
                        u_h(m,v,gk,gj,gis) = ot*(4.0*favg - u_h(m,v,gk+ok,gj+oj,gis+oi));
                      }
                    }
                  }
                } else if (ox2 != 0) {
                  int fj = (ox2 > 0) ? ngh : ngh + ncells - 1;
                  for (int v = 0; v < nvar; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int fi0 = ngh + 2*(gi - (ngh + sub_x*half));
                        int fk0 = ngh + 2*(gk - (ngh + sub_z*half));
                        Real favg = 0.25*(u_h(dm,v,fk0,fj,fi0)+u_h(dm,v,fk0,fj,fi0+1)
                                         +u_h(dm,v,fk0+1,fj,fi0)+u_h(dm,v,fk0+1,fj,fi0+1));
                        u_h(m,v,gk,gjs,gi) = ot*(4.0*favg - u_h(m,v,gk+ok,gjs+oj,gi+oi));
                      }
                    }
                  }
                } else {
                  int fk = (ox3 > 0) ? ngh : ngh + ncells - 1;
                  for (int v = 0; v < nvar; ++v) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int fi0 = ngh + 2*(gi - (ngh + sub_x*half));
                        int fj0 = ngh + 2*(gj - (ngh + sub_y*half));
                        Real favg = 0.25*(u_h(dm,v,fk,fj0,fi0)+u_h(dm,v,fk,fj0,fi0+1)
                                         +u_h(dm,v,fk,fj0+1,fi0)+u_h(dm,v,fk,fj0+1,fi0+1));
                        u_h(m,v,gks,gj,gi) = ot*(4.0*favg - u_h(m,v,gks+ok,gj+oj,gi+oi));
                      }
                    }
                  }
                }
                modified = true;

              } else if (nlev < m_lev) {
                // ==== COARSER neighbor, EDGE/CORNER: simple injection ====
                int gis, gie, gjs, gje, gks, gke;
                if (ox1 < 0)      { gis = 0;            gie = ngh - 1; }
                else if (ox1 > 0) { gis = ngh + ncells;  gie = ngh + ncells + ngh - 1; }
                else              { gis = ngh;            gie = ngh + ncells - 1; }
                if (ox2 < 0)      { gjs = 0;            gje = ngh - 1; }
                else if (ox2 > 0) { gjs = ngh + ncells;  gje = ngh + ncells + ngh - 1; }
                else              { gjs = ngh;            gje = ngh + ncells - 1; }
                if (ox3 < 0)      { gks = 0;            gke = ngh - 1; }
                else if (ox3 > 0) { gks = ngh + ncells;  gke = ngh + ncells + ngh - 1; }
                else              { gks = ngh;            gke = ngh + ncells - 1; }

                for (int v = 0; v < nvar; ++v) {
                  for (int gk = gks; gk <= gke; ++gk) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int si, sj, sk;
                        if (ox1 < 0)      si = ngh + ncells - 1;
                        else if (ox1 > 0) si = ngh;
                        else si = ngh + child_x*(half) + (gi - ngh)/2;
                        if (ox2 < 0)      sj = ngh + ncells - 1;
                        else if (ox2 > 0) sj = ngh;
                        else sj = ngh + child_y*(half) + (gj - ngh)/2;
                        if (ox3 < 0)      sk = ngh + ncells - 1;
                        else if (ox3 > 0) sk = ngh;
                        else sk = ngh + child_z*(half) + (gk - ngh)/2;

                        u_h(m, v, gk, gj, gi) = u_h(dm, v, sk, sj, si);
                      }
                    }
                  }
                }
                modified = true;

              } else {
                int sub_x = 0, sub_y = 0, sub_z = 0;
                if (nface == 2) {
                  if (ox1 == 0) sub_x = f1;
                  if (ox2 == 0) sub_y = f1;
                  if (ox3 == 0) sub_z = f1;
                }
                int gis, gie, gjs, gje, gks, gke;
                if (ox1 < 0)      { gis = 0;            gie = ngh - 1; }
                else if (ox1 > 0) { gis = ngh + ncells;  gie = ngh + ncells + ngh - 1; }
                else { gis = ngh + sub_x*half; gie = ngh + sub_x*half + half - 1; }
                if (ox2 < 0)      { gjs = 0;            gje = ngh - 1; }
                else if (ox2 > 0) { gjs = ngh + ncells;  gje = ngh + ncells + ngh - 1; }
                else { gjs = ngh + sub_y*half; gje = ngh + sub_y*half + half - 1; }
                if (ox3 < 0)      { gks = 0;            gke = ngh - 1; }
                else if (ox3 > 0) { gks = ngh + ncells;  gke = ngh + ncells + ngh - 1; }
                else { gks = ngh + sub_z*half; gke = ngh + sub_z*half + half - 1; }

                for (int v = 0; v < nvar; ++v) {
                  for (int gk = gks; gk <= gke; ++gk) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int fi0, fi1, fj0, fj1, fk0, fk1;
                        if (ox1 < 0) {
                          fi0 = ngh + ncells - 2; fi1 = ngh + ncells - 1;
                        } else if (ox1 > 0) {
                          fi0 = ngh; fi1 = ngh + 1;
                        } else {
                          fi0 = ngh + 2*(gi - (ngh + sub_x*half)); fi1 = fi0 + 1;
                        }
                        if (ox2 < 0) {
                          fj0 = ngh + ncells - 2; fj1 = ngh + ncells - 1;
                        } else if (ox2 > 0) {
                          fj0 = ngh; fj1 = ngh + 1;
                        } else {
                          fj0 = ngh + 2*(gj - (ngh + sub_y*half)); fj1 = fj0 + 1;
                        }
                        if (ox3 < 0) {
                          fk0 = ngh + ncells - 2; fk1 = ngh + ncells - 1;
                        } else if (ox3 > 0) {
                          fk0 = ngh; fk1 = ngh + 1;
                        } else {
                          fk0 = ngh + 2*(gk - (ngh + sub_z*half)); fk1 = fk0 + 1;
                        }
                        u_h(m, v, gk, gj, gi) = 0.125 * (
                          u_h(dm,v,fk0,fj0,fi0) + u_h(dm,v,fk0,fj0,fi1) +
                          u_h(dm,v,fk0,fj1,fi0) + u_h(dm,v,fk0,fj1,fi1) +
                          u_h(dm,v,fk1,fj0,fi0) + u_h(dm,v,fk1,fj0,fi1) +
                          u_h(dm,v,fk1,fj1,fi0) + u_h(dm,v,fk1,fj1,fi1));
                      }
                    }
                  }
                }
                modified = true;
              }
            }
          }
        }
      }
    }
  }

  if (modified) {
    Kokkos::deep_copy(u, u_h);
  }

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

  {
  int nmnv = nmb * nnghbr * nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("PackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank() / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
    const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

    if (nghbr.d_view(m, n).gid >= 0 &&
        nghbr.d_view(m, n).lev == mblev.d_view(m)) {
      int il = sbuf[n].isame[0].bis;
      int iu = sbuf[n].isame[0].bie;
      int jl = sbuf[n].isame[0].bjs;
      int ju = sbuf[n].isame[0].bje;
      int kl = sbuf[n].isame[0].bks;
      int ku = sbuf[n].isame[0].bke;

      int sh = shift_;
      int nx = nx1_;
    
      while (sh > 0) {
        if (sbuf[n].faces.d_view(0) && (il == nx)) {
          int d = iu - il; il = il >> 1; iu = il + d;
        } else if (!sbuf[n].faces.d_view(0)) {
          iu = ((iu - il) >> 1) + il;
        }
        if (sbuf[n].faces.d_view(1) && (jl == nx)) {
          int d = ju - jl; jl = jl >> 1; ju = jl + d;
        } else if (!sbuf[n].faces.d_view(1)) {
          ju = ((ju - jl) >> 1) + jl;
        }
        if (sbuf[n].faces.d_view(2) && (kl == nx)) {
          int d = ku - kl; kl = kl >> 1; ku = kl + d;
        } else if (!sbuf[n].faces.d_view(2)) {
          ku = ((ku - kl) >> 1) + kl;
        }
        sh--;
        nx = nx >> 1;
      }

      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj = nk * nj;

      int dm = nghbr.d_view(m, n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m, n).dest;

      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
      [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        if (nghbr.d_view(m, n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = u(m, v, k, j, i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))))
                = u(m, v, k, j, i);
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
      if (nghbr.h_view(m,n).gid >= 0
          && nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m)) {
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          int data_size = nvar;
          data_size *= sendbuf[n].isame_ndat;
          
          if (not(sendbuf[n].faces.h_view(0)))
            data_size >>= shift_;
          if (not(sendbuf[n].faces.h_view(1)))
            data_size >>= shift_;
          if (not(sendbuf[n].faces.h_view(2)))
            data_size >>= shift_;

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
  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0
          && nghbr.h_view(m,n).lev == mblev.h_view(m)) {
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
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
  MPI_Barrier(comm_vars);
#endif

  //----- STEP 2: buffers have all completed, so unpack
  int nvar = u.extent_int(1);
  int ngh = pmy_mg->GetGhostCells();

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

      int sh = shift_;
      while (sh > 0) {
        if (rbuf[n].faces.d_view(0) && il > 1) {
          int d = iu - il; il = (il + ngh) >> 1; iu = il + d;
        } else if (!rbuf[n].faces.d_view(0)) {
          iu = ((iu - il) >> 1) + il;
        }
        if (rbuf[n].faces.d_view(1) && jl > 1) {
          int d = ju - jl; jl = (jl + ngh) >> 1; ju = jl + d;
        } else if (!rbuf[n].faces.d_view(1)) {
          ju = ((ju - jl) >> 1) + jl;
        }
        if (rbuf[n].faces.d_view(2) && kl > 1) {
          int d = ku - kl; kl = (kl + ngh) >> 1; ku = kl + d;
        } else if (!rbuf[n].faces.d_view(2)) {
          ku = ((ku - kl) >> 1) + kl;
        }
        sh--;
      }

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

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0
          && nghbr.h_view(m,n).lev == mblev.h_view(m)) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars;
          data_size *= sendbuf[n].isame_ndat;
          
          if (not(recvbuf[n].faces.h_view(0)))
            data_size >>= shift_;
          if (not(recvbuf[n].faces.h_view(1)))
            data_size >>= shift_;
          if (not(recvbuf[n].faces.h_view(2)))
            data_size >>= shift_;

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