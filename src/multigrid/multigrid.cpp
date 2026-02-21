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
#include "../mesh/mesh.hpp"
#include "../mesh/nghbr_index.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

//namespace multigrid{ // NOLINT (build/namespace)
//----------------------------------------------------------------------------------------
//! \fn Multigrid::Multigrid(MultigridDriver *pmd, MeshBlock *pmb, int nghost)
//  \brief Multigrid constructor

Multigrid::Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost):
  pmy_driver_(pmd), pmy_pack_(pmbp), pmy_mesh_(pmd->pmy_mesh_), ngh_(nghost), nvar_(pmd->nvar_), defscale_(1.0)  {
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

  Kokkos::realloc(block_rdx_, nmmb_);
  {
    auto brdx_h = Kokkos::create_mirror_view(block_rdx_);
    if (pmy_pack_ != nullptr) {
      auto &mb_size = pmy_pack_->pmb->mb_size;
      Real rnx1 = static_cast<Real>(indcs_.nx1);
      for (int m = 0; m < nmmb_; ++m) {
        brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
      }
    } else {
      brdx_h(0) = rdx_;
    }
    Kokkos::deep_copy(block_rdx_, brdx_h);
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
  u_ = new DvceArray5D<Real>[nlevel_];
  src_ = new DvceArray5D<Real>[nlevel_];
  def_ = new DvceArray5D<Real>[nlevel_];
  coeff_ = new DvceArray5D<Real>[nlevel_];
  matrix_ = new DvceArray5D<Real>[nlevel_];
  uold_ = new DvceArray5D<Real>[nlevel_];

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
//! \fn void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh)
//! \brief Fill the inital guess in the active zone of the finest level

void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh) {
  DvceArray5D<Real> &dst = u_[nlevel_-1];
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1 - 1; je = js + indcs_.nx2 - 1; ke = ks + indcs_.nx3 - 1;

  // copy locals for safe capture in device lambda
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
  // ngh is the number of ghost zones in src (hydro)
  // ngh_ is the number of ghost zones in dst (multigrid)
  // Copy active zone + min(ngh_,ngh) ghost cells, aligning active zones.

  auto &dst = src_[nlevel_-1];
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
  DvceArray5D<Real> &cm = coeff_[nlevel_-1];
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
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1;
  je = js + indcs_.nx2;
  ke = ks + indcs_.nx3;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictCoefficients()
//! \brief restrict coefficients within a Multigrid object

void Multigrid::RestrictCoefficients() {
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  for (int lev = nlevel_ - 1; lev > 0; lev--) {
    int ll = nlevel_ - lev;
    ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
    Restrict(coeff_[lev-1], coeff_[lev], ncoeff_, is, ie, js, je, ks, ke, false);
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh)
//! \brief Set the result, including the ghost zone

void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh) {
  const auto &src = u_[nlevel_-1];
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
  const DvceArray5D<Real> &src = def_[nlevel_-1];
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
  Kokkos::deep_copy(u_[current_level_], 0.0);
  return;
}

void Multigrid::CopySourceToData() {
  Kokkos::deep_copy(u_[nlevel_-1], src_[nlevel_-1]);
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictPack()
//! \brief Restrict the defect to the source

void Multigrid::RestrictPack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  CalculateDefectPack();
  is=js=ks= ngh_;
  ie = is + (indcs_.nx1>>ll) - 1;
  je = js + (indcs_.nx2>>ll) - 1;
  ke = ks + (indcs_.nx3>>ll) - 1;
  Restrict(src_[current_level_-1], def_[current_level_],
           nvar_, is, ie, js, je, ks, ke, th);
  // Full Approximation Scheme - restrict the variable itself
  Restrict(u_[current_level_-1], u_[current_level_],
             nvar_, is, ie, js, je, ks, ke, th);
  current_level_--;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictSourcePack()
//! \brief Restrict the source (and solution) without forming defect

void Multigrid::RestrictSourcePack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  is=js=ks= ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;
  Restrict(src_[current_level_-1], src_[current_level_],
           nvar_, is, ie, js, je, ks, ke, th);
  current_level_--;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrectPack()
//! \brief Prolongate the potential using tri-linear interpolation

void Multigrid::ProlongateAndCorrectPack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  ComputeCorrection();
  
  ProlongateAndCorrect(u_[current_level_+1], u_[current_level_],
                       is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, th);

  current_level_++;
  return;
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

  FMGProlongate(u_[current_level_+1], u_[current_level_],
                is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);

  current_level_++;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void Multigrid::SmoothPack(int color)
//! \brief Apply Smoother on the Pack

void Multigrid::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  is = js = ks = ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;
  Smooth(u_[current_level_], src_[current_level_],  coeff_[current_level_],
         matrix_[current_level_], -ll, is, ie, js, je, ks, ke, color, th);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::CalculateDefectPack()
//! \brief calculate the residual

void Multigrid::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  is = js = ks = ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;

  CalculateDefect(def_[current_level_], u_[current_level_], src_[current_level_],
                  coeff_[current_level_], matrix_[current_level_],
                  -ll, is, ie, js, je, ks, ke, th);

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::CalculateFASRHSPack()
//! \brief calculate the RHS for the Full Approximation Scheme

void Multigrid::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  is = js = ks = ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;
  CalculateFASRHS(src_[current_level_], u_[current_level_], coeff_[current_level_],
                  matrix_[current_level_], -ll, is, ie, js, je, ks, ke, th);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::SetFromRootGrid(bool folddata)
//! \brief Load the data from the root grid or octets (Athena++ style per-cell octets)

void Multigrid::SetFromRootGrid(bool folddata) {
  current_level_ = 0;
  auto &dst = u_[current_level_];
  auto &odst = uold_[current_level_];
  const auto &rsrc = pmy_driver_->mgroot_->GetCurrentData();
  const auto &rosrc = pmy_driver_->mgroot_->GetCurrentOldData();
  int padding = pmy_mesh_->gids_eachrank[global_variable::my_rank];
  auto dst_h = Kokkos::create_mirror_view(dst);
  auto odst_h = Kokkos::create_mirror_view(odst);
  const auto src_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rsrc);
  const auto osrc_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rosrc);
  for (int m = 0; m < nmmb_; ++m) {
    auto loc = pmy_mesh_->lloc_eachmb[m + padding];
    int lev = loc.level - pmy_driver_->locrootlevel_;
    if (lev == 0) {
      // Root-level block: read 3x3x3 neighborhood from root grid
      int ci = static_cast<int>(loc.lx1);
      int cj = static_cast<int>(loc.lx2);
      int ck = static_cast<int>(loc.lx3);
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2; ++k) {
          for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
              dst_h(m, v, k, j, i) = src_h(0, v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = osrc_h(0, v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    } else {
      // Refined block: read from parent octet
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
        for (int k = 0; k <= 2; ++k) {
          for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
              dst_h(m, v, k, j, i) = oct.U(v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = oct.Uold(v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(dst, dst_h);
  if (folddata)
    Kokkos::deep_copy(odst, odst_h);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the residual norm

Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n) {
  auto &def=def_[current_level_];
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
    // Grid spacing at current level (coarser by factor 2^ll)
  Real dx = rdx_ * static_cast<Real>(1 << ll);
  Real dy = rdy_ * static_cast<Real>(1 << ll);
  Real dz = rdz_ * static_cast<Real>(1 << ll);
  Real dV = dx * dy * dz;
  // Compute defect (residual) at current level
  CalculateDefect(def_[current_level_], u_[current_level_], src_[current_level_],
                  coeff_[current_level_], matrix_[current_level_],
                  -ll, is, ie, js, je, ks, ke, false);

  // Calculate norm over active zone on device
  Real norm = 0.0;
  
  if (nrm == MGNormType::max) {
    // L-infinity norm: max absolute value
    Kokkos::parallel_reduce("MG::DefectNorm_Linf", 
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, n, ks, js, is}, 
                                              {nmmb_, n+1, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i, Real &local_max) {
        local_max = std::max(local_max, std::abs(def(m, v, k, j, i)));
      }, Kokkos::Max<Real>(norm));
      return norm;
  } else if (nrm == MGNormType::l1) {
    // L1 norm: sum of absolute values
    Kokkos::parallel_reduce("MG::DefectNorm_L1",
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, n, ks, js, is},
                                              {nmmb_, n+1, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i, Real &local_sum) {
        local_sum += std::abs(def(m, v, k, j, i));
      }, Kokkos::Sum<Real>(norm));
  } else { // L2 norm (default)
    // L2 norm: sqrt(sum of squares)
    Kokkos::parallel_reduce("MG::DefectNorm_L2",
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, n, ks, js, is},
                                              {nmmb_, n+1, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i, Real &local_sum) {
        Real val = def(m, v, k, j, i);
        local_sum += val * val;
        }, Kokkos::Sum<Real>(norm));
  }
  norm *= defscale_;
  return norm;

}

//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateAverage(MGVariable type)
//! \brief Calculate volume-weighted average of variable 0 on current level

Real Multigrid::CalculateAverage(MGVariable type) {
  const auto &src = (type == MGVariable::src) ? src_[current_level_] : u_[current_level_];
  int ll = nlevel_ - 1 - current_level_;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + (indcs_.nx1 >> ll) - 1;
  je = js + (indcs_.nx2 >> ll) - 1;
  ke = ks + (indcs_.nx3 >> ll) - 1;

  auto brdx = block_rdx_;
  int ll_l = ll;

  Real sum = 0.0;
  Kokkos::parallel_reduce("MG::Average",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0, ks, js, is},
                                            {nmmb_, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
      Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
      Real dV_m = dx_m * dx_m * dx_m;
      local_sum += src(m, 0, k, j, i) * dV_m;
    }, Kokkos::Sum<Real>(sum));

  Real volume = 0.0;
  {
    auto brdx_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), block_rdx_);
    Real nx = static_cast<Real>(indcs_.nx1);
    for (int m = 0; m < nmmb_; ++m) {
      Real len = brdx_h(m) * nx;
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
  DvceArray5D<Real> &dst = (type == MGVariable::src) ? src_[nlevel_-1] : u_[nlevel_-1];
  int is, ie, js, je, ks, ke;
  is = js = ks = 0;
  ie = is + indcs_.nx1 + 2*ngh_ - 1;
  je = js + indcs_.nx2 + 2*ngh_ - 1;
  ke = ks + indcs_.nx3 + 2*ngh_ - 1;

  // local copies for device lambda capture
  const int m0 = 0, m1 = nmmb_ - 1;
  const int mk0 = ks, mk1 = ke;
  const int mj0 = js, mj1 = je;
  const int mi0 = is, mi1 = ie;
  const int vn = n;
  const Real lave = ave;

  auto dst_ = dst;

  par_for("Multigrid::SubtractAverage", DevExeSpace(),
          m0, m1, mk0, mk1, mj0, mj1, mi0, mi1,
  KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
    dst_(m, vn, mk, mj, mi) -= lave;
  });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::StoreOldData()
//! \brief store the old u data in the uold array

void Multigrid::StoreOldData() {
  Kokkos::deep_copy(DevExeSpace(),uold_[current_level_], u_[current_level_]);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::Restrict(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
//                      int nvar, int il, int iu, int jl, int ju, int kl, int ku, bool th)
//  \brief Actual implementation of prolongation and correction

void Multigrid::Restrict(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
                int nvar, int i0, int i1, int j0, int j1, int k0, int k1, bool th) {

  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar - 1;
  const int ngh = ngh_;
                
  par_for("Multigrid::Restrict", DevExeSpace(),
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
  DvceArray5D<Real> &u = u_[current_level_];
  const DvceArray5D<Real> &uold = uold_[current_level_];
  
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  int ll = nlevel_ - 1 - current_level_;
  int is = 0, ie = is + (indcs_.nx1 >> ll) + 2*ngh_ -1;
  int js = 0, je = js + (indcs_.nx2 >> ll) + 2*ngh_ -1;
  int ks = 0, ke = ks + (indcs_.nx3 >> ll) + 2*ngh_ -1;
  
  par_for("Multigrid::ComputeCorrection", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    u(m, v, k, j, i) -= uold(m, v, k, j, i);
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrect(DvceArray5D<Real> &dst,
//!     const DvceArray5D<Real> &src, int il, int iu, int jl, int ju, int kl, int ku,
//!     int fil, int fjl, int fkl, bool th)
//! \brief Actual implementation of prolongation and correction

void Multigrid::ProlongateAndCorrect(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th) {

  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  const int ll = pmy_driver_->fprolongation_; // copy host flag for capture

  auto dst_ = dst;
  auto src_ = src;

  if (ll == 1) { // tricubic
    par_for("Multigrid::ProlongateAndCorrect_tricubic", DevExeSpace(),
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
    par_for("Multigrid::ProlongateAndCorrect_trilinear", DevExeSpace(),
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
//! \fn void Multigrid::FMGProlongate(DvceArray5D<Real> &dst,
//!     const DvceArray5D<Real> &src, int il, int iu, int jl, int ju, int kl, int ku,
//!     int fil, int fjl, int fkl)
//! \brief FMG prolongation: direct overwrite (=) with tricubic interpolation.
//! Unlike ProlongateAndCorrect (+=), this overwrites the destination array.

void Multigrid::FMGProlongate(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl) {

  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  auto dst_ = dst;
  auto src_ = src;

  par_for("Multigrid::FMGProlongate", DevExeSpace(),
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
   MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg){
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
//! \brief Fill ghost cells at fine-coarse boundaries using injection prolongation
//! from coarser neighbors and restriction from finer neighbors (same-rank only).

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

  for (int m = 0; m < nmb; ++m) {
    int m_lev = mblev.h_view(m);
    int m_gid = mbgid.h_view(m);
    LogicalLocation m_loc = lloc[m_gid];

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;

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

              // Ghost cell ranges in each dimension
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

              if (nlev < m_lev) {
                // ---- Neighbor is COARSER: injection prolongation ----
                int child_x = m_loc.lx1 & 1;
                int child_y = m_loc.lx2 & 1;
                int child_z = m_loc.lx3 & 1;

                for (int v = 0; v < nvar; ++v) {
                  for (int gk = gks; gk <= gke; ++gk) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int si, sj, sk;
                        if (ox1 < 0)      si = ngh + ncells - 1;
                        else if (ox1 > 0) si = ngh;
                        else si = ngh + child_x*(ncells/2) + (gi - ngh)/2;

                        if (ox2 < 0)      sj = ngh + ncells - 1;
                        else if (ox2 > 0) sj = ngh;
                        else sj = ngh + child_y*(ncells/2) + (gj - ngh)/2;

                        if (ox3 < 0)      sk = ngh + ncells - 1;
                        else if (ox3 > 0) sk = ngh;
                        else sk = ngh + child_z*(ncells/2) + (gk - ngh)/2;

                        u_h(m, v, gk, gj, gi) = u_h(dm, v, sk, sj, si);
                      }
                    }
                  }
                }
                modified = true;

              } else {
                // ---- Neighbor is FINER: restriction (average 2x2x2 fine cells) ----
                // Determine which portion of my ghost cells this fine subblock covers.
                // For non-face dimensions, the subblock (f1,f2) splits my active range
                // in half.
                int sub_x = 0, sub_y = 0, sub_z = 0;
                // x1-face: non-face dims are y,z → f1=fy, f2=fz
                // x2-face: non-face dims are x,z → f1=fx, f2=fz
                // x3-face: non-face dims are x,y → f1=fx, f2=fy
                // edges/corners: only 1 or 0 non-face dims
                int nface = (ox1 != 0 ? 1:0) + (ox2 != 0 ? 1:0) + (ox3 != 0 ? 1:0);
                if (nface == 1) {
                  if (ox1 != 0) { sub_y = f1; sub_z = f2; }
                  if (ox2 != 0) { sub_x = f1; sub_z = f2; }
                  if (ox3 != 0) { sub_x = f1; sub_y = f2; }
                } else if (nface == 2) {
                  if (ox1 == 0) sub_x = f1;
                  if (ox2 == 0) sub_y = f1;
                  if (ox3 == 0) sub_z = f1;
                }

                // Restrict ghost range for ox==0 dims to the subblock half
                int half = ncells / 2;
                if (ox1 == 0) { gis = ngh + sub_x*half; gie = ngh + sub_x*half + half - 1; }
                if (ox2 == 0) { gjs = ngh + sub_y*half; gje = ngh + sub_y*half + half - 1; }
                if (ox3 == 0) { gks = ngh + sub_z*half; gke = ngh + sub_z*half + half - 1; }

                for (int v = 0; v < nvar; ++v) {
                  for (int gk = gks; gk <= gke; ++gk) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        // Map each coarse ghost cell to 2x2x2 fine cells
                        int fi0, fi1, fj0, fj1, fk0, fk1;
                        if (ox1 < 0) {
                          fi0 = ngh + ncells - 2; fi1 = ngh + ncells - 1;
                        } else if (ox1 > 0) {
                          fi0 = ngh; fi1 = ngh + 1;
                        } else {
                          int local_i = gi - (ngh + sub_x*half);
                          fi0 = ngh + 2*local_i; fi1 = fi0 + 1;
                        }
                        if (ox2 < 0) {
                          fj0 = ngh + ncells - 2; fj1 = ngh + ncells - 1;
                        } else if (ox2 > 0) {
                          fj0 = ngh; fj1 = ngh + 1;
                        } else {
                          int local_j = gj - (ngh + sub_y*half);
                          fj0 = ngh + 2*local_j; fj1 = fj0 + 1;
                        }
                        if (ox3 < 0) {
                          fk0 = ngh + ncells - 2; fk1 = ngh + ncells - 1;
                        } else if (ox3 > 0) {
                          fk0 = ngh; fk1 = ngh + 1;
                        } else {
                          int local_k = gk - (ngh + sub_z*half);
                          fk0 = ngh + 2*local_k; fk1 = fk0 + 1;
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

void Multigrid::PrintActiveRegion(const DvceArray5D<Real> &u) {
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u);
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

void Multigrid::PrintAll(const DvceArray5D<Real> &u) {
  auto u_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u);
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