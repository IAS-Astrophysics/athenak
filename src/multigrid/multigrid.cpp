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
  // ngh is the number of ghost zones in src
  // ngh_ is the number of ghost zones in dst

  auto &dst = src_[nlevel_-1];
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_-ngh;
  ie = is + indcs_.nx1 + 2*ngh - 1;
  je = js + indcs_.nx2 + 2*ngh - 1;
  ke = ks + indcs_.nx3 + 2*ngh - 1;

  // local copies for device lambda capture
  const Real lfac = fac;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;

  par_for("Multigrid::LoadSource", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = ns + v;
    const int k = mk - ks;
    const int j = mj - js;
    const int i = mi - is;
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
  const int lks = ks, ljs = js, lis = is;
  const int lngh = ngh;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = ncoeff_ - 1;

  auto cm_ = cm;
  auto coeff_ = coeff;

  par_for("Multigrid::LoadCoefficients", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int k = mk + lngh - ngh_; // mk + (ngh - ngh_)
    const int j = mj + lngh - ngh_;
    const int i = mi + lngh - ngh_;
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
  int is, ie, js, je, ks, ke;
  int sngh = std::min(ngh_,ngh);
  is = js = ks = ngh_-sngh;
  ie = indcs_.nx1 + ngh_ + sngh - 1;
  je = indcs_.nx2 + ngh_ + sngh - 1;
  ke = indcs_.nx3 + ngh_ + sngh - 1;

  par_for("Multigrid::RetrieveResult", DevExeSpace(),
          0, nmmb_-1, 0, nvar_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int ndst = ns + v;
    const int k = mk - ks;
    const int j = mj - js;
    const int i = mi - is;
    dst(m, ndst, k, j, i) = src(m, v, mk, mj, mi);
  });

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

  auto dst_ = dst;
  auto src_ = src;

  par_for("Multigrid::RetrieveDefect", DevExeSpace(),
          m0, m1, v0, v1, mk0, mk1, mj0, mj1, mi0, mi1,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int ndst = ns + v;
    const int k = mk - ngh_ + ngh;
    const int j = mj - ngh_ + ngh;
    const int i = mi - ngh_ + ngh;
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


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictPack()
//! \brief Restrict the defect to the source

void Multigrid::RestrictPack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  CalculateDefectPack();
  int ngc = ngh_-(ngh_>>1);
  is=js=ks= ngc;
  ie = is+(indcs_.nx1>>ll)+ngc-1;
  je = js+(indcs_.nx2>>ll)+ngc-1;
  ke = ks+(indcs_.nx3>>ll)+ngc-1;
  Restrict(src_[current_level_-1], def_[current_level_],
           nvar_, is, ie, js, je, ks, ke, th);
  // Full Approximation Scheme - restrict the variable itself
  Restrict(u_[current_level_-1], u_[current_level_],
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
//! \fn  void Multigrid::SmoothPack(int color)
//! \brief Apply Smoother on the Pack

void Multigrid::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  int th = false;
  is = js = ks = 1;
  ie = is+(indcs_.nx1>>ll) + 2*(ngh_-1) - 1;
  je = js+(indcs_.nx2>>ll) + 2*(ngh_-1) - 1;
  ke = ks+(indcs_.nx3>>ll) + 2*(ngh_-1) - 1;
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
  is = js = ks = 1;
  ie = is+(indcs_.nx1>>ll) + 2*(ngh_-1) - 1;
  je = js+(indcs_.nx2>>ll) + 2*(ngh_-1) - 1;
  ke = ks+(indcs_.nx3>>ll) + 2*(ngh_-1) - 1;

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
  is = js = ks = 1;
  ie = is+(indcs_.nx1>>ll) + 2*(ngh_-1) - 1;
  je = js+(indcs_.nx2>>ll) + 2*(ngh_-1) - 1;
  ke = ks+(indcs_.nx3>>ll) + 2*(ngh_-1) - 1;
  CalculateFASRHS(src_[current_level_], u_[current_level_], coeff_[current_level_],
                  matrix_[current_level_], -ll, is, ie, js, je, ks, ke, th);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::SetFromRootGrid(bool folddata)
//! \brief Load the data from the root grid or octets

void Multigrid::SetFromRootGrid(bool folddata) {
  current_level_=0;
  auto &dst = u_[current_level_];
  auto &odst = uold_[current_level_];
  const auto &src=pmy_driver_->mgroot_->GetCurrentData();
  const auto &osrc = pmy_driver_->mgroot_->GetCurrentOldData();
  int lev = loc_.level - pmy_driver_->locrootlevel_;
  int padding = pmy_mesh_->gids_eachrank[global_variable::my_rank];
  //Host copy/mirror this should be optimized later
  auto dst_h = Kokkos::create_mirror_view(dst);
  auto odst_h = Kokkos::create_mirror_view(odst);
  const auto src_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), src);
  const auto osrc_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), osrc);

  if (lev == 0) { // from the root grid
    for(int m=0; m<nmmb_; ++m) {
      auto loc = pmy_mesh_->lloc_eachmb[m+padding];
      int ci = static_cast<int>(loc.lx1);
      int cj = static_cast<int>(loc.lx2);
      int ck = static_cast<int>(loc.lx3);
      
      for (int v=0; v<nvar_; ++v) {
        for (int k=ngh_-1; k<=ngh_+1; ++k) {
          for (int j=ngh_-1; j<=ngh_+1; ++j) {
            for (int i=ngh_-1; i<=ngh_+1; ++i){
              dst_h(m, v, k, j, i) = src_h(0, v, ck+k, cj+j, ci+i);
              if(folddata)
                odst_h(m,v, k, j, i) = osrc_h(0, v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    }
  }
  //Copy back to device
  Kokkos::deep_copy(dst, dst_h);
  if(folddata)
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

  } else if (nrm == MGNormType::l1) {
    // L1 norm: sum of absolute values
    Kokkos::parallel_reduce("MG::DefectNorm_L1",
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, n, ks, js, is},
                                              {nmmb_, n+1, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i, Real &local_sum) {
        local_sum += std::abs(def(m, v, k, j, i));
      }, Kokkos::Sum<Real>(norm));
    norm *= dV;
  } else { // L2 norm (default)
    // L2 norm: sqrt(sum of squares)
    Kokkos::parallel_reduce("MG::DefectNorm_L2",
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>(DevExeSpace(), {0, n, ks, js, is},
                                              {nmmb_, n+1, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i, Real &local_sum) {
        Real val = def(m, v, k, j, i);
        local_sum += val * val;
        }, Kokkos::Sum<Real>(norm));
    norm *= dV;
  }
  norm *= defscale_;
  return norm;

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
//! \fn MultigridBoundaryValues::MultigridBoundaryValues()
//! \brief Constructor for multigrid boundary values object
//----------------------------------------------------------------------------------------

MultigridBoundaryValues::MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse, Multigrid *pmg) 
  :
   MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg){
  return;
}
//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PackAndSend()
//! \brief Pack restricted fluxes of multigrid variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors. Adapts to different block sizes per level.

TaskStatus MultigridBoundaryValues::PackAndSendMG(const DvceArray5D<Real> &u) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = u.extent_int(1);

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  int current_level = pmy_mg->GetCurrentLevel();
  int nlevels = pmy_mg->GetNumberOfLevels();
  int shift_ = pmy_mg->GetLevelShift();
  int nx1_ = pmy_mg->GetSize();

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);

  Kokkos::parallel_for("MG::PackAndSendCC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    int shift = shift_;
    int nx1 = nx1_;
    int diff;
    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      // For multigrid, all neighbors are at the same level, so always use isame indices
      int il = sbuf[n].isame[0].bis;
      int iu = sbuf[n].isame[0].bie;
      int jl = sbuf[n].isame[0].bjs;
      int ju = sbuf[n].isame[0].bje;
      int kl = sbuf[n].isame[0].bks;
      int ku = sbuf[n].isame[0].bke;

      while(shift>0){
        if(rbuf[n].faces.d_view(0) and il==nx1){
          diff = iu-il;
          il = (il)>>1;  
          iu = il + diff;
        }
        else if(rbuf[n].faces.d_view(0)-1){
          iu = ((iu-il)>>1) + il;
        }
        if(rbuf[n].faces.d_view(1) and jl==nx1){
          diff = ju-jl;
          jl = (jl)>>1;  
          ju = jl + diff;
        }
        else if(rbuf[n].faces.d_view(1)-1){
          ju = ((ju-jl)>>1) + jl;
        }
        if(rbuf[n].faces.d_view(2) and kl==nx1){
          diff = ku-kl;
          kl = (kl)>>1;  
          ku = kl + diff;
        }
        else if(rbuf[n].faces.d_view(2)-1){
          ku = ((ku-kl)>>1) + kl;
        }
        shift--;
        nx1 = nx1 >> 1;
      }     

      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj = nk*nj;

      // index of receiving (destination) MB in neighbor rank's pack
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu+1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = u(m, v, k, j, i);
          });

        // else copy into send buffer for MPI communication
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu+1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = u(m, v, k, j, i);
          });
        }
      });
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for

  #if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
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
  auto &rbuf = recvbuf;
  int shift_ = pmy_mg->GetLevelShift();
  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
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
  int rank = global_variable::my_rank;
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nnghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("MG::RecvAndUnpackCC", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    int shift = shift_;
    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      int diff;
      // For multigrid all neighbors at same level, so use isame indices
      il = rbuf[n].isame[0].bis;
      iu = rbuf[n].isame[0].bie;
      jl = rbuf[n].isame[0].bjs;
      ju = rbuf[n].isame[0].bje;
      kl = rbuf[n].isame[0].bks;
      ku = rbuf[n].isame[0].bke;

      while(shift>0){
        if(rbuf[n].faces.d_view(0) and il>1){
          diff = iu-il;
          il = (il+ngh)>>1;  
          iu = il + diff;
        }
        else if(rbuf[n].faces.d_view(0)-1){
          iu = ((iu-il)>>1) + il;
        }
        if(rbuf[n].faces.d_view(1) and jl>1){
          diff = ju-jl;
          jl = (jl+ngh)>>1;  
          ju = jl + diff;
        }
        else if(rbuf[n].faces.d_view(1)-1){
          ju = ((ju-jl)>>1) + jl;
        }
        if(rbuf[n].faces.d_view(2) and kl>1){
          diff = ku-kl;
          kl = (kl+ngh)>>1;  
          ku = kl + diff;
        }
        else if(rbuf[n].faces.d_view(2)-1){
          ku = ((ku-kl)>>1) + kl;
        }
        shift--;
      }
       
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i: unpack from buffer into ghost cells
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
        [&](const int i) {
          u(m,v,k,j,i) = rbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
        });
      });
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for

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
  int shift_ = pmy_mg->GetLevelShift();

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