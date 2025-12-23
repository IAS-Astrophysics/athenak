//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.cpp
//! \brief create multigrid solver for gravity

// C headers

// C++ headers
#include <algorithm>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../driver/driver.hpp"

class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
//! \brief MGGravityDriver constructor

MGGravityDriver::MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
    four_pi_G_ = pin->GetOrAddReal("gravity", "four_pi_G", -1);
    omega_ = pin->GetOrAddReal("gravity", "omega", 1.15);
    eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
    niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
    npresmooth_ = pin->GetOrAddReal("gravity", "npresmooth", npresmooth_);
    npostsmooth_ = pin->GetOrAddReal("gravity", "npostsmooth", npostsmooth_);
    fshowdef_ = pin->GetOrAddBoolean("gravity", "show_defect", false);
    if (eps_ < 0.0 && niter_ < 0) {
        std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <gravity> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
        exit(EXIT_FAILURE);
  }
  if (four_pi_G_ < 0.0) {
    std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Allocate the root multigrid
  int nghost = pmbp->pmesh->mb_indcs.ng;
  std::cout<< "Allocating MGGravity root multigrid object." <<std::endl;
  mgroot_ = new MGGravity(this, nullptr, nghost);
  std::cout<< "MGGravity root multigrid object allocated." <<std::endl;
  mglevels_ = new MGGravity(this, pmbp, nghost);
  std::cout<< "MGGravity meshblock multigrid object allocated." <<std::endl;
  // allocate boundary buffers
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers((nvar_));
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//! \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
  delete mglevels_;
}

//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//! \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost) : Multigrid(pmd, pmbp, nghost) {
  //btype = BoundaryQuantity::mg;
  //btypef = BoundaryQuantity::mg_faceonly;
  //pmgbval = new MGGravityBoundaryValues(this, mg_block_bcs_);
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::~MGGravity()
//! \brief MGGravity deconstructor

MGGravity::~MGGravity() {
  //delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

void MGGravityDriver::Solve(Driver *pdriver, int stage, Real dt) {
  std::cout << "MGGravityDriver::Solve called at stage " << stage << std::endl;
  RegionIndcs &indcs_ = pmy_pack_->pmesh->mb_indcs;
  // mglevels_ points to the Multigrid object for all MeshBlocks
  mglevels_->LoadSource(pmy_pack_->phydro->u0, IDN, indcs_.ng, four_pi_G_);
  //mglevels_->PrintAll(mglevels_->GetCurrentSource()); // DEBUG
  // iterative mode - load initial guess
  mglevels_->LoadFinestData(pmy_pack_->pgrav->phi, 0, indcs_.ng);
  std::cout << "Starting Multigrid Solve..." << std::endl;
  SetupMultigrid(dt, false);
  std::cout << "Multigrid Setup complete." << std::endl;
  SolveIterative(pdriver);

  gravity::Gravity *pgrav = pmy_pack_->pgrav;
  mglevels_->RetrieveResult(pgrav->phi, 0, indcs_.ng);
  mglevels_->RetrieveDefect(pgrav->def, 0, indcs_.ng);

  //if (vmg_[0]->pmy_block_->pgrav->fill_ghost)
  //  gtlist_->DoTaskListOneStage(pmy_mesh_, stage);

  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void MGGravity::Smooth(DvceArray5D<Real> &u, const DvceArray5D<Real> &src,
//!           const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &mmatrix, int rlev,
//!           int il, int iu, int jl, int ju, int kl, int ku, int color, bool th)
//! \brief Implementation of the Red-Black Gauss-Seidel Smoother
//!        rlev = relative level from the finest level of this Multigrid block

void MGGravity::Smooth(DvceArray5D<Real> &u, const DvceArray5D<Real> &src,
                const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrix, int rlev,
                int il, int iu, int jl, int ju, int kl, int ku, int color, bool th) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  std::cout<<"h="<< dx <<", rlev="<< rlev <<std::endl;
  Real dx2 = SQR(dx);
  Real isix = static_cast<MGGravityDriver*>(pmy_driver_)->omega_/6.0;
  color ^= pmy_driver_->GetCoffset();
  
  std::cout<<"omega = "<< static_cast<MGGravityDriver*>(pmy_driver_)->omega_ <<", isix="<< isix <<std::endl;
  //auto u_ = u;
  //auto src_ = src;
  std::cout<< "Smoothing with shift "<< rlev <<", il="<< il <<", iu="<< iu <<", color="<< color <<std::endl;
  par_for("MGGravity::Smooth", DevExeSpace(),0 ,nmmb_-1, kl, ku, jl, ju,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    const int c = (color + k + j) & 1;
    for (int i = il + c; i <= iu; i += 2) {
      u(m,0,k,j,i) -= ((6.0*u(m,0,k,j,i) - u(m,0,k+1,j,i) - u(m,0,k,j+1,i) - u(m,0,k,j,i+1)
                         - u(m,0,k-1,j,i) - u(m,0,k,j-1,i) - u(m,0,k,j,i-1)) + src(m,0,k,j,i)*dx2)*isix;
    }
  });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void MGGravity::CalculateDefect(DvceArray5D<Real> &def,
//!             const DvceArray5D<Real> &u, const DvceArray5D<Real> &src,
//!             const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrix,
//!            int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th)
//! \brief Implementation of the Defect calculation
//!        rlev = relative level from the finest level of this Multigrid block

void MGGravity::CalculateDefect(DvceArray5D<Real> &def, const DvceArray5D<Real> &u,
                const DvceArray5D<Real> &src, const DvceArray5D<Real> &coeff,
                const DvceArray5D<Real> &matrix, int rlev,
                int il, int iu, int jl, int ju, int kl, int ku, bool th) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);

  // local copies for safe capture in device lambda
  const int li = il, lu = iu;
  const Real lidx2 = idx2;

  auto def_ = def;
  auto u_ = u;
  auto src_ = src;
  std::cout<< "Calculating defect with shift "<< rlev <<", il="<< il <<", iu="<< iu <<std::endl;
  par_for("MGGravity::CalculateDefect", DevExeSpace(),
          0, nmmb_-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    def_(m,0,k,j,i) = (6.0*u_(m,0,k,j,i) - u_(m,0,k+1,j,i) - u_(m,0,k,j+1,i)
                           - u_(m,0,k,j,i+1) - u_(m,0,k-1,j,i) - u_(m,0,k,j-1,i)
                           - u_(m,0,k,j,i-1)) * lidx2 + src_(m,0,k,j,i);
  });

  return;
}


//----------------------------------------------------------------------------------------
//! \fn  void MGGravity::CalculateFASRHS(DvceArray5D<Real> &src,
//!             const DvceArray5D<Real> &u, const DvceArray5D<Real> &coeff,
//!             const DvceArray5D<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
//!             int kl, int ku, bool th)
//! \brief Implementation of the RHS calculation for FAS
//!        rlev = relative level from the finest level of this Multigrid block

void MGGravity::CalculateFASRHS(DvceArray5D<Real> &src, const DvceArray5D<Real> &u,
                const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrix,
                int shift, int il, int iu, int jl, int ju, int kl, int ku, bool th) {
  Real dx;
  if (shift <= 0) dx = rdx_*static_cast<Real>(1<<(-shift));
  else           dx = rdx_/static_cast<Real>(1<<shift);
  Real idx2 = 1.0/SQR(dx);

  // locals for capture
  const Real lidx2 = idx2;
  const int li = il, lu = iu;

  //auto src_ = src;
  //auto u_ = u;
  par_for("MGGravity::CalculateFASRHS", DevExeSpace(),
          0, nmmb_-1, kl, ku, jl, ju, li, lu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    src(m,0,k,j,i) -=  (6.0*u(m,0,k,j,i) - u(m,0,k+1,j,i) - u(m,0,k,j+1,i)
                         - u(m,0,k,j,i+1) - u(m,0,k-1,j,i) - u(m,0,k,j-1,i)
                         - u(m,0,k,j,i-1)) * lidx2;    
  });
  return;
}
