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
#include <chrono>
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
#include "../mhd/mhd.hpp"
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
    four_pi_G_ = pin->GetOrAddReal("gravity", "four_pi_G", -1.0);
    omega_ = pin->GetOrAddReal("gravity", "omega", 1.15);
    eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
    niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
    npresmooth_ = pin->GetOrAddReal("gravity", "npresmooth", npresmooth_);
    npostsmooth_ = pin->GetOrAddReal("gravity", "npostsmooth", npostsmooth_);
    full_multigrid_ = pin->GetOrAddBoolean("gravity", "full_multigrid", false);
    fmg_ncycle_ = pin->GetOrAddInteger("gravity", "fmg_ncycle", 1);
    fshowdef_ = pin->GetOrAddBoolean("gravity", "show_defect", false);
    fsubtract_average_ = pin->GetOrAddBoolean("gravity", "subtract_average", true);
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
  // Override MG boundary conditions if specified in input file.
  // Options: "zerofixed" (Dirichlet zero), "zerograd" (Neumann zero), "multipole".
  std::string mg_bc_str = pin->GetOrAddString("gravity", "mg_bc", "none");
  if (mg_bc_str != "none") {
    BoundaryFlag mg_bc;
    if (mg_bc_str == "zerofixed") {
      mg_bc = BoundaryFlag::mg_zerofixed;
    } else if (mg_bc_str == "zerograd") {
      mg_bc = BoundaryFlag::mg_zerograd;
    } else if (mg_bc_str == "multipole") {
      mg_bc = BoundaryFlag::mg_multipole;
    } else {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "Unknown mg_bc type: " << mg_bc_str << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int f = 0; f < 6; ++f) {
      if (mg_mesh_bcs_[f] != BoundaryFlag::periodic) {
        mg_mesh_bcs_[f] = mg_bc;
      }
    }
  }
  if (!pmy_mesh_->strictly_periodic) {
    fsubtract_average_ = false;
  }

  // Check if multipole BCs are active and configure
  for (int f = 0; f < 6; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) {
      mporder_ = 0;  // mark as detected
      break;
    }
  }
  if (mporder_ >= 0) {
    mporder_ = pin->GetOrAddInteger("gravity", "mporder", 4);
    autompo_ = pin->GetOrAddBoolean("gravity", "auto_mporigin", true);
    nodipole_ = pin->GetOrAddBoolean("gravity", "nodipole", false);
    if (mporder_ != 2 && mporder_ != 4) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "mporder must be 2 (quadrupole) or 4 (hexadecapole)." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (autompo_ && nodipole_) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "auto_mporigin and nodipole cannot be used together." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!autompo_) {
      mpo_[0] = pin->GetReal("gravity", "mporigin_x1");
      mpo_[1] = pin->GetReal("gravity", "mporigin_x2");
      mpo_[2] = pin->GetReal("gravity", "mporigin_x3");
    }
    AllocateMultipoleCoefficients();
    fsubtract_average_ = false;
  }

  // Source masking parameters
  mask_radius_ = pin->GetOrAddReal("gravity", "mask_radius", -1.0);
  mask_origin_[0] = pin->GetOrAddReal("gravity", "mask_origin_x1", 0.0);
  mask_origin_[1] = pin->GetOrAddReal("gravity", "mask_origin_x2", 0.0);
  mask_origin_[2] = pin->GetOrAddReal("gravity", "mask_origin_x3", 0.0);

  // Allocate the root multigrid
  int nghost = pin->GetOrAddInteger("gravity", "mg_nghost", 1);
  bool root_on_host = pin->GetOrAddBoolean("gravity", "root_on_host", false);
  mgroot_ = new MGGravity(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGGravity(this, pmbp, nghost);
  // allocate boundary buffers
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers((nvar_));
  mglevels_->pbval->RemapIndicesForMG();
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//! \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGGravityDriver::SetFourPiG(Real four_pi_G) {
  four_pi_G_ = four_pi_G;
}

//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//! \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
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
  RegionIndcs &indcs_ = pmy_pack_->pmesh->mb_indcs;

  // Reallocate MG arrays and phi if AMR has changed the mesh
  PrepareForAMR();
  {
    int nmb = pmy_pack_->nmb_thispack;
    if (static_cast<int>(pmy_pack_->pgrav->phi.extent_int(0)) != nmb) {
      int ncells1 = indcs_.nx1 + 2*indcs_.ng;
      int ncells2 = (indcs_.nx2 > 1) ? (indcs_.nx2 + 2*indcs_.ng) : 1;
      int ncells3 = (indcs_.nx3 > 1) ? (indcs_.nx3 + 2*indcs_.ng) : 1;
      Kokkos::realloc(pmy_pack_->pgrav->phi, nmb, 1, ncells3, ncells2, ncells1);
    }
  }

  // mglevels_ points to the Multigrid object for all MeshBlocks
  // The MG smoother solves -∇²u = src (note the minus sign from the Laplacian
  // convention: Laplacian(u) = 6u - neighbors = -dx²∇²u).  To obtain the
  // standard Poisson equation ∇²φ = 4πGρ we must load the source with a
  // negative sign so that -∇²φ = -4πGρ, i.e. ∇²φ = +4πGρ.
  auto &u0 = (pmy_pack_->pmhd != nullptr) ? pmy_pack_->pmhd->u0
                                            : pmy_pack_->phydro->u0;
  mglevels_->LoadSource(u0, IDN, indcs_.ng, -four_pi_G_);

  // Apply source mask (zero source outside mask_radius_)
  mglevels_->ApplyMask();

  // iterative mode - load initial guess
  if(!full_multigrid_) 
    mglevels_->LoadFinestData(pmy_pack_->pgrav->phi, 0, indcs_.ng);

  // Finalize setup (SubtractAverage, level counts) after data is loaded
  SetupMultigrid(dt, false);

  // Compute multipole coefficients for isolated boundaries
  if (mporder_ > 0) {
    if (autompo_) CalculateCenterOfMass();
    CalculateMultipoleCoefficients();
    SyncMultipoleToDevice();
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  if (full_multigrid_)
    SolveFMG(pdriver);
  else
    SolveMG(pdriver);

  Kokkos::fence();

  if (fshowdef_) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double mg_elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "mg_solve_time = " << std::scientific << std::setprecision(6)
              << mg_elapsed << std::endl;
    Real norm = CalculateDefectNorm(MGNormType::l2, 0);
    std::cout << "MGGravityDriver::Solve: Final defect norm = " << norm << std::endl;
  }

  mglevels_->RetrieveResult(pmy_pack_->pgrav->phi, 0, indcs_.ng);

  return;
}

void MGGravity::SmoothPack(int color) {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{static_cast<MGGravityDriver*>(pmy_driver_)->omega_/6.0};
  if (on_host_) {
    Smooth(u_[current_level_].h_view, src_[current_level_].h_view,
           coeff_[current_level_].h_view, matrix_[current_level_].h_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  } else {
    Smooth(u_[current_level_].d_view, src_[current_level_].d_view,
           coeff_[current_level_].d_view, matrix_[current_level_].d_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  }
}

void MGGravity::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateDefect(def_[current_level_].h_view, u_[current_level_].h_view,
                    src_[current_level_].h_view, coeff_[current_level_].h_view,
                    matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateDefect(def_[current_level_].d_view, u_[current_level_].d_view,
                    src_[current_level_].d_view, coeff_[current_level_].d_view,
                    matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}

void MGGravity::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateFASRHS(src_[current_level_].h_view, u_[current_level_].h_view,
                    coeff_[current_level_].h_view, matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateFASRHS(src_[current_level_].d_view, u_[current_level_].d_view,
                    coeff_[current_level_].d_view, matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}


//----------------------------------------------------------------------------------------
// Host-side octet physics for MGGravityDriver

static inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i) {
  return (6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
          - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
          - o.U(v,k,j,i-1));
}

void MGGravityDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  Real isix = omega_ / 6.0;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real lap = OctLaplacian(oct, 0, k, j, i);
        oct.U(0,k,j,i) -= (lap - oct.Src(0,k,j,i)*dx2)*isix;
      }
    }
  }
}

void MGGravityDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Def(0,k,j,i) = oct.Src(0,k,j,i) - OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}

void MGGravityDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Src(0,k,j,i) += OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::ProlongateOctetBoundariesFluxCons(...)
//! \brief Conservative prolongation of face ghost cells at fine-coarse level boundaries.
//!        Implements the "Conservative Formulation" from Tomida & Stone (2023) Eq. 24-27.
//!        Ghost = (2/3)*coarse_interpolated + (1/3)*fine_active, where transverse
//!        gradients from the coarse buffer provide sub-cell interpolation.
//!        Only face neighbors are handled (edges/corners are unused by the 7-point stencil).

void MGGravityDriver::ProlongateOctetBoundariesFluxCons(MGOctet &oct,
     std::vector<Real> &cbuf, const std::vector<bool> &ncoarse) {
  constexpr Real ot = 1.0/3.0;
  const int ngh = mgroot_->GetGhostCells();
  const int l = ngh, r = ngh + 1;

  // x1 face
  for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
    if (ncoarse[1*9 + 1*3 + (ox1+1)]) {
      int i, fi, fig;
      if (ox1 > 0) { i = ngh + 1; fi = ngh + 1; fig = ngh + 2; }
      else         { i = ngh - 1; fi = ngh;     fig = ngh - 1; }
      Real ccval = BufRef(cbuf, 3, 0, ngh, ngh, i);
      Real gx2c = 0.125*(BufRef(cbuf, 3, 0, ngh, ngh+1, i)
                        - BufRef(cbuf, 3, 0, ngh, ngh-1, i));
      Real gx3c = 0.125*(BufRef(cbuf, 3, 0, ngh+1, ngh, i)
                        - BufRef(cbuf, 3, 0, ngh-1, ngh, i));
      oct.U(0, l, l, fig) = ot*(2.0*(ccval - gx2c - gx3c) + oct.U(0, l, l, fi));
      oct.U(0, l, r, fig) = ot*(2.0*(ccval + gx2c - gx3c) + oct.U(0, l, r, fi));
      oct.U(0, r, l, fig) = ot*(2.0*(ccval - gx2c + gx3c) + oct.U(0, r, l, fi));
      oct.U(0, r, r, fig) = ot*(2.0*(ccval + gx2c + gx3c) + oct.U(0, r, r, fi));
    }
  }

  // x2 face
  for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
    if (ncoarse[1*9 + (ox2+1)*3 + 1]) {
      int j, fj, fjg;
      if (ox2 > 0) { j = ngh + 1; fj = ngh + 1; fjg = ngh + 2; }
      else         { j = ngh - 1; fj = ngh;     fjg = ngh - 1; }
      Real ccval = BufRef(cbuf, 3, 0, ngh, j, ngh);
      Real gx1c = 0.125*(BufRef(cbuf, 3, 0, ngh, j, ngh+1)
                        - BufRef(cbuf, 3, 0, ngh, j, ngh-1));
      Real gx3c = 0.125*(BufRef(cbuf, 3, 0, ngh+1, j, ngh)
                        - BufRef(cbuf, 3, 0, ngh-1, j, ngh));
      oct.U(0, l, fjg, l) = ot*(2.0*(ccval - gx1c - gx3c) + oct.U(0, l, fj, l));
      oct.U(0, l, fjg, r) = ot*(2.0*(ccval + gx1c - gx3c) + oct.U(0, l, fj, r));
      oct.U(0, r, fjg, l) = ot*(2.0*(ccval - gx1c + gx3c) + oct.U(0, r, fj, l));
      oct.U(0, r, fjg, r) = ot*(2.0*(ccval + gx1c + gx3c) + oct.U(0, r, fj, r));
    }
  }

  // x3 face
  for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
    if (ncoarse[(ox3+1)*9 + 1*3 + 1]) {
      int k, fk, fkg;
      if (ox3 > 0) { k = ngh + 1; fk = ngh + 1; fkg = ngh + 2; }
      else         { k = ngh - 1; fk = ngh;     fkg = ngh - 1; }
      Real ccval = BufRef(cbuf, 3, 0, k, ngh, ngh);
      Real gx1c = 0.125*(BufRef(cbuf, 3, 0, k, ngh, ngh+1)
                        - BufRef(cbuf, 3, 0, k, ngh, ngh-1));
      Real gx2c = 0.125*(BufRef(cbuf, 3, 0, k, ngh+1, ngh)
                        - BufRef(cbuf, 3, 0, k, ngh-1, ngh));
      oct.U(0, fkg, l, l) = ot*(2.0*(ccval - gx1c - gx2c) + oct.U(0, fk, l, l));
      oct.U(0, fkg, l, r) = ot*(2.0*(ccval + gx1c - gx2c) + oct.U(0, fk, l, r));
      oct.U(0, fkg, r, l) = ot*(2.0*(ccval - gx1c + gx2c) + oct.U(0, fk, r, l));
      oct.U(0, fkg, r, r) = ot*(2.0*(ccval + gx1c + gx2c) + oct.U(0, fk, r, r));
    }
  }
}
