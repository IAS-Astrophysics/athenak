//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bhl.cpp
//! \brief Problem generator to initialize relativistic Bondi-Hoyle-Lyttleton (BHL) accretion.
//!
//! Sets up a uniform flow (wind) at infinity with specified Mach number and impact parameter,
//! accounting for relativistic thermodynamics and magnetic fields in Kerr-Schild coordinates.

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

#include <Kokkos_Random.hpp>

// prototypes for functions used internally to this pgen
namespace {

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real spin,
                                         Real *pr, Real *ptheta, Real *pphi);

// Useful container for physical parameters of BHL Wind
struct bhl_pgen {
  Real spin;                    // black hole spin
  Real dexcise, pexcise;        // excision parameters
  Real gamma_adi;               // EOS parameters
  Real arad;                    // radiation constant

  // ---------------------------------------------------------------------------
  // Bondi–Hoyle–Lyttleton (BHL) wind parameters
  // Units: G = M = c = 1.  Upstream boundary is +x (outer_x1).
  // ---------------------------------------------------------------------------
  Real rho_inf, cs_inf, beta_inf, r_acc;
  Real v_inf, uux_inf;     // v_inf and u^{x'} = -v_inf/sqrt(1-v^2)
  Real e_inf, p_inf, temp_inf;
  Real sigma_inf, B0, Bx_tilde, By_tilde, Bz_tilde;
  Real urad_inf;
};

  bhl_pgen bhl;

} // namespace

// Prototypes for user-defined BCs and history functions
void NoInflowBHL(Mesh *pm);
void BHLFluxes(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for BHL accretion in GR
//! Compile with '-D PROBLEM=gr_bhl' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR BHL problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  user_bcs_func = NoInflowBHL;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  bool use_dyngr = (pmbp->pdyngr != nullptr);

  // Extract BH parameters
  bhl.spin = coord.bh_spin;
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux =
    (is_radiation_enabled) ? ceil(r_excise + 1.0) : 1.0 + sqrt(1.0 - SQR(bhl.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  // Additional radii for flux analysis
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));
  user_hist_func = BHLFluxes;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Extract radiation parameters if enabled
  int nangles_;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled) {
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    bhl.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bhl.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bhl.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr) {
    bhl.arad = pmbp->prad->arad;
  }

  // ---------------------------------------------------------------------------
  // BHL wind parameters (needed even on restart, because BC uses them)
  // Inputs: rho_inf, r_acc (=R_a), cs_inf, beta_inf, theta_B (deg)
  // ---------------------------------------------------------------------------
  bhl.rho_inf  = pin->GetOrAddReal("problem", "rho_inf", 1.0);
  bhl.r_acc    = pin->GetOrAddReal("problem", "r_acc",   200.0);
  bhl.cs_inf   = pin->GetOrAddReal("problem", "cs_inf",  0.05);
  bhl.beta_inf = pin->GetOrAddReal("problem", "beta_inf",10.0);
  const Real thetaB = pin->GetOrAddReal("problem", "theta_B", 90.0) * (M_PI/180.0);

  Real v2 = 2.0/bhl.r_acc;
  v2 = fmin(v2, 1.0 - 1.0e-12);
  bhl.v_inf   = sqrt(v2);
  bhl.uux_inf = -bhl.v_inf / sqrt(1.0 - v2);   // Eq. (init-velocity), c=1

  const Real cs2 = SQR(bhl.cs_inf);
  const Real denom = bhl.gamma_adi * (bhl.gamma_adi - 1.0 - cs2);
  bhl.e_inf = (denom > 0.0) ? (cs2 * bhl.rho_inf / denom) : 0.0; // Eq. (init-e)
  bhl.p_inf = bhl.e_inf * (bhl.gamma_adi - 1.0);                 // Eq. (init-p)
  bhl.temp_inf = (bhl.rho_inf > 0.0) ? (bhl.p_inf/bhl.rho_inf) : 0.0;

  bhl.sigma_inf = (bhl.beta_inf > 0.0)
    ? ((2.0/bhl.gamma_adi) * cs2 / bhl.beta_inf) : 0.0;          // Eq. (init-beta)
  const Real B0sq = bhl.sigma_inf * (bhl.rho_inf + bhl.gamma_adi*bhl.e_inf) * (1.0 - v2);
  bhl.B0 = sqrt(fmax(B0sq, 0.0));                                // Eq. (init-sigma)
  bhl.Bx_tilde = 0.0;
  bhl.By_tilde = bhl.B0 * sin(thetaB);                             // Eq. (initial b field)
  bhl.Bz_tilde = bhl.B0 * cos(thetaB);

  bhl.urad_inf = (is_radiation_enabled) ? (bhl.arad * SQR(SQR(bhl.temp_inf))) : 0.0;

  // excision parameters (used for internal boundary)
  bhl.dexcise = coord.dexcise;
  bhl.pexcise = coord.pexcise;

  // return if restart
  if (restart) return;

  // initialize primitive variables for new run ---------------------------------------

  auto bhl_params = bhl;
  auto &size = pmbp->pmb->mb_size;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  Kokkos::parallel_reduce("pgen_bhl", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_ptot) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    // Extract metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);

    // Calculate Boyer-Lindquist coordinates of cell for excision check
    Real r_cell, theta_cell, phi_cell;
    GetBoyerLindquistCoordinates(x1v,
                                 x2v,
                                 x3v,
                                 bhl_params.spin, &r_cell,
                                 &theta_cell, &phi_cell);

    // -----------------------------------------------------------------------
    // BHL wind primitives everywhere (except excision zone).
    // w0_(IV*) stores u^{i'} (normal-frame spatial velocity), so we set
    // u^{x'} = -v_inf/sqrt(1-v_inf^2), u^{y'}=u^{z'}=0.
    // -----------------------------------------------------------------------
    Real rho  = bhl_params.rho_inf;
    Real pgas = bhl_params.p_inf;
    Real uu1  = bhl_params.uux_inf;
    Real uu2  = 0.0;
    Real uu3  = 0.0;
    Real urad = (is_radiation_enabled) ? bhl_params.urad_inf : 0.0;

    // Set Vacuum/Atmosphere inside excision radius
    if (r_cell <= r_excise) {
      rho  = bhl_params.dexcise;
      pgas = bhl_params.pexcise;
      uu1 = uu2 = uu3 = 0.0;
      urad = 0.0;
    }

    // Set primitive values
    w0_(m,IDN,k,j,i) = rho;
    if (!use_dyngr) {
      w0_(m,IEN,k,j,i) = pgas / gm1;
    } else {
      w0_(m,IPR,k,j,i) = pgas;
    }
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;

    // Set coordinate frame intensity (if radiation enabled)
    if (is_radiation_enabled) {
      Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
             + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
             + glower[3][3]*uu3*uu3;
      Real uu0 = sqrt(1.0 + q);
      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      // Go through each angle
      for (int n=0; n<nangles_; ++n) {
        // Calculate direction in fluid frame
        Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                     u_tet_[3]*nh_c_.d_view(n,3));
        Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t;

        // Calculate intensity in tetrad frame
        Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
        i0_(m,n,k,j,i) = n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
      }
    }

    // Compute total pressure (equal to gas pressure in non-radiating runs)
    Real ptot;
    if (!use_dyngr) {
      ptot = gm1*w0_(m,IEN,k,j,i);
    } else {
      ptot = w0_(m,IPR,k,j,i);
    }
    if (is_radiation_enabled) ptot += urad/3.0;
    max_ptot = fmax(ptot, max_ptot);
  }, Kokkos::Max<Real>(ptotmax));

  // initialize ADM variables -----------------------------------------

  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  // initialize magnetic fields ---------------------------------------

  if (pmbp->pmhd != nullptr) {
    // Uniform BHL wind magnetic field (densitized Btilde^i), divergence-free in code:
    //   Btilde^i = (0, B0 sin(theta_B), B0 cos(theta_B))
    auto &b0   = pmbp->pmhd->b0;
    auto &bcc_ = pmbp->pmhd->bcc0;
    auto bhl_params = bhl;

    // Fill all face-centered components (including ghost faces)
    const int nk_x1 = b0.x1f.extent_int(1);
    const int nj_x1 = b0.x1f.extent_int(2);
    const int ni_x1 = b0.x1f.extent_int(3);
    par_for("pgen_b0_wind_x1", DevExeSpace(), 0,nmb-1, 0,nk_x1-1, 0,nj_x1-1, 0,ni_x1-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = bhl_params.Bx_tilde;
    });

    const int nk_x2 = b0.x2f.extent_int(1);
    const int nj_x2 = b0.x2f.extent_int(2);
    const int ni_x2 = b0.x2f.extent_int(3);
    par_for("pgen_b0_wind_x2", DevExeSpace(), 0,nmb-1, 0,nk_x2-1, 0,nj_x2-1, 0,ni_x2-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x2f(m,k,j,i) = bhl_params.By_tilde;
    });

    const int nk_x3 = b0.x3f.extent_int(1);
    const int nj_x3 = b0.x3f.extent_int(2);
    const int ni_x3 = b0.x3f.extent_int(3);
    par_for("pgen_b0_wind_x3", DevExeSpace(), 0,nmb-1, 0,nk_x3-1, 0,nj_x3-1, 0,ni_x3-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) = bhl_params.Bz_tilde;
    });

    // Fill cell-centered (densitized) B directly (including ghost zones)
    const int nk_cc = bcc_.extent_int(2);
    const int nj_cc = bcc_.extent_int(3);
    const int ni_cc = bcc_.extent_int(4);
    par_for("pgen_bcc_wind", DevExeSpace(), 0,nmb-1, 0,nk_cc-1, 0,nj_cc-1, 0,ni_cc-1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc_(m,IBX,k,j,i) = bhl_params.Bx_tilde;
      bcc_(m,IBY,k,j,i) = bhl_params.By_tilde;
      bcc_(m,IBZ,k,j,i) = bhl_params.Bz_tilde;
    });
  }

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  } else {
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
//   spin: black hole spin parameter (a)
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real spin,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                      + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-spin*x1, spin*x2+r*x1) -
          spin*r/(SQR(r)-2.0*r+SQR(spin));
  return;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn NoInflowBHL
//  \brief Sets boundary condition on surfaces of computational domain
//  Enforces Dirichlet wind inflow at Outer X1 (+x) and Outflow elsewhere.

void NoInflowBHL(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto &coord  = pm->pmb_pack->pcoord->coord_data;
  auto &size   = pm->pmb_pack->pmb->mb_size;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  // EOS gamma and DynGRMHD switch (needed to set either IEN or IPR)
  Real gamma_adi = 0.0;
  if (pm->pmb_pack->phydro != nullptr) {
    gamma_adi = pm->pmb_pack->phydro->peos->eos_data.gamma;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    gamma_adi = pm->pmb_pack->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = gamma_adi - 1.0;
  const bool use_dyngr = (pm->pmb_pack->pdyngr != nullptr);

  // Wind primitives/B (set in UserProblem even on restart)
  const Real rho_inf = bhl.rho_inf;
  const Real p_inf   = bhl.p_inf;
  const Real uux_inf = bhl.uux_inf;
  const Real uuy_inf = 0.0;
  const Real uuz_inf = 0.0;
  const Real bx_tilde = bhl.Bx_tilde;
  const Real by_tilde = bhl.By_tilde;
  const Real bz_tilde = bhl.Bz_tilde;
  const Real urad_inf = bhl.urad_inf;

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    nh_c_ = pm->pmb_pack->prad->nh_c;
    norm_to_tet_ = pm->pmb_pack->prad->norm_to_tet;
    tet_c_ = pm->pmb_pack->prad->tet_c;
    tetcov_c_ = pm->pmb_pack->prad->tetcov_c;
  }

  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        // Upstream (+x) Dirichlet inflow: enforce constant wind Btilde on boundary face
        // Only set faces strictly inside the ghost zone (ie+2 and beyond).
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = bx_tilde;
          b0.x2f(m,k,j,ie+i+1) = by_tilde;
          if (j == n2-1) { b0.x2f(m,k,j+1,ie+i+1) = by_tilde; }
          b0.x3f(m,k,j,ie+i+1) = bz_tilde;
          if (k == n3-1) { b0.x3f(m,k+1,j,ie+i+1) = bz_tilde; }
        }
      }
    });
  }

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  }

  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        // Free-streaming: copy primitives
        w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        // Upstream (+x) Dirichlet inflow: inject constant wind profile
        const int ii = ie+i+1;
        if (n == IDN) {
          w0_(m,n,k,j,ii) = rho_inf;
        } else if (!use_dyngr && n == IEN) {
          w0_(m,n,k,j,ii) = p_inf / gm1;
        } else if (use_dyngr && n == IPR) {
          w0_(m,n,k,j,ii) = p_inf;
        } else if (n == IVX) {
          w0_(m,n,k,j,ii) = uux_inf;
        } else if (n == IVY) {
          w0_(m,n,k,j,ii) = uuy_inf;
        } else if (n == IVZ) {
          w0_(m,n,k,j,ii) = uuz_inf;
        } else {
          // scalars/etc: keep free-streaming behavior
          w0_(m,n,k,j,ii) = w0_(m,n,k,j,ie);
        }
      }
    }
  });

  if (is_radiation_enabled) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = i0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        // ---------------------------------------------------------------------
        // FIX #1: Do NOT use norm_to_tet_/tet_* at ghost indices (ii).
        // Compute the inflow intensity pattern using a reference ACTIVE cell (iref=ie),
        // then apply it to all ghosts in +x.
        // ---------------------------------------------------------------------
        const int iref = ie;

        // Coordinates of reference active cell center (iref)
        Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
        Real x1v_ref = CellCenterX(iref-is, indcs.nx1, x1min, x1max);
        Real x2v_ref = CellCenterX(j -js,  indcs.nx2, x2min, x2max);
        Real x3v_ref = CellCenterX(k -ks,  indcs.nx3, x3min, x3max);

        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v_ref, x2v_ref, x3v_ref,
                                coord.is_minkowski, coord.bh_spin, glower, gupper);

        // Normal-frame 4-velocity components (uu0, uu1, uu2, uu3) with uu^i = u^{i'}
        Real uu1 = uux_inf, uu2 = uuy_inf, uu3 = uuz_inf;
        Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
               + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
               + glower[3][3]*uu3*uu3;
        Real uu0 = sqrt(1.0 + q);

        Real u_tet_[4];
        u_tet_[0] = (norm_to_tet_(m,0,0,k,j,iref)*uu0 + norm_to_tet_(m,0,1,k,j,iref)*uu1 +
                     norm_to_tet_(m,0,2,k,j,iref)*uu2 + norm_to_tet_(m,0,3,k,j,iref)*uu3);
        u_tet_[1] = (norm_to_tet_(m,1,0,k,j,iref)*uu0 + norm_to_tet_(m,1,1,k,j,iref)*uu1 +
                     norm_to_tet_(m,1,2,k,j,iref)*uu2 + norm_to_tet_(m,1,3,k,j,iref)*uu3);
        u_tet_[2] = (norm_to_tet_(m,2,0,k,j,iref)*uu0 + norm_to_tet_(m,2,1,k,j,iref)*uu1 +
                     norm_to_tet_(m,2,2,k,j,iref)*uu2 + norm_to_tet_(m,2,3,k,j,iref)*uu3);
        u_tet_[3] = (norm_to_tet_(m,3,0,k,j,iref)*uu0 + norm_to_tet_(m,3,1,k,j,iref)*uu1 +
                     norm_to_tet_(m,3,2,k,j,iref)*uu2 + norm_to_tet_(m,3,3,k,j,iref)*uu3);

        // Direction in fluid frame
        Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                     u_tet_[3]*nh_c_.d_view(n,3));
        Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t;

        // Guard against pathological tiny/negative n0_f (prevents 1/n0_f^4 blowups)
        Real n0_f_safe = fmax(n0_f, (Real)1.0e-20);

        // Intensity in tetrad frame
        Real n0 = tet_c_(m,0,0,k,j,iref);
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) { n_0 += tetcov_c_(m,d,0,k,j,iref)*nh_c_.d_view(n,d); }

        Real i0_inflow = n0*n_0*(urad_inf/(4.0*M_PI))/SQR(SQR(n0_f_safe));

        for (int i=0; i<ng; ++i) {
          const int ii = ie+i+1;
          i0_(m,n,k,j,ii) = i0_inflow;
        }
      }
    });
  }

  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  }

  // X2-Boundary
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
      }
    }
  });

  // ---------------------------------------------------------------------------
  // FIX #2: Add radiation i0 BCs on X2 (copy), matching working gr_torus behavior
  // ---------------------------------------------------------------------------
  if (is_radiation_enabled) {
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = i0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
        }
      }
    });
  }

  // X3-Boundary
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
      }
    }
  });

  // ---------------------------------------------------------------------------
  // FIX #2 (cont.): Add radiation i0 BCs on X3 (copy), matching gr_torus
  // ---------------------------------------------------------------------------
  if (is_radiation_enabled) {
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = i0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
        }
      }
    });
  }
}


//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void BHLFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // Calculate conversion for P to e if using DynGRMHD.
  Real to_ien = 1.;
  if (pmbp->pdyngr != nullptr) {
    to_ien = 1.0 / (gamma - 1.);
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = (is_mhd) ? 4 : 3;

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
    if (is_mhd) {
      pdata->label[nflux*g+3] = "phi_" + rad_str;
    }
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;
    if (is_mhd) pdata->hdata[nflux*g+3] = 0.0;

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN)*to_ien;

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
               2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
               2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor/alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real a2 = SQR(spin);
      Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r*x1/(2.0*r2 - rad2 + a2);
      Real drdy = r*x2/(2.0*r2 - rad2 + a2);
      Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
      // contravariant r component of 4-velocity
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // compute mass flux
      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*sqrtmdet*domega;

      // compute energy flux
      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega;

      // compute angular momentum flux
      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      pdata->hdata[nflux*g+2] += t1_3*sqrtmdet*domega;

      // compute magnetic flux
      if (is_mhd) {
        pdata->hdata[nflux*g+3] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
      }
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
