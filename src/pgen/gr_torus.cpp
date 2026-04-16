//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_torus.cpp
//! \brief Problem generator to initialize rotational equilibrium tori in GR, using either
//! Fishbone-Moncrief (1976) or Chakrabarti (1985) ICs, specialized for cartesian
//! Kerr-Schild coordinates.  Based on gr_torus.cpp in Athena++, with edits by CJW and SR.
//! Simplified and implemented in Kokkos by JMS.
//!
//! References:
//!    Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//!    Fishbone 1977, ApJ 215 323 (F)
//!    Chakrabarti, S. 1985, ApJ 288, 1

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <sstream>    // stringstream
#include <string>     // c_str(), string
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
static void CalculateCN(struct torus_pgen pgen, Real *cparam, Real *nparam);

KOKKOS_INLINE_FUNCTION
static Real CalculateL(struct torus_pgen pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(struct torus_pgen pgen, Real r, Real sin_theta, Real l);

KOKKOS_INLINE_FUNCTION
static Real CalculateLFromRPeak(struct torus_pgen pgen, Real r);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(struct torus_pgen pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static Real CalculateT(struct torus_pgen pgen, Real rho, Real ptot_over_rho);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct torus_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct torus_pgen pgen,
                                           Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct torus_pgen pgen,
                                     Real r, Real sin_theta, Real *pu0, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotentialInTiltedTorus(struct torus_pgen pgen,
                                                  Real r, Real theta, Real phi,
                                                  Real *patheta, Real *paphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct torus_pgen pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
Real A1(struct torus_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct torus_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct torus_pgen pgen, Real x1, Real x2, Real x3);

// Useful container for physical parameters of torus
struct torus_pgen {
  Real spin;                                  // black hole spin
  Real dexcise, pexcise;                      // excision parameters
  Real gamma_adi;                             // EOS parameters
  Real arad;                                  // radiation constant
  bool prograde;                              // flag indicating disk is prograde (FM)
  Real r_edge, r_peak, l, rho_max;            // fixed torus parameters
  Real l_peak;                                // fixed torus parameters
  Real c_param;                               // calculated chakrabarti parameter
  Real n_param;                               // fixed or calculated chakrabarti parameter
  Real log_h_edge, log_h_peak;                // calculated torus parameters
  Real ptot_over_rho_peak, rho_peak;          // more calculated torus parameters
  Real r_outer_edge;                          // even more calculated torus parameters
  Real psi, sin_psi, cos_psi;                 // tilt parameters
  Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
  bool is_vertical_field;                     // use vertical field configuration
  bool fm_torus, chakrabarti_torus;           // FM versus Chakrabarti torus ICs
  Real potential_cutoff, potential_falloff;   // sets region of torus to magnetize
  Real potential_r_pow;                       // set how vector potential scales
  Real potential_beta_min;                    // set how vector potential scales (cont.)
  Real potential_rho_pow;                     // set vector potential dependence on rho
};

  torus_pgen torus;

} // namespace

// Prototypes for user-defined BCs and history functions
void NoInflowTorus(Mesh *pm);
void TorusFluxes(HistoryData *pdata, Mesh *pm);

// prototype for custom history function
void TorusHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for either Fishbone-Moncrief or Chakrabarti torus in GR
//! Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator
//!  assumes x3 is axisymmetric direction

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR torus problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  user_bcs_func = NoInflowTorus;
  //user_hist_func = &TorusHistory;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  bool use_dyngr = (pmbp->pdyngr != nullptr);

  // Extract BH parameters
  torus.spin = coord.bh_spin;
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux =
    (is_radiation_enabled) ? ceil(r_excise + 1.0) : 1.0 + sqrt(1.0 - SQR(torus.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  // NOTE(@pdmullen): Enroll additional radii for flux analysis by
  // pushing back the grids vector with additional SphericalGrid instances
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));
  user_hist_func = TorusFluxes;

  // return if restart
  if (restart) return;

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
    torus.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    torus.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = torus.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr) {
    torus.arad = pmbp->prad->arad;
  }

  // Read problem-specific parameters from input file
  // global parameters
  torus.rho_min = pin->GetReal("problem", "rho_min");
  torus.rho_pow = pin->GetReal("problem", "rho_pow");
  torus.pgas_min = pin->GetReal("problem", "pgas_min");
  torus.pgas_pow = pin->GetReal("problem", "pgas_pow");
  torus.psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * (M_PI/180.0);
  torus.sin_psi = sin(torus.psi);
  torus.cos_psi = cos(torus.psi);
  torus.rho_max = pin->GetReal("problem", "rho_max");
  torus.r_edge = pin->GetReal("problem", "r_edge");
  torus.r_peak = pin->GetReal("problem", "r_peak");
  torus.n_param = pin->GetOrAddReal("problem", "n_param",0.0);
  torus.prograde = pin->GetOrAddBoolean("problem","prograde",true);
  torus.fm_torus = pin->GetOrAddBoolean("problem", "fm_torus", false);
  torus.chakrabarti_torus = pin->GetOrAddBoolean("problem", "chakrabarti_torus", false);

  // local parameters
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);

  // excision parameters
  torus.dexcise = coord.dexcise;
  torus.pexcise = coord.pexcise;

  // Compute angular momentum and prepare constants describing primitives
  if (torus.fm_torus) {
    torus.l_peak = CalculateLFromRPeak(torus, torus.r_peak);
  } else if (torus.chakrabarti_torus) {
    CalculateCN(torus, &torus.c_param, &torus.n_param);
    torus.l_peak = CalculateL(torus, torus.r_peak, 1.0);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unrecognized torus type in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Common to both tori:
  torus.log_h_edge = LogHAux(torus, torus.r_edge, 1.0);
  torus.log_h_peak = LogHAux(torus, torus.r_peak, 1.0) - torus.log_h_edge;
  torus.ptot_over_rho_peak = gm1/torus.gamma_adi * (exp(torus.log_h_peak)-1.0);
  torus.rho_peak = pow(torus.ptot_over_rho_peak, 1.0/gm1) / torus.rho_max;

  // find "outer edge" of torus (first place log_h > 0)
  Real ra = torus.r_peak;
  Real rb = 2. * ra;
  Real log_h_trial = LogHAux(torus, rb, 1.) - torus.log_h_edge;
  for (int iter=0; iter<10000; ++iter) {
    if (log_h_trial <= 0) {
      break;
    }
    rb *= 2.;
    log_h_trial = LogHAux(torus, rb, 1.) - torus.log_h_edge;
  }
  for (int iter=0; iter<10000; ++iter) {
    if (fabs(ra - rb) < 1.e-3) {
      break;
    }
    Real r_trial = (ra + rb) / 2.;
    if (LogHAux(torus, r_trial, 1.) > torus.log_h_edge) {
      ra = r_trial;
    } else {
      rb = r_trial;
    }
  }
  torus.r_outer_edge = ra;
  std::cout << "Found torus outer edge: " << torus.r_outer_edge << std::endl;

  // initialize primitive variables for new run ---------------------------------------

  auto trs = torus;
  auto &size = pmbp->pmb->mb_size;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  Kokkos::parallel_reduce("pgen_torus1", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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

    // Calculate Boyer-Lindquist coordinates of cell
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);
    Real sin_theta = sin(theta);
    Real cos_theta = cos(theta);
    Real sin_phi = sin(phi);
    Real cos_phi = cos(phi);

    // Account for tilt
    Real sin_vartheta;
    if (trs.psi != 0.0) {
      Real x = sin_theta * cos_phi;
      Real y = sin_theta * sin_phi;
      Real z = cos_theta;
      Real varx = trs.cos_psi * x - trs.sin_psi * z;
      Real vary = y;
      sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    } else {
      sin_vartheta = fabs(sin_theta);
    }

    // Determine if we are in the torus
    Real log_h;
    bool in_torus = false;
    if (r >= trs.r_edge) {
      log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
      if (log_h >= 0.0) {
        in_torus = true;
      }
    }

    // Calculate background primitives -- to be consistent with the excision algorithm,
    // we have to recalculate r; we try to avoid excising cells within the horizon which
    // might have a corner sticking out of the horizon.
    Real r_excise, theta_excise, phi_excise;
    GetBoyerLindquistCoordinates(trs, x1v + copysign(0.5*dx1,x1v),
                                      x2v + copysign(0.5*dx2,x2v),
                                      x3v + copysign(0.5*dx3,x3v), &r_excise,
                                      &theta_excise, &phi_excise);
    Real rho_bg, pgas_bg;
    if (r_excise > 1.0) {
      rho_bg = trs.rho_min * pow(r, trs.rho_pow);
      pgas_bg = trs.pgas_min * pow(r, trs.pgas_pow);
    } else {
      rho_bg = trs.dexcise;
      pgas_bg = trs.pexcise;
    }

    Real rho = rho_bg;
    Real pgas = pgas_bg;
    Real uu1 = 0.0;
    Real uu2 = 0.0;
    Real uu3 = 0.0;
    Real urad = 0.0;

    Real perturbation = 0.0;
    // Overwrite primitives inside torus
    if (in_torus) {
      // Calculate perturbation
      auto rand_gen = rand_pool64.get_state(); // get random number state this thread
      perturbation = 2.0*pert_amp*(rand_gen.frand() - 0.5);
      rand_pool64.free_state(rand_gen);        // free state for use by other threads

      // Calculate thermodynamic variables
      Real ptot_over_rho = gm1/trs.gamma_adi * (exp(log_h) - 1.0);
      rho = pow(ptot_over_rho, 1.0/gm1) / trs.rho_peak;
      Real temp = ptot_over_rho;
      if (is_radiation_enabled) temp = CalculateT(trs, rho, ptot_over_rho);
      pgas = temp * rho;

      // Calculate radiation variables (if radiation enabled)
      if (is_radiation_enabled) urad = trs.arad * SQR(SQR(temp));

      // Calculate velocities in Boyer-Lindquist coordinates
      Real u0_bl, u1_bl, u2_bl, u3_bl;
      CalculateVelocityInTiltedTorus(trs, r, theta, phi,
                                     &u0_bl, &u1_bl, &u2_bl, &u3_bl);

      // Transform to preferred coordinates
      Real u0, u1, u2, u3;
      TransformVector(trs, u0_bl, 0.0, u2_bl, u3_bl,
                      x1v, x2v, x3v, &u0, &u1, &u2, &u3);

      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);
      uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
      uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
      uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
    }

    // Set primitive values, including random perturbations to pressure
    w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
    if (!use_dyngr) {
      w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg) * (1.0 + perturbation) / gm1;
    } else {
      w0_(m,IPR,k,j,i) = fmax(pgas, pgas_bg) * (1.0 + perturbation);
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
    // parse some more parameters from input
    torus.potential_beta_min = pin->GetOrAddReal("problem", "potential_beta_min", 100.0);
    torus.potential_cutoff   = pin->GetOrAddReal("problem", "potential_cutoff", 0.2);

    torus.is_vertical_field = pin->GetOrAddBoolean("problem", "vertical_field", false);

    // for FM torus:
    // Eqns 33, 34 of arxiv/2202.11721
    //  SANE -> potential_r_pow = 0
    //          potential_falloff = 0  (when 0 -> falloff is disabled)
    //          potential_rho_pow = 1
    //   MAD -> potential_r_pow = 3
    //          potential_falloff = 400
    //          potential_rho_pow = 1
    torus.potential_falloff  = pin->GetOrAddReal("problem", "potential_falloff", 0.0);
    torus.potential_r_pow    = pin->GetOrAddReal("problem", "potential_r_pow", 0.0);
    torus.potential_rho_pow  = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;
    auto trs = torus;

    par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(trs, x1v, x2f, x3f);
      a2(m,k,j,i) = A2(trs, x1f, x2v, x3f);
      a3(m,k,j,i) = A3(trs, x1f, x2f, x3v);

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(trs, xl,x2f,x3f) + A1(trs, xr,x2f,x3f));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(trs, x1f,xl,x3f) + A2(trs, x1f,xr,x3f));
      }

      // Correct A3 at x1-faces, x2-faces, and x1x2-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
          (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
        Real xl = x3v + 0.25*dx3;
        Real xr = x3v - 0.25*dx3;
        a3(m,k,j,i) = 0.5*(A3(trs, x1f,x2f,xl) + A3(trs, x1f,x2f,xr));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // find maximum bsq
    Real bsqmax = std::numeric_limits<float>::min();
    Real bsqmax_intorus = std::numeric_limits<float>::min();
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    const int nji  = indcs.nx2*indcs.nx1;
    Kokkos::parallel_reduce("torus_beta", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &max_bsq, Real &max_bsq_intorus) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      // Extract metric components
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);

      // Calculate Boyer-Lindquist coordinates of cell
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);
      Real sin_theta = sin(theta);
      Real cos_theta = cos(theta);
      Real sin_phi = sin(phi);
      Real cos_phi = cos(phi);

      // Account for tilt
      Real sin_vartheta;
      if (trs.psi != 0.0) {
        Real x = sin_theta * cos_phi;
        Real y = sin_theta * sin_phi;
        Real z = cos_theta;
        Real varx = trs.cos_psi * x - trs.sin_psi * z;
        Real vary = y;
        sin_vartheta = sqrt(SQR(varx) + SQR(vary));
      } else {
        sin_vartheta = fabs(sin_theta);
      }

      // Determine if we are in the torus
      Real log_h;
      bool in_torus = false;
      if (r >= trs.r_edge) {
        log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
        if (log_h >= 0.0) {
          in_torus = true;
        }
      }

      // Extract primitive velocity, magnetic field B^i, and gas pressure
      Real &wvx = w0_(m,IVX,k,j,i);
      Real &wvy = w0_(m,IVY,k,j,i);
      Real &wvz = w0_(m,IVZ,k,j,i);
      Real &wbx = bcc_(m,IBX,k,j,i);
      Real &wby = bcc_(m,IBY,k,j,i);
      Real &wbz = bcc_(m,IBZ,k,j,i);

      // Calculate 4-velocity (exploiting symmetry of metric)
      Real q = glower[1][1]*wvx*wvx +2.0*glower[1][2]*wvx*wvy +2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy +2.0*glower[2][3]*wvy*wvz
             + glower[3][3]*wvz*wvz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = wvx - alpha * lor * gupper[0][1];
      Real u2 = wvy - alpha * lor * gupper[0][2];
      Real u3 = wvz - alpha * lor * gupper[0][3];

      // lower vector indices
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field
      Real b0 = u_1*wbx + u_2*wby + u_3*wbz;
      Real b1 = (wbx + b0 * u1) / u0;
      Real b2 = (wby + b0 * u2) / u0;
      Real b3 = (wbz + b0 * u3) / u0;

      // lower vector indices and compute bsq
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      max_bsq = fmax(bsq, max_bsq);
      if (in_torus) {
        max_bsq_intorus = fmax(bsq, max_bsq_intorus);
      }
    }, Kokkos::Max<Real>(bsqmax), Kokkos::Max<Real>(bsqmax_intorus));

#if MPI_PARALLEL_ENABLED
    // get maximum value of gas pressure and bsq over all MPI ranks
    MPI_Allreduce(MPI_IN_PLACE, &ptotmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bsqmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bsqmax_intorus, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

    // Apply renormalization of magnetic field
    Real bnorm = sqrt((ptotmax/(0.5*bsqmax))/torus.potential_beta_min);
    // Since vertical field extends beyond torus, normalize based on values in torus
    if (torus.is_vertical_field) {
      bnorm = sqrt((ptotmax/(0.5*bsqmax_intorus))/torus.potential_beta_min);
    }

    par_for("pgen_normb0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) *= bnorm;
      b0.x2f(m,k,j,i) *= bnorm;
      b0.x3f(m,k,j,i) *= bnorm;
      if (i==ie) { b0.x1f(m,k,j,i+1) *= bnorm; }
      if (j==je) { b0.x2f(m,k,j+1,i) *= bnorm; }
      if (k==ke) { b0.x3f(m,k+1,j,i) *= bnorm; }
    });

    // Recompute cell-centered magnetic field
    par_for("pgen_normbcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
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
    //pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for calculating angular momentum variable l in Fishbone-Moncrief torus
// Inputs:
//   r: desired radius of pressure maximum
// Outputs:
//   returned value: l = u^t u_\phi such that pressure maximum occurs at r_peak
// Notes:
//   beware many different definitions of l abound; this is *not* -u_phi/u_t
//   Harm has a similar function: lfish_calc() in init.c
//     Harm's function assumes M = 1 and that corotation is desired
//     it is equivalent to this, though seeing this requires much manipulation
//   implements (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation

KOKKOS_INLINE_FUNCTION
static Real CalculateLFromRPeak(struct torus_pgen pgen, Real r) {
  Real sgn = (pgen.prograde) ? 1.0 : -1.0;
  Real num = sgn*(SQR(r*r) + SQR(pgen.spin*r) - 2.0*SQR(pgen.spin)*r)
           - pgen.spin*(r*r - pgen.spin*pgen.spin)*sqrt(r);
  Real denom = SQR(r) - 3.0*r + sgn*2.0*pgen.spin*sqrt(r);
  return 1.0/r * sqrt(1.0/r) * num/denom;
}


//----------------------------------------------------------------------------------------
// Function to calculate enthalpy in Fishbone-Moncrief torus or Chakrabarti torus
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)
//   references Chakrabarti, S. 1985, ApJ 288, 1

KOKKOS_INLINE_FUNCTION
static Real LogHAux(struct torus_pgen pgen, Real r, Real sin_theta) {
  Real tol_trunc=1e-15;
  Real logh;
  if (pgen.fm_torus) {
    Real sin_sq_theta = SQR(sin_theta);
    Real cos_sq_theta = 1.0 - sin_sq_theta;
    Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);            // \Delta
    Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;       // \Sigma
    Real aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
    Real exp_2nu = sigma * delta / aa;                       // \exp(2\nu) (FM 3.5)
    Real exp_2psi = aa / sigma * sin_sq_theta;               // \exp(2\psi) (FM 3.5)
    Real exp_neg2chi = exp_2nu / exp_2psi;                   // \exp(-2\chi) (cf. FM 2.15)
    Real omega = 2.0*pgen.spin*r/aa;                         // \omega (FM 3.5)
    Real var_a = sqrt(1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi);
    Real var_b = 0.5 * log((1.0+var_a) / (sigma*delta/aa));
    Real var_c = -0.5 * var_a;
    Real var_d = -pgen.l_peak * omega;
    logh = var_b + var_c + var_d;                            // (FM 3.4)
  } else { // Chakrabarti
    Real l = CalculateL(pgen, r, sin_theta);
    Real u_t = CalculateCovariantUT(pgen, r, sin_theta, l);
    Real l_edge = CalculateL(pgen, pgen.r_edge, 1.0);
    Real u_t_edge = CalculateCovariantUT(pgen, pgen.r_edge, 1.0, l_edge);
    Real h = u_t_edge/u_t;
    if (pgen.n_param==1.0) {
      h *= pow(l_edge/l, SQR(pgen.c_param)/(SQR(pgen.c_param)-1.0));
    } else {
      Real pow_c = 2.0/pgen.n_param;
      Real pow_l = 2.0-2.0/pgen.n_param;
      Real pow_abs = pgen.n_param/(2.0-2.0*pgen.n_param);
      h *= (pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l   , pow_l)), pow_abs) *
            pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l_edge, pow_l)), -1.0*pow_abs));
    }
    if (isfinite(h) && h >= 1.0) {
      logh = log(h);
    } else if (fabs(h-1.0) <= 1e-15) {
      logh = 0.0;
    } else {
      logh = -1.0;
    }
  }
  return logh;
}

//----------------------------------------------------------------------------------------
// Function to calculate T for radiating runs, assuming pressure and temp equilibrium
// Outputs:
//   returned value: temperature (p_gas / rho)
// Notes:
//   equation has form b4 * T^4 + T + b0 = 0

KOKKOS_INLINE_FUNCTION
static Real CalculateT(struct torus_pgen pgen, Real rho, Real ptot_over_rho) {
  // Calculate quartic coefficients
  Real b4 = pgen.arad / (3.0 * rho);
  Real b0 = -ptot_over_rho;

  // Calculate real root of z^3 - 4*b0/b4 * z - 1/b4^2 = 0
  Real delta1 = 0.25 - 64.0 * b0 * b0 * b0 * b4 / 27.0;
  if (delta1 < 0.0) {
    return 0.0;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return 0.0;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = pow(0.5 + delta1, 1.0/3.0) - pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return 0.0;
  }
  zroot *= pow(b4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (b4 * rcoef);
  if (delta2 < 0.0) {
    return 0.0;
  }
  delta2 = sqrt(delta2);
  Real root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return 0.0;
  }
  return root;
}

//----------------------------------------------------------------------------------------
// Function for calculating c, n parameters controlling angular momentum profile
// in Chakrabarti torus, where l = c * lambda^n. edited so that n can be pre-specified
// such that the assumption of keplerian angular momentum at the inner edge is dropped

KOKKOS_INLINE_FUNCTION
static void CalculateCN(struct torus_pgen pgen, Real *cparam, Real *nparam) {
  Real n_input = pgen.n_param;
  Real nn; // slope of angular momentum profile
  Real cc; // constant of angular momentum profile
  Real l_edge = ((SQR(pgen.r_edge) + SQR(pgen.spin) - 2.0*pgen.spin*sqrt(pgen.r_edge))/
                 (sqrt(pgen.r_edge)*(pgen.r_edge - 2.0) + pgen.spin));
  Real l_peak = ((SQR(pgen.r_peak) + SQR(pgen.spin) - 2.0*pgen.spin*sqrt(pgen.r_peak))/
                 (sqrt(pgen.r_peak)*(pgen.r_peak - 2.0) + pgen.spin));
  Real lambda_edge = sqrt((l_edge*(-2.0*pgen.spin*l_edge + SQR(pgen.r_edge)*pgen.r_edge
                                   + SQR(pgen.spin)*(2.0+pgen.r_edge)))/
                          (2.0*pgen.spin + l_edge*(pgen.r_edge - 2.0)));
  Real lambda_peak = sqrt((l_peak*(-2.0*pgen.spin*l_peak + SQR(pgen.r_peak)*pgen.r_peak
                                   + SQR(pgen.spin)*(2.0+pgen.r_peak)))/
                          (2.0*pgen.spin + l_peak*(pgen.r_peak - 2.0)));
  if (n_input == 0.0) {
    nn = log(l_peak/l_edge)/log(lambda_peak/lambda_edge);
    cc = l_edge*pow(lambda_edge, -nn);
  } else {
    nn = n_input;
    cc = l_peak*pow(lambda_peak, -nn);
  }
  *cparam = cc;
  *nparam = nn;
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating l in Chakrabarti torus

KOKKOS_INLINE_FUNCTION
static Real CalculateL(struct torus_pgen pgen, Real r, Real sin_theta) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(pgen.spin)*(1.0-SQR(sin_theta));
  Real g_00 = -1.0 + 2.0*r/sigma;
  Real g_03 = -2.0*pgen.spin*r/sigma*SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0*SQR(pgen.spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Perform bisection
  Real l_min = 1.0;
  Real l_max = 100.0;
  Real l_val = 0.5*(l_min + l_max);
  int max_iterations = 25;
  Real tol_rel = 1.0e-8;
  for (int n=0; n<max_iterations; ++n) {
    Real error_rel = 0.5*(l_max - l_min)/l_val;
    if (error_rel < tol_rel) {
      break;
    }
    Real residual = pow(l_val/pgen.c_param, 2.0/pgen.n_param) +
                    (l_val*g_33 + SQR(l_val)*g_03)/(g_03 + l_val*g_00);
    if (residual < 0.0) {
      l_min = l_val;
      l_val = 0.5 * (l_min + l_max);
    } else if (residual > 0.0) {
      l_max = l_val;
      l_val = 0.5 * (l_min + l_max);
    } else if (residual == 0.0) {
      break;
    }
  }
  return l_val;
}

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(struct torus_pgen pgen, Real r, Real sin_theta, Real l) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(pgen.spin)*(1.0-SQR(sin_theta));
  Real g_00 = -1.0 + 2.0*r/sigma;
  Real g_03 = -2.0*pgen.spin*r/sigma*SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0*SQR(pgen.spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(fmax((SQR(g_03) - g_00*g_33)/(g_33 + 2.0*l*g_03 + SQR(l)*g_00), 0.0));
  return u_t;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct torus_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
          pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside tilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   theta,phi: Boyer-Lindquist theta and phi in BH-aligned coordinates
// Outputs:
//   pu0,pu1,pu2,pu3: u^\mu set (Boyer-Lindquist coordinates)
// Notes:
//   first finds corresponding location in untilted torus
//   next calculates velocity at that point in untilted case
//   finally transforms that velocity into coordinates in which torus is tilted

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct torus_pgen pgen,
                                           Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3) {
  // Calculate corresponding location
  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real sin_phi = sin(phi);
  Real cos_phi = cos(phi);
  Real sin_vartheta, cos_vartheta, varphi;
  if (pgen.psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = pgen.cos_psi * x - pgen.sin_psi * z;
    Real vary = y;
    Real varz = pgen.sin_psi * x + pgen.cos_psi * z;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = atan2(vary, varx);
  } else {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
  }
  Real sin_varphi = sin(varphi);
  Real cos_varphi = cos(varphi);

  // Calculate untilted velocity
  Real u0_tilt, u3_tilt;
  CalculateVelocityInTorus(pgen, r, sin_vartheta, &u0_tilt, &u3_tilt);
  Real u1_tilt = 0.0;
  Real u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (pgen.psi != 0.0) {
    Real dtheta_dvartheta =
        (pgen.cos_psi * sin_vartheta
         + pgen.sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    Real dtheta_dvarphi = -pgen.sin_psi * sin_vartheta * sin_varphi / sin_theta;
    Real dphi_dvartheta = pgen.sin_psi * sin_varphi / SQR(sin_theta);
    Real dphi_dvarphi = sin_vartheta / SQR(sin_theta)
        * (pgen.cos_psi * sin_vartheta + pgen.sin_psi * cos_vartheta * cos_varphi);
    *pu2 = dtheta_dvartheta * u2_tilt + dtheta_dvarphi * u3_tilt;
    *pu3 = dphi_dvartheta * u2_tilt + dphi_dvarphi * u3_tilt;
  } else {
    *pu2 = u2_tilt;
    *pu3 = u3_tilt;
  }
  if (sin_theta < 0.0) {
    *pu2 *= -1.0;
    *pu3 *= -1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside untilted disk
// Inputs:
//   r: Boyer-Lindquist r
//   sin_theta: sine of Boyer-Lindquist theta
// Outputs:
//   pu0: u^t set (Boyer-Lindquist coordinates)
//   pu3: u^\phi set (Boyer-Lindquist coordinates)
// Notes:
//   The formula for u^3 as a function of u_{(\phi)} is tedious to derive, but this
//       matches the formula used in Harm (init.c).

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct torus_pgen pgen,
                                    Real r, Real sin_theta, Real *pu0, Real *pu3) {
  // Compute BL metric components
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);              // \Delta
  Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;         // \Sigma
  Real aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
  Real g_00 = -(1.0 - 2.0*r/sigma); // g_tt
  Real g_03 = -2.0*pgen.spin*r/sigma * sin_sq_theta; // g_tp
  Real g_33 = (sigma + (1.0 + 2.0*r/sigma) *
              SQR(pgen.spin) * sin_sq_theta) * sin_sq_theta; // g_pp
  Real g00 = -aa/(delta*sigma); // g^tt
  Real g03 = -2.0*pgen.spin*r/(delta*sigma); // g^tp

  Real u0 = 0.0, u3 = 0.0;
  // Compute non-zero components of 4-velocity
  if (pgen.fm_torus) {
    Real exp_2nu = sigma * delta / aa;                 // \exp(2\nu) (FM 3.5)
    Real exp_2psi = aa / sigma * sin_sq_theta;         // \exp(2\psi) (FM 3.5)
    Real exp_neg2chi = exp_2nu / exp_2psi;             // \exp(-2\chi) (cf. FM 2.15)
    Real u_phi_proj_a = 1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi;
    Real u_phi_proj_b = -1.0 + sqrt(u_phi_proj_a);
    Real u_phi_proj = sqrt(0.5 * u_phi_proj_b);        // (FM 3.3)
    u_phi_proj *= (pgen.prograde) ? 1.0 : -1.0;
    Real u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
    Real u3_b = 2.0*pgen.spin*r * sqrt(u3_a);
    Real u3_c = sqrt(sigma/aa) / sin_theta;
    u3 = u3_b + u3_c * u_phi_proj;
    Real u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
    Real u0_b = sqrt(u0_a - g_00);
    u0 = -1.0/g_00 * (g_03*u3 + u0_b);
  } else { // Chakrabarti torus
    Real l = CalculateL(pgen, r, sin_theta);
    Real u_0 = CalculateCovariantUT(pgen, r, sin_theta, l); // u_t
    Real omega = -(g_03 + l*g_00)/(g_33 + l*g_03);
    u0 = (g00 - l*g03) * u_0; // u^t
    u3 = omega * u0; // u^p
  }
  *pu0 = u0;
  *pu3 = u3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct torus_pgen pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - x2*pgen.spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + x1*pgen.spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(pgen.spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating vector potential in Spherical KS given CKS coordinates
// Inputs:
//   r,theta,phi spherical Boyer-Lindquist coordinates of point
// Outputs:
//   patheta,paphi: pointers to lower theta, phi components in desired coordinates

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotentialInTiltedTorus(struct torus_pgen pgen,
                                                  Real r, Real theta, Real phi,
                                                  Real *patheta, Real *paphi) {
  // Find vector potential components, accounting for tilt
  Real atheta = 0.0, aphi = 0.0;

  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real sin_phi = sin(phi);
  Real cos_phi = cos(phi);
  Real sin_vartheta;

  if (pgen.psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = pgen.cos_psi * x - pgen.sin_psi * z;
    Real vary = y;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
  } else {
    sin_vartheta = fabs(sin(theta));
  }

  if (pgen.is_vertical_field) {
    // Determine if we are in the torus
    Real rho;
    Real gm1 = pgen.gamma_adi - 1.0;
    bool in_torus = false;
    Real log_h = LogHAux(pgen, r, sin_vartheta) - pgen.log_h_edge;  // (FM 3.6)
    if (log_h >= 0.0) {
      in_torus = true;
      Real ptot_over_rho = gm1/pgen.gamma_adi * (exp(log_h) - 1.0);
      rho = pow(ptot_over_rho, 1.0/gm1) / pgen.rho_peak;
    }

    // more-or-less vertical geometry but falling to zero on edges
    Real cyl_radius = r * sin_vartheta;
    Real rcyl_in = pgen.r_edge;
    Real rcyl_falloff = pgen.potential_falloff;

    Real aphi_tilt = pow(cyl_radius/rcyl_in, pgen.potential_r_pow);
    if (pgen.potential_falloff != 0) {
      aphi_tilt *= exp(-cyl_radius/rcyl_falloff);
    }

    Real aphi_offset = exp(-rcyl_in/rcyl_falloff);
    if (cyl_radius < rcyl_in) {
      aphi_tilt = 0.0;
    } else {
      aphi_tilt -= aphi_offset;
    }

    if (pgen.potential_rho_pow != 0) {
      if (in_torus) {
        aphi_tilt *= pow(rho/pgen.rho_max, pgen.potential_rho_pow);
      } else {
        aphi_tilt = 0.0;
      }
    }

    if (pgen.psi != 0.0) {
      Real dvarphi_dtheta = -pgen.sin_psi * sin_phi / SQR(sin_vartheta);
      Real dvarphi_dphi = sin_theta / SQR(sin_vartheta)
          * (pgen.cos_psi * sin_theta - pgen.sin_psi * cos_theta * cos_phi);
      atheta = dvarphi_dtheta * aphi_tilt;
      aphi = dvarphi_dphi * aphi_tilt;
    } else {
      atheta = 0.0;
      aphi = aphi_tilt;
    }

  } else {
    if (r >= pgen.r_edge) {
      // Determine if we are in the torus
      Real rho;
      Real gm1 = pgen.gamma_adi-1.0;
      bool in_torus = false;
      Real log_h = LogHAux(pgen, r, sin_vartheta) - pgen.log_h_edge;  // (FM 3.6)
      if (log_h >= 0.0) {
        in_torus = true;
        Real ptot_over_rho = gm1/pgen.gamma_adi * (exp(log_h) - 1.0);
        rho = pow(ptot_over_rho, 1.0/gm1) / pgen.rho_peak;
      }

      Real aphi_tilt = 0.0;
      if (in_torus) {
        Real scaling_param = pow((r/pgen.r_edge)*sin_vartheta, pgen.potential_r_pow);
        if (pgen.potential_falloff != 0) {
          scaling_param *= exp(-r/pgen.potential_falloff);
        }
        aphi_tilt = pow(rho/pgen.rho_max, pgen.potential_rho_pow)*scaling_param;
        aphi_tilt -= pgen.potential_cutoff;
        aphi_tilt = fmax(aphi_tilt, 0.0);
        if (pgen.psi != 0.0) {
          Real dvarphi_dtheta = -pgen.sin_psi * sin_phi / SQR(sin_vartheta);
          Real dvarphi_dphi = sin_theta / SQR(sin_vartheta)
              * (pgen.cos_psi * sin_theta - pgen.sin_psi * cos_theta * cos_phi);
          atheta = dvarphi_dtheta * aphi_tilt;
          aphi = dvarphi_dphi * aphi_tilt;
        } else {
          atheta = 0.0;
          aphi = aphi_tilt;
        }
      }
    }
  }

  *patheta = atheta;
  *paphi = aphi;

  return;
}
//----------------------------------------------------------------------------------------
// Function to compute 1-component of vector potential.  First computes phi-componenent
// in spherical KS coordinates, then transforms to Cartesian KS

KOKKOS_INLINE_FUNCTION
Real A1(struct torus_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x1*x3*isin_term/(r*sqrt_term)) +
         aphi*(-x2/(SQR(x1)+SQR(x2))+pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(struct torus_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x2*x3*isin_term/(r*sqrt_term)) +
         aphi*(x1/(SQR(x1)+SQR(x2))+pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(struct torus_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) +
         aphi*(pgen.spin*x3/(r*sqrt_term));
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn NoInflowTorus
//  \brief Sets boundary condition on surfaces of computational domain
// FIXME: Boundaries need to be adjusted for DynGRMHD

void NoInflowTorus(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

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

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
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
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
  }
  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
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
        if (n==(IVX)) {
          w0_(m,n,k,j,is-i-1) = fmin(0.0,w0_(m,n,k,j,is));
        } else {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,ie+i+1) = fmax(0.0,w0_(m,n,k,j,ie));
        } else {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
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
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = i0_(m,n,k,j,ie);
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
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,js-j-1,i) = fmin(0.0,w0_(m,n,k,js,i));
        } else {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,je+j+1,i) = fmax(0.0,w0_(m,n,k,je,i));
        } else {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
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
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
  }
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pm->pmb_pack->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = fmin(0.0,w0_(m,n,ks,j,i));
        } else {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ke+k+1,j,i) = fmax(0.0,w0_(m,n,ke,j,i));
        } else {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
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
  // PrimToCons on X3 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void TorusFluxes(HistoryData *pdata, Mesh *pm) {
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
