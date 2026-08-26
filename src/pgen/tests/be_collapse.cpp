//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file be_collapse.cpp
//! \brief Problem generator for collapse of a Bonnor-Ebert-like sphere with AMR.
//!
//! Port of the Athena++ collapse.cpp problem generator (MHD + hydro).
//! Sets up an enhanced Bonnor-Ebert density profile and lets it collapse under
//! self-gravity solved by the multigrid Poisson solver.  Adaptive mesh refinement
//! is driven by a Jeans-length criterion registered through user_ref_func.
//!
//! Supports both isothermal and adiabatic (with barotropic cooling) EOS.
//! When using adiabatic EOS, a user source term enforces a barotropic relation
//! (isothermal below rhocrit, adiabatic above) and resets velocity outside the cloud.
//!
//! References:
//!   Tomida (2011) PhD Thesis (BE profile approximation)
//!   Tomida & Stone (2023) ApJS 266, 7 (multigrid + collapse test)

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

namespace {

// Dimensionless constants for BE sphere
constexpr Real rc_default = 6.45;
constexpr Real rcsq_fac   = 1.0 / 3.0;   // rcsq = rc^2 / 3
constexpr Real bemass      = 197.561;

// Physical constants (cgs)
constexpr Real cs10  = 1.9e4;           // sound speed at 10 K [cm/s]
constexpr Real msun  = 1.9891e33;       // solar mass [g]
constexpr Real au    = 1.4959787e13;    // astronomical unit [cm]
constexpr Real yr    = 3.15569e7;       // year [s]
constexpr Real G_cgs = 6.67259e-8;      // gravitational constant [dyn cm^2 g^-2]

// Unit system (computed from mass and temperature)
Real m0, v0, t0, l0, rho0_phys, gauss_unit;

// Runtime parameters stored for cooling source term
Real rc_global;
Real rhocrit_code;     // critical density in code units (0 = not used / isothermal)
Real gamma_global;     // gamma for adiabatic EOS
bool is_ideal_global;  // true if adiabatic EOS

// AMR parameter (set from input in pgen, read in refinement function)
Real njeans_threshold;
Real cs_global;  // effective sound speed for Jeans criterion

// Approximated Bonnor-Ebert density profile (Tomida 2011)
KOKKOS_INLINE_FUNCTION
Real BEProfile(Real r, Real rcsq) {
  return Kokkos::pow(1.0 + r * r / rcsq, -1.5);
}

}  // namespace

// Forward declarations
void JeansRefinement(MeshBlockPack *pmbp);
void BarotropicCooling(Mesh *pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BECollapse()
//! \brief Sets up a Bonnor-Ebert sphere for gravitational collapse with Jeans AMR.

void ProblemGenerator::BECollapse(ParameterInput *pin, const bool restart) {
  // --- AMR Jeans criterion (must be set on both fresh start and restart) ---
  njeans_threshold = pin->GetOrAddReal("problem", "njeans", 16.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Determine EOS type and sound speed
  is_ideal_global = false;
  gamma_global = 5.0/3.0;
  cs_global = 1.0;

  if (pmbp->pmhd != nullptr) {
    is_ideal_global = pmbp->pmhd->peos->eos_data.is_ideal;
    if (is_ideal_global) {
      gamma_global = pmbp->pmhd->peos->eos_data.gamma;
      cs_global = 1.0;  // isothermal cs=1 at low density by construction
    } else {
      cs_global = pmbp->pmhd->peos->eos_data.iso_cs;
    }
  } else if (pmbp->phydro != nullptr) {
    is_ideal_global = pmbp->phydro->peos->eos_data.is_ideal;
    if (is_ideal_global) {
      gamma_global = pmbp->phydro->peos->eos_data.gamma;
      cs_global = 1.0;
    } else {
      cs_global = pmbp->phydro->peos->eos_data.iso_cs;
    }
  }

  user_ref_func = JeansRefinement;

  // --- barotropic cooling source term (adiabatic EOS only) ---
  Real rc = pin->GetOrAddReal("problem", "cloud_radius", rc_default);
  rc_global = rc;

  Real mass = pin->GetOrAddReal("problem", "mass", 0.0);
  Real temp = pin->GetOrAddReal("problem", "temperature", 0.0);
  Real f    = pin->GetOrAddReal("problem", "f", 1.2);

  // Compute unit system from physical parameters (if mass > 0)
  if (mass > 0.0 && temp > 0.0) {
    m0 = mass * msun / (bemass * f);
    v0 = cs10 * std::sqrt(temp / 10.0);
    rho0_phys = std::pow(v0, 6) / (SQR(m0) * 64.0 * M_PI*M_PI*M_PI
                                   * G_cgs*G_cgs*G_cgs);
    t0 = 1.0 / std::sqrt(4.0 * M_PI * G_cgs * rho0_phys);
    l0 = v0 * t0;
    gauss_unit = std::sqrt(rho0_phys * SQR(v0) * 4.0 * M_PI);
  } else {
    m0 = v0 = t0 = l0 = rho0_phys = gauss_unit = 0.0;
  }

  // Critical density for barotropic transition
  Real rhocrit_cgs = pin->GetOrAddReal("problem", "rhocrit", 0.0);
  if (rhocrit_cgs > 0.0 && rho0_phys > 0.0) {
    rhocrit_code = rhocrit_cgs / rho0_phys;
  } else {
    rhocrit_code = 0.0;
  }

  if (is_ideal_global && rhocrit_code > 0.0) {
    user_srcs = true;
    user_srcs_func = BarotropicCooling;
  }

  if (restart) return;

  // --- gravity coupling ---
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (pmbp->pgrav != nullptr) {
    pmbp->pgrav->four_pi_G = four_pi_G;
    if (pmbp->pgrav->pmgd != nullptr) {
      pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
    }
  }

  // --- problem parameters ---
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);
  Real x_center = pin->GetOrAddReal("problem", "x_center", 0.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  Real rcsq = SQR(rc) * rcsq_fac;

  // Solid-body rotation: omega = omegatff / tff, where tff = pi*sqrt(3/(8f))
  Real tff = std::sqrt(3.0 / (8.0 * f)) * M_PI;
  Real omegatff = pin->GetOrAddReal("problem", "omegatff", 0.0);
  Real omega = omegatff / tff;

  // --- magnetic field strength ---
  // Prefer mass-to-flux ratio mu if provided; otherwise use b0_z directly
  Real mu = pin->GetOrAddReal("problem", "mu", 0.0);
  Real bz = 0.0;
  if (mu > 0.0 && mass > 0.0) {
    Real mucrit1 = 0.53 / (3.0 * M_PI) * std::sqrt(5.0 / G_cgs);
    bz = mass * msun / (mucrit1 * mu * M_PI * SQR(rc * l0)) / gauss_unit;
  } else {
    bz = pin->GetOrAddReal("problem", "b0_z", 0.0);
  }

  // Initial internal energy factor for adiabatic EOS: e_int = rho / (gamma-1)
  // (cs=1 at the initial state means P = rho * cs^2 = rho, so e = P/(gamma-1))
  Real igm1 = 0.0;
  if (is_ideal_global) {
    igm1 = 1.0 / (gamma_global - 1.0);
  }

  // --- initialize density, momentum, and energy ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;

  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    bool eos_ideal = is_ideal_global;
    Real igm1_l = igm1;
    par_for("be_collapse_hydro", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

      Real r = Kokkos::sqrt(SQR(x - x_center) + SQR(y - y_center)
                           + SQR(z - z_center));
      Real r_clamped = Kokkos::fmin(r, rc);

      Real rho = f * BEProfile(r_clamped, rcsq);
      if (amp > 0.0 && r < rc) {
        rho *= (1.0 + amp * SQR(r) / SQR(rc)
                * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
      }

      u0(m, IDN, k, j, i) = rho;
      Real mx = 0.0, my = 0.0;
      if (r < rc) {
        mx =  rho * omega * (y - y_center);
        my = -rho * omega * (x - x_center);
      }
      u0(m, IM1, k, j, i) = mx;
      u0(m, IM2, k, j, i) = my;
      u0(m, IM3, k, j, i) = 0.0;

      if (eos_ideal) {
        Real ke = 0.5 * (SQR(mx) + SQR(my)) / rho;
        u0(m, IEN, k, j, i) = igm1_l * rho + ke;
      }
    });
  } else if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    bool eos_ideal = is_ideal_global;
    Real igm1_l = igm1;
    par_for("be_collapse_mhd", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

      Real r = Kokkos::sqrt(SQR(x - x_center) + SQR(y - y_center)
                           + SQR(z - z_center));
      Real r_clamped = Kokkos::fmin(r, rc);

      Real rho = f * BEProfile(r_clamped, rcsq);
      if (amp > 0.0 && r < rc) {
        rho *= (1.0 + amp * SQR(r) / SQR(rc)
                * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
      }

      u0(m, IDN, k, j, i) = rho;
      Real mx = 0.0, my = 0.0;
      if (r < rc) {
        mx =  rho * omega * (y - y_center);
        my = -rho * omega * (x - x_center);
      }
      u0(m, IM1, k, j, i) = mx;
      u0(m, IM2, k, j, i) = my;
      u0(m, IM3, k, j, i) = 0.0;

      if (eos_ideal) {
        Real ke = 0.5 * (SQR(mx) + SQR(my)) / rho;
        Real me = 0.5 * SQR(bz);
        u0(m, IEN, k, j, i) = igm1_l * rho + ke + me;
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("be_collapse_bfield", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = bz;
      if (i==ie) b0.x1f(m,k,j,i+1) = 0.0;
      if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
      if (k==ke) b0.x3f(m,k+1,j,i) = bz;
      bcc0(m,IBX,k,j,i) = 0.0;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = bz;
    });
  } else {
    return;
  }

  // --- diagnostic output ---
  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- Bonnor-Ebert Collapse ---" << std::endl
      << "Density enhancement f   = " << f << std::endl
      << "Cloud radius rc         = " << rc << std::endl
      << "Free-fall time tff      = " << tff << std::endl
      << "Omega * tff             = " << omegatff << std::endl
      << "Angular velocity omega  = " << omega << std::endl
      << "Perturbation amplitude  = " << amp << std::endl
      << "Magnetic field b0_z     = " << bz << std::endl;
    if (mu > 0.0) {
      std::cout
        << "Mass-to-flux ratio mu   = " << mu << std::endl;
    }
    std::cout
      << "Jeans AMR threshold     = " << njeans_threshold << std::endl
      << "EOS type                = " << (is_ideal_global ? "ideal" : "isothermal")
      << std::endl;
    if (is_ideal_global) {
      std::cout
        << "Gamma                   = " << gamma_global << std::endl
        << "rhocrit (code units)    = " << rhocrit_code << std::endl;
    } else {
      std::cout
        << "Sound speed cs          = " << cs_global << std::endl;
    }
    std::cout
      << "four_pi_G               = " << four_pi_G << std::endl;
    if (mass > 0.0 && temp > 0.0) {
      std::cout << std::endl
        << "---  Dimensional parameters  ---" << std::endl
        << "Total mass          : " << mass      << " [Msun]" << std::endl
        << "Initial temperature : " << temp      << " [K]" << std::endl
        << "Central density     : " << rho0_phys*f << " [g/cm^3]" << std::endl
        << "Cloud radius        : " << rc*l0/au  << " [au]" << std::endl
        << "Free fall time      : " << tff*t0/yr << " [yr]" << std::endl
        << std::endl
        << "---   Normalization Units    ---" << std::endl
        << "Mass                : " << m0/msun   << " [Msun]" << std::endl
        << "Length              : " << l0/au     << " [au]" << std::endl
        << "Time                : " << t0/yr     << " [yr]" << std::endl
        << "Velocity            : " << v0        << " [cm/s]" << std::endl
        << "Density             : " << rho0_phys << " [g/cm^3]" << std::endl;
      if (gauss_unit > 0.0) {
        std::cout
          << "Magnetic field      : " << gauss_unit << " [Gauss]" << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BarotropicCooling()
//! \brief User source term: enforces barotropic relation
//!        and resets velocity outside cloud.
//!
//! Internal energy is set to:
//!   e_int = rho/(gamma-1) * sqrt(1 + (rho/rhocrit)^(2*(gamma-1)))
//! This is isothermal (cs=1) at low density and adiabatic above rhocrit.
//! Momentum is zeroed outside the initial cloud radius.

void BarotropicCooling(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  Real rc = rc_global;
  Real rhocrit = rhocrit_code;
  Real gm1 = gamma_global - 1.0;
  Real igm1 = 1.0 / gm1;

  if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("barotropic_cooling_mhd", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
      Real r2 = SQR(x) + SQR(y) + SQR(z);

      // Zero momentum outside the initial cloud
      if (r2 > SQR(rc)) {
        u0(m, IM1, k, j, i) = 0.0;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = 0.0;
      }

      Real rho = u0(m, IDN, k, j, i);
      Real ke = 0.5 / rho * (SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i))
                            + SQR(u0(m, IM3, k, j, i)));
      Real me = 0.5 * (SQR(bcc0(m,IBX,k,j,i)) + SQR(bcc0(m,IBY,k,j,i))
                      + SQR(bcc0(m,IBZ,k,j,i)));

      // Barotropic internal energy: isothermal at rho << rhocrit, adiabatic above
      Real te = igm1 * rho
              * Kokkos::sqrt(1.0 + Kokkos::pow(rho/rhocrit, 2.0*gm1));
      u0(m, IEN, k, j, i) = te + ke + me;
    });
  } else if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    par_for("barotropic_cooling_hydro", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
      Real r2 = SQR(x) + SQR(y) + SQR(z);

      if (r2 > SQR(rc)) {
        u0(m, IM1, k, j, i) = 0.0;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = 0.0;
      }

      Real rho = u0(m, IDN, k, j, i);
      Real ke = 0.5 / rho * (SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i))
                            + SQR(u0(m, IM3, k, j, i)));
      Real te = igm1 * rho
              * Kokkos::sqrt(1.0 + Kokkos::pow(rho/rhocrit, 2.0*gm1));
      u0(m, IEN, k, j, i) = te + ke;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void JeansRefinement()
//! \brief Jeans-length AMR criterion for self-gravitating gas.
//!
//! For each meshblock, computes the minimum Jeans number:
//!   nJ = cs / sqrt(rho_max) * (2*pi / dx)
//! For MHD, includes Alfven speed: cs_eff = cs + v_A.

void JeansRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int ng = indcs.ng;
  const int nkji = (nx3 + 2 * ng) * (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int nji  = (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int ni   = (nx1 + 2 * ng);
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  DvceArray5D<Real> u0;
  bool has_bfield = false;
  DvceArray5D<Real> bcc0;
  if (pmbp->pmhd != nullptr) {
    u0 = pmbp->pmhd->u0;
    bcc0 = pmbp->pmhd->bcc0;
    has_bfield = true;
  } else {
    u0 = pmbp->phydro->u0;
  }
  auto &size = pmbp->pmb->mb_size;
  Real cs = cs_global;
  Real njeans = njeans_threshold;
  bool ideal = is_ideal_global;

  par_for_outer("JeansAMR", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_rhomax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &rhomax) {
        int k = idx / nji;
        int j = (idx - k * nji) / ni;
        int i = (idx - k * nji - j * ni);
        rhomax = Kokkos::fmax(u0(m, IDN, k, j, i), rhomax);
      },
      Kokkos::Max<Real>(team_rhomax));

    Real dx = size.d_view(m).dx1;
    Real v_eff = cs;

    // For MHD with isothermal EOS, add Alfven speed contribution
    if (has_bfield && !ideal) {
      Real bsq_max = 0.0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [&](const int idx, Real &bmax) {
          int k = idx / nji;
          int j = (idx - k * nji) / ni;
          int i = (idx - k * nji - j * ni);
          Real bsq = SQR(bcc0(m,IBX,k,j,i)) + SQR(bcc0(m,IBY,k,j,i))
                   + SQR(bcc0(m,IBZ,k,j,i));
          bmax = Kokkos::fmax(bsq / u0(m, IDN, k, j, i), bmax);
        },
        Kokkos::Max<Real>(bsq_max));
      v_eff = cs + Kokkos::sqrt(bsq_max);
    }

    Real nj_min = v_eff / Kokkos::sqrt(team_rhomax) * (2.0 * M_PI / dx);

    if (nj_min < njeans) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (nj_min > njeans * 2.5) {
      refine_flag.d_view(m + mbs) = -1;
    } else {
      refine_flag.d_view(m + mbs) = 0;
    }
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
