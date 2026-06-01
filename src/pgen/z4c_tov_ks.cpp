//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_tov_ks.cpp
//! \brief TOV-star residual data on top of an analytic Kerr-Schild background

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pgen/pgen.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"

namespace {

Real bh_spin = 0.0;
Real bh_mass = 1.0;
bool use_minkowski_background = false;
Real bh_center_x1 = 0.0;
Real bh_center_x2 = 0.0;
Real bh_center_x3 = 0.0;
Real bh_horizon_radius = 1.0;
Real star_center_x1 = 8.0;
Real star_center_x2 = 0.0;
Real star_center_x3 = 0.0;
Real star_boost_x = 0.0;
Real star_boost_y = 0.0;
Real star_boost_z = 0.0;
Real star_boost_mag = 0.0;
bool star_orbit_circular_geodesic = false;
Real star_orbit_radius = -1.0;
Real star_orbit_radius_factor = 2.0;
Real star_orbit_phase = 0.0;
Real star_orbit_omega = 0.0;
Real star_orbit_ut = 1.0;
Real star_orbit_uphi = 0.0;
Real star_orbit_tidal_radius = 0.0;
Real star_orbit_boost_speed = 0.0;
Real star_rot_00 = 1.0;
Real star_rot_01 = 0.0;
Real star_rot_02 = 0.0;
Real star_rot_10 = 0.0;
Real star_rot_11 = 1.0;
Real star_rot_12 = 0.0;
Real star_rot_20 = 0.0;
Real star_rot_21 = 0.0;
Real star_rot_22 = 1.0;
bool star_isotropic = true;
Real excision_damp_rate = 50.0;
bool excision_project_state = true;
Real excision_freeze_radius = 0.0;
Real excision_ramp_radius = 0.0;
Real excision_atmo_density = 0.0;
Real excision_atmo_energy = 0.0;
Real amr_rho_slope_threshold = 0.5;
Real amr_rho_min = 0.0;
Real amr_bh_exclusion_radius = 0.0;
bool amr_star_refine = false;
Real amr_star_refine_radius = 0.0;
int amr_star_refine_level = -1;

void TOVKerrSchildHistory(HistoryData *pdata, Mesh *pm);

KOKKOS_INLINE_FUNCTION
Real KerrSchildRadius(Real x, Real y, Real z, Real a) {
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real discr = SQR(SQR(rad) - SQR(a)) + 4.0*SQR(a)*SQR(z);
  Real r = sqrt((SQR(rad) - SQR(a) + sqrt(discr))/2.0);
  Real eps = 1.0e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  return r;
}

KOKKOS_INLINE_FUNCTION
Real SmootherStep(Real q) {
  q = fmax(0.0, fmin(1.0, q));
  return q*q*q*(10.0 + q*(-15.0 + 6.0*q));
}

KOKKOS_INLINE_FUNCTION
Real InnerExcisionRamp(Real x, Real y, Real z, Real spin, Real freeze_radius,
                       Real ramp_radius) {
  Real r_ks = KerrSchildRadius(x, y, z, spin);
  if (r_ks <= freeze_radius) {
    return 0.0;
  }
  if (r_ks >= ramp_radius) {
    return 1.0;
  }
  Real width = ramp_radius - freeze_radius;
  if (width <= 0.0) {
    return 1.0;
  }
  return SmootherStep((r_ks - freeze_radius)/width);
}

KOKKOS_INLINE_FUNCTION
Real BlendFiniteToTarget(Real value, Real target, Real ramp) {
  if (!(isfinite(value))) {
    return target;
  }
  return ramp*value + (1.0 - ramp)*target;
}

template <class TOVEOS>
KOKKOS_INLINE_FUNCTION
void SampleIsotropicTOV(const TOVEOS &eos, const tov::TOVStar &tov_star, Real r,
                        Real &rho, Real &p, Real &mass, Real &alp, Real &psi4) {
  tov_star.GetPrimitivesAtIsoPoint(eos, r, rho, p, mass, alp);
  Real r_schw = tov_star.FindSchwarzschildR(r, mass);
  Real fmet = 1.0;
  if (r > 0.0) {
    fmet = r_schw/r;
  }
  psi4 = fmet*fmet;
}

template <class TOVEOS, class Array5D>
KOKKOS_INLINE_FUNCTION
void SetYeScalar(std::true_type, const TOVEOS &eos, Array5D w0, Real rho, Real ye_atmo,
                 int nscalars, int scalar_index, int m, int k, int j, int i) {
  if (nscalars >= 1) {
    Real ye = ye_atmo;
    if (rho > 0.0) {
      ye = eos.template GetYeFromRho<tov::LocationTag::Device>(rho);
    }
    w0(m,scalar_index,k,j,i) = ye;
  }
}

template <class TOVEOS, class Array5D>
KOKKOS_INLINE_FUNCTION
void SetYeScalar(std::false_type, const TOVEOS &, Array5D, Real, Real, int, int,
                 int, int, int, int) {}

inline void SetStarBoostRotation() {
  star_boost_mag = std::sqrt(SQR(star_boost_x) + SQR(star_boost_y) + SQR(star_boost_z));
  if (star_boost_mag <= 0.0) {
    star_rot_00 = 1.0; star_rot_01 = 0.0; star_rot_02 = 0.0;
    star_rot_10 = 0.0; star_rot_11 = 1.0; star_rot_12 = 0.0;
    star_rot_20 = 0.0; star_rot_21 = 0.0; star_rot_22 = 1.0;
    return;
  }

  Real nx = star_boost_x/star_boost_mag;
  Real ny = star_boost_y/star_boost_mag;
  Real nz = star_boost_z/star_boost_mag;

  Real rx = (std::fabs(nx) < 0.9) ? 1.0 : 0.0;
  Real ry = (std::fabs(nx) < 0.9) ? 0.0 : 1.0;
  Real rz = 0.0;

  Real dot = nx*rx + ny*ry + nz*rz;
  Real e2x = rx - dot*nx;
  Real e2y = ry - dot*ny;
  Real e2z = rz - dot*nz;
  Real e2norm = std::sqrt(e2x*e2x + e2y*e2y + e2z*e2z);
  if (e2norm <= 0.0) {
    e2x = 0.0;
    e2y = 0.0;
    e2z = 1.0;
    e2norm = 1.0;
  }
  e2x /= e2norm;
  e2y /= e2norm;
  e2z /= e2norm;

  Real e3x = ny*e2z - nz*e2y;
  Real e3y = nz*e2x - nx*e2z;
  Real e3z = nx*e2y - ny*e2x;

  star_rot_00 = nx;  star_rot_01 = ny;  star_rot_02 = nz;
  star_rot_10 = e2x; star_rot_11 = e2y; star_rot_12 = e2z;
  star_rot_20 = e3x; star_rot_21 = e3y; star_rot_22 = e3z;
}

inline void StarCenterAtTime(Real time, Real &x, Real &y, Real &z) {
  if (star_orbit_circular_geodesic) {
    Real phi = star_orbit_phase + star_orbit_omega*time;
    x = star_orbit_radius*std::cos(phi);
    y = star_orbit_radius*std::sin(phi);
    z = star_center_x3;
    return;
  }
  x = star_center_x1 + star_boost_x*time;
  y = star_center_x2 + star_boost_y*time;
  z = star_center_x3 + star_boost_z*time;
}

inline void ConfigureCircularGeodesicOrbit(const tov::TOVStar &tov_star) {
  if (!star_orbit_circular_geodesic) {
    return;
  }
  if (use_minkowski_background || std::fabs(bh_mass - 1.0) > 1.0e-12 ||
      std::fabs(bh_spin) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "star_orbit = circular_geodesic currently requires "
              << "a unit-mass Schwarzschild Kerr-Schild background." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  star_orbit_tidal_radius =
      tov_star.R_edge*std::pow(bh_mass/fmax(tov_star.M_edge, 1.0e-300), 1.0/3.0);
  if (star_orbit_radius <= 0.0) {
    star_orbit_radius = star_orbit_radius_factor*star_orbit_tidal_radius;
  }
  if (star_orbit_radius <= 3.0*bh_mass) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Circular Schwarzschild geodesic requested at r = "
              << star_orbit_radius << ", but circular timelike geodesics require r > 3M."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  star_orbit_omega = std::sqrt(bh_mass/(star_orbit_radius*star_orbit_radius*
                                        star_orbit_radius));
  star_orbit_ut = 1.0/std::sqrt(1.0 - 3.0*bh_mass/star_orbit_radius);
  star_orbit_uphi = star_orbit_omega*star_orbit_ut;

  Real u_tangent = star_orbit_radius*star_orbit_uphi;
  star_orbit_boost_speed = u_tangent/std::sqrt(1.0 + SQR(u_tangent));
  Real ephi_x = -std::sin(star_orbit_phase);
  Real ephi_y =  std::cos(star_orbit_phase);

  star_center_x1 = star_orbit_radius*std::cos(star_orbit_phase);
  star_center_x2 = star_orbit_radius*std::sin(star_orbit_phase);
  star_boost_x = star_orbit_boost_speed*ephi_x;
  star_boost_y = star_orbit_boost_speed*ephi_y;
  star_boost_z = 0.0;
  SetStarBoostRotation();

  if (global_variable::my_rank == 0) {
    std::cout << "CIRCULAR SCHWARZSCHILD GEODESIC STAR ORBIT\n"
              << "------------------------------------------\n"
              << "Tidal radius: " << star_orbit_tidal_radius << "\n"
              << "Orbit radius: " << star_orbit_radius << "\n"
              << "Orbit phase: " << star_orbit_phase << "\n"
              << "Omega=dphi/dt: " << star_orbit_omega << "\n"
              << "u^t: " << star_orbit_ut << "\n"
              << "u^phi: " << star_orbit_uphi << "\n"
              << "Tangential boost speed: " << star_orbit_boost_speed << "\n"
              << "Initial star center: (" << star_center_x1 << ", "
              << star_center_x2 << ", " << star_center_x3 << ")\n"
              << "Initial star_boost: (" << star_boost_x << ", "
              << star_boost_y << ", " << star_boost_z << ")\n\n";
  }
}

void ApplyInnerExcision(Mesh *pm, Real bdt, bool project_mhd) {
  if (excision_damp_rate <= 0.0 && !excision_project_state) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;
  const Real bh_center_x1_l = bh_center_x1;
  const Real bh_center_x2_l = bh_center_x2;
  const Real bh_center_x3_l = bh_center_x3;
  const Real bh_spin_l = bh_spin;
  const Real excision_freeze_radius_l = excision_freeze_radius;
  const Real excision_ramp_radius_l = excision_ramp_radius;
  const Real excision_damp_rate_l = excision_damp_rate;
  const Real excision_atmo_density_l = excision_atmo_density;
  const Real excision_atmo_energy_l = excision_atmo_energy;
  const bool excision_project_state_l = excision_project_state;

  if (project_mhd && pmbp->pmhd != nullptr) {
    auto &mhd_u0 = pmbp->pmhd->u0;
    auto &mhd_u1 = pmbp->pmhd->u1;
    int nmhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    int nbase = pmbp->pmhd->nmhd;
    par_for("z4c_tov_ks_inner_excision_mhd", DevExeSpace(), 0, nmb - 1, 0, nmhd - 1,
            ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - bh_center_x1_l;
      Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - bh_center_x2_l;
      Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - bh_center_x3_l;
      Real ramp = InnerExcisionRamp(x, y, z, bh_spin_l, excision_freeze_radius_l,
                                    excision_ramp_radius_l);
      if (ramp >= 1.0 && isfinite(mhd_u0(m,n,k,j,i)) && isfinite(mhd_u1(m,n,k,j,i))) {
        return;
      }

      Real target = 0.0;
      if (n == IDN) {
        target = excision_atmo_density_l;
      } else if (n == IEN && nbase > IEN) {
        target = excision_atmo_energy_l;
      }

      // Only cell-centered hydrodynamic conserved variables are projected here.
      // Face- and cell-centered magnetic fields are intentionally untouched.
      if (excision_project_state_l) {
        mhd_u0(m,n,k,j,i) = BlendFiniteToTarget(mhd_u0(m,n,k,j,i), target, ramp);
        mhd_u1(m,n,k,j,i) = BlendFiniteToTarget(mhd_u1(m,n,k,j,i), target, ramp);
      } else {
        if (!(isfinite(mhd_u0(m,n,k,j,i)))) {
          mhd_u0(m,n,k,j,i) = target;
        }
        if (!(isfinite(mhd_u1(m,n,k,j,i)))) {
          mhd_u1(m,n,k,j,i) = target;
        }
      }
    });
  }

  auto &z4c_u0 = pmbp->pz4c->u0;
  auto &z4c_u1 = pmbp->pz4c->u1;
  auto &z4c_rhs = pmbp->pz4c->u_rhs;
  int nz4c = pmbp->pz4c->nz4c;
  par_for("z4c_tov_ks_inner_excision_z4c", DevExeSpace(), 0, nmb - 1, 0, nz4c - 1,
          ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - bh_center_x1_l;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - bh_center_x2_l;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - bh_center_x3_l;
    Real ramp = InnerExcisionRamp(x, y, z, bh_spin_l, excision_freeze_radius_l,
                                  excision_ramp_radius_l);
    if (ramp >= 1.0 && isfinite(z4c_rhs(m,n,k,j,i)) && isfinite(z4c_u0(m,n,k,j,i)) &&
        isfinite(z4c_u1(m,n,k,j,i))) {
      return;
    }
    if (excision_damp_rate_l > 0.0) {
      Real damp = fmax(0.0, 1.0 - bdt*excision_damp_rate_l*(1.0 - ramp));
      z4c_rhs(m,n,k,j,i) = isfinite(z4c_rhs(m,n,k,j,i)) ?
                            damp*ramp*z4c_rhs(m,n,k,j,i) : 0.0;
    } else {
      z4c_rhs(m,n,k,j,i) = isfinite(z4c_rhs(m,n,k,j,i)) ?
                            ramp*z4c_rhs(m,n,k,j,i) : 0.0;
    }
    if (excision_project_state_l) {
      z4c_u0(m,n,k,j,i) = isfinite(z4c_u0(m,n,k,j,i)) ?
                           ramp*z4c_u0(m,n,k,j,i) : 0.0;
      z4c_u1(m,n,k,j,i) = isfinite(z4c_u1(m,n,k,j,i)) ?
                           ramp*z4c_u1(m,n,k,j,i) : 0.0;
    }
  });
}

void ApplyInnerExcision(Mesh *pm, Real bdt) {
  ApplyInnerExcision(pm, bdt, true);
}

void TOVKerrSchildHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  auto &w0 = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nmkji = pm->pmb_pack->nmb_thispack*nx3*nx2*nx1;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;

  Real rho_max = -std::numeric_limits<Real>::max();
  Real alpha_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "TOVKerrSchildHistory", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &rho_local, Real &alpha_local) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = idx - m*nkji - k*nji - j*nx1 + is;
        k += ks;
        j += js;

        rho_local = fmax(rho_local, w0(m, IDN, k, j, i));
        alpha_local = fmin(alpha_local, adm.alpha(m, k, j, i));
      },
      Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.0;
    alpha_min = 0.0;
  }
#endif

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

void RefinementCondition(MeshBlockPack *pmbp) {
  if (pmbp->pmhd == nullptr) {
    pmbp->pz4c->pamr->Refine(pmbp);
    return;
  }

  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  int nmb = pmbp->nmb_thispack;
  bool multi_d = pmbp->pmesh->multi_d;
  bool three_d = pmbp->pmesh->three_d;
  auto &w0 = pmbp->pmhd->w0;
  auto &size = pmbp->pmb->mb_size;
  const Real rho_slope_threshold_l = amr_rho_slope_threshold;
  const Real rho_min_l = amr_rho_min;
  const Real bh_exclusion_radius_l = amr_bh_exclusion_radius;
  const Real bh_center_x1_l = bh_center_x1;
  const Real bh_center_x2_l = bh_center_x2;
  const Real bh_center_x3_l = bh_center_x3;
  const Real bh_spin_l = bh_spin;

  par_for_outer("z4c_tov_ks_rho_gradient_refinement", DevExeSpace(), 0, 0, 0, nmb - 1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real x1min = size.d_view(m).x1min - bh_center_x1_l;
    Real x1max = size.d_view(m).x1max - bh_center_x1_l;
    Real x2min = size.d_view(m).x2min - bh_center_x2_l;
    Real x2max = size.d_view(m).x2max - bh_center_x2_l;
    Real x3min = size.d_view(m).x3min - bh_center_x3_l;
    Real x3max = size.d_view(m).x3max - bh_center_x3_l;
    Real cx = 0.5*(x1min + x1max);
    Real cy = 0.5*(x2min + x2max);
    Real cz = 0.5*(x3min + x3max);
    Real r_ks = KerrSchildRadius(cx, cy, cz, bh_spin_l);

    Real team_dqmax = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, Real& dqmax) {
      int k = idx/nji;
      int j = (idx - k*nji)/nx1;
      int i = (idx - k*nji - j*nx1) + is;
      j += js;
      k += ks;
      Real rho = w0(m,IDN,k,j,i);
      if (!isfinite(rho) || rho <= rho_min_l) {
        return;
      }
      Real d2 = SQR(w0(m,IDN,k,j,i+1) - w0(m,IDN,k,j,i-1));
      if (multi_d) {
        d2 += SQR(w0(m,IDN,k,j+1,i) - w0(m,IDN,k,j-1,i));
      }
      if (three_d) {
        d2 += SQR(w0(m,IDN,k+1,j,i) - w0(m,IDN,k-1,j,i));
      }
      Real denom = fmax(rho, rho_min_l);
      dqmax = fmax(0.5*sqrt(d2)/denom, dqmax);
    }, Kokkos::Max<Real>(team_dqmax));

    int &flag = refine_flag.d_view(m + mbs);
    if (bh_exclusion_radius_l > 0.0 && r_ks < bh_exclusion_radius_l) {
      if (flag == 0) {
        flag = -1;
      }
      return;
    }
    if (team_dqmax > rho_slope_threshold_l) {
      flag = 1;
    } else if (flag == 0) {
      flag = -1;
    }
  });
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  if (amr_star_refine && amr_star_refine_radius > 0.0 && amr_star_refine_level >= 0) {
    const Real time = pmbp->pmesh->time;
    Real xstar, ystar, zstar;
    StarCenterAtTime(time, xstar, ystar, zstar);
    const Real rstar2 = SQR(amr_star_refine_radius);
    for (int m = 0; m < nmb; ++m) {
      const int level = pmbp->pmesh->lloc_eachmb[m + mbs].level - pmbp->pmesh->root_level;
      const auto &mb_size = size.h_view(m);
      const bool contains =
          (xstar >= mb_size.x1min && xstar <= mb_size.x1max) &&
          (ystar >= mb_size.x2min && ystar <= mb_size.x2max) &&
          (zstar >= mb_size.x3min && zstar <= mb_size.x3max);
      const Real d2[8] = {
          SQR(mb_size.x1min - xstar) + SQR(mb_size.x2min - ystar) + SQR(mb_size.x3min - zstar),
          SQR(mb_size.x1max - xstar) + SQR(mb_size.x2min - ystar) + SQR(mb_size.x3min - zstar),
          SQR(mb_size.x1min - xstar) + SQR(mb_size.x2max - ystar) + SQR(mb_size.x3min - zstar),
          SQR(mb_size.x1max - xstar) + SQR(mb_size.x2max - ystar) + SQR(mb_size.x3min - zstar),
          SQR(mb_size.x1min - xstar) + SQR(mb_size.x2min - ystar) + SQR(mb_size.x3max - zstar),
          SQR(mb_size.x1max - xstar) + SQR(mb_size.x2min - ystar) + SQR(mb_size.x3max - zstar),
          SQR(mb_size.x1min - xstar) + SQR(mb_size.x2max - ystar) + SQR(mb_size.x3max - zstar),
          SQR(mb_size.x1max - xstar) + SQR(mb_size.x2max - ystar) + SQR(mb_size.x3max - zstar),
      };
      const Real dmin2 = *std::min_element(&d2[0], &d2[8]);
      if (contains || dmin2 < rstar2) {
        if (level < amr_star_refine_level) {
          refine_flag.h_view(m + mbs) = 1;
        } else if (level == amr_star_refine_level && refine_flag.h_view(m + mbs) == -1) {
          refine_flag.h_view(m + mbs) = 0;
        }
      }
    }
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
  }
}

template <typename ADMState>
void FillFlatADM(MeshBlockPack *pmbp, ADMState &adm_state) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  par_for("z4c_tov_ks_flat_adm", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm_state.alpha(m,k,j,i) = 1.0;
    adm_state.beta_u(m,0,k,j,i) = 0.0;
    adm_state.beta_u(m,1,k,j,i) = 0.0;
    adm_state.beta_u(m,2,k,j,i) = 0.0;
    adm_state.psi4(m,k,j,i) = 1.0;

    adm_state.g_dd(m,0,0,k,j,i) = 1.0;
    adm_state.g_dd(m,0,1,k,j,i) = 0.0;
    adm_state.g_dd(m,0,2,k,j,i) = 0.0;
    adm_state.g_dd(m,1,1,k,j,i) = 1.0;
    adm_state.g_dd(m,1,2,k,j,i) = 0.0;
    adm_state.g_dd(m,2,2,k,j,i) = 1.0;

    adm_state.vK_dd(m,0,0,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,2,2,k,j,i) = 0.0;
  });
}

template <typename ADMState>
void FillKerrSchildADM(MeshBlockPack *pmbp, ADMState &adm_state) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;
  const Real bh_center_x1_l = bh_center_x1;
  const Real bh_center_x2_l = bh_center_x2;
  const Real bh_center_x3_l = bh_center_x3;
  const Real bh_spin_l = bh_spin;
  const Real excision_freeze_radius_l = excision_freeze_radius;
  const bool excision_project_state_l = excision_project_state;

  par_for("z4c_tov_ks_background", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - bh_center_x1_l;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - bh_center_x2_l;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - bh_center_x3_l;
    Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
    Real r_ks = KerrSchildRadius(x, y, z, bh_spin_l);
    if (excision_project_state_l && r_ks < excision_freeze_radius_l) {
      Real scale = excision_freeze_radius_l/fmax(rad, 1.0e-12);
      x = (rad > 1.0e-12) ? x*scale : excision_freeze_radius_l;
      y = (rad > 1.0e-12) ? y*scale : 0.0;
      z = (rad > 1.0e-12) ? z*scale : 0.0;
    }

    ComputeADMDecomposition(
        x, y, z, false, bh_spin_l,
        &adm_state.alpha(m,k,j,i),
        &adm_state.beta_u(m,0,k,j,i), &adm_state.beta_u(m,1,k,j,i),
        &adm_state.beta_u(m,2,k,j,i), &adm_state.psi4(m,k,j,i),
        &adm_state.g_dd(m,0,0,k,j,i), &adm_state.g_dd(m,0,1,k,j,i),
        &adm_state.g_dd(m,0,2,k,j,i), &adm_state.g_dd(m,1,1,k,j,i),
        &adm_state.g_dd(m,1,2,k,j,i), &adm_state.g_dd(m,2,2,k,j,i),
        &adm_state.vK_dd(m,0,0,k,j,i), &adm_state.vK_dd(m,0,1,k,j,i),
        &adm_state.vK_dd(m,0,2,k,j,i), &adm_state.vK_dd(m,1,1,k,j,i),
        &adm_state.vK_dd(m,1,2,k,j,i), &adm_state.vK_dd(m,2,2,k,j,i));
  });
}

void SetADMBackgroundKerrSchild(MeshBlockPack *pmbp, Real /*time*/) {
  if (use_minkowski_background) {
    FillFlatADM(pmbp, pmbp->pz4c->adm_bg);
  } else {
    FillKerrSchildADM(pmbp, pmbp->pz4c->adm_bg);
  }
}

template <class TOVEOS>
void FillTOVPrimitivesAndADM(ParameterInput *pin, Mesh *pmy_mesh, TOVEOS &eos,
                             const tov::TOVStar &tov_star) {
  MeshBlockPack *pmbp = pmy_mesh->pmb_pack;
  auto &w0 = pmbp->pmhd->w0;
  auto &adm_state = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmy_mesh->mb_indcs;

  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  Real dfloor = pin->GetOrAddReal("mhd", "dfloor", tov_star.dfloor);
  Real pfloor = pin->GetOrAddReal("mhd", "pfloor", tov_star.pfloor);
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;
  Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);
  int nvars = pmbp->pmhd->nmhd;
  int nscalars = pmbp->pmhd->nscalars;
  Real lorentz = 1.0/std::sqrt(fmax(1.0e-16, 1.0 - SQR(star_boost_mag)));
  const Real star_center_x1_l = star_center_x1;
  const Real star_center_x2_l = star_center_x2;
  const Real star_center_x3_l = star_center_x3;
  const Real star_boost_mag_l = star_boost_mag;
  const bool star_isotropic_l = star_isotropic;
  const Real star_rot_00_l = star_rot_00;
  const Real star_rot_01_l = star_rot_01;
  const Real star_rot_02_l = star_rot_02;
  const Real star_rot_10_l = star_rot_10;
  const Real star_rot_11_l = star_rot_11;
  const Real star_rot_12_l = star_rot_12;
  const Real star_rot_20_l = star_rot_20;
  const Real star_rot_21_l = star_rot_21;
  const Real star_rot_22_l = star_rot_22;

  par_for("z4c_tov_ks_star", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - star_center_x1_l;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - star_center_x2_l;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - star_center_x3_l;

    Real xb = star_rot_00_l*x + star_rot_01_l*y + star_rot_02_l*z;
    Real yb = star_rot_10_l*x + star_rot_11_l*y + star_rot_12_l*z;
    Real zb = star_rot_20_l*x + star_rot_21_l*y + star_rot_22_l*z;
    Real xr = lorentz*xb;
    Real yr = yb;
    Real zr = zb;
    Real r = sqrt(SQR(xr) + SQR(yr) + SQR(zr));

    Real rho, p, mass, alp, psi4_rest;
    if (star_isotropic_l) {
      SampleIsotropicTOV(eos, tov_star, r, rho, p, mass, alp, psi4_rest);
    } else {
      tov_star.GetPrimitivesAtPoint(eos, r, rho, p, mass, alp);
      psi4_rest = 1.0;
    }

    w0(m,IDN,k,j,i) = fmax(rho, dfloor);
    w0(m,IPR,k,j,i) = fmax(p, pfloor);
    SetYeScalar(std::integral_constant<bool, use_ye>{}, eos, w0, rho, ye_atmo,
                nscalars, nvars, m, k, j, i);

    if (star_isotropic_l) {
      Real a_rest = SQR(alp);
      Real A = SQR(lorentz)*(psi4_rest - SQR(star_boost_mag_l)*a_rest);
      Real Bcov = SQR(lorentz)*star_boost_mag_l*(a_rest - psi4_rest);
      Real alpha_lab = std::sqrt(fmax(1.0e-16, a_rest*psi4_rest/fmax(A, 1.0e-16)));
      Real beta_par = Bcov/fmax(A, 1.0e-16);

      adm_state.alpha(m,k,j,i) = alpha_lab;
      adm_state.beta_u(m,0,k,j,i) = beta_par*star_rot_00_l;
      adm_state.beta_u(m,1,k,j,i) = beta_par*star_rot_01_l;
      adm_state.beta_u(m,2,k,j,i) = beta_par*star_rot_02_l;

      adm_state.g_dd(m,0,0,k,j,i) =
          psi4_rest + (A - psi4_rest)*SQR(star_rot_00_l);
      adm_state.g_dd(m,0,1,k,j,i) =
          (A - psi4_rest)*star_rot_00_l*star_rot_01_l;
      adm_state.g_dd(m,0,2,k,j,i) =
          (A - psi4_rest)*star_rot_00_l*star_rot_02_l;
      adm_state.g_dd(m,1,1,k,j,i) =
          psi4_rest + (A - psi4_rest)*SQR(star_rot_01_l);
      adm_state.g_dd(m,1,2,k,j,i) =
          (A - psi4_rest)*star_rot_01_l*star_rot_02_l;
      adm_state.g_dd(m,2,2,k,j,i) =
          psi4_rest + (A - psi4_rest)*SQR(star_rot_02_l);
      adm_state.psi4(m,k,j,i) = pow(fmax(A*psi4_rest*psi4_rest, 1.0e-16), 1.0/3.0);

      Real u0 = lorentz/fmax(alp, 1.0e-16);
      Real ui_par = lorentz*star_boost_mag_l/fmax(alp, 1.0e-16);
      Real uu_par = ui_par + u0*beta_par;
      w0(m,IVX,k,j,i) = uu_par*star_rot_00_l;
      w0(m,IVY,k,j,i) = uu_par*star_rot_01_l;
      w0(m,IVZ,k,j,i) = uu_par*star_rot_02_l;
    } else {
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      adm_state.alpha(m,k,j,i) = alp;
      adm_state.beta_u(m,0,k,j,i) = 0.0;
      adm_state.beta_u(m,1,k,j,i) = 0.0;
      adm_state.beta_u(m,2,k,j,i) = 0.0;
      Real fmet = 0.0;
      if (r > 0.0) {
        fmet = (1.0/(1.0 - 2.0*mass/r) - 1.0)/(r*r);
      }
      adm_state.g_dd(m,0,0,k,j,i) = x*x*fmet + 1.0;
      adm_state.g_dd(m,0,1,k,j,i) = x*y*fmet;
      adm_state.g_dd(m,0,2,k,j,i) = x*z*fmet;
      adm_state.g_dd(m,1,1,k,j,i) = y*y*fmet + 1.0;
      adm_state.g_dd(m,1,2,k,j,i) = y*z*fmet;
      adm_state.g_dd(m,2,2,k,j,i) = z*z*fmet + 1.0;
      Real det = adm::SpatialDet(adm_state.g_dd(m,0,0,k,j,i),
                                 adm_state.g_dd(m,0,1,k,j,i),
                                 adm_state.g_dd(m,0,2,k,j,i),
                                 adm_state.g_dd(m,1,1,k,j,i),
                                 adm_state.g_dd(m,1,2,k,j,i),
                                 adm_state.g_dd(m,2,2,k,j,i));
      adm_state.psi4(m,k,j,i) = pow(det, 1.0/3.0);
    }
    adm_state.vK_dd(m,0,0,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,2,2,k,j,i) = 0.0;
  });

  if (!(star_isotropic) || star_boost_mag <= 0.0) {
    return;
  }

  par_for("z4c_tov_ks_boosted_K", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - star_center_x1_l;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - star_center_x2_l;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - star_center_x3_l;

    Real xb = star_rot_00_l*x + star_rot_01_l*y + star_rot_02_l*z;
    Real yb = star_rot_10_l*x + star_rot_11_l*y + star_rot_12_l*z;
    Real zb = star_rot_20_l*x + star_rot_21_l*y + star_rot_22_l*z;
    Real xr = lorentz*xb;
    Real yr = yb;
    Real zr = zb;
    Real r = sqrt(SQR(xr) + SQR(yr) + SQR(zr));

    Real rho0, p0, mass0, alp0, psi0;
    SampleIsotropicTOV(eos, tov_star, r, rho0, p0, mass0, alp0, psi0);

    Real dx1 = (indcs.nx1 > 1) ? size.d_view(m).dx1 : 1.0;
    Real dx2 = (indcs.nx2 > 1) ? size.d_view(m).dx2 : dx1;
    Real dx3 = (indcs.nx3 > 1) ? size.d_view(m).dx3 : dx1;
    Real dr = 0.25*fmax(1.0e-6, fmin(dx1, fmin(dx2, dx3)));
    Real rp = r + dr;
    Real rm = fmax(0.0, r - dr);

    Real rho_p, p_p, mass_p, alp_p, psi_p;
    Real rho_m, p_m, mass_m, alp_m, psi_m;
    SampleIsotropicTOV(eos, tov_star, rp, rho_p, p_p, mass_p, alp_p, psi_p);
    SampleIsotropicTOV(eos, tov_star, rm, rho_m, p_m, mass_m, alp_m, psi_m);

    Real inv_dr = 1.0/fmax(rp - rm, 1.0e-12);
    Real da_dr = (SQR(alp_p) - SQR(alp_m))*inv_dr;
    Real dpsi_dr = (psi_p - psi_m)*inv_dr;

    Real dr_dxb = (r > 0.0) ? (SQR(lorentz)*xb/r) : 0.0;
    Real dr_dyb = (r > 0.0) ? (yb/r) : 0.0;
    Real dr_dzb = (r > 0.0) ? (zb/r) : 0.0;

    Real da_dxb = da_dr*dr_dxb;
    Real da_dyb = da_dr*dr_dyb;
    Real da_dzb = da_dr*dr_dzb;
    Real dpsi_dxb = dpsi_dr*dr_dxb;
    Real dpsi_dyb = dpsi_dr*dr_dyb;
    Real dpsi_dzb = dpsi_dr*dr_dzb;

    Real a_rest = SQR(alp0);
    Real A = SQR(lorentz)*(psi0 - SQR(star_boost_mag_l)*a_rest);
    Real Bcov = SQR(lorentz)*star_boost_mag_l*(a_rest - psi0);
    Real alpha_lab = std::sqrt(fmax(1.0e-16, a_rest*psi0/fmax(A, 1.0e-16)));

    Real dA_dxb = SQR(lorentz)*(dpsi_dxb - SQR(star_boost_mag_l)*da_dxb);
    Real dA_dyb = SQR(lorentz)*(dpsi_dyb - SQR(star_boost_mag_l)*da_dyb);
    Real dA_dzb = SQR(lorentz)*(dpsi_dzb - SQR(star_boost_mag_l)*da_dzb);
    Real dB_dxb = SQR(lorentz)*star_boost_mag_l*(da_dxb - dpsi_dxb);
    Real dB_dyb = SQR(lorentz)*star_boost_mag_l*(da_dyb - dpsi_dyb);
    Real dB_dzb = SQR(lorentz)*star_boost_mag_l*(da_dzb - dpsi_dzb);

    Real inv_A = 1.0/fmax(A, 1.0e-16);
    Real Kbb_xx = (star_boost_mag_l*dA_dxb + 2.0*dB_dxb - inv_A*dA_dxb*Bcov)/
                  (2.0*fmax(alpha_lab, 1.0e-16));
    Real Kbb_yy = (star_boost_mag_l + Bcov*inv_A)*dpsi_dxb/
                  (2.0*fmax(alpha_lab, 1.0e-16));
    Real Kbb_zz = Kbb_yy;
    Real Kbb_xy = (dB_dyb - inv_A*dA_dyb*Bcov)/(2.0*fmax(alpha_lab, 1.0e-16));
    Real Kbb_xz = (dB_dzb - inv_A*dA_dzb*Bcov)/(2.0*fmax(alpha_lab, 1.0e-16));

    Real K00 = Kbb_xx;
    Real K01 = Kbb_xy;
    Real K02 = Kbb_xz;
    Real K11 = Kbb_yy;
    Real K12 = 0.0;
    Real K22 = Kbb_zz;

    auto rotate_K = [&](int a, int b) {
      Real R[3][3] = {
        {star_rot_00_l, star_rot_01_l, star_rot_02_l},
        {star_rot_10_l, star_rot_11_l, star_rot_12_l},
        {star_rot_20_l, star_rot_21_l, star_rot_22_l}
      };
      Real Kab[3][3] = {
        {K00, K01, K02},
        {K01, K11, K12},
        {K02, K12, K22}
      };
      Real sum = 0.0;
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          sum += R[c][a]*R[d][b]*Kab[c][d];
        }
      }
      return sum;
    };

    adm_state.vK_dd(m,0,0,k,j,i) = rotate_K(0, 0);
    adm_state.vK_dd(m,0,1,k,j,i) = rotate_K(0, 1);
    adm_state.vK_dd(m,0,2,k,j,i) = rotate_K(0, 2);
    adm_state.vK_dd(m,1,1,k,j,i) = rotate_K(1, 1);
    adm_state.vK_dd(m,1,2,k,j,i) = rotate_K(1, 2);
    adm_state.vK_dd(m,2,2,k,j,i) = rotate_K(2, 2);
  });
}

template <int NGHOST>
void ConvertADMToResidualOnBackground(MeshBlockPack *pmbp, ParameterInput *pin) {
  auto *pz4c = pmbp->pz4c;
  pz4c->ADMToZ4c<NGHOST>(pmbp, pin);
  Kokkos::deep_copy(DevExeSpace(), pz4c->u_full, pz4c->u0);

  FillFlatADM(pmbp, pmbp->padm->adm);
  pz4c->ADMToZ4c<NGHOST>(pmbp, pin);
  Kokkos::deep_copy(DevExeSpace(), pz4c->u_bg, pz4c->u0);

  auto &indcs = pmbp->pmesh->mb_indcs;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;
  int nmb = pmbp->nmb_thispack;
  int nz4c = pz4c->nz4c;

  int ialpha = pz4c->I_Z4C_ALPHA;
  int ibetax = pz4c->I_Z4C_BETAX;
  int ibetaz = pz4c->I_Z4C_BETAZ;
  int ibx = pz4c->I_Z4C_BX;
  int ibz = pz4c->I_Z4C_BZ;
  auto &u0 = pz4c->u0;
  auto &u_full = pz4c->u_full;
  auto &u_bg = pz4c->u_bg;

  par_for("z4c_tov_ks_residual", DevExeSpace(), 0, nmb - 1, 0, nz4c - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    u0(m,n,k,j,i) = u_full(m,n,k,j,i) - u_bg(m,n,k,j,i);
    if (n == ialpha || (n >= ibetax && n <= ibetaz) || (n >= ibx && n <= ibz)) {
      u0(m,n,k,j,i) = 0.0;
    }
  });
}

void ZeroMagneticFields(MeshBlockPack *pmbp) {
  if (pmbp->pmhd == nullptr) {
    return;
  }
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->bcc0, 0.0);
}

template <class TOVEOS>
KOKKOS_INLINE_FUNCTION
Real MagneticPressureProfile(const TOVEOS &eos, const tov::TOVStar &tov_star,
                             Real pcut, Real magindex, Real x, Real y, Real z,
                             Real scx1, Real scx2, Real scx3, Real boost_mag,
                             bool isotropic, Real sr00, Real sr01, Real sr02,
                             Real sr10, Real sr11, Real sr12,
                             Real sr20, Real sr21, Real sr22) {
  x -= scx1;
  y -= scx2;
  z -= scx3;

  Real xb = sr00*x + sr01*y + sr02*z;
  Real yb = sr10*x + sr11*y + sr12*z;
  Real zb = sr20*x + sr21*y + sr22*z;
  Real lorentz = 1.0/sqrt(fmax(1.0e-16, 1.0 - SQR(boost_mag)));
  Real xr = lorentz*xb;
  Real yr = yb;
  Real zr = zb;
  Real r = sqrt(SQR(xr) + SQR(yr) + SQR(zr));

  Real rho, p;
  if (isotropic) {
    tov_star.GetPandRhoIso(eos, r, rho, p);
  } else {
    tov_star.GetPandRho(eos, r, rho, p);
  }

  Real rho_factor = fmax(1.0 - rho/tov_star.rhoc, 0.0);
  return fmax(p - pcut, 0.0)*pow(rho_factor, magindex);
}

template <class TOVEOS>
KOKKOS_INLINE_FUNCTION
Real MagneticVectorPotentialX(const TOVEOS &eos, const tov::TOVStar &tov_star,
                              Real pcut, Real magindex, Real x, Real y, Real z,
                              Real scx1, Real scx2, Real scx3, Real boost_mag,
                              bool isotropic, Real sr00, Real sr01, Real sr02,
                              Real sr10, Real sr11, Real sr12,
                              Real sr20, Real sr21, Real sr22) {
  Real profile = MagneticPressureProfile(eos, tov_star, pcut, magindex, x, y, z,
                                         scx1, scx2, scx3, boost_mag, isotropic,
                                         sr00, sr01, sr02, sr10, sr11, sr12,
                                         sr20, sr21, sr22);
  return -(y - scx2)*profile;
}

template <class TOVEOS>
KOKKOS_INLINE_FUNCTION
Real MagneticVectorPotentialY(const TOVEOS &eos, const tov::TOVStar &tov_star,
                              Real pcut, Real magindex, Real x, Real y, Real z,
                              Real scx1, Real scx2, Real scx3, Real boost_mag,
                              bool isotropic, Real sr00, Real sr01, Real sr02,
                              Real sr10, Real sr11, Real sr12,
                              Real sr20, Real sr21, Real sr22) {
  Real profile = MagneticPressureProfile(eos, tov_star, pcut, magindex, x, y, z,
                                         scx1, scx2, scx3, boost_mag, isotropic,
                                         sr00, sr01, sr02, sr10, sr11, sr12,
                                         sr20, sr21, sr22);
  return (x - scx1)*profile;
}

template <class TOVEOS>
void InitializeDipoleMagneticField(ParameterInput *pin, Mesh *pmy_mesh, TOVEOS &eos,
                                   const tov::TOVStar &tov_star) {
  MeshBlockPack *pmbp = pmy_mesh->pmb_pack;
  ZeroMagneticFields(pmbp);

  Real b_norm = pin->GetOrAddReal("problem", "b_norm", 0.0);
  if (b_norm == 0.0) {
    return;
  }
  Real pcut = pin->GetOrAddReal("problem", "pcut", 1.0e-6);
  Real magindex = pin->GetOrAddReal("problem", "magindex", 2.0);
  if (pin->GetOrAddBoolean("problem", "use_pcut_rel", false)) {
    Real pmax = eos.template GetPFromRho<tov::LocationTag::Device>(tov_star.rhoc);
    pcut *= pmax;
  }

  auto &indcs = pmy_mesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb = pmbp->nmb_thispack;

  DvceArray4D<Real> a1;
  DvceArray4D<Real> a2;
  DvceArray4D<Real> a3;
  Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;
  auto &eos_ = eos;
  auto &tov_star_ = tov_star;
  const Real star_center_x1_l = star_center_x1;
  const Real star_center_x2_l = star_center_x2;
  const Real star_center_x3_l = star_center_x3;
  const Real star_boost_mag_l = star_boost_mag;
  const bool star_isotropic_l = star_isotropic;
  const Real star_rot_00_l = star_rot_00;
  const Real star_rot_01_l = star_rot_01;
  const Real star_rot_02_l = star_rot_02;
  const Real star_rot_10_l = star_rot_10;
  const Real star_rot_11_l = star_rot_11;
  const Real star_rot_12_l = star_rot_12;
  const Real star_rot_20_l = star_rot_20;
  const Real star_rot_21_l = star_rot_21;
  const Real star_rot_22_l = star_rot_22;
  par_for("z4c_tov_ks_potential", DevExeSpace(), 0, nmb - 1, ks, ke + 1,
          js, je + 1, is, ie + 1,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real x1f = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real x2f = LeftEdgeX(j - js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3f = LeftEdgeX(k - ks, indcs.nx3, x3min, x3max);
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;

    a1(m,k,j,i) = MagneticVectorPotentialX(eos_, tov_star_, pcut, magindex,
                                           x1v, x2f, x3f, star_center_x1_l,
                                           star_center_x2_l, star_center_x3_l,
                                           star_boost_mag_l, star_isotropic_l,
                                           star_rot_00_l, star_rot_01_l,
                                           star_rot_02_l, star_rot_10_l,
                                           star_rot_11_l, star_rot_12_l,
                                           star_rot_20_l, star_rot_21_l,
                                           star_rot_22_l);
    a2(m,k,j,i) = MagneticVectorPotentialY(eos_, tov_star_, pcut, magindex,
                                           x1f, x2v, x3f, star_center_x1_l,
                                           star_center_x2_l, star_center_x3_l,
                                           star_boost_mag_l, star_isotropic_l,
                                           star_rot_00_l, star_rot_01_l,
                                           star_rot_02_l, star_rot_10_l,
                                           star_rot_11_l, star_rot_12_l,
                                           star_rot_20_l, star_rot_21_l,
                                           star_rot_22_l);
    a3(m,k,j,i) = 0.0;

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
      a1(m,k,j,i) = 0.5*(
          MagneticVectorPotentialX(eos_, tov_star_, pcut, magindex, xl, x2f, x3f,
                                   star_center_x1_l, star_center_x2_l,
                                   star_center_x3_l, star_boost_mag_l,
                                   star_isotropic_l, star_rot_00_l, star_rot_01_l,
                                   star_rot_02_l, star_rot_10_l, star_rot_11_l,
                                   star_rot_12_l, star_rot_20_l, star_rot_21_l,
                                   star_rot_22_l) +
          MagneticVectorPotentialX(eos_, tov_star_, pcut, magindex, xr, x2f, x3f,
                                   star_center_x1_l, star_center_x2_l,
                                   star_center_x3_l, star_boost_mag_l,
                                   star_isotropic_l, star_rot_00_l, star_rot_01_l,
                                   star_rot_02_l, star_rot_10_l, star_rot_11_l,
                                   star_rot_12_l, star_rot_20_l, star_rot_21_l,
                                   star_rot_22_l));
    }

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
      a2(m,k,j,i) = 0.5*(
          MagneticVectorPotentialY(eos_, tov_star_, pcut, magindex, x1f, xl, x3f,
                                   star_center_x1_l, star_center_x2_l,
                                   star_center_x3_l, star_boost_mag_l,
                                   star_isotropic_l, star_rot_00_l, star_rot_01_l,
                                   star_rot_02_l, star_rot_10_l, star_rot_11_l,
                                   star_rot_12_l, star_rot_20_l, star_rot_21_l,
                                   star_rot_22_l) +
          MagneticVectorPotentialY(eos_, tov_star_, pcut, magindex, x1f, xr, x3f,
                                   star_center_x1_l, star_center_x2_l,
                                   star_center_x3_l, star_boost_mag_l,
                                   star_isotropic_l, star_rot_00_l, star_rot_01_l,
                                   star_rot_02_l, star_rot_10_l, star_rot_11_l,
                                   star_rot_12_l, star_rot_20_l, star_rot_21_l,
                                   star_rot_22_l));
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("z4c_tov_ks_Bfc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = b_norm*((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                       (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = b_norm*((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                       (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = b_norm*((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                       (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

    if (i == ie) {
      b0.x1f(m,k,j,i+1) = b_norm*((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                           (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j == je) {
      b0.x2f(m,k,j+1,i) = b_norm*((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                           (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k == ke) {
      b0.x3f(m,k+1,j,i) = b_norm*((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                           (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

  auto &bcc = pmbp->pmhd->bcc0;
  par_for("z4c_tov_ks_Bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    bcc(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    bcc(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });
}

template <class TOVEOS>
void SetupTOVKerrSchild(ParameterInput *pin, Mesh *pmy_mesh) {
  MeshBlockPack *pmbp = pmy_mesh->pmb_pack;
  TOVEOS eos{pin};
  auto tov_star = tov::TOVStar::ConstructTOV(pin, eos);
  ConfigureCircularGeodesicOrbit(tov_star);

  FillTOVPrimitivesAndADM(pin, pmy_mesh, eos, tov_star);

  auto &indcs = pmy_mesh->mb_indcs;
  switch (indcs.ng) {
    case 2:
      ConvertADMToResidualOnBackground<2>(pmbp, pin);
      break;
    case 3:
      ConvertADMToResidualOnBackground<3>(pmbp, pin);
      break;
    case 4:
      ConvertADMToResidualOnBackground<4>(pmbp, pin);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unsupported nghost for z4c_tov_ks" << std::endl;
      std::exit(EXIT_FAILURE);
  }

  auto *pz4c = pmbp->pz4c;
  pz4c->SetADMBackground = &SetADMBackgroundKerrSchild;
  pz4c->UpdateBackgroundState(pmy_mesh->time);
  pz4c->ReconstructFullState();
  pz4c->EnforceAlgConstrOn(pz4c->full);
  pz4c->RecastResidualState();
  pz4c->PrescribeGaugeResidual();
  ApplyInnerExcision(pmy_mesh, 0.0, false);
  pz4c->Z4cToADM(pmbp);

  InitializeDipoleMagneticField(pin, pmy_mesh, eos, tov_star);

  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  pmbp->pdyngr->PrimToConInit(0, n1 - 1, 0, n2 - 1, 0, n3 - 1);
  ApplyInnerExcision(pmy_mesh, 0.0);

  switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
  }
}

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic || pmbp->pdyngr == nullptr ||
      pmbp->padm == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires dynamical GR, ADM, DynGRMHD, and Z4c"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pz4c->use_analytic_background) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires <z4c>/use_analytic_background = true"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  user_hist_func = &TOVKerrSchildHistory;

  bh_mass = pin->GetOrAddReal("problem", "bh_mass", 1.0);
  use_minkowski_background = fabs(bh_mass) <= 1.0e-12;
  const bool coord_minkowski = pin->GetOrAddBoolean("coord", "minkowski", false);
  if (!use_minkowski_background && fabs(bh_mass - 1.0) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks supports bh_mass = 0 for a Minkowski "
              << "background or bh_mass = 1 for the unit-mass Kerr-Schild helper."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (use_minkowski_background && !coord_minkowski) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks with bh_mass = 0 requires "
              << "<coord>/minkowski = true so the source-term metric is flat."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  bh_spin = pin->GetOrAddReal("problem", "bh_spin", 0.0);
  Real coord_spin = pin->GetOrAddReal("coord", "a", 0.0);
  if (use_minkowski_background && fabs(bh_spin) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks with bh_mass = 0 requires bh_spin = 0."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!use_minkowski_background && fabs(coord_spin - bh_spin) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires <coord>/a to match "
              << "<problem>/bh_spin so the coordinate source terms and analytic "
              << "background use the same Kerr-Schild spin." << std::endl;
    exit(EXIT_FAILURE);
  }
  bh_center_x1 = pin->GetOrAddReal("problem", "bh_center_x1", 0.0);
  bh_center_x2 = pin->GetOrAddReal("problem", "bh_center_x2", 0.0);
  bh_center_x3 = pin->GetOrAddReal("problem", "bh_center_x3", 0.0);
  if (fabs(bh_center_x1) > 1.0e-12 || fabs(bh_center_x2) > 1.0e-12 ||
      fabs(bh_center_x3) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks currently requires the BH to remain at the "
              << "grid origin so the coordinate metric and analytic background agree."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  star_center_x1 = pin->GetOrAddReal("problem", "star_center_x1", 8.0);
  star_center_x2 = pin->GetOrAddReal("problem", "star_center_x2", 0.0);
  star_center_x3 = pin->GetOrAddReal("problem", "star_center_x3", 0.0);
  star_boost_x = pin->GetOrAddReal("problem", "star_boost_x", 0.0);
  star_boost_y = pin->GetOrAddReal("problem", "star_boost_y", 0.0);
  star_boost_z = pin->GetOrAddReal("problem", "star_boost_z", 0.0);
  std::string star_orbit = pin->GetOrAddString("problem", "star_orbit", "none");
  star_orbit_circular_geodesic = (star_orbit == "circular_geodesic");
  if (!(star_orbit == "none" || star_orbit_circular_geodesic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unsupported <problem>/star_orbit = " << star_orbit
              << ". Supported values are none and circular_geodesic." << std::endl;
    exit(EXIT_FAILURE);
  }
  star_orbit_radius = pin->GetOrAddReal("problem", "star_orbit_radius", -1.0);
  star_orbit_radius_factor =
      pin->GetOrAddReal("problem", "star_orbit_radius_factor", 2.0);
  star_orbit_phase = pin->GetOrAddReal("problem", "star_orbit_phase", 0.0);
  Real star_boost_sq = SQR(star_boost_x) + SQR(star_boost_y) + SQR(star_boost_z);
  if (star_boost_sq >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires star_boost_x^2 + star_boost_y^2 + "
              << "star_boost_z^2 < 1." << std::endl;
    exit(EXIT_FAILURE);
  }
  star_isotropic = pin->GetOrAddBoolean("problem", "isotropic", true);
  SetStarBoostRotation();
  if (!star_isotropic && star_boost_mag > 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks currently supports boosted stars only for "
              << "isotropic TOV coordinates." << std::endl;
    exit(EXIT_FAILURE);
  }
  bh_horizon_radius = use_minkowski_background ?
      0.0 : 1.0 + sqrt(fmax(0.0, 1.0 - SQR(bh_spin)));
  excision_damp_rate = pin->GetOrAddReal("problem", "excision_damp_rate", 50.0);
  excision_project_state = pin->GetOrAddBoolean("problem", "excision_project_state", true);
  excision_freeze_radius =
      pin->GetOrAddReal("problem", "excision_freeze_radius",
                        use_minkowski_background ? 0.0 : 0.85*bh_horizon_radius);
  excision_ramp_radius =
      pin->GetOrAddReal("problem", "excision_ramp_radius",
                        use_minkowski_background ? 0.0 : 0.95*bh_horizon_radius);
  if (excision_freeze_radius < 0.0 || excision_ramp_radius < excision_freeze_radius ||
      excision_ramp_radius > bh_horizon_radius) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires 0 <= excision_freeze_radius <= "
              << "excision_ramp_radius <= Kerr-Schild horizon radius." << std::endl;
    exit(EXIT_FAILURE);
  }
  Real dfloor = pin->GetOrAddReal("mhd", "dfloor", 1.0e-16);
  Real pfloor = pin->GetOrAddReal("mhd", "pfloor", 1.0e-22);
  Real gamma = pin->GetOrAddReal("mhd", "gamma", 5.0/3.0);
  excision_atmo_density = pin->GetOrAddReal("problem", "excision_atmo_density", dfloor);
  excision_atmo_energy =
      pin->GetOrAddReal("problem", "excision_atmo_energy", pfloor/fmax(gamma - 1.0, 1.0e-12));
  amr_rho_slope_threshold =
      pin->GetOrAddReal("problem", "amr_rho_slope_threshold", 0.5);
  amr_rho_min = pin->GetOrAddReal("problem", "amr_rho_min", 100.0*dfloor);
  amr_bh_exclusion_radius =
      pin->GetOrAddReal("problem", "amr_bh_exclusion_radius",
                        use_minkowski_background ? 0.0 : 4.0*bh_horizon_radius);
  amr_star_refine = pin->GetOrAddBoolean("problem", "amr_star_refine", false);
  amr_star_refine_radius = pin->GetOrAddReal("problem", "amr_star_refine_radius", 0.0);
  amr_star_refine_level = pin->GetOrAddInteger("problem", "amr_star_refine_level", -1);
  user_srcs = true;
  user_srcs_func = &ApplyInnerExcision;
  user_ref_func = &RefinementCondition;
  pmbp->pz4c->SetADMBackground = &SetADMBackgroundKerrSchild;

  if (restart) {
    pmbp->pz4c->UpdateBackgroundState(pmy_mesh_->time);
    pmbp->pz4c->ReconstructFullState();
    pmbp->pz4c->EnforceAlgConstrOn(pmbp->pz4c->full);
    pmbp->pz4c->RecastResidualState();
    pmbp->pz4c->PrescribeGaugeResidual();
    ApplyInnerExcision(pmy_mesh_, 0.0, false);
    pmbp->pz4c->Z4cToADM(pmbp);
    ApplyInnerExcision(pmy_mesh_, 0.0);
    return;
  }

  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupTOVKerrSchild<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupTOVKerrSchild<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupTOVKerrSchild<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupTOVKerrSchild<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unsupported EOS policy for z4c_tov_ks" << std::endl;
    exit(EXIT_FAILURE);
  }
}
