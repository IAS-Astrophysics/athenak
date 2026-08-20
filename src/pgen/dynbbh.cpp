//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dynbbh.cpp
//! \brief Problem generator for superimposed Kerr-Schild black holes

#include <math.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "outputs/outputs.hpp"
#include "utils/flux_generalized.hpp"

#define D2(comp, h) ((met_p1.g).comp - (met_m1.g).comp) / (2*h)

namespace {

enum {
  TT, XX, YY, ZZ, NDIM
};

constexpr Real kDefaultMetricFdStep = 5.0e-5;

KOKKOS_INLINE_FUNCTION Real metric_sqrt(const Real x) { return sqrt(x); }
KOKKOS_INLINE_FUNCTION Real value_of(const Real x) { return x; }

struct dual1_real {
  Real val;
  Real deriv;

  KOKKOS_INLINE_FUNCTION dual1_real() : val(0.0), deriv(0.0) {}
  KOKKOS_INLINE_FUNCTION dual1_real(const Real value) : val(value), deriv(0.0) {}
  KOKKOS_INLINE_FUNCTION dual1_real(const Real value, const Real derivative)
      : val(value), deriv(derivative) {}
};

KOKKOS_INLINE_FUNCTION dual1_real operator+(const dual1_real &a,
                                             const dual1_real &b) {
  return dual1_real(a.val + b.val, a.deriv + b.deriv);
}
KOKKOS_INLINE_FUNCTION dual1_real operator-(const dual1_real &a,
                                             const dual1_real &b) {
  return dual1_real(a.val - b.val, a.deriv - b.deriv);
}
KOKKOS_INLINE_FUNCTION dual1_real operator-(const dual1_real &a) {
  return dual1_real(-a.val, -a.deriv);
}
KOKKOS_INLINE_FUNCTION dual1_real operator*(const dual1_real &a,
                                             const dual1_real &b) {
  return dual1_real(a.val*b.val, a.deriv*b.val + a.val*b.deriv);
}
KOKKOS_INLINE_FUNCTION dual1_real operator/(const dual1_real &a,
                                             const dual1_real &b) {
  const Real inv = 1.0/b.val;
  return dual1_real(a.val*inv,
                    (a.deriv*b.val - a.val*b.deriv)*inv*inv);
}
KOKKOS_INLINE_FUNCTION dual1_real metric_sqrt(const dual1_real &x) {
  const Real root = sqrt(x.val);
  return dual1_real(root, 0.5*x.deriv/root);
}
KOKKOS_INLINE_FUNCTION Real value_of(const dual1_real &x) { return x.val; }

template <typename T>
KOKKOS_INLINE_FUNCTION T metric_norm3(const T x, const T y, const T z) {
  const T radius2 = x*x + y*y + z*z;
  if (value_of(radius2) <= 0.0) return T(0.0);
  return metric_sqrt(radius2);
}

enum {
  X1, Y1, Z1, X2, Y2, Z2,
  VX1, VY1, VZ1, VX2, VY2, VZ2,
  AX1, AY1, AZ1, AX2, AY2, AZ2,
  M1T, M2T, NTRAJ
};

struct dd_sym {
  Real tt;
  Real tx;
  Real ty;
  Real tz;
  Real xx;
  Real xy;
  Real xz;
  Real yy;
  Real yz;
  Real zz;
};

struct four_metric {
  struct dd_sym g;
  struct dd_sym g_t;
  struct dd_sym g_x;
  struct dd_sym g_y;
  struct dd_sym g_z;
};

struct three_metric {
  Real gxx;
  Real gxy;
  Real gxz;
  Real gyy;
  Real gyz;
  Real gzz;
  Real alpha;
  Real betax;
  Real betay;
  Real betaz;
  Real kxx;
  Real kxy;
  Real kxz;
  Real kyy;
  Real kyz;
  Real kzz;
};

enum class MetricDerivativeMethod {
  finite_difference,
  ad
};

struct bbh_pgen {
  Real sep;
  Real om;
  Real q;
  Real a1, a2;
  Real th_a1, th_a2;
  Real ph_a1, ph_a2;
  Real spin_ramp_timescale;
  Real spin_ramp_start_time;
  Real dfloor;
  Real pfloor;
  Real gamma_adi;
  Real a1_buffer, a2_buffer;
  Real cutoff_floor;
  Real metric_fd_step = kDefaultMetricFdStep;
  Real alpha_thr;
  Real radius_thr;
  MetricDerivativeMethod metric_derivative_method =
      MetricDerivativeMethod::finite_difference;
  bool use_traj_table = false;
  bool spin_ramp = false;
};

struct bbh_pgen bbh;

struct bbh_traj_table {
  std::vector<Real> t;
  std::vector<Real> x1, y1, z1, x2, y2, z2;
  std::vector<Real> vx1, vy1, vz1, vx2, vy2, vz2;
  std::vector<Real> chix1, chiy1, chiz1, chix2, chiy2, chiz2;
  std::vector<Real> m1, m2;
  std::size_t active_segment = 0;
};

bbh_traj_table bbh_table;

/* Declare functions */
void find_traj_t(Real tt, Real traj_array[NTRAJ]);
void find_traj_t_with_deriv(Real tt, Real traj_array[NTRAJ],
                            Real dtraj_array[NTRAJ]);
void LoadTrajectoryTable(const std::string &fname);

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const Real hm, const Real hp, const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met, struct three_metric &gam);
KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
void get_metric_and_derivatives(const Real t, const Real x, const Real y,
                                const Real z, struct four_metric &met,
                                const Real bbh_traj_loc[NTRAJ],
                                const Real dbbh_traj_loc[NTRAJ],
                                const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                   Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen& bbh_);
void SetADMVariablesToBBH(MeshBlockPack *pmbp);
void RefineAlphaMin(MeshBlockPack* pmbp);
void RefineTracker(MeshBlockPack* pmbp);
void DynBBHFluxHistory(HistoryData *pdata, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ShockTube_()
//! \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "BBH problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string amr_cond = pin->GetOrAddString("problem", "amr_condition", "track");
  if (amr_cond == "alpha_min") {
    user_ref_func = RefineAlphaMin;
  } else {
    user_ref_func = RefineTracker;
  }
  if (pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh requires an <adm> object" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pmbp->padm->SetADMVariables = &SetADMVariablesToBBH;

  bbh.sep = pin->GetOrAddReal("problem", "sep", 20.0);
  bbh.q = pin->GetOrAddReal("problem", "q", 1.0);
  bbh.a1 = pin->GetOrAddReal("problem", "a1", 0.0);
  bbh.a2 = pin->GetOrAddReal("problem", "a2", 0.0);
  const Real degrees = std::acos(-1.0)/180.0;
  bbh.th_a1 = pin->GetOrAddReal("problem", "th_a1", 0.0)*degrees;
  bbh.th_a2 = pin->GetOrAddReal("problem", "th_a2", 0.0)*degrees;
  bbh.ph_a1 = pin->GetOrAddReal("problem", "ph_a1", 0.0)*degrees;
  bbh.ph_a2 = pin->GetOrAddReal("problem", "ph_a2", 0.0)*degrees;
  bbh.spin_ramp = pin->GetOrAddBoolean("problem", "spin_ramp", false);
  bbh.spin_ramp_timescale = pin->GetOrAddReal(
      "problem", "spin_ramp_timescale", 50.0);
  bbh.spin_ramp_start_time = pin->GetOrAddReal(
      "problem", "spin_ramp_start_time", pmbp->pmesh->time);
  if (!(bbh.sep > 0.0) || !(bbh.q > 0.0) || !std::isfinite(bbh.sep) ||
      !std::isfinite(bbh.q) || !std::isfinite(bbh.a1) ||
      !std::isfinite(bbh.a2) || !std::isfinite(bbh.th_a1) ||
      !std::isfinite(bbh.th_a2) || !std::isfinite(bbh.ph_a1) ||
      !std::isfinite(bbh.ph_a2) || bbh.a1 < 0.0 || bbh.a2 < 0.0 ||
      bbh.a1 > 1.0 || bbh.a2 > 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem/sep and q must be positive and 0 <= a1,a2 <= 1"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bbh.spin_ramp && (!(bbh.spin_ramp_timescale > 0.0) ||
                        !std::isfinite(bbh.spin_ramp_start_time))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "spin_ramp requires a positive timescale and finite start time"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bbh.om = std::pow(bbh.sep, -1.5);
  bbh.dfloor = pin->GetOrAddReal("problem", "dfloor", (FLT_MIN));
  bbh.pfloor = pin->GetOrAddReal("problem", "pfloor", (FLT_MIN));
  if (pin->DoesParameterExist("problem", "adjust_mass1") ||
      pin->DoesParameterExist("problem", "adjust_mass2")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem/adjust_mass1 and adjust_mass2 are no longer supported; "
              << "trajectory masses are the physical Kerr-Schild masses"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bbh.a1_buffer = pin->GetOrAddReal("problem", "a1_buffer", 0.0);
  bbh.a2_buffer = pin->GetOrAddReal("problem", "a2_buffer", 0.0);
  bbh.cutoff_floor = pin->GetOrAddReal("problem", "cutoff_floor", 1e-10);
  bbh.metric_fd_step = pin->GetOrAddReal(
      "problem", "metric_fd_step", kDefaultMetricFdStep);
  if (!(bbh.metric_fd_step > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "problem/metric_fd_step must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string metric_derivative = pin->GetOrAddString(
      "problem", "metric_derivative", "finite_difference");
  if (metric_derivative == "finite_difference" || metric_derivative == "fd") {
    bbh.metric_derivative_method = MetricDerivativeMethod::finite_difference;
  } else if (metric_derivative == "ad") {
    bbh.metric_derivative_method = MetricDerivativeMethod::ad;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown problem/metric_derivative='"
              << metric_derivative
              << "'. Use 'finite_difference' or 'ad'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bbh.alpha_thr = pin->GetOrAddReal("problem", "alpha_thr", 0.6);
  bbh.radius_thr = pin->GetOrAddReal("problem", "radius_thr", 6.0);
  bbh.use_traj_table = pin->GetOrAddBoolean(
      "problem", "use_traj_table", false);
  const std::string traj_file = pin->GetOrAddString("problem", "traj_file", "");
  const Real analytic_vmax = bbh.om*bbh.sep*
      std::max(bbh.q, 1.0)/(1.0 + bbh.q);
  if (!bbh.use_traj_table && !(analytic_vmax < 1.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "analytic binary orbit has superluminal velocity"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bbh.use_traj_table && bbh.spin_ramp) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "spin_ramp applies only to analytic orbits; put evolving chi "
              << "directly in a trajectory table" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bbh.use_traj_table) {
    if (traj_file.empty()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "use_traj_table=true requires problem/traj_file"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    LoadTrajectoryTable(traj_file);
    const Real required_start = pmbp->pmesh->time;
    const Real required_end = std::max(
        required_start, pin->GetOrAddReal("time", "tlim", required_start));
    const Real scale = std::max({1.0, std::abs(required_start),
                                 std::abs(required_end)});
    const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon()*scale;
    if (bbh_table.t.front() > required_start + tolerance ||
        bbh_table.t.back() < required_end - tolerance) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "trajectory table [" << bbh_table.t.front()
                << ", " << bbh_table.t.back() << "] does not cover required time ["
                << required_start << ", " << required_end << "]" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (user_hist) {
    const int flux_ntheta = pin->GetOrAddInteger("problem", "flux_ntheta", 32);
    const int flux_nphi = pin->GetOrAddInteger("problem", "flux_nphi", 64);
    const int flux_interp_order = pin->GetOrAddInteger(
        "problem", "flux_interp_order", 1);
    const Real radial_inner = pin->GetOrAddReal(
        "problem", "flux_rsurf_inner", -1.0);
    const Real radial_outer = pin->GetOrAddReal(
        "problem", "flux_rsurf_outer", -1.0);
    const Real radial_step = pin->GetOrAddReal("problem", "flux_dr_surf", 1.0);
    const bool horizon1 = pin->GetOrAddBoolean(
        "problem", "flux_horizon1", false);
    const bool horizon2 = pin->GetOrAddBoolean(
        "problem", "flux_horizon2", false);
    const Real horizon_radius1 = pin->GetOrAddReal(
        "problem", "flux_radius1", -1.0);
    const Real horizon_radius2 = pin->GetOrAddReal(
        "problem", "flux_radius2", -1.0);
    if (flux_ntheta < 3 || flux_nphi < 3 || flux_interp_order < 1 ||
        ((radial_outer >= radial_inner && radial_inner > 0.0) &&
         !(radial_step > 0.0)) || (horizon1 && !(horizon_radius1 > 0.0)) ||
        (horizon2 && !(horizon_radius2 > 0.0))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "invalid dynbbh flux-surface configuration"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    surface_flux_grids.clear();
    const Real origin[3] = {0.0, 0.0, 0.0};
    if (radial_inner > 0.0 && radial_outer >= radial_inner) {
      for (Real radius = radial_inner;
           radius <= radial_outer + 16.0*std::numeric_limits<Real>::epsilon()*
                                      std::max(1.0, radial_outer);
           radius += radial_step) {
        std::ostringstream label;
        label << "r" << radius;
        surface_flux_grids.emplace_back(std::make_unique<SphericalSurfaceGrid>(
            pmbp, flux_ntheta, flux_nphi,
            [radius](Real, Real) { return radius; }, label.str(), origin,
            flux_interp_order));
        if (2*surface_flux_grids.size() > NHISTORY_VARIABLES) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "too many dynbbh flux surfaces" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
    Real trajectory[NTRAJ];
    find_traj_t(pmbp->pmesh->time, trajectory);
    if (horizon1) {
      const Real center[3] = {trajectory[X1], trajectory[Y1], trajectory[Z1]};
      surface_flux_grids.emplace_back(std::make_unique<SphericalSurfaceGrid>(
          pmbp, flux_ntheta, flux_nphi,
          [horizon_radius1](Real, Real) { return horizon_radius1; }, "h1",
          center, flux_interp_order));
    }
    if (horizon2) {
      const Real center[3] = {trajectory[X2], trajectory[Y2], trajectory[Z2]};
      surface_flux_grids.emplace_back(std::make_unique<SphericalSurfaceGrid>(
          pmbp, flux_ntheta, flux_nphi,
          [horizon_radius2](Real, Real) { return horizon_radius2; }, "h2",
          center, flux_interp_order));
    }
    if (2*surface_flux_grids.size() > NHISTORY_VARIABLES) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "too many dynbbh flux surfaces" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    user_hist_func = DynBBHFluxHistory;
  }

  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  auto &bbh_ = bbh;

  if (pmbp->phydro != nullptr) {
    auto &eos = pmbp->phydro->peos->eos_data;
    auto &w0 = pmbp->phydro->w0;
    auto &nscal = pmbp->phydro->nscalars;
    par_for("pgen_hydro", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = bbh_.dfloor;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IPR,k,j,i) = bbh_.pfloor; //bbh.fluid.pfloor;
      for (int r=0; r<nscal; ++r) {
        w0(m,IYF+r,k,j,i) = 0.0;
      }
    });

    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    if (pmbp->padm == nullptr) {
      pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
    }
  } // End initialization of Hydro variables

  // Initialize MHD variables -------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &eos = pmbp->pmhd->peos->eos_data;
    auto &w0 = pmbp->pmhd->w0;
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    auto &nscal = pmbp->pmhd->nscalars;
    par_for("pgen_shock1", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = bbh_.dfloor;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IPR,k,j,i) = bbh_.pfloor; //bbh.fluid.pfloor;
      for (int r=0; r<nscal; ++r) {
        w0(m,IYF+r,k,j,i) = 0.0;
      }
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = 0.0;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
    // Convert primitives to conserved
    auto &u0 = pmbp->pmhd->u0;
    if (!pmbp->pcoord->is_dynamical_relativistic) {
      pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
    }
  } // End initialization of MHD variables

  // Initialize ADM variables -------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    // If we're using the ADM variables, then we've got dynamic GR enabled.
    // Because we need the metric, we can't initialize the conserved variables
    // until we've filled out the ADM variables.
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }
  return;
}

namespace {

void SetADMVariablesToBBH(MeshBlockPack *pmbp) {
  const Real tt = pmbp->pmesh->time;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  Real bbh_traj_p1[NTRAJ];
  Real bbh_traj_0[NTRAJ];
  Real bbh_traj_m1[NTRAJ];
  Real dbbh_traj_0[NTRAJ];
  auto& bbh_ = bbh;

  /* Load trajectories */

  /* Whether we load traj from a table or we compute analytical trajectories */
  find_traj_t_with_deriv(tt, bbh_traj_0, dbbh_traj_0);
  Real hm = bbh_.metric_fd_step;
  Real hp = bbh_.metric_fd_step;
  if (bbh_.use_traj_table) {
    hm = std::min(hm, std::max(tt - bbh_table.t.front(), 0.0));
    hp = std::min(hp, std::max(bbh_table.t.back() - tt, 0.0));
  }
  if (hp > 0.0) {
    find_traj_t(tt + hp, bbh_traj_p1);
  } else {
    for (int n = 0; n < NTRAJ; ++n) bbh_traj_p1[n] = bbh_traj_0[n];
  }
  if (hm > 0.0) {
    find_traj_t(tt - hm, bbh_traj_m1);
  } else {
    for (int n = 0; n < NTRAJ; ++n) bbh_traj_m1[n] = bbh_traj_0[n];
  }


  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    struct four_metric met4;
    struct three_metric met3;
    if (bbh_.metric_derivative_method == MetricDerivativeMethod::ad) {
      get_metric_and_derivatives(tt, x1v, x2v, x3v, met4, bbh_traj_0,
                                 dbbh_traj_0, bbh_);
    } else {
      numerical_4metric(tt, x1v, x2v, x3v, met4, bbh_traj_m1, bbh_traj_0,
                        bbh_traj_p1, hm, hp, bbh_);
    }

    /* Transform 4D metric to 3+1 variables*/
    four_metric_to_three_metric(met4, met3);

    /* Load (Cartesian) components of the metric and curvature */
    // g_ab
    adm.g_dd(m,0,0,k,j,i) = met3.gxx;
    adm.g_dd(m,0,1,k,j,i) = met3.gxy;
    adm.g_dd(m,0,2,k,j,i) = met3.gxz;
    adm.g_dd(m,1,1,k,j,i) = met3.gyy;
    adm.g_dd(m,1,2,k,j,i) = met3.gyz;
    adm.g_dd(m,2,2,k,j,i) = met3.gzz;

    adm.vK_dd(m,0,0,k,j,i) = met3.kxx;
    adm.vK_dd(m,0,1,k,j,i) = met3.kxy;
    adm.vK_dd(m,0,2,k,j,i) = met3.kxz;
    adm.vK_dd(m,1,1,k,j,i) = met3.kyy;
    adm.vK_dd(m,1,2,k,j,i) = met3.kyz;
    adm.vK_dd(m,2,2,k,j,i) = met3.kzz;

    adm.alpha(m,k,j,i) = met3.alpha;
    adm.beta_u(m,0,k,j,i) = met3.betax;
    adm.beta_u(m,1,k,j,i) = met3.betay;
    adm.beta_u(m,2,k,j,i) = met3.betaz;
  });
  return;
}

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const Real hm, const Real hp, const bbh_pgen& bbh_) {
  struct four_metric met_m1;
  struct four_metric met_p1;
  const Real step = bbh_.metric_fd_step;

  // Time
  get_metric(t, x, y, z, outmet, nz_0, bbh_);
  if (hm > 0.0) get_metric(t-hm, x, y, z, met_m1, nz_m1, bbh_);
  if (hp > 0.0) get_metric(t+hp, x, y, z, met_p1, nz_p1, bbh_);
#define DT_METRIC(comp) \
  outmet.g_t.comp = (hm > 0.0 && hp > 0.0) ? \
      (met_p1.g.comp - met_m1.g.comp)/(hm + hp) : \
      ((hp > 0.0) ? (met_p1.g.comp - outmet.g.comp)/hp : \
                    (outmet.g.comp - met_m1.g.comp)/hm)
  DT_METRIC(tt);
  DT_METRIC(tx);
  DT_METRIC(ty);
  DT_METRIC(tz);
  DT_METRIC(xx);
  DT_METRIC(xy);
  DT_METRIC(xz);
  DT_METRIC(yy);
  DT_METRIC(yz);
  DT_METRIC(zz);
#undef DT_METRIC

  // X
  get_metric(t, x-step, y, z, met_m1, nz_0, bbh_);
  get_metric(t, x+step, y, z, met_p1, nz_0, bbh_);

  outmet.g_x.tt = D2(tt, step);
  outmet.g_x.tx = D2(tx, step);
  outmet.g_x.ty = D2(ty, step);
  outmet.g_x.tz = D2(tz, step);
  outmet.g_x.xx = D2(xx, step);
  outmet.g_x.xy = D2(xy, step);
  outmet.g_x.xz = D2(xz, step);
  outmet.g_x.yy = D2(yy, step);
  outmet.g_x.yz = D2(yz, step);
  outmet.g_x.zz = D2(zz, step);

  // Y
  get_metric(t, x, y-step, z, met_m1, nz_0, bbh_);
  get_metric(t, x, y+step, z, met_p1, nz_0, bbh_);

  outmet.g_y.tt = D2(tt, step);
  outmet.g_y.tx = D2(tx, step);
  outmet.g_y.ty = D2(ty, step);
  outmet.g_y.tz = D2(tz, step);
  outmet.g_y.xx = D2(xx, step);
  outmet.g_y.xy = D2(xy, step);
  outmet.g_y.xz = D2(xz, step);
  outmet.g_y.yy = D2(yy, step);
  outmet.g_y.yz = D2(yz, step);
  outmet.g_y.zz = D2(zz, step);

  // Z
  get_metric(t, x, y, z-step, met_m1, nz_0, bbh_);
  get_metric(t, x, y, z+step, met_p1, nz_0, bbh_);

  outmet.g_z.tt = D2(tt, step);
  outmet.g_z.tx = D2(tx, step);
  outmet.g_z.ty = D2(ty, step);
  outmet.g_z.tz = D2(tz, step);
  outmet.g_z.xx = D2(xx, step);
  outmet.g_z.xy = D2(xy, step);
  outmet.g_z.xz = D2(xz, step);
  outmet.g_z.yy = D2(yy, step);
  outmet.g_z.yz = D2(yz, step);
  outmet.g_z.zz = D2(zz, step);

  return;
}

KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met,
                                struct three_metric &gam) {
  /* Check determinant first */
  gam.gxx = met.g.xx;
  gam.gxy = met.g.xy;
  gam.gxz = met.g.xz;
  gam.gyy = met.g.yy;
  gam.gyz = met.g.yz;
  gam.gzz = met.g.zz;

  Real det = adm::SpatialDet(gam.gxx, gam.gxy, gam.gxz,
                                   gam.gyy, gam.gyz, gam.gzz);

  /* If determinant is not >0  something is wrong with the metric */
  /* This could occur during the transition to merger at certain points
     so here we restart to Minkowski */
  if (!(det > 0)) {
    //std::fprintf(stderr, "det < 0: %e\n", det);
    //std::fprintf(stderr, "%e %e %e\n", gam.gxx, gam.gxy, gam.gxz);
    //std::fprintf(stderr, "%e %e %e\n", gam.gyy, gam.gyz, gam.gzz);
    //std::fflush(stderr);
    Kokkos::printf("det < 0: %e\n" // NOLINT
                   "%e %e %e\n"
                   "%e %e %e\n",
                   det, gam.gxx, gam.gxy, gam.gxz, gam.gyy, gam.gyz, gam.gzz);
    det = 1.0;
    gam.gxx = 1.0;
    gam.gxy = 0.0;
    gam.gxz = 0.0;
    gam.gyy = 1.0;
    gam.gyz = 0.0;
    gam.gzz = 1.0;
    Real betadownx = 0.0;
    Real betadowny = 0.0;
    Real betadownz = 0.0;

    Real dbetadownxx = 0.0;
    Real dbetadownyx = 0.0;
    Real dbetadownzx = 0.0;

    Real dbetadownxy = 0.0;
    Real dbetadownyy = 0.0;
    Real dbetadownzy = 0.0;

    Real dbetadownxz = 0.0;
    Real dbetadownyz = 0.0;
    Real dbetadownzz = 0.0;

    Real dtgxx = 0.0;
    Real dtgxy = 0.0;
    Real dtgxz = 0.0;
    Real dtgyy = 0.0;
    Real dtgyz = 0.0;
    Real dtgzz = 0.0;

    Real dgxxx = 0.0;
    Real dgxyx = 0.0;
    Real dgxzx = 0.0;
    Real dgyyx = 0.0;
    Real dgyzx = 0.0;
    Real dgzzx = 0.0;

    Real dgxxy = 0.0;
    Real dgxyy = 0.0;
    Real dgxzy = 0.0;
    Real dgyyy = 0.0;
    Real dgyzy = 0.0;
    Real dgzzy = 0.0;

    Real dgxxz = 0.0;
    Real dgxyz = 0.0;
    Real dgxzz = 0.0;
    Real dgyyz = 0.0;
    Real dgyzz = 0.0;
    Real dgzzz = 0.0;

    Real idetgxx = -gam.gyz * gam.gyz + gam.gyy * gam.gzz;
    Real idetgxy = gam.gxz * gam.gyz - gam.gxy * gam.gzz;
    Real idetgxz = -(gam.gxz * gam.gyy) + gam.gxy * gam.gyz;
    Real idetgyy = -gam.gxz * gam.gxz + gam.gxx * gam.gzz;
    Real idetgyz = gam.gxy * gam.gxz - gam.gxx * gam.gyz;
    Real idetgzz = -gam.gxy * gam.gxy + gam.gxx * gam.gyy;
    Real invgxx = idetgxx / det;
    Real invgxy = idetgxy / det;
    Real invgxz = idetgxz / det;
    Real invgyy = idetgyy / det;
    Real invgyz = idetgyz / det;
    Real invgzz = idetgzz / det;

    gam.betax = 0.0;
    gam.betay = 0.0;
    gam.betaz = 0.0;

    gam.alpha = 1.0;
    gam.kxx = 0.0;
    gam.kxy = 0.0;
    gam.kxz = 0.0;
    gam.kyy = 0.0;
    gam.kyz = 0.0;
    gam.kzz = 0.0;

  } else {
    /* Compute components if detg is not <0 */
    Real betadownx = met.g.tx;
    Real betadowny = met.g.ty;
    Real betadownz = met.g.tz;

    Real dbetadownxx = met.g_x.tx;
    Real dbetadownyx = met.g_x.ty;
    Real dbetadownzx = met.g_x.tz;

    Real dbetadownxy = met.g_y.tx;
    Real dbetadownyy = met.g_y.ty;
    Real dbetadownzy = met.g_y.tz;

    Real dbetadownxz = met.g_z.tx;
    Real dbetadownyz = met.g_z.ty;
    Real dbetadownzz = met.g_z.tz;

    Real dtgxx = met.g_t.xx;
    Real dtgxy = met.g_t.xy;
    Real dtgxz = met.g_t.xz;
    Real dtgyy = met.g_t.yy;
    Real dtgyz = met.g_t.yz;
    Real dtgzz = met.g_t.zz;

    Real dgxxx = met.g_x.xx;
    Real dgxyx = met.g_x.xy;
    Real dgxzx = met.g_x.xz;
    Real dgyyx = met.g_x.yy;
    Real dgyzx = met.g_x.yz;
    Real dgzzx = met.g_x.zz;

    Real dgxxy = met.g_y.xx;
    Real dgxyy = met.g_y.xy;
    Real dgxzy = met.g_y.xz;
    Real dgyyy = met.g_y.yy;
    Real dgyzy = met.g_y.yz;
    Real dgzzy = met.g_y.zz;

    Real dgxxz = met.g_z.xx;
    Real dgxyz = met.g_z.xy;
    Real dgxzz = met.g_z.xz;
    Real dgyyz = met.g_z.yy;
    Real dgyzz = met.g_z.yz;
    Real dgzzz = met.g_z.zz;

    Real idetgxx = -gam.gyz * gam.gyz + gam.gyy * gam.gzz;
    Real idetgxy = gam.gxz * gam.gyz - gam.gxy * gam.gzz;
    Real idetgxz = -(gam.gxz * gam.gyy) + gam.gxy * gam.gyz;
    Real idetgyy = -gam.gxz * gam.gxz + gam.gxx * gam.gzz;
    Real idetgyz = gam.gxy * gam.gxz - gam.gxx * gam.gyz;
    Real idetgzz = -gam.gxy * gam.gxy + gam.gxx * gam.gyy;

    Real invgxx = idetgxx / det;
    Real invgxy = idetgxy / det;
    Real invgxz = idetgxz / det;
    Real invgyy = idetgyy / det;
    Real invgyz = idetgyz / det;
    Real invgzz = idetgzz / det;

    gam.betax =
      betadownx * invgxx + betadowny * invgxy + betadownz * invgxz;

    gam.betay =
      betadownx * invgxy + betadowny * invgyy + betadownz * invgyz;

    gam.betaz =
      betadownx * invgxz + betadowny * invgyz + betadownz * invgzz;

    Real b2 =
      betadownx * gam.betax + betadowny * gam.betay +
      betadownz * gam.betaz;


    gam.alpha = sqrt(fabs(b2 - met.g.tt));

    gam.kxx = -(-2 * dbetadownxx - gam.betax * dgxxx - gam.betay * dgxxy -
      gam.betaz * dgxxz + 2 * (gam.betax * dgxxx + gam.betay * dgxyx +
        gam.betaz * dgxzx) + dtgxx) / (2. * gam.alpha);

    gam.kxy = -(-dbetadownxy - dbetadownyx + gam.betax * dgxxy -
      gam.betaz * dgxyz + gam.betaz * dgxzy + gam.betay * dgyyx +
      gam.betaz * dgyzx + dtgxy) / (2. * gam.alpha);

    gam.kxz = -(-dbetadownxz - dbetadownzx + gam.betax * dgxxz +
      gam.betay * dgxyz - gam.betay * dgxzy + gam.betay * dgyzx +
      gam.betaz * dgzzx + dtgxz) / (2. * gam.alpha);

    gam.kyy = -(-2 * dbetadownyy - gam.betax * dgyyx - gam.betay * dgyyy -
      gam.betaz * dgyyz + 2 * (gam.betax * dgxyy + gam.betay * dgyyy +
        gam.betaz * dgyzy) + dtgyy) / (2. * gam.alpha);

    gam.kyz = -(-dbetadownyz - dbetadownzy + gam.betax * dgxyz +
      gam.betax * dgxzy + gam.betay * dgyyz - gam.betax * dgyzx +
      gam.betaz * dgzzy + dtgyz) / (2. * gam.alpha);

    gam.kzz = -(-2 * dbetadownzz - gam.betax * dgzzx - gam.betay * dgzzy -
      gam.betaz * dgzzz + 2 * (gam.betax * dgxzz + gam.betay * dgyzz +
        gam.betaz * dgzzz) + dtgzz) / (2. * gam.alpha);
  }
  return 0;
}

bool SpinVectorWithinExtremality(const Real x, const Real y, const Real z) {
  const Real magnitude2 = x*x + y*y + z*z;
  return std::isfinite(magnitude2) && magnitude2 <= 1.0;
}

Real SmoothRamp01(const Real time, const Real start, const Real timescale,
                  Real *derivative) {
  *derivative = 0.0;
  if (time <= start) return 0.0;
  if (time >= start + timescale) return 1.0;
  const Real u = (time - start)/timescale;
  *derivative = 6.0*u*(1.0 - u)/timescale;
  return u*u*(3.0 - 2.0*u);
}

void LoadTrajectoryTable(const std::string &fname) {
  std::ifstream input(fname);
  if (!input.is_open()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "could not open trajectory file '" << fname << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bbh_table = bbh_traj_table();
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream row(line);
    Real value[21];
    for (int n = 0; n < 21; ++n) {
      if (!(row >> value[n]) || !std::isfinite(value[n])) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "invalid trajectory value in '" << fname
                  << "' line " << line_number << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    row >> std::ws;
    if (!row.eof() && row.peek() != '#') {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "unexpected extra trajectory column in '" << fname
                << "' line " << line_number << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real velocity1_sq = SQR(value[15]) + SQR(value[16]) + SQR(value[17]);
    const Real velocity2_sq = SQR(value[18]) + SQR(value[19]) + SQR(value[20]);
    if (!(value[1] > 0.0) || !(value[2] > 0.0) ||
        !(velocity1_sq < 1.0) || !(velocity2_sq < 1.0) ||
        !SpinVectorWithinExtremality(value[9], value[10], value[11]) ||
        !SpinVectorWithinExtremality(value[12], value[13], value[14])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "trajectory masses must be positive, velocities "
                << "subluminal, and |chi| <= 1 "
                << "in '" << fname << "' line " << line_number << std::endl;
      std::exit(EXIT_FAILURE);
    }
    bbh_table.t.push_back(value[0]);
    bbh_table.m1.push_back(value[1]);
    bbh_table.m2.push_back(value[2]);
    bbh_table.x1.push_back(value[3]);
    bbh_table.y1.push_back(value[4]);
    bbh_table.z1.push_back(value[5]);
    bbh_table.x2.push_back(value[6]);
    bbh_table.y2.push_back(value[7]);
    bbh_table.z2.push_back(value[8]);
    bbh_table.chix1.push_back(value[9]);
    bbh_table.chiy1.push_back(value[10]);
    bbh_table.chiz1.push_back(value[11]);
    bbh_table.chix2.push_back(value[12]);
    bbh_table.chiy2.push_back(value[13]);
    bbh_table.chiz2.push_back(value[14]);
    bbh_table.vx1.push_back(value[15]);
    bbh_table.vy1.push_back(value[16]);
    bbh_table.vz1.push_back(value[17]);
    bbh_table.vx2.push_back(value[18]);
    bbh_table.vy2.push_back(value[19]);
    bbh_table.vz2.push_back(value[20]);
  }
  if (bbh_table.t.size() < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "trajectory file must contain at least two rows"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t n = 1; n < bbh_table.t.size(); ++n) {
    if (!(bbh_table.t[n] > bbh_table.t[n - 1])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "trajectory times must be strictly increasing"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Loaded BBH trajectory table '" << fname << "' with "
              << bbh_table.t.size() << " rows" << std::endl;
  }
}

void find_traj_t(Real t, Real bbh_t[NTRAJ]) {
  Real dbbh_t[NTRAJ];
  find_traj_t_with_deriv(t, bbh_t, dbbh_t);
}

void find_traj_t_with_deriv(Real t, Real bbh_t[NTRAJ],
                            Real dbbh_t[NTRAJ]) {
  if (!bbh.use_traj_table) {
    const Real r1 = bbh.q/(1.0 + bbh.q)*bbh.sep;
    const Real r2 = -bbh.sep/(1.0 + bbh.q);
    const Real phase = bbh.om*t;
    const Real c = std::cos(phase);
    const Real s = std::sin(phase);
    const Real omega2 = bbh.om*bbh.om;
    bbh_t[X1] = r1*c;
    bbh_t[Y1] = r1*s;
    bbh_t[Z1] = 0.0;
    bbh_t[X2] = r2*c;
    bbh_t[Y2] = r2*s;
    bbh_t[Z2] = 0.0;
    bbh_t[VX1] = -r1*bbh.om*s;
    bbh_t[VY1] = r1*bbh.om*c;
    bbh_t[VZ1] = 0.0;
    bbh_t[VX2] = -r2*bbh.om*s;
    bbh_t[VY2] = r2*bbh.om*c;
    bbh_t[VZ2] = 0.0;
    Real spin_factor = 1.0;
    Real dspin_factor = 0.0;
    if (bbh.spin_ramp) {
      spin_factor = SmoothRamp01(t, bbh.spin_ramp_start_time,
                                 bbh.spin_ramp_timescale, &dspin_factor);
    }
    const Real e1x = std::sin(bbh.th_a1)*std::cos(bbh.ph_a1);
    const Real e1y = std::sin(bbh.th_a1)*std::sin(bbh.ph_a1);
    const Real e1z = std::cos(bbh.th_a1);
    const Real e2x = std::sin(bbh.th_a2)*std::cos(bbh.ph_a2);
    const Real e2y = std::sin(bbh.th_a2)*std::sin(bbh.ph_a2);
    const Real e2z = std::cos(bbh.th_a2);
    bbh_t[AX1] = bbh.a1*spin_factor*e1x;
    bbh_t[AY1] = bbh.a1*spin_factor*e1y;
    bbh_t[AZ1] = bbh.a1*spin_factor*e1z;
    bbh_t[AX2] = bbh.a2*spin_factor*e2x;
    bbh_t[AY2] = bbh.a2*spin_factor*e2y;
    bbh_t[AZ2] = bbh.a2*spin_factor*e2z;
    bbh_t[M1T] = 1.0/(bbh.q + 1.0);
    bbh_t[M2T] = 1.0 - bbh_t[M1T];
    dbbh_t[X1] = bbh_t[VX1];
    dbbh_t[Y1] = bbh_t[VY1];
    dbbh_t[Z1] = 0.0;
    dbbh_t[X2] = bbh_t[VX2];
    dbbh_t[Y2] = bbh_t[VY2];
    dbbh_t[Z2] = 0.0;
    dbbh_t[VX1] = -omega2*bbh_t[X1];
    dbbh_t[VY1] = -omega2*bbh_t[Y1];
    dbbh_t[VZ1] = 0.0;
    dbbh_t[VX2] = -omega2*bbh_t[X2];
    dbbh_t[VY2] = -omega2*bbh_t[Y2];
    dbbh_t[VZ2] = 0.0;
    dbbh_t[AX1] = bbh.a1*dspin_factor*e1x;
    dbbh_t[AY1] = bbh.a1*dspin_factor*e1y;
    dbbh_t[AZ1] = bbh.a1*dspin_factor*e1z;
    dbbh_t[AX2] = bbh.a2*dspin_factor*e2x;
    dbbh_t[AY2] = bbh.a2*dspin_factor*e2y;
    dbbh_t[AZ2] = bbh.a2*dspin_factor*e2z;
    dbbh_t[M1T] = 0.0;
    dbbh_t[M2T] = 0.0;
    return;
  }

  const std::vector<Real> &times = bbh_table.t;
  const Real span = times.back() - times.front();
  const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon()*
      std::max({1.0, std::abs(times.front()), std::abs(times.back()), span});
  if (t < times.front() - tolerance || t > times.back() + tolerance) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "requested time " << t
              << " is outside the trajectory-table range" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::size_t i0;
  std::size_t i1;
  const std::size_t cached = std::min(bbh_table.active_segment,
                                      times.size() - 2);
  if (t >= times[cached] && t <= times[cached + 1]) {
    i0 = cached;
    i1 = cached + 1;
  } else {
    const auto upper = std::upper_bound(times.begin(), times.end(), t);
    if (upper == times.begin()) {
      i0 = 0;
      i1 = 1;
    } else if (upper == times.end()) {
      i0 = times.size() - 2;
      i1 = times.size() - 1;
    } else {
      i1 = static_cast<std::size_t>(upper - times.begin());
      i0 = i1 - 1;
    }
    bbh_table.active_segment = i0;
  }
  const Real dt = times[i1] - times[i0];
  const Real w = (t - times[i0])/dt;
  const auto hermite = [w, dt](Real p0, Real v0, Real p1, Real v1,
                                Real *position, Real *velocity,
                                Real *acceleration) {
    const Real w2 = w*w;
    const Real w3 = w2*w;
    *position = (2*w3 - 3*w2 + 1)*p0 + (w3 - 2*w2 + w)*dt*v0 +
                (-2*w3 + 3*w2)*p1 + (w3 - w2)*dt*v1;
    *velocity = ((6*w2 - 6*w)*p0 + (3*w2 - 4*w + 1)*dt*v0 +
                 (-6*w2 + 6*w)*p1 + (3*w2 - 2*w)*dt*v1)/dt;
    *acceleration = ((12*w - 6)*p0 + (6*w - 4)*dt*v0 +
                     (-12*w + 6)*p1 + (6*w - 2)*dt*v1)/(dt*dt);
  };
  hermite(bbh_table.x1[i0], bbh_table.vx1[i0], bbh_table.x1[i1],
          bbh_table.vx1[i1], &bbh_t[X1], &bbh_t[VX1], &dbbh_t[VX1]);
  hermite(bbh_table.y1[i0], bbh_table.vy1[i0], bbh_table.y1[i1],
          bbh_table.vy1[i1], &bbh_t[Y1], &bbh_t[VY1], &dbbh_t[VY1]);
  hermite(bbh_table.z1[i0], bbh_table.vz1[i0], bbh_table.z1[i1],
          bbh_table.vz1[i1], &bbh_t[Z1], &bbh_t[VZ1], &dbbh_t[VZ1]);
  hermite(bbh_table.x2[i0], bbh_table.vx2[i0], bbh_table.x2[i1],
          bbh_table.vx2[i1], &bbh_t[X2], &bbh_t[VX2], &dbbh_t[VX2]);
  hermite(bbh_table.y2[i0], bbh_table.vy2[i0], bbh_table.y2[i1],
          bbh_table.vy2[i1], &bbh_t[Y2], &bbh_t[VY2], &dbbh_t[VY2]);
  hermite(bbh_table.z2[i0], bbh_table.vz2[i0], bbh_table.z2[i1],
          bbh_table.vz2[i1], &bbh_t[Z2], &bbh_t[VZ2], &dbbh_t[VZ2]);
  dbbh_t[X1] = bbh_t[VX1];
  dbbh_t[Y1] = bbh_t[VY1];
  dbbh_t[Z1] = bbh_t[VZ1];
  dbbh_t[X2] = bbh_t[VX2];
  dbbh_t[Y2] = bbh_t[VY2];
  dbbh_t[Z2] = bbh_t[VZ2];
  const auto linear = [w](Real a, Real b) { return (1.0 - w)*a + w*b; };
  const auto slope = [dt](Real a, Real b) { return (b - a)/dt; };
  const auto interpolate = [&](const std::vector<Real> &field, int index) {
    bbh_t[index] = linear(field[i0], field[i1]);
    dbbh_t[index] = slope(field[i0], field[i1]);
  };
  interpolate(bbh_table.chix1, AX1);
  interpolate(bbh_table.chiy1, AY1);
  interpolate(bbh_table.chiz1, AZ1);
  interpolate(bbh_table.chix2, AX2);
  interpolate(bbh_table.chiy2, AY2);
  interpolate(bbh_table.chiz2, AZ2);
  interpolate(bbh_table.m1, M1T);
  interpolate(bbh_table.m2, M2T);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T BoostGammaMinusOneOverV2(const T v2, const T gamma) {
  // This series avoids the removable 0/0 singularity as v -> 0.
  if (value_of(v2) < 1.0e-12) {
    return T(0.5) + T(0.375)*v2 + T(0.3125)*v2*v2;
  }
  return (gamma - T(1.0))/v2;
}

template <typename T>
KOKKOS_INLINE_FUNCTION void BoostedCoordinates(
    const T x, const T y, const T z, const T x0, const T y0, const T z0,
    const T vx, const T vy, const T vz, T *xbh, T *ybh, T *zbh) {
  const T dx = x - x0;
  const T dy = y - y0;
  const T dz = z - z0;
  const T v2 = vx*vx + vy*vy + vz*vz;
  const T gamma = T(1.0)/metric_sqrt(T(1.0) - v2);
  const T q = BoostGammaMinusOneOverV2(v2, gamma);
  const T vdotx = vx*dx + vy*dy + vz*dz;
  *xbh = dx + q*vx*vdotx;
  *ybh = dy + q*vy*vdotx;
  *zbh = dz + q*vz*vdotx;
}

template <typename T>
KOKKOS_INLINE_FUNCTION void BuildBoostJacobian(
    const T vx, const T vy, const T vz, T jac[NDIM][NDIM]) {
  const T v2 = vx*vx + vy*vy + vz*vz;
  const T gamma = T(1.0)/metric_sqrt(T(1.0) - v2);
  const T q = BoostGammaMinusOneOverV2(v2, gamma);
  for (int a = 0; a < NDIM; ++a) {
    for (int b = 0; b < NDIM; ++b) jac[a][b] = T(0.0);
  }
  jac[0][0] = gamma;
  jac[0][1] = -gamma*vx;
  jac[0][2] = -gamma*vy;
  jac[0][3] = -gamma*vz;
  jac[1][0] = jac[0][1];
  jac[2][0] = jac[0][2];
  jac[3][0] = jac[0][3];
  jac[1][1] = T(1.0) + q*vx*vx;
  jac[1][2] = q*vx*vy;
  jac[1][3] = q*vx*vz;
  jac[2][1] = jac[1][2];
  jac[2][2] = T(1.0) + q*vy*vy;
  jac[2][3] = q*vy*vz;
  jac[3][1] = jac[1][3];
  jac[3][2] = jac[2][3];
  jac[3][3] = T(1.0) + q*vz*vz;
}

template <typename T>
KOKKOS_INLINE_FUNCTION void KerrSchildPerturbation(
    const T x, const T y, const T z, const T ax, const T ay, const T az,
    const T mass, T ks[NDIM][NDIM]) {
  const T root2 = T(1.4142135623730951);
  const T iroot2 = T(1.0)/root2;
  const T spin2 = ax*ax + ay*ay + az*az;
  const T radius2 = x*x + y*y + z*z;
  const T adotx = ax*x + ay*y + az*z;
  const T term = radius2 - spin2;
  const T rho2 = term + metric_sqrt(T(4.0)*adotx*adotx + term*term);
  const T rho = metric_sqrt(rho2);
  const T fac = iroot2*rho2*rho*mass/
      (adotx*adotx + T(0.25)*rho2*rho2);
  const T den = spin2 + T(0.5)*rho2;
  T ell[3];
  ell[0] = y*az - z*ay + root2*adotx*ax/rho + rho*x*iroot2;
  ell[1] = -x*az + z*ax + root2*adotx*ay/rho + rho*y*iroot2;
  ell[2] = x*ay - y*ax + root2*adotx*az/rho + rho*z*iroot2;
  for (int a = 0; a < NDIM; ++a) {
    for (int b = 0; b < NDIM; ++b) ks[a][b] = T(0.0);
  }
  ks[0][0] = fac;
  for (int a = 0; a < 3; ++a) {
    ks[0][a + 1] = fac*ell[a]/den;
    ks[a + 1][0] = ks[0][a + 1];
    for (int b = a; b < 3; ++b) {
      ks[a + 1][b + 1] = fac*ell[a]*ell[b]/(den*den);
      ks[b + 1][a + 1] = ks[a + 1][b + 1];
    }
  }
}

template <typename T>
KOKKOS_INLINE_FUNCTION void AddBoostedHole(
    const T ks[NDIM][NDIM], const T jac[NDIM][NDIM],
    T gcov[NDIM][NDIM]) {
  for (int a = 0; a < NDIM; ++a) {
    for (int b = a; b < NDIM; ++b) {
      T sum = T(0.0);
      for (int m = 0; m < NDIM; ++m) {
        for (int n = 0; n < NDIM; ++n) {
          sum = sum + jac[m][a]*jac[n][b]*ks[m][n];
        }
      }
      gcov[a][b] = gcov[a][b] + sum;
      gcov[b][a] = gcov[a][b];
    }
  }
}

template <typename T>
KOKKOS_INLINE_FUNCTION void SuperposedBBHTemplate(
    const T x, const T y, const T z, T gcov[NDIM][NDIM],
    const T tr[NTRAJ], const bbh_pgen& b) {
  const T v1x = tr[VX1];
  const T v1y = tr[VY1];
  const T v1z = tr[VZ1];
  const T v2x = tr[VX2];
  const T v2y = tr[VY2];
  const T v2z = tr[VZ2];
  const T mass1 = tr[M1T];
  const T mass2 = tr[M2T];
  // Trajectory spins are dimensionless chi; the Kerr parameter is a=M*chi.
  const T a1x = tr[AX1]*mass1;
  const T a1y = tr[AY1]*mass1;
  const T a1z = tr[AZ1]*mass1;
  const T a2x = tr[AX2]*mass2;
  const T a2y = tr[AY2]*mass2;
  const T a2z = tr[AZ2]*mass2;

  T x1, y1, z1, x2, y2, z2;
  BoostedCoordinates(x, y, z, tr[X1], tr[Y1], tr[Z1],
                           v1x, v1y, v1z, &x1, &y1, &z1);
  BoostedCoordinates(x, y, z, tr[X2], tr[Y2], tr[Z2],
                           v2x, v2y, v2z, &x2, &y2, &z2);
  const T radius1 = metric_norm3(x1, y1, z1);
  const T radius2 = metric_norm3(x2, y2, z2);
  const T spin1_norm = metric_sqrt(a1x*a1x + a1y*a1y + a1z*a1z + T(1e-40));
  const T spin2_norm = metric_sqrt(a2x*a2x + a2y*a2y + a2z*a2z + T(1e-40));
  const T cutoff1 = spin1_norm*(T(1.0) + b.a1_buffer) + b.cutoff_floor;
  const T cutoff2 = spin2_norm*(T(1.0) + b.a2_buffer) + b.cutoff_floor;
  if (value_of(radius1) < value_of(cutoff1)) {
    z1 = (value_of(z1) > 0.0) ? cutoff1 : -cutoff1;
  }
  if (value_of(radius2) < value_of(cutoff2)) {
    z2 = (value_of(z2) > 0.0) ? cutoff2 : -cutoff2;
  }

  T ks1[NDIM][NDIM], ks2[NDIM][NDIM];
  T jac1[NDIM][NDIM], jac2[NDIM][NDIM];
  KerrSchildPerturbation(x1, y1, z1, a1x, a1y, a1z, mass1, ks1);
  KerrSchildPerturbation(x2, y2, z2, a2x, a2y, a2z, mass2, ks2);
  BuildBoostJacobian(v1x, v1y, v1z, jac1);
  BuildBoostJacobian(v2x, v2y, v2z, jac2);
  for (int a = 0; a < NDIM; ++a) {
    for (int bidx = 0; bidx < NDIM; ++bidx) {
      gcov[a][bidx] = (a == bidx) ? ((a == 0) ? T(-1.0) : T(1.0)) : T(0.0);
    }
  }
  AddBoostedHole(ks1, jac1, gcov);
  AddBoostedHole(ks2, jac2, gcov);
}

KOKKOS_INLINE_FUNCTION void SuperposedBBH(
    const Real time, const Real x, const Real y, const Real z,
    Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen& bbh_) {
  (void)time;
  SuperposedBBHTemplate(x, y, z, gcov, traj_array, bbh_);
}

KOKKOS_INLINE_FUNCTION void FillMetricDerivative(
    const dual1_real gcov[NDIM][NDIM], struct dd_sym &dg) {
  dg.tt = gcov[TT][TT].deriv;
  dg.tx = gcov[TT][XX].deriv;
  dg.ty = gcov[TT][YY].deriv;
  dg.tz = gcov[TT][ZZ].deriv;
  dg.xx = gcov[XX][XX].deriv;
  dg.xy = gcov[XX][YY].deriv;
  dg.xz = gcov[XX][ZZ].deriv;
  dg.yy = gcov[YY][YY].deriv;
  dg.yz = gcov[YY][ZZ].deriv;
  dg.zz = gcov[ZZ][ZZ].deriv;
}

KOKKOS_INLINE_FUNCTION void MetricDerivativeAD(
    const Real x, const Real y, const Real z, const int direction,
    const Real tr[NTRAJ], const Real dtr[NTRAJ], const bbh_pgen& b,
    struct dd_sym &dg) {
  dual1_real xd(x, direction == 1 ? 1.0 : 0.0);
  dual1_real yd(y, direction == 2 ? 1.0 : 0.0);
  dual1_real zd(z, direction == 3 ? 1.0 : 0.0);
  dual1_real trd[NTRAJ];
  for (int n = 0; n < NTRAJ; ++n) {
    trd[n] = dual1_real(tr[n], direction == 0 ? dtr[n] : 0.0);
  }
  dual1_real gcov[NDIM][NDIM];
  SuperposedBBHTemplate(xd, yd, zd, gcov, trd, b);
  FillMetricDerivative(gcov, dg);
}

KOKKOS_INLINE_FUNCTION void get_metric_and_derivatives(
    const Real t, const Real x, const Real y, const Real z,
    struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
    const Real dbbh_traj_loc[NTRAJ], const bbh_pgen& bbh_) {
  // Preserve the exact existing real-valued metric evaluation.
  get_metric(t, x, y, z, met, bbh_traj_loc, bbh_);
  MetricDerivativeAD(x, y, z, 0, bbh_traj_loc, dbbh_traj_loc, bbh_, met.g_t);
  MetricDerivativeAD(x, y, z, 1, bbh_traj_loc, dbbh_traj_loc, bbh_, met.g_x);
  MetricDerivativeAD(x, y, z, 2, bbh_traj_loc, dbbh_traj_loc, bbh_, met.g_y);
  MetricDerivativeAD(x, y, z, 3, bbh_traj_loc, dbbh_traj_loc, bbh_, met.g_z);
}

KOKKOS_INLINE_FUNCTION
void get_metric(const Real t,
                const Real x,
                const Real y,
                const Real z,
                struct four_metric &met,
                const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen& bbh_) {
  Real gcov[NDIM][NDIM];

  SuperposedBBH(t, x, y, z, gcov, bbh_traj_loc, bbh_);

  met.g.tt = gcov[TT][TT];
  met.g.tx = gcov[TT][XX];
  met.g.ty = gcov[TT][YY];
  met.g.tz = gcov[TT][ZZ];
  met.g.xx = gcov[XX][XX];
  met.g.xy = gcov[XX][YY];
  met.g.xz = gcov[XX][ZZ];
  met.g.yy = gcov[YY][YY];
  met.g.yz = gcov[YY][ZZ];
  met.g.zz = gcov[ZZ][ZZ];

  return;
}

void DynBBHFluxHistory(HistoryData *pdata, Mesh *pm) {
  Real trajectory[NTRAJ];
  find_traj_t(pm->time, trajectory);
  const Real center1[3] = {trajectory[X1], trajectory[Y1], trajectory[Z1]};
  const Real center2[3] = {trajectory[X2], trajectory[Y2], trajectory[Z2]};
  std::vector<SphericalSurfaceGrid*> surfaces;
  surfaces.reserve(pm->pgen->surface_flux_grids.size());
  for (const auto &owned : pm->pgen->surface_flux_grids) {
    if (owned->Label() == "h1") owned->SetCenter(center1);
    if (owned->Label() == "h2") owned->SetCenter(center2);
    surfaces.push_back(owned.get());
  }
  TorusFluxes_General(pdata, pm->pmb_pack, surfaces);
}

// refine region within a certain distance from each compact object
void RefineAlphaMin(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->padm->u_adm;
  int I_ADM_ALPHA  = pmbp->padm->I_ADM_ALPHA;
  // note: we need this to prevent capture by this in the lambda expr.
  auto &bbh_ = bbh;

  par_for_outer(
  "AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_dmin;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &dmin) {
        int k = (idx) / nji;
        int j = (idx - k * nji) / nx1;
        int i = (idx - k * nji - j * nx1) + is;
        j += js;
        k += ks;
        dmin = fmin(u0(m, I_ADM_ALPHA, k, j, i), dmin);
      },
      Kokkos::Min<Real>(team_dmin));

    if (team_dmin < bbh_.alpha_thr) {
      refine_flag.d_view(m + mbs) = 1;
    }
    if (team_dmin > 1.25 * bbh_.alpha_thr) {
      refine_flag.d_view(m + mbs) = -1;
    }
  });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

void RefineTracker(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  Real bbh_traj[NTRAJ];

  Real tt = pmesh->time;
  find_traj_t(tt, bbh_traj);
  Real x1_BH1 = bbh_traj[X1];
  Real x2_BH1 = bbh_traj[Y1];
  Real x3_BH1 = bbh_traj[Z1];
  Real x1_BH2 = bbh_traj[X2];
  Real x2_BH2 = bbh_traj[Y2];
  Real x3_BH2 = bbh_traj[Z2];
  for (int m = 0; m < nmb; ++m) {
    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    Real d2_bh1[8] = {
      SQR(x1min - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1min - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1min - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3max - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3max - x3_BH1),
      SQR(x1min - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3max - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3max - x3_BH1),
    };

    Real d2_bh2[8] = {
      SQR(x1min - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1min - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1min - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3max - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3max - x3_BH2),
      SQR(x1min - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3max - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3max - x3_BH2),
    };
    Real dmin2_bh1 = *std::min_element(&d2_bh1[0], &d2_bh1[8]);
    Real dmin2_bh2 = *std::min_element(&d2_bh2[0], &d2_bh2[8]);
    bool iscontained_bh1 =
      (x1_BH1 >= x1min && x1_BH1 <= x1max) &&
      (x2_BH1 >= x2min && x2_BH1 <= x2max) &&
      (x3_BH1 >= x3min && x3_BH1 <= x3max);
    bool iscontained_bh2 =
      (x1_BH2 >= x1min && x1_BH2 <= x1max) &&
      (x2_BH2 >= x2min && x2_BH2 <= x2max) &&
      (x3_BH2 >= x3min && x3_BH2 <= x3max);

    if (dmin2_bh1 < SQR(bbh.radius_thr) || dmin2_bh2 < SQR(bbh.radius_thr) ||
        iscontained_bh1 || iscontained_bh2) {
      refine_flag.d_view(m + mbs) = 1;
    } else {
      refine_flag.d_view(m + mbs) = -1;
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // namespace
