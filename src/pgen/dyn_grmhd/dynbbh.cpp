#include <stdio.h>
#include <math.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <Kokkos_Random.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "diffusion/current_density.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "dyn_radiation/dyn_radiation.hpp"
#include "units/units.hpp"
#include "utils/flux_generalized.hpp"


#define D2(comp, h) ((met_p1.g).comp - (met_m1.g).comp) / (2*h)

namespace {

enum {
  TT, XX, YY, ZZ, NDIM
};

constexpr Real kDefaultMetricFdStep = 5.0e-5;

KOKKOS_INLINE_FUNCTION Real metric_sqrt(const Real x) { return sqrt(x); }
KOKKOS_INLINE_FUNCTION Real value_of(const Real x) { return x; }

struct dual2_real {
  Real val;
  Real deriv0;
  Real deriv1;

  KOKKOS_INLINE_FUNCTION dual2_real()
      : val(0.0), deriv0(0.0), deriv1(0.0) {}
  KOKKOS_INLINE_FUNCTION dual2_real(const Real value)
      : val(value), deriv0(0.0), deriv1(0.0) {}
  KOKKOS_INLINE_FUNCTION dual2_real(const Real value, const Real derivative0,
                                    const Real derivative1)
      : val(value), deriv0(derivative0), deriv1(derivative1) {}
};

KOKKOS_INLINE_FUNCTION dual2_real operator+(const dual2_real &a,
                                             const dual2_real &b) {
  return dual2_real(a.val + b.val, a.deriv0 + b.deriv0,
                    a.deriv1 + b.deriv1);
}
KOKKOS_INLINE_FUNCTION dual2_real operator-(const dual2_real &a,
                                             const dual2_real &b) {
  return dual2_real(a.val - b.val, a.deriv0 - b.deriv0,
                    a.deriv1 - b.deriv1);
}
KOKKOS_INLINE_FUNCTION dual2_real operator-(const dual2_real &a) {
  return dual2_real(-a.val, -a.deriv0, -a.deriv1);
}
KOKKOS_INLINE_FUNCTION dual2_real operator*(const dual2_real &a,
                                             const dual2_real &b) {
  return dual2_real(a.val*b.val,
                    a.deriv0*b.val + a.val*b.deriv0,
                    a.deriv1*b.val + a.val*b.deriv1);
}
KOKKOS_INLINE_FUNCTION dual2_real operator/(const dual2_real &a,
                                             const dual2_real &b) {
  const Real inv = 1.0/b.val;
  return dual2_real(a.val*inv,
                    (a.deriv0*b.val - a.val*b.deriv0)*inv*inv,
                    (a.deriv1*b.val - a.val*b.deriv1)*inv*inv);
}
KOKKOS_INLINE_FUNCTION dual2_real metric_sqrt(const dual2_real &x) {
  const Real root = sqrt(x.val);
  const Real factor = 0.5/root;
  return dual2_real(root, factor*x.deriv0, factor*x.deriv1);
}
KOKKOS_INLINE_FUNCTION Real value_of(const dual2_real &x) { return x.val; }

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

// Device kernels only need this small subset of the full problem state when
// evaluating the binary metric.  Keeping it separate avoids copying all torus,
// sink, and refinement parameters into every metric-kernel closure.
struct bbh_metric_params {
  Real a1_buffer;
  Real a2_buffer;
  Real cutoff_floor;
  Real metric_fd_step;
  MetricDerivativeMethod metric_derivative_method;
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
  Real d;
  Real gamma_adi;
  Real a1_buffer, a2_buffer;
  Real cutoff_floor;
  Real metric_fd_step = kDefaultMetricFdStep;
  Real alpha_thr;
  Real radius_thr;
  Real smooth_b_damping_eta;
  Real smooth_b_damping_cfl;
  Real puncture_excise_rad1;
  Real puncture_excise_rad2;
  Real puncture_excise_shrink_timescale;
  Real puncture_excise_shrink_start_time;
  Real sink_radius;
  Real sink_width;
  Real sink_timescale;
  Real sink_density_floor;
  Real sink_pressure_floor;
  Real sink_cells_per_radius;
  Real sink_resolved_cells_across_horizon;
  Real test_bz_gradient;
  MetricDerivativeMethod metric_derivative_method =
      MetricDerivativeMethod::finite_difference;
  bool use_traj_table = false;
  bool spin_ramp = false;
  bool smooth_b_damping = false;
  bool puncture_excise_cap_to_horizon = false;
  bool puncture_excise_to_horizon = false;
  bool puncture_excise_shrink_to_horizon = false;
  Real puncture_excise_horizon_fraction = 1.0;
  bool require_resolved_horizon = false;
  bool unresolved_sink = false;
  Real spin;

  Real dexcise, pexcise;                      // excision parameters
  Real arad;                                  // radiation constant
  Real restart_seed_erad_fraction;            // multiplier for old GRMHD restart seed
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
  Real potential_cutoff, potential_falloff;   // sets region of torus to magnetize
  Real potential_r_pow;                       // set how vector potential scales
  Real potential_beta_min;                    // set how vector potential scales (cont.)
  Real potential_rho_pow;                     // set vector potential dependence on rho
};

static_assert(std::is_trivially_copyable<bbh_metric_params>::value,
              "metric kernel parameters must remain device-copyable");
static_assert(std::is_trivially_copyable<bbh_pgen>::value,
              "torus kernel parameters must remain device-copyable");

enum class RefineBasePolicy {
  none,
  alpha_min,
  tracker
};

struct bbh_refine {
  RefineBasePolicy base = RefineBasePolicy::none;
  Real tracker_radius[2] = {0.0, 0.0};
  int tracker_reflevel[2] = {-1, -1};
  Real hysteresis = 1.25;
  std::vector<Real> com_radius;
  std::vector<int> com_reflevel;
};

struct bbh_pgen bbh;
struct bbh_refine bbh_ref;

struct bbh_sink_hole_state {
  Real x, y, z;
  Real horizon_radius;
  Real mesh_dx;
  Real sink_radius;
  Real sink_width;
  int active;
};

struct bbh_sink_state {
  bbh_sink_hole_state hole1;
  bbh_sink_hole_state hole2;
};

static_assert(std::is_trivially_copyable<bbh_sink_state>::value,
              "sink kernel state must remain device-copyable");

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

Real LocalFinestMeshSpacing(MeshBlockPack *pmbp) {
  Real min_dx = std::numeric_limits<Real>::max();
  auto &mb_size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    min_dx = std::min(min_dx, mb_size.h_view(m).dx1);
    if (indcs.nx2 > 1) min_dx = std::min(min_dx, mb_size.h_view(m).dx2);
    if (indcs.nx3 > 1) min_dx = std::min(min_dx, mb_size.h_view(m).dx3);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &min_dx, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  return min_dx;
}

Real LocalMeshSpacingAtPoint(MeshBlockPack *pmbp, Real x, Real y, Real z) {
  Real local_dx = std::numeric_limits<Real>::max();
  auto &mb_size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    auto mb = mb_size.h_view(m);
    const Real eps = 16.0*std::numeric_limits<Real>::epsilon()*
        std::max({1.0, std::abs(mb.x1min), std::abs(mb.x1max),
                  std::abs(mb.x2min), std::abs(mb.x2max),
                  std::abs(mb.x3min), std::abs(mb.x3max)});
    bool contains = (x >= mb.x1min-eps && x <= mb.x1max+eps);
    if (indcs.nx2 > 1) contains = contains &&
        (y >= mb.x2min-eps && y <= mb.x2max+eps);
    if (indcs.nx3 > 1) contains = contains &&
        (z >= mb.x3min-eps && z <= mb.x3max+eps);
    if (contains) {
      Real dx = mb.dx1;
      if (indcs.nx2 > 1) dx = std::max(dx, mb.dx2);
      if (indcs.nx3 > 1) dx = std::max(dx, mb.dx3);
      local_dx = std::min(local_dx, dx);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &local_dx, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  return (local_dx == std::numeric_limits<Real>::max()) ?
      LocalFinestMeshSpacing(pmbp) : local_dx;
}

Real HorizonRadiusFromMassAndChi(Real mass, Real chix, Real chiy, Real chiz) {
  const Real chi2 = SQR(chix) + SQR(chiy) + SQR(chiz);
  return mass*(1.0 + std::sqrt(std::max(1.0-chi2, 0.0)));
}

Real SmoothExcisionRadiusToHorizon(Real requested, Real horizon, Real elapsed,
                                   Real timescale, bool set_to_horizon,
                                   bool shrink_to_horizon) {
  Real start = (requested > 0.0) ? requested : horizon;
  if (set_to_horizon) return horizon;
  if (!shrink_to_horizon) return start;
  start = std::max(start, horizon);
  Real s = std::min(std::max(elapsed/timescale, 0.0), 1.0);
  s = s*s*(3.0 - 2.0*s);
  return (1.0-s)*start + s*horizon;
}

Real MinDynBBHHorizonRadius() {
  if (bbh.use_traj_table && !bbh_table.t.empty()) {
    Real rmin = std::numeric_limits<Real>::max();
    for (std::size_t n = 0; n < bbh_table.t.size(); ++n) {
      rmin = std::min(rmin, HorizonRadiusFromMassAndChi(
          bbh_table.m1[n], bbh_table.chix1[n], bbh_table.chiy1[n], bbh_table.chiz1[n]));
      rmin = std::min(rmin, HorizonRadiusFromMassAndChi(
          bbh_table.m2[n], bbh_table.chix2[n], bbh_table.chiy2[n], bbh_table.chiz2[n]));
    }
    return rmin;
  }
  Real q[NTRAJ];
  find_traj_t(0.0, q);
  return std::min(HorizonRadiusFromMassAndChi(q[M1T], q[AX1], q[AY1], q[AZ1]),
                  HorizonRadiusFromMassAndChi(q[M2T], q[AX2], q[AY2], q[AZ2]));
}

bbh_sink_hole_state MakeSinkHoleState(MeshBlockPack *pmbp, Real x, Real y, Real z,
                                      Real horizon) {
  bbh_sink_hole_state h{x, y, z, horizon, LocalMeshSpacingAtPoint(pmbp, x, y, z),
                        0.0, 0.0, 0};
  const Real cells = 2.0*horizon/std::max(h.mesh_dx, 1.0e-300);
  h.active = (cells < bbh.sink_resolved_cells_across_horizon) ? 1 : 0;
  h.sink_radius = std::max(horizon, bbh.sink_cells_per_radius*h.mesh_dx);
  if (bbh.sink_radius > 0.0) h.sink_radius = std::max(h.sink_radius, bbh.sink_radius);
  h.sink_width = (bbh.sink_width > 0.0) ?
      std::min(bbh.sink_width, h.sink_radius) :
      std::max(h.mesh_dx, 0.25*h.sink_radius);
  return h;
}

bbh_sink_state ComputeUnresolvedSinkState(MeshBlockPack *pmbp,
                                          const Real q[NTRAJ]) {
  bbh_sink_state state;
  state.hole1 = MakeSinkHoleState(pmbp, q[X1], q[Y1], q[Z1],
      HorizonRadiusFromMassAndChi(q[M1T], q[AX1], q[AY1], q[AZ1]));
  state.hole2 = MakeSinkHoleState(pmbp, q[X2], q[Y2], q[Z2],
      HorizonRadiusFromMassAndChi(q[M2T], q[AX2], q[AY2], q[AZ2]));
  return state;
}

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const Real hm, const Real hp, const bbh_metric_params& metric);
KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met, struct three_metric &gam);
KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
                const bbh_metric_params& metric);
KOKKOS_INLINE_FUNCTION
void get_adm_and_derivatives_ad(const Real t, const Real x, const Real y,
                                const Real z, struct three_metric &gam,
                                const Real bbh_traj_loc[NTRAJ],
                                const Real dbbh_traj_loc[NTRAJ],
                                const bbh_metric_params& metric);
KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                   Real gcov[][NDIM], const Real traj_array[NTRAJ],
                   const bbh_metric_params& metric);
void SetADMVariablesToBBH(MeshBlockPack *pmbp);
void RefineAlphaMin(MeshBlockPack* pmbp);
void RefineSpatialPolicy(MeshBlockPack* pmbp);
void Refine(MeshBlockPack* pmbp);
void DynBBHFluxHistory(HistoryData *pdata, Mesh *pm);
void AddUnresolvedBHSink(Mesh *pm, const Real bdt);
void AddSmoothExcisionMagneticDamping(Mesh *pm, DvceEdgeFld4D<Real> &efld);

KOKKOS_INLINE_FUNCTION
static void GetSuperposedAndInverse(const Real t,
                            const Real x, const Real y, const Real z,
                            Real gcov[][NDIM], Real gcon[][NDIM], const Real bbh_traj_loc[NTRAJ],
                            const bbh_metric_params& metric);



KOKKOS_INLINE_FUNCTION
static void CalculateCN(const bbh_pgen& pgen, Real *cparam, Real *nparam);

KOKKOS_INLINE_FUNCTION
static Real CalculateL(const bbh_pgen& pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(const bbh_pgen& pgen, Real r, Real sin_theta, Real l);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(const bbh_pgen& pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static Real CalculateT(const bbh_pgen& pgen, Real rho, Real ptot_over_rho);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(const bbh_pgen& pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(const bbh_pgen& pgen,
                                           Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);
KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(const bbh_pgen& pgen,
                                     Real r, Real sin_theta, Real *pu0, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void TransformVector(const bbh_pgen& pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotentialInTiltedTorus(const bbh_pgen& pgen,
                                                  Real r, Real theta, Real phi,
                                                  Real *patheta, Real *paphi);


KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(const bbh_pgen& pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void InvertMetric(Real gcov[][NDIM], Real gcon[][NDIM]);

KOKKOS_INLINE_FUNCTION
Real A1(const bbh_pgen& pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(const bbh_pgen& pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(const bbh_pgen& pgen, Real x1, Real x2, Real x3);

void InitializeDynBBHRestartDynRadiation(Mesh *pm);
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
  bbh.test_bz_gradient = pin->GetOrAddReal(
      "problem", "test_bz_gradient", 0.0);
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
  const std::string amr_cond = pin->GetOrAddString(
      "problem", "amr_condition", "track");
  if (amr_cond == "none") {
    bbh_ref.base = RefineBasePolicy::none;
  } else if (amr_cond == "alpha_min") {
    bbh_ref.base = RefineBasePolicy::alpha_min;
  } else if (amr_cond == "tracker" || amr_cond == "track") {
    bbh_ref.base = RefineBasePolicy::tracker;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown problem/amr_condition='" << amr_cond
              << "'. Use 'none', 'alpha_min', or 'tracker'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int common_tracker_level = pin->GetOrAddInteger(
      "problem", "tracker_reflevel", -1);
  bbh_ref.tracker_radius[0] = pin->GetOrAddReal(
      "problem", "tracker_1_rad", bbh.radius_thr);
  bbh_ref.tracker_radius[1] = pin->GetOrAddReal(
      "problem", "tracker_2_rad", bbh.radius_thr);
  bbh_ref.tracker_reflevel[0] = pin->GetOrAddInteger(
      "problem", "tracker_1_reflevel", common_tracker_level);
  bbh_ref.tracker_reflevel[1] = pin->GetOrAddInteger(
      "problem", "tracker_2_reflevel", common_tracker_level);
  bbh_ref.hysteresis = pin->GetOrAddReal(
      "problem", "refinement_hysteresis", 1.25);
  bbh_ref.com_radius.clear();
  bbh_ref.com_reflevel.clear();
  for (int nr = 0; nr < 16; ++nr) {
    const std::string prefix = "radius_" + std::to_string(nr);
    if (pin->DoesParameterExist("problem", prefix + "_rad")) {
      bbh_ref.com_radius.push_back(pin->GetReal("problem", prefix + "_rad"));
      bbh_ref.com_reflevel.push_back(pin->GetOrAddInteger(
          "problem", prefix + "_reflevel", -1));
    }
  }
  const int max_physical_level = pmbp->pmesh->max_level - pmbp->pmesh->root_level;
  auto invalid_reflevel = [=](const int level) {
    return level < -1 || (pmbp->pmesh->adaptive && level > max_physical_level);
  };
  bool invalid_refinement = !(bbh.radius_thr > 0.0) ||
      !(bbh.alpha_thr > 0.0) || !std::isfinite(bbh.alpha_thr) ||
      !(bbh_ref.tracker_radius[0] > 0.0) ||
      !(bbh_ref.tracker_radius[1] > 0.0) ||
      !(bbh_ref.hysteresis >= 1.0) || !std::isfinite(bbh_ref.hysteresis) ||
      invalid_reflevel(bbh_ref.tracker_reflevel[0]) ||
      invalid_reflevel(bbh_ref.tracker_reflevel[1]);
  for (std::size_t nr = 0; nr < bbh_ref.com_radius.size(); ++nr) {
    invalid_refinement = invalid_refinement ||
        !(bbh_ref.com_radius[nr] > 0.0) ||
        !std::isfinite(bbh_ref.com_radius[nr]) ||
        invalid_reflevel(bbh_ref.com_reflevel[nr]);
  }
  if (invalid_refinement) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh refinement radii must be positive, "
              << "refinement_hysteresis must be finite and >= 1, and target "
              << "levels must be -1 or valid physical AMR levels" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_ref_func = Refine;
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

  bbh.require_resolved_horizon = pin->GetOrAddBoolean(
      "coord", "require_resolved_horizon", false);
  bbh.puncture_excise_rad1 = pin->GetOrAddReal("coord", "excise_1_rad", -1.0);
  bbh.puncture_excise_rad2 = pin->GetOrAddReal("coord", "excise_2_rad", -1.0);
  bbh.puncture_excise_cap_to_horizon = pin->GetOrAddBoolean(
      "coord", "excise_cap_to_horizon", false);
  bbh.puncture_excise_to_horizon = pin->GetOrAddBoolean(
      "coord", "excise_to_horizon", false);
  bbh.puncture_excise_shrink_to_horizon = pin->GetOrAddBoolean(
      "coord", "excise_shrink_to_horizon", false);
  bbh.puncture_excise_shrink_timescale = pin->GetOrAddReal(
      "coord", "excise_shrink_timescale", 50.0);
  bbh.puncture_excise_shrink_start_time = pin->GetOrAddReal(
      "coord", "excise_shrink_start_time", 0.0);
  bbh.puncture_excise_horizon_fraction = pin->GetOrAddReal(
      "coord", "excise_horizon_fraction", 1.0);
  if (!(bbh.puncture_excise_horizon_fraction > 0.0 &&
        bbh.puncture_excise_horizon_fraction <= 1.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "coord/excise_horizon_fraction must be in (0,1]"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bbh.puncture_excise_shrink_to_horizon &&
      !(bbh.puncture_excise_shrink_timescale > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "excise_shrink_to_horizon requires positive "
              << "coord/excise_shrink_timescale" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bbh.puncture_excise_to_horizon &&
      bbh.puncture_excise_shrink_to_horizon) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "excise_to_horizon and excise_shrink_to_horizon "
              << "are mutually exclusive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bbh.smooth_b_damping = pin->GetOrAddBoolean(
      "coord", "smooth_excision_b_damping", false);
  bbh.smooth_b_damping_eta = pin->GetOrAddReal(
      "coord", "smooth_excision_b_damping_eta", 0.0);
  bbh.smooth_b_damping_cfl = pin->GetOrAddReal(
      "coord", "smooth_excision_b_damping_cfl", 0.25);
  if (bbh.smooth_b_damping &&
      (!(bbh.smooth_b_damping_eta > 0.0) || bbh.smooth_b_damping_cfl < 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "smooth_excision_b_damping requires positive eta "
              << "and non-negative cfl cap" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bbh.unresolved_sink = pin->GetOrAddBoolean("problem", "unresolved_sink", false);
  bbh.sink_radius = pin->GetOrAddReal("problem", "sink_radius", 0.0);
  bbh.sink_width = pin->GetOrAddReal(
      "problem", "sink_width", bbh.sink_radius > 0.0 ? 0.25*bbh.sink_radius : -1.0);
  bbh.sink_timescale = pin->GetOrAddReal("problem", "sink_timescale", 0.0);
  bbh.sink_density_floor = pin->GetOrAddReal("problem", "sink_density_floor", -1.0);
  bbh.sink_pressure_floor = pin->GetOrAddReal("problem", "sink_pressure_floor", -1.0);
  bbh.sink_cells_per_radius = pin->GetOrAddReal(
      "problem", "sink_cells_per_radius", 10.0);
  bbh.sink_resolved_cells_across_horizon = pin->GetOrAddReal(
      "problem", "sink_resolved_cells_across_horizon", 20.0);
  if (bbh.unresolved_sink &&
      (!(bbh.sink_timescale > 0.0) || bbh.sink_radius < 0.0 ||
       bbh.sink_width == 0.0 || !(bbh.sink_cells_per_radius > 0.0) ||
       !(bbh.sink_resolved_cells_across_horizon > 0.0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "unresolved_sink requires positive sink_timescale, "
              << "sink_cells_per_radius, and sink_resolved_cells_across_horizon; "
              << "sink_radius must be non-negative and sink_width nonzero" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &coord_data = pmbp->pcoord->coord_data;
  if (coord_data.bh_excise &&
      coord_data.excision_scheme == ExcisionScheme::puncture) {
    const Real finest_dx = LocalFinestMeshSpacing(pmbp);
    const Real min_horizon = MinDynBBHHorizonRadius();
    if (finest_dx > min_horizon) {
      if (bbh.require_resolved_horizon) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "puncture excision is under-resolved: finest dx="
                  << finest_dx << " exceeds minimum horizon radius=" << min_horizon
                  << std::endl;
        std::exit(EXIT_FAILURE);
      } else if (global_variable::my_rank == 0) {
        std::cout << "WARNING: puncture excision is under-resolved: finest dx="
                  << finest_dx << " exceeds minimum horizon radius=" << min_horizon;
        if (bbh.unresolved_sink) {
          std::cout << "; resolution-scaled sink fallback is enabled";
        }
        std::cout << std::endl;
      }
    }
  }
  if (bbh.unresolved_sink) {
    if (pmbp->pmhd == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "unresolved_sink currently requires MHD" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    user_srcs = true;
    user_srcs_func = AddUnresolvedBHSink;
  }
  if (bbh.smooth_b_damping) {
    if (pmbp->pmhd == nullptr || !coord_data.bh_excise ||
        !coord_data.smooth_excise ||
        coord_data.excision_scheme != ExcisionScheme::puncture) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "smooth_excision_b_damping requires MHD and "
                << "smooth puncture excision" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    user_efield = true;
    user_efield_func = AddSmoothExcisionMagneticDamping;
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

  if (pmbp->prad != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh uses ADM background data and is incompatible "
              << "with legacy <radiation>; use <dyn_radiation> instead." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pdynrad != nullptr) {
    if (!(pmbp->pdynrad->use_adm_geometry)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "dynbbh <dyn_radiation> requires geometry='adm'."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmbp->pdynrad->are_units_enabled) {
      bbh.arad = pmbp->punit->rad_constant_cgs*
                 SQR(SQR(pmbp->punit->temperature_cgs()))/
                 pmbp->punit->pressure_cgs();
    } else {
      bbh.arad = pin->GetReal("dyn_radiation", "arad");
    }
    if (!(bbh.arad > 0.0) || !std::isfinite(bbh.arad)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "dynbbh <dyn_radiation> requires a positive finite "
                << "radiation constant." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    bbh.restart_seed_erad_fraction =
        pin->GetOrAddReal("dyn_radiation", "restart_seed_erad_fraction", 1.0);
    if (!(bbh.restart_seed_erad_fraction >= 0.0) ||
        !std::isfinite(bbh.restart_seed_erad_fraction)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "dynbbh dyn_radiation/restart_seed_erad_fraction "
                << "must be finite and non-negative." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pmbp->padm->SetADMVariables(pmbp);
    pmbp->pdynrad->PrepareADMGeometry();
    if (restart_missing_dynrad_i0) {
      if (pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "dynbbh restart activation of <dyn_radiation> "
                  << "requires Valencia GRMHD." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      post_restart_primitive_init_func = InitializeDynBBHRestartDynRadiation;
    }
  }

  if (restart) {
    // Coordinates/masks are not checkpoint payloads. Restore the current
    // puncture geometry before startup primitive recovery and timestep checks.
    pmbp->padm->SetADMVariables(pmbp);
    if (pmbp->pcoord->coord_data.bh_excise) pmbp->pcoord->UpdateExcisionMasks();
  }

  // The reconstructed problem generator initializes a Chakrabarti torus by
  // default.  Retain an explicit atmosphere mode for metric, excision, and
  // refinement regressions and for compatibility with earlier dynbbh inputs.
  const bool initialize_torus = pin->GetOrAddBoolean(
      "problem", "initialize_torus", true);
  if (!initialize_torus) {
    if (restart) return;
    const auto indcs = pmy_mesh_->mb_indcs;
    const int is = indcs.is; const int ie = indcs.ie;
    const int js = indcs.js; const int je = indcs.je;
    const int ks = indcs.ks; const int ke = indcs.ke;
    auto size = pmbp->pmb->mb_size.d_view;
    int nmb = pmbp->nmb_thispack;
    const Real atmosphere_dfloor = bbh.dfloor;
    const Real atmosphere_pfloor = bbh.pfloor;
    const Real test_bz_gradient = bbh.test_bz_gradient;

    if (pmbp->phydro != nullptr) {
      auto w0 = pmbp->phydro->w0;
      const int nscal = pmbp->phydro->nscalars;
      par_for("pgen_dynbbh_hydro_atmosphere", DevExeSpace(),
      0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        w0(m,IDN,k,j,i) = atmosphere_dfloor;
        w0(m,IVX,k,j,i) = 0.0;
        w0(m,IVY,k,j,i) = 0.0;
        w0(m,IVZ,k,j,i) = 0.0;
        w0(m,IPR,k,j,i) = atmosphere_pfloor;
        for (int r = 0; r < nscal; ++r) w0(m,IYF+r,k,j,i) = 0.0;
      });
      if (pmbp->padm == nullptr) {
        pmbp->phydro->peos->PrimToCons(
            w0, pmbp->phydro->u0, is, ie, js, je, ks, ke);
      }
    }

    if (pmbp->pmhd != nullptr) {
      auto w0 = pmbp->pmhd->w0;
      auto b0 = pmbp->pmhd->b0;
      auto bcc0 = pmbp->pmhd->bcc0;
      const int nscal = pmbp->pmhd->nscalars;
      par_for("pgen_dynbbh_mhd_atmosphere", DevExeSpace(),
      0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        w0(m,IDN,k,j,i) = atmosphere_dfloor;
        w0(m,IVX,k,j,i) = 0.0;
        w0(m,IVY,k,j,i) = 0.0;
        w0(m,IVZ,k,j,i) = 0.0;
        w0(m,IPR,k,j,i) = atmosphere_pfloor;
        for (int r = 0; r < nscal; ++r) w0(m,IYF+r,k,j,i) = 0.0;
        const Real x1v = CellCenterX(i-is, indcs.nx1,
                                     size(m).x1min, size(m).x1max);
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = test_bz_gradient*x1v;
        if (i == ie) b0.x1f(m,k,j,i+1) = 0.0;
        if (j == je) b0.x2f(m,k,j+1,i) = 0.0;
        if (k == ke) b0.x3f(m,k+1,j,i) = test_bz_gradient*x1v;
        bcc0(m,IBX,k,j,i) = 0.0;
        bcc0(m,IBY,k,j,i) = 0.0;
        bcc0(m,IBZ,k,j,i) = test_bz_gradient*x1v;
      });
      if (!pmbp->pcoord->is_dynamical_relativistic) {
        pmbp->pmhd->peos->PrimToCons(
            w0, bcc0, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
      }
    }

    if (pmbp->padm != nullptr) {
      pmbp->padm->SetADMVariables(pmbp);
      pmbp->pcoord->UpdateExcisionMasks();
      pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
    }
    if (pmbp->pdynrad != nullptr) {
      Kokkos::deep_copy(pmbp->pdynrad->i0, 0.0);
    }
    return;
  }

  if (restart) return;

  bbh.spin = 0.0;
  bbh.d = 1.0;
  const bool is_radiation_enabled = (pmbp->pdynrad != nullptr);

  // capture variables for the kernel
  const auto indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is; const int ie = indcs.ie;
  const int js = indcs.js; const int je = indcs.je;
  const int ks = indcs.ks; const int ke = indcs.ke;
  auto size = pmbp->pmb->mb_size.d_view;
  int nmb = pmbp->nmb_thispack;
  //auto bbh_ = bbh;
  auto &coord = pmbp->pcoord->coord_data;
  bool use_dyngr = (pmbp->pdyngr != nullptr);


  // copied form torus PG, needs to be rewritten?

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
  DvceArray6D<Real> norm_to_tet_;
  DvceArray4D<Real> sqrt_detg_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled) {
    nangles_ = pmbp->pdynrad->prgeo->nangles;
    nh_c_ = pmbp->pdynrad->nh_c;
    norm_to_tet_ = pmbp->pdynrad->norm_to_tet;
    sqrt_detg_c_ = pmbp->pdynrad->sqrt_detg_c;
    i0_ = pmbp->pdynrad->i0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    bbh.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bbh.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bbh.gamma_adi - 1.0;

  // global parameters
  bbh.rho_min = pin->GetReal("problem", "rho_min");
  bbh.rho_pow = pin->GetReal("problem", "rho_pow");
  bbh.pgas_min = pin->GetReal("problem", "pgas_min");
  bbh.pgas_pow = pin->GetReal("problem", "pgas_pow");
  bbh.psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * (M_PI/180.0);
  bbh.sin_psi = sin(bbh.psi);
  bbh.cos_psi = cos(bbh.psi);
  bbh.rho_max = pin->GetReal("problem", "rho_max");
  bbh.r_edge = pin->GetReal("problem", "r_edge");
  bbh.r_peak = pin->GetReal("problem", "r_peak");
  bbh.n_param = pin->GetOrAddReal("problem", "n_param",0.0);

  // local parameters
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);

  // excision parameters
  bbh.dexcise = coord.dexcise;
  bbh.pexcise = coord.pexcise;

  // Compute angular momentum and prepare constants describing primitives
  CalculateCN(bbh, &bbh.c_param, &bbh.n_param);
  bbh.l_peak = CalculateL(bbh, bbh.r_peak, 1.0);
  // Common to both tori:
  bbh.log_h_edge = LogHAux(bbh, bbh.r_edge, 1.0);
  bbh.log_h_peak = LogHAux(bbh, bbh.r_peak, 1.0) - bbh.log_h_edge;
  bbh.ptot_over_rho_peak = gm1/bbh.gamma_adi * (exp(bbh.log_h_peak)-1.0);
  bbh.rho_peak = pow(bbh.ptot_over_rho_peak, 1.0/gm1) / bbh.rho_max;

  // find "outer edge" of torus (first place log_h > 0)
  Real ra = bbh.r_peak;
  Real rb = 2. * ra;
  Real log_h_trial = LogHAux(bbh, rb, 1.) - bbh.log_h_edge;
  for (int iter=0; iter<10000; ++iter) {
    if (log_h_trial <= 0) {
      break;
    }
    rb *= 2.;
    log_h_trial = LogHAux(bbh, rb, 1.) - bbh.log_h_edge;
  }
  for (int iter=0; iter<10000; ++iter) {
    if (fabs(ra - rb) < 1.e-3) {
      break;
    }
    Real r_trial = (ra + rb) / 2.;
    if (LogHAux(bbh, r_trial, 1.) > bbh.log_h_edge) {
      ra = r_trial;
    } else {
      rb = r_trial;
    }
  }
  bbh.r_outer_edge = ra;
  std::cout << "Found torus outer edge: " << bbh.r_outer_edge << std::endl;

  // initialize primitive variables for new run ---------------------------------------

  auto trs = bbh;
  const bbh_metric_params metric = {
      bbh.a1_buffer, bbh.a2_buffer, bbh.cutoff_floor,
      bbh.metric_fd_step, bbh.metric_derivative_method};
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  Real bbh_traj_t0[NTRAJ];
  find_traj_t(0.0, bbh_traj_t0);

  Kokkos::parallel_reduce("pgen_torus1", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_ptot) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size(m).x1min;
    Real &x1max = size(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size(m).x2min;
    Real &x2max = size(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size(m).x3min;
    Real &x3max = size(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real &dx1 = size(m).dx1;
    Real &dx2 = size(m).dx2;
    Real &dx3 = size(m).dx3;

    // Extract metric and inverse -- presumably should get actual metric?????
    Real glower[4][4], gupper[4][4];
    GetSuperposedAndInverse(0.0, x1v, x2v, x3v, glower, gupper,
                            bbh_traj_t0, metric);

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
      log_h = LogHAux(trs, r, sin_vartheta)- trs.log_h_edge;  // (FM 3.6)
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
      GetSuperposedAndInverse(0.0, x1v, x2v, x3v, glower, gupper,
                              bbh_traj_t0, metric);

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
        const Real sqrt_detg = sqrt_detg_c_(m,k,j,i);
        Real intensity = 0.0;
        if (Kokkos::isfinite(n0_f) && n0_f > 0.0 &&
            Kokkos::isfinite(sqrt_detg) && sqrt_detg > 0.0 &&
            Kokkos::isfinite(urad) && urad >= 0.0) {
          intensity = sqrt_detg*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
          if (!(Kokkos::isfinite(intensity)) || intensity < 0.0) intensity = 0.0;
        }
        i0_(m,n,k,j,i) = intensity;
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

  // Initialize ADM variables -------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    pmbp->pcoord->UpdateExcisionMasks();
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  // initialize magnetic fields ---------------------------------------

  if (pmbp->pmhd != nullptr) {
    // parse some more parameters from input
    bbh.potential_beta_min = pin->GetOrAddReal("problem", "potential_beta_min", 100.0);
    bbh.potential_cutoff   = pin->GetOrAddReal("problem", "potential_cutoff", 0.2);

    bbh.is_vertical_field = pin->GetOrAddBoolean("problem", "vertical_field", false);

    bbh.potential_falloff  = pin->GetOrAddReal("problem", "potential_falloff", 0.0);
    bbh.potential_r_pow    = pin->GetOrAddReal("problem", "potential_r_pow", 0.0);
    bbh.potential_rho_pow  = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);

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
    auto trs = bbh;

    par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size(m).x1min;
      Real &x1max = size(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size(m).x2min;
      Real &x2max = size(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size(m).x3min;
      Real &x3max = size(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

      Real dx1 = size(m).dx1;
      Real dx2 = size(m).dx2;
      Real dx3 = size(m).dx3;

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
      Real dx1 = size(m).dx1;
      Real dx2 = size(m).dx2;
      Real dx3 = size(m).dx3;

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
      if (trs.test_bz_gradient != 0.0) {
        const Real x1v = CellCenterX(i-is, indcs.nx1,
                                     size(m).x1min, size(m).x1max);
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = trs.test_bz_gradient*x1v;
        if (i == ie) b0.x1f(m,k,j,i+1) = 0.0;
        if (j == je) b0.x2f(m,k,j+1,i) = 0.0;
        if (k == ke) b0.x3f(m,k+1,j,i) = trs.test_bz_gradient*x1v;
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
      Real &x1min = size(m).x1min;
      Real &x1max = size(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size(m).x2min;
      Real &x2max = size(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size(m).x3min;
      Real &x3max = size(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      GetSuperposedAndInverse(0.0, x1v, x2v, x3v, glower, gupper,
                              bbh_traj_t0, metric);

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
    Real bnorm = sqrt((ptotmax/(0.5*bsqmax))/trs.potential_beta_min);
    // Since vertical field extends beyond torus, normalize based on values in torus
    if (trs.is_vertical_field) {
      bnorm = sqrt((ptotmax/(0.5*bsqmax_intorus))/trs.potential_beta_min);
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

void InitializeDynBBHRestartDynRadiation(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pdynrad == nullptr || pmbp->pmhd == nullptr ||
      pmbp->pdyngr == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh restart radiation initialization requires "
              << "<dyn_radiation>, MHD, Valencia GRMHD, and ADM variables."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdynrad->PrepareADMGeometry();
  if (pmbp->pcoord->coord_data.bh_excise) {
    pmbp->pcoord->UpdateExcisionMasks();
  }
  if (pmbp->nmb_thispack == 0) return;

  const auto indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang1 = pmbp->pdynrad->prgeo->nangles - 1;

  auto w0 = pmbp->pmhd->w0;
  auto i0 = pmbp->pdynrad->i0;
  auto nh_c = pmbp->pdynrad->nh_c;
  auto norm_to_tet = pmbp->pdynrad->norm_to_tet;
  auto sqrt_detg_c = pmbp->pdynrad->sqrt_detg_c;
  auto adm_g_dd_c = pmbp->pdynrad->adm_g_dd_c;
  const bool use_excise = pmbp->pcoord->coord_data.bh_excise;
  auto excision_floor = pmbp->pcoord->excision_floor;

  const Real density_floor = 10.0*pmbp->pmhd->peos->eos_data.dfloor;
  const Real pressure_floor = pmbp->pmhd->peos->eos_data.pfloor;
  const Real arad = bbh.arad;
  const Real seed_erad_fraction = bbh.restart_seed_erad_fraction;

  par_for("dynbbh_restart_dynrad_i0", DevExeSpace(),
          0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real intensity = 0.0;
    bool seed_radiation = !(use_excise && excision_floor(m,k,j,i));

    const Real rho = w0(m,IDN,k,j,i);
    const Real pgas = w0(m,IPR,k,j,i);
    if (!(rho > density_floor) || !(pgas > pressure_floor) ||
        !(Kokkos::isfinite(rho)) || !(Kokkos::isfinite(pgas))) {
      seed_radiation = false;
    }

    if (seed_radiation) {
      const Real uu1 = w0(m,IVX,k,j,i);
      const Real uu2 = w0(m,IVY,k,j,i);
      const Real uu3 = w0(m,IVZ,k,j,i);
      const Real q = adm_g_dd_c(m,0,0,k,j,i)*uu1*uu1 +
                     2.0*adm_g_dd_c(m,0,1,k,j,i)*uu1*uu2 +
                     2.0*adm_g_dd_c(m,0,2,k,j,i)*uu1*uu3 +
                     adm_g_dd_c(m,1,1,k,j,i)*uu2*uu2 +
                     2.0*adm_g_dd_c(m,1,2,k,j,i)*uu2*uu3 +
                     adm_g_dd_c(m,2,2,k,j,i)*uu3*uu3;
      if (Kokkos::isfinite(q) && 1.0 + q > 0.0) {
        const Real uu0 = sqrt(1.0 + q);
        Real u_tet[4];
        for (int a=0; a<4; ++a) {
          u_tet[a] = norm_to_tet(m,a,0,k,j,i)*uu0 +
                     norm_to_tet(m,a,1,k,j,i)*uu1 +
                     norm_to_tet(m,a,2,k,j,i)*uu2 +
                     norm_to_tet(m,a,3,k,j,i)*uu3;
        }
        const Real n0_f = u_tet[0]*nh_c.d_view(n,0) -
                          u_tet[1]*nh_c.d_view(n,1) -
                          u_tet[2]*nh_c.d_view(n,2) -
                          u_tet[3]*nh_c.d_view(n,3);
        const Real temp = pgas/rho;
        const Real urad = seed_erad_fraction*arad*SQR(SQR(temp));
        if (Kokkos::isfinite(n0_f) && n0_f > 0.0 &&
            Kokkos::isfinite(urad) && urad >= 0.0) {
          intensity = sqrt_detg_c(m,k,j,i)*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
          if (!(Kokkos::isfinite(intensity)) || intensity < 0.0) intensity = 0.0;
        }
      }
    }
    i0(m,n,k,j,i) = intensity;
  });
  DevExeSpace().fence();
}

void SetADMVariablesToBBH(MeshBlockPack *pmbp) {
  const Real tt = pmbp->pcoord->coord_data.metric_time;
  auto adm = pmbp->padm->adm;
  auto size = pmbp->pmb->mb_size.d_view;
  const auto indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
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
  const bbh_metric_params metric = {
      bbh.a1_buffer, bbh.a2_buffer, bbh.cutoff_floor,
      bbh.metric_fd_step, bbh.metric_derivative_method};

  /* Load trajectories */

  /* Whether we load traj from a table or we compute analytical trajectories */
  find_traj_t_with_deriv(tt, bbh_traj_0, dbbh_traj_0);
  auto &coord = pmbp->pcoord->coord_data;
  coord.punc_0[0] = bbh_traj_0[X1];
  coord.punc_0[1] = bbh_traj_0[Y1];
  coord.punc_0[2] = bbh_traj_0[Z1];
  coord.punc_1[0] = bbh_traj_0[X2];
  coord.punc_1[1] = bbh_traj_0[Y2];
  coord.punc_1[2] = bbh_traj_0[Z2];
  const Real m1 = bbh_traj_0[M1T];
  const Real m2 = bbh_traj_0[M2T];
  coord.punc_0_spin[0] = m1*bbh_traj_0[AX1];
  coord.punc_0_spin[1] = m1*bbh_traj_0[AY1];
  coord.punc_0_spin[2] = m1*bbh_traj_0[AZ1];
  coord.punc_1_spin[0] = m2*bbh_traj_0[AX2];
  coord.punc_1_spin[1] = m2*bbh_traj_0[AY2];
  coord.punc_1_spin[2] = m2*bbh_traj_0[AZ2];
  coord.punc_0_vel[0] = bbh_traj_0[VX1];
  coord.punc_0_vel[1] = bbh_traj_0[VY1];
  coord.punc_0_vel[2] = bbh_traj_0[VZ1];
  coord.punc_1_vel[0] = bbh_traj_0[VX2];
  coord.punc_1_vel[1] = bbh_traj_0[VY2];
  coord.punc_1_vel[2] = bbh_traj_0[VZ2];
  // Host-side stage-time targets; no additional device work or global capture.
  const Real rH1 = bbh.puncture_excise_horizon_fraction*HorizonRadiusFromMassAndChi(
      m1, bbh_traj_0[AX1], bbh_traj_0[AY1], bbh_traj_0[AZ1]);
  const Real rH2 = bbh.puncture_excise_horizon_fraction*HorizonRadiusFromMassAndChi(
      m2, bbh_traj_0[AX2], bbh_traj_0[AY2], bbh_traj_0[AZ2]);
  coord.punc_0_rad = SmoothExcisionRadiusToHorizon(
      bbh.puncture_excise_rad1, rH1,
      tt-bbh.puncture_excise_shrink_start_time,
      bbh.puncture_excise_shrink_timescale,
      bbh.puncture_excise_to_horizon,
      bbh.puncture_excise_shrink_to_horizon);
  coord.punc_1_rad = SmoothExcisionRadiusToHorizon(
      bbh.puncture_excise_rad2, rH2,
      tt-bbh.puncture_excise_shrink_start_time,
      bbh.puncture_excise_shrink_timescale,
      bbh.puncture_excise_to_horizon,
      bbh.puncture_excise_shrink_to_horizon);
  if (bbh.puncture_excise_cap_to_horizon &&
      !bbh.puncture_excise_to_horizon &&
      !bbh.puncture_excise_shrink_to_horizon) {
    coord.punc_0_rad = std::min(coord.punc_0_rad, rH1);
    coord.punc_1_rad = std::min(coord.punc_1_rad, rH2);
  }
  Real hm = metric.metric_fd_step;
  Real hp = metric.metric_fd_step;
  if (bbh.use_traj_table) {
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
    Real &x1min = size(m).x1min;
    Real &x1max = size(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size(m).x2min;
    Real &x2max = size(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size(m).x3min;
    Real &x3max = size(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    struct three_metric met3;
    if (metric.metric_derivative_method == MetricDerivativeMethod::ad) {
      get_adm_and_derivatives_ad(tt, x1v, x2v, x3v, met3, bbh_traj_0,
                                 dbbh_traj_0, metric);
    } else {
      struct four_metric met4;
      numerical_4metric(tt, x1v, x2v, x3v, met4, bbh_traj_m1, bbh_traj_0,
                        bbh_traj_p1, hm, hp, metric);
      four_metric_to_three_metric(met4, met3);
    }

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
    const Real hm, const Real hp, const bbh_metric_params& metric) {
  struct four_metric met_m1;
  struct four_metric met_p1;
  const Real step = metric.metric_fd_step;

  // Time
  get_metric(t, x, y, z, outmet, nz_0, metric);
  if (hm > 0.0) get_metric(t-hm, x, y, z, met_m1, nz_m1, metric);
  if (hp > 0.0) get_metric(t+hp, x, y, z, met_p1, nz_p1, metric);
#define DT_METRIC(comp) \
  outmet.g_t.comp = (hm > 0.0 && hp > 0.0) ? \
      (-hp*met_m1.g.comp/(hm*(hm + hp)) \
       + (hp - hm)*outmet.g.comp/(hm*hp) \
       + hm*met_p1.g.comp/(hp*(hm + hp))) : \
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
  get_metric(t, x-step, y, z, met_m1, nz_0, metric);
  get_metric(t, x+step, y, z, met_p1, nz_0, metric);

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
  get_metric(t, x, y-step, z, met_m1, nz_0, metric);
  get_metric(t, x, y+step, z, met_p1, nz_0, metric);

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
  get_metric(t, x, y, z-step, met_m1, nz_0, metric);
  get_metric(t, x, y, z+step, met_p1, nz_0, metric);

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
                                struct three_metric &gam)
{
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
  /* This could occur during the transition to merger at certain points so here we restart to Minkowski */
  if (!(det > 0)) {
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
  const bool cached_is_last = (cached + 1 == times.size() - 1);
  // Use half-open intervals at interior knots so the derivative is selected
  // deterministically from the segment to the right.  The final segment is
  // closed at its upper endpoint so the last tabulated time remains valid.
  if (t >= times[cached] &&
      (t < times[cached + 1] ||
       (cached_is_last && t <= times[cached + 1]))) {
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
  const Real v1_sq = SQR(bbh_t[VX1]) + SQR(bbh_t[VY1]) + SQR(bbh_t[VZ1]);
  const Real v2_sq = SQR(bbh_t[VX2]) + SQR(bbh_t[VY2]) + SQR(bbh_t[VZ2]);
  if (!std::isfinite(v1_sq) || !std::isfinite(v2_sq) ||
      v1_sq >= 1.0 || v2_sq >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "interpolated trajectory velocity must be finite "
              << "and subluminal at time " << t << " in segment ["
              << times[i0] << ", " << times[i1] << "]: |v1|^2=" << v1_sq
              << ", |v2|^2=" << v2_sq << std::endl;
    std::exit(EXIT_FAILURE);
  }
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
    const T tr[NTRAJ], const bbh_metric_params& metric) {
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
  const T cutoff1 = spin1_norm*(T(1.0) + metric.a1_buffer) + metric.cutoff_floor;
  const T cutoff2 = spin2_norm*(T(1.0) + metric.a2_buffer) + metric.cutoff_floor;
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
    Real gcov[][NDIM], const Real traj_array[NTRAJ],
    const bbh_metric_params& metric) {
  (void)time;
  SuperposedBBHTemplate(x, y, z, gcov, traj_array, metric);
}

KOKKOS_INLINE_FUNCTION void FillMetricDerivatives(
    const dual2_real gcov[NDIM][NDIM], struct dd_sym &dg0,
    struct dd_sym &dg1) {
  dg0.tt = gcov[TT][TT].deriv0; dg1.tt = gcov[TT][TT].deriv1;
  dg0.tx = gcov[TT][XX].deriv0; dg1.tx = gcov[TT][XX].deriv1;
  dg0.ty = gcov[TT][YY].deriv0; dg1.ty = gcov[TT][YY].deriv1;
  dg0.tz = gcov[TT][ZZ].deriv0; dg1.tz = gcov[TT][ZZ].deriv1;
  dg0.xx = gcov[XX][XX].deriv0; dg1.xx = gcov[XX][XX].deriv1;
  dg0.xy = gcov[XX][YY].deriv0; dg1.xy = gcov[XX][YY].deriv1;
  dg0.xz = gcov[XX][ZZ].deriv0; dg1.xz = gcov[XX][ZZ].deriv1;
  dg0.yy = gcov[YY][YY].deriv0; dg1.yy = gcov[YY][YY].deriv1;
  dg0.yz = gcov[YY][ZZ].deriv0; dg1.yz = gcov[YY][ZZ].deriv1;
  dg0.zz = gcov[ZZ][ZZ].deriv0; dg1.zz = gcov[ZZ][ZZ].deriv1;
}

KOKKOS_INLINE_FUNCTION void FillMetricValue(
    const dual2_real gcov[NDIM][NDIM], struct dd_sym &g) {
  g.tt = gcov[TT][TT].val;
  g.tx = gcov[TT][XX].val;
  g.ty = gcov[TT][YY].val;
  g.tz = gcov[TT][ZZ].val;
  g.xx = gcov[XX][XX].val;
  g.xy = gcov[XX][YY].val;
  g.xz = gcov[XX][ZZ].val;
  g.yy = gcov[YY][YY].val;
  g.yz = gcov[YY][ZZ].val;
  g.zz = gcov[ZZ][ZZ].val;
}

KOKKOS_INLINE_FUNCTION void MetricDerivativesAD2(
    const Real x, const Real y, const Real z, const int direction0,
    const int direction1, const Real tr[NTRAJ], const Real dtr[NTRAJ],
    const bbh_metric_params& metric, struct dd_sym &dg0,
    struct dd_sym &dg1, struct dd_sym *g) {
  dual2_real xd(x, direction0 == 1 ? 1.0 : 0.0,
                direction1 == 1 ? 1.0 : 0.0);
  dual2_real yd(y, direction0 == 2 ? 1.0 : 0.0,
                direction1 == 2 ? 1.0 : 0.0);
  dual2_real zd(z, direction0 == 3 ? 1.0 : 0.0,
                direction1 == 3 ? 1.0 : 0.0);
  dual2_real trd[NTRAJ];
  for (int n = 0; n < NTRAJ; ++n) {
    trd[n] = dual2_real(tr[n], direction0 == 0 ? dtr[n] : 0.0,
                       direction1 == 0 ? dtr[n] : 0.0);
  }
  dual2_real gcov[NDIM][NDIM];
  SuperposedBBHTemplate(xd, yd, zd, gcov, trd, metric);
  if (g != nullptr) FillMetricValue(gcov, *g);
  FillMetricDerivatives(gcov, dg0, dg1);
}

KOKKOS_INLINE_FUNCTION void get_adm_and_derivatives_ad(
    const Real t, const Real x, const Real y, const Real z,
    struct three_metric &gam, const Real bbh_traj_loc[NTRAJ],
    const Real dbbh_traj_loc[NTRAJ], const bbh_metric_params& metric) {
  (void)t;
  struct dd_sym g;
  struct dd_sym dg0;
  struct dd_sym dg1;
  MetricDerivativesAD2(x, y, z, 0, 1, bbh_traj_loc, dbbh_traj_loc, metric,
                       dg0, dg1, &g);

  gam.gxx = g.xx;
  gam.gxy = g.xy;
  gam.gxz = g.xz;
  gam.gyy = g.yy;
  gam.gyz = g.yz;
  gam.gzz = g.zz;
  const Real det = adm::SpatialDet(gam.gxx, gam.gxy, gam.gxz,
                                   gam.gyy, gam.gyz, gam.gzz);
  if (!(det > 0.0)) {
    gam.gxx = 1.0;
    gam.gxy = 0.0;
    gam.gxz = 0.0;
    gam.gyy = 1.0;
    gam.gyz = 0.0;
    gam.gzz = 1.0;
    gam.alpha = 1.0;
    gam.betax = 0.0;
    gam.betay = 0.0;
    gam.betaz = 0.0;
    gam.kxx = 0.0;
    gam.kxy = 0.0;
    gam.kxz = 0.0;
    gam.kyy = 0.0;
    gam.kyz = 0.0;
    gam.kzz = 0.0;
    return;
  }

  const Real idetgxx = -gam.gyz*gam.gyz + gam.gyy*gam.gzz;
  const Real idetgxy = gam.gxz*gam.gyz - gam.gxy*gam.gzz;
  const Real idetgxz = -gam.gxz*gam.gyy + gam.gxy*gam.gyz;
  const Real idetgyy = -gam.gxz*gam.gxz + gam.gxx*gam.gzz;
  const Real idetgyz = gam.gxy*gam.gxz - gam.gxx*gam.gyz;
  const Real idetgzz = -gam.gxy*gam.gxy + gam.gxx*gam.gyy;
  const Real invgxx = idetgxx/det;
  const Real invgxy = idetgxy/det;
  const Real invgxz = idetgxz/det;
  const Real invgyy = idetgyy/det;
  const Real invgyz = idetgyz/det;
  const Real invgzz = idetgzz/det;

  gam.betax = g.tx*invgxx + g.ty*invgxy + g.tz*invgxz;
  gam.betay = g.tx*invgxy + g.ty*invgyy + g.tz*invgyz;
  gam.betaz = g.tx*invgxz + g.ty*invgyz + g.tz*invgzz;
  const Real b2 = g.tx*gam.betax + g.ty*gam.betay + g.tz*gam.betaz;
  gam.alpha = sqrt(fabs(b2 - g.tt));

  // Accumulate the six K_ij numerators from two paired dual evaluations rather
  // than retaining a four_metric with all 40 derivative components.
  Real kxx = dg0.xx;
  Real kxy = dg0.xy;
  Real kxz = dg0.xz;
  Real kyy = dg0.yy;
  Real kyz = dg0.yz;
  Real kzz = dg0.zz;

  kxx += -2.0*dg1.tx + gam.betax*dg1.xx + 2.0*gam.betay*dg1.xy
       + 2.0*gam.betaz*dg1.xz;
  kxy += -dg1.ty + gam.betay*dg1.yy + gam.betaz*dg1.yz;
  kxz += -dg1.tz + gam.betay*dg1.yz + gam.betaz*dg1.zz;
  kyy += -gam.betax*dg1.yy;
  kyz += -gam.betax*dg1.yz;
  kzz += -gam.betax*dg1.zz;

  MetricDerivativesAD2(x, y, z, 2, 3, bbh_traj_loc, dbbh_traj_loc, metric,
                       dg0, dg1, nullptr);
  kxx += -gam.betay*dg0.xx;
  kxy += -dg0.tx + gam.betax*dg0.xx + gam.betaz*dg0.xz;
  kxz += -gam.betay*dg0.xz;
  kyy += -2.0*dg0.ty + 2.0*gam.betax*dg0.xy + gam.betay*dg0.yy
       + 2.0*gam.betaz*dg0.yz;
  kyz += -dg0.tz + gam.betax*dg0.xz + gam.betaz*dg0.zz;
  kzz += -gam.betay*dg0.zz;

  kxx += -gam.betaz*dg1.xx;
  kxy += -gam.betaz*dg1.xy;
  kxz += -dg1.tx + gam.betax*dg1.xx + gam.betay*dg1.xy;
  kyy += -gam.betaz*dg1.yy;
  kyz += -dg1.ty + gam.betax*dg1.xy + gam.betay*dg1.yy;
  kzz += -2.0*dg1.tz + 2.0*gam.betax*dg1.xz
       + 2.0*gam.betay*dg1.yz + gam.betaz*dg1.zz;

  const Real factor = -0.5/gam.alpha;
  gam.kxx = factor*kxx;
  gam.kxy = factor*kxy;
  gam.kxz = factor*kxz;
  gam.kyy = factor*kyy;
  gam.kyz = factor*kyz;
  gam.kzz = factor*kzz;
}

KOKKOS_INLINE_FUNCTION
void get_metric(const Real t,
	       	const Real x,
	       	const Real y,
	       	const Real z,
	       	struct four_metric &met,
          const Real bbh_traj_loc[NTRAJ], const bbh_metric_params& metric)
{
  Real gcov[NDIM][NDIM];

  SuperposedBBH(t, x, y, z, gcov, bbh_traj_loc, metric);

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
  auto refine_flag_d = refine_flag.d_view;
  const auto indcs = pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto u0 = pmbp->padm->u_adm;
  const int I_ADM_ALPHA = pmbp->padm->I_ADM_ALPHA;
  const Real alpha_threshold = bbh.alpha_thr;
  const Real alpha_hysteresis = bbh_ref.hysteresis;

  par_for_outer(
  "AMR::AlphaMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
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

    if (team_dmin < alpha_threshold) {
      refine_flag_d(m + mbs) = 1;
    } else if (team_dmin > alpha_hysteresis * alpha_threshold &&
               refine_flag_d(m + mbs) <= 0) {
      refine_flag_d(m + mbs) = -1;
    }
  });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

Real PointToBlockDistanceSquared(const Real point[3], const RegionSize &block,
                                 const bool multi_d, const bool three_d) {
  const Real dx1 = std::max({block.x1min - point[0], 0.0,
                             point[0] - block.x1max});
  const Real dx2 = multi_d ? std::max({block.x2min - point[1], 0.0,
                                       point[1] - block.x2max}) : 0.0;
  const Real dx3 = three_d ? std::max({block.x3min - point[2], 0.0,
                                       point[2] - block.x3max}) : 0.0;
  return SQR(dx1) + SQR(dx2) + SQR(dx3);
}

void RefineSpatialPolicy(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  const bool use_tracker = bbh_ref.base == RefineBasePolicy::tracker;
  const bool has_spatial_policy = use_tracker || !bbh_ref.com_radius.empty();
  if (!has_spatial_policy) return;

  Real trajectory[NTRAJ];
  find_traj_t(pmesh->time, trajectory);
  const Real hole1[3] = {trajectory[X1], trajectory[Y1], trajectory[Z1]};
  const Real hole2[3] = {trajectory[X2], trajectory[Y2], trajectory[Z2]};
  const Real origin[3] = {0.0, 0.0, 0.0};
  const int maximum_level = pmesh->max_level - pmesh->root_level;

  auto configured_target = [=](const int requested) {
    return requested < 0 ? maximum_level : requested;
  };

  for (int m = 0; m < nmb; ++m) {
    const int gid = m + mbs;
    const int level = pmesh->lloc_eachmb[gid].level - pmesh->root_level;
    const RegionSize &block = size.h_view(m);
    int hard_target = -1;
    int buffer_target = -1;

    auto add_region = [&](const Real center[3], const Real radius,
                          const int requested_level) {
      const Real distance2 = PointToBlockDistanceSquared(
          center, block, pmesh->multi_d, pmesh->three_d);
      const int target = configured_target(requested_level);
      if (distance2 < SQR(radius)) {
        hard_target = std::max(hard_target, target);
      }
      if (distance2 < SQR(bbh_ref.hysteresis*radius)) {
        buffer_target = std::max(buffer_target, target);
      }
    };

    if (use_tracker) {
      add_region(hole1, bbh_ref.tracker_radius[0],
                 bbh_ref.tracker_reflevel[0]);
      add_region(hole2, bbh_ref.tracker_radius[1],
                 bbh_ref.tracker_reflevel[1]);
    }
    for (std::size_t nr = 0; nr < bbh_ref.com_radius.size(); ++nr) {
      add_region(origin, bbh_ref.com_radius[nr], bbh_ref.com_reflevel[nr]);
    }

    int &flag = refine_flag.h_view(gid);
    if (hard_target >= 0) {
      if (level < hard_target) {
        flag = 1;
      } else if (level == hard_target) {
        if (flag < 0) flag = 0;
      } else if (flag <= 0) {
        flag = -1;
      }
    } else if (buffer_target >= level && flag <= 0) {
      // Retain the requested resolution until the block leaves the expanded
      // region.  This avoids refine/derefine chatter as a puncture crosses a
      // MeshBlock boundary.
      flag = 0;
    } else if (flag <= 0) {
      flag = -1;
    }
  }

  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// 1 refines, -1 derefines, and 0 leaves a MeshBlock unchanged.
void Refine(MeshBlockPack *pmbp) {
  if (bbh_ref.base == RefineBasePolicy::alpha_min) {
    RefineAlphaMin(pmbp);
  }
  RefineSpatialPolicy(pmbp);
}


KOKKOS_INLINE_FUNCTION
Real SinkSmoothStep01(const Real x) {
  Real s = fmin(1.0, fmax(0.0, x));
  return s*s*(3.0 - 2.0*s);
}

void AddUnresolvedBHSink(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (!bbh.unresolved_sink || pmbp->pmhd == nullptr ||
      pmbp->padm == nullptr || !(bdt > 0.0)) {
    return;
  }

  const auto indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int nscal = pmbp->pmhd->nscalars;
  int nmhd = pmbp->pmhd->nmhd;
  auto size = pmbp->pmb->mb_size.d_view;
  auto u0 = pmbp->pmhd->u0;
  auto eos = pmbp->pmhd->peos->eos_data;
  const Real rho_target = (bbh.sink_density_floor > 0.0) ?
      bbh.sink_density_floor : eos.dfloor;
  const Real p_target = (bbh.sink_pressure_floor > 0.0) ?
      bbh.sink_pressure_floor : eos.pfloor;
  const Real e_target = p_target/(eos.gamma - 1.0);
  const Real tau = bbh.sink_timescale;
  Real q[NTRAJ];
  find_traj_t(pmbp->pcoord->coord_data.metric_time, q);
  const bbh_sink_state state = ComputeUnresolvedSinkState(pmbp, q);

  par_for("dynbbh_unresolved_sink", DevExeSpace(),
          0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, indcs.nx1,
                         size(m).x1min, size(m).x1max);
    Real y = CellCenterX(j-js, indcs.nx2,
                         size(m).x2min, size(m).x2max);
    Real z = CellCenterX(k-ks, indcs.nx3,
                         size(m).x3min, size(m).x3max);
    Real weight = 0.0;
    if (state.hole1.active) {
      Real radius = sqrt(SQR(x-state.hole1.x) + SQR(y-state.hole1.y) +
                         SQR(z-state.hole1.z));
      weight = fmax(weight, SinkSmoothStep01(
          (state.hole1.sink_radius-radius)/state.hole1.sink_width));
    }
    if (state.hole2.active) {
      Real radius = sqrt(SQR(x-state.hole2.x) + SQR(y-state.hole2.y) +
                         SQR(z-state.hole2.z));
      weight = fmax(weight, SinkSmoothStep01(
          (state.hole2.sink_radius-radius)/state.hole2.sink_width));
    }
    if (!(weight > 0.0) || !Kokkos::isfinite(weight)) return;
    Real damp = 1.0-exp(-bdt*weight/tau);
    Real keep = 1.0-damp;
    u0(m,IDN,k,j,i) = keep*u0(m,IDN,k,j,i) + damp*rho_target;
    u0(m,IM1,k,j,i) *= keep;
    u0(m,IM2,k,j,i) *= keep;
    u0(m,IM3,k,j,i) *= keep;
    u0(m,IEN,k,j,i) = keep*u0(m,IEN,k,j,i) + damp*e_target;
    for (int n = 0; n < nscal; ++n) u0(m,nmhd+n,k,j,i) *= keep;
  });
}

KOKKOS_INLINE_FUNCTION
Real SmoothExcisionBWeight(Real w) {
  return fmin(fmax(w, 0.0), 1.0);
}

KOKKOS_INLINE_FUNCTION
Real StrictSmoothExcisionBWeight(const Real w0, const Real w1) {
  if (!Kokkos::isfinite(w0) || !Kokkos::isfinite(w1)) return 0.0;
  return SmoothExcisionBWeight(fmin(w0, w1));
}

KOKKOS_INLINE_FUNCTION
Real StrictSmoothExcisionBWeight(const Real w0, const Real w1,
                                 const Real w2, const Real w3) {
  if (!Kokkos::isfinite(w0) || !Kokkos::isfinite(w1) ||
      !Kokkos::isfinite(w2) || !Kokkos::isfinite(w3)) {
    return 0.0;
  }
  return SmoothExcisionBWeight(fmin(fmin(w0, w1), fmin(w2, w3)));
}

template <typename WeightView>
KOKKOS_INLINE_FUNCTION Real EdgeWeightX1D(const WeightView &w, const int m,
                                          const int k, const int j, const int i) {
  return StrictSmoothExcisionBWeight(w(m,k,j,i), w(m,k,j,i-1));
}

template <typename WeightView>
KOKKOS_INLINE_FUNCTION Real EdgeWeightX1(const WeightView &w, const int m, const int k,
                                         const int j, const int i) {
  return StrictSmoothExcisionBWeight(w(m,k,j,i), w(m,k,j-1,i),
                                     w(m,k-1,j,i), w(m,k-1,j-1,i));
}

template <typename WeightView>
KOKKOS_INLINE_FUNCTION Real EdgeWeightX2(const WeightView &w, const int m, const int k,
                                         const int j, const int i) {
  return StrictSmoothExcisionBWeight(w(m,k,j,i), w(m,k-1,j,i),
                                     w(m,k,j,i-1), w(m,k-1,j,i-1));
}

template <typename WeightView>
KOKKOS_INLINE_FUNCTION Real EdgeWeightX3(const WeightView &w, const int m, const int k,
                                         const int j, const int i) {
  return StrictSmoothExcisionBWeight(w(m,k,j,i), w(m,k,j-1,i),
                                     w(m,k,j,i-1), w(m,k,j-1,i-1));
}

KOKKOS_INLINE_FUNCTION
Real SmoothExcisionDampingEta(const RegionSize &size, const bool multi_d,
                              const bool three_d, const Real eta0,
                              const Real cfl_cap, const Real dt) {
  Real eta = eta0;
  if (cfl_cap > 0.0 && dt > 0.0) {
    Real dxmin = size.dx1;
    if (multi_d) dxmin = fmin(dxmin, size.dx2);
    if (three_d) dxmin = fmin(dxmin, size.dx3);
    eta = fmin(eta, cfl_cap*SQR(dxmin)/dt);
  }
  return eta;
}

//! \fn void AddSmoothExcisionMagneticDamping(Mesh *pm, DvceEdgeFld4D<Real> &efld)
//! \brief Add a smooth-excision resistive EMF, E_damp = eta W curl(B), at edges.
void AddSmoothExcisionMagneticDamping(Mesh *pm, DvceEdgeFld4D<Real> &efld) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr || !(bbh.smooth_b_damping_eta > 0.0)) return;

  const auto indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmbp->nmb_thispack - 1;
  auto b0 = pmbp->pmhd->b0;
  auto weight = pmbp->pcoord->excision_weight;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto mbsize = pmbp->pmb->mb_size.d_view;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;
  const Real eta0 = bbh.smooth_b_damping_eta;
  const Real cfl_cap = bbh.smooth_b_damping_cfl;
  const Real dt = pm->dt;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  if (pm->one_d) {
    par_for_outer("dynbbh_b_damp1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);
      auto size = mbsize(m);
      Real eta = SmoothExcisionDampingEta(size, multi_d, three_d, eta0, cfl_cap, dt);
	      CurrentDensity(member, m, ks, js, is, ie+1, b0, size, j1, j2, j3);
	      par_for_inner(member, is, ie+1, [&](const int i) {
	        Real w = EdgeWeightX1D(weight, m, ks, js, i);
	        if (w > 0.0 && Kokkos::isfinite(w)) {
	          Real damp = eta*w;
	          e2(m,ks,  js,i) += damp*j2(i);
	          e2(m,ke+1,js,i) += damp*j2(i);
	          e3(m,ks,  js,i) += damp*j3(i);
	          e3(m,ks,je+1,i) += damp*j3(i);
	        }
	      });
	    });
    return;
  }

  if (pm->two_d) {
    par_for_outer("dynbbh_b_damp2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);
      auto size = mbsize(m);
      Real eta = SmoothExcisionDampingEta(size, multi_d, three_d, eta0, cfl_cap, dt);
	      CurrentDensity(member, m, ks, j, is, ie+1, b0, size, j1, j2, j3);
	      par_for_inner(member, is, ie+1, [&](const int i) {
	        Real w1 = StrictSmoothExcisionBWeight(weight(m,ks,j,i),
	                                             weight(m,ks,j-1,i));
	        if (w1 > 0.0 && Kokkos::isfinite(w1)) {
	          e1(m,ks,  j,i) += eta*w1*j1(i);
	          e1(m,ke+1,j,i) += eta*w1*j1(i);
	        }
	        Real w2 = EdgeWeightX1D(weight, m, ks, j, i);
	        if (w2 > 0.0 && Kokkos::isfinite(w2)) {
	          e2(m,ks,  j,i) += eta*w2*j2(i);
	          e2(m,ke+1,j,i) += eta*w2*j2(i);
	        }
	        Real w3 = EdgeWeightX3(weight, m, ks, j, i);
	        if (w3 > 0.0 && Kokkos::isfinite(w3)) {
	          e3(m,ks,  j,i) += eta*w3*j3(i);
	        }
	      });
	    });
    return;
  }

  par_for_outer("dynbbh_b_damp3", DevExeSpace(), scr_size, scr_level,
                0, nmb1, ks, ke+1, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);
    auto size = mbsize(m);
	    Real eta = SmoothExcisionDampingEta(size, multi_d, three_d, eta0, cfl_cap, dt);
	    CurrentDensity(member, m, k, j, is, ie+1, b0, size, j1, j2, j3);
	    par_for_inner(member, is, ie+1, [&](const int i) {
	      Real w1 = EdgeWeightX1(weight, m, k, j, i);
	      if (w1 > 0.0 && Kokkos::isfinite(w1)) {
	        e1(m,k,j,i) += eta*w1*j1(i);
	      }
	      Real w2 = EdgeWeightX2(weight, m, k, j, i);
	      if (w2 > 0.0 && Kokkos::isfinite(w2)) {
	        e2(m,k,j,i) += eta*w2*j2(i);
	      }
	      Real w3 = EdgeWeightX3(weight, m, k, j, i);
	      if (w3 > 0.0 && Kokkos::isfinite(w3)) {
	        e3(m,k,j,i) += eta*w3*j3(i);
	      }
	    });
	  });
	}
//nere hardcoding zero spin
KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(const bbh_pgen& pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  //Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  //Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
  //                    + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  //*pr = r;
  //*ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  //*pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
  //        pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));

  Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2, r*x1);
  return;
}


//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

// Needs to be updated to use actual metric?

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(const bbh_pgen& pgen, Real r, Real sin_theta, Real l) {
  // Compute BL metric components
  Real sigma = SQR(r);
  Real g_00 = -1.0 + 2.0*r/sigma;
  Real g_03 = 0.0;
  Real g_33 = SQR(r)*SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  Real u_t = -sqrt(fmax((SQR(g_03) - g_00*g_33)/(g_33 + 2.0*l*g_03 + SQR(l)*g_00), 0.0));
  return u_t;
}

//----------------------------------------------------------------------------------------
// Function to calculate enthalpy in Chakrabarti torus
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Chakrabarti, S. 1985, ApJ 288, 1

KOKKOS_INLINE_FUNCTION
static Real LogHAux(const bbh_pgen& pgen, Real r, Real sin_theta) {
  Real logh;
  // Chakrabarti
  Real l = CalculateL(pgen, r, sin_theta);
  Real u_t = CalculateCovariantUT(pgen, r, sin_theta, l);
  Real l_edge = CalculateL(pgen, pgen.r_edge, 1.0);
  Real u_t_edge = CalculateCovariantUT(pgen, pgen.r_edge, 1.0, l_edge);
  Real hh = u_t_edge/u_t;
  if (pgen.n_param==1.0) {
    hh *= pow(l_edge/l, SQR(pgen.c_param)/(SQR(pgen.c_param)-1.0));
  } else {
    Real pow_c = 2.0/pgen.n_param;
    Real pow_l = 2.0-2.0/pgen.n_param;
    Real pow_abs = pgen.n_param/(2.0-2.0*pgen.n_param);
    hh *= (pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l   , pow_l)), pow_abs) *
          pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l_edge, pow_l)), -1.0*pow_abs));
  }
  if (Kokkos::isfinite(hh) && hh >= 1.0) {
    logh = log(hh);
  } else {
    logh = -1.0;
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
static Real CalculateT(const bbh_pgen& pgen, Real rho, Real ptot_over_rho) {
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
static void CalculateCN(const bbh_pgen& pgen, Real *cparam, Real *nparam) {
  Real n_input = pgen.n_param;
  Real nn; // slope of angular momentum profile
  Real cc; // constant of angular momentum profile
  Real l_edge = SQR(pgen.r_edge)/(sqrt(pgen.r_edge)*(pgen.r_edge - 2.0));
  Real l_peak = SQR(pgen.r_peak)/(sqrt(pgen.r_peak)*(pgen.r_peak - 2.0));
  Real lambda_edge = sqrt((l_edge*(SQR(pgen.r_edge)*pgen.r_edge))/(l_edge*(pgen.r_edge - 2.0)));
  Real lambda_peak = sqrt((l_peak*(SQR(pgen.r_peak)*pgen.r_peak))/(l_peak*(pgen.r_peak - 2.0)));
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
// N.B. Here assumer zero spin
KOKKOS_INLINE_FUNCTION
static Real CalculateL(const bbh_pgen& pgen, Real r, Real sin_theta) {
  // Compute BL metric components
  Real sigma = SQR(r);
  Real g_00 = -1.0 + 2.0*r/sigma;
  Real g_03 = 0.0;
  Real g_33 = sigma*SQR(sin_theta);

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
    Real residual = pow(l_val/pgen.c_param, 2.0/pgen.n_param) + l_val*g_33/(l_val*g_00);
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

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotentialInTiltedTorus(const bbh_pgen& pgen,
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

KOKKOS_INLINE_FUNCTION
Real A1(const bbh_pgen& pgen, Real x1, Real x2, Real x3) {
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
  //return -0.5*x2;
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(const bbh_pgen& pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  //Real r, theta, phi;
  //GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
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
  //return 0.5*x1;
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(const bbh_pgen& pgen, Real x1, Real x2, Real x3) {
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


  return 0.0;
}

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(const bbh_pgen& pgen,
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


KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(const bbh_pgen& pgen,
                                    Real r, Real sin_theta, Real *pu0, Real *pu3) {
  // Compute BL metric components
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*r;              // \Delta
  Real sigma = SQR(r);         // \Sigma
  Real aa = SQR(SQR(r));  // A
  Real g_00 = -(1.0 - 2.0*r/sigma); // g_tt
  //Real g_03 = 0.0;
  Real g_33 = sigma; // g_pp
  Real g00 = -aa/(delta*sigma); // g^tt
  //Real g03 = 0.0;  // g^tp

  Real u0 = 0.0, u3 = 0.0;
  // Compute non-zero components of 4-velocity
  // Chakrabarti torus
  Real l = CalculateL(pgen, r, sin_theta);
  Real u_0 = CalculateCovariantUT(pgen, r, sin_theta, l); // u_t
  Real omega = -l*g_00/g_33;
  u0 = g00*u_0; // u^t
  u3 = omega * u0; // u^p

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
static void TransformVector(const bbh_pgen& pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) + sqrt(SQR(SQR(rad))) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r;
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1)/(SQR(r))) +
         a2_bl * x1*x3/r * sqrt((SQR(r))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2)/(SQR(r))) +
         a2_bl * x2*x3/r * sqrt((SQR(r))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) ));
  return;
}


KOKKOS_INLINE_FUNCTION
static void GetSuperposedAndInverse(const Real t,
                            const Real x, const Real y, const Real z,
                            Real gcov[][NDIM], Real gcon[][NDIM], const Real bbh_traj_loc[NTRAJ],
                            const bbh_metric_params& metric){
  //Real gcov[NDIM][NDIM];
  //Real gcon[NDIM][NDIM];
  SuperposedBBH(t, x, y, z, gcov, bbh_traj_loc, metric);
  InvertMetric(gcov, gcon);

  return;
}


KOKKOS_INLINE_FUNCTION
static void InvertMetric(Real gcov[][NDIM], Real gcon[][NDIM]){
  Real A2323 = gcov[YY][YY] * gcov[ZZ][ZZ] - gcov[YY][ZZ] * gcov[ZZ][YY] ;
  Real A1323 = gcov[YY][XX] * gcov[ZZ][ZZ] - gcov[YY][ZZ] * gcov[ZZ][XX] ;
  Real A1223 = gcov[YY][XX] * gcov[ZZ][YY] - gcov[YY][YY] * gcov[ZZ][XX] ;
  Real A0323 = gcov[YY][TT] * gcov[ZZ][ZZ] - gcov[YY][ZZ] * gcov[ZZ][TT] ;
  Real A0223 = gcov[YY][TT] * gcov[ZZ][YY] - gcov[YY][YY] * gcov[ZZ][TT] ;
  Real A0123 = gcov[YY][TT] * gcov[ZZ][XX] - gcov[YY][XX] * gcov[ZZ][TT] ;
  Real A2313 = gcov[XX][YY] * gcov[ZZ][ZZ] - gcov[XX][ZZ] * gcov[ZZ][YY] ;
  Real A1313 = gcov[XX][XX] * gcov[ZZ][ZZ] - gcov[XX][ZZ] * gcov[ZZ][XX] ;
  Real A1213 = gcov[XX][XX] * gcov[ZZ][YY] - gcov[XX][YY] * gcov[ZZ][XX] ;
  Real A2312 = gcov[XX][YY] * gcov[YY][ZZ] - gcov[XX][ZZ] * gcov[YY][YY] ;
  Real A1312 = gcov[XX][XX] * gcov[YY][ZZ] - gcov[XX][ZZ] * gcov[YY][XX] ;
  Real A1212 = gcov[XX][XX] * gcov[YY][YY] - gcov[XX][YY] * gcov[YY][XX] ;
  Real A0313 = gcov[XX][TT] * gcov[ZZ][ZZ] - gcov[XX][ZZ] * gcov[ZZ][TT] ;
  Real A0213 = gcov[XX][TT] * gcov[ZZ][YY] - gcov[XX][YY] * gcov[ZZ][TT] ;
  Real A0312 = gcov[XX][TT] * gcov[YY][ZZ] - gcov[XX][ZZ] * gcov[YY][TT] ;
  Real A0212 = gcov[XX][TT] * gcov[YY][YY] - gcov[XX][YY] * gcov[YY][TT] ;
  Real A0113 = gcov[XX][TT] * gcov[ZZ][XX] - gcov[XX][XX] * gcov[ZZ][TT] ;
  Real A0112 = gcov[XX][TT] * gcov[YY][XX] - gcov[XX][XX] * gcov[YY][TT] ;

  Real det = gcov[TT][TT] * ( gcov[XX][XX] * A2323 - gcov[XX][YY] * A1323 + gcov[XX][ZZ] * A1223 )
    - gcov[TT][XX] * ( gcov[XX][TT] * A2323 - gcov[XX][YY] * A0323 + gcov[XX][ZZ] * A0223 )
    + gcov[TT][YY] * ( gcov[XX][TT] * A1323 - gcov[XX][XX] * A0323 + gcov[XX][ZZ] * A0123 )
    - gcov[TT][ZZ] * ( gcov[XX][TT] * A1223 - gcov[XX][XX] * A0223 + gcov[XX][YY] * A0123 ) ;
  det = 1 / det;

   gcon[TT][TT] = det *   ( gcov[XX][XX] * A2323 - gcov[XX][YY] * A1323 + gcov[XX][ZZ] * A1223 );
   gcon[TT][XX] = det * - ( gcov[TT][XX] * A2323 - gcov[TT][YY] * A1323 + gcov[TT][ZZ] * A1223 );
   gcon[TT][YY] = det *   ( gcov[TT][XX] * A2313 - gcov[TT][YY] * A1313 + gcov[TT][ZZ] * A1213 );
   gcon[TT][ZZ] = det * - ( gcov[TT][XX] * A2312 - gcov[TT][YY] * A1312 + gcov[TT][ZZ] * A1212 );
   gcon[XX][TT] = det * - ( gcov[XX][TT] * A2323 - gcov[XX][YY] * A0323 + gcov[XX][ZZ] * A0223 );
   gcon[XX][XX] = det *   ( gcov[TT][TT] * A2323 - gcov[TT][YY] * A0323 + gcov[TT][ZZ] * A0223 );
   gcon[XX][YY] = det * - ( gcov[TT][TT] * A2313 - gcov[TT][YY] * A0313 + gcov[TT][ZZ] * A0213 );
   gcon[XX][ZZ] = det *   ( gcov[TT][TT] * A2312 - gcov[TT][YY] * A0312 + gcov[TT][ZZ] * A0212 );
   gcon[YY][TT] = det *   ( gcov[XX][TT] * A1323 - gcov[XX][XX] * A0323 + gcov[XX][ZZ] * A0123 );
   gcon[YY][XX] = det * - ( gcov[TT][TT] * A1323 - gcov[TT][XX] * A0323 + gcov[TT][ZZ] * A0123 );
   gcon[YY][YY] = det *   ( gcov[TT][TT] * A1313 - gcov[TT][XX] * A0313 + gcov[TT][ZZ] * A0113 );
   gcon[YY][ZZ] = det * - ( gcov[TT][TT] * A1312 - gcov[TT][XX] * A0312 + gcov[TT][ZZ] * A0112 );
   gcon[ZZ][TT] = det * - ( gcov[XX][TT] * A1223 - gcov[XX][XX] * A0223 + gcov[XX][YY] * A0123 );
   gcon[ZZ][XX] = det *   ( gcov[TT][TT] * A1223 - gcov[TT][XX] * A0223 + gcov[TT][YY] * A0123 );
   gcon[ZZ][YY] = det * - ( gcov[TT][TT] * A1213 - gcov[TT][XX] * A0213 + gcov[TT][YY] * A0113 );
   gcon[ZZ][ZZ] = det *   ( gcov[TT][TT] * A1212 - gcov[TT][XX] * A0212 + gcov[TT][YY] * A0112 );

   return;
}


}//namespace
