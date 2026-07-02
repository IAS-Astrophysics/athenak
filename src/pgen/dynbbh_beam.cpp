#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
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
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/current_density.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "dyn_radiation/dyn_radiation.hpp"
#include "particles/particles.hpp"
#include "utils/flux_generalized.hpp"
#include "units/units.hpp"
#include "srcterms/ismcooling.hpp"



#include <Kokkos_Random.hpp>


namespace {

enum {
  TT, XX, YY, ZZ, NDIM
};

enum {
  X1, Y1, Z1, X2, Y2, Z2,
  VX1, VY1, VZ1, VX2, VY2, VZ2,
  AX1, AY1, AZ1, AX2, AY2, AZ2,
  M1T, M2T, NTRAJ
};

constexpr Real metric_fd_step = 5.0e-5;

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

KOKKOS_INLINE_FUNCTION Real metric_sqrt(Real a) { return sqrt(a); }
KOKKOS_INLINE_FUNCTION Real value_of(Real a) { return a; }

struct dual1_real {
  Real val, d;
  KOKKOS_INLINE_FUNCTION dual1_real() : val(0.0), d(0.0) {}
  KOKKOS_INLINE_FUNCTION dual1_real(Real v) : val(v), d(0.0) {}
  KOKKOS_INLINE_FUNCTION dual1_real(Real v, Real deriv) : val(v), d(deriv) {}
};
KOKKOS_INLINE_FUNCTION dual1_real operator+(const dual1_real &a,
                                            const dual1_real &b) {
  return dual1_real(a.val + b.val, a.d + b.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator+(const dual1_real &a, Real b) {
  return dual1_real(a.val + b, a.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator+(Real a, const dual1_real &b) {
  return b + a;
}
KOKKOS_INLINE_FUNCTION dual1_real operator-(const dual1_real &a,
                                            const dual1_real &b) {
  return dual1_real(a.val - b.val, a.d - b.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator-(const dual1_real &a, Real b) {
  return dual1_real(a.val - b, a.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator-(Real a, const dual1_real &b) {
  return dual1_real(a - b.val, -b.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator-(const dual1_real &a) {
  return dual1_real(-a.val, -a.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator*(const dual1_real &a,
                                            const dual1_real &b) {
  return dual1_real(a.val * b.val, a.d * b.val + a.val * b.d);
}
KOKKOS_INLINE_FUNCTION dual1_real operator*(const dual1_real &a, Real b) {
  return dual1_real(a.val * b, a.d * b);
}
KOKKOS_INLINE_FUNCTION dual1_real operator*(Real a, const dual1_real &b) {
  return b * a;
}
KOKKOS_INLINE_FUNCTION dual1_real operator/(const dual1_real &a,
                                            const dual1_real &b) {
  const Real inv = 1.0 / b.val;
  return dual1_real(a.val * inv, (a.d * b.val - a.val * b.d) * inv * inv);
}
KOKKOS_INLINE_FUNCTION dual1_real operator/(const dual1_real &a, Real b) {
  const Real inv = 1.0 / b;
  return dual1_real(a.val * inv, a.d * inv);
}
KOKKOS_INLINE_FUNCTION dual1_real operator/(Real a, const dual1_real &b) {
  const Real inv = 1.0 / b.val;
  return dual1_real(a * inv, -a * b.d * inv * inv);
}
KOKKOS_INLINE_FUNCTION dual1_real metric_sqrt(const dual1_real &a) {
  const Real s = sqrt(a.val);
  return dual1_real(s, 0.5 * a.d / s);
}
KOKKOS_INLINE_FUNCTION Real value_of(const dual1_real &a) { return a.val; }
KOKKOS_INLINE_FUNCTION Real deriv_of(const dual1_real &a) { return a.d; }

template <typename T>
KOKKOS_INLINE_FUNCTION T metric_norm3(T x, T y, T z) {
  T r2 = x * x + y * y + z * z;
  if (value_of(r2) <= 0.0) {
    return T(0.0);
  }
  return metric_sqrt(r2);
}

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

struct bbh_traj_state {
  Real q[NTRAJ];
  Real dq[NTRAJ];
};

enum class MetricDerivativeMethod {
  ad,
  finite_difference
};

enum class CoolingSource {
  none,
  ism,
  thin_disk
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
  Real d;
  Real gamma_adi;
  Real a1_buffer, a2_buffer;
  Real cutoff_floor;
  Real metric_fd_step;
  Real alpha_thr;
  Real radius_thr;
  Real smooth_b_damping_eta;
  Real smooth_b_damping_cfl;
  Real puncture_excise_rad1;
  Real puncture_excise_rad2;
  Real puncture_excise_shrink_timescale;
  Real puncture_excise_shrink_start_time;
  Real thin_cooling_h_over_r;
  Real thin_cooling_timescale_orbits;
  Real thin_cooling_cfl;
  Real thin_cooling_r_inner;
  Real thin_cooling_r_outer;
  bool use_traj_table;
  bool spin_ramp;
  bool smooth_b_damping;
  bool puncture_excise_cap_to_horizon;
  bool puncture_excise_to_horizon;
  bool puncture_excise_shrink_to_horizon;
  bool require_resolved_horizon;
  CoolingSource cooling_source;
  MetricDerivativeMethod metric_derivative_method;

  Real spin;

  Real dexcise, pexcise;                      // excision parameters
  Real arad;                                  // radiation constant
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

// a separate struc for refinement method, etc.
struct bbh_refine {
  bool AlphaMin = false;
  bool Tracker = false;
  Real tracker_radius[2] = {0.0, 0.0};
  int tracker_reflevel[2] = {-1, -1};
  std::vector<Real> radius;
  std::vector<int> reflevel;
};

// Parameters controlling the moving-tetrad "flash" radiation beam source. Unlike the
// generic (continuous, coordinate-static) <rad_srcterms> BeamSource, this source (a) is a
// short pulse whose amplitude decays exponentially in time and then turns off completely,
// and (b) is anchored to one orbiting black hole's equatorial photon ring: the launch
// point sits at the Cartesian Kerr-Schild ring radius sqrt(r_ph^2+a^2), the beam axis is
// the ring-tangent null direction, and the angular weights are re-projected against the
// current (moving) metric/tetrad every substage.  The construction mirrors the validated
// single-hole rad_kerr_orbit_beam test (src/pgen/tests/rad_beam.cpp).
// See DynBBHFlashBeamSource().
struct flash_beam_pgen {
  bool enabled = false;
  Real amp = 0.0;          // peak amplitude A0
  Real tau = 1.0;          // exponential decay timescale of the flash
  Real t0 = 0.0;           // time the flash turns on
  Real toff = 0.0;         // time the flash turns off (no injection afterwards)
  Real width = 0.2;        // spatial Gaussian sigma of the source spot
  int  src_bh = 1;         // which BH the beam is launched from (1 or 2)
  Real ring_frac = 1.0;    // launch radius as a fraction of the photon-ring radius
  Real ring_radius = -1.0; // explicit ring radius override (<=0 -> analytic CKS radius)
  Real ring_angle = 0.0;   // azimuth (rad) of the launch point on the ring (t=t0 frame)
  Real sense = -1.0;       // tangent sense: +1 prograde (+phi), -1 retrograde (-phi)
  Real aim_offset = 0.0;   // extra in-plane rotation (rad) of the tangent beam axis
  bool corotate = true;    // rigidly co-rotate launch point + aim with the binary
};

struct bbh_pgen bbh;
struct bbh_refine bbh_ref;
struct flash_beam_pgen flash;
struct bbh_traj_table {
  std::vector<Real> t;
  std::vector<Real> x1, y1, z1, x2, y2, z2;
  std::vector<Real> vx1, vy1, vz1, vx2, vy2, vz2;
  std::vector<Real> ax1, ay1, az1, ax2, ay2, az2;
  std::vector<Real> m1, m2;
  std::size_t active_segment = 0;
};
struct bbh_traj_table bbh_table;

bool SpinVectorWithinExtremality(Real chix, Real chiy, Real chiz) {
  Real chi2 = SQR(chix) + SQR(chiy) + SQR(chiz);
  return std::isfinite(chi2) && chi2 <= 1.0;
}

bool SpinMagnitudeWithinExtremality(Real chi) {
  return std::isfinite(chi) && chi >= 0.0 && SQR(chi) <= 1.0;
}

void find_traj_t(Real tt, Real traj_array[NTRAJ]);
void find_traj_t_with_deriv(Real tt, Real traj_array[NTRAJ],
                            Real dtraj_array[NTRAJ]);

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
    Real eps1 = 16.0*std::numeric_limits<Real>::epsilon()*
                std::max({1.0, std::abs(mb.x1min), std::abs(mb.x1max)});
    Real eps2 = 16.0*std::numeric_limits<Real>::epsilon()*
                std::max({1.0, std::abs(mb.x2min), std::abs(mb.x2max)});
    Real eps3 = 16.0*std::numeric_limits<Real>::epsilon()*
                std::max({1.0, std::abs(mb.x3min), std::abs(mb.x3max)});
    bool contains = (x >= mb.x1min - eps1 && x <= mb.x1max + eps1);
    if (indcs.nx2 > 1) contains = contains &&
        (y >= mb.x2min - eps2 && y <= mb.x2max + eps2);
    if (indcs.nx3 > 1) contains = contains &&
        (z >= mb.x3min - eps3 && z <= mb.x3max + eps3);
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
  if (local_dx == std::numeric_limits<Real>::max()) {
    local_dx = LocalFinestMeshSpacing(pmbp);
  }
  return local_dx;
}

Real HorizonRadiusFromMassAndChi(Real mass, Real chix, Real chiy, Real chiz) {
  Real chi2 = SQR(chix) + SQR(chiy) + SQR(chiz);
  return mass * (1.0 + std::sqrt(std::max(1.0 - chi2, 0.0)));
}

Real SmoothExcisionRadiusToHorizon(const Real requested_radius,
                                   const Real horizon_radius,
                                   const Real time,
                                   const Real timescale,
                                   const bool set_to_horizon,
                                   const bool shrink_to_horizon) {
  Real start_radius = (requested_radius > 0.0) ? requested_radius : horizon_radius;
  if (set_to_horizon) return horizon_radius;
  if (!shrink_to_horizon) return start_radius;
  start_radius = std::max(start_radius, horizon_radius);
  if (!(timescale > 0.0)) return horizon_radius;
  Real f = std::min(std::max(time/timescale, 0.0), 1.0);
  f = f*f*(3.0 - 2.0*f);
  return (1.0 - f)*start_radius + f*horizon_radius;
}

Real SmoothRamp01(const Real time, const Real start_time, const Real timescale,
                  Real *dfdt) {
  *dfdt = 0.0;
  if (!(timescale > 0.0)) return (time >= start_time) ? 1.0 : 0.0;
  Real u = (time - start_time) / timescale;
  if (u <= 0.0) return 0.0;
  if (u >= 1.0) return 1.0;
  *dfdt = 6.0*u*(1.0 - u) / timescale;
  return u*u*(3.0 - 2.0*u);
}

Real MinDynBBHHorizonRadius() {
  if (bbh.use_traj_table && !bbh_table.t.empty()) {
    Real rmin = std::numeric_limits<Real>::max();
    for (std::size_t n = 0; n < bbh_table.t.size(); ++n) {
      Real m1 = bbh_table.m1[n];
      Real m2 = bbh_table.m2[n];
      rmin = std::min(rmin, HorizonRadiusFromMassAndChi(
          m1, bbh_table.ax1[n], bbh_table.ay1[n], bbh_table.az1[n]));
      rmin = std::min(rmin, HorizonRadiusFromMassAndChi(
          m2, bbh_table.ax2[n], bbh_table.ay2[n], bbh_table.az2[n]));
    }
    return rmin;
  }
  Real state[NTRAJ];
  find_traj_t(0.0, state);
  Real m1 = state[M1T];
  Real m2 = state[M2T];
  return std::min(
      HorizonRadiusFromMassAndChi(m1, state[AX1], state[AY1], state[AZ1]),
      HorizonRadiusFromMassAndChi(m2, state[AX2], state[AY2], state[AZ2]));
}

void CheckPunctureExcisionResolution(MeshBlockPack *pmbp, const bbh_traj_state &traj,
                                     const Real r0, const Real r1,
                                     const char *context) {
  constexpr Real min_cells_across = 10.0;
  Real dx0 = LocalMeshSpacingAtPoint(pmbp, traj.q[X1], traj.q[Y1], traj.q[Z1]);
  Real dx1 = LocalMeshSpacingAtPoint(pmbp, traj.q[X2], traj.q[Y2], traj.q[Z2]);
  Real cells0 = (r0 > 0.0) ? 2.0*r0/std::max(dx0, 1.0e-300) : 0.0;
  Real cells1 = (r1 > 0.0) ? 2.0*r1/std::max(dx1, 1.0e-300) : 0.0;
  if ((cells0 < min_cells_across || cells1 < min_cells_across) &&
      global_variable::my_rank == 0) {
    std::cout << "WARNING: puncture excision radius is under-resolved during "
              << context << ": cells across excision diameter are "
              << "hole1=" << cells0 << " (r=" << r0 << ", dx=" << dx0 << "), "
              << "hole2=" << cells1 << " (r=" << r1 << ", dx=" << dx1 << "); "
              << "recommended minimum is " << min_cells_across << std::endl;
  }
}

/* Declare functions */
void find_traj_t(Real tt, Real traj_array[NTRAJ]);
void find_traj_t_with_deriv(Real tt, Real traj_array[NTRAJ],
                            Real dtraj_array[NTRAJ]);
bbh_traj_state find_traj_state(Real tt);
void LoadTrajectoryTable(const std::string &fname);

int four_metric_to_three_metric(const struct four_metric &met,
                                struct three_metric &gam);
KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen bbh_);
KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
                       const Real z, struct four_metric &outmet,
                       const Real traj_m[NTRAJ], const Real traj_0[NTRAJ],
                       const Real traj_p[NTRAJ], const Real hm, const Real hp,
                       const bbh_pgen bbh_);
KOKKOS_INLINE_FUNCTION
void get_metric_and_derivatives(const Real t, const Real x, const Real y,
                                const Real z, struct four_metric &met,
                                const Real bbh_traj_loc[NTRAJ],
                                const Real dtraj_array[NTRAJ],
                                const bbh_pgen bbh_);
KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                   Real gcov[][NDIM], const Real traj_array[NTRAJ],
                   const bbh_pgen bbh_);
void SetADMVariablesToBBH(MeshBlockPack *pmbp);
void RefineAlphaMin(MeshBlockPack* pmbp);
void RefineTracker(MeshBlockPack* pmbp);
void RefineRadii(MeshBlockPack* pmbp);
void Refine(MeshBlockPack* pmbp);
void AddValenciaGRCooling(Mesh *pm, const Real bdt);
void AddThinDiskCooling(Mesh *pm, const Real bdt);
void AddDynBBHUserSources(Mesh *pm, const Real bdt);
void DynBBHFlashBeamSource(Mesh *pm, const Real bdt);
void FlashLaunchGeometry(const Real t, Real pos[3], Real dir[3]);
Real FlashPhotonRingRadiusCKS(const Real mass, const Real chi, const Real sense);
void FlashMetricAt(const Real t, const Real x, const Real y, const Real z,
                   Real g4[NDIM][NDIM]);
bool FlashCovariantNull(const Real g4[NDIM][NDIM], const Real dir[3], Real k_cov[NDIM]);
void AddSmoothExcisionMagneticDamping(Mesh *pm, DvceEdgeFld4D<Real> &efld);

//----------------------------------------------------------------------------------------
//! \fn void TorusHistory(HistoryData *pdata, Mesh *pm)
//! \brief User history function that centers horizon grids and calls flux integrator.
//----------------------------------------------------------------------------------------
void TorusHistory(HistoryData *pdata, Mesh *pm) {
    ProblemGenerator *pgen = pm->pgen.get();
    if (pgen->surface_grids.empty()) {
        pdata->nhist = 0;
        return;
    }
    MeshBlockPack *pmbp = pm->pmb_pack;

    // 1. Calculate current Black Hole Trajectory
    Real tt = pm->time;
    Real btraj[NTRAJ];

    // Ensure find_traj_t is visible here (it is declared global in your snippet)
    find_traj_t(tt, btraj);

    // 2. Update Surface Grid Centers
    // We iterate through all grids and check labels to see if they are horizons
    for(auto& grid_ptr : pgen->surface_grids) {
        if (grid_ptr->Label() == "H1") {
            // Move grid to BH1 location (Indices 0, 1, 2)
            grid_ptr->SetCenter(&btraj[0]);
        }
        else if (grid_ptr->Label() == "H2") {
            // Move grid to BH2 location (Indices 3, 4, 5)
            grid_ptr->SetCenter(&btraj[3]);
        }
    }

    // 3. Prepare pointers for the flux calculator
    std::vector<SphericalSurfaceGrid*> surf_raw_ptrs;
    surf_raw_ptrs.reserve(pgen->surface_grids.size());
    for(const auto& s : pgen->surface_grids) {
        surf_raw_ptrs.push_back(s.get());
    }

    // 4. Calculate fluxes
    TorusFluxes_General(pdata, pmbp, surf_raw_ptrs);
}


KOKKOS_INLINE_FUNCTION
static void GetSuperposedAndInverse(const Real t,
                            const Real x, const Real y, const Real z,
                            Real gcov[][NDIM], Real gcon[][NDIM], const Real bbh_traj_loc[NTRAJ],
                            const bbh_pgen bbh_);



KOKKOS_INLINE_FUNCTION
static void CalculateCN(struct bbh_pgen pgen, Real *cparam, Real *nparam);

KOKKOS_INLINE_FUNCTION
static Real CalculateL(struct bbh_pgen pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static Real CalculateCovariantUT(struct bbh_pgen pgen, Real r, Real sin_theta, Real l);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(struct bbh_pgen pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static Real CalculateT(struct bbh_pgen pgen, Real rho, Real ptot_over_rho);

KOKKOS_INLINE_FUNCTION
static Real LogHAux(struct bbh_pgen pgen, Real r, Real sin_theta);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct bbh_pgen pgen,
                                           Real r, Real theta, Real phi, Real *pu0,
                                           Real *pu1, Real *pu2, Real *pu3);
KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct bbh_pgen pgen,
                                     Real r, Real sin_theta, Real *pu0, Real *pu3);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bbh_pgen pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotentialInTiltedTorus(struct bbh_pgen pgen,
                                                  Real r, Real theta, Real phi,
                                                  Real *patheta, Real *paphi);


KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bbh_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void InvertMetric(Real gcov[][NDIM], Real gcon[][NDIM]);

KOKKOS_INLINE_FUNCTION
Real A1(struct bbh_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct bbh_pgen pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct bbh_pgen pgen, Real x1, Real x2, Real x3);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DynBBHBeam()
//! \brief DynBBH ADM radiation beam/coupling smoke test

void ProblemGenerator::DynBBHBeam(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "dynbbh_beam requires GR coordinates or ADM metric fields"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh_beam requires an <adm> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->prad != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh_beam uses ADM metric data and requires "
              << "<dyn_radiation>, not legacy <radiation>." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pdynrad == nullptr || !(pmbp->pdynrad->use_adm_geometry)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh_beam requires <dyn_radiation> geometry='adm'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr || pmbp->pdyngr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh_beam is a Valencia GRMHD radiation-coupling "
              << "smoke pgen and requires <mhd> with <adm>." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto read_real_alias = [pin](const char *primary, const char *alias,
                              const Real fallback) {
    if (pin->DoesParameterExist("problem", primary)) {
      return pin->GetReal("problem", primary);
    }
    return pin->GetOrAddReal("problem", alias, fallback);
  };

  bbh.spin = 0.0;
  bbh.sep = pin->GetOrAddReal("problem", "sep", 4.0);
  bbh.om = std::pow(bbh.sep, -1.5);
  bbh.q = pin->GetOrAddReal("problem", "q", 1.0);
  bbh.a1 = read_real_alias("a1", "spin_a1", 0.0);
  bbh.a2 = read_real_alias("a2", "spin_a2", 0.0);
  bbh.th_a1 = pin->GetOrAddReal("problem", "th_a1", 0.0) * (M_PI/180.0);
  bbh.th_a2 = pin->GetOrAddReal("problem", "th_a2", 0.0) * (M_PI/180.0);
  bbh.ph_a1 = pin->GetOrAddReal("problem", "ph_a1", 0.0) * (M_PI/180.0);
  bbh.ph_a2 = pin->GetOrAddReal("problem", "ph_a2", 0.0) * (M_PI/180.0);
  bbh.spin_ramp = pin->GetOrAddBoolean("problem", "spin_ramp", false);
  bbh.spin_ramp_timescale = pin->GetOrAddReal(
      "problem", "spin_ramp_timescale", 50.0);
  bbh.spin_ramp_start_time = pin->GetOrAddReal(
      "problem", "spin_ramp_start_time", pmbp->pmesh->time);
  bbh.d = 1.0;
  bbh.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  bbh.a1_buffer = pin->GetOrAddReal("problem", "a1_buffer", 0.01);
  bbh.a2_buffer = pin->GetOrAddReal("problem", "a2_buffer", 0.01);
  bbh.cutoff_floor = pin->GetOrAddReal("problem", "cutoff_floor", 1.0e-4);
  bbh.metric_fd_step = pin->GetOrAddReal("problem", "metric_fd_step", metric_fd_step);
  bbh.alpha_thr = pin->GetOrAddReal("problem", "alpha_thr", 0.2);
  bbh.radius_thr = pin->GetOrAddReal("problem", "radius_thr", 2.0);
  bbh.smooth_b_damping = false;
  bbh.smooth_b_damping_eta = 0.0;
  bbh.smooth_b_damping_cfl = 0.0;
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
  bbh.puncture_excise_shrink_start_time = pmbp->pmesh->time;
  bbh.cooling_source = CoolingSource::none;
  bbh.thin_cooling_h_over_r = 0.0;
  bbh.thin_cooling_timescale_orbits = 0.0;
  bbh.thin_cooling_cfl = 0.0;
  bbh.thin_cooling_r_inner = 0.0;
  bbh.thin_cooling_r_outer = 0.0;
  bbh.psi = 0.0;
  bbh.sin_psi = 0.0;
  bbh.cos_psi = 1.0;

  if (!SpinMagnitudeWithinExtremality(bbh.a1) ||
      !SpinMagnitudeWithinExtremality(bbh.a2)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh_beam spin magnitudes must satisfy 0 <= chi <= 1"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(bbh.metric_fd_step > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "problem/metric_fd_step must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string metric_derivative = pin->GetOrAddString(
      "problem", "metric_derivative", "ad");
  if (metric_derivative == "ad") {
    bbh.metric_derivative_method = MetricDerivativeMethod::ad;
  } else if (metric_derivative == "finite_difference" || metric_derivative == "fd") {
    bbh.metric_derivative_method = MetricDerivativeMethod::finite_difference;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown problem/metric_derivative='" << metric_derivative
              << "'. Use 'ad' or 'finite_difference'." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bbh.use_traj_table = pin->GetOrAddBoolean("problem", "use_traj_table", false);
  std::string traj_file = pin->GetOrAddString("problem", "traj_file", "");
  if (bbh.use_traj_table) {
    if (traj_file.empty()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "use_traj_table=true requires traj_file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    LoadTrajectoryTable(traj_file);
  }

  pmbp->padm->SetADMVariables = &SetADMVariablesToBBH;
  pmbp->padm->SetADMVariables(pmbp);
  if (pmbp->pcoord->coord_data.bh_excise) {
    pmbp->pcoord->UpdateExcisionMasks();
  }
  pmbp->pdynrad->PrepareADMGeometry();

  // -----------------------------------------------------------------------------------
  // Moving-tetrad "flash" beam source (enrolled as the user radiation source term).
  // Parameters live in a <flash_beam> block. This is the correct source for the orbiting
  // binary: the generic <rad_srcterms> BeamSource injects continuously at a fixed
  // coordinate position/direction, so as the binary orbits the launch point drifts off
  // the (moving) photon ring and the tetrad projection becomes stale.  Here we instead
  // anchor the launch point/aim to the chosen BH, re-derive them from the current
  // trajectory every substage, and modulate the amplitude as an exponential flash.
  flash.enabled = pin->GetOrAddBoolean("problem", "flash_enabled", false);
  if (flash.enabled) {
    flash.amp        = pin->GetOrAddReal("problem", "flash_amp", 1.0);
    flash.tau        = pin->GetOrAddReal("problem", "flash_tau", 0.2);
    flash.t0         = pin->GetOrAddReal("problem", "flash_t0", 0.0);
    flash.toff       = pin->GetOrAddReal("problem", "flash_toff", flash.t0 + 6.0*flash.tau);
    flash.width      = pin->GetOrAddReal("problem", "flash_width", 0.2);
    flash.src_bh     = pin->GetOrAddInteger("problem", "flash_src_bh", 1);
    flash.ring_frac  = pin->GetOrAddReal("problem", "flash_ring_frac", 1.0);
    flash.ring_radius= pin->GetOrAddReal("problem", "flash_ring_radius", -1.0);
    flash.ring_angle = pin->GetOrAddReal("problem", "flash_ring_angle_deg", 0.0)
                       *(M_PI/180.0);
    flash.sense      = (pin->GetOrAddString("problem", "flash_ring_sense",
                                            "retrograde") == "prograde") ? +1.0 : -1.0;
    flash.aim_offset = pin->GetOrAddReal("problem", "flash_aim_offset_deg", 0.0)
                       *(M_PI/180.0);
    flash.corotate   = pin->GetOrAddBoolean("problem", "flash_corotate", true);
    if (!(flash.tau > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "flash_beam/tau must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    user_srcs = true;
    user_srcs_func = DynBBHFlashBeamSource;
  }

  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const Real rho0 = read_real_alias("rho", "rho_min", 1.0e-10);
  const Real pgas0 = read_real_alias("pgas", "pgas_min", 1.0e-12);
  if (!(rho0 > 0.0) || !(pgas0 > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dynbbh_beam requires positive rho/pgas" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &w0 = pmbp->pmhd->w0;
  par_for("dynbbh_beam_prims", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = rho0;
    w0(m,IPR,k,j,i) = pgas0;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
  });

  Kokkos::deep_copy(pmbp->pmhd->bcc0, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(pmbp->pdynrad->i0, 0.0);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  bool init_beam_particles = pin->GetOrAddBoolean("problem", "init_beam_edge_particles",
                                                  false);
  if (init_beam_particles) {
    if (pmbp->ppart == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "init_beam_edge_particles=true requires a <particles> block"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Launch point + central aim direction of the edge tracers.  When the flash beam is
    // active the tracers are seeded at the flash source's launch point/aim (evaluated at
    // the initial time) so the null-geodesic edges coincide with the emitted radiation.
    Real p1, p2, p3, d1, d2, d3;
    Real edge_spread_default = 40.0;
    if (flash.enabled) {
      Real pos[3], dir[3];
      FlashLaunchGeometry(pmbp->pmesh->time, pos, dir);
      p1 = pos[0]; p2 = pos[1]; p3 = pos[2];
      d1 = dir[0]; d2 = dir[1]; d3 = dir[2];
      edge_spread_default = 6.0;
    } else {
      std::string beam_block = pin->DoesBlockExist("rad_srcterms") ? "rad_srcterms"
                                                                   : "problem";
      p1 = pin->GetOrAddReal(beam_block, "pos_1", 0.0);
      p2 = pin->GetOrAddReal(beam_block, "pos_2", 0.0);
      p3 = pin->GetOrAddReal(beam_block, "pos_3", 0.0);
      d1 = pin->GetOrAddReal(beam_block, "dir_1", 1.0);
      d2 = pin->GetOrAddReal(beam_block, "dir_2", 0.0);
      d3 = pin->GetOrAddReal(beam_block, "dir_3", 0.0);
      Real dnorm = std::sqrt(SQR(d1) + SQR(d2) + SQR(d3));
      if (!(dnorm > 0.0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "beam direction must be nonzero" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      d1 /= dnorm; d2 /= dnorm; d3 /= dnorm;
    }

    // A fixed number of tracers fanned across the beam's angular width, in the orbital
    // (x-y) plane (aim direction rotated about the z-axis).  The two extreme rays are the
    // beam edges; the interior rays fill in the fan for visualization.
    int n_edge = pin->GetOrAddInteger("problem", "n_edge_tracers", 21);
    if (n_edge < 1) n_edge = 1;
    Real spread = pin->GetOrAddReal("problem", "edge_spread_deg",
                                    edge_spread_default)*(M_PI/180.0);

    // Find the (single) local meshblock that contains the launch point, if any.
    int msel_h = -1;
    for (int m=0; m<nmb; ++m) {
      bool inside = (p1 >= size.h_view(m).x1min && p1 <= size.h_view(m).x1max) &&
                    (p2 >= size.h_view(m).x2min && p2 <= size.h_view(m).x2max) &&
                    (p3 >= size.h_view(m).x3min && p3 <= size.h_view(m).x3max);
      if (inside) msel_h = m;
    }
    int npart_local = (msel_h >= 0) ? n_edge : 0;
    pmbp->ppart->nprtcl_thispack = npart_local;
    Kokkos::resize(pmbp->ppart->prtcl_rdata, pmbp->ppart->nrdata, npart_local);
    Kokkos::resize(pmbp->ppart->prtcl_idata, pmbp->ppart->nidata, npart_local);

    pmbp->pmesh->nprtcl_thisrank = pmbp->ppart->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
    MPI_Allgather(&(pmbp->pmesh->nprtcl_thisrank), 1, MPI_INT,
                  pmbp->pmesh->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#else
    pmbp->pmesh->nprtcl_eachrank[0] = pmbp->pmesh->nprtcl_thisrank;
#endif
    pmbp->pmesh->nprtcl_total = 0;
    for (int n=0; n<global_variable::nranks; ++n) {
      pmbp->pmesh->nprtcl_total += pmbp->pmesh->nprtcl_eachrank[n];
    }
    pmbp->ppart->CreateParticleTags(pin);

    if (npart_local > 0) {
      // Seed the covariant momenta on the host from the ANALYTIC metric at the exact
      // launch point.  Each fan ray has coordinate direction d^i (the beam axis rotated
      // about z); its time component d^0 comes from the null condition, and the covariant
      // spatial momentum is k_i = g_{i mu} d^mu = gamma_ij d^j + beta_i d^0.  The
      // beta_i d^0 (shift) term is essential: the null-geodesic pusher then yields
      // coordinate velocity dx^i/dt = d^i/d^0, exactly parallel to the radiation beam's
      // null completion.  (Seeding k_i = gamma_ij d^j alone -- the old approach -- makes
      // the tracer velocity alpha*dhat - beta, which visibly disagrees with the beam
      // wherever the shift is significant, i.e. everywhere near the punctures.)
      Real g4l[NDIM][NDIM];
      FlashMetricAt(pmbp->pmesh->time, p1, p2, p3, g4l);
      DualArray2D<Real> kseed("kseed", npart_local, 3);
      for (int p=0; p<npart_local; ++p) {
        Real frac = (n_edge > 1)
                  ? (static_cast<Real>(p)/static_cast<Real>(n_edge - 1) - 0.5) : 0.0;
        Real psi = frac*spread;
        Real dirp[3] = {std::cos(psi)*d1 - std::sin(psi)*d2,
                        std::sin(psi)*d1 + std::cos(psi)*d2, d3};
        Real k_cov[NDIM];
        if (!FlashCovariantNull(g4l, dirp, k_cov)) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "edge tracer direction is not null-realizable"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        kseed.h_view(p,0) = k_cov[1];
        kseed.h_view(p,1) = k_cov[2];
        kseed.h_view(p,2) = k_cov[3];
      }
      kseed.template modify<HostMemSpace>();
      kseed.template sync<DevExeSpace>();

      auto &pr = pmbp->ppart->prtcl_rdata;
      auto &pi = pmbp->ppart->prtcl_idata;
      int pgid = pmbp->gids + msel_h;
      auto kseed_d = kseed.d_view;
      par_for("dynbbh_beam_edge_particles", DevExeSpace(), 0, npart_local-1,
      KOKKOS_LAMBDA(const int p) {
        pi(PGID,p) = pgid;
        pr(IPX,p) = p1;
        pr(IPY,p) = p2;
        pr(IPZ,p) = p3;
        pr(IPVX,p) = kseed_d(p,0);
        pr(IPVY,p) = kseed_d(p,1);
        pr(IPVZ,p) = kseed_d(p,2);
      });
    }
  }
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  DynBBHBeam(pin, restart);
}

namespace {

template <typename ADMVars>
KOKKOS_INLINE_FUNCTION
void StoreADMVariables(ADMVars adm_vars, const int m, const int k,
                       const int j, const int i, const struct three_metric &met3) {
  adm_vars.g_dd(m,0,0,k,j,i) = met3.gxx;
  adm_vars.g_dd(m,0,1,k,j,i) = met3.gxy;
  adm_vars.g_dd(m,0,2,k,j,i) = met3.gxz;
  adm_vars.g_dd(m,1,1,k,j,i) = met3.gyy;
  adm_vars.g_dd(m,1,2,k,j,i) = met3.gyz;
  adm_vars.g_dd(m,2,2,k,j,i) = met3.gzz;

  adm_vars.vK_dd(m,0,0,k,j,i) = met3.kxx;
  adm_vars.vK_dd(m,0,1,k,j,i) = met3.kxy;
  adm_vars.vK_dd(m,0,2,k,j,i) = met3.kxz;
  adm_vars.vK_dd(m,1,1,k,j,i) = met3.kyy;
  adm_vars.vK_dd(m,1,2,k,j,i) = met3.kyz;
  adm_vars.vK_dd(m,2,2,k,j,i) = met3.kzz;

  adm_vars.alpha(m,k,j,i) = met3.alpha;
  adm_vars.beta_u(m,0,k,j,i) = met3.betax;
  adm_vars.beta_u(m,1,k,j,i) = met3.betay;
  adm_vars.beta_u(m,2,k,j,i) = met3.betaz;
}

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

  auto &coord = pmbp->pcoord->coord_data;

  bbh_traj_state traj = find_traj_state(tt);
  auto bbh_ = bbh;

  // update punc location for excision
  coord.punc_0[0] = traj.q[X1];
  coord.punc_0[1] = traj.q[Y1];
  coord.punc_0[2] = traj.q[Z1];
  coord.punc_1[0] = traj.q[X2];
  coord.punc_1[1] = traj.q[Y2];
  coord.punc_1[2] = traj.q[Z2];
  Real m1_ex = traj.q[M1T];
  Real m2_ex = traj.q[M2T];
  Real a1x_ex = traj.q[AX1] * m1_ex;
  Real a1y_ex = traj.q[AY1] * m1_ex;
  Real a1z_ex = traj.q[AZ1] * m1_ex;
  Real a2x_ex = traj.q[AX2] * m2_ex;
  Real a2y_ex = traj.q[AY2] * m2_ex;
  Real a2z_ex = traj.q[AZ2] * m2_ex;
  coord.punc_0_spin[0] = a1x_ex;
  coord.punc_0_spin[1] = a1y_ex;
  coord.punc_0_spin[2] = a1z_ex;
  coord.punc_1_spin[0] = a2x_ex;
  coord.punc_1_spin[1] = a2y_ex;
  coord.punc_1_spin[2] = a2z_ex;
  coord.punc_0_vel[0] = traj.q[VX1];
  coord.punc_0_vel[1] = traj.q[VY1];
  coord.punc_0_vel[2] = traj.q[VZ1];
  coord.punc_1_vel[0] = traj.q[VX2];
  coord.punc_1_vel[1] = traj.q[VY2];
  coord.punc_1_vel[2] = traj.q[VZ2];
  Real rH1 = HorizonRadiusFromMassAndChi(m1_ex, traj.q[AX1], traj.q[AY1],
                                         traj.q[AZ1]);
  Real rH2 = HorizonRadiusFromMassAndChi(m2_ex, traj.q[AX2], traj.q[AY2],
                                         traj.q[AZ2]);
  coord.punc_0_rad = SmoothExcisionRadiusToHorizon(
      bbh_.puncture_excise_rad1, rH1, tt - bbh_.puncture_excise_shrink_start_time,
      bbh_.puncture_excise_shrink_timescale,
      bbh_.puncture_excise_to_horizon, bbh_.puncture_excise_shrink_to_horizon);
  coord.punc_1_rad = SmoothExcisionRadiusToHorizon(
      bbh_.puncture_excise_rad2, rH2, tt - bbh_.puncture_excise_shrink_start_time,
      bbh_.puncture_excise_shrink_timescale,
      bbh_.puncture_excise_to_horizon, bbh_.puncture_excise_shrink_to_horizon);
  if (bbh_.puncture_excise_cap_to_horizon &&
      !bbh_.puncture_excise_to_horizon &&
      !bbh_.puncture_excise_shrink_to_horizon) {
    coord.punc_0_rad = std::min(coord.punc_0_rad, rH1);
    coord.punc_1_rad = std::min(coord.punc_1_rad, rH2);
  }
  if (pmbp->pcoord->coord_data.bh_excise &&
      pmbp->pcoord->coord_data.excision_scheme == ExcisionScheme::puncture) {
    CheckPunctureExcisionResolution(pmbp, traj, coord.punc_0_rad,
                                    coord.punc_1_rad, "ADM update");
  }

  if (bbh_.metric_derivative_method == MetricDerivativeMethod::ad) {
    par_for("update_adm_vars_ad", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
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
      get_metric_and_derivatives(tt, x1v, x2v, x3v, met4, traj.q, traj.dq, bbh_);
      four_metric_to_three_metric(met4, met3);
      StoreADMVariables(adm, m, k, j, i, met3);
    });
    return;
  }

  Real traj_m[NTRAJ], traj_p[NTRAJ];
  Real hm = bbh_.metric_fd_step, hp = bbh_.metric_fd_step;

  if (bbh_.use_traj_table && !bbh_table.t.empty()) {
    Real tmin = bbh_table.t.front();
    Real tmax = bbh_table.t.back();
    hm = fmin(metric_fd_step, fmax(tt - tmin, 0.0));
    hp = fmin(metric_fd_step, fmax(tmax - tt, 0.0));
    if (hm == 0.0 && hp == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "trajectory table does not bracket time for metric derivatives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (hm > 0.0) {
    find_traj_t(tt - hm, traj_m);
  } else {
    for (int n = 0; n < NTRAJ; ++n) traj_m[n] = traj.q[n];
  }
  if (hp > 0.0) {
    find_traj_t(tt + hp, traj_p);
  } else {
    for (int n = 0; n < NTRAJ; ++n) traj_p[n] = traj.q[n];
  }

  par_for("update_adm_vars_fd", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
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
    numerical_4metric(tt, x1v, x2v, x3v, met4, traj_m, traj.q, traj_p,
                      hm, hp, bbh_);

    four_metric_to_three_metric(met4, met3);
    StoreADMVariables(adm, m, k, j, i, met3);
  });
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

void LoadTrajectoryTable(const std::string &fname) {
  std::ifstream fin(fname);
  if (!fin.is_open()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "could not open trajectory file: '" << fname << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bbh_table = bbh_traj_table();
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(fin, line)) {
    ++lineno;
    auto p = line.find_first_not_of(" \t\r\n");
    if (p == std::string::npos || line[p] == '#')
      continue;
    std::istringstream iss(line);
    Real c[21];
    for (int i = 0; i < 21; ++i) {
      if (!(iss >> c[i]) || !std::isfinite(c[i])) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                  << __LINE__ << std::endl
                  << "bad trajectory value in '" << fname << "' line " << lineno
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (!(c[1] > 0.0) || !(c[2] > 0.0) ||
        !SpinVectorWithinExtremality(c[9], c[10], c[11]) ||
        !SpinVectorWithinExtremality(c[12], c[13], c[14])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "invalid mass or spin in '" << fname << "' line " << lineno
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    bbh_table.t.push_back(c[0]);
    bbh_table.m1.push_back(c[1]);
    bbh_table.m2.push_back(c[2]);
    bbh_table.x1.push_back(c[3]);
    bbh_table.y1.push_back(c[4]);
    bbh_table.z1.push_back(c[5]);
    bbh_table.x2.push_back(c[6]);
    bbh_table.y2.push_back(c[7]);
    bbh_table.z2.push_back(c[8]);
    bbh_table.ax1.push_back(c[9]);
    bbh_table.ay1.push_back(c[10]);
    bbh_table.az1.push_back(c[11]);
    bbh_table.ax2.push_back(c[12]);
    bbh_table.ay2.push_back(c[13]);
    bbh_table.az2.push_back(c[14]);
    bbh_table.vx1.push_back(c[15]);
    bbh_table.vy1.push_back(c[16]);
    bbh_table.vz1.push_back(c[17]);
    bbh_table.vx2.push_back(c[18]);
    bbh_table.vy2.push_back(c[19]);
    bbh_table.vz2.push_back(c[20]);
  }
  if (bbh_table.t.size() < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "trajectory file has fewer than 2 rows" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 1; i < bbh_table.t.size(); ++i)
    if (!(bbh_table.t[i] > bbh_table.t[i - 1])) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "trajectory times must increase" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  bbh_table.active_segment = 0;
  if (global_variable::my_rank == 0)
    std::cout << "Loaded BBH trajectory table '" << fname << "' with "
              << bbh_table.t.size() << " rows" << std::endl;
}

void find_traj_t(Real t, Real bbh_t[NTRAJ]) {
  Real dbbh_t[NTRAJ];
  find_traj_t_with_deriv(t, bbh_t, dbbh_t);
}

bbh_traj_state find_traj_state(Real t) {
  bbh_traj_state state;
  find_traj_t_with_deriv(t, state.q, state.dq);
  return state;
}

void find_traj_t_with_deriv(Real t, Real bbh_t[NTRAJ], Real dbbh_t[NTRAJ]) {
  if (!bbh.use_traj_table) {
    Real r1 = bbh.q / (1.0 + bbh.q) * bbh.sep, r2 = -bbh.sep / (1.0 + bbh.q),
         c = std::cos(bbh.om * t), s = std::sin(bbh.om * t),
         om2 = bbh.om * bbh.om;
    bbh_t[X1] = r1 * c;
    bbh_t[Y1] = r1 * s;
    bbh_t[Z1] = 0;
    bbh_t[X2] = r2 * c;
    bbh_t[Y2] = r2 * s;
    bbh_t[Z2] = 0;
    bbh_t[VX1] = -r1 * bbh.om * s;
    bbh_t[VY1] = r1 * bbh.om * c;
    bbh_t[VZ1] = 0;
    bbh_t[VX2] = -r2 * bbh.om * s;
    bbh_t[VY2] = r2 * bbh.om * c;
    bbh_t[VZ2] = 0;
    Real spin_factor = 1.0, dspin_factor = 0.0;
    if (bbh.spin_ramp) {
      spin_factor = SmoothRamp01(t, bbh.spin_ramp_start_time,
                                 bbh.spin_ramp_timescale, &dspin_factor);
    }
    Real e1x = std::sin(bbh.th_a1) * std::cos(bbh.ph_a1);
    Real e1y = std::sin(bbh.th_a1) * std::sin(bbh.ph_a1);
    Real e1z = std::cos(bbh.th_a1);
    Real e2x = std::sin(bbh.th_a2) * std::cos(bbh.ph_a2);
    Real e2y = std::sin(bbh.th_a2) * std::sin(bbh.ph_a2);
    Real e2z = std::cos(bbh.th_a2);
    bbh_t[AX1] = bbh.a1 * spin_factor * e1x;
    bbh_t[AY1] = bbh.a1 * spin_factor * e1y;
    bbh_t[AZ1] = bbh.a1 * spin_factor * e1z;
    bbh_t[AX2] = bbh.a2 * spin_factor * e2x;
    bbh_t[AY2] = bbh.a2 * spin_factor * e2y;
    bbh_t[AZ2] = bbh.a2 * spin_factor * e2z;
    bbh_t[M1T] = 1.0 / (bbh.q + 1.0);
    bbh_t[M2T] = 1.0 - bbh_t[M1T];
    dbbh_t[X1] = bbh_t[VX1];
    dbbh_t[Y1] = bbh_t[VY1];
    dbbh_t[Z1] = 0;
    dbbh_t[X2] = bbh_t[VX2];
    dbbh_t[Y2] = bbh_t[VY2];
    dbbh_t[Z2] = 0;
    dbbh_t[VX1] = -om2 * bbh_t[X1];
    dbbh_t[VY1] = -om2 * bbh_t[Y1];
    dbbh_t[VZ1] = 0;
    dbbh_t[VX2] = -om2 * bbh_t[X2];
    dbbh_t[VY2] = -om2 * bbh_t[Y2];
    dbbh_t[VZ2] = 0;
    dbbh_t[AX1] = bbh.a1 * dspin_factor * e1x;
    dbbh_t[AY1] = bbh.a1 * dspin_factor * e1y;
    dbbh_t[AZ1] = bbh.a1 * dspin_factor * e1z;
    dbbh_t[AX2] = bbh.a2 * dspin_factor * e2x;
    dbbh_t[AY2] = bbh.a2 * dspin_factor * e2y;
    dbbh_t[AZ2] = bbh.a2 * dspin_factor * e2z;
    dbbh_t[M1T] = dbbh_t[M2T] = 0;
    return;
  }
  const auto &T = bbh_table.t;
  Real trange = T.back() - T.front();
  Real tol = 64.0 * std::numeric_limits<Real>::epsilon() *
             std::max({1.0, std::abs(T.front()), std::abs(T.back()), trange});
  if (t < T.front() - tol || t > T.back() + tol) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "requested time outside trajectory-table range" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::size_t i0, i1;
  std::size_t cached = std::min(bbh_table.active_segment, T.size() - 2);
  if (t >= T[cached] && t <= T[cached + 1]) {
    i0 = cached;
    i1 = cached + 1;
  } else {
    auto it = std::upper_bound(T.begin(), T.end(), t);
    if (it == T.begin()) {
      i0 = 0;
      i1 = 1;
    } else if (it == T.end()) {
      i0 = T.size() - 2;
      i1 = T.size() - 1;
    } else {
      i1 = it - T.begin();
      i0 = i1 - 1;
    }
    bbh_table.active_segment = i0;
  }
  Real dt = T[i1] - T[i0], w = (t - T[i0]) / dt;
  auto H = [w, dt](Real p0, Real v0, Real p1, Real v1, Real *p, Real *dp,
                   Real *ddp) {
    Real w2 = w * w, w3 = w2 * w;
    *p = (2 * w3 - 3 * w2 + 1) * p0 + (w3 - 2 * w2 + w) * dt * v0 +
         (-2 * w3 + 3 * w2) * p1 + (w3 - w2) * dt * v1;
    *dp = ((6 * w2 - 6 * w) * p0 + (3 * w2 - 4 * w + 1) * dt * v0 +
           (-6 * w2 + 6 * w) * p1 + (3 * w2 - 2 * w) * dt * v1) /
          dt;
    *ddp = ((12 * w - 6) * p0 + (6 * w - 4) * dt * v0 + (-12 * w + 6) * p1 +
            (6 * w - 2) * dt * v1) /
           (dt * dt);
  };
  H(bbh_table.x1[i0], bbh_table.vx1[i0], bbh_table.x1[i1], bbh_table.vx1[i1],
    &bbh_t[X1], &bbh_t[VX1], &dbbh_t[VX1]);
  H(bbh_table.y1[i0], bbh_table.vy1[i0], bbh_table.y1[i1], bbh_table.vy1[i1],
    &bbh_t[Y1], &bbh_t[VY1], &dbbh_t[VY1]);
  H(bbh_table.z1[i0], bbh_table.vz1[i0], bbh_table.z1[i1], bbh_table.vz1[i1],
    &bbh_t[Z1], &bbh_t[VZ1], &dbbh_t[VZ1]);
  H(bbh_table.x2[i0], bbh_table.vx2[i0], bbh_table.x2[i1], bbh_table.vx2[i1],
    &bbh_t[X2], &bbh_t[VX2], &dbbh_t[VX2]);
  H(bbh_table.y2[i0], bbh_table.vy2[i0], bbh_table.y2[i1], bbh_table.vy2[i1],
    &bbh_t[Y2], &bbh_t[VY2], &dbbh_t[VY2]);
  H(bbh_table.z2[i0], bbh_table.vz2[i0], bbh_table.z2[i1], bbh_table.vz2[i1],
    &bbh_t[Z2], &bbh_t[VZ2], &dbbh_t[VZ2]);
  dbbh_t[X1] = bbh_t[VX1];
  dbbh_t[Y1] = bbh_t[VY1];
  dbbh_t[Z1] = bbh_t[VZ1];
  dbbh_t[X2] = bbh_t[VX2];
  dbbh_t[Y2] = bbh_t[VY2];
  dbbh_t[Z2] = bbh_t[VZ2];
  auto linear = [w](Real f0, Real f1) { return (1.0 - w) * f0 + w * f1; };
  auto linear_deriv = [dt](Real f0, Real f1) { return (f1 - f0) / dt; };
  bbh_t[AX1] = linear(bbh_table.ax1[i0], bbh_table.ax1[i1]);
  bbh_t[AY1] = linear(bbh_table.ay1[i0], bbh_table.ay1[i1]);
  bbh_t[AZ1] = linear(bbh_table.az1[i0], bbh_table.az1[i1]);
  bbh_t[AX2] = linear(bbh_table.ax2[i0], bbh_table.ax2[i1]);
  bbh_t[AY2] = linear(bbh_table.ay2[i0], bbh_table.ay2[i1]);
  bbh_t[AZ2] = linear(bbh_table.az2[i0], bbh_table.az2[i1]);
  bbh_t[M1T] = linear(bbh_table.m1[i0], bbh_table.m1[i1]);
  bbh_t[M2T] = linear(bbh_table.m2[i0], bbh_table.m2[i1]);
  dbbh_t[AX1] = linear_deriv(bbh_table.ax1[i0], bbh_table.ax1[i1]);
  dbbh_t[AY1] = linear_deriv(bbh_table.ay1[i0], bbh_table.ay1[i1]);
  dbbh_t[AZ1] = linear_deriv(bbh_table.az1[i0], bbh_table.az1[i1]);
  dbbh_t[AX2] = linear_deriv(bbh_table.ax2[i0], bbh_table.ax2[i1]);
  dbbh_t[AY2] = linear_deriv(bbh_table.ay2[i0], bbh_table.ay2[i1]);
  dbbh_t[AZ2] = linear_deriv(bbh_table.az2[i0], bbh_table.az2[i1]);
  dbbh_t[M1T] = linear_deriv(bbh_table.m1[i0], bbh_table.m1[i1]);
  dbbh_t[M2T] = linear_deriv(bbh_table.m2[i0], bbh_table.m2[i1]);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T BoostGammaMinusOneOverV2(const T v2, const T gamma) {
  if (value_of(v2) < 1.0e-12) {
    return T(0.5) + T(0.375) * v2 + T(0.3125) * v2 * v2;
  }
  return (gamma - T(1.0)) / v2;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void BuildBoostJacobian(T vx, T vy, T vz,
                                               T J[NDIM][NDIM]) {
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = 0; j < NDIM; ++j)
      J[i][j] = T(0.0);
  T v2 = vx * vx + vy * vy + vz * vz;
  T g = T(1.0) / metric_sqrt(T(1.0) - v2);
  T q = BoostGammaMinusOneOverV2(v2, g);
  J[0][0] = g;
  J[0][1] = -g * vx;
  J[0][2] = -g * vy;
  J[0][3] = -g * vz;
  J[1][0] = J[0][1];
  J[2][0] = J[0][2];
  J[3][0] = J[0][3];
  J[1][1] = T(1.0) + q * vx * vx;
  J[1][2] = q * vx * vy;
  J[1][3] = q * vx * vz;
  J[2][1] = J[1][2];
  J[2][2] = T(1.0) + q * vy * vy;
  J[2][3] = q * vy * vz;
  J[3][1] = J[1][3];
  J[3][2] = J[2][3];
  J[3][3] = T(1.0) + q * vz * vz;
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void BoostedSpatialCoordinates(T x, T y, T z, T x0, T y0,
                                                      T z0, T vx, T vy, T vz,
                                                      T *xbh, T *ybh, T *zbh) {
  T dx = x - x0, dy = y - y0, dz = z - z0;
  T v2 = vx * vx + vy * vy + vz * vz;
  T g = T(1.0) / metric_sqrt(T(1.0) - v2);
  T q = BoostGammaMinusOneOverV2(v2, g);
  T vd = vx * dx + vy * dy + vz * dz;
  *xbh = dx + q * vx * vd;
  *ybh = dy + q * vy * vd;
  *zbh = dz + q * vz * vd;
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void
KerrSchildPerturbation(T x, T y, T z, T ax, T ay, T az, T m, T KS[NDIM][NDIM]) {
  T rt2 = T(1.4142135623730951), irt2 = T(1.0) / rt2,
    a2 = ax * ax + ay * ay + az * az, x2 = x * x + y * y + z * z,
    ad = ax * x + ay * y + az * z, term = x2 - a2,
    rho2 = term + metric_sqrt(T(4.0) * ad * ad + term * term),
    rho = metric_sqrt(rho2),
    fac = irt2 * rho2 * rho * m / (ad * ad + T(0.25) * rho2 * rho2),
    den = a2 + T(0.5) * rho2, iden = T(1.0) / den;
  T ell[3];
  ell[0] = y * az - z * ay + rt2 * ad * ax / rho + rho * x * irt2;
  ell[1] = -x * az + z * ax + rt2 * ad * ay / rho + rho * y * irt2;
  ell[2] = x * ay - y * ax + rt2 * ad * az / rho + rho * z * irt2;
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = 0; j < NDIM; ++j)
      KS[i][j] = T(0.0);
  KS[0][0] = fac;
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    KS[0][i + 1] = fac * ell[i] * iden;
    KS[i + 1][0] = KS[0][i + 1];
  }
  #pragma unroll
  for (int i = 0; i < 3; ++i)
    #pragma unroll
    for (int j = i; j < 3; ++j) {
      KS[i + 1][j + 1] = fac * ell[i] * ell[j] * iden * iden;
      KS[j + 1][i + 1] = KS[i + 1][j + 1];
    }
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void AddBoostedKerrSchildHole(const T KS[NDIM][NDIM],
                                                     const T J[NDIM][NDIM],
                                                     T gcov[NDIM][NDIM]) {
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = i; j < NDIM; ++j) {
      T sum = T(0.0);
      #pragma unroll
      for (int m = 0; m < NDIM; ++m)
        #pragma unroll
        for (int n = 0; n < NDIM; ++n)
          sum = sum + J[m][i] * J[n][j] * KS[m][n];
      gcov[i][j] = gcov[i][j] + sum;
      gcov[j][i] = gcov[i][j];
    }
}
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void
SuperposedBBHTemplate(T x, T y, T z, T gcov[NDIM][NDIM], const T tr[NTRAJ],
                      const bbh_pgen b) {
  T v1x = tr[VX1], v1y = tr[VY1], v1z = tr[VZ1], v2x = tr[VX2],
    v2y = tr[VY2], v2z = tr[VZ2];
  T a1x = tr[AX1], a1y = tr[AY1], a1z = tr[AZ1], a2x = tr[AX2], a2y = tr[AY2],
    a2z = tr[AZ2];
  T m1 = tr[M1T], m2 = tr[M2T];
  a1x = a1x * m1;
  a1y = a1y * m1;
  a1z = a1z * m1;
  a2x = a2x * m2;
  a2y = a2y * m2;
  a2z = a2z * m2;
  T a1n = metric_sqrt(a1x * a1x + a1y * a1y + a1z * a1z + T(1e-40)),
    a2n = metric_sqrt(a2x * a2x + a2y * a2y + a2z * a2z + T(1e-40)),
    a1 = a1n, a2 = a2n;
  T x1, y1, z1, x2, y2, z2;
  BoostedSpatialCoordinates(x, y, z, tr[X1], tr[Y1], tr[Z1], v1x, v1y, v1z, &x1,
                            &y1, &z1);
  BoostedSpatialCoordinates(x, y, z, tr[X2], tr[Y2], tr[Z2], v2x, v2y, v2z, &x2,
                            &y2, &z2);
  T r1 = metric_norm3(x1, y1, z1),
    r2 = metric_norm3(x2, y2, z2),
    c1 = metric_sqrt(a1 * a1) * (T(1.0) + b.a1_buffer) + b.cutoff_floor,
    c2 = metric_sqrt(a2 * a2) * (T(1.0) + b.a2_buffer) + b.cutoff_floor;
  if (value_of(r1) < value_of(c1))
    z1 = (value_of(z1) > 0.0) ? c1 : -c1;
  if (value_of(r2) < value_of(c2))
    z2 = (value_of(z2) > 0.0) ? c2 : -c2;
  T KS1[NDIM][NDIM], KS2[NDIM][NDIM], J1[NDIM][NDIM], J2[NDIM][NDIM];
  KerrSchildPerturbation(x1, y1, z1, a1x, a1y, a1z, m1, KS1);
  KerrSchildPerturbation(x2, y2, z2, a2x, a2y, a2z, m2, KS2);
  BuildBoostJacobian(v1x, v1y, v1z, J1);
  BuildBoostJacobian(v2x, v2y, v2z, J2);
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = 0; j < NDIM; ++j)
      gcov[i][j] = (i == j) ? ((i == 0) ? T(-1.0) : T(1.0)) : T(0.0);
  AddBoostedKerrSchildHole(KS1, J1, gcov);
  AddBoostedKerrSchildHole(KS2, J2, gcov);
}

template <typename V>
KOKKOS_FORCEINLINE_FUNCTION void BuildBoostJacobianMixed(V vx, V vy, V vz,
                                                    V J[NDIM][NDIM]) {
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = 0; j < NDIM; ++j)
      J[i][j] = V(0.0);
  auto v2 = vx * vx + vy * vy + vz * vz;
  auto g = V(1.0) / metric_sqrt(V(1.0) - v2);
  auto q = BoostGammaMinusOneOverV2(v2, g);
  J[0][0] = g;
  J[0][1] = -g * vx;
  J[0][2] = -g * vy;
  J[0][3] = -g * vz;
  J[1][0] = J[0][1];
  J[2][0] = J[0][2];
  J[3][0] = J[0][3];
  J[1][1] = V(1.0) + q * vx * vx;
  J[1][2] = q * vx * vy;
  J[1][3] = q * vx * vz;
  J[2][1] = J[1][2];
  J[2][2] = V(1.0) + q * vy * vy;
  J[2][3] = q * vy * vz;
  J[3][1] = J[1][3];
  J[3][2] = J[2][3];
  J[3][3] = V(1.0) + q * vz * vz;
}

template <typename C, typename P, typename G>
KOKKOS_FORCEINLINE_FUNCTION void
BoostedSpatialCoordinatesMixed(C x, C y, C z, P x0, P y0, P z0,
                               P vx, P vy, P vz, G *xbh, G *ybh, G *zbh) {
  auto dx = x - x0, dy = y - y0, dz = z - z0;
  auto v2 = vx * vx + vy * vy + vz * vz;
  auto g = P(1.0) / metric_sqrt(P(1.0) - v2);
  auto q = BoostGammaMinusOneOverV2(v2, g);
  auto vd = vx * dx + vy * dy + vz * dz;
  *xbh = dx + q * vx * vd;
  *ybh = dy + q * vy * vd;
  *zbh = dz + q * vz * vd;
}

template <typename C, typename P, typename G>
KOKKOS_FORCEINLINE_FUNCTION void
KerrSchildPerturbationMixed(C x, C y, C z, P ax, P ay, P az, P m,
                            G KS[NDIM][NDIM]) {
  G rt2 = G(1.4142135623730951), irt2 = G(1.0) / rt2;
  auto a2 = ax * ax + ay * ay + az * az;
  auto x2 = x * x + y * y + z * z;
  auto ad = ax * x + ay * y + az * z;
  auto term = x2 - a2;
  auto rho2 = term + metric_sqrt(G(4.0) * ad * ad + term * term);
  auto rho = metric_sqrt(rho2);
  auto fac = irt2 * rho2 * rho * m /
             (ad * ad + G(0.25) * rho2 * rho2);
  auto den = a2 + G(0.5) * rho2;
  auto iden = G(1.0) / den;
  G ell[3];
  ell[0] = y * az - z * ay + rt2 * ad * ax / rho + rho * x * irt2;
  ell[1] = -x * az + z * ax + rt2 * ad * ay / rho + rho * y * irt2;
  ell[2] = x * ay - y * ax + rt2 * ad * az / rho + rho * z * irt2;
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = 0; j < NDIM; ++j)
      KS[i][j] = G(0.0);
  KS[0][0] = fac;
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    KS[0][i + 1] = fac * ell[i] * iden;
    KS[i + 1][0] = KS[0][i + 1];
  }
  #pragma unroll
  for (int i = 0; i < 3; ++i)
    #pragma unroll
    for (int j = i; j < 3; ++j) {
      KS[i + 1][j + 1] = fac * ell[i] * ell[j] * iden * iden;
      KS[j + 1][i + 1] = KS[i + 1][j + 1];
    }
}

template <typename K, typename J, typename G>
KOKKOS_FORCEINLINE_FUNCTION void
AddBoostedKerrSchildHoleMixed(const K KS[NDIM][NDIM],
                              const J Jmat[NDIM][NDIM],
                              G gcov[NDIM][NDIM]) {
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = i; j < NDIM; ++j) {
      G sum = G(0.0);
      #pragma unroll
      for (int m = 0; m < NDIM; ++m)
        #pragma unroll
        for (int n = 0; n < NDIM; ++n)
          sum = sum + Jmat[m][i] * Jmat[n][j] * KS[m][n];
      gcov[i][j] = gcov[i][j] + sum;
      gcov[j][i] = gcov[i][j];
    }
}

template <typename C, typename P, typename G>
KOKKOS_FORCEINLINE_FUNCTION void
SuperposedBBHTemplateMixed(C x, C y, C z, G gcov[NDIM][NDIM],
                           const P tr[NTRAJ], const bbh_pgen b) {
  auto v1x = tr[VX1], v1y = tr[VY1], v1z = tr[VZ1];
  auto v2x = tr[VX2], v2y = tr[VY2], v2z = tr[VZ2];
  auto a1x = tr[AX1], a1y = tr[AY1], a1z = tr[AZ1];
  auto a2x = tr[AX2], a2y = tr[AY2], a2z = tr[AZ2];
  auto m1 = tr[M1T];
  auto m2 = tr[M2T];
  a1x = a1x * m1;
  a1y = a1y * m1;
  a1z = a1z * m1;
  a2x = a2x * m2;
  a2y = a2y * m2;
  a2z = a2z * m2;
  auto a1 = metric_sqrt(a1x * a1x + a1y * a1y + a1z * a1z + P(1e-40));
  auto a2 = metric_sqrt(a2x * a2x + a2y * a2y + a2z * a2z + P(1e-40));

  G x1, y1, z1, x2, y2, z2;
  BoostedSpatialCoordinatesMixed<C, P, G>(
      x, y, z, tr[X1], tr[Y1], tr[Z1], v1x, v1y, v1z, &x1, &y1, &z1);
  BoostedSpatialCoordinatesMixed<C, P, G>(
      x, y, z, tr[X2], tr[Y2], tr[Z2], v2x, v2y, v2z, &x2, &y2, &z2);

  auto r1 = metric_norm3(x1, y1, z1);
  auto r2 = metric_norm3(x2, y2, z2);
  auto c1 = metric_sqrt(a1 * a1) * (P(1.0) + b.a1_buffer) + b.cutoff_floor;
  auto c2 = metric_sqrt(a2 * a2) * (P(1.0) + b.a2_buffer) + b.cutoff_floor;
  if (value_of(r1) < value_of(c1))
    z1 = (value_of(z1) > 0.0) ? c1 : -c1;
  if (value_of(r2) < value_of(c2))
    z2 = (value_of(z2) > 0.0) ? c2 : -c2;

  using JType = decltype(v1x + v1y + v1z);
  G KS1[NDIM][NDIM], KS2[NDIM][NDIM];
  JType J1[NDIM][NDIM], J2[NDIM][NDIM];
  KerrSchildPerturbationMixed<G, P, G>(x1, y1, z1, a1x, a1y, a1z, m1, KS1);
  KerrSchildPerturbationMixed<G, P, G>(x2, y2, z2, a2x, a2y, a2z, m2, KS2);
  BuildBoostJacobianMixed(v1x, v1y, v1z, J1);
  BuildBoostJacobianMixed(v2x, v2y, v2z, J2);
  #pragma unroll
  for (int i = 0; i < NDIM; ++i)
    #pragma unroll
    for (int j = 0; j < NDIM; ++j)
      gcov[i][j] = (i == j) ? ((i == 0) ? G(-1.0) : G(1.0)) : G(0.0);
  AddBoostedKerrSchildHoleMixed(KS1, J1, gcov);
  AddBoostedKerrSchildHoleMixed(KS2, J2, gcov);
}
KOKKOS_INLINE_FUNCTION
void
SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
              Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen b)
{
  (void)time;
  SuperposedBBHTemplate<Real>(x, y, z, gcov, traj_array, b);
}

KOKKOS_FORCEINLINE_FUNCTION
void FillMetricValue(const dual1_real gcov[NDIM][NDIM], struct dd_sym &g) {
  g.tt = value_of(gcov[TT][TT]);
  g.tx = value_of(gcov[TT][XX]);
  g.ty = value_of(gcov[TT][YY]);
  g.tz = value_of(gcov[TT][ZZ]);
  g.xx = value_of(gcov[XX][XX]);
  g.xy = value_of(gcov[XX][YY]);
  g.xz = value_of(gcov[XX][ZZ]);
  g.yy = value_of(gcov[YY][YY]);
  g.yz = value_of(gcov[YY][ZZ]);
  g.zz = value_of(gcov[ZZ][ZZ]);
}

KOKKOS_FORCEINLINE_FUNCTION
void FillMetricDerivative(const dual1_real gcov[NDIM][NDIM],
                          struct dd_sym &dg) {
  dg.tt = deriv_of(gcov[TT][TT]);
  dg.tx = deriv_of(gcov[TT][XX]);
  dg.ty = deriv_of(gcov[TT][YY]);
  dg.tz = deriv_of(gcov[TT][ZZ]);
  dg.xx = deriv_of(gcov[XX][XX]);
  dg.xy = deriv_of(gcov[XX][YY]);
  dg.xz = deriv_of(gcov[XX][ZZ]);
  dg.yy = deriv_of(gcov[YY][YY]);
  dg.yz = deriv_of(gcov[YY][ZZ]);
  dg.zz = deriv_of(gcov[ZZ][ZZ]);
}

KOKKOS_INLINE_FUNCTION
void DifferenceMetric(const struct dd_sym &plus, const struct dd_sym &minus,
                      const Real denom, struct dd_sym &dg) {
  Real inv = 1.0/denom;
  dg.tt = (plus.tt - minus.tt)*inv;
  dg.tx = (plus.tx - minus.tx)*inv;
  dg.ty = (plus.ty - minus.ty)*inv;
  dg.tz = (plus.tz - minus.tz)*inv;
  dg.xx = (plus.xx - minus.xx)*inv;
  dg.xy = (plus.xy - minus.xy)*inv;
  dg.xz = (plus.xz - minus.xz)*inv;
  dg.yy = (plus.yy - minus.yy)*inv;
  dg.yz = (plus.yz - minus.yz)*inv;
  dg.zz = (plus.zz - minus.zz)*inv;
}

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
                       const Real z, struct four_metric &outmet,
                       const Real traj_m[NTRAJ], const Real traj_0[NTRAJ],
                       const Real traj_p[NTRAJ], const Real hm, const Real hp,
                       const bbh_pgen b) {
  struct four_metric met_m, met_p;
  Real hx = b.metric_fd_step;

  get_metric(t, x, y, z, outmet, traj_0, b);

  if (hm > 0.0 && hp > 0.0) {
    get_metric(t - hm, x, y, z, met_m, traj_m, b);
    get_metric(t + hp, x, y, z, met_p, traj_p, b);
    DifferenceMetric(met_p.g, met_m.g, hm + hp, outmet.g_t);
  } else if (hp > 0.0) {
    get_metric(t + hp, x, y, z, met_p, traj_p, b);
    DifferenceMetric(met_p.g, outmet.g, hp, outmet.g_t);
  } else {
    get_metric(t - hm, x, y, z, met_m, traj_m, b);
    DifferenceMetric(outmet.g, met_m.g, hm, outmet.g_t);
  }

  get_metric(t, x - hx, y, z, met_m, traj_0, b);
  get_metric(t, x + hx, y, z, met_p, traj_0, b);
  DifferenceMetric(met_p.g, met_m.g, 2.0*hx, outmet.g_x);

  get_metric(t, x, y - hx, z, met_m, traj_0, b);
  get_metric(t, x, y + hx, z, met_p, traj_0, b);
  DifferenceMetric(met_p.g, met_m.g, 2.0*hx, outmet.g_y);

  get_metric(t, x, y, z - hx, met_m, traj_0, b);
  get_metric(t, x, y, z + hx, met_p, traj_0, b);
  DifferenceMetric(met_p.g, met_m.g, 2.0*hx, outmet.g_z);
}

void TimeMetricAD(const Real x, const Real y, const Real z,
                  const Real tr[NTRAJ], const Real dtr[NTRAJ],
                  const bbh_pgen b, struct dd_sym &g, struct dd_sym &dg) {
  dual1_real gcov[NDIM][NDIM];
  dual1_real td[NTRAJ];
  for (int n = 0; n < NTRAJ; ++n)
    td[n] = dual1_real(tr[n], dtr[n]);
  SuperposedBBHTemplateMixed<Real, dual1_real, dual1_real>(x, y, z, gcov, td,
                                                           b);
  FillMetricValue(gcov, g);
  FillMetricDerivative(gcov, dg);
}

KOKKOS_FORCEINLINE_FUNCTION
void SpatialMetricAD(const Real x, const Real y, const Real z, const int dir,
                     const Real tr[NTRAJ], const bbh_pgen b,
                     struct dd_sym &dg) {
  dual1_real gcov[NDIM][NDIM];
  dual1_real xd(x, dir == 0 ? 1.0 : 0.0);
  dual1_real yd(y, dir == 1 ? 1.0 : 0.0);
  dual1_real zd(z, dir == 2 ? 1.0 : 0.0);
  SuperposedBBHTemplateMixed<dual1_real, Real, dual1_real>(
      xd, yd, zd, gcov, tr, b);
  FillMetricDerivative(gcov, dg);
}

KOKKOS_FORCEINLINE_FUNCTION void
get_metric_and_derivatives(const Real t, const Real x, const Real y,
                           const Real z, struct four_metric &met,
                           const Real tr[NTRAJ], const Real dtr[NTRAJ],
                           const bbh_pgen b) {
  (void)t;
  TimeMetricAD(x, y, z, tr, dtr, b, met.g, met.g_t);
  SpatialMetricAD(x, y, z, 0, tr, b, met.g_x);
  SpatialMetricAD(x, y, z, 1, tr, b, met.g_y);
  SpatialMetricAD(x, y, z, 2, tr, b, met.g_z);
}

KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen bbh_) {
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

  // note: we need this to prevent capture by this in the lambda expr.
  auto bbh_ = bbh;

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
  const Real tracker_radius1 = bbh_ref.tracker_radius[0];
  const Real tracker_radius2 = bbh_ref.tracker_radius[1];
  const int tracker_reflevel1 = bbh_ref.tracker_reflevel[0];
  const int tracker_reflevel2 = bbh_ref.tracker_reflevel[1];
  for (int m = 0; m < nmb; ++m) {

    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

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

    bool in_tracker1 = dmin2_bh1 < SQR(tracker_radius1) || iscontained_bh1;
    bool in_tracker2 = dmin2_bh2 < SQR(tracker_radius2) || iscontained_bh2;
    bool unlimited = (in_tracker1 && tracker_reflevel1 < 0) ||
                     (in_tracker2 && tracker_reflevel2 < 0);
    int target_level = -1;
    if (in_tracker1) target_level = std::max(target_level, tracker_reflevel1);
    if (in_tracker2) target_level = std::max(target_level, tracker_reflevel2);

    if (unlimited) {
      refine_flag.h_view(m + mbs) = 1;
    } else if (target_level >= 0) {
      if (level < target_level) {
        refine_flag.h_view(m + mbs) = 1;
      } else if (level == target_level) {
        refine_flag.h_view(m + mbs) = 0;
      } else {
        refine_flag.h_view(m + mbs) = -1;
      }
    } else {
      refine_flag.h_view(m + mbs) = -1;
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// Enforce some minimum resolution within a certain spherical region
void RefineRadii(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    Real r2[8] = {
      SQR(x1min) + SQR(x2min) + SQR(x3min),
      SQR(x1max) + SQR(x2min) + SQR(x3min),
      SQR(x1min) + SQR(x2max) + SQR(x3min),
      SQR(x1max) + SQR(x2max) + SQR(x3min),
      SQR(x1min) + SQR(x2min) + SQR(x3max),
      SQR(x1max) + SQR(x2min) + SQR(x3max),
      SQR(x1min) + SQR(x2max) + SQR(x3max),
      SQR(x1max) + SQR(x2max) + SQR(x3max),
    };
    Real rmin2 = *std::min_element(&r2[0], &r2[8]);

    for (int ir = 0; ir < bbh_ref.radius.size(); ++ir) {
      if (rmin2 < SQR(bbh_ref.radius[ir])) {
        if (level < bbh_ref.reflevel[ir]) {
          refine_flag.h_view(m + mbs) = 1;
        } else if (level == bbh_ref.reflevel[ir] && refine_flag.h_view(m + mbs) == -1) {
          refine_flag.h_view(m + mbs) = 0;
        }
      }
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// 1: refines, -1: de-refines, 0: does nothing
void Refine(MeshBlockPack *pmy_pack) {
  if (bbh_ref.AlphaMin) {
    RefineAlphaMin(pmy_pack);
  } else if (bbh_ref.Tracker) {
    RefineTracker(pmy_pack);
  }
  RefineRadii(pmy_pack);
}

//nere hardcoding zero spin
KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bbh_pgen pgen,
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
  if (r > 0.0) {
    *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
    *pphi = atan2(r*x2, r*x1);
  } else {
    *ptheta = 0.0;
    *pphi = 0.0;
  }
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
static Real CalculateCovariantUT(struct bbh_pgen pgen, Real r, Real sin_theta, Real l) {
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
static Real LogHAux(struct bbh_pgen pgen, Real r, Real sin_theta) {
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
  if (isfinite(hh) && hh >= 1.0) {
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
static Real CalculateT(struct bbh_pgen pgen, Real rho, Real ptot_over_rho) {
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
static void CalculateCN(struct bbh_pgen pgen, Real *cparam, Real *nparam) {
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
static Real CalculateL(struct bbh_pgen pgen, Real r, Real sin_theta) {
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
static void CalculateVectorPotentialInTiltedTorus(struct bbh_pgen pgen,
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
Real A1(struct bbh_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if (r <= 1.0e-14) {
    return 0.0;
  }

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real cyl2 = SQR(x1) + SQR(x2);
  Real safe_cyl2 = fmax(cyl2, 1.0e-12);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/safe_cyl2);

  return atheta*(x1*x3*isin_term/(r*sqrt_term)) +
         aphi*(-x2/safe_cyl2 +
               pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
  //return -0.5*x2;
}

//----------------------------------------------------------------------------------------
// Function to compute 2-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A2(struct bbh_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  //Real r, theta, phi;
  //GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if (r <= 1.0e-14) {
    return 0.0;
  }

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real cyl2 = SQR(x1) + SQR(x2);
  Real safe_cyl2 = fmax(cyl2, 1.0e-12);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/safe_cyl2);

  return atheta*(x2*x3*isin_term/(r*sqrt_term)) +
         aphi*(x1/safe_cyl2 +
               pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
  //return 0.5*x1;
}

//----------------------------------------------------------------------------------------
// Function to compute 3-component of vector potential. See comments for A1.

KOKKOS_INLINE_FUNCTION
Real A3(struct bbh_pgen pgen, Real x1, Real x2, Real x3) {
  // BL coordinates
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if (r <= 1.0e-14) {
    return 0.0;
  }

  // calculate vector potential in spherical KS
  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real cyl2 = SQR(x1) + SQR(x2);
  Real safe_cyl2 = fmax(cyl2, 1.0e-12);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/safe_cyl2);

  return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) +
         aphi*(pgen.spin*x3/(r*sqrt_term));


  return 0.0;
}

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct bbh_pgen pgen,
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
static void CalculateVelocityInTorus(struct bbh_pgen pgen,
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
static void TransformVector(struct bbh_pgen pgen,
                            Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) + sqrt(SQR(SQR(rad))) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r;
  Real cyl2 = SQR(x1) + SQR(x2);
  Real safe_cyl = sqrt(fmax(cyl2, 1.0e-12));
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1)/(SQR(r))) +
         a2_bl * x1*x3/safe_cyl -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2)/(SQR(r))) +
         a2_bl * x2*x3/safe_cyl +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * sqrt(cyl2);
  return;
}


KOKKOS_INLINE_FUNCTION
static void GetSuperposedAndInverse(const Real t,
                            const Real x, const Real y, const Real z,
                            Real gcov[][NDIM], Real gcon[][NDIM], const Real bbh_traj_loc[NTRAJ],
                            const bbh_pgen bbh_){
  //Real gcov[NDIM][NDIM];
  //Real gcon[NDIM][NDIM];
  SuperposedBBH(t, x, y, z, gcov, bbh_traj_loc, bbh_);
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

KOKKOS_INLINE_FUNCTION
Real SmoothExcisionBWeight(Real w) {
  return fmin(fmax(w, 0.0), 1.0);
}

KOKKOS_INLINE_FUNCTION
Real StrictSmoothExcisionBWeight(const Real w0, const Real w1) {
  if (!isfinite(w0) || !isfinite(w1)) return 0.0;
  return SmoothExcisionBWeight(fmin(w0, w1));
}

KOKKOS_INLINE_FUNCTION
Real StrictSmoothExcisionBWeight(const Real w0, const Real w1,
                                 const Real w2, const Real w3) {
  if (!isfinite(w0) || !isfinite(w1) || !isfinite(w2) || !isfinite(w3)) {
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

  auto &indcs = pm->mb_indcs;
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
  auto &mbsize = pmbp->pmb->mb_size;
  bool multi_d = pm->multi_d;
  bool three_d = pm->three_d;
  Real eta0 = bbh.smooth_b_damping_eta;
  Real cfl_cap = bbh.smooth_b_damping_cfl;
  Real dt = pm->dt;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  if (pm->one_d) {
    par_for_outer("dynbbh_b_damp1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);
      auto size = mbsize.d_view(m);
      Real eta = SmoothExcisionDampingEta(size, multi_d, three_d, eta0, cfl_cap, dt);
	      CurrentDensity(member, m, ks, js, is, ie+1, b0, size, j1, j2, j3);
	      par_for_inner(member, is, ie+1, [&](const int i) {
	        Real w = EdgeWeightX1D(weight, m, ks, js, i);
	        if (w > 0.0 && isfinite(w)) {
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
      auto size = mbsize.d_view(m);
      Real eta = SmoothExcisionDampingEta(size, multi_d, three_d, eta0, cfl_cap, dt);
	      CurrentDensity(member, m, ks, j, is, ie+1, b0, size, j1, j2, j3);
	      par_for_inner(member, is, ie+1, [&](const int i) {
	        Real w1 = StrictSmoothExcisionBWeight(weight(m,ks,j,i),
	                                             weight(m,ks,j-1,i));
	        if (w1 > 0.0 && isfinite(w1)) {
	          e1(m,ks,  j,i) += eta*w1*j1(i);
	          e1(m,ke+1,j,i) += eta*w1*j1(i);
	        }
	        Real w2 = EdgeWeightX1D(weight, m, ks, j, i);
	        if (w2 > 0.0 && isfinite(w2)) {
	          e2(m,ks,  j,i) += eta*w2*j2(i);
	          e2(m,ke+1,j,i) += eta*w2*j2(i);
	        }
	        Real w3 = EdgeWeightX3(weight, m, ks, j, i);
	        if (w3 > 0.0 && isfinite(w3)) {
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
    auto size = mbsize.d_view(m);
	    Real eta = SmoothExcisionDampingEta(size, multi_d, three_d, eta0, cfl_cap, dt);
	    CurrentDensity(member, m, k, j, is, ie+1, b0, size, j1, j2, j3);
	    par_for_inner(member, is, ie+1, [&](const int i) {
	      Real w1 = EdgeWeightX1(weight, m, k, j, i);
	      if (w1 > 0.0 && isfinite(w1)) {
	        e1(m,k,j,i) += eta*w1*j1(i);
	      }
	      Real w2 = EdgeWeightX2(weight, m, k, j, i);
	      if (w2 > 0.0 && isfinite(w2)) {
	        e2(m,k,j,i) += eta*w2*j2(i);
	      }
	      Real w3 = EdgeWeightX3(weight, m, k, j, i);
	      if (w3 > 0.0 && isfinite(w3)) {
	        e3(m,k,j,i) += eta*w3*j3(i);
	      }
	    });
	  });
	}

//----------------------------------------------------------------------------------------
// Flash-beam helpers.  The construction below mirrors the validated single-hole
// rad_kerr_orbit_beam test (src/pgen/tests/rad_beam.cpp): the launch point sits on the
// equatorial photon orbit at the Cartesian Kerr-Schild radius sqrt(r_ph^2 + a^2), the
// beam axis is the ring-tangent coordinate direction null-completed against the actual
// (superposed, boosted) metric, converted to the Eulerian tetrad frame with the same
// Cholesky triad used by dyn_radiation, and moment-projected onto <=3 angular-grid
// directions so the discrete first moment is exactly parallel to the requested ray.

//! \brief Equatorial circular photon-orbit radius in Cartesian Kerr-Schild coordinates
//! for a hole of mass m and dimensionless spin chi.  sense=+1 selects the prograde
//! (inner) orbit, sense=-1 the retrograde (outer) orbit; the Boyer-Lindquist radius is
//! r = 2m(1 + cos((2/3) acos(∓chi))) and the CKS cylindrical radius is sqrt(r^2 + a^2).
Real FlashPhotonRingRadiusCKS(const Real mass, const Real chi, const Real sense) {
  const Real arg = fmax(-1.0, fmin(1.0, (sense > 0.0) ? -fabs(chi) : fabs(chi)));
  const Real r_bl = 2.0*(1.0 + std::cos((2.0/3.0)*std::acos(arg)));
  return mass*std::sqrt(SQR(r_bl) + SQR(chi));
}

//! \brief Four-metric (lower indices) of the superposed BBH spacetime at (t,x,y,z).
void FlashMetricAt(const Real t, const Real x, const Real y, const Real z,
                   Real g4[NDIM][NDIM]) {
  Real traj[NTRAJ];
  find_traj_t(t, traj);
  SuperposedBBH(t, x, y, z, g4, traj, bbh);
}

//! \brief Completes coordinate direction dir[3] to a future-pointing null 4-vector and
//! returns its covariant components k_mu = g_{mu nu} d^nu (with d^0 from the null
//! condition).  Returns false if the direction is not null-realizable.
bool FlashCovariantNull(const Real g4[NDIM][NDIM], const Real dir[3], Real k_cov[NDIM]) {
  const Real ta = g4[0][0];
  const Real tb = 2.0*(g4[0][1]*dir[0] + g4[0][2]*dir[1] + g4[0][3]*dir[2]);
  const Real tc = g4[1][1]*dir[0]*dir[0] + 2.0*g4[1][2]*dir[0]*dir[1]
                + 2.0*g4[1][3]*dir[0]*dir[2] + g4[2][2]*dir[1]*dir[1]
                + 2.0*g4[2][3]*dir[1]*dir[2] + g4[3][3]*dir[2]*dir[2];
  const Real disc = tb*tb - 4.0*ta*tc;
  if (!(disc > 0.0)) return false;
  const Real d0 = (-tb - std::sqrt(disc))/(2.0*ta);
  const Real d[NDIM] = {d0, dir[0], dir[1], dir[2]};
  for (int mu=0; mu<NDIM; ++mu) {
    k_cov[mu] = 0.0;
    for (int nu=0; nu<NDIM; ++nu) {
      k_cov[mu] += g4[mu][nu]*d[nu];
    }
  }
  return true;
}

//! \brief Converts a coordinate direction at a point of the superposed BBH metric into
//! Eulerian-tetrad direction cosines ell[3], using the same Cholesky triad construction
//! as dyn_radiation's tetrad (adapted from CoordinateDirectionToADMTetrad in the
//! single-hole test, evaluated on the superposed metric instead of a single Kerr hole).
bool FlashDirectionToTetrad(const Real g4[NDIM][NDIM], const Real dir[3], Real ell[3]) {
  Real k_cov[NDIM];
  if (!FlashCovariantNull(g4, dir, k_cov)) return false;

  // ADM decomposition of g4
  const Real gxx = g4[1][1], gxy = g4[1][2], gxz = g4[1][3];
  const Real gyy = g4[2][2], gyz = g4[2][3], gzz = g4[3][3];
  Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
  Real gu[6];
  adm::SpatialInv(1.0/detg, gxx, gxy, gxz, gyy, gyz, gzz,
                  &gu[0], &gu[1], &gu[2], &gu[3], &gu[4], &gu[5]);
  const Real beta_l[3] = {g4[0][1], g4[0][2], g4[0][3]};
  const Real beta_u[3] = {gu[0]*beta_l[0] + gu[1]*beta_l[1] + gu[2]*beta_l[2],
                          gu[1]*beta_l[0] + gu[3]*beta_l[1] + gu[4]*beta_l[2],
                          gu[2]*beta_l[0] + gu[4]*beta_l[1] + gu[5]*beta_l[2]};
  const Real beta2 = beta_l[0]*beta_u[0] + beta_l[1]*beta_u[1] + beta_l[2]*beta_u[2];
  const Real alpha = std::sqrt(fmax(beta2 - g4[0][0], 1.0e-30));

  // Cholesky triad (columns map orthonormal directions to coordinate vectors); this is
  // BuildADMSpatialTriad from dyn_radiation_tetrad.cpp.
  constexpr Real mfloor = 1.0e-30;
  const Real l00 = std::sqrt(fmax(gxx, mfloor));
  const Real l10 = gxy/l00;
  const Real l20 = gxz/l00;
  const Real l11 = std::sqrt(fmax(gyy - SQR(l10), mfloor));
  const Real l21 = (gyz - l20*l10)/l11;
  const Real l22 = std::sqrt(fmax(gzz - SQR(l20) - SQR(l21), mfloor));
  Real triad[3][3];
  triad[0][0] = 1.0/l00; triad[1][0] = 0.0;      triad[2][0] = 0.0;
  triad[0][1] = -l10/(l00*l11); triad[1][1] = 1.0/l11; triad[2][1] = 0.0;
  triad[0][2] = l10*l21/(l00*l11*l22) - l20/(l00*l22);
  triad[1][2] = -l21/(l11*l22);
  triad[2][2] = 1.0/l22;

  // photon energy measured by the Eulerian observer
  const Real e0_mu[NDIM] = {1.0/alpha, -beta_u[0]/alpha, -beta_u[1]/alpha,
                            -beta_u[2]/alpha};
  Real energy = 0.0;
  for (int mu=0; mu<NDIM; ++mu) {
    energy -= e0_mu[mu]*k_cov[mu];
  }
  if (!(energy > 0.0)) return false;
  for (int a=0; a<3; ++a) {
    ell[a] = 0.0;
    for (int i=0; i<3; ++i) {
      ell[a] += triad[i][a]*k_cov[i+1];
    }
    ell[a] /= energy;
  }
  return true;
}

//! \brief Positive angular weights on <=3 grid directions whose first moment is exactly
//! parallel to the requested tetrad-frame ray (convex-hull moment projection; local
//! adaptation of SetProjectedAngularWeights from the single-hole test).
bool FlashProjectedWeights(const DualArray2D<Real> &nh_c, const int nangles,
                           const Real ell[3], std::vector<Real> &weights) {
  weights.assign(nangles, 0.0);
  const Real qn = std::sqrt(SQR(ell[0]) + SQR(ell[1]) + SQR(ell[2]));
  if (!(qn > 0.0)) return false;
  const Real qx = ell[0]/qn, qy = ell[1]/qn, qz = ell[2]/qn;

  std::vector<std::pair<Real,int>> ranked;
  ranked.reserve(nangles);
  for (int n=0; n<nangles; ++n) {
    ranked.emplace_back(nh_c.h_view(n,1)*qx + nh_c.h_view(n,2)*qy
                        + nh_c.h_view(n,3)*qz, n);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  const int ncand = std::min(nangles, 80);
  int best[3] = {ranked[0].second, ranked[0].second, ranked[0].second};
  Real best_lam[3] = {1.0, 0.0, 0.0};
  Real best_r = -1.0;
  for (int ia=0; ia<ncand; ++ia) {
    const int aidx = ranked[ia].second;
    for (int ib=ia+1; ib<ncand; ++ib) {
      const int bidx = ranked[ib].second;
      for (int ic=ib+1; ic<ncand; ++ic) {
        const int cidx = ranked[ic].second;
        // solve [na nb nc][lam] = r*q with lam summing to 1 (4x4 gaussian elimination)
        Real a[4][5] = {
          {nh_c.h_view(aidx,1), nh_c.h_view(bidx,1), nh_c.h_view(cidx,1), -qx, 0.0},
          {nh_c.h_view(aidx,2), nh_c.h_view(bidx,2), nh_c.h_view(cidx,2), -qy, 0.0},
          {nh_c.h_view(aidx,3), nh_c.h_view(bidx,3), nh_c.h_view(cidx,3), -qz, 0.0},
          {1.0, 1.0, 1.0, 0.0, 1.0},
        };
        Real sol[4];
        bool ok = true;
        for (int col=0; col<4 && ok; ++col) {
          int piv = col;
          Real mx = std::fabs(a[col][col]);
          for (int r=col+1; r<4; ++r) {
            if (std::fabs(a[r][col]) > mx) { mx = std::fabs(a[r][col]); piv = r; }
          }
          if (mx < 1.0e-14) { ok = false; break; }
          if (piv != col) {
            for (int c=col; c<5; ++c) std::swap(a[col][c], a[piv][c]);
          }
          const Real inv = 1.0/a[col][col];
          for (int c=col; c<5; ++c) a[col][c] *= inv;
          for (int r=0; r<4; ++r) {
            if (r == col) continue;
            const Real f = a[r][col];
            for (int c=col; c<5; ++c) a[r][c] -= f*a[col][c];
          }
        }
        if (!ok) continue;
        for (int r=0; r<4; ++r) sol[r] = a[r][4];
        const Real min_lam = std::min(sol[0], std::min(sol[1], sol[2]));
        const Real r = sol[3];
        if (min_lam >= -1.0e-10 && r > best_r && r <= 1.0 + 1.0e-10) {
          best[0] = aidx; best[1] = bidx; best[2] = cidx;
          best_lam[0] = std::max(sol[0], 0.0);
          best_lam[1] = std::max(sol[1], 0.0);
          best_lam[2] = std::max(sol[2], 0.0);
          best_r = r;
        }
      }
    }
  }
  const Real sum_lam = best_lam[0] + best_lam[1] + best_lam[2];
  if (best_r <= 0.0 || sum_lam <= 0.0) return false;
  for (int n=0; n<3; ++n) {
    weights[best[n]] += best_lam[n]/sum_lam;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void FlashLaunchGeometry()
//! \brief Returns the launch point and (unit, in-plane) ring-tangent coordinate aim
//! direction of the flash beam at time t.  The launch point sits on the chosen orbiting
//! hole's equatorial photon orbit at the CKS ring radius (retrograde by default) and
//! optionally co-rotates rigidly with the binary; the aim is the local tangent in the
//! chosen orbital sense, optionally rotated by aim_offset.

void FlashLaunchGeometry(const Real t, Real pos[3], Real dir[3]) {
  Real traj[NTRAJ];
  find_traj_t(t, traj);
  Real cx, cy, cz, vx, vy, vz, m_src, chi_src;
  if (flash.src_bh == 2) {
    cx = traj[X2]; cy = traj[Y2]; cz = traj[Z2];
    vx = traj[VX2]; vy = traj[VY2]; vz = traj[VZ2];
    m_src = traj[M2T];
    chi_src = std::sqrt(SQR(traj[AX2]) + SQR(traj[AY2]) + SQR(traj[AZ2]));
  } else {
    cx = traj[X1]; cy = traj[Y1]; cz = traj[Z1];
    vx = traj[VX1]; vy = traj[VY1]; vz = traj[VZ1];
    m_src = traj[M1T];
    chi_src = std::sqrt(SQR(traj[AX1]) + SQR(traj[AY1]) + SQR(traj[AZ1]));
  }
  Real rring = (flash.ring_radius > 0.0)
             ? flash.ring_radius
             : FlashPhotonRingRadiusCKS(m_src, chi_src, flash.sense);
  Real dphi = flash.corotate ? bbh.om*(t - flash.t0) : 0.0;
  Real ra = flash.ring_angle + dphi;
  pos[0] = cx + rring*flash.ring_frac*std::cos(ra);
  pos[1] = cy + rring*flash.ring_frac*std::sin(ra);
  pos[2] = cz;

  // Ring tangent in the HOLE'S REST FRAME: +phi for prograde (sense=+1), -phi for
  // retrograde (sense=-1), optionally rotated in-plane by aim_offset.  The hole is a
  // boosted Kerr-Schild puncture moving at v~0.25c, so a rest-frame tangent must be
  // aberrated into the global coordinate frame (relativistic velocity addition for
  // light); without this the launched ray does not load onto the moving hole's ring.
  Real ta = ra + flash.sense*0.5*M_PI + flash.aim_offset;
  Real np[3] = {std::cos(ta), std::sin(ta), 0.0};
  Real v2 = vx*vx + vy*vy + vz*vz;
  if (v2 > 0.0 && v2 < 1.0) {
    Real gam = 1.0/std::sqrt(1.0 - v2);
    Real vdotn = vx*np[0] + vy*np[1] + vz*np[2];
    Real denom = 1.0 + vdotn;
    Real fac = 1.0 + (gam/(gam + 1.0))*vdotn;
    Real nn[3] = {(np[0]/gam + vx*fac)/denom,
                  (np[1]/gam + vy*fac)/denom,
                  (np[2]/gam + vz*fac)/denom};
    Real nl = std::sqrt(SQR(nn[0]) + SQR(nn[1]) + SQR(nn[2]));
    dir[0] = nn[0]/nl; dir[1] = nn[1]/nl; dir[2] = nn[2]/nl;
  } else {
    dir[0] = np[0]; dir[1] = np[1]; dir[2] = np[2];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DynBBHFlashBeamSource()
//! \brief Custom radiation source term for the orbiting BBH background.  A compact,
//! exponentially-decaying "flash" of radiation is injected on one BH's equatorial photon
//! ring, along the local ring-tangent null direction (retrograde by default).  The launch
//! point, tangent direction, and the moment-projected angular weights are re-derived from
//! the CURRENT (moving) metric/tetrad every substage, so the source stays locked to the
//! orbiting photon ring.  After t=toff nothing is injected and the radiation
//! free-streams.
//!
//! This replaces the generic <rad_srcterms> BeamSource, which is wrong here because it
//! injects continuously at a fixed coordinate position/direction: as the binary orbits,
//! that launch point drifts off the moving photon ring and never produces a clean pulse.

void DynBBHFlashBeamSource(Mesh *pm, const Real bdt) {
  if (!flash.enabled) return;
  const Real t = pm->time;
  if (t < flash.t0 || t > flash.toff) return;

  // user_srcs_func is invoked once by the radiation source-term task and once by the MHD
  // source-term task within the same substage.  (time,bdt) uniquely identifies a substage
  // (RK stages carry distinct beta*dt), so skip the duplicate call to avoid double
  // injection.
  static Real last_t = -1.0e300, last_bdt = -1.0e300;
  if (t == last_t && bdt == last_bdt) return;
  last_t = t; last_bdt = bdt;

  const Real amp_t = flash.amp*std::exp(-(t - flash.t0)/flash.tau);
  if (!(amp_t > 0.0)) return;

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pdynrad == nullptr) return;
  const int nangles = pmbp->pdynrad->prgeo->nangles;

  // Launch geometry and angular weights against the current metric/tetrad (host side).
  Real pos[3], dir[3];
  FlashLaunchGeometry(t, pos, dir);
  Real g4[NDIM][NDIM];
  FlashMetricAt(t, pos[0], pos[1], pos[2], g4);
  Real ell[3];
  if (!FlashDirectionToTetrad(g4, dir, ell)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "flash beam direction is not null-realizable"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  static std::vector<Real> h_weights;
  if (!FlashProjectedWeights(pmbp->pdynrad->nh_c, nangles, ell, h_weights)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "flash beam angular projection failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // heap-allocated and intentionally leaked (like the single-hole test's
  // kerr_orbit_beam.angular_weights) so the View is never destroyed after
  // Kokkos::finalize
  static DvceArray1D<Real> *d_weights_ptr = nullptr;
  if (d_weights_ptr == nullptr) {
    d_weights_ptr = new DvceArray1D<Real>();
  }
  if (static_cast<int>(d_weights_ptr->extent(0)) != nangles) {
    Kokkos::realloc(*d_weights_ptr, nangles);
  }
  auto h_mirror = Kokkos::create_mirror_view(*d_weights_ptr);
  for (int n=0; n<nangles; ++n) {
    h_mirror(n) = h_weights[n];
  }
  Kokkos::deep_copy(*d_weights_ptr, h_mirror);

  const Real sxp = pos[0], syp = pos[1], szp = pos[2];
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  int nang1 = nangles - 1;

  auto i0 = pmbp->pdynrad->i0;
  auto solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  auto sqrt_detg_c = pmbp->pdynrad->sqrt_detg_c;
  auto &size = pmbp->pmb->mb_size;
  auto &excise = pmbp->pcoord->coord_data.bh_excise;
  auto &rad_mask = pmbp->pcoord->excision_floor;
  auto weights = *d_weights_ptr;

  const Real width2 = SQR(flash.width);
  const Real inject = amp_t*bdt;

  par_for("dynbbh_flash_beam", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    if (weights(n) <= 0.0) return;
    if (excise && rad_mask(m,k,j,i)) return;
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real dist2 = SQR(x1v - sxp) + SQR(x2v - syp) + SQR(x3v - szp);
    Real spatial = exp(-0.5*dist2/width2);
    if (spatial < 1.0e-8) return;
    i0(m,n,k,j,i) += sqrt_detg_c(m,k,j,i)*inject*spatial
                     *weights(n)/solid_angles.d_view(n);
  });
}

void AddDynBBHUserSources(Mesh *pm, const Real bdt) {
  if (bbh.cooling_source == CoolingSource::ism) {
    AddValenciaGRCooling(pm, bdt);
  } else if (bbh.cooling_source == CoolingSource::thin_disk) {
    AddThinDiskCooling(pm, bdt);
  }
}

void AddThinDiskCooling(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (bbh.cooling_source != CoolingSource::thin_disk ||
      pmbp->pmhd == nullptr || pmbp->padm == nullptr) {
    return;
  }

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &adm = pmbp->padm->adm;
  auto &w0 = pmbp->pmhd->w0;
  auto &u0 = pmbp->pmhd->u0;
  auto &size = pmbp->pmb->mb_size;
  auto eos = pmbp->pmhd->peos->eos_data;
  Real gamma_adi = eos.gamma;
  Real gm1 = gamma_adi - 1.0;
  Real rho_floor = eos.dfloor;
  Real p_floor = eos.pfloor;
  Real h_over_r = bbh.thin_cooling_h_over_r;
  Real tcool_orbits = bbh.thin_cooling_timescale_orbits;
  Real cfl_limit = bbh.thin_cooling_cfl;
  Real r_inner = bbh.thin_cooling_r_inner;
  Real r_outer = bbh.thin_cooling_r_outer;
  Real two_pi = 2.0*M_PI;
  auto pgen = bbh;

  constexpr Real tiny = 1.0e-30;

  par_for("dynbbh_thin_disk_cooling", DevExeSpace(),
          0, nmb-1, ks, ke, js, je, is, ie,
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

    Real r, theta, phi;
    GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);
    if (!isfinite(r)) return;
    if (!(r >= r_inner && r <= r_outer)) return;

    Real rho = w0(m,IDN,k,j,i);
    if (!(rho > rho_floor) || !isfinite(rho)) return;

    Real pres = w0(m,IPR,k,j,i);
    Real e_int = pres / gm1;
    Real e_floor = p_floor / gm1;
    if (!(e_int > e_floor) || !isfinite(e_int)) return;

    Real vk2 = 1.0 / fmax(r, 1.0);
    Real p_target = rho*SQR(h_over_r)*vk2 / gamma_adi;
    Real e_target = fmax(p_target/gm1, e_floor);
    if (e_int <= e_target) return;

    Real tcool = tcool_orbits*two_pi*pow(fmax(r, 1.0), 1.5);
    Real de = (e_int - e_target) * (1.0 - exp(-bdt/tcool));
    if (cfl_limit > 0.0) {
      de = fmin(de, cfl_limit*(e_int - e_floor));
    }
    de = fmin(de, e_int - e_floor);
    if (de <= 0.0) return;

    Real gxx = adm.g_dd(m,0,0,k,j,i);
    Real gxy = adm.g_dd(m,0,1,k,j,i);
    Real gxz = adm.g_dd(m,0,2,k,j,i);
    Real gyy = adm.g_dd(m,1,1,k,j,i);
    Real gyz = adm.g_dd(m,1,2,k,j,i);
    Real gzz = adm.g_dd(m,2,2,k,j,i);

    Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
    if (!(detg > tiny) || !isfinite(detg)) return;
    detg = fmax(detg, tiny);
    Real sqrt_gamma = sqrt(detg);
    Real alpha = adm.alpha(m,k,j,i);
    if (!(alpha > tiny) || !isfinite(alpha)) return;
    alpha = fmax(alpha, tiny);

    Real u1p = w0(m,IVX,k,j,i);
    Real u2p = w0(m,IVY,k,j,i);
    Real u3p = w0(m,IVZ,k,j,i);
    if (!isfinite(u1p) || !isfinite(u2p) || !isfinite(u3p)) return;
    Real u_sq = gxx*u1p*u1p + 2.0*gxy*u1p*u2p + 2.0*gxz*u1p*u3p
              + gyy*u2p*u2p + 2.0*gyz*u2p*u3p + gzz*u3p*u3p;
    if (!isfinite(u_sq)) return;
    Real W = sqrt(fmax(1.0 + u_sq, 1.0));

    Real u1_cov = gxx*u1p + gxy*u2p + gxz*u3p;
    Real u2_cov = gxy*u1p + gyy*u2p + gyz*u3p;
    Real u3_cov = gxz*u1p + gyz*u2p + gzz*u3p;

    Real q_dt = de * (W / alpha);
    if (!isfinite(q_dt)) return;
    u0(m,IEN,k,j,i) -= sqrt_gamma * alpha * W * q_dt;
    u0(m,IM1,k,j,i) -= sqrt_gamma * alpha * u1_cov * q_dt;
    u0(m,IM2,k,j,i) -= sqrt_gamma * alpha * u2_cov * q_dt;
    u0(m,IM3,k,j,i) -= sqrt_gamma * alpha * u3_cov * q_dt;
  });
}

void AddValenciaGRCooling(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->padm == nullptr) {
    return;
  }

  // ---- Units ----
  Real temp_unit     = pmbp->punit->temperature_cgs();
  Real density_unit  = pmbp->punit->density_cgs();
  Real time_unit     = pmbp->punit->time_cgs();
  Real pressure_unit = pmbp->punit->pressure_cgs();

  Real mu  = pmbp->punit->mu();
  Real amu = pmbp->punit->atomic_mass_unit_cgs;

  Real n_unit = density_unit / (mu * amu);
  Real cooling_unit = pressure_unit / time_unit / (n_unit * n_unit);

  // ---- Indices ----
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  // ---- Accessors ----
  auto &adm = pmbp->padm->adm;
  auto &w0  = pmbp->pmhd->w0;  // primitives
  auto &u0  = pmbp->pmhd->u0;  // conserved

  // ---- EOS ----
  auto &eos_data = pmbp->pmhd->peos->eos_data;
  Real gamma_adi = eos_data.gamma;
  Real gm1       = gamma_adi - 1.0;
  Real rho_floor = eos_data.dfloor;
  Real p_floor   = eos_data.pfloor;

  // ---- Stability Control ----
  // Use the simulation's global CFL number for consistency
  // instead of a hardcoded "by-hand" value.
  Real cfl_limit = pm->cfl_no;

  constexpr int  max_sub  = 64;
  constexpr Real tiny     = 1.0e-30;

  par_for("Valencia_IsotropicCooling", DevExeSpace(),
          0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {

    // --- metric gamma_ij and sqrt(gamma) ---
    Real gxx = adm.g_dd(m,0,0,k,j,i);
    Real gxy = adm.g_dd(m,0,1,k,j,i);
    Real gxz = adm.g_dd(m,0,2,k,j,i);
    Real gyy = adm.g_dd(m,1,1,k,j,i);
    Real gyz = adm.g_dd(m,1,2,k,j,i);
    Real gzz = adm.g_dd(m,2,2,k,j,i);

    Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
    if (!(detg > tiny) || !isfinite(detg)) return;
    detg = fmax(detg, tiny);
    Real sqrt_gamma = sqrt(detg);

    Real alpha = adm.alpha(m,k,j,i);
    if (!(alpha > tiny) || !isfinite(alpha)) return;

    // --- primitive state ---
    Real rho  = w0(m,IDN,k,j,i);
    if (!(rho > rho_floor) || !isfinite(rho)) return;

    Real pres = w0(m,IPR,k,j,i);
    if (!isfinite(pres)) return;

    // primitive stores u^{i'}
    Real u1p = w0(m,IVX,k,j,i);
    Real u2p = w0(m,IVY,k,j,i);
    Real u3p = w0(m,IVZ,k,j,i);
    if (!isfinite(u1p) || !isfinite(u2p) || !isfinite(u3p)) return;

    // W = sqrt(1 + gamma_ij u^{i'} u^{j'})
    Real u_sq = gxx*u1p*u1p + 2.0*gxy*u1p*u2p + 2.0*gxz*u1p*u3p
              + gyy*u2p*u2p + 2.0*gyz*u2p*u3p + gzz*u3p*u3p;
    if (!isfinite(u_sq)) return;
    Real W = sqrt(fmax(1.0 + u_sq, 1.0));

    // covariant spatial components u_i
    Real u1_cov = gxx*u1p + gxy*u2p + gxz*u3p;
    Real u2_cov = gxy*u1p + gyy*u2p + gyz*u3p;
    Real u3_cov = gxz*u1p + gyz*u2p + gzz*u3p;

    // comoving internal energy density
    Real e_int = pres / gm1;
    Real e_floor = p_floor / gm1;
    if (!(e_int > e_floor) || !isfinite(e_int)) return;

    // Subcycling in coordinate time
    Real dt_rem = bdt;

    // Accumulated conserved decrements
    Real dTau_total = 0.0;
    Real dS1_total  = 0.0;
    Real dS2_total  = 0.0;
    Real dS3_total  = 0.0;

    for (int n = 0; n < max_sub && dt_rem > 0.0; ++n) {
      // Temperature proxy
      Real T_cgs = ( (e_int * gm1) / rho ) * temp_unit;
      if (!isfinite(T_cgs)) break;

      // Determine Cooling Rate
      Real Lambda_cgs = 0.0;
      if (T_cgs >= eos_data.tfloor * temp_unit) {
         Lambda_cgs = ISMCoolFn(T_cgs);
      }

      // q = n^2 Lambda
      Real q = (rho * rho) * (Lambda_cgs / cooling_unit); // code units

      if (!(q > 0.0) || !isfinite(q)) break;

      // Coordinate-time cooling rate for e_int: de_int/dt = -(alpha/W) q
      Real rate_e_dt = (alpha / W) * q;
      if (!isfinite(rate_e_dt)) break;

      // === DYNAMIC CLAMP (Based on CFL) ===
      // Max allowed rate is one that removes 'cfl_limit' fraction of e_int
      // over the full timestep 'bdt'.
      // This ensures operator splitting doesn't shock the hydro solver.
      Real rate_max = (cfl_limit * e_int) / (bdt + tiny);

      rate_e_dt = fmin(rate_e_dt, rate_max);
      // ====================================

      // Choose substep using the same CFL limit
      Real dt_sub = cfl_limit * e_int / (rate_e_dt + tiny);
      dt_sub = fmin(dt_sub, dt_rem);
      dt_sub = fmax(dt_sub, tiny * bdt);

      // Proposed decrement
      Real de = rate_e_dt * dt_sub;
      if (!isfinite(de)) break;

      // Enforce floor on e_int
      Real de_applied = de;
      if (e_int - de_applied < e_floor) {
        de_applied = e_int - e_floor;
        e_int = e_floor;
      } else {
        e_int -= de_applied;
      }

      if (de_applied <= 0.0) break;

      // Convert applied decrement back to q*dt (source term magnitude)
      Real q_dt = de_applied * (W / alpha);
      if (!isfinite(q_dt)) break;

      // Valencia isotropic cooling sources:
      Real dTau = sqrt_gamma * alpha * W * q_dt;
      Real dS1  = sqrt_gamma * alpha * u1_cov * q_dt;
      Real dS2  = sqrt_gamma * alpha * u2_cov * q_dt;
      Real dS3  = sqrt_gamma * alpha * u3_cov * q_dt;

      dTau_total += dTau;
      dS1_total  += dS1;
      dS2_total  += dS2;
      dS3_total  += dS3;

      dt_rem -= dt_sub;
    }

    // Apply to conserved variables
    u0(m,IEN,k,j,i) -= dTau_total;
    u0(m,IM1,k,j,i) -= dS1_total;
    u0(m,IM2,k,j,i) -= dS2_total;
    u0(m,IM3,k,j,i) -= dS3_total;
  });
}


}//namespace
