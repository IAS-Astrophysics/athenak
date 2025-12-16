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
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/flux_generalized.hpp"

#include <Kokkos_Random.hpp>


#define h 5e-5
#define D2(comp, h) ((met_p1.g).comp - (met_m1.g).comp) / (2*h)

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

struct bbh_pgen {
  Real sep;
  Real om;
  Real q;
  Real a1, a2;
  Real th_a1, th_a2;
  Real ph_a1, ph_a2;
  Real d;
  Real gamma_adi;
  Real a1_buffer, a2_buffer;
  Real adjust_mass1, adjust_mass2;
  Real cutoff_floor;
  Real alpha_thr;
  Real radius_thr;

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
  std::vector<Real> radius;
  std::vector<int> reflevel;
};

struct bbh_pgen bbh;
struct bbh_refine bbh_ref;

/* Declare functions */
void find_traj_t(Real tt, Real traj_array[NTRAJ]);

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ], const bbh_pgen bbh_);
KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met, struct three_metric &gam);
KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
	       	        struct four_metric &met, const Real bbh_traj_loc[NTRAJ], const bbh_pgen bbh_);
KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                   Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen bbh_);
void SetADMVariablesToBBH(MeshBlockPack *pmbp);
void RefineAlphaMin(MeshBlockPack* pmbp);
void RefineTracker(MeshBlockPack* pmbp);
void RefineRadii(MeshBlockPack* pmbp);
void Refine(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn void TorusHistory(HistoryData *pdata, Mesh *pm)
//! \brief New user history function that calls the general-purpose flux integrator.
//----------------------------------------------------------------------------------------
void TorusHistory(HistoryData *pdata, Mesh *pm) {
    ProblemGenerator *pgen = pm->pgen.get();
    if (pgen->surface_grids.empty()) {
        pdata->nhist = 0;
        return;
    }
    MeshBlockPack *pmbp = pm->pmb_pack;

    // Convert the vector of unique_ptrs to a vector of raw pointers for the function.
    std::vector<SphericalSurfaceGrid*> surf_raw_ptrs;
    surf_raw_ptrs.reserve(pgen->surface_grids.size());
    for(const auto& s : pgen->surface_grids) {
        surf_raw_ptrs.push_back(s.get());
    }

    // Call the generalized flux calculator from "utils/flux_generalized.cpp"
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

  bbh.spin = 0.0; //delete eventually?

  bbh.sep = pin->GetOrAddReal("problem", "sep", 25.0);
  bbh.om = std::pow(bbh.sep, -1.5);
  bbh.q = pin->GetOrAddReal("problem", "q", 1.0);
  bbh.a1 = pin->GetOrAddReal("problem", "a1", 0.0);
  bbh.a2 = pin->GetOrAddReal("problem", "a2", 0.0);
  bbh.th_a1 = pin->GetOrAddReal("problem", "th_a1", 0.0);
  bbh.th_a2 = pin->GetOrAddReal("problem", "th_a2", 0.0);
  bbh.ph_a1 = pin->GetOrAddReal("problem", "ph_a1", 0.0);
  bbh.ph_a2 = pin->GetOrAddReal("problem", "ph_a2", 0.0);
  bbh.d = pin->GetOrAddReal("problem", "duniform", 1.0);
  bbh.gamma_adi = pin->GetOrAddReal("problem", "gamma_adi", 1.6666666);
  bbh.adjust_mass1 = pin->GetOrAddReal("problem", "adjust_mass1", 1.0);
  bbh.adjust_mass2 = pin->GetOrAddReal("problem", "adjust_mass2", 1.0);
  bbh.a1_buffer = pin->GetOrAddReal("problem", "a1_buffer", 0.01);
  bbh.a2_buffer = pin->GetOrAddReal("problem", "a2_buffer", 0.01);
  bbh.cutoff_floor = pin->GetOrAddReal("problem", "cutoff_floor", 1e-4);
  bbh.alpha_thr = pin->GetOrAddReal("problem", "alpha_thr", 0.2);
  bbh.radius_thr = pin->GetOrAddReal("problem", "radius_thr", 2.);

  for (int nr = 0; nr < 16; ++nr) {
    std::string name = "radius_" + std::to_string(nr) + "_rad";
    if (pin->DoesParameterExist("problem", name)) {
      bbh_ref.radius.push_back(pin->GetReal("problem", name));
      bbh_ref.reflevel.push_back(pin->GetOrAddInteger(
          "problem", "radius_" + std::to_string(nr) + "_reflevel", -1));
    } else {
      break;
    }
  }

  std::string amr_cond = pin->GetOrAddString("problem", "amr_condition", "none");
  if (amr_cond == "alpha_min") {
    std::cout << "Using Lapse-Based Refinement" << std::endl;
    bbh_ref.AlphaMin = true;
  } else if (amr_cond == "tracker") {
    std::cout << "Using Tracker-Based Refinement" << std::endl;
    bbh_ref.Tracker = true;
  }

  user_ref_func = Refine;
  user_hist_func = TorusHistory;

  pmbp->padm->SetADMVariables = &SetADMVariablesToBBH;

  // Flux diagnostics setup
  // Resolution of surface grids
  const int ntheta = pin->GetOrAddInteger("problem", "flux_ntheta", 64);
  const int nphi = pin->GetOrAddInteger("problem", "flux_nphi", 128);
  // Setup Radius of surface grids
  const Real r_surf_inner = pin->GetOrAddReal("problem", "flux_rsurf_inner", 10.0);
  const Real dr_surf = pin->GetOrAddReal("problem", "flux_dr_surf", 5.0);
  const Real r_surf_outer = pin->GetOrAddReal("problem", "flux_rsurf_outer", 20.0);
  // Create surfaces at three different radii and store them in the class member
  // This avoids the crash-on-exit from using a static variable.
  for (Real r_surf = r_surf_inner; r_surf <= r_surf_outer; r_surf += dr_surf) {
    auto r_func = [=](Real th, Real ph){ return r_surf; };
    this->surface_grids.push_back(std::make_unique<SphericalSurfaceGrid>(
        pmbp, ntheta, nphi, r_func, "R" + std::to_string(static_cast<int>(r_surf))));
  }

  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
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
    bbh.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    bbh.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bbh.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr) {
    bbh.arad = pmbp->prad->arad;
  }


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
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  Real bbh_traj_t0[NTRAJ];
  find_traj_t(0.0, bbh_traj_t0);
  auto& bbh_traj_ = bbh_traj_t0;

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

    // Extract metric and inverse -- presumably should get actual metric?????
    Real glower[4][4], gupper[4][4];
    GetSuperposedAndInverse(0.0, x1v, x2v, x3v, glower, gupper, bbh_traj_, trs);

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
      GetSuperposedAndInverse(0.0, x1v, x2v, x3v, glower, gupper, bbh_traj_, trs);

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

  // Initialize ADM variables -------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
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
      GetSuperposedAndInverse(0.0, x1v, x2v, x3v, glower, gupper, bbh_traj_, trs);

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

  Real bbh_traj_p1[NTRAJ];
  Real bbh_traj_0[NTRAJ];
  Real bbh_traj_m1[NTRAJ];
  auto bbh_ = bbh;

  /* Load trajectories */

  /* Whether we load traj from a table or we compute analytical trajectories */
  find_traj_t(tt+h, bbh_traj_p1);
  find_traj_t(tt, bbh_traj_0);
  find_traj_t(tt-h, bbh_traj_m1);

  // update punc location for excision
  coord.punc_0[0] = bbh_traj_0[X1];
  coord.punc_0[1] = bbh_traj_0[Y1];
  coord.punc_0[2] = bbh_traj_0[Z1];
  coord.punc_1[0] = bbh_traj_0[X2];
  coord.punc_1[1] = bbh_traj_0[Y2];
  coord.punc_1[2] = bbh_traj_0[Z2];



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
    numerical_4metric(tt, x1v, x2v, x3v, met4, bbh_traj_m1, bbh_traj_0, bbh_traj_p1, bbh_);
  
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
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ], const bbh_pgen bbh_)
{
  struct four_metric met_m1;
  struct four_metric met_p1;

  // Time
  get_metric(t-1*h, x, y, z, met_m1, nz_m1, bbh_);
  get_metric(t+1*h, x, y, z, met_p1, nz_p1, bbh_);
  get_metric(t, x, y, z, outmet, nz_0, bbh_);

  outmet.g_t.tt = D2(tt, h);
  outmet.g_t.tx = D2(tx, h);
  outmet.g_t.ty = D2(ty, h);
  outmet.g_t.tz = D2(tz, h);
  outmet.g_t.xx = D2(xx, h);
  outmet.g_t.xy = D2(xy, h);
  outmet.g_t.xz = D2(xz, h);
  outmet.g_t.yy = D2(yy, h);
  outmet.g_t.yz = D2(yz, h);
  outmet.g_t.zz = D2(zz, h);

  // X
  get_metric(t, x-1*h, y, z, met_m1, nz_0, bbh_);
  get_metric(t, x+1*h, y, z, met_p1, nz_0, bbh_);

  outmet.g_x.tt = D2(tt, h);
  outmet.g_x.tx = D2(tx, h);
  outmet.g_x.ty = D2(ty, h);
  outmet.g_x.tz = D2(tz, h);
  outmet.g_x.xx = D2(xx, h);
  outmet.g_x.xy = D2(xy, h);
  outmet.g_x.xz = D2(xz, h);
  outmet.g_x.yy = D2(yy, h);
  outmet.g_x.yz = D2(yz, h);
  outmet.g_x.zz = D2(zz, h);

  // Y
  get_metric(t, x, y-1*h, z, met_m1, nz_0, bbh_);
  get_metric(t, x, y+1*h, z, met_p1, nz_0, bbh_);

  outmet.g_y.tt = D2(tt, h);
  outmet.g_y.tx = D2(tx, h);
  outmet.g_y.ty = D2(ty, h);
  outmet.g_y.tz = D2(tz, h);
  outmet.g_y.xx = D2(xx, h);
  outmet.g_y.xy = D2(xy, h);
  outmet.g_y.xz = D2(xz, h);
  outmet.g_y.yy = D2(yy, h);
  outmet.g_y.yz = D2(yz, h);
  outmet.g_y.zz = D2(zz, h);

  // Z
  get_metric(t, x, y, z-1*h, met_m1, nz_0, bbh_);
  get_metric(t, x, y, z+1*h, met_p1, nz_0, bbh_);

  outmet.g_z.tt = D2(tt, h);
  outmet.g_z.tx = D2(tx, h);
  outmet.g_z.ty = D2(ty, h);
  outmet.g_z.tz = D2(tz, h);
  outmet.g_z.xx = D2(xx, h);
  outmet.g_z.xy = D2(xy, h);
  outmet.g_z.xz = D2(xz, h);
  outmet.g_z.yy = D2(yy, h);
  outmet.g_z.yz = D2(yz, h);
  outmet.g_z.zz = D2(zz, h);

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

// Function to calculate the position and velocity of m1 and m2 at time t
void find_traj_t(Real t, Real bbh_t[NTRAJ]) {

  Real const r_BH1_0 = bbh.q/(1.0+bbh.q)*bbh.sep;
  Real const r_BH2_0 = -bbh.sep/(1.0+bbh.q);
  bbh_t[X1] = r_BH1_0*std::cos(bbh.om*t);
  bbh_t[Y1] = r_BH1_0*std::sin(bbh.om*t);
  bbh_t[Z1] = 0.0;
  bbh_t[X2] = r_BH2_0*std::cos(bbh.om*t);
  bbh_t[Y2] = r_BH2_0*std::sin(bbh.om*t);
  bbh_t[Z2] = 0.0;
  bbh_t[VX1] = -r_BH1_0*bbh.om*std::sin(bbh.om*t);
  bbh_t[VY1] = r_BH1_0*bbh.om*std::cos(bbh.om*t);
  bbh_t[VZ1] = 0.0;
  bbh_t[VX2] = -r_BH2_0*bbh.om*std::sin(bbh.om*t);
  bbh_t[VY2] = r_BH2_0*bbh.om*std::cos(bbh.om*t);
  bbh_t[VZ2] = 0.0;
  bbh_t[AX1] = bbh.a1*std::sin(bbh.th_a1)*std::cos(bbh.ph_a1);
  bbh_t[AY1] = bbh.a1*std::sin(bbh.th_a1)*std::sin(bbh.ph_a1);
  bbh_t[AZ1] = bbh.a1*std::cos(bbh.th_a1);
  bbh_t[AX2] = bbh.a1*std::sin(bbh.th_a2)*std::cos(bbh.ph_a2);
  bbh_t[AY2] = bbh.a1*std::sin(bbh.th_a2)*std::sin(bbh.ph_a2);
  bbh_t[AZ2] = bbh.a1*std::cos(bbh.th_a2);
  bbh_t[M1T] = 1.0/(bbh.q+1.0);
  bbh_t[M2T] = 1.0 - bbh_t[M1T];
}

KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                    Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen bbh_)
{
  /* Superposition components*/
  Real KS1[NDIM][NDIM];
  Real KS2[NDIM][NDIM];
  Real J1[NDIM][NDIM];
  Real J2[NDIM][NDIM];

  /* Load trajectories */
  Real xi1x = traj_array[X1];
  Real xi1y = traj_array[Y1]; 
  Real xi1z = traj_array[Z1];
  Real xi2x = traj_array[X2];
  Real xi2y = traj_array[Y2];
  Real xi2z = traj_array[Z2];
  Real v1x  = traj_array[VX1] + 1e-40;
  Real v1y  = traj_array[VY1] + 1e-40;
  Real v1z  = traj_array[VZ1] + 1e-40;
  Real v2x =  traj_array[VX2] + 1e-40;
  Real v2y =  traj_array[VY2] + 1e-40;
  Real v2z =  traj_array[VZ2] + 1e-40;
  
  Real v2  =  sqrt( v2x * v2x + v2y * v2y + v2z * v2z );
  Real v1  =  sqrt( v1x * v1x + v1y * v1y + v1z * v1z ); 
  
  Real a1x  = traj_array[AX1];
  Real a1y  = traj_array[AY1];
  Real a1z  = traj_array[AZ1];
  
  Real a2x =  traj_array[AX2];
  Real a2y =  traj_array[AY2];
  Real a2z =  traj_array[AZ2];
  
  Real m1_t = traj_array[M1T];
  Real m2_t = traj_array[M2T];

  Real a1_t = sqrt( a1x*a1x + a1y*a1y + a1z*a1z + 1e-40) ;
  Real a2_t = sqrt( a2x*a2x + a2y*a2y + a2z*a2z + 1e-40) ;
 
  /* Load coordinates */  

   Real oo1 = v1 * v1;
   Real oo2 = oo1 * -1;
   Real oo3 = 1 + oo2;
   Real oo4 = sqrt(oo3);
   Real oo5 = 1 / oo4;
   Real oo6 = x * -1;
   Real oo7 = oo6 + xi1x;
   Real oo8 = v1x * oo7;
   Real oo9 = y * -1;
   Real oo10 = z * -1;
   Real oo11 = v2 * v2;
   Real oo12 = oo11 * -1;
   Real oo13 = 1 + oo12;
   Real oo14 = sqrt(oo13);
   Real oo15 = 1 / oo14;
   Real oo16 = oo6 + xi2x;
   Real oo17 = v2x * oo16;
   Real oo18 = xi1x * -1;
   Real oo19 = 1 / oo1;
   Real oo20 = -1 + oo4;
   Real oo21 = xi1y * -1;
   Real oo22 = xi1z * -1;
   Real oo23 = xi2x * -1;
   Real oo24 = 1 / oo11;
   Real oo25 = -1 + oo14;
   Real oo26 = xi2y * -1;
   Real oo27 = xi2z * -1;
   Real oo28 = xi1y * v1y;
   Real oo29 = xi1z * v1z;
   Real oo30 = v1y * (y * -1);
   Real oo31 = v1z * (z * -1);
   Real oo32 = oo28 + (oo29 + (oo30 + (oo31 + oo8)));
   Real oo33 = xi2y * v2y;
   Real oo34 = xi2z * v2z;
   Real oo35 = v2y * (y * -1);
   Real oo36 = v2z * (z * -1);
   Real oo37 = oo17 + (oo33 + (oo34 + (oo35 + oo36)));
   //Real x0BH1 = (oo8 + ((oo9 + xi1y) * v1y + (oo10 + xi1z) * v1z)) * oo5;
   //Real x0BH2 = (oo17 + ((oo9 + xi2y) * v2y + (oo10 + xi2z) * v2z)) * oo15;
   Real x1BH1 = (oo18 + x) - oo20 * (oo5 * (v1x * (((oo18 + x) * v1x + ((oo21 + y) * v1y + (oo22 + z) * v1z)) * oo19)));
   Real x1BH2 = (oo23 + x) - oo24 * (oo25 * (v2x * (((oo23 + x) * v2x + ((oo26 + y) * v2y + (oo27 + z) * v2z)) * oo15)));
   Real x2BH1 = oo21 + (oo20 * (oo32 * (oo5 * (v1y * oo19))) + y);
   Real x2BH2 = oo26 + (oo24 * (oo25 * (oo37 * (v2y * oo15))) + y);
   Real x3BH1 = oo22 + (oo20 * (oo32 * (oo5 * (v1z * oo19))) + z);
   Real x3BH2 = oo27 + (oo24 * (oo25 * (oo37 * (v2z * oo15))) + z);

 
  /* Adjust mass */
  /* This is useful for reducing the effective mass of each BH */
  /* Adjust by hand to get the correct irreducible mass of the BH */
  Real a1 = a1_t * bbh_.adjust_mass1;
  Real m1 = m1_t * bbh_.adjust_mass1;
  Real a2 = a2_t * bbh_.adjust_mass2;
  Real m2 = m2_t * bbh_.adjust_mass2;
 
  //============================================// 
  // Regularize horizon and apply excision mask //
  //============================================//

  /* Define radius with respect to BH frame */
  Real rBH1 = sqrt( x1BH1*x1BH1 + x2BH1*x2BH1 + x3BH1*x3BH1) ;
  Real rBH2 = sqrt( x1BH2*x1BH2 + x2BH2*x2BH2 + x3BH2*x3BH2) ;

   /* Define radius cutoff */
  Real rBH1_Cutoff = fabs(a1) * ( 1.0 + bbh_.a1_buffer) + bbh_.cutoff_floor ;
  Real rBH2_Cutoff = fabs(a2) * ( 1.0 + bbh_.a2_buffer) + bbh_.cutoff_floor ;

  /* Apply excision */
  if ((rBH1) < rBH1_Cutoff) { if(x3BH1>0) {x3BH1 = rBH1_Cutoff;} else {x3BH1 = -1.0*rBH1_Cutoff;}}
  if ((rBH2) < rBH2_Cutoff) { if(x3BH2>0) {x3BH2 = rBH2_Cutoff;} else {x3BH2 = -1.0*rBH2_Cutoff;}}
 
  //=================//
  //     Metric      //
  //=================//
  Real o1 = 1.4142135623730951;
  Real o2 = 1 / o1;
  Real o3 = a1x * a1x;
  Real o4 = o3 * -1;
  Real o5 = a1z * a1z;
  Real o6 = o5 * -1;
  Real o7 = a2x * a2x;
  Real o8 = o7 * -1;
  Real o9 = x1BH1 * x1BH1;
  Real o10 = x2BH1 * x2BH1;
  Real o11 = x3BH1 * x3BH1;
  Real o12 = x1BH1 * a1x;
  Real o13 = x2BH1 * a2x;
  Real o14 = x3BH1 * a1z;
  Real o15 = o12 + (o13 + o14);
  Real o16 = o15 * o15;
  Real o17 = o16 * 4;
  Real o18 = o10 + (o11 + (o4 + (o6 + (o8 + o9))));
  Real o19 = o18 * o18;
  Real o20 = o17 + o19;
  Real o21 = sqrt(o20);
  Real o22 = o10 + (o11 + (o21 + (o4 + (o6 + (o8 + o9)))));
  Real o23 = pow(o22, 1.5);
  Real o24 = o22 * o22;
  Real o25 = o24 * 0.25;
  Real o26 = o16 + o25;
  Real o27 = 1 / o26;
  Real o28 = x2BH1 * a1z;
  Real o29 = a2x * (x3BH1 * -1);
  Real o30 = sqrt(o22);
  Real o31 = 1 / o30;
  Real o32 = o1 * (o15 * (o31 * a1x));
  Real o33 = o30 * (x1BH1 * o2);
  Real o34 = o28 + (o29 + (o32 + o33));
  Real o35 = o22 * 0.5;
  Real o36 = o3 + (o35 + (o5 + o7));
  Real o37 = 1 / o36;
  Real o38 = o2 * (o23 * (o27 * (o34 * (o37 * m1))));
  Real o39 = a1z * (x1BH1 * -1);
  Real o40 = x3BH1 * a1x;
  Real o41 = o1 * (o15 * (o31 * a2x));
  Real o42 = o30 * (x2BH1 * o2);
  Real o43 = o39 + (o40 + (o41 + o42));
  Real o44 = o2 * (o23 * (o27 * (o37 * (o43 * m1))));
  Real o45 = x1BH1 * a2x;
  Real o46 = a1x * (x2BH1 * -1);
  Real o47 = o1 * (o15 * (o31 * a1z));
  Real o48 = o30 * (x3BH1 * o2);
  Real o49 = o45 + (o46 + (o47 + o48));
  Real o50 = o2 * (o23 * (o27 * (o37 * (o49 * m1))));
  Real o51 = o36 * o36;
  Real o52 = 1 / o51;
  Real o53 = o2 * (o23 * (o27 * (o34 * (o43 * (o52 * m1)))));
  Real o54 = o2 * (o23 * (o27 * (o34 * (o49 * (o52 * m1)))));
  Real o55 = o2 * (o23 * (o27 * (o43 * (o49 * (o52 * m1)))));
  Real o56 = a2y * a2y;
  Real o57 = o56 * -1;
  Real o58 = a2z * a2z;
  Real o59 = o58 * -1;
  Real o60 = x1BH2 * x1BH2;
  Real o61 = x2BH2 * x2BH2;
  Real o62 = x3BH2 * x3BH2;
  Real o63 = x1BH2 * a2x;
  Real o64 = x2BH2 * a2y;
  Real o65 = x3BH2 * a2z;
  Real o66 = o63 + (o64 + o65);
  Real o67 = o66 * o66;
  Real o68 = o67 * 4;
  Real o69 = o57 + (o59 + (o60 + (o61 + (o62 + o8))));
  Real o70 = o69 * o69;
  Real o71 = o68 + o70;
  Real o72 = sqrt(o71);
  Real o73 = o57 + (o59 + (o60 + (o61 + (o62 + (o72 + o8)))));
  Real o74 = pow(o73, 1.5);
  Real o75 = o73 * o73;
  Real o76 = o75 * 0.25;
  Real o77 = o67 + o76;
  Real o78 = 1 / o77;
  Real o79 = x2BH2 * a2z;
  Real o80 = a2y * (x3BH2 * -1);
  Real o81 = sqrt(o73);
  Real o82 = 1 / o81;
  Real o83 = o1 * (o66 * (o82 * a2x));
  Real o84 = o81 * (x1BH2 * o2);
  Real o85 = o79 + (o80 + (o83 + o84));
  Real o86 = o73 * 0.5;
  Real o87 = o56 + (o58 + (o7 + o86));
  Real o88 = 1 / o87;
  Real o89 = o2 * (o74 * (o78 * (o85 * (o88 * m2))));
  Real o90 = a2z * (x1BH2 * -1);
  Real o91 = x3BH2 * a2x;
  Real o92 = o1 * (o66 * (o82 * a2y));
  Real o93 = o81 * (x2BH2 * o2);
  Real o94 = o90 + (o91 + (o92 + o93));
  Real o95 = o2 * (o74 * (o78 * (o88 * (o94 * m2))));
  Real o96 = x1BH2 * a2y;
  Real o97 = a2x * (x2BH2 * -1);
  Real o98 = o1 * (o66 * (o82 * a2z));
  Real o99 = o81 * (x3BH2 * o2);
  Real o100 = o96 + (o97 + (o98 + o99));
  Real o101 = o100 * (o2 * (o74 * (o78 * (o88 * m2))));
  Real o102 = o87 * o87;
  Real o103 = 1 / o102;
  Real o104 = o103 * (o2 * (o74 * (o78 * (o85 * (o94 * m2)))));
  Real o105 = o100 * (o103 * (o2 * (o74 * (o78 * (o85 * m2)))));
  Real o106 = o100 * (o103 * (o2 * (o74 * (o78 * (o94 * m2)))));
  Real o107 = v1 * v1;
  Real o108 = o107 * -1;
  Real o109 = 1 + o108;
  Real o110 = sqrt(o109);
  Real o111 = 1 / o110;
  Real o112 = o111 * (v1x * -1);
  Real o113 = o111 * (v1y * -1);
  Real o114 = o111 * (v1z * -1);
  Real o115 = 1 / o107;
  Real o116 = -1 + o111;
  Real o117 = o116 * (v1x * (v1y * o115));
  Real o118 = o116 * (v1x * (v1z * o115));
  Real o119 = o116 * (v1y * (v1z * o115));
  Real o120 = v2 * v2;
  Real o121 = o120 * -1;
  Real o122 = 1 + o121;
  Real o123 = sqrt(o122);
  Real o124 = 1 / o123;
  Real o125 = o124 * (v2x * -1);
  Real o126 = o124 * (v2y * -1);
  Real o127 = o124 * (v2z * -1);
  Real o128 = 1 / o120;
  Real o129 = -1 + o124;
  Real o130 = o129 * (v2x * (v2y * o128));
  Real o131 = o129 * (v2x * (v2z * o128));
  Real o132 = o129 * (v2y * (v2z * o128));
  KS1[0][0] = o2 * (o23 * (o27 * m1));
  KS1[0][1] = o38;
  KS1[0][2] = o44;
  KS1[0][3] = o50;
  KS1[1][0] = o38;
  KS1[1][1] = o2 * (o23 * (o27 * ((o34 * o34) * (o52 * m1))));
  KS1[1][2] = o53;
  KS1[1][3] = o54;
  KS1[2][0] = o44;
  KS1[2][1] = o53;
  KS1[2][2] = o2 * (o23 * (o27 * ((o43 * o43) * (o52 * m1))));
  KS1[2][3] = o55;
  KS1[3][0] = o50;
  KS1[3][1] = o54;
  KS1[3][2] = o55;
  KS1[3][3] = o2 * (o23 * (o27 * ((o49 * o49) * (o52 * m1))));
  KS2[0][0] = o2 * (o74 * (o78 * m2));
  KS2[0][1] = o89;
  KS2[0][2] = o95;
  KS2[0][3] = o101;
  KS2[1][0] = o89;
  KS2[1][1] = o103 * (o2 * (o74 * (o78 * ((o85 * o85) * m2))));
  KS2[1][2] = o104;
  KS2[1][3] = o105;
  KS2[2][0] = o95;
  KS2[2][1] = o104;
  KS2[2][2] = o103 * (o2 * (o74 * (o78 * ((o94 * o94) * m2))));
  KS2[2][3] = o106;
  KS2[3][0] = o101;
  KS2[3][1] = o105;
  KS2[3][2] = o106;
  KS2[3][3] = (o100 * o100) * (o103 * (o2 * (o74 * (o78 * m2))));
  J1[0][0] = o111;
  J1[0][1] = o112;
  J1[0][2] = o113;
  J1[0][3] = o114;
  J1[1][0] = o112;
  J1[1][1] = 1 + o116 * ((v1x * v1x) * o115);
  J1[1][2] = o117;
  J1[1][3] = o118;
  J1[2][0] = o113;
  J1[2][1] = o117;
  J1[2][2] = 1 + o116 * ((v1y * v1y) * o115);
  J1[2][3] = o119;
  J1[3][0] = o114;
  J1[3][1] = o118;
  J1[3][2] = o119;
  J1[3][3] = 1 + o116 * ((v1z * v1z) * o115);
  J2[0][0] = o124;
  J2[0][1] = o125;
  J2[0][2] = o126;
  J2[0][3] = o127;
  J2[1][0] = o125;
  J2[1][1] = 1 + o129 * ((v2x * v2x) * o128);
  J2[1][2] = o130;
  J2[1][3] = o131;
  J2[2][0] = o126;
  J2[2][1] = o130;
  J2[2][2] = 1 + o129 * ((v2y * v2y) * o128);
  J2[2][3] = o132;
  J2[3][0] = o127;
  J2[3][1] = o131;
  J2[3][2] = o132;
  J2[3][3] = 1 + o129 * ((v2z * v2z) * o128);
  /* Initialize the flat part */
  Real eta[4][4] = {{-1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  for (int i=0; i < 4; i++ ){ for (int j=0; j < 4; j++ ){ gcov[i][j] = eta[i][j]; }}

  /* Load symmetric gcov (from chatGPT3)*/
  for (int i = 0; i < 4; ++i) {
      for (int j = i; j < 4; ++j) {
  
          Real sum = 0.0;
          for (int m = 0; m < 4; ++m) {
              Real term1 = J2[m][i];
              Real term2 = J1[m][i];
  
              for (int n = 0; n < 4; ++n) {
                  Real term3 = J2[n][j];
                  Real term4 = J1[n][j];
  
                  sum += (term1 * term3 * KS2[m][n] + term2 * term4 * KS1[m][n]);
              }
          }
  
          gcov[i][j] += sum;
          gcov[j][i] = gcov[i][j];
      }
  }

  return;
}

KOKKOS_INLINE_FUNCTION
void get_metric(const Real t,
	       	const Real x,
	       	const Real y,
	       	const Real z,
	       	struct four_metric &met,
          const Real bbh_traj_loc[NTRAJ], const bbh_pgen bbh_)
{
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
Real A2(struct bbh_pgen pgen, Real x1, Real x2, Real x3) {
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
Real A3(struct bbh_pgen pgen, Real x1, Real x2, Real x3) {
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


}//namespace
