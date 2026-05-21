#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>  // mkdir

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "z4c/z4c.hpp"
#include "z4c/BHaHAHA_horizon_finder.hpp"
#include "z4c/compact_object_tracker.hpp"
extern "C" {
  #include "z4c/bhahaha/BHaHAHA.h"
}
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/arbitrary_grid_interpolator.hpp"

BHAHAHorizonFinder::BHAHAHorizonFinder(MeshBlockPack *pmbp, ParameterInput *pin)
  : pmbp_(pmbp), pin_(pin), last_find_time_(-std::numeric_limits<double>::max()),
    min_lapse_center_{0.0, 0.0, 0.0} {
  LoadParameters();
  // Initialize storage
  bah_horizon_active_.assign(max_num_horizons_, 1);
  use_fixed_radius_guess_on_full_sphere_.assign(max_num_horizons_, 1);

  // center location of black holes, will be set to the puncture location
  // if not common horizon.
  x_center_m1_.assign(max_num_horizons_, 0.0);
  y_center_m1_.assign(max_num_horizons_, 0.0);
  z_center_m1_.assign(max_num_horizons_, 0.0);
  t_m1_.assign(max_num_horizons_, -1.0);
  t_m2_ = t_m1_; t_m3_ = t_m1_;
  r_min_m1_.assign(max_num_horizons_, 0.0);
  r_min_m2_ = r_min_m1_; r_min_m3_ = r_min_m1_;
  r_max_m1_.assign(max_num_horizons_, max_search_radius_);
  r_max_m2_ = r_max_m1_; r_max_m3_ = r_max_m1_;
  prev_horizon_m1_.assign(max_num_horizons_, std::vector<double>(Ntheta_*Nphi_));
  prev_horizon_m2_ = prev_horizon_m1_;
  prev_horizon_m3_ = prev_horizon_m1_;

  params_data_.resize(max_num_horizons_);
  radii_.resize(Nr_interp_);
  cart_coords_.resize(static_cast<size_t>(Nr_interp_) * Ntheta_ * Nphi_);
  agrid_ = std::make_unique<ArbitraryGrid>(pmbp_, cart_coords_, 4);

  // initialize input_metric_data to null pointers
  int h = 0;
  for (auto &pd : params_data_) {
    bah_poisoning_set_inputs(&pd);
    pd.input_metric_data = nullptr;
    pd.use_fixed_radius_guess_on_full_sphere = 1;
    pd.cfl_factor = pin_->GetOrAddReal("z4c", "bah_cfl", 0.4);
    pd.M_scale = m_guess[h];
    pd.eta_damping_times_M = 1.6;
    pd.KO_strength = 0;
    pd.max_iterations = pin->GetOrAddInteger("z4c", "bah_max_itr", 10000);
    pd.Theta_Linf_times_M_tolerance = pin_->GetOrAddReal("z4c", "bah_Theta_Linf_tol", 1e-2);
    pd.Theta_L2_times_M_tolerance = pin_->GetOrAddReal("z4c", "bah_Theta_L2_tol", 2e-5);
    pd.enable_eta_varying_alg_for_precision_common_horizon = 0;
    pd.verbosity_level = pin->GetOrAddInteger("z4c", "bah_verbosity", 0);
    r_max_m1_[h] *= pd.M_scale;
    h++;
  }
}

BHAHAHorizonFinder::~BHAHAHorizonFinder() = default;

void BHAHAHorizonFinder::LoadParameters() {
  int legacy_find_every = pin_->GetOrAddInteger("z4c", "bah_find_every", 1);
  find_dcycle_ = pin_->GetOrAddInteger("z4c", "bah_find_dcycle", legacy_find_every);
  double legacy_find_dt = pin_->GetOrAddReal("z4c", "horizon_dt", 1.0);
  find_dt_ = pin_->GetOrAddReal("z4c", "bah_find_dt", legacy_find_dt);
  std::string find_mode = pin_->GetOrAddString("z4c", "bah_find_mode", "dcycle");
  if (find_mode == "dcycle" || find_mode == "cycle") {
    find_schedule_mode_ = FindScheduleMode::Dcycle;
  } else if (find_mode == "dt" || find_mode == "time") {
    find_schedule_mode_ = FindScheduleMode::Dt;
  } else {
    std::cerr << "Invalid <z4c>/bah_find_mode='" << find_mode
              << "'; expected 'dcycle' or 'dt'." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string center_mode = pin_->GetOrAddString("z4c", "bah_center_mode", "tracker");
  bool use_min_lapse = pin_->GetOrAddBoolean("z4c", "bah_use_min_lapse_centroid", false);
  if (use_min_lapse || center_mode == "min_lapse" || center_mode == "minimum_lapse" ||
      center_mode == "lapse") {
    center_mode_ = CenterMode::MinimumLapse;
  } else if (center_mode == "tracker") {
    center_mode_ = CenterMode::Tracker;
  } else {
    std::cerr << "Invalid <z4c>/bah_center_mode='" << center_mode
              << "'; expected 'tracker' or 'min_lapse'." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  max_num_horizons_ = pin_->GetOrAddInteger("z4c", "bah_num_horizons", 1);

  bah_num_resolutions_multigrid_ = pin_->GetOrAddInteger("z4c", "bah_num_resolutions_multigrid", 1);
  bah_Ntheta_array_multigrid_.resize(bah_num_resolutions_multigrid_);
  bah_Nphi_array_multigrid_.resize(bah_num_resolutions_multigrid_);
  Nr_interp_  = pin_->GetOrAddInteger("z4c", "bah_Nr_interp", 48);
  Ntheta_     = pin_->GetOrAddInteger("z4c", "bah_Ntheta", 32);
  Nphi_       = pin_->GetOrAddInteger("z4c", "bah_Nphi", 64);
  max_search_radius_ = pin_->GetOrAddReal("z4c", "bah_max_search_radius", 2);
  for (int i = 0; i < bah_num_resolutions_multigrid_; ++i) {
    bah_Ntheta_array_multigrid_[i] = pin_->GetOrAddInteger("z4c", "bah_Ntheta_array_multigrid_"+std::to_string(i), Ntheta_);
    bah_Nphi_array_multigrid_[i]   = pin_->GetOrAddInteger("z4c", "bah_Nphi_array_multigrid_"+std::to_string(i), Nphi_);
  }
  bah_BBH_mode_enable_ = pin_->GetOrAddBoolean("z4c", "bah_BBH_mode_enable", false);
  bah_BBH_mode_inspiral_BH_idxs_[0] = pin_->GetOrAddInteger("z4c", "bah_BBH_mode_inspiral_BH_idxs_0", 0);
  bah_BBH_mode_inspiral_BH_idxs_[1] = pin_->GetOrAddInteger("z4c", "bah_BBH_mode_inspiral_BH_idxs_1", 1);
  bah_BBH_mode_common_horizon_idx_ = pin_->GetOrAddInteger("z4c", "bah_BBH_mode_common_horizon_idx", 2);

  m_guess.assign(max_num_horizons_,0.0);
  for (int h = 0; h < max_num_horizons_; ++h) {
    m_guess[h] = pin_->GetOrAddInteger("z4c", "bah_mass"+std::to_string(h), 1);
  }
}

bool BHAHAHorizonFinder::ShouldFindHorizons() const {
  if (find_schedule_mode_ == FindScheduleMode::Dcycle) {
    return find_dcycle_ > 0 && (pmbp_->pmesh->ncycle % find_dcycle_ == 0);
  }

  if (find_dt_ <= 0.0) return false;
  double const time = pmbp_->pmesh->time;
  double const next_time = last_find_time_ + find_dt_;
  double const tol = 10.0 * std::numeric_limits<double>::epsilon() *
                     std::max(1.0, std::abs(next_time));
  return pmbp_->pmesh->ncycle == 0 || time + tol >= next_time;
}

void BHAHAHorizonFinder::checkMultigridResolutionInputs() {
  if (bah_num_resolutions_multigrid_ <= 0) {
    std::cerr << "Invalid multigrid level count" << std::endl;
    abort();
  }
  for (int i = 0; i < bah_num_resolutions_multigrid_; ++i) {
    if (bah_Ntheta_array_multigrid_[i] <= 0 || bah_Nphi_array_multigrid_[i] <= 0) {
      std::cerr << "Invalid multigrid resolution at level " << i << std::endl;
      abort();
    }
  }
}

void BHAHAHorizonFinder::FindHorizons() {
  if (!ShouldFindHorizons()) return;

  checkMultigridResolutionInputs();

  // Initialize variables at first time step
  if (pmbp_->pmesh->ncycle == 0) {
    for (int h = 0; h < max_num_horizons_; ++h) {
      t_m1_[h] = t_m2_[h] = t_m3_[h] = -1;
      r_min_m1_[h] = r_min_m2_[h] = r_min_m3_[h] = 0.0;
      r_max_m1_[h] = r_max_m2_[h] = r_max_m3_[h] = max_search_radius_*m_guess[h];
      use_fixed_radius_guess_on_full_sphere_[h] = 1;
    }

    // STEP: process BBH mode initial activation
    if (bah_BBH_mode_enable_) {
      if (max_num_horizons_ != 3) {
        std::cerr << "BBH mode requires 3 horizons" << std::endl;
        abort();
      }
      // common horizon not active at initial time
      bah_horizon_active_[bah_BBH_mode_common_horizon_idx_] = 0;
    }
  }

  if (center_mode_ == CenterMode::MinimumLapse) {
    UpdateMinimumLapseCenter();
  }

  // Loop horizons: read persistent data
  for (int h = 0; h < max_num_horizons_; ++h) {
    readPersistentData(h);
  }

  // BBH logic
  processBBHMode();

  // Interpolation
  for (int h = 0; h < max_num_horizons_; ++h) {
    if (!bah_horizon_active_[h]) continue;
    InterpolateMetricData(h);
  }

  // Solve
  for (int h = 0; h < max_num_horizons_; ++h) {
    if (!bah_horizon_active_[h]) {
      free(params_data_[h].input_metric_data);
      params_data_[h].input_metric_data = nullptr;
      continue;
    }
    SolveHorizon(h);
    writePersistentData(h);
  }
  last_find_time_ = pmbp_->pmesh->time;
}

void BHAHAHorizonFinder::UpdateMinimumLapseCenter() {
  if (pmbp_->padm == nullptr) {
    std::cerr << "BHaHAHA minimum-lapse centering requires ADM variables." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmbp_->pmesh->mb_indcs;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  int const nmkji = pmbp_->nmb_thispack * nx3 * nx2 * nx1;
  int const nji = nx2 * nx1;
  int const nkji = nx3 * nji;

  auto alpha = pmbp_->padm->adm.alpha;
  Real local_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("BHaHAHA minimum lapse",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_alpha) {
    int const m = idx / nkji;
    int const rem = idx - m * nkji;
    int const k0 = rem / nji;
    int const rem_j = rem - k0 * nji;
    int const j0 = rem_j / nx1;
    int const i0 = rem_j - j0 * nx1;
    min_alpha = fmin(alpha(m, k0 + ks, j0 + js, i0 + is), min_alpha);
  }, Kokkos::Min<Real>(local_min));

  double global_min = static_cast<double>(local_min);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

  auto &size = pmbp_->pmb->mb_size;
  Real count = 0.0;
  Real sx = 0.0;
  Real sy = 0.0;
  Real sz = 0.0;
  Real const min_alpha = static_cast<Real>(global_min);
  Real const tol = static_cast<Real>(10.0 * std::numeric_limits<double>::epsilon() *
                                    std::max(1.0, std::abs(global_min)));
  Kokkos::parallel_reduce("BHaHAHA minimum lapse center",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &nmatch, Real &sumx, Real &sumy, Real &sumz) {
    int const m = idx / nkji;
    int const rem = idx - m * nkji;
    int const k0 = rem / nji;
    int const rem_j = rem - k0 * nji;
    int const j0 = rem_j / nx1;
    int const i0 = rem_j - j0 * nx1;

    Real const lapse = alpha(m, k0 + ks, j0 + js, i0 + is);
    if (fabs(lapse - min_alpha) <= tol) {
      nmatch += 1.0;
      sumx += CellCenterX(i0, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      sumy += CellCenterX(j0, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      sumz += CellCenterX(k0, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    }
  }, Kokkos::Sum<Real>(count), Kokkos::Sum<Real>(sx), Kokkos::Sum<Real>(sy),
     Kokkos::Sum<Real>(sz));

  double global_sums[] = {
    static_cast<double>(count),
    static_cast<double>(sx),
    static_cast<double>(sy),
    static_cast<double>(sz),
  };
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, global_sums, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (global_sums[0] <= 0.0) {
    std::cerr << "BHaHAHA minimum-lapse centering could not locate a minimum-lapse cell."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  min_lapse_center_[0] = global_sums[1] / global_sums[0];
  min_lapse_center_[1] = global_sums[2] / global_sums[0];
  min_lapse_center_[2] = global_sums[3] / global_sums[0];
}

void BHAHAHorizonFinder::readPersistentData(int h) {
  // Load into params_data_[h]
  auto &pd = params_data_[h];
  // Scalar historical
  if (center_mode_ == CenterMode::MinimumLapse) {
    pd.x_center_m1 = min_lapse_center_[0];
    pd.y_center_m1 = min_lapse_center_[1];
    pd.z_center_m1 = min_lapse_center_[2];
  } else if (h!=2) {
    if (h >= static_cast<int>(pmbp_->pz4c->ptracker.size())) {
      std::cerr << "BHaHAHA tracker centering requested horizon " << h
                << " but only " << pmbp_->pz4c->ptracker.size()
                << " compact-object trackers are configured." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real * ptrack =  pmbp_->pz4c->ptracker[h]->GetPos();
    pd.x_center_m1 = ptrack[0];
    pd.y_center_m1 = ptrack[1];
    pd.z_center_m1 = ptrack[2];
  } else {
    if (pmbp_->pz4c->ptracker.size() < 2) {
      std::cerr << "BHaHAHA common-horizon tracker centering requires at least two "
                << "compact-object trackers." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real * ptrack0 =  pmbp_->pz4c->ptracker[0]->GetPos();
    Real * ptrack1 =  pmbp_->pz4c->ptracker[1]->GetPos();
    int bh1 = bah_BBH_mode_inspiral_BH_idxs_[0];
    int bh2 = bah_BBH_mode_inspiral_BH_idxs_[1];

    pd.x_center_m1 = (m_guess[bh1]*ptrack0[0]+m_guess[bh2]*ptrack1[0])/(m_guess[bh1]+m_guess[bh2]);
    pd.y_center_m1 = (m_guess[bh1]*ptrack0[1]+m_guess[bh2]*ptrack1[1])/(m_guess[bh1]+m_guess[bh2]);
    pd.z_center_m1 = (m_guess[bh1]*ptrack0[2]+m_guess[bh2]*ptrack1[2])/(m_guess[bh1]+m_guess[bh2]);
  }

  pd.t_m1 = t_m1_[h]; pd.t_m2 = t_m2_[h]; pd.t_m3 = t_m3_[h];
  pd.r_min_m1 = r_min_m1_[h]; pd.r_max_m1 = r_max_m1_[h];
  pd.r_min_m2 = r_min_m2_[h]; pd.r_max_m2 = r_max_m2_[h];
  pd.r_min_m3 = r_min_m3_[h]; pd.r_max_m3 = r_max_m3_[h];

  pd.use_fixed_radius_guess_on_full_sphere = use_fixed_radius_guess_on_full_sphere_[h];
  // Array historical shapes
  pd.prev_horizon_m1 = prev_horizon_m1_[h].data();
  pd.prev_horizon_m2 = prev_horizon_m2_[h].data();
  pd.prev_horizon_m3 = prev_horizon_m3_[h].data();
}

void BHAHAHorizonFinder::writePersistentData(int h) {
  auto &pd = params_data_[h];
  r_min_m1_[h] = pd.r_min_m1; r_max_m1_[h] = pd.r_max_m1;
  r_min_m2_[h] = pd.r_min_m2; r_max_m2_[h] = pd.r_max_m2;
  r_min_m3_[h] = pd.r_min_m3; r_max_m3_[h] = pd.r_max_m3;
  t_m1_[h] = pd.t_m1;
  t_m2_[h] = pd.t_m2;
  t_m3_[h] = pd.t_m3;
  use_fixed_radius_guess_on_full_sphere_[h] = pd.use_fixed_radius_guess_on_full_sphere;
  // shapes updated in pd.prev_horizon_* by solver
}

void BHAHAHorizonFinder::processBBHMode() {
  if (!bah_BBH_mode_enable_) return;
  int bh1 = bah_BBH_mode_inspiral_BH_idxs_[0];
  int bh2 = bah_BBH_mode_inspiral_BH_idxs_[1];
  int com = bah_BBH_mode_common_horizon_idx_;
  bool common_found = (use_fixed_radius_guess_on_full_sphere_[com]==0);
  if (common_found && bah_horizon_active_[bh1] && bah_horizon_active_[bh2]) {
    bah_horizon_active_[bh1]=0; bah_horizon_active_[bh2]=0;
  }
  if (bah_horizon_active_[bh1] && bah_horizon_active_[bh2] && !bah_horizon_active_[com]) {
    // compute separation + radii and possibly activate common
    double dx = params_data_[bh1].x_center_m1-params_data_[bh2].x_center_m1;
    double dy = params_data_[bh1].y_center_m1-params_data_[bh2].y_center_m1;
    double dz = params_data_[bh1].z_center_m1-params_data_[bh2].z_center_m1;
    double dist = std::sqrt(dx*dx+dy*dy+dz*dz);
    double thr = 2.0*r_max_m1_[com];
    if (dist + r_max_m1_[bh1] + r_max_m1_[bh2] <= thr) {
      bah_horizon_active_[com] = 1;
      // coarse COM center
      double xc=(m_guess[bh1]*params_data_[bh1].x_center_m1+m_guess[bh2]*params_data_[bh2].x_center_m1)/(m_guess[bh1]+m_guess[bh2]);
      double yc=(m_guess[bh1]*params_data_[bh1].y_center_m1+m_guess[bh2]*params_data_[bh2].y_center_m1)/(m_guess[bh1]+m_guess[bh2]);
      double zc=(m_guess[bh1]*params_data_[bh1].z_center_m1+m_guess[bh2]*params_data_[bh2].z_center_m1)/(m_guess[bh1]+m_guess[bh2]);
      x_center_m1_[com]=xc; y_center_m1_[com]=yc; z_center_m1_[com]=zc;
      t_m1_[com]=t_m2_[com]=t_m3_[com]=-1.0;
      r_min_m1_[com]=0.0;
    }
  }
}

void BHAHAHorizonFinder::SetGridCoordinates(int h) {
  auto &pd = params_data_[h];

  // number of radial points (== Nr_interp_)
  int Nr = static_cast<int>(radii_.size());

  // angular cell widths
  double dtheta = M_PI / static_cast<double>(Ntheta_);
  double dphi   = 2.0 * M_PI / static_cast<double>(Nphi_);

  // horizon center for this index
  double x0 = pd.x_center_m1;
  double y0 = pd.y_center_m1;
  double z0 = pd.z_center_m1;

  // loop over azimuth
  for (int iphi = 0; iphi < Nphi_; ++iphi) {
    double phi    = -M_PI + (iphi + 0.5) * dphi;
    double sinphi = std::sin(phi), cosphi = std::cos(phi);

    // loop over polar
    for (int itheta = 0; itheta < Ntheta_; ++itheta) {
      double theta    = (itheta + 0.5) * dtheta;
      double sintheta = std::sin(theta), costheta = std::cos(theta);

      // loop over radius
      for (int ir = 0; ir < Nr; ++ir) {
        // linear index: r fastest, then θ, then φ
        size_t idx = ir + Nr * (itheta + Ntheta_ * iphi);
        double r   = radii_[ir];

        // fill the coordinate triplet
        cart_coords_[idx][0] = x0 + r * sintheta * cosphi;
        cart_coords_[idx][1] = y0 + r * sintheta * sinphi;
        cart_coords_[idx][2] = z0 + r * costheta;
      }
    }
  }
}

void BHAHAHorizonFinder::InterpolateMetricData(int h) {
  auto &pd = params_data_[h];
  pd.which_horizon = h+1;
  pd.num_horizons = max_num_horizons_;
  pd.iteration_external_input = pmbp_->pmesh->ncycle;
  pd.time_external_input = pmbp_->pmesh->time;
  pd.num_resolutions_multigrid = bah_num_resolutions_multigrid_;
  for (int i=0;i<bah_num_resolutions_multigrid_;++i) {
    pd.Ntheta_array_multigrid[i] = bah_Ntheta_array_multigrid_[i];
    pd.Nphi_array_multigrid[i]   = bah_Nphi_array_multigrid_[i];
  }

  Real x_extrap, y_extrap, z_extrap, r_min_extrap, r_max_extrap; // Variables to store extrapolated values.
  // `bah_xyz_center_r_minmax` performs the extrapolation using historical data (e.g., centers, radii, times at m1, m2, m3)
  bah_xyz_center_r_minmax(&pd, &x_extrap, &y_extrap, &z_extrap, &r_min_extrap, &r_max_extrap);

  // If extrapolation suggests (or if it was already set that) a full sphere guess is needed, enforce it.
  //if (pd.use_fixed_radius_guess_on_full_sphere) {

  //}
  r_min_extrap = 0.0;                      // No inner boundary for a full sphere guess.
  r_max_extrap = max_search_radius_*m_guess[h]; // Use the maximum search radius parameter.
  // radial grid
  bah_radial_grid_cell_centered_set_up(Nr_interp_, max_search_radius_*m_guess[h], r_min_extrap, r_max_extrap,
                                       &pd.Nr_external_input, &pd.r_min_external_input, &pd.dr_external_input,
                                       radii_.data());
  // coords
  SetGridCoordinates(h);

  size_t pts = static_cast<size_t>(pd.Nr_external_input)*Ntheta_*Nphi_;
  size_t total = pts*NUM_EXT_INPUT_CARTESIAN_GFS;
  pd.input_metric_data = (double*)malloc(total*sizeof(double));
  agrid_->ResetGrid(cart_coords_);
  agrid_->ResetCenter(pd.x_center_m1,pd.y_center_m1,pd.z_center_m1);

  // loop gridfunctions
  for (int gf=0;gf<NUM_EXT_INPUT_CARTESIAN_GFS;++gf) {
    agrid_->InterpolateToGrid(gf, pmbp_->padm->u_adm);
    for (size_t i=0;i<pts;++i) {
      pd.input_metric_data[gf*pts + i] = agrid_->interp_vals.h_view(i);
    }
  }
  // MPI reduce here
  // Reduction to the master rank for data_out
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, pd.input_metric_data, total,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(pd.input_metric_data, pd.input_metric_data, total, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  #endif
}

void BHAHAHorizonFinder::SolveHorizon(int h) {
  auto &pd = params_data_[h];
  bhahaha_diagnostics_struct diags;
  if (0 == global_variable::my_rank) {
    std::cout << "Finding Horizon" << std::endl;
    bah_poisoning_check_inputs(&pd);
    int rc = bah_find_horizon(&pd, &diags);
    if (rc == BHAHAHA_SUCCESS) {
      std::cout << "Success" << std::endl;
      bah_diagnostics_file_output(&diags, &pd, max_num_horizons_, pd.x_center_m1, pd.y_center_m1, pd.z_center_m1, "./horizon");
      pd.use_fixed_radius_guess_on_full_sphere = 0;
      pd.t_m1 = pmbp_->pmesh->time;
    } else {
      std::cout << "Failed with Error Flag " << rc << std::endl; 
      // ATHENA_ERROR("Horizon %d find failed rc=%d: %s", h+1, rc, bah_error_message((bhahaha_error_codes)rc));
      pd.use_fixed_radius_guess_on_full_sphere = 1;
      pd.t_m1 = -1.0;
    }
  }

  #if MPI_PARALLEL_ENABLED
  // Keep every rank's persistent horizon history consistent with the rank-0 solve.
  double scalars[] = {
    pd.t_m1, pd.t_m2, pd.t_m3,
    pd.r_min_m1, pd.r_max_m1,
    pd.r_min_m2, pd.r_max_m2,
    pd.r_min_m3, pd.r_max_m3,
  };
  MPI_Bcast(scalars, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  pd.t_m1 = scalars[0]; pd.t_m2 = scalars[1]; pd.t_m3 = scalars[2];
  pd.r_min_m1 = scalars[3]; pd.r_max_m1 = scalars[4];
  pd.r_min_m2 = scalars[5]; pd.r_max_m2 = scalars[6];
  pd.r_min_m3 = scalars[7]; pd.r_max_m3 = scalars[8];
  MPI_Bcast(&pd.use_fixed_radius_guess_on_full_sphere, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(pd.prev_horizon_m1,
            Ntheta_*Nphi_,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);
  MPI_Bcast(bah_horizon_active_.data(),
            max_num_horizons_,
            MPI_INT,
            0,
            MPI_COMM_WORLD);
  #endif

  free(pd.input_metric_data);
  pd.input_metric_data = nullptr;
}
