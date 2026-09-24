#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>  // mkdir

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
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
#include <Kokkos_Timer.hpp>
#include "athena.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/arbitrary_grid_interpolator.hpp"

BHAHAHorizonFinder::BHAHAHorizonFinder(MeshBlockPack *pmbp, ParameterInput *pin)
  : pmbp_(pmbp), pin_(pin) {
  LoadParameters();
  checkMultigridResolutionInputs();

  bah_horizon_active_.assign(max_num_horizons_, 1);
  prev_horizon_m1_.assign(max_num_horizons_, std::vector<double>(Ntheta_*Nphi_, 0.0));
  prev_horizon_m2_ = prev_horizon_m1_;
  prev_horizon_m3_ = prev_horizon_m1_;
  grid_center_.assign(max_num_horizons_, {0.0, 0.0, 0.0});
  nfinds_.assign(max_num_horizons_, 0);
  found_.assign(max_num_horizons_, 0);
  mass_.assign(max_num_horizons_, 0.0);
  center_.assign(max_num_horizons_, {0.0, 0.0, 0.0});
  spin_.assign(max_num_horizons_, {0.0, 0.0, 0.0});
  input_buf_.resize(max_num_horizons_);
  time_interp_.assign(max_num_horizons_, 0.0);
  time_mpi_.assign(max_num_horizons_, 0.0);
  time_solve_.assign(max_num_horizons_, 0.0);

  params_data_.resize(max_num_horizons_);
  radii_.resize(Nr_interp_);
  cart_coords_.resize(static_cast<size_t>(Nr_interp_) * Ntheta_ * Nphi_);
  agrid_ = std::make_unique<ArbitraryGrid>(pmbp_, cart_coords_, 4, interp_half_width_);

  for (int h = 0; h < max_num_horizons_; ++h) {
    auto &pd = params_data_[h];
    bah_poisoning_set_inputs(&pd);
    pd.input_metric_data = nullptr;
    pd.prev_horizon_m1 = prev_horizon_m1_[h].data();
    pd.prev_horizon_m2 = prev_horizon_m2_[h].data();
    pd.prev_horizon_m3 = prev_horizon_m3_[h].data();
    pd.cfl_factor = pin_->GetOrAddReal("bhahaha", "bah_cfl", 0.4);
    pd.M_scale = m_guess[h];
    pd.eta_damping_times_M = 1.6;
    pd.KO_strength = 0;
    pd.max_iterations = pin_->GetOrAddInteger("bhahaha", "bah_max_itr", 10000);
    pd.Theta_Linf_times_M_tolerance = pin_->GetOrAddReal("bhahaha", "bah_Theta_Linf_tol", 1e-2);
    pd.Theta_L2_times_M_tolerance = pin_->GetOrAddReal("bhahaha", "bah_Theta_L2_tol", 2e-5);
    pd.enable_eta_varying_alg_for_precision_common_horizon = 0;
    pd.verbosity_level = verbosity_;
    resetHorizonHistory(h);
  }

  if (bah_BBH_mode_enable_) {
    if (max_num_horizons_ != 3) {
      std::cerr << "BBH mode requires 3 horizons" << std::endl;
      abort();
    }
    // common horizon not active at initial time
    bah_horizon_active_[bah_BBH_mode_common_horizon_idx_] = 0;
  }
}

BHAHAHorizonFinder::~BHAHAHorizonFinder() = default;

void BHAHAHorizonFinder::Find(Driver *pdrive, int stage) {
  if (stage != pdrive->nexp_stages) return;
  FindHorizons();
}

void BHAHAHorizonFinder::LoadParameters() {
  find_every_ = pin_->GetOrAddInteger("bhahaha", "bah_find_every", 1);
  dt_find_ = pin_->GetOrAddReal("bhahaha", "bah_dt", 0.0);
  last_find_time_ = -1.0e30;
  output_shape_every_ = pin_->GetOrAddInteger("bhahaha", "bah_output_shape_every", 1);
  interp_half_width_ = pin_->GetOrAddInteger("bhahaha", "bah_interp_half_width", 0);
  verbosity_ = pin_->GetOrAddInteger("bhahaha", "bah_verbosity", 0);
  max_num_horizons_ = pin_->GetOrAddInteger("bhahaha", "bah_num_horizons", 1);

  bah_num_resolutions_multigrid_ = pin_->GetOrAddInteger("bhahaha", "bah_num_resolutions_multigrid", 1);
  bah_Ntheta_array_multigrid_.resize(bah_num_resolutions_multigrid_);
  bah_Nphi_array_multigrid_.resize(bah_num_resolutions_multigrid_);
  Nr_interp_  = pin_->GetOrAddInteger("bhahaha", "bah_Nr_interp", 48);
  Ntheta_     = pin_->GetOrAddInteger("bhahaha", "bah_Ntheta", 32);
  Nphi_       = pin_->GetOrAddInteger("bhahaha", "bah_Nphi", 64);
  max_search_radius_ = pin_->GetOrAddReal("bhahaha", "bah_max_search_radius", 2);
  for (int i = 0; i < bah_num_resolutions_multigrid_; ++i) {
    bah_Ntheta_array_multigrid_[i] = pin_->GetOrAddInteger("bhahaha", "bah_Ntheta_array_multigrid_"+std::to_string(i), Ntheta_);
    bah_Nphi_array_multigrid_[i]   = pin_->GetOrAddInteger("bhahaha", "bah_Nphi_array_multigrid_"+std::to_string(i), Nphi_);
  }
  // BHaHAHA expects the input data at the resolution of the finest multigrid level
  if (bah_num_resolutions_multigrid_ > 0) {
    Ntheta_ = bah_Ntheta_array_multigrid_[bah_num_resolutions_multigrid_ - 1];
    Nphi_   = bah_Nphi_array_multigrid_[bah_num_resolutions_multigrid_ - 1];
  }
  bah_BBH_mode_enable_ = pin_->GetOrAddBoolean("bhahaha", "bah_BBH_mode_enable", false);
  bah_BBH_mode_inspiral_BH_idxs_[0] = pin_->GetOrAddInteger("bhahaha", "bah_BBH_mode_inspiral_BH_idxs_0", 0);
  bah_BBH_mode_inspiral_BH_idxs_[1] = pin_->GetOrAddInteger("bhahaha", "bah_BBH_mode_inspiral_BH_idxs_1", 1);
  bah_BBH_mode_common_horizon_idx_ = pin_->GetOrAddInteger("bhahaha", "bah_BBH_mode_common_horizon_idx", 2);

  m_guess.assign(max_num_horizons_,0.0);
  for (int h = 0; h < max_num_horizons_; ++h) {
    m_guess[h] = pin_->GetOrAddReal("bhahaha", "bah_mass_"+std::to_string(h), 1.0);
  }
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
  Real time = pmbp_->pmesh->time;
  if (dt_find_ > 0.0) {
    if (static_cast<float>(time) < static_cast<float>(last_find_time_ + dt_find_)) return;
    last_find_time_ = time;
  } else if (find_every_ == 0 || pmbp_->pmesh->ncycle % find_every_ != 0) {
    return;
  }

  for (int h = 0; h < max_num_horizons_; ++h) {
    readPersistentData(h);
  }

  processBBHMode();

  for (int h = 0; h < max_num_horizons_; ++h) {
    if (!bah_horizon_active_[h]) found_[h] = 0;
  }

  for (int h = 0; h < max_num_horizons_; ++h) {
    if (!bah_horizon_active_[h]) continue;
    InterpolateMetricData(h);
  }
  for (int h = 0; h < max_num_horizons_; ++h) {
    if (!bah_horizon_active_[h]) continue;
    if (global_variable::my_rank == rootRank(h)) SolveHorizon(h);
  }
  for (int h = 0; h < max_num_horizons_; ++h) {
    if (!bah_horizon_active_[h]) continue;
    broadcastHorizonState(h);
    auto &pd = params_data_[h];
    center_[h] = {pd.x_center_m1, pd.y_center_m1, pd.z_center_m1};
    if (verbosity_ > 0 && global_variable::my_rank == rootRank(h)) {
      std::cout << "BHaHAHA horizon " << h << " timing [s]: interpolation "
                << time_interp_[h] << ", MPI " << time_mpi_[h] << ", solve "
                << time_solve_[h] << std::endl;
    }
  }
}

int BHAHAHorizonFinder::rootRank(int h) const {
  return h % global_variable::nranks;
}

void BHAHAHorizonFinder::resetHorizonHistory(int h) {
  auto &pd = params_data_[h];
  pd.t_m1 = pd.t_m2 = pd.t_m3 = -1.0;
  pd.x_center_m1 = pd.x_center_m2 = pd.x_center_m3 = 0.0;
  pd.y_center_m1 = pd.y_center_m2 = pd.y_center_m3 = 0.0;
  pd.z_center_m1 = pd.z_center_m2 = pd.z_center_m3 = 0.0;
  pd.r_min_m1 = pd.r_min_m2 = pd.r_min_m3 = 0.0;
  pd.r_max_m1 = pd.r_max_m2 = pd.r_max_m3 = max_search_radius_*m_guess[h];
  pd.use_fixed_radius_guess_on_full_sphere = 1;
}

void BHAHAHorizonFinder::readPersistentData(int h) {
  // Once a horizon has been found, BHaHAHA extrapolates the center from its own history.
  // Only without history is the search centered on the puncture tracker(s).
  auto &pd = params_data_[h];
  if (!pd.use_fixed_radius_guess_on_full_sphere) return;

  if (bah_BBH_mode_enable_ && h == bah_BBH_mode_common_horizon_idx_) {
    int bh1 = bah_BBH_mode_inspiral_BH_idxs_[0];
    int bh2 = bah_BBH_mode_inspiral_BH_idxs_[1];
    Real *ptrack1 = pmbp_->pz4c->ptracker[bh1]->GetPos();
    Real *ptrack2 = pmbp_->pz4c->ptracker[bh2]->GetPos();
    double mtot = m_guess[bh1] + m_guess[bh2];
    pd.x_center_m1 = (m_guess[bh1]*ptrack1[0] + m_guess[bh2]*ptrack2[0])/mtot;
    pd.y_center_m1 = (m_guess[bh1]*ptrack1[1] + m_guess[bh2]*ptrack2[1])/mtot;
    pd.z_center_m1 = (m_guess[bh1]*ptrack1[2] + m_guess[bh2]*ptrack2[2])/mtot;
  } else {
    Real *ptrack = pmbp_->pz4c->ptracker[h]->GetPos();
    pd.x_center_m1 = ptrack[0];
    pd.y_center_m1 = ptrack[1];
    pd.z_center_m1 = ptrack[2];
  }
}

void BHAHAHorizonFinder::broadcastHorizonState(int h) {
#if MPI_PARALLEL_ENABLED
  Kokkos::Timer timer;
  auto &pd = params_data_[h];
  double buf[24] = {pd.t_m1, pd.t_m2, pd.t_m3,
                    pd.x_center_m1, pd.x_center_m2, pd.x_center_m3,
                    pd.y_center_m1, pd.y_center_m2, pd.y_center_m3,
                    pd.z_center_m1, pd.z_center_m2, pd.z_center_m3,
                    pd.r_min_m1, pd.r_min_m2, pd.r_min_m3,
                    pd.r_max_m1, pd.r_max_m2, pd.r_max_m3,
                    static_cast<double>(pd.use_fixed_radius_guess_on_full_sphere),
                    static_cast<double>(found_[h]), mass_[h],
                    spin_[h][0], spin_[h][1], spin_[h][2]};
  MPI_Bcast(buf, 24, MPI_DOUBLE, rootRank(h), MPI_COMM_WORLD);
  pd.t_m1 = buf[0];  pd.t_m2 = buf[1];  pd.t_m3 = buf[2];
  pd.x_center_m1 = buf[3];  pd.x_center_m2 = buf[4];  pd.x_center_m3 = buf[5];
  pd.y_center_m1 = buf[6];  pd.y_center_m2 = buf[7];  pd.y_center_m3 = buf[8];
  pd.z_center_m1 = buf[9];  pd.z_center_m2 = buf[10]; pd.z_center_m3 = buf[11];
  pd.r_min_m1 = buf[12]; pd.r_min_m2 = buf[13]; pd.r_min_m3 = buf[14];
  pd.r_max_m1 = buf[15]; pd.r_max_m2 = buf[16]; pd.r_max_m3 = buf[17];
  pd.use_fixed_radius_guess_on_full_sphere = static_cast<int>(buf[18]);
  found_[h] = static_cast<int>(buf[19]);
  mass_[h] = buf[20];
  spin_[h] = {buf[21], buf[22], buf[23]};
  time_mpi_[h] += timer.seconds();
#endif
}

void BHAHAHorizonFinder::processBBHMode() {
  if (!bah_BBH_mode_enable_) return;
  int bh1 = bah_BBH_mode_inspiral_BH_idxs_[0];
  int bh2 = bah_BBH_mode_inspiral_BH_idxs_[1];
  int com = bah_BBH_mode_common_horizon_idx_;
  auto &pd1 = params_data_[bh1];
  auto &pd2 = params_data_[bh2];
  auto &pdc = params_data_[com];
  bool common_found = (pdc.use_fixed_radius_guess_on_full_sphere == 0);
  if (common_found && bah_horizon_active_[bh1] && bah_horizon_active_[bh2]) {
    bah_horizon_active_[bh1]=0; bah_horizon_active_[bh2]=0;
  }
  if (bah_horizon_active_[bh1] && bah_horizon_active_[bh2] && !bah_horizon_active_[com]) {
    // compute separation + radii and possibly activate common
    double dx = pd1.x_center_m1 - pd2.x_center_m1;
    double dy = pd1.y_center_m1 - pd2.y_center_m1;
    double dz = pd1.z_center_m1 - pd2.z_center_m1;
    double dist = std::sqrt(dx*dx+dy*dy+dz*dz);
    double thr = 2.0*pdc.r_max_m1;
    if (dist + pd1.r_max_m1 + pd2.r_max_m1 <= thr) {
      bah_horizon_active_[com] = 1;
      resetHorizonHistory(com);
      double mtot = m_guess[bh1] + m_guess[bh2];
      pdc.x_center_m1 = (m_guess[bh1]*pd1.x_center_m1 + m_guess[bh2]*pd2.x_center_m1)/mtot;
      pdc.y_center_m1 = (m_guess[bh1]*pd1.y_center_m1 + m_guess[bh2]*pd2.y_center_m1)/mtot;
      pdc.z_center_m1 = (m_guess[bh1]*pd1.z_center_m1 + m_guess[bh2]*pd2.z_center_m1)/mtot;
    }
  }
}

void BHAHAHorizonFinder::SetGridCoordinates(int h, int Nr) {
  cart_coords_.resize(static_cast<size_t>(Nr) * Ntheta_ * Nphi_);

  // angular cell widths
  double dtheta = M_PI / static_cast<double>(Ntheta_);
  double dphi   = 2.0 * M_PI / static_cast<double>(Nphi_);

  // center of the interpolation grid for this horizon
  double x0 = grid_center_[h][0];
  double y0 = grid_center_[h][1];
  double z0 = grid_center_[h][2];

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
  Kokkos::Timer timer;
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

  // Full sphere around the tracker(s) without history, otherwise a shell around the
  // extrapolated center and radii of the previous horizons.
  const double r_search = max_search_radius_*m_guess[h];
  double xc, yc, zc, r_min, r_max;
  if (pd.use_fixed_radius_guess_on_full_sphere) {
    xc = pd.x_center_m1;
    yc = pd.y_center_m1;
    zc = pd.z_center_m1;
    r_min = 0.0;
    r_max = r_search;
  } else {
    bah_xyz_center_r_minmax(&pd, &xc, &yc, &zc, &r_min, &r_max);
  }
  grid_center_[h] = {xc, yc, zc};

  bah_radial_grid_cell_centered_set_up(Nr_interp_, r_search, r_min, r_max,
                                       &pd.Nr_external_input, &pd.r_min_external_input,
                                       &pd.dr_external_input, radii_.data());
  SetGridCoordinates(h, pd.Nr_external_input);

  size_t pts = static_cast<size_t>(pd.Nr_external_input)*Ntheta_*Nphi_;
  agrid_->ResetCenter(xc, yc, zc);
  agrid_->ResetGrid(cart_coords_);
  agrid_->InterpolateToGrid(0, NUM_EXT_INPUT_CARTESIAN_GFS, pmbp_->padm->u_adm);
  time_interp_[h] = timer.seconds();

  gatherMetricData(h, pts);
}

void BHAHAHorizonFinder::gatherMetricData(int h, size_t pts) {
  Kokkos::Timer timer;
  auto &pd = params_data_[h];
  const int ngf = NUM_EXT_INPUT_CARTESIAN_GFS;
  const int root = rootRank(h);
  const bool is_root = (global_variable::my_rank == root);
  auto &ivals = agrid_->interp_vals.h_view;
  auto &iindcs = agrid_->interp_indcs.h_view;

  double *input = nullptr;
  if (is_root) {
    size_t total = pts*ngf;
    auto &buf = input_buf_[h];
    if (buf.size() < total) {
      buf.resize(std::max(total, static_cast<size_t>(Nr_interp_)*Ntheta_*Nphi_*ngf));
    }
    std::fill(buf.begin(), buf.begin() + total, 0.0);
    input = buf.data();
  }
  pd.input_metric_data = input;

#if MPI_PARALLEL_ENABLED
  const int rec = ngf + 1;
  sendbuf_.clear();
  for (size_t n = 0; n < pts; ++n) {
    if (iindcs(n,0) == -1) continue;
    sendbuf_.push_back(static_cast<double>(n));
    for (int gf = 0; gf < ngf; ++gf) sendbuf_.push_back(ivals(n,gf));
  }
  int nsend = static_cast<int>(sendbuf_.size());
  if (is_root) recvcounts_.assign(global_variable::nranks, 0);
  MPI_Gather(&nsend, 1, MPI_INT, is_root ? recvcounts_.data() : nullptr, 1, MPI_INT,
             root, MPI_COMM_WORLD);
  if (is_root) {
    recvdispls_.assign(global_variable::nranks, 0);
    for (int r = 1; r < global_variable::nranks; ++r) {
      recvdispls_[r] = recvdispls_[r-1] + recvcounts_[r-1];
    }
    recvbuf_.resize(recvdispls_.back() + recvcounts_.back());
  }
  MPI_Gatherv(sendbuf_.data(), nsend, MPI_DOUBLE,
              is_root ? recvbuf_.data() : nullptr,
              is_root ? recvcounts_.data() : nullptr,
              is_root ? recvdispls_.data() : nullptr,
              MPI_DOUBLE, root, MPI_COMM_WORLD);
  if (is_root) {
    size_t nrec = recvbuf_.size()/rec;
    for (size_t q = 0; q < nrec; ++q) {
      const double *r = &recvbuf_[q*rec];
      size_t n = static_cast<size_t>(r[0]);
      for (int gf = 0; gf < ngf; ++gf) input[gf*pts + n] = r[1 + gf];
    }
  }
#else
  for (int gf = 0; gf < ngf; ++gf) {
    for (size_t n = 0; n < pts; ++n) input[gf*pts + n] = ivals(n,gf);
  }
#endif
  time_mpi_[h] = timer.seconds();
}

void BHAHAHorizonFinder::SolveHorizon(int h) {
  Kokkos::Timer timer;
  auto &pd = params_data_[h];
  bhahaha_diagnostics_struct diags;
  std::cout << "Finding Horizon" << std::endl;
  bah_poisoning_check_inputs(&pd);
  int rc = bah_find_horizon(&pd, &diags);
  if (rc == BHAHAHA_SUCCESS) {
    std::cout << "Success" << std::endl;
    int write_shape = (output_shape_every_ > 0) && (nfinds_[h] % output_shape_every_ == 0);
    bah_diagnostics_file_output(&diags, &pd, max_num_horizons_, grid_center_[h][0],
                                grid_center_[h][1], grid_center_[h][2], "./horizon",
                                write_shape);
    ++nfinds_[h];
    pd.use_fixed_radius_guess_on_full_sphere = 0;
    found_[h] = 1;
    mass_[h] = std::sqrt(diags.area/(16.0*M_PI));
    spin_[h] = {diags.J_x, diags.J_y, diags.J_z};
  } else {
    std::cout << "Failed with Error Flag " << rc << std::endl;
    found_[h] = 0;
    resetHorizonHistory(h);
  }
  pd.input_metric_data = nullptr;
  time_solve_[h] = timer.seconds();
}
