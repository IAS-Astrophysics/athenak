#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>  // mkdir

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <utility>

extern "C" {
  #include "BHaHAHA.h"
}

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "z4c/z4c.hpp"
#include "z4c/BHaHAHA_horizon_finder.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "coordinates/adm.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/arbitrary_grid_interpolator.hpp"

BHAHAHorizonFinder::BHAHAHorizonFinder(MeshBlockPack *pmbp, ParameterInput *pin)
  : pmbp_(pmbp), pin_(pin) {
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
  agrid_ = new ArbitraryGrid(pmbp_, cart_coords_, 4);
  
  // initialize input_metric_data to null pointers
  int h = 0;
  for (auto &pd : params_data_) {
    bah_poisoning_set_inputs(&pd);
    pd.input_metric_data = nullptr;
    pd.use_fixed_radius_guess_on_full_sphere = 1;
    pd.cfl_factor = 1.05;
    pd.M_scale = m_guess[h];
    pd.eta_damping_times_M = 1.6;
    pd.KO_strength = 0;
    pd.max_iterations = 10000;
    pd.Theta_Linf_times_M_tolerance = 1e-2;
    pd.Theta_L2_times_M_tolerance = 2e-5;
    pd.enable_eta_varying_alg_for_precision_common_horizon = 0;
    pd.verbosity_level = 2;
    h++;
  }
}

BHAHAHorizonFinder::~BHAHAHorizonFinder() {
  delete agrid_;
}

void BHAHAHorizonFinder::LoadParameters() {
  find_every_ = pin_->GetOrAddInteger("z4c", "bah_find_every", 1);
  max_num_horizons_ = pin_->GetOrAddInteger("z4c", "bah_num_horizons", 1);

  bah_num_resolutions_multigrid_ = pin_->GetOrAddInteger("z4c", "bah_num_resolutions_multigrid", 1);
  bah_Ntheta_array_multigrid_.resize(bah_num_resolutions_multigrid_);
  bah_Nphi_array_multigrid_.resize(bah_num_resolutions_multigrid_);
  max_Ntheta_ = pin_->GetOrAddInteger("z4c", "bah_Ntheta", 32);
  max_Nphi_   = pin_->GetOrAddInteger("z4c", "bah_Nphi", 64);
  Nr_interp_  = pin_->GetOrAddInteger("z4c", "bah_Nr_interp", 48);
  Ntheta_     = pin_->GetOrAddInteger("z4c", "bah_Ntheta", 32);
  Nphi_       = pin_->GetOrAddInteger("z4c", "bah_Nphi", 64);
  max_search_radius_ = pin_->GetOrAddReal("z4c", "bah_max_search_radius", 0.6);
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
  if (find_every_ == 0) return;
  if (pmbp_->pmesh->ncycle % find_every_ != 0) return;

  checkMultigridResolutionInputs();

  // STEP: process BBH mode initial activation
  if (bah_BBH_mode_enable_) {
    if (max_num_horizons_ != 3) {
      std::cerr << "BBH mode requires 3 horizons" << std::endl;
      abort();
    }
    bah_horizon_active_.assign(max_num_horizons_, 1);
    use_fixed_radius_guess_on_full_sphere_.assign(max_num_horizons_, 1);
  }

  timeval start, mid;
  gettimeofday(&start, nullptr);

  // Loop horizons: read persistent data
  for (int h = 0; h < max_num_horizons_; ++h) {
    readPersistentData(h);
  }

  // BBH logic
  processBBHMode();

  // Diagnostics pre-interp
  gettimeofday(&mid, nullptr);
  //diagnosticPrintPreInterpolation(timevalToSeconds(start, mid));

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
}

void BHAHAHorizonFinder::readPersistentData(int h) {
  int bh1 = bah_BBH_mode_inspiral_BH_idxs_[0];
  int bh2 = bah_BBH_mode_inspiral_BH_idxs_[1];
  // Load into params_data_[h]
  auto &pd = params_data_[h];
  // Scalar historical
  if (h!=2) {
    Real * ptrack =  pmbp_->pz4c->ptracker[h]->GetPos();
    pd.x_center_m1 = ptrack[0];
    pd.y_center_m1 = ptrack[1];
    pd.z_center_m1 = ptrack[2];

  } else {
    Real * ptrack0 =  pmbp_->pz4c->ptracker[0]->GetPos();
    Real * ptrack1 =  pmbp_->pz4c->ptracker[1]->GetPos();

    pd.x_center_m1 = (m_guess[bh1]*ptrack0[0]+m_guess[bh2]*ptrack1[0])/(m_guess[bh1]+m_guess[bh2]);
    pd.y_center_m1 = (m_guess[bh1]*ptrack0[1]+m_guess[bh2]*ptrack1[1])/(m_guess[bh1]+m_guess[bh2]);
    pd.z_center_m1 = (m_guess[bh1]*ptrack0[2]+m_guess[bh2]*ptrack1[2])/(m_guess[bh1]+m_guess[bh2]);
  }

  pd.t_m1 = t_m1_[h]; pd.t_m2 = t_m2_[h]; pd.t_m3 = t_m3_[h];
  pd.r_min_m1 = r_min_m1_[h]; pd.r_max_m1 = r_max_m1_[h];
  pd.r_min_m2 = r_min_m2_[h]; pd.r_max_m2 = r_max_m2_[h];
  pd.r_min_m3 = r_min_m3_[h]; pd.r_max_m3 = r_max_m3_[h];
  // Array historical shapes
  pd.prev_horizon_m1 = prev_horizon_m1_[h].data();
  pd.prev_horizon_m2 = prev_horizon_m2_[h].data();
  pd.prev_horizon_m3 = prev_horizon_m3_[h].data();
}

void BHAHAHorizonFinder::writePersistentData(int h) {
  auto &pd = params_data_[h];
  x_center_m1_[h] = pd.x_center_m1; y_center_m1_[h] = pd.y_center_m1; z_center_m1_[h] = pd.z_center_m1;
  t_m1_[h] = pd.t_m1;
  r_min_m1_[h] = pd.r_min_m1; r_max_m1_[h] = pd.r_max_m1;
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
    double dx = x_center_m1_[bh1]-x_center_m1_[bh2];
    double dy = y_center_m1_[bh1]-y_center_m1_[bh2];
    double dz = z_center_m1_[bh1]-z_center_m1_[bh2];
    double dist = std::sqrt(dx*dx+dy*dy+dz*dz);
    double thr = 2.0*max_search_radius_;
    if (dist + r_max_m1_[bh1] + r_max_m1_[bh2] <= thr) {
      bah_horizon_active_[com] = 1;
      // coarse COM center
      double xc=(m_guess[bh1]*x_center_m1_[bh1]+m_guess[bh2]*x_center_m1_[bh2])/(m_guess[bh1]+m_guess[bh2]);
      double yc=(m_guess[bh1]*y_center_m1_[bh1]+m_guess[bh2]*y_center_m1_[bh2])/(m_guess[bh1]+m_guess[bh2]);
      double zc=(m_guess[bh1]*z_center_m1_[bh1]+m_guess[bh2]*z_center_m1_[bh2])/(m_guess[bh1]+m_guess[bh2]);
      x_center_m1_[com]=xc; y_center_m1_[com]=yc; z_center_m1_[com]=zc;
      t_m1_[com]=t_m2_[com]=t_m3_[com]=-1.0;
      r_min_m1_[com]=0.0;
    }
  }
}
/*
void BHAHAHorizonFinder::diagnosticPrintPreInterpolation(double start_time) {
  // print states
  ATHENA_INFO("--- Pre-Interp State at time = %f ---", pmbp_->pmesh->time);
  for (int h=0; h<max_num_horizons_; ++h) {
    ATHENA_INFO("H%d: active=%d, fixed_full=%d, center=(%f,%f,%f), r=[%f,%f]", h,
                bah_horizon_active_[h], use_fixed_radius_guess_on_full_sphere_[h],
                x_center_m1_[h], y_center_m1_[h], z_center_m1_[h], r_min_m1_[h], r_max_m1_[h]);
  }
}
*/
double BHAHAHorizonFinder::timevalToSeconds(const timeval &start, const timeval &end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
}

void BHAHAHorizonFinder::SetGridCoordinates(int h) {
  // number of radial points (== Nr_interp_)
  int Nr = static_cast<int>(radii_.size());

  // angular cell widths
  double dtheta = M_PI / static_cast<double>(Ntheta_);
  double dphi   = 2.0 * M_PI / static_cast<double>(Nphi_);

  // horizon center for this index
  double x0 = x_center_m1_[h];
  double y0 = y_center_m1_[h];
  double z0 = z_center_m1_[h];

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
  pd.verbosity_level=2;

  // radial grid
  bah_radial_grid_cell_centered_set_up(Nr_interp_, max_search_radius_, r_min_m1_[h], r_max_m1_[h],
                                       &pd.Nr_external_input, &pd.r_min_external_input, &pd.dr_external_input,
                                       radii_.data());
  // coords
  SetGridCoordinates(h);

  size_t pts = static_cast<size_t>(pd.Nr_external_input)*Ntheta_*Nphi_;
  size_t total = pts*NUM_EXT_INPUT_CARTESIAN_GFS;
  pd.input_metric_data = (double*)malloc(total*sizeof(double));
  agrid_->ResetGrid(cart_coords_);
  agrid_->ResetCenter(x_center_m1_[h],y_center_m1_[h],z_center_m1_[h]);

  // loop gridfunctions
  for (int gf=0;gf<NUM_EXT_INPUT_CARTESIAN_GFS;++gf) {
    agrid_->InterpolateToGrid(gf, pmbp_->padm->u_adm);
    for (size_t i=0;i<pts;++i) {
      /*double r = std::sqrt(SQR(cart_coords_[i][0])+SQR(cart_coords_[i][1])+SQR(cart_coords_[i][2]));
      double psi4 = std::pow(1.0 + 0.5*1/r,4);
      if (gf == 0 || gf == 3 || gf==5) {
        pd.input_metric_data[gf*pts + i] = psi4;
        std::cout << abs(psi4-agrid_->interp_vals.h_view(i)) << std::endl;
      } else {
        pd.input_metric_data[gf*pts + i] = 0;
      }*/
      pd.input_metric_data[gf*pts + i] = agrid_->interp_vals.h_view(i);
    }
  }
  // MPI reduce here
  // Reduction to the master rank for data_out
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, pd.input_metric_data, total,
              MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(pd.input_metric_data, pd.input_metric_data, total, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  #endif
}

void BHAHAHorizonFinder::SolveHorizon(int h) {
  auto &pd = params_data_[h];
  bhahaha_diagnostics_struct diags;
  if (0 == global_variable::my_rank) {
    bah_poisoning_check_inputs(&pd);
    int rc = bah_find_horizon(&pd, &diags);
    if (rc == BHAHAHA_SUCCESS) {
      bah_diagnostics_file_output(&diags, &pd, max_num_horizons_, pd.x_center_m1, pd.y_center_m1, pd.z_center_m1, ".");
      pd.use_fixed_radius_guess_on_full_sphere = 0;
    } else {
      // ATHENA_ERROR("Horizon %d find failed rc=%d: %s", h+1, rc, bah_error_message((bhahaha_error_codes)rc));
      pd.use_fixed_radius_guess_on_full_sphere = 1;
      t_m1_[h] = -1.0;
    }
  }

  #if MPI_PARALLEL_ENABLED
  // ---- broadcast the just-found horizon radius array to all ranks ----
  // prev_horizon_m1_[h] is a std::vector<double> of length Ntheta_*Nphi_
  MPI_Bcast(pd.prev_horizon_m1,
            Ntheta_*Nphi_,
            MPI_DOUBLE,        // or MPI_ATHENA_REAL if you prefer
            0,                 // root rank
            MPI_COMM_WORLD);
  #endif

  free(pd.input_metric_data);
  pd.input_metric_data = nullptr;
}
