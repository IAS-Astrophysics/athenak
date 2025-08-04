// BHaHAHA_horizon_finder.hpp
#ifndef BHAHAHA_HORIZON_FINDER_HPP
#define BHAHAHA_HORIZON_FINDER_HPP

#include <vector>
#include <array>
#include <sys/time.h>
#include "BHaHAHA.h"

class MeshBlockPack;
class ParameterInput;
class ArbitraryGrid;

class BHAHAHorizonFinder {
public:
  BHAHAHorizonFinder(MeshBlockPack *pmbp, ParameterInput *pin);
  ~BHAHAHorizonFinder();

  // Main entry point: find all active horizons at current timestep
  void FindHorizons();
  int max_num_horizons_;

private:
  // Initialization
  void LoadParameters();
  void checkMultigridResolutionInputs();
  void initializePersistentState();

  // Persistence sync (stubbed for single-process)
  void syncPersistentAcrossRanks();
  void readPersistentData(int h);
  void writePersistentData(int h);

  // BBH mode handling
  void processBBHMode();

  // Diagnostics
  void diagnosticPrintPreInterpolation(double start_time);
  double timevalToSeconds(const timeval &start, const timeval &end);

  // Grid & interpolation
  void SetGridCoordinates(int h);
  void InterpolateMetricData(int h);

  // Core solver & cleanup
  void SolveHorizon(int h);

  // AthenaK handles
  MeshBlockPack *pmbp_;
  ParameterInput *pin_;

  // User-configurable parameters
  int find_every_;
  int bah_num_resolutions_multigrid_;
  std::vector<int> bah_Ntheta_array_multigrid_;
  std::vector<int> bah_Nphi_array_multigrid_;
  int max_Ntheta_;
  int max_Nphi_;
  int Nr_interp_;
  int Ntheta_;
  int Nphi_;
  double max_search_radius_;
  bool bah_BBH_mode_enable_;
  std::array<int,3> bah_BBH_mode_inspiral_BH_idxs_;
  int bah_BBH_mode_common_horizon_idx_;

  // Active flags and guess enforcement
  std::vector<int> bah_horizon_active_;
  std::vector<int> use_fixed_radius_guess_on_full_sphere_;

  // Persistent historical data
  // center location for bh
  std::vector<double> x_center_m1_, y_center_m1_, z_center_m1_;
  // time at which previous horizon found
  std::vector<double> t_m1_, t_m2_, t_m3_;
  // range of radius to search for each hole
  std::vector<double> r_min_m1_, r_max_m1_, r_min_m2_, r_max_m2_, r_min_m3_, r_max_m3_;
  // location of the previous horizon, used to initialize the next finder
  std::vector<std::vector<double>> prev_horizon_m1_, prev_horizon_m2_, prev_horizon_m3_;
  // guess for the mass of black hole
  std::vector<double> m_guess;

  // Per-horizon parameters and data
  std::vector<bhahaha_params_and_data_struct> params_data_;

  // Interpolation buffers
  std::vector<std::array<double,3>> cart_coords_;
  std::vector<double> radii_;
  ArbitraryGrid *agrid_;
};

#endif // BHAHAHA_HORIZON_FINDER_HPP