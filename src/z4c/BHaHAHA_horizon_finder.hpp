// BHaHAHA_horizon_finder.hpp
#ifndef BHAHAHA_HORIZON_FINDER_HPP
#define BHAHAHA_HORIZON_FINDER_HPP

#include <memory>
#include <vector>
#include <array>
#include <sys/time.h>
#include "z4c/bhahaha/BHaHAHA.h"

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

  // Persistence: set the search center from the trackers if no horizon history exists,
  // and keep the BHaHAHA history in params_data_ identical on all ranks
  void readPersistentData(int h);
  void broadcastHorizonState(int h);
  int rootRank(int h) const;
  void resetHorizonHistory(int h);

  // BBH mode handling
  void processBBHMode();

  // Diagnostics
  void diagnosticPrintPreInterpolation(double start_time);
  double timevalToSeconds(const timeval &start, const timeval &end);

  // Grid & interpolation
  void SetGridCoordinates(int h, int Nr);
  void InterpolateMetricData(int h);
  void gatherMetricData(int h, size_t pts);

  // Core solver & cleanup
  void SolveHorizon(int h);

  // AthenaK handles
  MeshBlockPack *pmbp_;
  ParameterInput *pin_;

  // User-configurable parameters
  int find_every_;
  double dt_find_;
  double last_find_time_;
  int output_shape_every_;
  int interp_half_width_;
  int verbosity_;
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

  // Active flags
  std::vector<int> bah_horizon_active_;

  // Storage for the horizon shapes of the previous three finds. The time, center and
  // radius history lives in params_data_ and is cycled by BHaHAHA itself.
  std::vector<std::vector<double>> prev_horizon_m1_, prev_horizon_m2_, prev_horizon_m3_;

  // Center of the spherical interpolation grid used for the latest find
  std::vector<std::array<double,3>> grid_center_;
  // guess for the mass of black hole
  std::vector<double> m_guess;

  std::vector<int> nfinds_;
  std::vector<std::vector<double>> input_buf_;
  std::vector<double> time_interp_, time_mpi_, time_solve_;

  // Per-horizon parameters and data
  std::vector<bhahaha_params_and_data_struct> params_data_;

  // Interpolation buffers
  std::vector<std::array<double,3>> cart_coords_;
  std::vector<double> radii_;
  std::vector<double> sendbuf_, recvbuf_;
  std::vector<int> recvcounts_, recvdispls_;
  std::unique_ptr<ArbitraryGrid> agrid_;
};

#endif // BHAHAHA_HORIZON_FINDER_HPP