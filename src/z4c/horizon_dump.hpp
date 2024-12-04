//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_HORIZON_DUMP_HPP_
#define Z4C_HORIZON_DUMP_HPP_

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c_macros.hpp"

// Forward declaration
class Mesh;
class MeshBlockPack;
class ParameterInput;
class CartesianGrid;

//! \class CompactObjectTracker
//! \brief Tracks a single puncture
class HorizonDump {
 public:
  //! Initialize a tracker
  HorizonDump(MeshBlockPack *pmbp, ParameterInput *pin, int n, int common_horizon);
  //! Destructor (will close output file)
  ~HorizonDump();

  // Interpolate field to Cartesian Grid centered at the puncture locations
  void SetGridAndInterpolate(Real center[NDIM]);

  //! Write parameter file for Einstein Toolkit
  void ETK_setup_parfile();

  int horizon_nx;  // number of points in each direction
  int common_horizon; // common horizon or not, triggering when to start dumping data
  int horizon_ind; // indices for horizon
  // TODO(hzhu) : check if this works with rst
  int output_count; // counting the output number (for naming subfolders)

  Real horizon_dt;
  Real horizon_last_output_time;
  Real horizon_extent; // radius for dumping data in a cube
  CartesianGrid *pcat_grid=nullptr; // pointer to cartesian grid
  Real pos[NDIM]; // position of the puncture
  Real r_guess;  // nominal radius of the object (for the AMR driver)

 private:
  MeshBlockPack const *pmbp;
  // first element store variable index second
  // store whether from z4c (true) or adm (false) array
  std::vector<std::pair<int, bool>> variable_to_dump;
};

#endif // Z4C_HORIZON_DUMP_HPP_
