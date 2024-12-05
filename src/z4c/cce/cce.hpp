//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_CCE_CCE_HPP_
#define Z4C_CCE_CCE_HPP_

#include <string>
#include <vector>
#include <utility>
#include <memory>

#include "athena.hpp"
#include "globals.hpp"

// max absolute value of spin in spin weighted Ylm
#define MAX_SPIN (2)

// Forward Declarations
class Mesh;
class MeshBlock;
class MeshBlockPack;
class ParameterInput;
class GaussLegendreGrid;
namespace decomp_decompose {class decomp_info;};

namespace z4c {

class CCE {
 private:
  int index;       // radius number/shell number
  Real rin;  // inner radius of shell
  Real rout; // outer radius of shell
  Mesh *pm;             // mesh
  MeshBlockPack *pmbp;  // meshblockpack
  ParameterInput *pin;  // param file

  int num_l_modes;    // number of l modes = n_theta (angular)
  int num_angular_modes; // total number of angular modes
  int num_n_modes;    // number of n modes = n_radius (radial)
  int nlmmodes;       // total number of coefficients to store

  int ntheta;         // number of collocation points in theta
  int nphi;           // number of collocation points in phi
  int nr;             // number of collocation points in radius
  int nangle;         // number of theta and phi points
  int npoint;         // total number of points

  // variables to dump
  std::vector<std::pair<int, bool>> variable_to_dump;

  // sphere for storing the indices, etc.
  std::vector<std::unique_ptr<GaussLegendreGrid>> grids;

 public:
  CCE(Mesh *const pm, ParameterInput *const pin, int index);
  ~CCE();
  void InterpolateAndDecompose(MeshBlockPack *pmbp);
};

} // end namespace z4c

#endif // Z4C_CCE_CCE_HPP_
