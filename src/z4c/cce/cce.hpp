//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef Z4C_Z4C_CCE_HPP_
#define Z4C_Z4C_CCE_HPP_

#include <string>
#include <vector>

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

class CCE
{
  private:
    int n;       // radius number/shell number
    Real rin;  // inner radius of shell
    Real rout; // outer radius of shell
    std::string output_dir; // write output file in this directory
    Mesh *pm;             // mesh
    ParameterInput *pin;  // param file

    int num_l_modes;    // number of l modes = n_theta (angular)
    int num_n_modes;    // number of n modes = n_radius (radial)
    int nlmmodes;       // total number of coefficients to store

    int ntheta;         // number of collocation points in theta
    int nphi;           // number of collocation points in phi
    int nr;             // number of collocation points in radius
    int nangle;         // number of theta and phi points
    int npoint;         // total number of points

    // sphere for storing the indices, etc.
    std::vector<GaussLegendreGrid> grids;

  public:
    CCE(Mesh *const pm, ParameterInput *const pin, int n);
    ~CCE();
    void SetRadialCoords();
    void Interpolate(MeshBlockPack *pmbp);
    void ReduceInterpolation();
    void DecomposeAndWrite(int iter, Real curr_time);
};

} // end namespace z4c

#endif // Z4C_Z4C_CCE_HPP_
