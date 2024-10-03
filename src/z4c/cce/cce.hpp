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
#include "geodesic-grid/gauss_legendre.hpp"

// max absolute value of spin in spin weighted Ylm
#define MAX_SPIN (2)

class Mesh;
class MeshBlock;
class MeshBlockPack;
class ParameterInput;
namespace decomp_decompose {class decomp_info;};

namespace z4c {

class CCE
{
  private:
    Real rin;  // inner radius of shell
    Real rout; // outer radius of shell
    Real ncycle; // num. of cycle(iter)
    Real *re_f; // the resultant interpolated field after MPI reduction
    std::string fieldname; // field name that used for pittnull code
    std::string output_dir; // write h5 file in this directory
    std::string bfname; // bookkeeping file name
    bool bitant;          // if bitant symmetry is on, true; otherwise, false.
    Mesh *pm;             // mesh
    ParameterInput *pin;  // param file

    int ntheta;  // number of points in theta direction(polar)
    int nl;    // number of l modes = n theta
    int nlmmodes;       // total number of coefficients to store
    int rn;       // radius number/shell number

    // sphere for storing the indices, etc.
    std::vector<GaussLegendreGrid> grids;

  public:
    CCE(Mesh *const pm, ParameterInput *const pin, int rn);
    ~CCE();
    void SetRadialCoords();
    void Interpolate(MeshBlockPack *pmbp);
    void ReduceInterpolation();
    void DecomposeAndWrite(int iter, Real curr_time);
};

} // end namespace z4c

#endif // Z4C_Z4C_CCE_HPP_
