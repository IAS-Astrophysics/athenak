#ifndef CCE_HPP
#define CCE_HPP

#include <string>
#include "athena.hpp"
#include "globals.hpp"

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
    Real *ifield; // interpolated values of the given field
    Real *re_f; // the resultant interpolated field after MPI reduction
    std::string fieldname; // field name that used for pittnull code
    std::string output_dir; // write h5 file in this directory
    std::string bfname; // bookkeeping file name
    bool bitant;          // if bitant symmetry is on, true; otherwise, false.
    Mesh *pm;             // mesh
    ParameterInput *pin;  // param file
    const decomp_decompose::decomp_info **dinfo_pp; // decomposition info
    int num_mu_points;  // number of points in theta direction(polar)
    int num_phi_points; // number of points in phi direction(azimuthal)
    int num_x_points;   // number of points in radius between the two shells
    int num_l_modes;    // number of l modes in -2Ylm (m modes calculated automatically)
    int num_n_modes;    // radial modes
    int nlmmodes;       // num_l_modes*(num_l_modes+2*ABS(MAX_SPIN));
    int nangle;         // num_mu_points*num_phi_points
    int npoint;         // num_mu_points*num_phi_points*num_x_points
    int spin;     // it's 0 and really not used
    int rn;       // radius number/shell number
    int count_interp_pnts; // count the number of interpolated points for a test
    Real *xb; // Cart. x coords. for spherical coords.
    Real *yb; // Cart. y coords. for spherical coords.
    Real *zb; // Cart. z coords. for spherical coords.
    
  public:
    CCE(Mesh *const pm, ParameterInput *const pin, std::string fname, int rn);
    ~CCE();
    void Interpolate(MeshBlockPack *pmbp);
    void ReduceInterpolation();
    void DecomposeAndWrite(int iter, Real curr_time);
};

#endif

} // end namespace z4c
