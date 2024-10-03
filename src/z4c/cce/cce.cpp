//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include "athena.hpp"
#include "globals.hpp"

#if Z4C_CCE_ENABLED

#include <algorithm> // for fill
#include <cmath>     // for NAN
#include <stdexcept>
#include <sstream>
#include <unistd.h> // for F_OK
#include <fstream> 

#define H5_USE_16_API (1)
#include <hdf5.h>


#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "cce.hpp"
#include "z4c/z4c.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "geodesic-grid/gauss_legendre.hpp"

#define BUFFSIZE  (1024)
#define MAX_RADII (100)
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

#define HDF5_CALL(fn_call)                                           \
{                                                                     \
  /* ex: error_code = group_id = H5Gcreate(file_id, metaname, 0) */   \
  hid_t _error_code = fn_call;                                        \
  if (_error_code < 0)                                                \
  {                                                                   \
    cerr << "File: " << __FILE__ << "\n"                              \
         << "line: " << __LINE__ << "\n"                              \
         << "HDF5 call " << #fn_call << ", "                          \
         << "returned error code: " << (int)_error_code << ".\n";     \
         exit((int)_error_code);                                      \
  }                                                                   \
}

namespace z4c {

CCE::CCE(Mesh *const pm, ParameterInput *const pin, int rn):
    pm(pm),
    pin(pin),
    spin(0),
    rn(rn) {
  bitant     = pin->GetOrAddBoolean("z4c", "bitant", false);
  output_dir = pin->GetString("cce","output_dir");
  rin  = pin->GetReal("cce", "rin_"  + std::to_string(rn));
  rout = pin->GetReal("cce", "rout_" + std::to_string(rn));
  ntheta  = pin->GetOrAddInteger("cce","ntheta",41);
  num_x_points   = pin->GetOrAddInteger("cce","num_r_inshell",28);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",7);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);
  nangle = num_mu_points*num_phi_points;
  npoint = nangle*num_x_points;
}

CCE::~CCE()
{
}

// given a Cartesian point, interpolate the pertinent field for that point
void CCE::Interpolate(MeshBlockPack *pmbp)
{
}

// reduce different parts of the interpolation array into one array
void CCE::ReduceInterpolation()
{
}

// decompose the field and write into an h5 file
void CCE::DecomposeAndWrite(int iter/* number of times writes into an h5 file */, Real curr_time)
{
  return 0;
}

} // end namespace z4c

#endif // Z4C_CCE_ENABLED
