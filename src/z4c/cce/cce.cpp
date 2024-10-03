//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include "athena.hpp"
#include "globals.hpp"

#include <algorithm> // for fill
#include <cmath>     // for NAN
#include <stdexcept>
#include <sstream>
#include <unistd.h> // for F_OK
#include <fstream> 


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


namespace z4c {

CCE::CCE(Mesh *const pm, ParameterInput *const pin, int n):
  pm(pm),
  pin(pin),
  n(n) {
  output_dir = pin->GetString("cce","output_dir");
  rin  = pin->GetOrAddReal("cce", "rin_"  + std::to_string(n),20.);
  rout = pin->GetOrAddReal("cce", "rout_" + std::to_string(n),40.);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",7);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);

  ntheta = num_l_modes;
  nphi   = 2*num_l_modes;
  nr     = num_n_modes;
  nangle = ntheta*nphi;
  npoint = nangle*nr;
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
}

} // end namespace z4c
