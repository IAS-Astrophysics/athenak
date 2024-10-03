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
#include "coordinates/adm.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "utils/spherical_harm.hpp"
#include "utils/write_vector.hpp"
#include "utils/chebyshev.hpp"

#define BUFFSIZE  (1024)
#define MAX_RADII (100)
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

namespace z4c {

CCE::CCE(Mesh *const pm, ParameterInput *const pin, int index):
  pm(pm),
  pin(pin),
  index(index) {
  // pointer to meshblockpack
  pmbp = pm->pmb_pack;

  output_dir = pin->GetString("cce","output_dir");
  rin  = pin->GetOrAddReal("cce", "rin_"  + std::to_string(index),20.);
  rout = pin->GetOrAddReal("cce", "rout_" + std::to_string(index),40.);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",7);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);
  num_angular_modes = (num_l_modes + 1) * (num_l_modes + 1);

  ntheta = num_l_modes;
  nphi   = 2*num_l_modes;
  nr     = num_n_modes;
  nangle = ntheta*nphi;
  npoint = nangle*nr;

  // set appropriate size for the coefficients for all fields
  // 10 fields we need to interpolate
  cnlm_real.resize(10);
  cnlm_imag.resize(10);
  data_real.resize(10);
  data_imag.resize(10);
  for (int i = 0; i < 10; ++i) {
    cnlm_real[i].resize(nr);
    cnlm_imag[i].resize(nr);
    data_real[i].resize(nr);
    data_imag[i].resize(nr);
    for (int j = 0; j < nr; ++j) {
      cnlm_real[i][j].resize(num_angular_modes);
      cnlm_imag[i][j].resize(num_angular_modes);
      data_real[i][j].resize(num_angular_modes);
      data_imag[i][j].resize(num_angular_modes);
    }
  }

  // Calculate radius for the Gauss Legendre Spheres 
  // and initialize them in a vector
  for (int k = 0; k < nr; ++k) {
    Real x_k = std::cos( M_PI * (k + 1) / (nr + 2) );
    Real rad = 0.5 * ( (rout - rin) * x_k + (rout + rin) );
    grids.push_back(std::make_unique<GaussLegendreGrid>(pmbp, ntheta, rad));
  }
}

CCE::~CCE() {}

// Interpolate all fields to Gauss-Legendre Sphere
void CCE::InterpolateAndDecompose(MeshBlockPack *pmbp) {
  auto &u_adm = pmbp->padm->u_adm;
  // outer loop over the number of variables
  // inner loop over the number of spheres
  Real ylmR,ylmI;

  Real dat_real[10][nr][num_angular_modes];
  Real dat_imag[10][nr][num_angular_modes];

  // counts for mpi
  int count = 10*nr*num_angular_modes;
  for (int nvar = 0; nvar < 10; ++nvar) {
    for (int k = 0; k < nr; ++k) {
      // call interpolation routine here
      // TODO: replace nvar with whatever field I need to interpolate
      grids[k]->InterpolateToSphere(nvar, u_adm);
      // Decompose into Spherical Harmonics and store value into cnlm[nvar,k,lm]
      for (int l = 1; l < num_l_modes+1; ++l) {
        for (int m = -l; m < l+1 ; ++m) {
          Real psilmR = 0.0;
          Real psilmI = 0.0;
          for (int ip = 0; ip < grids[k]->nangles; ++ip) {
            Real theta = grids[k]->polar_pos.h_view(ip,0);
            Real phi = grids[k]->polar_pos.h_view(ip,1);
            Real data = grids[k]->interp_vals.h_view(ip);
            Real weight = grids[k]->int_weights.h_view(ip);
            // calculate spherical harmonics
            SWSphericalHarm(&ylmR,&ylmI, l, m, 0, theta, phi);
            psilmR += weight*data*ylmR;
            psilmI += weight*data*ylmI;
          }
          dat_real[nvar][k][l*l+l+m]=psilmR;
          dat_imag[nvar][k][l*l+l+m]=psilmI;
        }
      }
    }
  }
  // Reduction to the master rank for cnlm_real and cnlm_imag
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, dat_real, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, dat_imag, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(dat_real, dat_real, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(dat_imag, dat_imag, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  #endif

  // converting static array to std::vector
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < nr; ++j) {
      for (int k = 0; k < num_angular_modes; ++k) {
        data_real[i][j][k] = dat_real[i][j][k];
        data_imag[i][j][k] = dat_imag[i][j][k];
      }
    }
  }

  // Convert to Chebyshev coefficients
  ChebyshevDecomposition(data_real,cnlm_real);
  ChebyshevDecomposition(data_imag,cnlm_imag);

  // output here
  write3DVectorToFile(cnlm_real,"real.bin");
  write3DVectorToFile(cnlm_imag,"imag.bin");
}

void CCE::ChebyshevDecomposition(std::vector<std::vector<std::vector<Real>>> data,
                            std::vector<std::vector<std::vector<Real>>> cnlm) {
  for (int nvar = 0; nvar < 10; ++nvar) {
    for (int l = 0; l < num_angular_modes; ++l) {
      double factor = 2.0 / (nr + 2);
      for (int k = 0; k <= nr; ++k) {
        double sum = 0.0;
        for (int i = 0; i <= nr; ++i) {
            // Compute U_k(x_i)
            double Uk_xi = chebyshevSecondKindPolynomial(k, std::cos(M_PI * (i + 1) / (nr + 2)));
            sum += data[nvar][i][l] * Uk_xi;
        }
        cnlm[nvar][k][l] = factor * sum;
      }
    }
  }
}

} // end namespace z4c
