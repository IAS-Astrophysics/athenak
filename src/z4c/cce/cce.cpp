//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <unistd.h> // for F_OK
#include <algorithm> // for fill
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <utility>
#include <string>
#include <cstdio>

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

  rin  = pin->GetOrAddReal("cce", "rin_"  + std::to_string(index),20.);
  rout = pin->GetOrAddReal("cce", "rout_" + std::to_string(index),40.);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",16);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);
  num_angular_modes = (num_l_modes + 1) * (num_l_modes + 1);

  ntheta = num_l_modes + 1;
  nphi   = 2*num_l_modes;
  nr     = num_n_modes;
  nangle = ntheta*nphi;
  npoint = nangle*nr;

  // Calculate radius for the Gauss Legendre Spheres
  // and initialize them in a vector
  for (int k = 0; k < nr; ++k) {
    Real rad = ChebyshevSecondKindCollocationPoints(rin,rout,nr,k);
    grids.push_back(std::make_unique<GaussLegendreGrid>(pmbp, ntheta, rad));
  }

  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_ALPHA, true));
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_BETAX, true));
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_BETAY, true));
  variable_to_dump.push_back(std::make_pair(pmbp->pz4c->I_Z4C_BETAZ, true));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GXX, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GXY, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GXZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GYY, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GYZ, false));
  variable_to_dump.push_back(std::make_pair(pmbp->padm->I_ADM_GZZ, false));
}

CCE::~CCE() {}

// Interpolate all fields to Gauss-Legendre Sphere
void CCE::InterpolateAndDecompose(MeshBlockPack *pmbp) {
  Real ylmR,ylmI;

  // reinitialize interpolation indices and weights if AMR
  if(pmbp->pmesh->adaptive) {
    for (int k = 0; k < nr; ++k) {
      grids[k]->SetInterpolationIndices();
      grids[k]->SetInterpolationWeights();
    }
  }

  // raveled shape of array & counts for mpi
  int count = 10*nr*num_angular_modes;
  // Dynamically allocate memory for the 4D array flattened into 1D
  Real* data_real = new Real[count];
  Real* data_imag = new Real[count];
  for(int nvar=0; nvar<10; nvar++) {
    for (int k = 0; k < nr; ++k) {
      // Interpolate here
      if (variable_to_dump[nvar].second) {
        grids[k]->InterpolateToSphere(variable_to_dump[nvar].first,pmbp->pz4c->u0);
      } else {
        grids[k]->InterpolateToSphere(variable_to_dump[nvar].first,pmbp->padm->u_adm);
      }
      for (int l = 0; l < num_l_modes+1; ++l) {
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
          data_real[k * 10 * num_angular_modes // first over the different radii
                    + nvar * num_angular_modes // then over the variables
                    + l*l+l+m]                 // lastly over the angular harmonic index
                    = psilmR;
          data_imag[k * 10 * num_angular_modes // first over the different radii
                    + nvar * num_angular_modes // then over the variables
                    + l*l+l+m]                 // lastly over the angular harmonic index
                    = psilmI;
        }
      }
    }
  }

  // Reduction to the master rank for cnlm_real and cnlm_imag
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, data_real, count, MPI_ATHENA_REAL,
              MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, data_imag, count, MPI_ATHENA_REAL,
              MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(data_real, data_real, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(data_imag, data_imag, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  #endif

  // Then write output file
  // Open the file in binary write mode

  if (0 == global_variable::my_rank) {
    std::string filename = "cce/cce_";
    std::stringstream strObj;
    strObj << std::setfill('0') << std::setw(8) << pmbp->pmesh->time;
    filename += strObj.str();
    filename += ".bin";

    FILE* cce_file = fopen(filename.c_str(), "wb");
    if (cce_file == nullptr) {
      perror("Error opening file");
      return;
    }
    // write number of radius and angular modes for reshaping data
    fwrite(&nr, sizeof(int), 1, cce_file);
    fwrite(&num_l_modes, sizeof(int), 1, cce_file);
    // write time
    fwrite(&pmbp->pmesh->time, sizeof(Real), 1, cce_file);
    // write inner and outer radial boundary
    fwrite(&rin, sizeof(Real), 1, cce_file);
    fwrite(&rout, sizeof(Real), 1, cce_file);
    // Write the 4D array to the binary file
    size_t elementsWritten = fwrite(data_real, sizeof(Real), count, cce_file);
    if (elementsWritten != count) {
      perror("Error writing to file");
    }
    elementsWritten = fwrite(data_imag, sizeof(Real), count, cce_file);
    if (elementsWritten != count) {
      perror("Error writing to file");
    }
    // Close the file
    fclose(cce_file);
  }
  delete[] data_real;
  delete[] data_imag;
}
} // end namespace z4c
