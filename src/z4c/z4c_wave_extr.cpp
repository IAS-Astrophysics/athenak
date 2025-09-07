//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class to interpolate weyl scalar
//  and output the waveform

// C++ standard headers
#include <unistd.h>
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "globals.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "utils/surface_grid.hpp"

namespace z4c {

//Factorial
Real fac(Real n) {
  if(n==0 || n==1) {
    return 1.0;
  } else {
    n=n*fac(n-1);
    return n;
  }
}

// Calculate spin weighted spherical harmonics sw=-2 using Wigner-d matrix notation
// see e.g. Eq II.7, II.8 in 0709.0093
void swsh(Real * ylmR, Real * ylmI, int l, int m, Real theta, Real phi) {
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-2);
  k2 = std::min(l+m,l-2);
  for (k = k1; k<k2+1; ++k) {
    wignerd += pow((-1),k)*sqrt(fac(l+m)*fac(l-m)*fac(l+2)*fac(l-2))
      *pow(std::cos(theta/2.0),2*l+m-2-2*k)*pow(std::sin(theta/2.0),2*k+2-m)
      /(fac(l+m-k)*fac(l-2-k)*fac(k)*fac(k+2-m));
  }
  *ylmR = sqrt((2*l+1)/(4*M_PI))*wignerd*std::cos(m*phi);
  *ylmI = sqrt((2*l+1)/(4*M_PI))*wignerd*std::sin(m*phi);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::WaveExtr(MeshBlockPack *pmbp)
// \brief Computes Weyl scalar multipoles and outputs the waveform.
//
// This function interpolates the Weyl scalar Psi4 onto spherical surfaces, then
// performs a spherical harmonic decomposition using the proper relativistic surface
// area element derived from the interpolated spatial metric.
void Z4c::WaveExtr(MeshBlockPack *pmbp) {
  // --- Setup ---
  auto &grids = pmbp->pz4c->spherical_grids;
  auto &u_weyl = pmbp->pz4c->u_weyl;
  auto &psi_out = pmbp->pz4c->psi_out;

  int nradii = grids.size();
  int lmax = 8;

  int count = 0;
  for (int g=0; g<nradii; ++g) {
    // --- 1. Interpolate data to the surface ---
    // Interpolate Weyl scalars to a temporary buffer
    DualArray2D<Real> interpolated_weyl = grids[g]->InterpolateToSurface(u_weyl, 0, 2);
    // Interpolate the metric, which now AUTOMATICALLY calculates the proper area element
    grids[g]->InterpolateMetric();

    // --- 2. Sync the pre-calculated area element to the host for the integration loop ---
    auto& proper_area_element = grids[g]->ProperAreaElement();
    proper_area_element.template sync<HostMemSpace>();

    // --- 3. Perform spherical harmonic decomposition ---
    const int npts = grids[g]->Npts();
    for (int l = 2; l < lmax+1; ++l) {
      for (int m = -l; m < l+1; ++m) {
        Real psilmR = 0.0;
        Real psilmI = 0.0;
        for (int ip = 0; ip < npts; ++ip) {
          Real theta = grids[g]->Thetas().h_view(ip);
          Real phi = grids[g]->Phis().h_view(ip);
          Real datareal = interpolated_weyl.h_view(ip,0);
          Real dataim = interpolated_weyl.h_view(ip,1);

          // Get the pre-calculated proper area element
          Real weight = proper_area_element.h_view(ip);

          Real ylmR, ylmI;
          swsh(&ylmR, &ylmI, l, m, theta, phi);

          psilmR += weight * (datareal * ylmR + dataim * ylmI);
          psilmI += weight * (dataim * ylmR - datareal * ylmI);
        }
        psi_out[count++] = psilmR;
        psi_out[count++] = psilmI;
      }
    }
  }

  // --- 4. MPI Reduction and File Output ---
  #if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, psi_out, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(psi_out, psi_out, count, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  #endif

  if (0 == global_variable::my_rank) {
    int idx = 0;
    for (int g=0; g<nradii; ++g) {
      // Get the label (e.g., "R100") from the surface grid
      std::string label = grids[g]->Label();

      // Extract the radius part of the string (e.g., "100" from "R100")
      // This assumes the name format is "R<radius>"
      std::string radius_str = (label.length() > 1) ? label.substr(1) : label;
      
      // Format the radius string for the filename (e.g., zero-padding)
      std::stringstream strObj;
      strObj << std::setfill('0') << std::setw(4) << radius_str;
      
      std::string filename = "waveforms/rpsi4_real_" + strObj.str() + ".txt";
      std::string filename2 = "waveforms/rpsi4_imag_" + strObj.str() + ".txt";

      std::ifstream fileCheck(filename);
      bool fileExists = fileCheck.good();
      fileCheck.close();
      std::ifstream fileCheck2(filename2);
      bool fileExists2 = fileCheck2.good();
      fileCheck2.close();

      if (!fileExists) {
        std::ofstream createFile(filename);
        std::ofstream outFile(filename, std::ios::out | std::ios::app);
        outFile << "# 1:time" << "\t";
        int a = 2;
        for (int l = 2; l < lmax+1; ++l) {
          for (int m = -l; m < l+1; ++m) {
            outFile << std::to_string(a++) + ":" + std::to_string(l) + std::to_string(m) << '\t';
          }
        }
        outFile << '\n';
        outFile.close();
      }
      if (!fileExists2) {
        std::ofstream createFile(filename2);
        std::ofstream outFile(filename2, std::ios::out | std::ios::app);
        outFile << "# 1:time" << "\t";
        int a = 2;
        for (int l = 2; l < lmax+1; ++l) {
          for (int m = -l; m < l+1; ++m) {
            outFile << std::to_string(a++) + ":" + std::to_string(l) + std::to_string(m) << '\t';
          }
        }
        outFile << '\n';
        outFile.close();
      }

      std::ofstream outFile(filename, std::ios::out | std::ios::app);
      std::ofstream outFile2(filename2, std::ios::out | std::ios::app);

      outFile << pmbp->pmesh->time << "\t";
      outFile2 << pmbp->pmesh->time << "\t";

      for (int l = 2; l < lmax+1; ++l) {
        for (int m = -l; m < l+1; ++m) {
          outFile << std::setprecision(15) << psi_out[idx++] << '\t';
          outFile2 << std::setprecision(15) << psi_out[idx++] << '\t';
        }
      }
      outFile << '\n';
      outFile2 << '\n';

      outFile.close();
      outFile2.close();
    }
  }
}

} // namespace z4c