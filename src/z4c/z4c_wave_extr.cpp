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
#include <vector>

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
  
  // Prepare storage for areal radii (only on root) and coordinate radii (all ranks)
  std::vector<Real> areal_radii;
  if (0 == global_variable::my_rank) {
    areal_radii.resize(nradii);
  }
  std::vector<Real> coord_radii(nradii);
  for (int g = 0; g < nradii; ++g) {
    std::string label = grids[g]->Label();
    std::string radius_str = (label.length() > 1) ? label.substr(1) : label;
    coord_radii[g] = std::stod(radius_str);
  }

  int count = 0;
  for (int g=0; g<nradii; ++g) {
    // Rebuild surface geometry if AMR is enabled ---
    // If the mesh has adapted, the old surface geometry (coordinates, interpolation
    // map, etc.) is invalid and must be recomputed.
    if (pmbp->pmesh->adaptive) {
      grids[g]->RebuildAll();
    }

    // --- 1. Interpolate data to the surface ---
    DualArray2D<Real> interpolated_weyl = grids[g]->InterpolateToSurface(u_weyl, 0, 2);
    interpolated_weyl.template sync<HostMemSpace>();
    grids[g]->InterpolateMetric();

    // --- 2. Sync area element and calculate areal radius ---
    auto& proper_area_element = grids[g]->ProperAreaElement();
    proper_area_element.template sync<HostMemSpace>();
    
    // Step 1: Every rank calculates its local contribution to the surface area.
    Real local_total_area = 0.0;
    const int npts_g = grids[g]->Npts();
    auto area_view = proper_area_element.h_view;
    for (int ip = 0; ip < npts_g; ++ip) {
      local_total_area += area_view(ip);
    }

    // Step 2: Reduce the local sums from all ranks into a global sum on the root rank.
    Real global_total_area = 0.0;
    #if MPI_PARALLEL_ENABLED
    MPI_Reduce(&local_total_area, &global_total_area, 1, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    #else
    global_total_area = local_total_area; // Fallback for serial execution
    #endif

    // Step 3: The root rank now has the correct total area and calculates the areal radius.
    if (0 == global_variable::my_rank) {
      areal_radii[g] = sqrt(global_total_area / (4.0 * M_PI));
    }

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

          Real weight = proper_area_element.h_view(ip);

          Real ylmR, ylmI;
          swsh(&ylmR, &ylmI, l, m, theta, phi);

          psilmR += weight * (datareal * ylmR + dataim * ylmI);
          psilmI += weight * (dataim * ylmR - datareal * ylmI);
        }
        // Factor out the coordinate radius and use areal radius instead
        psi_out[count++] = psilmR / SQR(coord_radii[g]);
        psi_out[count++] = psilmI / SQR(coord_radii[g]);
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
      int radius_int = static_cast<int>(coord_radii[g]);
      std::stringstream strObj;
      strObj << std::setfill('0') << std::setw(4) << radius_int;
      
      std::string filename = "waveforms/rpsi4_real_" + strObj.str() + ".txt";
      std::string filename2 = "waveforms/rpsi4_imag_" + strObj.str() + ".txt";

      std::ifstream fileCheck(filename);
      bool fileExists = fileCheck.good();
      fileCheck.close();
      std::ifstream fileCheck2(filename2);
      bool fileExists2 = fileCheck2.good();
      fileCheck2.close();

      // Add Areal Radius to header
      if (!fileExists) {
        std::ofstream createFile(filename);
        std::ofstream outFile(filename, std::ios::out | std::ios::app);
        outFile << "# 1:time\t2:areal_radius\t";
        int a = 3;
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
        outFile << "# 1:time\t2:areal_radius\t";
        int a = 3;
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

      outFile << std::setprecision(15) << pmbp->pmesh->time << "\t" << areal_radii[g] << "\t";
      outFile2 << std::setprecision(15) << pmbp->pmesh->time << "\t" << areal_radii[g] << "\t";

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
