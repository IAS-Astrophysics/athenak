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
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"

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
int LmIndex(int l,int m) {
    return l*l+m+l-4;
}
//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(MeshBlockPack *pmbp)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock
void Z4c::WaveExtr(MeshBlockPack *pmbp) {
  // Spherical Grid for user-defined history
  auto &grids = pmbp->pz4c->spherical_grids;
  auto &u_weyl = pmbp->pz4c->u_weyl;
  auto &psi_out = pmbp->pz4c->psi_out;

  // number of radii
  int nradii = grids.size();

  // maximum l; TODO(@hzhu): read in from input file
  int lmax = 8;
  // bool bitant = false;

  Real ylmR,ylmI;
  int count = 0;
  for (int g=0; g<nradii; ++g) {
    // Interpolate Weyl scalars to the surface
    grids[g]->InterpolateToSphere(2, u_weyl);
    for (int l = 2; l < lmax+1; ++l) {
      for (int m = -l; m < l+1 ; ++m) {
        Real psilmR = 0.0;
        Real psilmI = 0.0;
          for (int ip = 0; ip < grids[g]->nangles; ++ip) {
            Real theta = grids[g]->polar_pos.h_view(ip,0);
            Real phi = grids[g]->polar_pos.h_view(ip,1);
            Real datareal = grids[g]->interp_vals.h_view(ip,0);
            Real dataim = grids[g]->interp_vals.h_view(ip,1);
            Real weight = grids[g]->solid_angles.h_view(ip);
            swsh(&ylmR,&ylmI,l,m,theta,phi);
            // The spherical harmonics transform as
            // Y^s_{l m}( Pi-th, ph ) = (-1)^{l+s} Y^s_{l -m}(th, ph)
            // but the PoisitionPolar function returns theta \in [0,\pi],
            // so these are correct for bitant.
            // With bitant, under reflection the imaginary part of
            // the weyl scalar should pick a - sign,
            // which is accounted for here.
            // Real bitant_z_fac = (bitant && theta > M_PI/2) ? -1 : 1;
            psilmR += weight*(datareal*ylmR + dataim*ylmI);
            psilmI += weight*(dataim*ylmR - datareal*ylmI);
          }
        psi_out[count++] = psilmR;
        psi_out[count++] = psilmI;
      }
    }
  }

  // write output
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
      // Output file names
      std::string filename = "waveforms/rpsi4_real_";
      std::string filename2 = "waveforms/rpsi4_imag_";
      std::stringstream strObj;
      strObj << std::setfill('0') << std::setw(4) << grids[g]->radius;
      filename += strObj.str();
      filename += ".txt";
      filename2 += strObj.str();
      filename2 += ".txt";

      // Check if the file already exists
      std::ifstream fileCheck(filename);
      bool fileExists = fileCheck.good();
      fileCheck.close();
      std::ifstream fileCheck2(filename2);
      bool fileExists2 = fileCheck2.good();
      fileCheck2.close();


      // If the file doesn't exist, create it
      if (!fileExists) {
        std::ofstream createFile(filename);
        createFile.close();

        // Open a file stream for writing header
        std::ofstream outFile;
        // append mode
        outFile.open(filename, std::ios::out | std::ios::app);
        // first append time
        outFile << "# 1:time" << "\t";
        // append waveform
        int a = 2;
        for (int l = 2; l < lmax+1; ++l) {
          for (int m = -l; m < l+1 ; ++m) {
            outFile << std::to_string(a)+":"+std::to_string(l)+std::to_string(m) << '\t';
            a++;
          }
        }
        outFile << '\n';

        // Close the file stream
        outFile.close();
      }
      if (!fileExists2) {
        std::ofstream createFile(filename2);
        createFile.close();

        // Open a file stream for writing header
        std::ofstream outFile;
        // append mode
        outFile.open(filename2, std::ios::out | std::ios::app);
        // first append time
        outFile << "# 1:time" << "\t";
        // append waveform
        int a = 2;
        for (int l = 2; l < lmax+1; ++l) {
          for (int m = -l; m < l+1 ; ++m) {
            outFile << std::to_string(a)+":"+std::to_string(l)+std::to_string(m) << '\t';
            a++;
          }
        }
        outFile << '\n';

        // Close the file stream
        outFile.close();
      }
      // Open a file stream for writing header
      std::ofstream outFile;
      std::ofstream outFile2;

      // append mode
      outFile.open(filename, std::ios::out | std::ios::app);
      outFile2.open(filename2, std::ios::out | std::ios::app);

      // first append time
      outFile << pmbp->pmesh->time << "\t";
      outFile2 << pmbp->pmesh->time << "\t";

      // append waveform
      for (int l = 2; l < lmax+1; ++l) {
        for (int m = -l; m < l+1 ; ++m) {
          outFile << std::setprecision(15) << psi_out[idx++] << '\t';
          outFile2 << std::setprecision(15) << psi_out[idx++] << '\t';
        }
      }
      outFile << '\n';
      outFile2 << '\n';

      // Close the file stream
      outFile.close();
      outFile2.close();
    }
  }
}


}  // namespace z4c
