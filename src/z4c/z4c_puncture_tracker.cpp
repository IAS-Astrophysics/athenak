//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file puncture_tracker.cpp
//  \brief implementation of functions in the PunctureTracker classes

#include <cmath>
#include <sstream>
#include <unistd.h>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


#include "athena.hpp"
#include "athena_tensor.hpp"
#include "z4c/z4c_puncture_tracker.hpp"
// #include "globals.hpp"
// #include "coordinates/coordinates.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "utils/lagrange_interpolator.hpp"

#include "z4c.hpp"

//----------------------------------------------------------------------------------------
PunctureTracker::PunctureTracker(MeshBlockPack * pmbp, ParameterInput * pin, int n):
    owns_puncture{false}, pos{NAN, NAN, NAN}, betap{NAN, NAN, NAN}, pmbp{pmbp} {
  ofname = pin->GetString("job", "problem_id") + ".";
  ofname += pin->GetOrAddString("z4c", "filename", "puncture_");
  ofname += std::to_string(n) + ".txt";
  pos[0] = 0.0; // pin->GetOrAddReal("z4c", "bh_" + std::to_string(n) + "_x", 0.0);
  pos[1] = 0.0; // pin->GetOrAddReal("z4c", "bh_" + std::to_string(n) + "_y", 0.0);
  pos[2] = 0.0; // pin->GetOrAddReal("z4c", "bh_" + std::to_string(n) + "_z", 0.0);
  if (0 == Globals::my_rank) {
    // check if output file already exists
    if (access(ofname.c_str(), F_OK) == 0) {
      pofile = fopen(ofname.c_str(), "a");
    }
    else {
      pofile = fopen(ofname.c_str(), "w");
      if (NULL == pofile) {
        std::stringstream msg;
        msg << "### FATAL ERROR in PunctureTracker constructor" << std::endl;
        msg << "Could not open file '" << ofname << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile, "# 1:iter 2:time 3:x 4:y 5:z 6:betax 7:betay 8:betaz\n");
    }
  }
}

//----------------------------------------------------------------------------------------
PunctureTracker::~PunctureTracker() {
  if (0 == Globals::my_rank) {
    fclose(pofile);
  }
}

//----------------------------------------------------------------------------------------
void PunctureTracker::InterpolateShift(MeshBlockPack * pmbp, AthenaTensor<Real, TensorSymm::NONE, 3, 1> & beta_u) {
#pragma omp atomic write
  owns_puncture = true;

  LagrangeInterpolator *S = nullptr;
  S = new LagrangeInterpolator(pmbp,pos);

  for (int a = 0; a < 3; ++a) {
#pragma omp atomic write
    betap[a] = S->Interpolate(beta_u,a);
  }
}


//----------------------------------------------------------------------------------------
void PunctureTracker::EvolveTracker() {
  if (owns_puncture) {
    for (int a = 0; a < 3; ++a) {
      pos[a] -= pmbp->pmesh->dt * betap[a];
    }
  }
#ifndef MPI_PARALLEL
  else {
    std::stringstream msg;
    msg << "### FATAL ERROR in PunctureTracker::EvolveTracker" << std::endl;
    msg << "The puncture is MIA" << std::endl;
    exit(EXIT_FAILURE);
  }
#else
  Real buf[2*NDIM + 1] = {0., 0., 0., 0., 0., 0., 0.};
  if (owns_puncture) {
    buf[0] = pos[0];
    buf[1] = pos[1];
    buf[2] = pos[2];
    buf[3] = betap[0];
    buf[4] = betap[1];
    buf[5] = betap[2];
    buf[6] = 1.0;
  }
  MPI_Allreduce(MPI_IN_PLACE, buf, 2*NDIM + 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (buf[6] < 1.) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PunctureTracker::EvolveTracker" << std::endl;
    msg << "The puncture has left the grid" << std::endl;
    ATHENA_ERROR(msg);
  }
  pos[0] = buf[0]/buf[6];
  pos[1] = buf[1]/buf[6];
  pos[2] = buf[2]/buf[6];
  betap[0] = buf[3]/buf[6];
  betap[1] = buf[4]/buf[6];
  betap[2] = buf[5]/buf[6];
#endif // MPI_PARALLEL

  // After the puncture has moved it might have changed ownership
  owns_puncture = false;
}

//----------------------------------------------------------------------------------------
void PunctureTracker::WriteTracker(int iter, Real time) const {
  // if (0 == Globals::my_rank) {
    fprintf(pofile, "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
        iter, time, pos[0], pos[1], pos[2], betap[0], betap[1], betap[2]);
  // }
}
