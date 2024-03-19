//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <assert.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "z4c/z4c.hpp"
#include "z4c_puncture_tracker.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
PunctureTracker::PunctureTracker(Mesh *pmesh, ParameterInput *pin, int n):
              owns_puncture{false}, pos{NAN, NAN, NAN}, betap{NAN, NAN, NAN},
              pmesh{pmesh} {
  ofname = pin->GetString("job", "basename") + ".";
  ofname += pin->GetOrAddString("z4c", "filename", "puncture_");
  ofname += std::to_string(n) + ".txt";

  pos[0] = pin->GetOrAddReal("z4c", "bh_" + std::to_string(n) + "_x", 0.0);
  pos[1] = pin->GetOrAddReal("z4c", "bh_" + std::to_string(n) + "_y", 0.0);
  pos[2] = pin->GetOrAddReal("z4c", "bh_" + std::to_string(n) + "_z", 0.0);
  bitant = pin->GetOrAddBoolean("z4c", "bitant", false);
  if (0 == global_variable::my_rank) {
    // check if output file already exists
    if (access(ofname.c_str(), F_OK) == 0) {
      pofile = fopen(ofname.c_str(), "a");
      if (NULL == pofile) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                  << __LINE__ << std::endl;
        std::cout << "Could not open file '" << ofname << "' for writing!"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // read the last iteration and position (used after a restart)
      std::ifstream file(ofname);
      std::string line;
      std::istringstream lastline;
      Real time{0};
      if (!file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Read and process each line until the end of the file is reached
      while (std::getline(file, line)) {
        if (line.length() > 1)
          lastline.str(line);
      }
      file.close();

      // get iter, pos, beta
      lastline >> time >> pos[0] >> pos[1] >> pos[2] >> betap[0] >> betap[1]
        >> betap[2];
    } else {
      pofile = fopen(ofname.c_str(), "w");
      if (NULL == pofile) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                  << __LINE__ << std::endl;
        std::cout << "Could not open file '" << ofname << "' for writing!"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      fprintf(pofile, "# 1:time 2:x 3:y 4:z 5:betax 6:betay 7:betaz\n");
      fflush(pofile);
    }
  }
}

//----------------------------------------------------------------------------------------
PunctureTracker::~PunctureTracker() {
  if (0 == global_variable::my_rank) {
    fclose(pofile);
  }
}

//----------------------------------------------------------------------------------------
void PunctureTracker::InterpolateShift(MeshBlockPack *pmbp) {
  auto &pz4c = pmbp->pz4c;
  auto &u0   = pz4c->u0;
  auto *S    = new LagrangeInterpolator(pmbp, pos);

  if (S->point_exist) {
    betap[0]      = S->Interpolate(u0, pz4c->I_Z4C_BETAX);
    betap[1]      = S->Interpolate(u0, pz4c->I_Z4C_BETAY);
    betap[2]      = S->Interpolate(u0, pz4c->I_Z4C_BETAZ);
    owns_puncture = true;
  }

  delete S;
}

//----------------------------------------------------------------------------------------
void PunctureTracker::EvolveTracker() {
  if (owns_puncture) {
    for (int a = 0; a < NDIM; ++a) {
      pos[a] -= pmesh->dt * betap[a];
    }
    // Impose the motion on the z = 0 plane with bitant.
    if (bitant)
      pos[2] = 0;
#if !(MPI_PARALLEL_ENABLED)
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "couldn't find the puncture!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#else
  }
  Real buf[2 * NDIM + 1] = {0., 0., 0., 0., 0., 0., 0.};
  if (owns_puncture) {
    buf[0] = pos[0];
    buf[1] = pos[1];
    buf[2] = pos[2];
    buf[3] = betap[0];
    buf[4] = betap[1];
    buf[5] = betap[2];
    buf[6] = 1.0;
  }
  MPI_Allreduce(
    MPI_IN_PLACE, buf, 2 * NDIM + 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (buf[6] < 1.) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "The puncture has left the grid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pos[0]   = buf[0] / buf[6];
  pos[1]   = buf[1] / buf[6];
  pos[2]   = buf[2] / buf[6];
  betap[0] = buf[3] / buf[6];
  betap[1] = buf[4] / buf[6];
  betap[2] = buf[5] / buf[6];
#endif // MPI_PARALLEL_ENABLED

  // After the puncture has moved it might have changed ownership
  owns_puncture = false;
}

//----------------------------------------------------------------------------------------
void PunctureTracker::WriteTracker() const {
  if (0 == global_variable::my_rank) {
    fprintf(
      pofile, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n", pmesh->time,
      pos[0], pos[1], pos[2], betap[0], betap[1], betap[2]);
    fflush(pofile);
  }
}
} // namespace z4c
