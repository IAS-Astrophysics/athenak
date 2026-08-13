//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driftcontrol/driftcontrol.cpp
//! \brief Implementation of the DriftControl class.

#include <assert.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "driftcontrol.hpp"

#include "globals.hpp"
#include "parameter_input.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_macros.hpp"

//----------------------------------------------------------------------------------------
DriftControl::Variety DriftControl::VarietyFromString(std::string const &str) {
  if (str.compare("oscillator") == 0) {
    return Oscillator;
  } else if (str.compare("pid") == 0) {
    return PID;
  } else if (str.compare("relaxation") == 0) {
    return Relaxation;
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
            << __LINE__ << std::endl;
  std::cout << "Unknown DriftControl variety: " << str << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
DriftControl::Center DriftControl::CenterFromString(std::string const &str) {
  if (str.compare("fixed") == 0) {
    return CenterFixed;
  } else if (str.compare("tracker") == 0) {
    return CenterTracker;
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
            << __LINE__ << std::endl;
  std::cout << "Unknown DriftControl gaussian center: " << str << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
DriftControl::DriftControl(Mesh *pmesh, ParameterInput *pin) :
  pmesh(pmesh) {
  // Initialize drift control
  dc_variety = VarietyFromString(
      pin->GetOrAddString("z4c", "dc_variety", "oscillator"));

  dc_tracker_index = pin->GetOrAddInteger("z4c", "dc_tracker_index", 0);
  dc_first_step    = true;
  dc_vel_cap       = pin->GetOrAddReal("z4c", "dc_vel_cap", 1.0);
  dc_integral_cap  = pin->GetOrAddReal("z4c", "dc_integral_cap", 5.0);
  out_every        = pin->GetOrAddInteger("z4c", "dc_out_every", 1);

  dc_fixed[0]      = pin->GetOrAddReal("z4c", "dc_fixed_x", 0.0);
  dc_fixed[1]      = pin->GetOrAddReal("z4c", "dc_fixed_y", 0.0);
  dc_fixed[2]      = pin->GetOrAddReal("z4c", "dc_fixed_z", 0.0);

  for (int a = 0; a < NDIM; ++a) {
    dc_pos[a]        = dc_fixed[a];
    dc_pos_old[a]    = dc_fixed[a];
    dc_vel[a]        = 0.0;
    dc_integral[a]   = 0.0;
    dc_prev_error[a] = 0.0;
  }

  // Register on every rank. ParameterDump() feeds the header offset written by
  // the binary output, while each rank places its MeshBlocks using its own dump
  // length, so a parameter held by rank 0 alone corrupts the header/data
  // boundary of every .bin file.
  std::string dc_file = pin->GetOrAddString("z4c", "dc_filename", "drift_control");

  if (0 == global_variable::my_rank) {
    std::string ofname = pin->GetString("job", "basename") + ".";
    ofname += dc_file;
    ofname += ".txt";
    ofile.open(ofname.c_str());
    ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz 9:ix 10:iy 11:iz\n";
    ofile << std::flush;
    ofile << std::setprecision(19);
  }
}

//----------------------------------------------------------------------------------------
DriftControl::~DriftControl() { }

//----------------------------------------------------------------------------------------
void DriftControl::EvolveDriftControl() {
  int const idx = dc_tracker_index;
  auto &ptracker = pmesh->pmb_pack->pz4c->ptracker;

  if (idx < 0 || static_cast<std::size_t>(idx) >= ptracker.size()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "dc_tracker_index " << idx << " is out of range for "
              << ptracker.size() << " trackers." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  CompactObjectTracker *pt = ptracker[idx].get();

  for (int a = 0; a < NDIM; ++a) {
    dc_pos_old[a] = dc_pos[a];
    dc_pos[a]     = pt->GetPos(a);
  }

  Real const dt = pmesh->dt;

  if (dc_first_step || dt <= 0.0) {
    for (int a = 0; a < NDIM; ++a) {
      dc_vel[a]        = 0.0;
      dc_integral[a]   = 0.0;
      dc_prev_error[a] = dc_pos[a] - dc_fixed[a];
    }
    dc_first_step = false;
  } else if (dc_variety == PID) {
    for (int a = 0; a < NDIM; ++a) {
      Real const e = dc_pos[a] - dc_fixed[a];
      dc_integral[a] += e * dt;

      // Clamp dc_integral/dc_vel to a limited range given by the
      // cap. If out this range, return the minimum.
      dc_integral[a] = std::clamp(dc_integral[a],
                                  -dc_integral_cap, dc_integral_cap);

      Real const vel_raw = (e - dc_prev_error[a]) / dt;
      dc_vel[a]          = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
      dc_prev_error[a]   = e;
    }
  } else {
    for (int a = 0; a < NDIM; ++a) {
      Real const vel_raw = (dc_pos[a] - dc_pos_old[a]) / dt;
      dc_vel[a] = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
    }
  }
}

//----------------------------------------------------------------------------------------
void DriftControl::WriteDriftControl() {
  if (0 == global_variable::my_rank && 0 == pmesh->ncycle % out_every) {
    ofile << pmesh->ncycle << " "
          << pmesh->time << " "
          << dc_pos[0] << " "
          << dc_pos[1] << " "
          << dc_pos[2] << " "
          << dc_vel[0] << " "
          << dc_vel[1] << " "
          << dc_vel[2] << " "
          << dc_integral[0] << " "
          << dc_integral[1] << " "
          << dc_integral[2] << std::endl << std::flush;
  }
}
