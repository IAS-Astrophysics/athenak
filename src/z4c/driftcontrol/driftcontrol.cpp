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

namespace {
// Parameter keys under which the PID integral is checkpointed.
char const *const integral_key[3] = {"dc_integral_x", "dc_integral_y", "dc_integral_z"};
// Parameter keys under which the DOB observer state is checkpointed.
char const *const dob_key[3] = {"dc_dob_px", "dc_dob_py", "dc_dob_pz"};
} // namespace

//----------------------------------------------------------------------------------------
DriftControl::Variety DriftControl::VarietyFromString(std::string const &str) {
  if (str.compare("oscillator") == 0) {
    return Oscillator;
  } else if (str.compare("pid") == 0) {
    return PID;
  } else if (str.compare("relaxation") == 0) {
    return Relaxation;
  } else if (str.compare("dob") == 0) {
    return DOB;
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
  pmesh(pmesh), pin(pin) {
  // Initialize drift control
  dc_variety = VarietyFromString(
      pin->GetOrAddString("z4c", "dc_variety", "oscillator"));

  dc_tracker_index = pin->GetOrAddInteger("z4c", "dc_tracker_index", 0);
  dc_first_step    = true;
  dc_dt_warned     = false;
  dc_vel_cap       = pin->GetOrAddReal("z4c", "dc_vel_cap", 1.0);
  dc_integral_cap  = pin->GetOrAddReal("z4c", "dc_integral_cap", 5.0);
  out_every        = pin->GetOrAddInteger("z4c", "dc_out_every", 1);

  // Ramp-down. Defaults leave the controller running forever, i.e. behaviour is
  // unchanged for every parfile that does not set these.
  dc_ramp_start    = pin->GetOrAddReal("z4c", "dc_ramp_start", -1.0);
  dc_ramp_time     = pin->GetOrAddReal("z4c", "dc_ramp_time", 0.0);
  dc_ramp_begun    = false;
  dc_ramp_done     = false;

  dc_fixed[0]      = pin->GetOrAddReal("z4c", "dc_fixed_x", 0.0);
  dc_fixed[1]      = pin->GetOrAddReal("z4c", "dc_fixed_y", 0.0);
  dc_fixed[2]      = pin->GetOrAddReal("z4c", "dc_fixed_z", 0.0);

  // DOB gains.
  dc_omega_c       = pin->GetOrAddReal("z4c", "dc_omega_c", 0.2);
  dc_omega_o       = pin->GetOrAddReal("z4c", "dc_omega_o", 1.0);
  dc_zeta          = pin->GetOrAddReal("z4c", "dc_zeta", 1.0);

  // The PID integral holds the controller's entire steady-state authority;
  // restore it across restarts
  for (int a = 0; a < NDIM; ++a) {
    dc_pos[a]        = dc_fixed[a];
    dc_pos_old[a]    = dc_fixed[a];
    dc_vel[a]        = 0.0;
    dc_integral[a]   = pin->GetOrAddReal("z4c", integral_key[a], 0.0);
    dc_prev_error[a] = 0.0;
    dc_p[a]          = pin->GetOrAddReal("z4c", dob_key[a], 0.0);
    dc_fhat[a]       = dc_p[a];  // v is suppressed on the first step
  }

  std::string const dc_fname =
      pin->GetOrAddString("z4c", "dc_filename", "drift_control");

  if (0 == global_variable::my_rank) {
    std::string ofname = pin->GetString("job", "basename") + ".";
    ofname += dc_fname;
    ofname += ".txt";
    ofile.open(ofname.c_str());
    if (dc_variety == DOB) {
      ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz 9:px 10:py 11:pz"
               " 12:fhatx 13:fhaty 14:fhatz\n";
    } else {
      ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz 9:ix 10:iy 11:iz\n";
    }
    ofile << std::flush;
    ofile << std::setprecision(19);
  }
}

//----------------------------------------------------------------------------------------
DriftControl::~DriftControl() { }

//----------------------------------------------------------------------------------------
//! \brief Smooth 1 -> 0 taper on the applied correction. See the header for why an
//! abrupt cut is not an option.
Real DriftControl::RampFactor(Real time) const {
  if (dc_ramp_start < 0.0) {
    return 1.0;                       // ramp disabled: unchanged behaviour
  }
  if (time <= dc_ramp_start) {
    return 1.0;
  }
  if (dc_ramp_time <= 0.0 || time >= dc_ramp_start + dc_ramp_time) {
    return 0.0;
  }
  // Raised cosine: C^1 at both ends, so neither the force nor its derivative jumps.
  Real const s = (time - dc_ramp_start) / dc_ramp_time;
  return 0.5 * (1.0 + std::cos(M_PI * s));
}

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

  // Once the ramp has begun the loop is being opened on purpose: the RHS applies only
  // w*u while e and v grow, so letting the observer/integrator keep accumulating would
  // wind the state up against the ramp and partly cancel it. Freeze the state instead;
  // the applied force is then w times a fixed quantity and goes to zero as w does.
  Real const ramp = RampFactor(pmesh->time);
  bool const ramping = (ramp < 1.0);

  if (dc_ramp_start >= 0.0 && 0 == global_variable::my_rank) {
    if (!dc_ramp_begun && ramping) {
      dc_ramp_begun = true;
      std::cout << "### Drift control ramp-down begun at time = " << pmesh->time
                << "; correction reaches zero at "
                << dc_ramp_start + dc_ramp_time << "." << std::endl;
    }
    if (!dc_ramp_done && ramp <= 0.0) {
      dc_ramp_done = true;
      std::cout << "### Drift control ramp-down complete at time = " << pmesh->time
                << "; the applied correction is now zero." << std::endl;
    }
  }

  // The observer pole sits at -omega_o and is integrated explicitly, so omega_o dt is
  // the binding numerical constraint on the DOB branch.
  if (dc_variety == DOB && !dc_dt_warned && dt > 0.0 && dc_omega_o * dt > 0.5) {
    dc_dt_warned = true;
    if (0 == global_variable::my_rank) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "dc_omega_o*dt = " << dc_omega_o * dt << " exceeds 0.5; the drift "
                << "control observer may be unstable. Reduce dc_omega_o below "
                << 0.5 / dt << "." << std::endl;
    }
  }

  if (dc_first_step || dt <= 0.0) {
    // Suppress the velocity term only. dc_integral and dc_p are restored from restart.
    for (int a = 0; a < NDIM; ++a) {
      dc_vel[a]        = 0.0;
      dc_prev_error[a] = dc_pos[a] - dc_fixed[a];
      dc_fhat[a]       = dc_p[a];
    }
    dc_first_step = false;
  } else if (dc_variety == DOB) {
    Real const wc2  = dc_omega_c * dc_omega_c;
    Real const twzc = 2.0 * dc_zeta * dc_omega_c;
    for (int a = 0; a < NDIM; ++a) {
      Real const e       = dc_pos[a] - dc_fixed[a];
      Real const vel_raw = (dc_pos[a] - dc_pos_old[a]) / dt;
      dc_vel[a]          = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
      dc_prev_error[a]   = e;

      // pdot = omega_o (u - fhat) = omega_o (omega_c^2 e + 2 zeta omega_c v)
      if (!ramping) {
        dc_p[a] += dt * dc_omega_o * (wc2 * e + twzc * dc_vel[a]);
      }
      // Recorded after the update so the logged fhat is what the next RHS applies.
      dc_fhat[a] = dc_p[a] + dc_omega_o * dc_vel[a];
    }
  } else if (dc_variety == PID) {
    for (int a = 0; a < NDIM; ++a) {
      Real const e = dc_pos[a] - dc_fixed[a];
      if (!ramping) {
        dc_integral[a] += e * dt;
      }

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
  for (int a = 0; a < NDIM; ++a) {
    pin->SetReal("z4c", integral_key[a], dc_integral[a]);
    pin->SetReal("z4c", dob_key[a], dc_p[a]);
  }

  if (0 == global_variable::my_rank && 0 == pmesh->ncycle % out_every) {
    ofile << pmesh->ncycle << " "
          << pmesh->time << " "
          << dc_pos[0] << " "
          << dc_pos[1] << " "
          << dc_pos[2] << " "
          << dc_vel[0] << " "
          << dc_vel[1] << " "
          << dc_vel[2] << " ";
    if (dc_variety == DOB) {
      ofile << dc_p[0] << " "
            << dc_p[1] << " "
            << dc_p[2] << " "
            << dc_fhat[0] << " "
            << dc_fhat[1] << " "
            << dc_fhat[2] << std::endl << std::flush;
    } else {
      ofile << dc_integral[0] << " "
            << dc_integral[1] << " "
            << dc_integral[2] << std::endl << std::flush;
    }
  }
}
