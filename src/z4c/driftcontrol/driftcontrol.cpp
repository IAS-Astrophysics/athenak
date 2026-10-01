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
// Parameter keys under which the BDOB applied correction is checkpointed.
char const *const bdob_u_key[3] = {"dc_bdob_ux", "dc_bdob_uy", "dc_bdob_uz"};
// Parameter keys under which the spent effort budget is checkpointed.
char const *const budget_key[3] = {"dc_budget_dx", "dc_budget_dy", "dc_budget_dz"};

//! \brief Smooth, strictly bounded soft limiter,
Real SoftLimit(Real u, Real umax, int n) {
  if (umax < 0.0) {
    return u;                                  // limiter disabled
  }
  if (umax == 0.0) {
    return 0.0;                                // zero authority, same as gain = 0
  }
  Real const q = std::fabs(u) / umax;
  if (q > 1.0e3) {
    return (u > 0.0) ? umax : -umax;           // saturated to better than 1e-9 relative
  }
  Real const pw = 2.0 * static_cast<Real>(n);
  return u / std::pow(1.0 + std::pow(q, pw), 1.0 / pw);
}

//! \brief Symmetric slew limit on the applied correction. Guarantees that a mid-chain
//! change of umax, of a per-axis gain, or of the variety cannot step dbeta/dt.
Real SlewLimit(Real u, Real u_prev, Real max_rate, Real dt) {
  if (max_rate < 0.0 || dt <= 0.0) {
    return u;
  }
  Real const du = max_rate * dt;
  return std::clamp(u, u_prev - du, u_prev + du);
}
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
  } else if (str.compare("bdob") == 0) {
    return BDOB;
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

  // BDOB effort budget. The defaults disable the limiter, so dc_variety = bdob with
  // nothing else set is dc_variety = dob.
  dc_umax[0]       = pin->GetOrAddReal("z4c", "dc_umax_x", -1.0);
  dc_umax[1]       = pin->GetOrAddReal("z4c", "dc_umax_y", -1.0);
  dc_umax[2]       = pin->GetOrAddReal("z4c", "dc_umax_z", -1.0);
  dc_urate         = pin->GetOrAddReal("z4c", "dc_urate", -1.0);
  dc_sat_order     = pin->GetOrAddInteger("z4c", "dc_sat_order", 4);
  if (dc_sat_order < 1) {
    dc_sat_order = 1;
  }

  // Self-scheduling effort budget.
  dc_budget[0]     = pin->GetOrAddReal("z4c", "dc_budget_x", -1.0);
  dc_budget[1]     = pin->GetOrAddReal("z4c", "dc_budget_y", -1.0);
  dc_budget[2]     = pin->GetOrAddReal("z4c", "dc_budget_z", -1.0);
  dc_budget_c      = pin->GetOrAddReal("z4c", "dc_budget_c", 0.025);
  dc_budget_tend   = pin->GetOrAddReal("z4c", "dc_budget_tend",
                                       pin->GetOrAddReal("time", "tlim", 0.0));

  // Velocity-budget schedule on omega_c. Negative disables it and omega_c is constant.
  dc_vmax          = pin->GetOrAddReal("z4c", "dc_vmax", -1.0);
  dc_sched_C       = pin->GetOrAddReal("z4c", "dc_sched_C", 0.341);
  dc_omega_c_eff   = dc_omega_c;

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
    dc_u[a]          = pin->GetOrAddReal("z4c", bdob_u_key[a], 0.0);
    dc_budget_used[a] = pin->GetOrAddReal("z4c", budget_key[a], 0.0);
    dc_umax_eff[a]   = dc_umax[a];
  }

  std::string const dc_fname =
      pin->GetOrAddString("z4c", "dc_filename", "drift_control");

  // Per-axis gain on the applied correction.
  dc_gain[0] = pin->GetOrAddReal("z4c", "dc_gain_x", 1.0);
  dc_gain[1] = pin->GetOrAddReal("z4c", "dc_gain_y", 1.0);
  dc_gain[2] = pin->GetOrAddReal("z4c", "dc_gain_z", 1.0);

  if (0 == global_variable::my_rank) {
    for (int a = 0; a < NDIM; ++a) {
      if (dc_gain[a] == 0.0) {
        std::cout << "### Drift control: axis " << a << " is open loop (dc_gain = 0); "
                  << (dc_variety == BDOB
                      ? "its observer keeps tracking f, which is correct with the loop "
                        "open and makes re-enabling the axis bumpless."
                      : "its integrator/observer state is frozen.") << std::endl;
      } else if (dc_variety == BDOB && dc_budget[a] >= 0.0) {
        std::cout << "### Drift control: axis " << a << " on an effort budget of "
                  << dc_budget[a] << " M of differential displacement to t = "
                  << dc_budget_tend << " (" << dc_budget_used[a]
                  << " M already spent, c = " << dc_budget_c << ")." << std::endl;
      } else if (dc_variety == BDOB && dc_umax[a] >= 0.0) {
        std::cout << "### Drift control: axis " << a << " effort ceiling |u| < "
                  << dc_umax[a] << " (soft, order " << dc_sat_order << ")." << std::endl;
      } else if (dc_variety == DOB && dc_gain[a] != 1.0) {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "dc_variety = dob with dc_gain[" << a << "] = " << dc_gain[a]
                  << ". The observer assumes the full correction is applied, so fhat "
                  << "is biased for any gain other than 0 or 1. Use the pid or "
                  << "oscillator variety for a partial gain." << std::endl;
      }
    }
  }

  if (0 == global_variable::my_rank) {
    std::string ofname = pin->GetString("job", "basename") + ".";
    ofname += dc_fname;
    ofname += ".txt";
    ofile.open(ofname.c_str());
    ofile << "# variety=" << pin->GetString("z4c", "dc_variety")
          << " fixed=(" << dc_fixed[0] << "," << dc_fixed[1] << "," << dc_fixed[2] << ")"
          << " gain=(" << dc_gain[0] << "," << dc_gain[1] << "," << dc_gain[2] << ")";
    if (dc_variety == PID) {
      ofile << " Kp=" << pin->GetOrAddReal("z4c", "dc_Kp", 1.0)
            << " Ki=" << pin->GetOrAddReal("z4c", "dc_Ki", 0.1)
            << " Kd=" << pin->GetOrAddReal("z4c", "dc_Kd", 2.0)
            << " integral_cap=" << dc_integral_cap;
    } else if (dc_variety == DOB || dc_variety == BDOB) {
      ofile << " omega_c=" << dc_omega_c << " omega_o=" << dc_omega_o
            << " zeta=" << dc_zeta;
      if (dc_variety == BDOB) {
        ofile << " umax=(" << dc_umax[0] << "," << dc_umax[1] << "," << dc_umax[2] << ")"
              << " sat_order=" << dc_sat_order << " urate=" << dc_urate
              << " budget=(" << dc_budget[0] << "," << dc_budget[1] << ","
              << dc_budget[2] << ")"
              << " budget_c=" << dc_budget_c << " budget_tend=" << dc_budget_tend
              << " vmax=" << dc_vmax;
      }
    } else if (dc_variety == Oscillator) {
      ofile << " tau=" << pin->GetOrAddReal("z4c", "dc_damping_time", 0.5)
            << " zeta=" << pin->GetOrAddReal("z4c", "dc_damping_coeff", 1.0);
    }
    ofile << " scale=" << pin->GetOrAddReal("z4c", "dc_damping_scale", 10.0)
          << " ramp_start=" << dc_ramp_start << " ramp_time=" << dc_ramp_time << "\n";
    if (dc_variety == BDOB) {
      ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz 9:px 10:py 11:pz"
               " 12:fhatx 13:fhaty 14:fhatz 15:ux 16:uy 17:uz"
               " 18:Dx 19:Dy 20:Dz 21:umaxx 22:umaxy 23:umaxz 24:omega_c\n";
    } else if (dc_variety == DOB) {
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
//! \brief Advance the bounded-authority observer and form the applied correction.
void DriftControl::UpdateBoundedObserver(Real ramp, Real dt, bool advance) {
  dc_omega_c_eff = dc_omega_c;
  if (dc_vmax > 0.0) {
    Real e2 = 0.0;
    for (int a = 0; a < NDIM; ++a) {
      Real const ea = dc_pos[a] - dc_fixed[a];
      e2 += ea * ea;
    }
    Real const enorm = std::sqrt(e2);
    if (enorm > 0.0 && dc_sched_C > 0.0) {
      dc_omega_c_eff = std::min(dc_omega_c, dc_vmax / (dc_sched_C * enorm));
    }
  }
  Real const wc2  = dc_omega_c_eff * dc_omega_c_eff;
  Real const twzc = 2.0 * dc_zeta * dc_omega_c_eff;

  for (int a = 0; a < NDIM; ++a) {
    Real const e     = dc_pos[a] - dc_fixed[a];
    Real const fhat  = dc_p[a] + dc_omega_o * dc_vel[a];
    Real const u_pd  = wc2 * e + twzc * dc_vel[a];
    Real const u_cmd = u_pd + fhat;

    // Effort ceiling in force this cycle.
    Real umax = dc_umax[a];
    if (dc_budget[a] >= 0.0 && dc_budget_c > 0.0) {
      Real const left  = std::max(dc_budget[a] - dc_budget_used[a], 0.0);
      Real const tleft = std::max(dc_budget_tend - pmesh->time, 1.0);
      Real const ubgt  = left / (dc_budget_c * tleft);
      umax = (umax < 0.0) ? ubgt : std::min(umax, ubgt);
    }
    dc_umax_eff[a] = umax;

    Real u_app = dc_gain[a] * ramp * SoftLimit(u_cmd, umax, dc_sat_order);
    u_app = SlewLimit(u_app, dc_u[a], dc_urate, dt);

    if (advance) {
      dc_p[a] += dt * dc_omega_o * (u_pd + (u_app - u_cmd));
      // Charge the budget with the correction that was actually applied.
      dc_budget_used[a] += dt * dc_budget_c * std::fabs(u_app);
    }
    // Recorded after the update so the logged fhat is the one the next RHS uses.
    dc_fhat[a]       = dc_p[a] + dc_omega_o * dc_vel[a];
    dc_prev_error[a] = e;
    dc_u[a]          = u_app;
  }
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
  if ((dc_variety == DOB || dc_variety == BDOB) && !dc_dt_warned && dt > 0.0 &&
      dc_omega_o * dt > 0.5) {
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
    if (dc_variety == BDOB) {
      UpdateBoundedObserver(ramp, dt, false);
    }
    dc_first_step = false;
  } else if (dc_variety == BDOB) {
    for (int a = 0; a < NDIM; ++a) {
      Real const vel_raw = (dc_pos[a] - dc_pos_old[a]) / dt;
      dc_vel[a] = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
    }
    UpdateBoundedObserver(ramp, dt, true);
  } else if (dc_variety == DOB) {
    Real const wc2  = dc_omega_c * dc_omega_c;
    Real const twzc = 2.0 * dc_zeta * dc_omega_c;
    for (int a = 0; a < NDIM; ++a) {
      Real const e       = dc_pos[a] - dc_fixed[a];
      Real const vel_raw = (dc_pos[a] - dc_pos_old[a]) / dt;
      dc_vel[a]          = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
      dc_prev_error[a]   = e;

      if (!ramping && dc_gain[a] != 0.0) {
        dc_p[a] += dt * dc_omega_o * (wc2 * e + twzc * dc_vel[a]);
      }
      // Recorded after the update so the logged fhat is what the next RHS applies.
      dc_fhat[a] = dc_p[a] + dc_omega_o * dc_vel[a];
    }
  } else if (dc_variety == PID) {
    for (int a = 0; a < NDIM; ++a) {
      Real const e = dc_pos[a] - dc_fixed[a];
      if (!ramping && dc_gain[a] != 0.0) {
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
    pin->SetReal("z4c", bdob_u_key[a], dc_u[a]);
    pin->SetReal("z4c", budget_key[a], dc_budget_used[a]);
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
    if (dc_variety == DOB || dc_variety == BDOB) {
      ofile << dc_p[0] << " "
            << dc_p[1] << " "
            << dc_p[2] << " "
            << dc_fhat[0] << " "
            << dc_fhat[1] << " "
            << dc_fhat[2];
      if (dc_variety == BDOB) {
        ofile << " " << dc_u[0] << " " << dc_u[1] << " " << dc_u[2]
              << " " << dc_budget_used[0] << " " << dc_budget_used[1]
              << " " << dc_budget_used[2]
              << " " << dc_umax_eff[0] << " " << dc_umax_eff[1]
              << " " << dc_umax_eff[2]
              << " " << dc_omega_c_eff;
      }
      ofile << std::endl << std::flush;
    } else {
      ofile << dc_integral[0] << " "
            << dc_integral[1] << " "
            << dc_integral[2] << std::endl << std::flush;
    }
  }
}
