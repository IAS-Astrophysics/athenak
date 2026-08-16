//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driftcontrol/driftcontrol.hpp
//! \brief Definitions for the DriftControl class.

#ifndef Z4C_DRIFTCONTROL_DRIFTCONTROL_HPP_
#define Z4C_DRIFTCONTROL_DRIFTCONTROL_HPP_

#include <cstdio>
#include <fstream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"

// Forward declaration
class Mesh;
class ParameterInput;

//! \class DriftControl
//! \brief Prevents coordinate drift of a fixed point during evolution.
//!
//! Tracks the position of a fixed point (e.g. post-merger remnant centre)
//! via a CompactObjectTracker and adds a damped-oscillator
//! restoring force to the shift vector RHS to pull it back toward the
//! desired origin.
class DriftControl {
 public:
  enum Variety { Oscillator, PID, Relaxation, DOB };
  enum Center { CenterFixed, CenterTracker };

  static Variety VarietyFromString(std::string const &str);
  static Center CenterFromString(std::string const &str);

  DriftControl(Mesh *pmesh, ParameterInput *pin);
  ~DriftControl();

  void EvolveDriftControl();
  //! Write the control state to file
  void WriteDriftControl();

  //! Get DriftControl position
  inline Real GetPos(int a) const {
    return dc_pos[a];
  }
  //! Get DriftControl velocity
  inline Real GetVel(int a) const {
    return dc_vel[a];
  }
  //! Get DriftControl integral
  inline Real GetIntegral(int a) const {
    return dc_integral[a];
  }
  //! Get the DOB observer state p (fhat = p + omega_o v)
  inline Real GetP(int a) const {
    return dc_p[a];
  }
  //! Get the DOB estimate of the ambient drive f
  inline Real GetFhat(int a) const {
    return dc_fhat[a];
  }
  //! Get the position at which the object is being held
  inline Real GetFixed(int a) const {
    return dc_fixed[a];
  }
  //! Get DriftControl variety
  inline Variety GetVariety() const {
    return dc_variety;
  }

  //! Scale factor applied to the whole correction, ramping 1 -> 0 over
  //! [dc_ramp_start, dc_ramp_start + dc_ramp_time]. Returns 1 when disabled.
  Real RampFactor(Real time) const;

 private:
  Mesh const *pmesh;
  ParameterInput *pin;
  Variety dc_variety;
  int dc_tracker_index;
  Real dc_fixed[3];
  Real dc_pos[3];
  Real dc_pos_old[3];
  Real dc_vel[3];
  Real dc_integral[3];
  Real dc_prev_error[3];
  Real dc_p[3];        // DOB observer state
  Real dc_fhat[3];     // DOB estimate of the ambient drive (diagnostic; RHS recomputes)
  Real dc_omega_c;     // closed-loop PD bandwidth
  Real dc_omega_o;     // observer bandwidth
  Real dc_zeta;        // PD damping ratio
  bool dc_first_step;
  bool dc_dt_warned;   // one-shot guard against omega_o dt beyond the explicit limit
  Real dc_vel_cap;
  Real dc_integral_cap;
  Real dc_ramp_start;  // time at which the ramp-down begins; < 0 disables the ramp
  Real dc_ramp_time;   // ramp duration; <= 0 with a valid start means an instant cut
  bool dc_ramp_begun;  // one-shot log guards
  bool dc_ramp_done;
  int out_every;
  std::ofstream ofile;
};
#endif // Z4C_DRIFTCONTROL_DRIFTCONTROL_HPP_
