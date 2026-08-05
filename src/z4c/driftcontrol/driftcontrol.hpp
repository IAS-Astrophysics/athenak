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
  enum Variety { Oscillator, PID, Relaxation };
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
  //! Get the position at which the object is being held
  inline Real GetFixed(int a) const {
    return dc_fixed[a];
  }
  //! Get DriftControl variety
  inline Variety GetVariety() const {
    return dc_variety;
  }

 private:
  Mesh const *pmesh;
  Variety dc_variety;
  int dc_tracker_index;
  Real dc_fixed[3];
  Real dc_pos[3];
  Real dc_pos_old[3];
  Real dc_vel[3];
  Real dc_integral[3];
  Real dc_prev_error[3];
  bool dc_first_step;
  Real dc_vel_cap;
  Real dc_integral_cap;
  int out_every;
  std::ofstream ofile;
};
#endif // Z4C_DRIFTCONTROL_DRIFTCONTROL_HPP_
