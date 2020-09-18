#ifndef DRIVER_DRIVER_HPP_
#define DRIVER_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.hpp
//  \brief definitions for Driver class

#include <ctime>

#include "parameter_input.hpp"
#include "outputs/outputs.hpp"

//----------------------------------------------------------------------------------------
//! \class Driver

class Driver {
 public:
  Driver(ParameterInput *pin, Mesh *pmesh);
  ~Driver() = default;

  // data
  bool time_evolution;

  // folowing data only relevant for runs involving time evolution
  Real tlim;      // stopping time
  int nlim;       // cycle-limit
  int ndiag;      // cycles between output of diagnostic information
  // variables for various SSP RK integrators
  std::string integrator;          // integrator name (rk1, rk2, rk3)
  int nstages;                     // total number of stages
  Real cfl_limit;                  // maximum CFL number for integrator
  Real gam0[3], gam1[3], beta[3];  // averaging weights and fractional timestep per stage

  // functions
  void Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout);
  void Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout);
  void Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout);

 private:
  Kokkos::Timer run_time_;   // generalized timer for cpu/gpu/etc
  int nmb_updated_;         // running total of MB updated during run
  void OutputCycleDiagnostics(Mesh *pm);

};
#endif // DRIVER_DRIVER_HPP_
