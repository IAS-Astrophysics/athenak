#ifndef DRIVER_DRIVER_HPP_
#define DRIVER_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.hpp
//  \brief definitions for Driver class
//
// Note ProblemGenerator object is stored in Driver and is called in Initialize(). If the
// pgen class contains analysis routines that are run at end of execution, they can be
// called in Finalize().

#include <ctime>
#include <memory>
#include <string>

#include "parameter_input.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \class Driver

class Driver {
 public:
  Driver(ParameterInput *pin, Mesh *pmesh, Real wtlim, Kokkos::Timer* ptimer);
  ~Driver() = default;

  // data
  TimeEvolution time_evolution;
  DvceArray6D<Real> impl_src;  // stiff source terms used in ImEx integrators

  // folowing data only relevant for runs involving time evolution
  Real tlim;      // stopping time
  int nlim;       // cycle-limit
  int ndiag;      // cycles between output of diagnostic information
  // variables for various SSP and ImEx RK integrators
  std::string integrator;          // integrator name (rk1, rk2, rk3)
  int nimp_stages;                 // number of implicit stages (ImEx only)
  int nexp_stages;                 // number of explicit stages (both SSP-RK and ImEx)
  Real gam0[4], gam1[4], beta[4];  // weights and fractional timestep per explicit stage
  Real delta[4];                   // weights for updating the intermediate stage (u1)
  Real a_twid[4][4], a_impl;       // matrix elements for implicit stages in ImEx
  Real cfl_limit;                  // maximum CFL number for integrator
  Real gamma;                      // gamma value for the IMEX_new integrator
  Kokkos::Timer* pwall_clock_;     // timer for tracking the wall clock
  Real wall_time;

  // functions
  void ExecuteTaskList(Mesh *pm, std::string tl, int stage);
  void Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool rflag);
  void Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout);
  void Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout);
  void InitBoundaryValuesAndPrimitives(Mesh *pm);

 private:
  Kokkos::Timer run_time_;      // generalized timer for cpu/gpu/etc
  std::uint64_t nmb_updated_;   // running total of MB updated during run
  std::uint64_t npart_updated_; // running total of particles updated during run
  float lb_efficiency_;         // measure of how efficient was load balancing
  void OutputCycleDiagnostics(Mesh *pm);
  Real UpdateWallClock();
};
#endif // DRIVER_DRIVER_HPP_
