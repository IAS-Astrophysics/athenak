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

#include "athena_arrays.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class Driver

class Driver {
 public:
  Driver(std::unique_ptr<ParameterInput> &pin, std::unique_ptr<Mesh> &pmesh,
         std::unique_ptr<Outputs> &pout);
  ~Driver() = default;

  // data
  bool time_evolution;

  // folowing data only relevant for runs involving time evolution
  Real tlim;      // stopping time
  int nlim;       // cycle-limit
  int ndiag;      // cycles between output of diagnostic information
  Real ssp_gam[3];  // averaging weights (gamma) for SSP RK integrators
  Real ssp_bet[3];  // fractional time step (beta) for SSP RK integrators
  int nstages;      // total number of stages in SSP RK integratorj:x

  // functions
  void Initialize(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);
  void Execute(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);
  void Finalize(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);

 private:
  clock_t tstart_, tstop_;  // variables to measure cpu time
  int nmb_updated_;         // running total of MB updated during run
  void OutputCycleDiagnostics(std::unique_ptr<Mesh> &pm);

};
#endif // DRIVER_DRIVER_HPP_
