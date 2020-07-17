#ifndef DRIVER_DRIVER_HPP_
#define DRIVER_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.hpp
//  \brief definitions for Driver class

#include "athena_arrays.hpp"
#include "parameter_input.hpp"

// constants that enumerate time evolution options
enum class DriverTimeEvolution {hydrostatic, kinematic, hydrodynamic, no_time_evolution};

//----------------------------------------------------------------------------------------
//! \class Driver

class Driver {
 public:
  Driver(std::unique_ptr<ParameterInput> &pin, std::unique_ptr<Mesh> &pmesh,
         std::unique_ptr<Outputs> &pout);
  ~Driver() = default;

  // data
  DriverTimeEvolution time_evolution;
  Real tlim;
  int nlim, ndiag;

  // functions
  void Execute(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);
  void OutputCycleDiagnostics(std::unique_ptr<Mesh> &pm);


 private:

};

#endif // DRIVER_DRIVER_HPP_
