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

class Driver {
 public:
  Driver(std::unique_ptr<ParameterInput> &pin, std::unique_ptr<Mesh> &pmesh,
         std::unique_ptr<Outputs> &pout);
  ~Driver() = default;

  // data

  // functions
  void Execute(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);


 private:

};

#endif // DRIVER_DRIVER_HPP_
