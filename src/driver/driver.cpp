//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.cpp
//  \brief implementation of functions in class Driver

#include <iostream>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "driver.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Driver::Driver(std::unique_ptr<ParameterInput> &pin, std::unique_ptr<Mesh> &pmesh,
     std::unique_ptr<Outputs> &pout) : tlim(0.0), nlim(0), ndiag(1) {

  // set time-evolution option (no default)
  {std::string evolution_t = pin->GetString("driver","evolution");
  if (evolution_t.compare("hydrostatic")) {
    time_evolution = DriverTimeEvolution::hydrostatic;
  } else if (evolution_t.compare("kinematic")) {
    time_evolution = DriverTimeEvolution::kinematic;
  } else if (evolution_t.compare("hydrodynamic")) {
    time_evolution = DriverTimeEvolution::hydrodynamic;
  } else if (evolution_t.compare("no_time_evolution")) {
    time_evolution = DriverTimeEvolution::hydrodynamic;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<driver> evolution = '" << evolution_t << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution != DriverTimeEvolution::no_time_evolution) {
    tlim = pin->GetReal("time", "tlim");
    nlim = pin->GetOrAddInteger("time", "nlim", -1);
    ndiag = pin->GetOrAddInteger("time", "ndiag", 1);
  }

}

//----------------------------------------------------------------------------------------
// dtor

//----------------------------------------------------------------------------------------
// Driver::Execute()

void Driver::Execute(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout) {

  // cycle through output Types and load data / write files.  This design allows for
  // asynchronous outputs to implemented in the future.
  // TODO: cycle through OutputTypes
  pout->poutput_list_.front()->LoadOutputData(pmesh);
  pout->poutput_list_.front()->WriteOutputFile(pmesh);

  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list...\n" << std::endl;
  }

//  while ((pmesh->time < tlim) &&
//         (pmesh->ncycle < nlim || nlim < 0)) {

    if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}

}

//----------------------------------------------------------------------------------------
// Driver::Execute()

void Driver::OutputCycleDiagnostics(std::unique_ptr<Mesh> &pm) {
  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  if (pm->ncycle % ndiag == 0) {
    std::cout << "cycle=" << pm->ncycle << std::scientific << std::setprecision(dtprcsn)
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}


