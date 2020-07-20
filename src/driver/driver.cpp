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
#include "hydro/hydro.hpp"
#include "driver.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Driver::Driver(std::unique_ptr<ParameterInput> &pin, std::unique_ptr<Mesh> &pmesh,
     std::unique_ptr<Outputs> &pout) : tlim(-1.0), nlim(-1), ndiag(1),
     time_evolution(false) {

  hydro::Hydro *phyd = pmesh->mblocks.front().phydro;
  if (phyd->hydro_evol != hydro::HydroEvolution::no_evolution) {
    time_evolution = true;
  }
  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution) {
    tlim = pin->GetReal("time", "tlim");
    nlim = pin->GetOrAddInteger("time", "nlim", -1);
    ndiag = pin->GetOrAddInteger("time", "ndiag", 1);
  }

}

//----------------------------------------------------------------------------------------
// dtor

//----------------------------------------------------------------------------------------
// Driver::Initialize()
// Tasks to be performed before execution of Driver, such as computing initial time step,
// setting boundary conditions, and outputing ICs

void Driver::Initialize(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout) {

  // cycle through output Types and load data / write files.  This design allows for
  // asynchronous outputs to implemented in the future.
  // TODO: cycle through OutputTypes
  pout->poutput_list_.front()->LoadOutputData(pmesh);
  pout->poutput_list_.front()->WriteOutputFile(pmesh);

  tstart_ = clock();
  nmb_updated_ = 0;

  if (time_evolution) {
    hydro::Hydro *phyd = pmesh->mblocks.front().phydro;
    phyd->NewTimeStep();
    pmesh->NewTimeStep(tlim);
  }
  return;
}


//----------------------------------------------------------------------------------------
// Driver::Execute()

void Driver::Execute(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout) {

  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list...\n" << std::endl;
  }

/*****/
// Start with simple one-step driver
/*****/


  while ((pmesh->time < tlim) &&
         (pmesh->ncycle < nlim || nlim < 0)) {

    if (time_evolution) {
      if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}

      hydro::Hydro *phyd = pmesh->mblocks.front().phydro;
      phyd->CopyConserved(phyd->u0, phyd->u1);
      phyd->HydroDivFlux();
      phyd->HydroUpdate();
  
      pmesh->time = pmesh->time + pmesh->dt;
      pmesh->ncycle++;
      nmb_updated_ += pmesh->nmbtotal;

      phyd->NewTimeStep();
      pmesh->NewTimeStep(tlim);
    }
  }

  return;

}

//----------------------------------------------------------------------------------------
// Driver::Finalize()
// Tasks to be performed after execution of Driver, such as making final output and
// printing diagnostic messages

void Driver::Finalize(std::unique_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout) {

  // cycle through output Types and load data / write files.  This design allows for
  // asynchronous outputs to implemented in the future.
  // TODO: cycle through OutputTypes
  pout->poutput_list_.front()->LoadOutputData(pmesh);
  pout->poutput_list_.front()->WriteOutputFile(pmesh);
    
  tstop_ = clock();

  if (time_evolution) { 
    if (global_variable::my_rank == 0) {
      // Print diagnostic messages related to the end of the simulation
      OutputCycleDiagnostics(pmesh);
      if (pmesh->ncycle == nlim) {
        std::cout << std::endl << "Terminating on cycle limit" << std::endl;
      } else {
        std::cout << std::endl << "Terminating on time limit" << std::endl;
      }

      std::cout << "time=" << pmesh->time << " cycle=" << pmesh->ncycle << std::endl;
      std::cout << "tlim=" << tlim << " nlim=" << nlim << std::endl;

      if (pmesh->adaptive) {
        std::cout << std::endl << "Current number of MeshBlocks = " << pmesh->nmbtotal
                  << std::endl << pmesh->nmb_created << " MeshBlocks were created, and "
                  << pmesh->nmb_deleted << " were deleted during this run." << std::endl;
      }
  
      // Calculate and print the zone-cycles/cpu-second and wall-second
      float cpu_time = (tstop_ > tstart_ ? static_cast<float>(tstop_ - tstart_) : 1.0) /
                        static_cast<float>(CLOCKS_PER_SEC);
      std::uint64_t zonecycles = nmb_updated_ *
          static_cast<std::uint64_t>(pmesh->mblocks.front().GetNumberOfMeshBlockCells());
      float zc_cpus = static_cast<float>(zonecycles) / cpu_time;

      std::cout << std::endl << "zone-cycles = " << zonecycles << std::endl;
      std::cout << "cpu time used  = " << cpu_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zc_cpus << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Driver::OutputCycleDiagnostics()

void Driver::OutputCycleDiagnostics(std::unique_ptr<Mesh> &pm) {
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    std::cout << "cycle=" << pm->ncycle << std::scientific << std::setprecision(dtprcsn)
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}


