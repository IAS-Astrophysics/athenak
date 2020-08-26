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
  // First, define each time-integrator by setting weights for each step of the algorithm
  // and the CFL number stability limit when coupled to the single-stage spatial operator.
  // Currently, the explicit, multistage time-integrators must be expressed as 2S-type
  // algorithms as in Ketcheson (2010) Algorithm 3, which incudes 2N (Williamson) and 2R
  // (van der Houwen) popular 2-register low-storage RK methods. The 2S-type integrators
  // depend on a bidiagonally sparse Shu-Osher representation; at each stage l:
  //
  //    U^{l} = a_{l,l-2}*U^{l-2} + a_{l-1}*U^{l-1}
  //          + b_{l,l-2}*dt*Div(F_{l-2}) + b_{l,l-1}*dt*Div(F_{l-1}),
  //
  // where U^{l-1} and U^{l-2} are previous stages and a_{l,l-2}, a_{l,l-1}=(1-a_{l,l-2}),
  // and b_{l,l-2}, b_{l,l-1} are weights that are different for each stage and
  // integrator. Previous timestep U^{0} = U^n is given, and the integrator solves
  // for U^{l} for 1 <= l <= nstages.
  //
  // The 2x RHS evaluations of Div(F) and source terms per stage is avoided by adding
  // another weighted average / caching of these terms each stage. The API and framework
  // is extensible to three register 3S* methods, although none are currently implemented.

  // Notation: exclusively using "stage", equivalent in lit. to "substage" or "substep"
  // (infrequently "step"), to refer to the intermediate values of U^{l} between each
  // "timestep" = "cycle" in explicit, multistage methods. This is to disambiguate the
  // temporal integration from other iterative sequences; "Step" is often used for generic
  // sequences in code, e.g. main.cpp: "Step 1: MPI"
  //
  // main.cpp invokes the tasklist in a for () loop from stage=1 to stage=ptlist->nstages

Driver::Driver(ParameterInput *pin, Mesh *pmesh,
     Outputs *pout) : tlim(-1.0), nlim(-1), ndiag(1),
     time_evolution(false) {

  hydro::Hydro *phyd = pmesh->mblocks.front().phydro;
  if (phyd->hydro_evol != hydro::HydroEvolution::no_evolution) {
    time_evolution = true;
  }
  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution) {
    integrator = pin->GetOrAddString("time", "integrator", "rk2");
    tlim = pin->GetReal("time", "tlim");
    nlim = pin->GetOrAddInteger("time", "nlim", -1);
    ndiag = pin->GetOrAddInteger("time", "ndiag", 1);

    if (integrator == "rk1") {
      // RK1: first-order Runge-Kutta / the forward Euler (FE) method
      nstages = 1;
      cfl_limit = 1.0;
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
    } else if (integrator == "rk2") {
      // Heun's method / SSPRK (2,2): Gottlieb (2009) equation 3.1
      // Optimal (in error bounds) explicit two-stage, second-order SSPRK
      nstages = 2;
      cfl_limit = 1.0;  // c_eff = c/nstages = 1/2 (Gottlieb (2009), pg 271)
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
  
      gam0[1] = 0.5;
      gam1[1] = 0.5;
      beta[1] = 0.5;
    } else if (integrator == "rk3") {
      // SSPRK (3,3): Gottlieb (2009) equation 3.2
      // Optimal (in error bounds) explicit three-stage, third-order SSPRK
      nstages = 3;
      cfl_limit = 1.0;  // c_eff = c/nstages = 1/3 (Gottlieb (2009), pg 271)
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;

      gam0[1] = 0.25;
      gam1[1] = 0.75;
      beta[1] = 0.25;

      gam0[2] = 2.0/3.0;
      gam1[2] = 1.0/3.0;
      beta[2] = 2.0/3.0;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "integrator=" << integrator << " not implemented" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

}

//----------------------------------------------------------------------------------------
// dtor

//----------------------------------------------------------------------------------------
// Driver::Initialize()
// Tasks to be performed before execution of Driver, such as computing initial time step,
// setting boundary conditions, and outputing ICs

void Driver::Initialize(Mesh *pmesh, Outputs *pout)
{
  //---- Step 1.  Set Boundary Conditions on conservd variables in all physics

  for (auto &mb : pmesh->mblocks) {
    TaskStatus tstatus;
    tstatus = mb.phydro->HydroSend(this, nstages);
    tstatus = mb.phydro->HydroReceive(this, nstages);
  }

  // convert conserved to primitive over whole mesh

  for (auto &mb : pmesh->mblocks) {
    TaskStatus tstatus;
    tstatus = mb.phydro->ConToPrim(this, nstages);
  }

  //---- Step 2.  Cycle through output Types and load data / write files.
  //  This design allows for asynchronous outputs to be implemented in the future.
  // TODO: cycle through OutputTypes

  pout->poutput_list_.front()->LoadOutputData(pmesh);
  pout->poutput_list_.front()->WriteOutputFile(pmesh);

  //---- Step 3.  Compute first time step (if problem involves time evolution

  if (time_evolution) {
    for (auto it = pmesh->mblocks.begin(); it < pmesh->mblocks.end(); ++it) {
      TaskStatus tstatus;
      tstatus = it->phydro->NewTimeStep(this, nstages);
    }
    pmesh->NewTimeStep(tlim);
  }

  //---- Step 4.  Initialize various counters, timers, etc.

  tstart_ = clock();
  nmb_updated_ = 0;

  return;
}


//----------------------------------------------------------------------------------------
// Driver::Execute()

void Driver::Execute(Mesh *pmesh, Outputs *pout) {

  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list...\n" << std::endl;
  }

  while ((pmesh->time < tlim) &&
         (pmesh->ncycle < nlim || nlim < 0)) {

    if (time_evolution) {
      if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}

/*** first implementation task list ***/

      // Do multi-stage time evolution TaskList
      for (int stage=1; stage<=nstages; ++stage) {

        for (auto &mb : pmesh->mblocks) {mb.tl_onestage.Reset();}
        int nmb_completed = 0;
        while (nmb_completed < pmesh->nmbthisrank) {
          // TODO(pgrete): need to let Kokkos::PartitionManager handle this
          for (auto &mb : pmesh->mblocks) {
            if (!mb.tl_onestage.IsComplete()) {
              auto status = mb.tl_onestage.DoAvailable(this,stage);
              if (status == TaskListStatus::complete) { nmb_completed++; }
            }
          }
        }

      }

      // Do STS TaskLists

/*** first implementation task list ***/

      pmesh->time = pmesh->time + pmesh->dt;
      pmesh->ncycle++;
      nmb_updated_ += pmesh->nmbtotal;

      pmesh->NewTimeStep(tlim);
    }
  }

  return;

}

//----------------------------------------------------------------------------------------
// Driver::Finalize()
// Tasks to be performed after execution of Driver, such as making final output and
// printing diagnostic messages

void Driver::Finalize(Mesh *pmesh, Outputs *pout) {

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
          static_cast<std::uint64_t>(pmesh->mblocks.front().NumberOfMeshBlockCells());
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

void Driver::OutputCycleDiagnostics(Mesh *pm) {
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    std::cout << "cycle=" << pm->ncycle << std::scientific << std::setprecision(dtprcsn)
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}


