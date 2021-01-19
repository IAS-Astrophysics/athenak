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
// "timestep" = "cycle" in explicit, multistage methods.
//
// Driver::Execute() invokes the tasklist from stage=1 to stage=ptlist->nstages

Driver::Driver(ParameterInput *pin, Mesh *pmesh) :
  tlim(-1.0), nlim(-1), ndiag(1)
{
  // set time-evolution option (default=dynamic)
  {std::string evolution_t = pin->GetOrAddString("time","evolution","dynamic");
  if (evolution_t.compare("stationary") == 0) {
    time_evolution = TimeEvolution::stationary;

  } else if (evolution_t.compare("kinematic") == 0) {
    time_evolution = TimeEvolution::kinematic;

  } else if (evolution_t.compare("dynamic") == 0) {
    time_evolution = TimeEvolution::dynamic;

  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> evolution = '" << evolution_t << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution != TimeEvolution::stationary) {
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

void Driver::Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout)
{
  //---- Step 1.  Set Boundary Conditions on conserved variables in all physics
  // Note sends on ALL MBs must be complete before receives execute

  // TODO: need to cycle through all physics modules/variables in this step

std::cout << "driver init 0" << std::endl;
  TaskStatus tstatus;
  tstatus = pmesh->pmb_pack->phydro->HydroInitRecv(this, 0);
std::cout << "driver init 1" << std::endl;
  tstatus = pmesh->pmb_pack->phydro->HydroSend(this, 0);
std::cout << "driver init 2" << std::endl;
  tstatus = pmesh->pmb_pack->phydro->HydroClearSend(this, 0);
std::cout << "driver init 3" << std::endl;
  tstatus = pmesh->pmb_pack->phydro->HydroClearRecv(this, 0);
std::cout << "driver init 4" << std::endl;
  tstatus = pmesh->pmb_pack->phydro->HydroReceive(this, 0);
std::cout << "driver init 5" << std::endl;
  pmesh->pmb_pack->phydro->HydroApplyPhysicalBCs(this, 0);
std::cout << "driver init 6" << std::endl;

  // convert conserved to primitive over whole mesh
  tstatus = pmesh->pmb_pack->phydro->ConToPrim(this, 0);

std::cout << "driver init 7" << std::endl;
  //---- Step 2.  Compute first time step (if problem involves time evolution

  // TODO: need to cycle through all physics modules/variables in this step

  if (time_evolution != TimeEvolution::stationary) {
    tstatus = pmesh->pmb_pack->phydro->NewTimeStep(this, nstages);
    pmesh->NewTimeStep(tlim);
  }
std::cout << "driver init 8" << std::endl;

  //---- Step 3.  Cycle through output Types and load data / write files.

  for (auto &out : pout->pout_list_) {
    out->LoadOutputData(pmesh);
    out->WriteOutputFile(pmesh, pin);
  }
std::cout << "driver init 9" << std::endl;

  //---- Step 4.  Initialize various counters, timers, etc.

  run_time_.reset();
  nmb_updated_ = 0;

  return;
}


//----------------------------------------------------------------------------------------
// Driver::Execute()

void Driver::Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout)
{
  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list...\n" << std::endl;
  }

  while ((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0)) {

    if (time_evolution != TimeEvolution::stationary) {
      if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}

      // Do multi-stage time evolution TaskLists
      int npacks = 1;
      MeshBlockPack* pmbp = pmesh->pmb_pack;
      for (int stage=1; stage<=nstages; ++stage) {

        // StageStart Tasks ---
        // tasks that must be completed over all MBPacks before start of each stage
        {for (int p=0; p<npacks; ++p) {
          if (!(pmbp->tl_stagestart.Empty())) {pmbp->tl_stagestart.Reset();}
        }
        int npack_left = npacks;
        while (npack_left > 0) {
          if (pmbp->tl_stagestart.Empty()) {
            npack_left--; 
          } else {
            if (!pmbp->tl_stagestart.IsComplete()) {
              auto status = pmbp->tl_stagestart.DoAvailable(this,stage);
              if (status == TaskListStatus::complete) { npack_left--; }
            }
          }
        }} // extra brace to enclose scope

        // StageRun Tasks ---
        // tasks in each stage
        {for (int p=0; p<npacks; ++p) {
          if (!(pmbp->tl_stagerun.Empty())) {pmbp->tl_stagerun.Reset();}
        }
        int npack_left = npacks;
        while (npack_left > 0) {
          if (pmbp->tl_stagerun.Empty()) {
            npack_left--; 
          } else {
            if (!pmbp->tl_stagerun.IsComplete()) {
              auto status = pmbp->tl_stagerun.DoAvailable(this,stage);
              if (status == TaskListStatus::complete) { npack_left--; }
            }
          }
        }} // extra brace to enclose scope

        // StageEnd Tasks ---
        // tasks that must be completed over all MBs at the end of each stage
        {for (int p=0; p<npacks; ++p) {
          if (!(pmbp->tl_stageend.Empty())) {pmbp->tl_stageend.Reset();}
        }
        int npack_left = npacks;
        while (npack_left > 0) {
          if (pmbp->tl_stageend.Empty()) {
            npack_left--; 
          } else {
            if (!pmbp->tl_stageend.IsComplete()) {
              auto status = pmbp->tl_stageend.DoAvailable(this,stage);
              if (status == TaskListStatus::complete) { npack_left--; }
            }
          }
        }} // extra brace to enclose scope

      } // end of loop over stages

      // Add STS TaskLists, etc here....

      // increment time, ncycle, etc.
      // Compute new timestep
      pmesh->time = pmesh->time + pmesh->dt;
      pmesh->ncycle++;
      nmb_updated_ += pmesh->nmb_total;
      pmesh->NewTimeStep(tlim);

      // Make outputs during execution
      for (auto &out : pout->pout_list_) {
        // compare at floating point (32-bit) precision to reduce effect of round off
        float time_32 = static_cast<float>(pmesh->time);
        float next_32 = static_cast<float>(out->out_params.last_time+out->out_params.dt);
        float tlim_32 = static_cast<float>(tlim);
        if (time_32 >= next_32 && time_32 < tlim_32) {
          out->LoadOutputData(pmesh);
          out->WriteOutputFile(pmesh, pin);
        }
      }

    }
  }

  return;

}

//----------------------------------------------------------------------------------------
// Driver::Finalize()
// Tasks to be performed after execution of Driver, such as making final output and
// printing diagnostic messages

void Driver::Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout)
{
  // cycle through output Types and load data / write files
  //  This design allows for asynchronous outputs to implemented in the future.
  for (auto &out : pout->pout_list_) {
    out->LoadOutputData(pmesh);
    out->WriteOutputFile(pmesh, pin);
  }
    
  float exe_time = run_time_.seconds();

  if (time_evolution != TimeEvolution::stationary) { 
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
        std::cout << std::endl << "Current number of MeshBlocks = " << pmesh->nmb_total
                  << std::endl << pmesh->nmb_created << " MeshBlocks were created, and "
                  << pmesh->nmb_deleted << " were deleted during this run." << std::endl;
      }
  
      // Calculate and print the zone-cycles/exe-second and wall-second
      std::uint64_t zonecycles = nmb_updated_ *
          static_cast<std::uint64_t>(pmesh->pmb_pack->NumberOfMeshBlockCells());
      float zcps = static_cast<float>(zonecycles) / exe_time;

      std::cout << std::endl << "zone-cycles = " << zonecycles << std::endl;
      std::cout << "cpu time used  = " << exe_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Driver::OutputCycleDiagnostics()

void Driver::OutputCycleDiagnostics(Mesh *pm)
{
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    std::cout << "cycle=" << pm->ncycle << std::scientific << std::setprecision(dtprcsn)
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}


