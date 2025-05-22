//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.cpp
//  \brief implementation of functions in class Driver

#include <iostream>
#include <iomanip>    // std::setprecision()
#include <limits>
#include <algorithm>
#include <string> // string

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "radiation/radiation.hpp"
#include "driver.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters
//
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

Driver::Driver(ParameterInput *pin, Mesh *pmesh, Real wtlim, Kokkos::Timer* ptimer) :
  tlim(-1.0),
  nlim(-1),
  ndiag(1),
  nmb_updated_(0),
  npart_updated_(0),
  lb_efficiency_(0),
  pwall_clock_(ptimer),
  wall_time(wtlim),
  impl_src("ru",1,1,1,1,1,1) {
  // set time-evolution option (no default)
  {
    std::string evolution_t = pin->GetString("time","evolution");
    if (evolution_t.compare("static") == 0) {
      time_evolution = TimeEvolution::tstatic;  // cannot use 'static' (keyword);
    } else if (evolution_t.compare("kinematic") == 0) {
      time_evolution = TimeEvolution::kinematic;
    } else if (evolution_t.compare("dynamic") == 0) {
      time_evolution = TimeEvolution::dynamic;
    } else {
      std::cout<<"### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
               <<"<hydro> evolution = '"<< evolution_t <<"' not implemented"<< std::endl;
      std::exit(EXIT_FAILURE);
    }
  } // extra brace to limit scope of string

  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution != TimeEvolution::tstatic) {
    integrator = pin->GetOrAddString("time", "integrator", "rk2");
    tlim = pin->GetReal("time", "tlim");
    nlim = pin->GetOrAddInteger("time", "nlim", -1);
    ndiag = pin->GetOrAddInteger("time", "ndiag", 1);

    if (integrator == "rk1") {
      // RK1: first-order Runge-Kutta / the forward Euler (FE) method
      nimp_stages = 0;
      nexp_stages = 1;
      cfl_limit = 1.0;
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
    } else if (integrator == "rk2") {
      // Heun's method / SSPRK (2,2): Gottlieb (2009) equation 3.1
      // Optimal (in error bounds) explicit two-stage, second-order SSPRK
      nimp_stages = 0;
      nexp_stages = 2;
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
      nimp_stages = 0;
      nexp_stages = 3;
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
    } else if (integrator == "rk4") {
      // RK4()4[2S] from Table 2 of Ketcheson (2010)
      // Non-SSP, explicit four-stage, fourth-order RK
      // Stability properties are similar to classical (non-SSP) RK4
      // (but ~2x L2 principal error norm).
      // Refer to Colella (2011) for linear stability analysis of constant coeff.
      nimp_stages = 0;
      nexp_stages = 4;

      // Colella (2011) eq 101; 1st order flux is most severe constraint
      cfl_limit = 1.3925;

      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.193743905974738;

      gam0[1] = 0.121098479554482;
      gam1[1] = 0.721781678111411;
      beta[1] = 0.099279895495783;

      gam0[2] = -3.843833699660025;
      gam1[2] = 2.121209265338722;
      beta[2] = 1.131678018054042;

      gam0[3] = 0.546370891121863;
      gam1[3] = 0.198653035682705;
      beta[3] = 0.310665766509336;

      delta[0] = 1.0;
      delta[1] = 0.217683334308543;
      delta[2] = 1.065841341361089;
      delta[3] = 0.0;
    } else if (integrator == "imex2") {
      // IMEX-SSP2(3,2,2): Pareschi & Russo (2005) Table III.
      // two-stage explicit, three-stage implicit, second-order ImEx
      // Note explicit steps identical to RK2
      nimp_stages = 3;
      nexp_stages = 2;
      cfl_limit = 1.0;
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = 1.0;

      gam0[1] = 0.5;
      gam1[1] = 0.5;
      beta[1] = 0.5;

      a_twid[0][0] = -1.0;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;

      a_twid[1][0] = 0.5;
      a_twid[1][1] = 0.0;
      a_twid[1][2] = 0.0;

      a_twid[2][0] = 0.0;
      a_twid[2][1] = 0.25;
      a_twid[2][2] = 0.25;
      a_impl = 0.5;
    } else if (integrator == "imex2+") {
      // IMEX(4,3,2): Krapp et al. (2024, arXiv:2310.04435), Eq.30.
      // three-stage explicit, four-stage implicit, second-order ImEx
      // two implicit stages added, adapting Athenak's overall architecture
      // Note explicit steps may not reduce to RK2 based on the parameters chosen
      nimp_stages = 4;
      nexp_stages = 3;
      cfl_limit = 1.0;
      gamma = 1.707106781186547;   //1+1/sqrt(2)
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = gamma;

      gam0[1] = (2.0*gamma-1.0)/(2.0*gamma*gamma);
      gam1[1] = (1.0-(2.0*gamma-1.0)/(2.0*gamma*gamma));
      beta[1] = 1.0/(2.0*gamma);

      gam0[2] = 1.0;
      gam1[2] = 0.0;
      beta[2] = 0.0;

      a_twid[0][0] = 0.0;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;
      a_twid[0][3] = 0.0;

      a_twid[1][0] = 0.0;
      a_twid[1][1] = 0.0;
      a_twid[1][2] = 0.0;
      a_twid[1][3] = 0.0;

      a_twid[2][0] = 0.0;
      a_twid[2][1] = 0.0;
      a_twid[2][2] = (1.0-2.0*gamma*gamma)/2.0/gamma;
      a_twid[2][3] = 0.0;

      a_twid[3][0] = 0.0;
      a_twid[3][1] = 0.0;
      a_twid[3][2] = 0.0;
      a_twid[3][3] = 0.0;

      a_impl = gamma;
    } else if (integrator == "imex3") {
      // IMEX-SSP3(4,3,3): Pareschi & Russo (2005) Table VI.
      // three-stage explicit, four-stage implicit, third-order ImEx
      // Note explicit steps identical to RK3
      nimp_stages = 4;
      nexp_stages = 3;
      cfl_limit = 1.0;
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;

      gam0[1] = 0.25;
      gam1[1] = 0.75;
      beta[1] = 0.25;

      gam0[2] = 2.0/3.0;
      gam1[2] = 1.0/3.0;
      beta[2] = 2.0/3.0;

      Real a = 0.24169426078821;
      Real b = 0.06042356519705;
      Real e = 0.12915286960590;
      a_twid[0][0] = -2.0*a;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;
      a_twid[0][3] = 0.0;

      a_twid[1][0] = a;
      a_twid[1][1] = 1.0 - 2.0*a;
      a_twid[1][2] = 0.0;
      a_twid[1][3] = 0.0;

      a_twid[2][0] = b;
      a_twid[2][1] = e - ((1.0-a)/4.0);
      a_twid[2][2] = 0.5 - b - e - 1.25*a;
      a_twid[2][3] = 0.0;

      a_twid[3][0] = (-2.0/3.0)*b;
      a_twid[3][1] = (1.0 - 4.0*e)/6.0;
      a_twid[3][2] = (4.0*(b + e + a) - 1.0)/6.0;
      a_twid[3][3] = 2.0*(1.0 - a)/3.0;
      a_impl = a;
    // Error, unrecognized integrator name.
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "integrator=" << integrator << " not implemented. "
         << "Valid choices are [rk1,rk2,rk3,rk4,imex2,imex3]." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Driver::ExecuteTaskList()
//! \brief Perform tasks over all MeshBlocks for the TaskList specified by string "tl".
//! Integer argument "stage" can be used to indicate at which step in overall algorithm
//! these tasks are to be performed, e.g. which stage of a multi-stage RK integrator.

void Driver::ExecuteTaskList(Mesh *pm, std::string tl, int stage) {
  MeshBlockPack* pmbp = pm->pmb_pack;
  for (int p=0; p<(pm->nmb_packs_thisrank); ++p) {
    if (!(pmbp->tl_map[tl]->Empty())) {pmbp->tl_map[tl]->Reset();}
  }
  int npack_left = (pm->nmb_packs_thisrank);
  while (npack_left > 0) {
    if (pmbp->tl_map[tl]->Empty()) {
      npack_left--;
    } else {
      if (!pmbp->tl_map[tl]->IsComplete()) {
        auto status = pmbp->tl_map[tl]->DoAvailable(this, stage);
        if (status == TaskListStatus::complete) { npack_left--; }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Driver::Initialize()
// Tasks to be performed before execution of Driver, such as setting ghost zones (BCs),
//  outputting ICs, and computing initial time step

void Driver::Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool res_flag) {
  //---- Step 1.  Set conserved variables in ghost zones for all physics
  InitBoundaryValuesAndPrimitives(pmesh);

  //---- Step 2.  Compute time step (if problem involves time evolution)
  hydro::Hydro *phydro = pmesh->pmb_pack->phydro;
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  radiation::Radiation *prad = pmesh->pmb_pack->prad;
  z4c::Z4c *pz4c = pmesh->pmb_pack->pz4c;
  if (time_evolution != TimeEvolution::tstatic) {
    if (phydro != nullptr) {
      (void) pmesh->pmb_pack->phydro->NewTimeStep(this, nexp_stages);
    }
    if (pmhd != nullptr) {
      (void) pmesh->pmb_pack->pmhd->NewTimeStep(this, nexp_stages);
    }
    if (prad != nullptr) {
      (void) pmesh->pmb_pack->prad->NewTimeStep(this, nexp_stages);
    }
    if (pz4c != nullptr) {
      (void) pmesh->pmb_pack->pz4c->NewTimeStep(this, nexp_stages);
    }

    pmesh->NewTimeStep(tlim);
  }

  //---- Step 3.  Cycle through output Types and load data / write files.
  if (!res_flag) { // only write outputs at the beginning of the run
    for (auto &out : pout->pout_list) {
      out->LoadOutputData(pmesh);
      out->WriteOutputFile(pmesh, pin);
    }
  }

  //---- Step 4.  Initialize various counters, timers, etc.
  run_time_.reset();
  nmb_updated_ = 0;

  // allocate memory for stiff source terms with ImEx integrators
  // only implemented for ion-neutral two fluid for now
  ion_neutral::IonNeutral *pionn = pmesh->pmb_pack->pionn;
  if (pionn != nullptr) {
    if (nimp_stages == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "IonNetral MHD can only be run with ImEx integrators."
          << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int nmb = std::max((pmesh->pmb_pack->nmb_thispack), (pmesh->nmb_maxperrank));
    auto &indcs = pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(impl_src, nimp_stages, nmb, 8, ncells3, ncells2, ncells1);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn Driver::Execute()
//! \brief Executes "main loop" by running all relevant task lists over all MeshBlockPacks
//! until a relevant stopping criteria is found (e.g. t > tlim). Calls AMR driver, and
//! performs outputs. Updates counters like (ncycle, time, etc.)

void Driver::Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list(s)...\n" << std::endl;
  }

  if (time_evolution == TimeEvolution::tstatic) {
    // TODO(@user): add work for time static problems here
  } else {
    Real elapsed_time = -1.;
    if (wall_time > 0.) {
      elapsed_time = UpdateWallClock();
    }
    while ((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0) &&
           (elapsed_time < wall_time)) {
      if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}

      // Execute TaskLists
      // Work before time integrator indicated by "0" in stage
      ExecuteTaskList(pmesh, "before_timeintegrator", 0);

      // time-integrator tasks for each stage of integrator
      for (int stage=1; stage<=(nexp_stages); ++stage) {
        ExecuteTaskList(pmesh, "before_stagen", stage);
        ExecuteTaskList(pmesh, "stagen", stage);
        ExecuteTaskList(pmesh, "after_stagen", stage);
      }

      // Work after time integrator indicated by "1" in stage
      ExecuteTaskList(pmesh, "after_timeintegrator", 1);

      // Work outside of TaskLists:
      // increment time, ncycle, etc.
      pmesh->time = pmesh->time + pmesh->dt;
      pmesh->ncycle++;
      nmb_updated_ += pmesh->nmb_total;
      npart_updated_ += pmesh->nprtcl_total;
      // load balancing efficiency
      if (global_variable::nranks > 1) {
        int minnmb = std::numeric_limits<int>::max();
        for (int i=0; i<global_variable::nranks; ++i) {
          minnmb = std::min(minnmb, pmesh->nmb_eachrank[i]);
        }
        lb_efficiency_ += static_cast<float>(minnmb*(global_variable::nranks))/
            static_cast<float>(pmesh->nmb_total);
      }

      // Test for/make outputs
      for (auto &out : pout->pout_list) {
        // compare at floating point (32-bit) precision to reduce effect of round off
        float time_32 = static_cast<float>(pmesh->time);
        float next_32 = static_cast<float>(out->out_params.last_time+out->out_params.dt);
        float tlim_32 = static_cast<float>(tlim);
        int &dcycle_ = out->out_params.dcycle;

        if (((out->out_params.dt > 0.0) && ((time_32 >= next_32) && (time_32<tlim_32))) ||
            ((dcycle_ > 0) && ((pmesh->ncycle)%(dcycle_) == 0)) ) {
          out->LoadOutputData(pmesh);
          out->WriteOutputFile(pmesh, pin);
        }
      }

      // AMR
      if (pmesh->adaptive) {pmesh->pmr->AdaptiveMeshRefinement(this, pin);}
      // compute new timestep AFTER all Meshblocks refined/derefined
      pmesh->NewTimeStep(tlim);

      // Update wall clock time if needed.
      if (wall_time > 0.) {
        elapsed_time = UpdateWallClock();
      }
    }  // end while
  }    // end of (time_evolution != tstatic) clause
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::Finalize()
//! \brief Tasks to be performed after execution of Driver, such as making final output
//!  and printing diagnostic messages

void Driver::Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  // cycle through output Types and load data / write files
  //  This design allows for asynchronous outputs to implemented in the future.
  for (auto &out : pout->pout_list) {
    out->LoadOutputData(pmesh);
    out->WriteOutputFile(pmesh, pin);
  }

  // call any problem specific functions to do work after main loop
  if (pmesh->pgen->pgen_final_func != nullptr) {
    (pmesh->pgen->pgen_final_func)(pin, pmesh);
  }

  float exe_time = run_time_.seconds();

  if (time_evolution != TimeEvolution::tstatic) {
#if MPI_PARALLEL_ENABLED
    // Collect number of MeshBlocks communicated during load balancing across all ranks
    if (pmesh->adaptive) {
      MPI_Allreduce(MPI_IN_PLACE, &(pmesh->pmr->nmb_sent_thisrank), 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
#endif
    if (global_variable::my_rank == 0) {
      // Print diagnostic messages related to the end of the simulation
      OutputCycleDiagnostics(pmesh);
      if (pmesh->ncycle == nlim) {
        std::cout << std::endl << "Terminating on cycle limit" << std::endl;
      } else if (pmesh->time >= tlim) {
        std::cout << std::endl << "Terminating on time limit" << std::endl;
      } else {
        std::cout << std::endl << "Terminating on wall clock limit" << std::endl;
      }

      std::cout << "time=" << pmesh->time << " cycle=" << pmesh->ncycle << std::endl;
      std::cout << "tlim=" << tlim << " nlim=" << nlim << std::endl;

      if (pmesh->adaptive) {
        std::cout << std::endl << "Current number of MeshBlocks = " << pmesh->nmb_total
          << std::endl << pmesh->pmr->nmb_created << " MeshBlocks created, "
          << pmesh->pmr->nmb_deleted << " deleted by AMR" << std::endl;
#if MPI_PARALLEL_ENABLED
        std::cout << pmesh->pmr->nmb_sent_thisrank << " communicated for load balancing, "
          <<"load balancing efficiency = " << (lb_efficiency_/pmesh->ncycle) << std::endl;
#endif
      }

      // Calculate and print the zone-cycles/cpu-second
      // Note the need for 64-bit integers since nmb_updated can easily exceed 2^32.
      std::uint64_t zonecycles = nmb_updated_ *
                                 static_cast<uint64_t>(pmesh->NumberOfMeshBlockCells());
      float zcps = static_cast<float>(zonecycles) / exe_time;
      float pups = static_cast<float>(npart_updated_) / exe_time;

      std::cout << std::endl << "MeshBlock-cycles = " << nmb_updated_ << std::endl;
      std::cout << "cpu time used  = " << exe_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
      std::cout << "particle-updates/cpu_second = " << pups << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::OutputCycleDiagnostics()
//! \brief Simple function to print diagnostics every 'ndiag' cycles to stdout

void Driver::OutputCycleDiagnostics(Mesh *pm) {
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    Real elapsed = pwall_clock_->seconds();
    std::cout << "elapsed=" << std::scientific << std::setprecision(dtprcsn) << elapsed
              << " cycle=" << pm->ncycle
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::UpdateWallClock()
//! \brief Update and sync the wall clock across all MPI ranks. This is necessary because
//! the different MPI ranks may 1) initialize their timers at slightly different times,
//! and 2) may reach the end of a loop to update their timers at slightly different times.
//! This may result in a weird problem where one or more ranks have timers that fall
//! slightly below the wall clock time while others determine that it's time to quit.

Real Driver::UpdateWallClock() {
  Real tnow;
  if (global_variable::my_rank == 0) {
    tnow = pwall_clock_->seconds();
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&tnow, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return tnow;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::InitBoundaryValuesAndPrimitives()
//! \brief Sets boundary conditions on conserved and initializes primitives.  Used both
//! on initialization, and when new MBs created with AMR.

void Driver::InitBoundaryValuesAndPrimitives(Mesh *pm) {
  // Note: with MPI, sends on ALL MBs must be complete before receives execute

  // Initialize Z4c
  z4c::Z4c *pz4c = pm->pmb_pack->pz4c;
  if (pz4c != nullptr) {
    (void) pz4c->RestrictU(this, 0);
    (void) pz4c->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) pz4c->SendU(this, 0);
    (void) pz4c->ClearSend(this, -1);
    (void) pz4c->ClearRecv(this, -1);
    (void) pz4c->RecvU(this, 0);
    (void) pz4c->Z4cBoundaryRHS(this, 0);
    (void) pz4c->ApplyPhysicalBCs(this, 0);
    (void) pz4c->Prolongate(this, 0);
  }

  // Initialize HYDRO: ghost zones and primitive variables (everywhere)
  // includes communications for shearing box boundaries
  hydro::Hydro *phydro = pm->pmb_pack->phydro;
  if (phydro != nullptr) {
    // following functions return a TaskStatus, but it is ignored so cast to (void)
    (void) phydro->RestrictU(this, 0);
    (void) phydro->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) phydro->SendU(this, 0);
    (void) phydro->ClearSend(this, -1); // stage = -1 only clear SendU
    (void) phydro->ClearRecv(this, -1); // stage = -1 only clear RecvU
    (void) phydro->RecvU(this, 0);
    (void) phydro->SendU_Shr(this, 0);
    (void) phydro->ClearSend(this, -4); // stage = -4 only clear SendU_Shr
    (void) phydro->ClearRecv(this, -4); // stage = -4 only clear RecvU_Shr
    (void) phydro->RecvU_Shr(this, 0);
    (void) phydro->ApplyPhysicalBCs(this, 0);
    (void) phydro->Prolongate(this, 0);
    (void) phydro->ConToPrim(this, 0);
  }

  // Initialize MHD: ghost zones and primitive variables (everywhere)
  // includes communications for shearing box boundaries
  mhd::MHD *pmhd = pm->pmb_pack->pmhd;
  dyngr::DynGRMHD *pdyngr = pm->pmb_pack->pdyngr;
  if (pmhd != nullptr) {
    (void) pmhd->RestrictU(this, 0);
    (void) pmhd->RestrictB(this, 0);
    (void) pmhd->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) pmhd->SendU(this, 0);
    (void) pmhd->SendB(this, 0);
    (void) pmhd->ClearSend(this, -1); // stage = -1 only clear SendU, SendB
    (void) pmhd->ClearRecv(this, -1); // stage = -1 only clear RecvU, RecvB
    (void) pmhd->RecvU(this, 0);
    (void) pmhd->RecvB(this, 0);
    (void) pmhd->SendU_Shr(this, 0);
    (void) pmhd->SendB_Shr(this, 0);
    (void) pmhd->ClearSend(this, -4); // stage = -4 only clear SendU_Shr, SendB_Shr
    (void) pmhd->ClearRecv(this, -4); // stage = -4 only clear RecvU_Shr, SendB_Shr
    (void) pmhd->RecvU_Shr(this, 0);
    (void) pmhd->RecvB_Shr(this, 0);
    (void) pmhd->ApplyPhysicalBCs(this, 0);
    (void) pmhd->Prolongate(this, 0);
    if (pdyngr == nullptr) {
      (void) pmhd->ConToPrim(this, 0);
    } else {
      if (pz4c != nullptr) {
        (void) pz4c->ConvertZ4cToADM(this, 0);
      }
      (void) pdyngr->ConToPrim(this, 0);
    }
  }

  // Initialize radiation: ghost zones and intensity (everywhere)
  // DOES NOT include communications for shearing box boundaries
  radiation::Radiation *prad = pm->pmb_pack->prad;
  if (prad != nullptr) {
    (void) prad->RestrictI(this, 0);
    (void) prad->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) prad->SendI(this, 0);
    (void) prad->ClearSend(this, -1);
    (void) prad->ClearRecv(this, -1);
    (void) prad->RecvI(this, 0);
    (void) prad->ApplyPhysicalBCs(this, 0);
    (void) prad->Prolongate(this, 0);
  }

  return;
}
