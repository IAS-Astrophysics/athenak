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
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion_neutral.hpp"
#include "driver.hpp"

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
//
// Driver::Execute() invokes the tasklist from stage=1 to stage=ptlist->nstages

Driver::Driver(ParameterInput *pin, Mesh *pmesh) :
  tlim(-1.0), nlim(-1), ndiag(1),
  impl_src("ru",1,1,1,1,1,1) {
  // set time-evolution option (no default)
  {std::string evolution_t = pin->GetString("time","evolution");
  if (evolution_t.compare("static") == 0) {
    time_evolution = TimeEvolution::tstatic;  // cannot use 'static' (keyword);
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
      //! - RK4()4[2S] from Table 2 of Ketcheson (2010)
      //! - Non-SSP, explicit four-stage, fourth-order RK
      //! - Stability properties are similar to classical (non-SSP) RK4
      //!   (but ~2x L2 principal error norm).
      //! - Refer to Colella (2011) for linear stability analysis of constant
      //!   coeff. advection of classical RK4 + 4th or 1st order (limiter engaged) fluxes

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
      gam0[0] = 0.0;
      gam1[0] = 1.0;
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
         << "Valid choices are [rk1,rk2,rk3,imex2,imex3]." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// Driver::Initialize()
// Tasks to be performed before execution of Driver, such as setting ghost zones (BCs),
//  outputting ICs, and computing initial time step

void Driver::Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  //---- Step 1.  Set conserved variables in ghost zones for all physics
  // Note: with MPI, sends on ALL MBs must be complete before receives execute

  // Initialize HYDRO: ghost zones and primitive variables (everywhere)
  hydro::Hydro *phydro = pmesh->pmb_pack->phydro;
  if (phydro != nullptr) {
    // following functions return a TaskStatus, but it is ignored so cast to (void)
    (void) phydro->RestrictU(this, 0);
    (void) phydro->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) phydro->SendU(this, 0);
    (void) phydro->ClearSend(this, -1);
    (void) phydro->ClearRecv(this, -1);
    (void) phydro->RecvU(this, 0);
    (void) phydro->ApplyPhysicalBCs(this, 0);
    (void) phydro->ConToPrim(this, 0);
  }

  // Initialize MHD: ghost zones and primitive variables (everywhere)
  // Note this requires communicating BOTH u and B
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  if (pmhd != nullptr) {
    (void) pmhd->RestrictU(this, 0);
    (void) pmhd->RestrictB(this, 0);
    (void) pmhd->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) pmhd->SendU(this, 0);
    (void) pmhd->SendB(this, 0);
    (void) pmhd->ClearSend(this, -1);
    (void) pmhd->ClearRecv(this, -1);
    (void) pmhd->RecvU(this, 0);
    (void) pmhd->RecvB(this, 0);
    (void) pmhd->ApplyPhysicalBCs(this, 0);
    (void) pmhd->ConToPrim(this, 0);
  }

  //---- Step 2.  Compute first time step (if problem involves time evolution

  if (time_evolution != TimeEvolution::tstatic) {
    if (phydro != nullptr) {
      (void) pmesh->pmb_pack->phydro->NewTimeStep(this, nexp_stages);
    }
    if (pmhd != nullptr) {
      (void) pmesh->pmb_pack->pmhd->NewTimeStep(this, nexp_stages);
    }
    pmesh->NewTimeStep(tlim);
  }

  //---- Step 3.  Cycle through output Types and load data / write files.

  for (auto &out : pout->pout_list) {
    out->LoadOutputData(pmesh);
    out->WriteOutputFile(pmesh, pin);
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
    int nmb = pmesh->pmb_pack->nmb_thispack;
    auto &indcs = pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(impl_src, nimp_stages, nmb, 6, ncells3, ncells2, ncells1);
  }

  return;
}


//----------------------------------------------------------------------------------------
// Driver::Execute()
// \brief Executes all relevant task lists over all MeshBlockPacks.  For static
// (non-evolving) problems, currently implemented task lists are:
//  (1) TODO
// For dynamic (time-evolving) problems, currently implemented task lists are:
//  (1) operator split physics (operator_split_tl)
//  (2) each stage of both explicit and ImEx RK integrators (start_tl, run_tl, end_tl)
//  [Note for ImEx integrators, the first two fully implicit updates should be performed
//  at the start of the first stage.]

void Driver::Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list...\n" << std::endl;
  }

  if (time_evolution == TimeEvolution::tstatic) {
    // TODO(@user): add work for time static problems here
  } else {
    while ((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0)) {
      if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}
      int npacks = 1;  // TODO(@user): extend for multiple MeshBlockPacks
      MeshBlockPack* pmbp = pmesh->pmb_pack;
      //---------------------------------------
      // (1) Do *** operator split *** TaskList
      {for (int p=0; p<npacks; ++p) {
        if (!(pmbp->operator_split_tl.Empty())) {pmbp->operator_split_tl.Reset();}
      }
      int npack_left = npacks;
      while (npack_left > 0) {
        if (pmbp->operator_split_tl.Empty()) {
          npack_left--;
        } else {
          if (!pmbp->operator_split_tl.IsComplete()) {
            // note 2nd argument to DoAvailable (stage) is not used, set to 0
            auto status = pmbp->operator_split_tl.DoAvailable(this, 0);
            if (status == TaskListStatus::complete) { npack_left--; }
          }
        }
      }} // extra brace to enclose scope

      //--------------------------------------------------------------
      // (2) Do *** explicit and ImEx RK time-integrator *** TaskLists
      for (int stage=1; stage<=(nexp_stages); ++stage) {
        // (2a) StageStart Tasks
        // tasks that must be completed over all MBPacks at start of each explicit stage
        {for (int p=0; p<npacks; ++p) {
          if (!(pmbp->start_tl.Empty())) {pmbp->start_tl.Reset();}
        }
        int npack_left = npacks;
        while (npack_left > 0) {
          if (pmbp->start_tl.Empty()) {
            npack_left--;
          } else {
            if (!pmbp->start_tl.IsComplete()) {
              auto status = pmbp->start_tl.DoAvailable(this, stage);
              if (status == TaskListStatus::complete) { npack_left--; }
            }
          }
        }} // extra brace to enclose scope

        // (2b) StageRun Tasks
        // tasks in each explicit stage
        {for (int p=0; p<npacks; ++p) {
          if (!(pmbp->run_tl.Empty())) {pmbp->run_tl.Reset();}
        }
        int npack_left = npacks;
        while (npack_left > 0) {
          if (pmbp->run_tl.Empty()) {
            npack_left--;
          } else {
            if (!pmbp->run_tl.IsComplete()) {
              auto status = pmbp->run_tl.DoAvailable(this, stage);
              if (status == TaskListStatus::complete) { npack_left--; }
            }
          }
        }} // extra brace to enclose scope

        // (2c) StageEnd Tasks
        // tasks that must be completed over all MBs at end of each explicit stage
        {for (int p=0; p<npacks; ++p) {
          if (!(pmbp->end_tl.Empty())) {pmbp->end_tl.Reset();}
        }
        int npack_left = npacks;
        while (npack_left > 0) {
          if (pmbp->end_tl.Empty()) {
            npack_left--;
          } else {
            if (!pmbp->end_tl.IsComplete()) {
              auto status = pmbp->end_tl.DoAvailable(this, stage);
              if (status == TaskListStatus::complete) { npack_left--; }
            }
          }
        }} // extra brace to enclose scope
      } // end of loop over stages

      //-------------------------------
      // (3) Work outside of TaskLists:
      // increment time, ncycle, etc. Compute new timestep, make outputs
      pmesh->time = pmesh->time + pmesh->dt;
      pmesh->ncycle++;
      nmb_updated_ += pmesh->nmb_total;
      pmesh->NewTimeStep(tlim);

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
    }
  }  // end of (time_evolution != tstatic) clause

  return;
}

//----------------------------------------------------------------------------------------
// Driver::Finalize()
// Tasks to be performed after execution of Driver, such as making final output and
// printing diagnostic messages

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
          static_cast<std::uint64_t>(pmesh->NumberOfMeshBlockCells());
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

void Driver::OutputCycleDiagnostics(Mesh *pm) {
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    std::cout << "cycle=" << pm->ncycle << std::scientific << std::setprecision(dtprcsn)
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}
