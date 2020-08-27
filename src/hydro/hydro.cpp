//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "hydro/eos/eos.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(Mesh *pm, ParameterInput *pin, int gid) :
  pmesh_(pm), my_mbgid_(gid)
{

  // set EOS option (no default)
  {std::string eqn_of_state   = pin->GetString("hydro","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {
    hydro_eos = HydroEOS::adiabatic;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    hydro_eos = HydroEOS::isothermal;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> eos = '" << eqn_of_state << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // set time-evolution option (no default)
  {std::string evolution_t = pin->GetString("hydro","evolution");
  if (evolution_t.compare("static") == 0) {
    hydro_evol = HydroEvolution::hydro_static;
  } else if (evolution_t.compare("kinematic") == 0) {
    hydro_evol = HydroEvolution::kinematic;
  } else if (evolution_t.compare("dynamic") == 0) {
    hydro_evol = HydroEvolution::hydro_dynamic;
  } else if (evolution_t.compare("none") == 0) {
    hydro_evol = HydroEvolution::no_evolution;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> evolution = '" << evolution_t << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // set reconstruction method option (default PLM)
  {std::string recon_method = pin->GetOrAddString("hydro","reconstruct","plm");
  if (recon_method.compare("dc") == 0) {
    hydro_recon = HydroReconMethod::donor_cell;
  } else if (recon_method.compare("plm") == 0) {
    hydro_recon = HydroReconMethod::piecewise_linear;
  } else if (recon_method.compare("ppm") == 0) {
    hydro_recon = HydroReconMethod::piecewise_parabolic;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> recon = '" << recon_method << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // set Riemann solver option (default depends on EOS and dynamics)
  {std::string rsolver;
  if (hydro_eos == HydroEOS::isothermal) {
    rsolver = pin->GetOrAddString("hydro","rsolver","hlle");
  } else {
    rsolver = pin->GetOrAddString("hydro","rsolver","hllc"); 
  }
  // always make solver=advection for kinematic problems
  if (rsolver.compare("advection") == 0 || hydro_evol == HydroEvolution::kinematic) {
    hydro_rsolver = HydroRiemannSolver::advection;
  } else if (rsolver.compare("llf") == 0) {
    hydro_rsolver = HydroRiemannSolver::llf;
  } else if (rsolver.compare("hlle") == 0) {
    hydro_rsolver = HydroRiemannSolver::hlle;
  } else if (rsolver.compare("hllc") == 0) {
    if (hydro_eos == HydroEOS::isothermal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> rsolver = '" << rsolver
                << "' cannot be used with isothermal EOS" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      hydro_rsolver = HydroRiemannSolver::hllc;
    }
  } else if (rsolver.compare("roe") == 0) {
    hydro_rsolver = HydroRiemannSolver::roe;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> rsolver = '" << rsolver << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // DO NOT ALLOCATE ARRAYS ABOVE THIS POINT as nhydro set here
  // construct EOS object
  switch (hydro_eos) {
    case HydroEOS::adiabatic:
      peos = new AdiabaticHydro(pmesh_, pin, my_mbgid_);
      nhydro = 5;
      break;
    case HydroEOS::isothermal:
      peos = new IsothermalHydro(pmesh_, pin, my_mbgid_);
      nhydro = 4;
      break;
  }

  // allocate memory for conserved variables
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ncells1 = pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng);
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng)) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*(pmb->mb_cells.ng)) : 1;

  u0.SetSize(nhydro, ncells3, ncells2, ncells1);
  w0.SetSize(nhydro, ncells3, ncells2, ncells1);

  // construct boundary values object
  pbvals = new BoundaryValues(pmesh_, pin, pmb->mb_gid, pmb->mb_bcs, nhydro);

  // for time-evolving problems, construct methods, allocate arrays
  if (hydro_evol != HydroEvolution::no_evolution) {

    // construct reconstruction object
    switch (hydro_recon) {
      case HydroReconMethod::donor_cell:
        precon = new DonorCell(pin);
        break;
    }

    // construct Riemann solver object
    switch (hydro_rsolver) {
      case HydroRiemannSolver::advection:
        prsolver = new Advection(pmesh_, pin, my_mbgid_);
        break;
      case HydroRiemannSolver::llf:
        prsolver = new LLF(pmesh_, pin, my_mbgid_);
        break;
      case HydroRiemannSolver::hllc:
        prsolver = new HLLC(pmesh_, pin, my_mbgid_);
        break;
    }

    // allocate registers, flux divergence, scratch arrays
    u1.SetSize(nhydro, ncells3, ncells2, ncells1);
    divf.SetSize(nhydro, ncells3, ncells2, ncells1);
    w_.SetSize(nhydro, ncells1);
    wl_.SetSize(nhydro, ncells1);
    wr_.SetSize(nhydro, ncells1);
    uflux_.SetSize(nhydro, ncells1);
  }

}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroAddTasks
//  \brief

void Hydro::HydroAddTasks(TaskList &tl) {

  TaskID none(0);
  auto hydro_copycons = tl.AddTask(&Hydro::CopyConserved, this, none);
  auto hydro_divflux  = tl.AddTask(&Hydro::HydroDivFlux, this, hydro_copycons);
  auto hydro_update  = tl.AddTask(&Hydro::HydroUpdate, this, hydro_divflux);
  auto hydro_send  = tl.AddTask(&Hydro::HydroSend, this, hydro_update);
  auto hydro_recv  = tl.AddTask(&Hydro::HydroReceive, this, hydro_send);
//  auto phy_bval  = tl.AddTask(&Hydro::PhysicalBoundaryValues, this, hydro_recv);
  auto hydro_con2prim  = tl.AddTask(&Hydro::ConToPrim, this, hydro_recv);
  auto hydro_newdt  = tl.AddTask(&Hydro::NewTimeStep, this, hydro_con2prim);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroSend
//  \brief

TaskStatus Hydro::HydroSend(Driver *pdrive, int stage) 
{
  TaskStatus tstat;
  tstat = pbvals->SendCellCenteredVariables(u0, nhydro);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroReceive
//  \brief

TaskStatus Hydro::HydroReceive(Driver *pdrive, int stage)
{
  TaskStatus tstat;
  tstat = pbvals->ReceiveCellCenteredVariables(u0, nhydro);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ConToPrim
//  \brief

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConservedToPrimitive(u0, w0);
  return TaskStatus::complete;
}

} // namespace hydro
