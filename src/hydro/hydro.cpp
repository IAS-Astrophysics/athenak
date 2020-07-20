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
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "hydro/eos/eos.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin) : pmy_mblock(pmb) {

  // set EOS option (no default)
  {std::string eqn_of_state   = pin->GetString("hydro","eos");
  if (eqn_of_state.compare("adiabatic")) {
    hydro_eos = HydroEOS::adiabatic;
  } else if (eqn_of_state.compare("isothermal")) {
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
  {std::string recon_method = pin->GetOrAddString("hydro","recon","plm");
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
    hydro_rsolver = HydroRiemannSolver::hllc;
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
      peos = new AdiabaticHydro(this, pin);
      nhydro = 5;
    case HydroEOS::isothermal:
      peos = new IsothermalHydro(this, pin);
      nhydro = 4;
  }

  // allocate memory for conserved variables
  u0.SetSize(nhydro, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);

  // for time-evolving problems, construct methods, allocate arrays
  if (hydro_evol != HydroEvolution::no_evolution) {
    // construct reconstruction object
    switch (hydro_recon) {
      case HydroReconMethod::donor_cell:
        precon = new DonorCell(pin);
    }

    // construct Riemann solver object
    switch (hydro_rsolver) {
      case HydroRiemannSolver::advection:
        prsolver = new Advection(this, pin);
      case HydroRiemannSolver::llf:
        prsolver = new LLF(this, pin);
    }

    // allocate registers, flux divergence, scratch arrays
    u1.SetSize(nhydro, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);
    divf.SetSize(nhydro, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);
    w_.SetSize(nhydro, pmb->indx.ncells1);
    wl_.SetSize(nhydro, pmb->indx.ncells1);
    wr_.SetSize(nhydro, pmb->indx.ncells1);
    uflux_.SetSize(nhydro, pmb->indx.ncells1);
  }

}

} // namespace hydro
