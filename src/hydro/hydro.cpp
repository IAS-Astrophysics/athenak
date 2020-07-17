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

  // allocate memory for conserved and primitive variables
  u.SetSize(5, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);
  w.SetSize(5, pmb->indx.ncells3, pmb->indx.ncells2, pmb->indx.ncells1);

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

  // set reconstruction method option (default PPM)
  {std::string recon_method = pin->GetOrAddString("hydro","recon","plm");
  if (recon_method.compare("dc")) {
    hydro_recon = HydroReconMethod::donor_cell;
  } else if (recon_method.compare("plm")) {
    hydro_recon = HydroReconMethod::piecewise_linear;
  } else if (recon_method.compare("ppm")) {
    hydro_recon = HydroReconMethod::piecewise_parabolic;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> recon = '" << recon_method << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // set Riemann solver option (default depends on EOS)
  {std::string rsolver;
  if (hydro_eos == HydroEOS::isothermal) {
    rsolver = pin->GetOrAddString("hydro","rsolver","hlle");
  } else {
    rsolver = pin->GetOrAddString("hydro","rsolver","hllc"); 
  }
  if (rsolver.compare("llf")) {
    hydro_rsolver = HydroRiemannSolver::llf;
  } else if (rsolver.compare("hlle")) {
    hydro_rsolver = HydroRiemannSolver::hlle;
  } else if (rsolver.compare("hllc")) {
    hydro_rsolver = HydroRiemannSolver::hllc;
  } else if (rsolver.compare("roe")) {
    hydro_rsolver = HydroRiemannSolver::roe;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> rsolver = '" << rsolver << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  
  // construct EOS object
  switch (hydro_eos) {
    case HydroEOS::adiabatic:
      peos = new AdiabaticHydro(this, pin);
  }

  // construct reconstruction object
//  switch (hydro_recon) {
//    case donor_cell:
//      precon = new DonorCell();
//  }

  // construct Riemann solver object
//  switch (hydro_rsolver) {
//    case llf:
//      prsolver = new LLF();
//////  }

}

} // namespace hydro
