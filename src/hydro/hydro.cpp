//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  u1("cons1",1,1,1,1,1),
  uflx("uflx",1,1,1,1,1)
{
  // (1) Start by selecting physics for this Hydro:

  // construct EOS object (no default)
  {std::string eqn_of_state = pin->GetString("hydro","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {
    peos = new AdiabaticHydro(ppack, pin);
    nhydro = 5;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    peos = new IsothermalHydro(ppack, pin);
    nhydro = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }}

  // Initialize number of scalars
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);

  // read time-evolution option [already error checked in driver constructor]
  std::string evolution_t = pin->GetString("time","evolution");

  // (2) Now initialize memory/algorithms

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &ncells = ppack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
  Kokkos::realloc(u0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new BoundaryValueCC(ppack, pin);
  pbval_u->AllocateBuffersCC((nhydro+nscalars));

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method_ = ReconstructionMethod::dc;

    } else if (xorder.compare("plm") == 0) {
      recon_method_ = ReconstructionMethod::plm;

    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (ncells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << ncells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::ppm;

    } else if (xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (ncells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "WENOZ reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << ncells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::wenoz;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("hydro","rsolver");
    if (rsolver.compare("advect") == 0) {
      if (evolution_t.compare("dynamic") == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with hydrodynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      } else {
        rsolver_method_ = Hydro_RSolver::advect;
      }

    } else  if (evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/rsolver = '" << rsolver
                << "' cannot be used with non-hydrodynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);

    } else if (rsolver.compare("llf") == 0) {
      rsolver_method_ = Hydro_RSolver::llf;

    } else if (rsolver.compare("hllc") == 0) {
      if (peos->eos_data.is_adiabatic) {
        rsolver_method_ = Hydro_RSolver::hllc;
      } else { 
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with isothermal EOS" << std::endl;
        std::exit(EXIT_FAILURE); 
      }  

//    } else if (rsolver.compare("roe") == 0) {
//      rsolver_method_ = Hydro_RSolver::roe;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate second registers, fluxes
    Kokkos::realloc(u1,       nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x1f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x2f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x3f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }

}

//----------------------------------------------------------------------------------------
// destructor
  
Hydro::~Hydro()
{
  delete peos;
  delete pbval_u;
}

} // namespace hydro
