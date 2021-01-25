//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//  \brief implementation of MHD class constructor and assorted functions

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "mhd/eos/mhd_eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  b0("Bfield",1,1,1,1),
  u1("cons1",1,1,1,1,1),
  b1("Bfield1",1,1,1,1),
  divf("divF",1,1,1,1,1),
  uflx_x1face("uflx_x1face",1,1,1),
  uflx_x2face("uflx_x2face",1,1,1),
  uflx_x3face("uflx_x3face",1,1,1)
{
  // construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("mhd","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {
    peos = new AdiabaticMHD(ppack, pin);
    nmhd = 5;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    peos = new IsothermalMHD(ppack, pin);
    nmhd = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<mhd> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Initialize number of scalars
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);

  // set time-evolution option (default=dynamic) [error checked in driver constructor]
  std::string evolution_t = pin->GetOrAddString("mhd","evolution","dynamic");

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &ncells = ppack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
  Kokkos::realloc(u0, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
  
  // allocate memory for boundary buffers
  pbvals = new BoundaryValues(ppack, pin);
  pbvals->AllocateBuffersCC((nhydro+nscalars));

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method_ = ReconstructionMethod::dc;

    } else if (xorder.compare("plm") == 0) {
      recon_method_ = ReconstructionMethod::plm;

    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (pmb->mb_cells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << pmb->mb_cells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::ppm;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("mhd","rsolver");
    if (rsolver.compare("advection") == 0) {
      if (evolution_t.compare("dynamic") == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver
                  << "' cannot be used with dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      } else {
        rsolver_method_ = MHD_RSolver::advect;
      }

    } else  if (evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/rsolver = '" << rsolver
                << "' cannot be used with non-dynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);

    } else if (rsolver.compare("llf") == 0) {
      rsolver_method_ = MHD_RSolver::llf;

//    } else if (rsolver.compare("hlld") == 0) {
//      if (peos->eos_data.is_adiabatic) {
//        rsolver_method_ = MHD_RSolver::hlld;
//      } else { 
//        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
//                  << std::endl << "<mhd>/rsolver = '" << rsolver
//                  << "' cannot be used with isothermal EOS" << std::endl;
//        std::exit(EXIT_FAILURE); 
//        }  

//    } else if (rsolver.compare("roe") == 0) {
//      rsolver_method_ = MHD_RSolver::roe;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate registers, flux divergence, scratch arrays for time-dep probs
    Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(divf,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);
  }
}

} // namespace mhd
