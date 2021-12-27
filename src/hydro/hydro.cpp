//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of Hydro class constructor and assorted other functions

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  coarse_u0("ccons",1,1,1,1,1),
  u1("cons1",1,1,1,1,1),
  uflx("uflx",1,1,1,1,1)
{
  // (1) Start by selecting physics for this Hydro:

  // Check for relativistic dynamics
  is_special_relativistic = pin->GetOrAddBoolean("hydro","special_rel",false);
  is_general_relativistic = pin->GetOrAddBoolean("hydro","general_rel",false);
  if (is_special_relativistic && is_general_relativistic) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Cannot specify both SR and GR at same time" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) construct EOS object (no default)
  {std::string eqn_of_state = pin->GetString("hydro","eos");

  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (is_special_relativistic){
      peos = new IdealSRHydro(ppack, pin);
    } else if (is_general_relativistic){
      peos = new IdealGRHydro(ppack, pin);
    } else {
      peos = new IdealHydro(ppack, pin);
    }
    nhydro = 5;

  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (is_special_relativistic || is_general_relativistic){
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "<hydro>/eos = isothermal cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalHydro(ppack, pin);
      nhydro = 4;
    }

  // EOS string not recognized
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro>/eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }}

  // (3) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);

  // Viscosity (only constructed if needed)
  if (pin->DoesParameterExist("hydro","viscosity")) {
    pvisc = new Viscosity("hydro", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Source terms (constructor parses input file to initialize only srcterms needed)
  psrc = new SourceTerms("hydro", ppack, pin);

  // (4) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(u0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nhydro+nscalars), nccells3, nccells2, nccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new BValCC(ppack, pin);
  pbval_u->AllocateBuffersCC((nhydro+nscalars));

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method = ReconstructionMethod::ppm;
    } else if (xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "WENOZ reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method = ReconstructionMethod::wenoz;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("hydro","rsolver");

    // Special relativistic solvers
    if (is_special_relativistic) {
      if (rsolver.compare("llf") == 0) {
        rsolver_method = Hydro_RSolver::llf_sr;
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle_sr;
      } else if (rsolver.compare("hllc") == 0) {
        rsolver_method = Hydro_RSolver::hllc_sr;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for SR dynamics" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic solvers
    } else if (is_general_relativistic){
      if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle_gr;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for GR dynamics" << std::endl;
        std::exit(EXIT_FAILURE); 
      }

    // Non-relativistic solvers
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        if (evolution_t.compare("dynamic") == 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = '" << rsolver
                    << "' cannot be used with hydrodynamic problems" << std::endl;
          std::exit(EXIT_FAILURE);
        } else {
          rsolver_method = Hydro_RSolver::advect;
        }
      // only advect RS can be used with non-dynamic problems; print error otherwise
      } else if (evolution_t.compare("dynamic") != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with non-hydrodynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      // LLF solver
      } else if (rsolver.compare("llf") == 0) {
        rsolver_method = Hydro_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle;
      // HLLC solver
      } else if (rsolver.compare("hllc") == 0) {
        if (peos->eos_data.is_ideal) {
          rsolver_method = Hydro_RSolver::hllc;
        } else { 
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = hllc cannot be used with "
                    << "isothermal EOS" << std::endl;
          std::exit(EXIT_FAILURE); 
        }  
      // Roe solver
      } else if (rsolver.compare("roe") == 0) {
        rsolver_method = Hydro_RSolver::roe;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << std::endl;
        std::exit(EXIT_FAILURE); 
      }
    }}

    // allocate second registers, fluxes
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u1,       nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x1f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x2f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x3f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }

  // (5) initialize metric (GR only)
  if (is_general_relativistic) {pmy_pack->pcoord->InitMetric(pin);}
}

//----------------------------------------------------------------------------------------
// destructor
  
Hydro::~Hydro()
{
  delete peos;
  delete pbval_u;
  if (pvisc != nullptr) {delete pvisc;}
  if (psrc != nullptr) {delete psrc;}
}

//----------------------------------------------------------------------------------------
//! \fn EnrollBoundaryFunction(BoundaryFace dir, BValFunc my_bc)
//! \brief Enroll a user-defined boundary function

void Hydro::EnrollBoundaryFunction(BoundaryFace dir, HydroBoundaryFnPtr my_bcfunc) {
  if (dir < 0 || dir > 5) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "EnrollBoundaryFunction called on bndry=" << dir << " not valid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->mesh_bcs[dir] != BoundaryFlag::user) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Boundary condition flag must be set to 'user' in the <mesh> block in input"
        << " file to use user-enrolled BCs on bndry=" << dir << std::endl;
    std::exit(EXIT_FAILURE);
  }
  HydroBoundaryFunc[static_cast<int>(dir)] = my_bcfunc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn CheckUserBoundaries()
//! \brief checks if user boundary functions are correctly enrolled
//! This compatibility check is performed in the Driver after calling ProblemGenerator()

void Hydro::CheckUserBoundaries() {
  for (int i=0; i<6; i++) {
    if (pmy_pack->pmesh->mesh_bcs[i] == BoundaryFlag::user) {
      if (HydroBoundaryFunc[i] == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "User-defined boundary function is specified in input "
                  << " file but boundary function not enrolled; bndry=" << i << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  return;
}


} // namespace hydro
