//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//! \brief implementation of MHD class constructor and assorted functions

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  b0("B_fc",1,1,1,1),
  bcc0("B_cc",1,1,1,1,1),
  coarse_u0("ccons",1,1,1,1,1),
  coarse_b0("cB_fc",1,1,1,1),
  u1("cons1",1,1,1,1,1),
  b1("B_fc1",1,1,1,1),
  uflx("uflx",1,1,1,1,1),
  efld("efld",1,1,1,1),
  e3x1("e3x1",1,1,1,1),
  e2x1("e2x1",1,1,1,1),
  e1x2("e1x2",1,1,1,1),
  e3x2("e3x2",1,1,1,1),
  e2x3("e2x3",1,1,1,1),
  e1x3("e1x3",1,1,1,1),
  e1_cc("e1_cc",1,1,1,1),
  e2_cc("e2_cc",1,1,1,1),
  e3_cc("e3_cc",1,1,1,1)
{
  // (1) Start by selecting physics for this MHD:

  // Check for relativistic dynamics
  is_special_relativistic = pin->GetOrAddBoolean("mhd","special_rel",false);
  is_general_relativistic = pin->GetOrAddBoolean("mhd","general_rel",false);
  if (is_special_relativistic && is_general_relativistic) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Cannot specify both SR and GR at same time" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) construct EOS object (no default)
  {std::string eqn_of_state = pin->GetString("mhd","eos");

  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (is_special_relativistic){
      peos = new IdealSRMHD(ppack, pin);
    } else if (is_general_relativistic){
      peos = new IdealGRMHD(ppack, pin);
    } else {
      peos = new IdealMHD(ppack, pin);
    }
    nmhd = 5;

  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (is_special_relativistic || is_general_relativistic){
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "<mhd> eos = isothermal cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalMHD(ppack, pin);
      nmhd = 4;
    }

  // EOS string not recognized
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<mhd> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }}

  // (3) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);

  // Viscosity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","viscosity")) {
    pvisc = new Viscosity("mhd", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Resistivity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","ohmic_resistivity")) {
    presist = new Resistivity(ppack, pin);
  } else {
    presist = nullptr;
  }

  // Thermal conduction (only constructed if needed)
  if (pin->DoesParameterExist("mhd","conductivity")) {
    pcond = new Conduction("mhd", ppack, pin);
  } else {
    pcond = nullptr;
  }

  // Source terms (constructor parses input file to initialize only srcterms needed)
  psrc = new SourceTerms("mhd", ppack, pin);

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
  Kokkos::realloc(u0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);

  // allocate memory for face-centered and cell-centered magnetic fields
  Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nmhd+nscalars), nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_b0.x1f, nmb, nccells3, nccells2, nccells1+1);
    Kokkos::realloc(coarse_b0.x2f, nmb, nccells3, nccells2+1, nccells1);
    Kokkos::realloc(coarse_b0.x3f, nmb, nccells3+1, nccells2, nccells1);
  }
  
  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new BoundaryValuesCC(ppack, pin);
  pbval_u->InitializeBuffers((nmhd+nscalars));

  // allocate boundary buffers for face-centered magnetic field
  pbval_b = new BoundaryValuesFC(ppack, pin);
  pbval_b->InitializeBuffers(3);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
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
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("mhd","rsolver");

    // Special relativistic solvers
    if (is_special_relativistic) {
      if (rsolver.compare("llf") == 0) {
        rsolver_method = MHD_RSolver::llf_sr;
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle_sr;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                  << " for SR dynamics" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic solvers
    } else if (is_general_relativistic){
      if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle_gr;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                  << " for GR dynamics" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic solvers
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        if (evolution_t.compare("dynamic") == 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd>/rsolver = '" << rsolver
                    << "' cannot be used with dynamic problems" << std::endl;
          std::exit(EXIT_FAILURE);
        } else {
          rsolver_method = MHD_RSolver::advect;
        }
      // only advect RS can be used with non-dynamic problems; print error otherwise
      } else  if (evolution_t.compare("dynamic") != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver
                  << "' cannot be used with non-dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      // LLF solver
      } else if (rsolver.compare("llf") == 0) {
        rsolver_method = MHD_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle;
      // HLLD solver
      } else if (rsolver.compare("hlld") == 0) {
          rsolver_method = MHD_RSolver::hlld;
      // Roe solver
//      } else if (rsolver.compare("roe") == 0) {
//        rsolver_method = MHD_RSolver::roe;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << std::endl;
        std::exit(EXIT_FAILURE); 
      }
    }}

    // allocate second registers
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);

    // allocate fluxes, electric fields
    Kokkos::realloc(uflx.x1f, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1+1);
    Kokkos::realloc(uflx.x2f, nmb, (nmhd+nscalars), ncells3, ncells2+1, ncells1);
    Kokkos::realloc(uflx.x3f, nmb, (nmhd+nscalars), ncells3+1, ncells2, ncells1);
    Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
    Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
    Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);

    // allocate scratch arrays for face- and cell-centered E used in CornerE
    Kokkos::realloc(e3x1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2x1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1x2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e3x2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2x3, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1x3, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1_cc, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2_cc, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e3_cc, nmb, ncells3, ncells2, ncells1);
  }

  // (5) initialize metric (GR only)
  if (is_general_relativistic) {pmy_pack->pcoord->InitMetric(pin);}
}

//----------------------------------------------------------------------------------------
// destructor
  
MHD::~MHD()
{
  delete peos;
  delete pbval_u;
  delete pbval_b;
  if (pvisc != nullptr) {delete pvisc;}
  if (presist!= nullptr) {delete presist;}
  if (pcond != nullptr) {delete pcond;}
  if (psrc!= nullptr) {delete psrc;}
}

//----------------------------------------------------------------------------------------
//! \fn EnrollBoundaryFunction(BoundaryFace dir, BValFunc my_bc)
//! \brief Enroll a user-defined boundary function

void MHD::EnrollBoundaryFunction(BoundaryFace dir, MHDBoundaryFnPtr my_bcfunc) {
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
  MHDBoundaryFunc[static_cast<int>(dir)] = my_bcfunc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn CheckUserBoundaries()
//! \brief checks if user boundary functions are correctly enrolled
//! This compatibility check is performed in the Driver after calling ProblemGenerator()

void MHD::CheckUserBoundaries() {
  for (int i=0; i<6; i++) {
    if (pmy_pack->pmesh->mesh_bcs[i] == BoundaryFlag::user) {
      if (MHDBoundaryFunc[i] == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "User-defined boundary function is specified in input "
                  << " file but boundary function not enrolled; bndry=" << i << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  return;
}

} // namespace mhd
