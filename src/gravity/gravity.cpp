//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravity.cpp
//! \brief implementation of functions in class Gravity

// C headers

// C++ headers
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../multigrid/multigrid.hpp"

namespace gravity { // NOLINT (build/namespace)
//! constructor, initializes data structures and parameters
//-------------------------------------------------------------------------------------
//! \fn Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Gravity constructor
Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin):
    pmy_pack(pmbp),
    phi("phi",1,1,1,1,1),
    coarse_phi("coarse",1,1,1,1,1),
    def("defect",1,1,1,1,1),
    four_pi_G(-1.0),
    output_defect(false),
    fill_ghost(false) {
    
    four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G",-1.0);
    output_defect = pin->GetOrAddBoolean("gravity", "output_defect", false);
    fill_ghost = pin->GetOrAddBoolean("gravity", "fill_ghost", true);

    if (four_pi_G == 0.0) {
        std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
        exit(EXIT_FAILURE);
    }

    // create multigrid driver/solver
    // The driver allocates multigrid instances for root level and meshblock levels
    pmgd = new MGGravityDriver(pmbp, pin);
    

    // Enroll CellCenteredBoundaryVariable object
    //gbvar.bvar_index = pmb->pbval->bvars.size();
    //pmb->pbval->bvars.push_back(&gbvar);
    //pmb->pbval->pgbvar = &gbvar;
    int nmb = pmy_pack->nmb_thispack;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(phi, nmb, 1, ncells3, ncells2, ncells1);
    Kokkos::realloc(def, nmb, 1, ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn Gravity::~Gravity()
//! \brief Gravity destructor
Gravity::~Gravity() {
    delete pmg;
}
} // namespace gravity
