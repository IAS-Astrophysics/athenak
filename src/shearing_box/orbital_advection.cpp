//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.cpp
//! \brief constructor for OrbitalAdvection abstract base class.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! OrbitalAdvection base class constructor
//! Called by Hydro and MHD constructors, so cannot access any data inside Hydro/MHD
//! classes as it may not be properly allocated yet.

OrbitalAdvection::OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin) :
    maxjshift(1),
    pmy_pack(ppack) {
  // First some error checks
std::cout << "constructing..." << std::endl;

  if (pin->DoesBlockExist("hydro")) {
    std::string xorder = pin->GetString("hydro","reconstruct");
    if (xorder.compare("dc") != 0 &&
        xorder.compare("plm") != 0 &&
        xorder.compare("wenoz") != 0) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Only dc, plm, or wenoz reconstruction can be used with the "
                << "shearing box and orbital advection" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pin->DoesBlockExist("mhd")) {
    std::string xorder = pin->GetString("mhd","reconstruct");
    if (xorder.compare("dc") != 0 &&
        xorder.compare("plm") != 0 &&
        xorder.compare("wenoz") != 0) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "Only dc, plm, or wenoz reconstruction can be used with the "
                << "shearing box and orbital advection" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // estimate maximum integer shift in x2-direction for orbital advection
  Real xmin = fabs(ppack->pmesh->mesh_size.x1min);
  Real xmax = fabs(ppack->pmesh->mesh_size.x1max);
  maxjshift = static_cast<int>((ppack->pmesh->cfl_no)*std::max(xmin,xmax)) + 1;

#if MPI_PARALLEL_ENABLED
  // For orbital advection, communication is only with x2-face neighbors
  // initialize vectors of MPI request in 2 elements of fixed length arrays
  for (int n=0; n<2; ++n) {
    int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
    }
  }
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_orb_advect);
#endif
}

//----------------------------------------------------------------------------------------
// OrbitalAdvection base class destructor

OrbitalAdvection::~OrbitalAdvection() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] recvbuf[n].vars_req;
  }
#endif
}
