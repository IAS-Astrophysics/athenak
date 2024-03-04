//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//! \brief implementation of ShearingBox class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ShearingBox::ShearingBox(MeshBlockPack *ppack, ParameterInput *pin, int nvar) :
    qshear(0.0),
    omega0(0.0),
    maxjshift(1),
    pmy_pack(ppack) {
  // read shear parameters
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");

  // estimate maximum integer shift in x2-direction for orbital advection
  Real xmin = fabs(ppack->pmesh->mesh_size.x1min);
  Real xmax = fabs(ppack->pmesh->mesh_size.x1max);
  maxjshift = static_cast<int>((ppack->pmesh->cfl_no)*std::max(xmin,xmax)) + 1;
std::cout << "maxjshift=" << maxjshift<<std::endl;

#if MPI_PARALLEL_ENABLED
  // For shearing box, communication is only with x2-face neighbors
  // initialize vectors of MPI request in 2 elements of fixed length arrays
  for (int n=0; n<2; ++n) {
    int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
    sendbuf_orb[n].vars_req = new MPI_Request[nmb];
    recvbuf_orb[n].vars_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf_orb[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf_orb[n].vars_req[m] = MPI_REQUEST_NULL;
    }
  }

  // create unique communicators for orbital advection
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_orb);
#endif

  // Allocate send/recv buffers for orbital advection
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells3 = indcs.nx3;
  int ncells2 = indcs.ng + maxjshift;
  int ncells1 = indcs.nx1;
  for (int n=0; n<2; ++n) {
    Kokkos::realloc(sendbuf_orb[n].vars,nmb,nvar,ncells3,ncells2,ncells1);
    Kokkos::realloc(recvbuf_orb[n].vars,nmb,nvar,ncells3,ncells2,ncells1);
  }
}

//----------------------------------------------------------------------------------------
// destructor

ShearingBox::~ShearingBox() {
}
