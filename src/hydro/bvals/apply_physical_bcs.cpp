//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file apply_physical_bcs.cpp
//  \brief

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::ApplyPhysicalBCsHydro()
// \brief Apply physical boundary conditions to faces of MB when they are at the edge of
// the computational domain

TaskStatus Hydro::HydroApplyPhysicalBCs(Driver* pdrive, int stage)
{
  // loop over all MeshBlocks in this MeshBlockPack
  int nmb = pmy_pack->nmb_thispack;

  for (int m=0; m<nmb; ++m) {
    // apply physical boundaries to inner_x1
    switch (pmy_pack->pmb->mb_bcs(m,BoundaryFace::inner_x1)) {
      case static_cast<int>(BoundaryFlag::reflect):
        ReflectInnerX1(m);
        break;
      case static_cast<int>(BoundaryFlag::outflow):
        OutflowInnerX1(m);
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x1
    switch (pmy_pack->pmb->mb_bcs(m,BoundaryFace::outer_x1)) {
      case static_cast<int>(BoundaryFlag::reflect):
        ReflectOuterX1(m);
        break;
      case static_cast<int>(BoundaryFlag::outflow):
        OutflowOuterX1(m);
        break;
      default:
        break;
    }
  }
  if (!(pmy_pack->pmesh->nx2gt1)) return TaskStatus::complete;

  for (int m=0; m<nmb; ++m) {
    // apply physical bounaries to inner_x2
    switch (pmy_pack->pmb->mb_bcs(m,BoundaryFace::inner_x2)) {
      case static_cast<int>(BoundaryFlag::reflect):
        ReflectInnerX2(m);
        break;
      case static_cast<int>(BoundaryFlag::outflow):
        OutflowInnerX2(m);
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x1
    switch (pmy_pack->pmb->mb_bcs(m,BoundaryFace::outer_x2)) {
      case static_cast<int>(BoundaryFlag::reflect):
        ReflectOuterX2(m);
        break;
      case static_cast<int>(BoundaryFlag::outflow):
        OutflowOuterX2(m);
        break;
      default:
        break;
    }
  }
  if (!(pmy_pack->pmesh->nx3gt1)) return TaskStatus::complete;

  for (int m=0; m<nmb; ++m) {
    // apply physical bounaries to inner_x3
    switch (pmy_pack->pmb->mb_bcs(m,BoundaryFace::inner_x3)) {
      case static_cast<int>(BoundaryFlag::reflect):
        ReflectInnerX3(m);
        break;
      case static_cast<int>(BoundaryFlag::outflow):
        OutflowInnerX3(m);
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x3
    switch (pmy_pack->pmb->mb_bcs(m,BoundaryFace::outer_x3)) {
      case static_cast<int>(BoundaryFlag::reflect):
        ReflectOuterX3(m);
        break;
      case static_cast<int>(BoundaryFlag::outflow):
        OutflowOuterX3(m);
        break;
      default:
        break;
    }
  }
  return TaskStatus::complete;
}
} // namespace hydro
