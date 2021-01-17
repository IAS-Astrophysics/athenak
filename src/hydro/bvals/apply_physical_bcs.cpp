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
  for (auto &mb : pmy_pack->mblocks) {
    // apply physical boundaries to inner_x1
    switch (mb.mb_bcs[BoundaryFace::inner_x1]) {
      case BoundaryFlag::reflect:
        ReflectInnerX1();
        break;
      case BoundaryFlag::outflow:
        OutflowInnerX1();
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x1
    switch (mb.mb_bcs[BoundaryFace::outer_x1]) {
      case BoundaryFlag::reflect:
        ReflectOuterX1();
        break;
      case BoundaryFlag::outflow:
        OutflowOuterX1();
        break;
      default:
        break;
    }
  }
  if (!(pmy_pack->pmesh->nx2gt1)) return TaskStatus::complete;

  for (auto &mb : pmy_pack->mblocks) {
    // apply physical bounaries to inner_x2
    switch (mb.mb_bcs[BoundaryFace::inner_x2]) {
      case BoundaryFlag::reflect:
        ReflectInnerX2();
        break;
      case BoundaryFlag::outflow:
        OutflowInnerX2();
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x1
    switch (mb.mb_bcs[BoundaryFace::outer_x2]) {
      case BoundaryFlag::reflect:
        ReflectOuterX2();
        break;
      case BoundaryFlag::outflow:
        OutflowOuterX2();
        break;
      default:
        break;
    }
  }
  if (!(pmy_pack->pmesh->nx3gt1)) return TaskStatus::complete;

  for (auto &mb : pmy_pack->mblocks) {
    // apply physical bounaries to inner_x3
    switch (mb.mb_bcs[BoundaryFace::inner_x3]) {
      case BoundaryFlag::reflect:
        ReflectInnerX3();
        break;
      case BoundaryFlag::outflow:
        OutflowInnerX3();
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x3
    switch (mb.mb_bcs[BoundaryFace::outer_x3]) {
      case BoundaryFlag::reflect:
        ReflectOuterX3();
        break;
      case BoundaryFlag::outflow:
        OutflowOuterX3();
        break;
      default:
        break;
    }
  }
  return TaskStatus::complete;
}
} // namespace hydro
