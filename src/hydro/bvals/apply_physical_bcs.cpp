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
#include "eos/eos.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::ApplyPhysicalBCsHydro()
// \brief Apply physical boundary conditions for all Hydro variables at faces of MB which
//  are at the edge of the computational domain

TaskStatus Hydro::ApplyPhysicalBCs(Driver* pdrive, int stage)
{
  // loop over all MeshBlocks in this MeshBlockPack
  int nmb = pmy_pack->nmb_thispack;
  auto &mbd = pmy_pack->pcoord->mbdata;
  auto &eos = peos->eos_data;
  auto u0_ = u0;

  for (int m=0; m<nmb; ++m) {
    // apply physical boundaries to inner_x1
    switch (pmy_pack->pmb->mbbcs(m,BoundaryFace::inner_x1)) {
      case BoundaryFlag::reflect:
        ReflectInnerX1(m);
        break;
      case BoundaryFlag::outflow:
        OutflowInnerX1(m);
        break;
      case BoundaryFlag::periodic:
        if (pmy_pack->pmesh->shearing_periodic) ShearInnerX1(m);
        break;
      case BoundaryFlag::user:
        pmy_pack->pmesh->BoundaryFunc[BoundaryFace::inner_x1](m, mbd, eos, u0_);
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x1
    switch (pmy_pack->pmb->mbbcs(m,BoundaryFace::outer_x1)) {
      case BoundaryFlag::reflect:
        ReflectOuterX1(m);
        break;
      case BoundaryFlag::outflow:
        OutflowOuterX1(m);
        break;
      case BoundaryFlag::periodic:
        if (pmy_pack->pmesh->shearing_periodic) ShearOuterX1(m);
        break;
      case BoundaryFlag::user:
        pmy_pack->pmesh->BoundaryFunc[BoundaryFace::outer_x1](m, mbd, eos, u0_);
        break;
      default:
        break;
    }
  }
  if (pmy_pack->pmesh->one_d) return TaskStatus::complete;

  for (int m=0; m<nmb; ++m) {
    // apply physical bounaries to inner_x2
    switch (pmy_pack->pmb->mbbcs(m,BoundaryFace::inner_x2)) {
      case BoundaryFlag::reflect:
        ReflectInnerX2(m);
        break;
      case BoundaryFlag::outflow:
        OutflowInnerX2(m);
        break;
      case BoundaryFlag::user:
        pmy_pack->pmesh->BoundaryFunc[BoundaryFace::inner_x2](m, mbd, eos, u0_);
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x1
    switch (pmy_pack->pmb->mbbcs(m,BoundaryFace::outer_x2)) {
      case BoundaryFlag::reflect:
        ReflectOuterX2(m);
        break;
      case BoundaryFlag::outflow:
        OutflowOuterX2(m);
        break;
      case BoundaryFlag::user:
        pmy_pack->pmesh->BoundaryFunc[BoundaryFace::outer_x2](m, mbd, eos, u0_);
        break;
      default:
        break;
    }
  }
  if (pmy_pack->pmesh->two_d) return TaskStatus::complete;

  for (int m=0; m<nmb; ++m) {
    // apply physical bounaries to inner_x3
    switch (pmy_pack->pmb->mbbcs(m,BoundaryFace::inner_x3)) {
      case BoundaryFlag::reflect:
        ReflectInnerX3(m);
        break;
      case BoundaryFlag::outflow:
        OutflowInnerX3(m);
        break;
      case BoundaryFlag::user:
        pmy_pack->pmesh->BoundaryFunc[BoundaryFace::inner_x3](m, mbd, eos, u0_);
        break;
      default:
        break;
    }

    // apply physical bounaries to outer_x3
    switch (pmy_pack->pmb->mbbcs(m,BoundaryFace::outer_x3)) {
      case BoundaryFlag::reflect:
        ReflectOuterX3(m);
        break;
      case BoundaryFlag::outflow:
        OutflowOuterX3(m);
        break;
      case BoundaryFlag::user:
        pmy_pack->pmesh->BoundaryFunc[BoundaryFace::outer_x3](m, mbd, eos, u0_);
        break;
      default:
        break;
    }
  }
  return TaskStatus::complete;
}
} // namespace hydro
