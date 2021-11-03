//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock_pack.cpp
//  \brief implementation of constructor and functions in MeshBlockPack class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/turb_driver.hpp"

//----------------------------------------------------------------------------------------
// MeshBlockPack constructor:

MeshBlockPack::MeshBlockPack(Mesh *pm, int igids, int igide)
  : pmesh(pm),
    gids(igids),
    gide(igide),
    nmb_thispack(igide - igids + 1)
{
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlockPack::~MeshBlockPack()
{
  delete pmb;
  delete pcoord;
  if (phydro != nullptr) {delete phydro;}
  if (pmhd   != nullptr) {delete pmhd;}
  if (pturb  != nullptr) {delete pturb;}
}

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddMeshBlocksAndCoordinates()

void MeshBlockPack::AddMeshBlocksAndCoordinates(ParameterInput *pin, RegionIndcs indcs)
{
  pmb = new MeshBlock(this, gids, nmb_thispack);
  pcoord = new Coordinates(this, indcs, gids, nmb_thispack);
}

