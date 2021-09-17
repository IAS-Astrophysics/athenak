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

MeshBlockPack::MeshBlockPack(Mesh *pm, ParameterInput *pin, int igids, int igide,
                             RegionIndcs indcs)
  : pmesh(pm),
    gids(igids),
    gide(igide),
    coord(pm, indcs, igids, (igide-igids+1))
{
  nmb_thispack = igide - igids + 1;

  // create MeshBlocks for this MeshBlockPack
  pmb = new MeshBlock(pm, gids, nmb_thispack);
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlockPack::~MeshBlockPack()
{
  delete pmb;
  if (phydro != nullptr) {delete phydro;}
  if (pmhd   != nullptr) {delete pmhd;}
  if (pturb  != nullptr) {delete pturb;}
}
