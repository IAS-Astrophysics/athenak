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
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"

//----------------------------------------------------------------------------------------
// MeshBlockPack constructor:

MeshBlockPack::MeshBlockPack(Mesh *pm, int igids, int igide, RegionCells icells) :
   pmesh(pm), gids(igids), gide(igide), mb_cells(icells)
{
  nmb_thispack = gide - gids + 1;

  // initialize MeshBlock cell indices
  mb_cells.is = mb_cells.ng;
  mb_cells.ie = mb_cells.is + mb_cells.nx1 - 1;

  if (mb_cells.nx2 > 1) {
    mb_cells.js = mb_cells.ng;
    mb_cells.je = mb_cells.js + mb_cells.nx2 - 1;
  } else {
    mb_cells.js = 0;
    mb_cells.je = 0;
  }

  if (mb_cells.nx3 > 1) {
    mb_cells.ks = mb_cells.ng;
    mb_cells.ke = mb_cells.ks + mb_cells.nx3 - 1;
  } else {
    mb_cells.ks = 0;
    mb_cells.ke = 0;
  }

  // initialize coarse grid indices
  if (pm->multilevel) {
    cmb_cells.ng = (mb_cells.ng + 1)/2 + 1;
    cmb_cells.is = cmb_cells.ng;
    cmb_cells.ie = cmb_cells.is + mb_cells.nx1/2 - 1;
    cmb_cells.nx1 = cmb_cells.ie - cmb_cells.is + 1;

    if (mb_cells.nx2 > 1) {
      cmb_cells.js = cmb_cells.ng;
      cmb_cells.je = cmb_cells.js + mb_cells.nx2/2 - 1;
      cmb_cells.nx2 = cmb_cells.je - cmb_cells.js + 1;
    } else {
      cmb_cells.js = 0;
      cmb_cells.je = 0;
      cmb_cells.nx2 = 1;
    }
  
    if (mb_cells.nx3 > 1) {
      cmb_cells.ks = cmb_cells.ng;
      cmb_cells.ke = cmb_cells.ks + mb_cells.nx3/2 - 1;
      cmb_cells.nx3 = cmb_cells.ke - cmb_cells.ks + 1;
    } else {
      cmb_cells.ks = 0;
      cmb_cells.ke = 0;
      cmb_cells.nx3 = 1;
    }
  }

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
  if (pvisc  != nullptr) {delete pvisc;}
  if (presist!= nullptr) {delete presist;}
}
