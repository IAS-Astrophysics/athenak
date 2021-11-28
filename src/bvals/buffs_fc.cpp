//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffs_fc.cpp
//  \brief functions to allocate and initialize buffers for face-centered variables

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "utils/create_mpitag.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BValFC::InitSendIndices
//! \brief Calculates indices of cells in mesh used to pack buffers and send FC data.
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BValFC::InitSendIndices(BValBufferFC &buf, int ox1, int ox2, int ox3, int f1, int f2)
{
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mb_cindcs = pmy_pack->pmesh->mb_cindcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

std::cout << "size of fc buffer = " << sizeof(BValBufferFC) << std::endl;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/fc/bvals_fc.cpp
  // for uniform grid: face-neighbors take care of the overlapping faces
  {auto &sindcs = buf.sindcs;
  if (ox1 == 0) {
    sindcs[0].bis = mb_indcs.is,           sindcs[0].bie = mb_indcs.ie + 1;
    sindcs[1].bis = mb_indcs.is,           sindcs[1].bie = mb_indcs.ie;
    sindcs[2].bis = mb_indcs.is,           sindcs[2].bie = mb_indcs.ie;
  } else if (ox1 > 0) {
    sindcs[0].bis = mb_indcs.ie - ng1,     sindcs[0].bie = mb_indcs.ie;
    sindcs[1].bis = mb_indcs.ie - ng1,     sindcs[1].bie = mb_indcs.ie;
    sindcs[2].bis = mb_indcs.ie - ng1,     sindcs[2].bie = mb_indcs.ie;
  } else {
    sindcs[0].bis = mb_indcs.is + 1,       sindcs[0].bie = mb_indcs.is + ng;
    sindcs[1].bis = mb_indcs.is,           sindcs[1].bie = mb_indcs.is + ng1;
    sindcs[2].bis = mb_indcs.is,           sindcs[2].bie = mb_indcs.is + ng1;
  }
  if (ox2 == 0) {
    sindcs[0].bjs = mb_indcs.js,           sindcs[0].bje = mb_indcs.je;
    sindcs[1].bjs = mb_indcs.js,           sindcs[1].bje = mb_indcs.je + 1;
    sindcs[2].bjs = mb_indcs.js,           sindcs[2].bje = mb_indcs.je;
  } else if (ox2 > 0) {
    sindcs[0].bjs = mb_indcs.je - ng1,     sindcs[0].bje = mb_indcs.je;
    sindcs[1].bjs = mb_indcs.je - ng1,     sindcs[1].bje = mb_indcs.je;
    sindcs[2].bjs = mb_indcs.je - ng1,     sindcs[2].bje = mb_indcs.je;
  } else {
    sindcs[0].bjs = mb_indcs.js,           sindcs[0].bje = mb_indcs.js + ng1;
    sindcs[1].bjs = mb_indcs.js + 1,       sindcs[1].bje = mb_indcs.js + ng;
    sindcs[2].bjs = mb_indcs.js,           sindcs[2].bje = mb_indcs.js + ng1;
  }
  if (ox3 == 0) {
    sindcs[0].bks = mb_indcs.ks,           sindcs[0].bke = mb_indcs.ke;
    sindcs[1].bks = mb_indcs.ks,           sindcs[1].bke = mb_indcs.ke;
    sindcs[2].bks = mb_indcs.ks,           sindcs[2].bke = mb_indcs.ke + 1;
  } else if (ox3 > 0) {
    sindcs[0].bks = mb_indcs.ke - ng1,     sindcs[0].bke = mb_indcs.ke;
    sindcs[1].bks = mb_indcs.ke - ng1,     sindcs[1].bke = mb_indcs.ke;
    sindcs[2].bks = mb_indcs.ke - ng1,     sindcs[2].bke = mb_indcs.ke;
  } else {
    sindcs[0].bks = mb_indcs.ks,           sindcs[0].bke = mb_indcs.ks + ng1;
    sindcs[1].bks = mb_indcs.ks,           sindcs[1].bke = mb_indcs.ks + ng1;
    sindcs[2].bks = mb_indcs.ks + 1,       sindcs[2].bke = mb_indcs.ks + ng;
  }
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
    if (ox1 > 0) {sindcs[0].bie++;}
    if (ox1 < 0) {sindcs[0].bis--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
    if (ox2 > 0) {sindcs[1].bje++;}
    if (ox2 < 0) {sindcs[1].bjs--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
    if (ox3 > 0) {sindcs[2].bke++;}
    if (ox3 < 0) {sindcs[2].bks--;}
  }
  for (int i=0; i<=2; ++i) {
    sindcs[i].ndat = (sindcs[i].bie - sindcs[i].bis + 1)*
                     (sindcs[i].bje - sindcs[i].bjs + 1)*
                     (sindcs[i].bke - sindcs[i].bks + 1);
  }}

  // set indices for sends to neighbors on COARSER level
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/fc/bvals_fc.cpp
  {auto &cindcs = buf.cindcs;
  if (ox1 == 0) { 
    cindcs[0].bis = mb_cindcs.is,          cindcs[0].bie = mb_cindcs.ie + 1;
    cindcs[1].bis = mb_cindcs.is,          cindcs[1].bie = mb_cindcs.ie;
    cindcs[2].bis = mb_cindcs.is,          cindcs[2].bie = mb_cindcs.ie;
  } else if (ox1 > 0) {
    cindcs[0].bis = mb_cindcs.ie - ng1,    cindcs[0].bie = mb_cindcs.ie;
    cindcs[1].bis = mb_cindcs.ie - ng1,    cindcs[1].bie = mb_cindcs.ie;
    cindcs[2].bis = mb_cindcs.ie - ng1,    cindcs[2].bie = mb_cindcs.ie;
  } else {
    cindcs[0].bis = mb_cindcs.is + 1,      cindcs[0].bie = mb_cindcs.is + ng;
    cindcs[1].bis = mb_cindcs.is,          cindcs[1].bie = mb_cindcs.is + ng1;
    cindcs[2].bis = mb_cindcs.is,          cindcs[2].bie = mb_cindcs.is + ng1;
  }
  if (ox2 == 0) { 
    cindcs[0].bjs = mb_cindcs.js,          cindcs[0].bje = mb_cindcs.je;
    cindcs[1].bjs = mb_cindcs.js,          cindcs[1].bje = mb_cindcs.je + 1;
    cindcs[2].bjs = mb_cindcs.js,          cindcs[2].bje = mb_cindcs.je;
  } else if (ox2 > 0) {
    cindcs[0].bjs = mb_cindcs.je - ng1,    cindcs[0].bje = mb_cindcs.je;
    cindcs[1].bjs = mb_cindcs.je - ng1,    cindcs[1].bje = mb_cindcs.je;
    cindcs[2].bjs = mb_cindcs.je - ng1,    cindcs[2].bje = mb_cindcs.je;
  } else {
    cindcs[0].bjs = mb_cindcs.js,          cindcs[0].bje = mb_cindcs.js + ng1;
    cindcs[1].bjs = mb_cindcs.js + 1,      cindcs[1].bje = mb_cindcs.js + ng;
    cindcs[2].bjs = mb_cindcs.js,          cindcs[2].bje = mb_cindcs.js + ng1;
  }
  if (ox3 == 0) {
    cindcs[0].bks = mb_cindcs.ks,          cindcs[0].bke = mb_cindcs.ke;
    cindcs[1].bks = mb_cindcs.ks,          cindcs[1].bke = mb_cindcs.ke;
    cindcs[2].bks = mb_cindcs.ks,          cindcs[2].bke = mb_cindcs.ke + 1;
  } else if (ox3 > 0) {
    cindcs[0].bks = mb_cindcs.ke - ng1,    cindcs[0].bke = mb_cindcs.ke;
    cindcs[1].bks = mb_cindcs.ke - ng1,    cindcs[1].bke = mb_cindcs.ke;
    cindcs[2].bks = mb_cindcs.ke - ng1,    cindcs[2].bke = mb_cindcs.ke;
  } else {              
    cindcs[0].bks = mb_cindcs.ks,          cindcs[0].bke = mb_cindcs.ks + ng1;
    cindcs[1].bks = mb_cindcs.ks,          cindcs[1].bke = mb_cindcs.ks + ng1;
    cindcs[2].bks = mb_cindcs.ks + 1,      cindcs[2].bke = mb_cindcs.ks + ng;
  }
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
    if (ox1 > 0) {cindcs[0].bie++;}
    if (ox1 < 0) {cindcs[0].bis--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
    if (ox2 > 0) {cindcs[1].bje++;}
    if (ox2 < 0) {cindcs[1].bjs--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
    if (ox3 > 0) {cindcs[2].bke++;}
    if (ox3 < 0) {cindcs[2].bks--;}
  }
  for (int i=0; i<=2; ++i) {
    cindcs[i].ndat = (cindcs[i].bie - cindcs[i].bis + 1)*
                     (cindcs[i].bje - cindcs[i].bjs + 1)*
                     (cindcs[i].bke - cindcs[i].bks + 1);
  }}

  // set indices for sends to neighbors on FINER level
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/fc/bvals_fc.cpp
  {auto &findcs = buf.findcs;
  int cnx1 = mb_indcs.nx1/2 - ng;
  int cnx2 = mb_indcs.nx2/2 - ng;
  int cnx3 = mb_indcs.nx3/2 - ng;
  if (ox1 == 0) {
    findcs[0].bis = mb_indcs.is,          findcs[0].bie = mb_indcs.ie + 1;
    findcs[1].bis = mb_indcs.is,          findcs[1].bie = mb_indcs.ie;
    findcs[2].bis = mb_indcs.is,          findcs[2].bie = mb_indcs.ie;
  } else if (ox1 > 0) {
    findcs[0].bis = mb_indcs.ie - ng1,    findcs[0].bie = mb_indcs.ie + 1;
    findcs[1].bis = mb_indcs.ie - ng1,    findcs[1].bie = mb_indcs.ie;
    findcs[2].bis = mb_indcs.ie - ng1,    findcs[2].bie = mb_indcs.ie;
  } else {
    findcs[0].bis = mb_indcs.is,          findcs[0].bie = mb_indcs.is + ng;
    findcs[1].bis = mb_indcs.is,          findcs[1].bie = mb_indcs.is + ng1;
    findcs[2].bis = mb_indcs.is,          findcs[2].bie = mb_indcs.is + ng1;
  }
  if (ox2 == 0) {
    findcs[0].bjs = mb_indcs.js,          findcs[0].bje = mb_indcs.je;
    findcs[1].bjs = mb_indcs.js,          findcs[1].bje = mb_indcs.je + 1;
    findcs[2].bjs = mb_indcs.js,          findcs[2].bje = mb_indcs.je;
  } else if (ox2 > 0) {
    findcs[0].bjs = mb_indcs.je - ng1,    findcs[0].bje = mb_indcs.je;
    findcs[1].bjs = mb_indcs.je - ng1,    findcs[1].bje = mb_indcs.je + 1;
    findcs[2].bjs = mb_indcs.je - ng1,    findcs[2].bje = mb_indcs.je;
  } else {
    findcs[0].bjs = mb_indcs.js,          findcs[0].bje = mb_indcs.js + ng1;
    findcs[1].bjs = mb_indcs.js,          findcs[1].bje = mb_indcs.js + ng;
    findcs[2].bjs = mb_indcs.js,          findcs[2].bje = mb_indcs.js + ng1;
  }
  if (ox3 == 0) {
    findcs[0].bks = mb_indcs.ks,          findcs[0].bke = mb_indcs.ke;
    findcs[1].bks = mb_indcs.ks,          findcs[1].bke = mb_indcs.ke;
    findcs[2].bks = mb_indcs.ks,          findcs[2].bke = mb_indcs.ke + 1;
  } else if (ox3 > 0) {
    findcs[0].bks = mb_indcs.ke - ng1,    findcs[0].bke = mb_indcs.ke;
    findcs[1].bks = mb_indcs.ke - ng1,    findcs[1].bke = mb_indcs.ke;
    findcs[2].bks = mb_indcs.ke - ng1,    findcs[2].bke = mb_indcs.ke + 1;
  } else {
    findcs[0].bks = mb_indcs.ks,          findcs[0].bke = mb_indcs.ks + ng1;
    findcs[1].bks = mb_indcs.ks,          findcs[1].bke = mb_indcs.ks + ng1;
    findcs[2].bks = mb_indcs.ks,          findcs[2].bke = mb_indcs.ks + ng;
  }
  // need to add internal edges on faces, and internal corners on edges
  if (ox1 == 0) {
    if (f1 == 1) {
      findcs[0].bis += mb_indcs.nx1/2 - mb_cindcs.ng;
      findcs[1].bis += mb_indcs.nx1/2 - mb_cindcs.ng;
      findcs[2].bis += mb_indcs.nx1/2 - mb_cindcs.ng;
    } else {
      findcs[0].bie -= mb_indcs.nx1/2 - mb_cindcs.ng;
      findcs[1].bie -= mb_indcs.nx1/2 - mb_cindcs.ng;
      findcs[2].bie -= mb_indcs.nx1/2 - mb_cindcs.ng;
    }
  }
  if (ox2 == 0 && mb_indcs.nx2 > 1) {
    if (ox1 != 0) {
      if (f1 == 1) {
        findcs[0].bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[1].bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[2].bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
      } else {
        findcs[0].bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[1].bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[2].bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
      }
    } else {
      if (f2 == 1) {
        findcs[0].bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[1].bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[2].bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
      } else {
        findcs[0].bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[1].bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
        findcs[2].bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
      }
    }
  }
  if (ox3 == 0 && mb_indcs.nx3 > 1) {
    if (ox1 != 0 && ox2 != 0) {
      if (f1 == 1) {
        findcs[0].bks += mb_indcs.nx3/2 - mb_cindcs.ng;
        findcs[1].bks += mb_indcs.nx3/2 - mb_cindcs.ng;
        findcs[2].bks += mb_indcs.nx3/2 - mb_cindcs.ng;
      } else {
        findcs[0].bke -= mb_indcs.nx3/2 - mb_cindcs.ng;
        findcs[1].bke -= mb_indcs.nx3/2 - mb_cindcs.ng;
        findcs[2].bke -= mb_indcs.nx3/2 - mb_cindcs.ng;
      }
    } else {
      if (f2 == 1) {
        findcs[0].bks += mb_indcs.nx3/2 - mb_cindcs.ng;
        findcs[1].bks += mb_indcs.nx3/2 - mb_cindcs.ng;
        findcs[2].bks += mb_indcs.nx3/2 - mb_cindcs.ng;
      } else {
        findcs[0].bke -= mb_indcs.nx3/2 - mb_cindcs.ng;}
        findcs[1].bke -= mb_indcs.nx3/2 - mb_cindcs.ng;}
        findcs[2].bke -= mb_indcs.nx3/2 - mb_cindcs.ng;}
    }
  }
  for (int i=0; i<=2; ++i) {
    findcs[i].ndat = (findcs[i].bie - findcs[i].bis + 1)*
                     (findcs[i].bje - findcs[i].bjs + 1)*
                     (findcs[i].bke - findcs[i].bks + 1);
  }}

  // indices for PROLONGATION not needed for sends, just initialize to zero
  {auto &pindcs = buf.pindcs;
  for (int i=0; i<=2; ++i) {
    pindcs[i].bis = 0; pindcs[i].bie = 0;
    pindcs[i].bjs = 0; pindcs[i].bje = 0;
    pindcs[i].bks = 0; pindcs[i].bke = 0;
    pindcs[i].ndat = 1;
  }}

}

//----------------------------------------------------------------------------------------
//! \fn void BValFC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for FC data
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BValFC::InitRecvIndices(BValBufferFC &buf, int ox1, int ox2, int ox3, int f1, int f2)
{ 
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  auto &mb_cindcs = pmy_pack->pmesh->mb_cindcs;
  int ng = mb_indcs.ng;

  // set indices for receives from neighbors on SAME level
  // Formulae taken from SetBoundarySameLevel() in src/bvals/fc/bvals_fc.cpp
  {auto &sindcs = buf.sindcs;   // indices of buffer at same level ("s")
  if (ox1 == 0) {
    sindcs[0].bis = mb_indcs.is,         sindcs[0].bie = mb_indcs.ie + 1;
    sindcs[1].bis = mb_indcs.is,         sindcs[1].bie = mb_indcs.ie;
    sindcs[2].bis = mb_indcs.is,         sindcs[2].bie = mb_indcs.ie;
  } else if (ox1 > 0) {
    sindcs[0].bis = mb_indcs.ie + 2,     sindcs[0].bie = mb_indcs.ie + ng + 1;
    sindcs[1].bis = mb_indcs.ie + 1,     sindcs[1].bie = mb_indcs.ie + ng;
    sindcs[2].bis = mb_indcs.ie + 1,     sindcs[2].bie = mb_indcs.ie + ng;
  } else {
    sindcs[0].bis = mb_indcs.is - ng,    sindcs[0].bie = mb_indcs.is - 1;
    sindcs[1].bis = mb_indcs.is - ng,    sindcs[1].bie = mb_indcs.is - 1;
    sindcs[2].bis = mb_indcs.is - ng,    sindcs[2].bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    sindcs[0].bjs = mb_indcs.js,          sindcs[0].bje = mb_indcs.je;
    sindcs[1].bjs = mb_indcs.js,          sindcs[1].bje = mb_indcs.je + 1;
    sindcs[2].bjs = mb_indcs.js,          sindcs[2].bje = mb_indcs.je;
  } else if (ox2 > 0) {
    sindcs[0].bjs = mb_indcs.je + 1,      sindcs[0].bje = mb_indcs.je + ng;
    sindcs[1].bjs = mb_indcs.je + 2,      sindcs[1].bje = mb_indcs.je + ng + 1;
    sindcs[2].bjs = mb_indcs.je + 1,      sindcs[2].bje = mb_indcs.je + ng;
  } else {
    sindcs[0].bjs = mb_indcs.js - ng,     sindcs[0].bje = mb_indcs.js - 1;
    sindcs[1].bjs = mb_indcs.js - ng,     sindcs[1].bje = mb_indcs.js - 1;
    sindcs[2].bjs = mb_indcs.js - ng,     sindcs[2].bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    sindcs[0].bks = mb_indcs.ks,          sindcs[0].bke = mb_indcs.ke;
    sindcs[1].bks = mb_indcs.ks,          sindcs[1].bke = mb_indcs.ke;
    sindcs[2].bks = mb_indcs.ks,          sindcs[2].bke = mb_indcs.ke + 1;
  } else if (ox3 > 0) {
    sindcs[0].bks = mb_indcs.ke + 1,      sindcs[0].bke = mb_indcs.ke + ng;
    sindcs[1].bks = mb_indcs.ke + 1,      sindcs[1].bke = mb_indcs.ke + ng;
    sindcs[2].bks = mb_indcs.ke + 2,      sindcs[2].bke = mb_indcs.ke + ng + 1;
  } else {
    sindcs[0].bks = mb_indcs.ks - ng,     sindcs[0].bke = mb_indcs.ks - 1;
    sindcs[1].bks = mb_indcs.ks - ng,     sindcs[1].bke = mb_indcs.ks - 1;
    sindcs[2].bks = mb_indcs.ks - ng,     sindcs[2].bke = mb_indcs.ks - 1;
  }
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
    if (ox1 > 0) {sindcs[0].bis--;}
    if (ox1 < 0) {sindcs[0].bie++;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
    if (ox2 > 0) {sindcs[1].bjs--;}
    if (ox2 < 0) {sindcs[1].bje++;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
    if (ox3 > 0) {sindcs[2].bks--;}
    if (ox3 < 0) {sindcs[2].bke++;}
  }
  for (int i=0; i<=2; ++i) {
    sindcs[i].ndat = (sindcs[i].bie - sindcs[i].bis + 1)*
                     (sindcs[i].bje - sindcs[i].bjs + 1)*
                     (sindcs[i].bke - sindcs[i].bks + 1);
  }}

  // set indices for receives from neighbors on COARSER level
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/fc/bvals_fc.cpp
  {auto &cindcs = buf.cindcs;   // indices of course buffer ("c")
  if (ox1 == 0) {
    cindcs[0].bis = mb_cindcs.is,         cindcs[0].bie = mb_cindcs.ie + 1;
    cindcs[1].bis = mb_cindcs.is,         cindcs[1].bie = mb_cindcs.ie;
    cindcs[2].bis = mb_cindcs.is,         cindcs[2].bie = mb_cindcs.ie;
    if (f1 == 0) {
      cindcs[0].bie += mb_indcs.ng;
      cindcs[1].bie += mb_indcs.ng;
      cindcs[2].bie += mb_indcs.ng;
    } else {
      cindcs[0].bis -= mb_indcs.ng;
      cindcs[1].bis -= mb_indcs.ng;
      cindcs[2].bis -= mb_indcs.ng;
    }
  } else if (ox1 > 0) {
    cindcs[0].bis = mb_cindcs.ie + 2,     cindcs[0].bie = mb_cindcs.ie + ng + 1;
    cindcs[1].bis = mb_cindcs.ie + 1,     cindcs[1].bie = mb_cindcs.ie + ng;
    cindcs[2].bis = mb_cindcs.ie + 1,     cindcs[2].bie = mb_cindcs.ie + ng;
  } else {
    cindcs[0].bis = mb_cindcs.is - ng,    cindcs[0].bie = mb_cindcs.is - 1;
    cindcs[1].bis = mb_cindcs.is - ng,    cindcs[1].bie = mb_cindcs.is - 1;
    cindcs[2].bis = mb_cindcs.is - ng,    cindcs[2].bie = mb_cindcs.is - 1;
  }
  if (ox2 == 0) {
    cindcs[0].bjs = mb_cindcs.js,          cindcs[0].bje = mb_cindcs.je;
    cindcs[1].bjs = mb_cindcs.js,          cindcs[1].bje = mb_cindcs.je + 1;
    cindcs[2].bjs = mb_cindcs.js,          cindcs[2].bje = mb_cindcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          cindcs[0].bje += mb_indcs.ng;
          cindcs[1].bje += mb_indcs.ng;
          cindcs[2].bje += mb_indcs.ng;
        } else {
          cindcs[0].bjs -= mb_indcs.ng;
          cindcs[1].bjs -= mb_indcs.ng;
          cindcs[2].bjs -= mb_indcs.ng;
        }
      } else {
        if (f2 == 0) {
          cindcs[0].bje += mb_indcs.ng;
          cindcs[1].bje += mb_indcs.ng;
          cindcs[2].bje += mb_indcs.ng;
        } else {
          cindcs[0].bjs -= mb_indcs.ng;
          cindcs[1].bjs -= mb_indcs.ng;
          cindcs[2].bjs -= mb_indcs.ng;
        }
      }
    }
  } else if (ox2 > 0) {
    cindcs[0].bjs = mb_cindcs.je + 1,      cindcs[0].bje = mb_cindcs.je + ng;
    cindcs[1].bjs = mb_cindcs.je + 2,      cindcs[1].bje = mb_cindcs.je + ng + 1;
    cindcs[2].bjs = mb_cindcs.je + 1,      cindcs[2].bje = mb_cindcs.je + ng;
  } else {
    cindcs[0].bjs = mb_cindcs.js - ng,     cindcs[0].bje = mb_cindcs.js - 1;
    cindcs[1].bjs = mb_cindcs.js - ng,     cindcs[1].bje = mb_cindcs.js - 1;
    cindcs[2].bjs = mb_cindcs.js - ng,     cindcs[2].bje = mb_cindcs.js - 1;
  }
  if (ox3 == 0) {
    cindcs[0].bks = mb_cindcs.ks,          cindcs[0].bke = mb_cindcs.ke;
    cindcs[1].bks = mb_cindcs.ks,          cindcs[1].bke = mb_cindcs.ke;
    cindcs[2].bks = mb_cindcs.ks,          cindcs[2].bke = mb_cindcs.ke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          cindcs[0].bke += mb_indcs.ng;
          cindcs[1].bke += mb_indcs.ng;
          cindcs[2].bke += mb_indcs.ng;
        } else {
          cindcs[0].bks -= mb_indcs.ng;}
          cindcs[1].bks -= mb_indcs.ng;}
          cindcs[2].bks -= mb_indcs.ng;}
      } else {
        if (f2 == 0) {
          cindcs[0].bke += mb_indcs.ng;
          cindcs[1].bke += mb_indcs.ng;
          cindcs[2].bke += mb_indcs.ng;
        } else {
          cindcs[0].bks -= mb_indcs.ng;}
          cindcs[1].bks -= mb_indcs.ng;}
          cindcs[2].bks -= mb_indcs.ng;}
      }
    }
  } else if (ox3 > 0) {
    cindcs[0].bks = mb_cindcs.ke + 1,      cindcs[0].bke = mb_cindcs.ke + ng;
    cindcs[1].bks = mb_cindcs.ke + 1,      cindcs[1].bke = mb_cindcs.ke + ng;
    cindcs[2].bks = mb_cindcs.ke + 2,      cindcs[2].bke = mb_cindcs.ke + ng + 1;
  } else {
    cindcs[0].bks = mb_cindcs.ks - ng,     cindcs[0].bke = mb_cindcs.ks - 1;
    cindcs[1].bks = mb_cindcs.ks - ng,     cindcs[1].bke = mb_cindcs.ks - 1;
    cindcs[2].bks = mb_cindcs.ks - ng,     cindcs[2].bke = mb_cindcs.ks - 1;
  }
  for (int i=0; i<=2; ++i) {
    cindcs[i].ndat = (cindcs[i].bie - cindcs[i].bis + 1)*
                     (cindcs[i].bje - cindcs[i].bjs + 1)*
                     (cindcs[i].bke - cindcs[i].bks + 1);
  }}

  // set indices for receives from neighbors on FINER level
  // Formulae taken from SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &findcs = buf.findcs;   // indices of fine buffer ("f")
  if (ox1 == 0) {
    findcs[0].bis = mb_indcs.is;               findcs[0].bie = mb_indcs.ie + 1;
    findcs[1].bis = mb_indcs.is;               findcs[1].bie = mb_indcs.ie;
    findcs[2].bis = mb_indcs.is;               findcs[2].bie = mb_indcs.ie;
    if (f1 == 1) {
      findcs[0].bis += mb_indcs.nx1/2;
      findcs[1].bis += mb_indcs.nx1/2;
      findcs[2].bis += mb_indcs.nx1/2;
    } else {
      findcs[0].bie -= mb_indcs.nx1/2;
      findcs[1].bie -= mb_indcs.nx1/2;
      findcs[2].bie -= mb_indcs.nx1/2;
    }
  } else if (ox1 > 0) {
    findcs[0].bis = mb_indcs.ie + 2;           findcs[0].bie = mb_indcs.ie + ng + 1;
    findcs[1].bis = mb_indcs.ie + 1;           findcs[1].bie = mb_indcs.ie + ng;
    findcs[2].bis = mb_indcs.ie + 1;           findcs[2].bie = mb_indcs.ie + ng;
  } else {
    findcs[0].bis = mb_indcs.is - ng;          findcs[0].bie = mb_indcs.is - 1;
    findcs[1].bis = mb_indcs.is - ng;          findcs[1].bie = mb_indcs.is - 1;
    findcs[2].bis = mb_indcs.is - ng;          findcs[2].bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    findcs[0].bjs = mb_indcs.js;             findcs[0].bje = mb_indcs.je;
    findcs[1].bjs = mb_indcs.js;             findcs[1].bje = mb_indcs.je;
    findcs[2].bjs = mb_indcs.js;             findcs[2].bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          findcs[0].bjs += mb_indcs.nx2/2;
          findcs[1].bjs += mb_indcs.nx2/2;
          findcs[2].bjs += mb_indcs.nx2/2;
        } else {
          findcs[0].bje -= mb_indcs.nx2/2;
          findcs[1].bje -= mb_indcs.nx2/2;
          findcs[2].bje -= mb_indcs.nx2/2;
        }
      } else {
        if (f2 == 1) {
          findcs[0].bjs += mb_indcs.nx2/2;
          findcs[1].bjs += mb_indcs.nx2/2;
          findcs[2].bjs += mb_indcs.nx2/2;
        } else {
          findcs[0].bje -= mb_indcs.nx2/2;
          findcs[1].bje -= mb_indcs.nx2/2;
          findcs[2].bje -= mb_indcs.nx2/2;
        }
      }
    }
  } else if (ox2 > 0) {
    findcs[0].bjs = mb_indcs.je + 1;          findcs[0].bje = mb_indcs.je + ng;
    findcs[1].bjs = mb_indcs.je + 1;          findcs[1].bje = mb_indcs.je + ng;
    findcs[2].bjs = mb_indcs.je + 1;          findcs[2].bje = mb_indcs.je + ng;
  } else {
    findcs[0].bjs = mb_indcs.js - ng;         findcs[0].bje = mb_indcs.js - 1;
    findcs[1].bjs = mb_indcs.js - ng;         findcs[1].bje = mb_indcs.js - 1;
    findcs[2].bjs = mb_indcs.js - ng;         findcs[2].bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    findcs[0].bks = mb_indcs.ks;              findcs[0].bke = mb_indcs.ke;
    findcs[1].bks = mb_indcs.ks;              findcs[1].bke = mb_indcs.ke;
    findcs[2].bks = mb_indcs.ks;              findcs[2].bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          findcs[0].bks += mb_indcs.nx3/2;
          findcs[1].bks += mb_indcs.nx3/2;
          findcs[2].bks += mb_indcs.nx3/2;
        } else {
          findcs[0].bke -= mb_indcs.nx3/2;
          findcs[1].bke -= mb_indcs.nx3/2;
          findcs[2].bke -= mb_indcs.nx3/2;
        }
      } else {
        if (f2 == 1) {
          findcs[0].bks += mb_indcs.nx3/2;
          findcs[1].bks += mb_indcs.nx3/2;
          findcs[2].bks += mb_indcs.nx3/2;
        } else {
          findcs[0].bke -= mb_indcs.nx3/2;
          findcs[1].bke -= mb_indcs.nx3/2;
          findcs[2].bke -= mb_indcs.nx3/2;
        }
      }
    }
  } else if (ox3 > 0) {
    findcs[0].bks = mb_indcs.ke + 1;         findcs[0].bke = mb_indcs.ke + ng;
    findcs[1].bks = mb_indcs.ke + 1;         findcs[1].bke = mb_indcs.ke + ng;
    findcs[2].bks = mb_indcs.ke + 1;         findcs[2].bke = mb_indcs.ke + ng;
  } else {
    findcs[0].bks = mb_indcs.ks - ng;        findcs[0].bke = mb_indcs.ks - 1;
    findcs[1].bks = mb_indcs.ks - ng;        findcs[1].bke = mb_indcs.ks - 1;
    findcs[2].bks = mb_indcs.ks - ng;        findcs[2].bke = mb_indcs.ks - 1;
  }
  for (int i=0; i<=2; ++i) {
    findcs[i].ndat = (findcs[i].bie - findcs[i].bis + 1)*
                     (findcs[i].bje - findcs[i].bjs + 1)*
                     (findcs[i].bke - findcs[i].bks + 1);
  }}

  // set indices for PROLONGATION in coarse cell buffers
  // Formulae taken from ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  // Identical to receives from coarser level, except ng --> ng/2
  {auto &pindcs = buf.pindcs;   // indices fpr prolongation ("p")
  int cn = mb_indcs.ng/2;       // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    pindcs[0].bis = mb_cindcs.is;          pindcs[0].bie = mb_cindcs.ie;
    pindcs[1].bis = mb_cindcs.is;          pindcs[1].bie = mb_cindcs.ie;
    pindcs[2].bis = mb_cindcs.is;          pindcs[2].bie = mb_cindcs.ie;
    if (f1 == 0) {
      pindcs[0].bie += cn;
      pindcs[1].bie += cn;
      pindcs[2].bie += cn;
    } else {
      pindcs[0].bis -= cn;}
      pindcs[1].bis -= cn;}
      pindcs[2].bis -= cn;}
  } else if (ox1 > 0)  {
    pindcs[0].bis = mb_cindcs.ie + 1;       pindcs[0].bie = mb_cindcs.ie + cn;
    pindcs[1].bis = mb_cindcs.ie + 1;       pindcs[1].bie = mb_cindcs.ie + cn;
    pindcs[2].bis = mb_cindcs.ie + 1;       pindcs[2].bie = mb_cindcs.ie + cn;
  } else {
    pindcs[0].bis = mb_cindcs.is - cn;      pindcs[0].bie = mb_cindcs.is - 1;
    pindcs[1].bis = mb_cindcs.is - cn;      pindcs[1].bie = mb_cindcs.is - 1;
    pindcs[2].bis = mb_cindcs.is - cn;      pindcs[2].bie = mb_cindcs.is - 1;
  }
  if (ox2 == 0) {
    pindcs[0].bjs = mb_cindcs.js;           pindcs[0].bje = mb_cindcs.je;
    pindcs[1].bjs = mb_cindcs.js;           pindcs[1].bje = mb_cindcs.je;
    pindcs[2].bjs = mb_cindcs.js;           pindcs[2].bje = mb_cindcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          pindcs[0].bje += cn;
          pindcs[1].bje += cn;
          pindcs[2].bje += cn;
        } else {
          pindcs[0].bjs -= cn;
          pindcs[1].bjs -= cn;
          pindcs[2].bjs -= cn;
        }
      } else {
        if (f2 == 0) {
          pindcs[0].bje += cn;
          pindcs[1].bje += cn;
          pindcs[2].bje += cn;
        } else {
          pindcs[0].bjs -= cn;
          pindcs[1].bjs -= cn;
          pindcs[2].bjs -= cn;
        }
      }
    }
  } else if (ox2 > 0) {
    pindcs[0].bjs = mb_cindcs.je + 1;        pindcs[0].bje = mb_cindcs.je + cn;
    pindcs[1].bjs = mb_cindcs.je + 1;        pindcs[1].bje = mb_cindcs.je + cn;
    pindcs[2].bjs = mb_cindcs.je + 1;        pindcs[2].bje = mb_cindcs.je + cn;
  } else {
    pindcs[0].bjs = mb_cindcs.js - cn;       pindcs[0].bje = mb_cindcs.js - 1;
    pindcs[1].bjs = mb_cindcs.js - cn;       pindcs[1].bje = mb_cindcs.js - 1;
    pindcs[2].bjs = mb_cindcs.js - cn;       pindcs[2].bje = mb_cindcs.js - 1;
  }
  if (ox3 == 0) {
    pindcs[0].bks = mb_cindcs.ks;            pindcs[0].bke = mb_cindcs.ke;
    pindcs[1].bks = mb_cindcs.ks;            pindcs[1].bke = mb_cindcs.ke;
    pindcs[2].bks = mb_cindcs.ks;            pindcs[2].bke = mb_cindcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          pindcs[0].bke += cn;
          pindcs[1].bke += cn;
          pindcs[2].bke += cn;
        } else {
          pindcs[0].bks -= cn;
          pindcs[1].bks -= cn;
          pindcs[2].bks -= cn;
        }
      } else {
        if (f2 == 0) {
          pindcs[0].bke += cn;
          pindcs[1].bke += cn;
          pindcs[2].bke += cn;
        } else {
          pindcs[0].bks -= cn;
          pindcs[1].bks -= cn;
          pindcs[2].bks -= cn;
        }
      }
    }
  } else if (ox3 > 0)  {
    pindcs[0].bks = mb_cindcs.ke + 1;          pindcs[0].bke = mb_cindcs.ke + cn;
    pindcs[1].bks = mb_cindcs.ke + 1;          pindcs[1].bke = mb_cindcs.ke + cn;
    pindcs[2].bks = mb_cindcs.ke + 1;          pindcs[2].bke = mb_cindcs.ke + cn;
  } else {
    pindcs[0].bks = mb_cindcs.ks - cn;         pindcs[0].bke = mb_cindcs.ks - 1;
    pindcs[1].bks = mb_cindcs.ks - cn;         pindcs[1].bke = mb_cindcs.ks - 1;
    pindcs[2].bks = mb_cindcs.ks - cn;         pindcs[2].bke = mb_cindcs.ks - 1;
  }
  for (int i=0; i<=2; ++i) {
    pindcs[i].ndat = (pindcs[i].bie - pindcs[i].bis + 1)*
                     (pindcs[i].bje - pindcs[i].bjs + 1)*
                     (pindcs[i].bke - pindcs[i].bks + 1);
  }}
}

//----------------------------------------------------------------------------------------
//! \fn void BValFC::AllocateBuffersFC
//! \brief initialize vector of send/recv BValBuffers for face-centered fields. 
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector

void BValFC::AllocateBuffersFC(const int nvar)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ng = indcs.ng;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  auto &cindcs = pmy_pack->pmesh->mb_cindcs;
  int cis = cindcs.is, cie = cindcs.ie;
  int cjs = cindcs.js, cje = cindcs.je;
  int cks = cindcs.ks, cke = cindcs.ke;

  int ng1 = ng-1;
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  // allocate size of (some) Views
  for (int n=0; n<nnghbr; ++n) {
    Kokkos::realloc(send_buf[n].bcomm_stat, nmb);
    Kokkos::realloc(recv_buf[n].bcomm_stat, nmb);
#if MPI_PARALLEL_ENABLED
    // cannot create Kokkos::View of type MPI_Request (not POD) so construct STL vector
    for (int m=0; m<nmb; ++m) {
      MPI_Request send_req, recv_req;
      send_buf[n].comm_req.push_back(send_req);
      recv_buf[n].comm_req.push_back(recv_req);
    }
#endif
  }

  // initialize buffers used for uniform grid nd SMR/AMR calculations
  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // x1 faces; NeighborIndex = [0,...,7]
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy = 0; fy<nfy; fy++) {
        int indx = pmy_pack->pmb->NeighborIndx(n,0,0,fy,fz);
        InitSendIndices(send_buf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recv_buf[indx],n, 0, 0, fy, fz);
        send_buf[indx].AllocateDataView(nmb, nvar);
        recv_buf[indx].AllocateDataView(nmb, nvar);
        indx++;
      }
    }
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {

    // x2 faces; NeighborIndex = [8,...,15]
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,0,fx,fz);
          InitSendIndices(send_buf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recv_buf[indx],0, m, 0, fx, fz);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,0,fz,0);
          InitSendIndices(send_buf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recv_buf[indx],n, m, 0, fz, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {

    // x3 faces; NeighborIndex = [24,...,31]
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) { 
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,0,l,fx,fy);
          InitSendIndices(send_buf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recv_buf[indx],0, 0, l, fx, fy);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = pmy_pack->pmb->NeighborIndx(n,0,l,fy,0);
          InitSendIndices(send_buf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recv_buf[indx],n, 0, l, fy, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = pmy_pack->pmb->NeighborIndx(0,m,l,fx,0);
          InitSendIndices(send_buf[indx],0, m, l, fx, 0);
          InitRecvIndices(recv_buf[indx],0, m, l, fx, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
          indx++;
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = pmy_pack->pmb->NeighborIndx(n,m,l,0,0);
          InitSendIndices(send_buf[indx],n, m, l, 0, 0);
          InitRecvIndices(recv_buf[indx],n, m, l, 0, 0);
          send_buf[indx].AllocateDataView(nmb, nvar);
          recv_buf[indx].AllocateDataView(nmb, nvar);
        }
      }
    }
  }

/***
  for (int m=0; m<nmb; ++m) {
  for (int n=0; n<=55; ++n) {
std::cout << std::endl << "MB= "<<m<<"  Buffer="<< n << std::endl;
std::cout <<"same:" <<send_buf[n].sindcs.bis<<"  "<<send_buf[n].sindcs.bie<<
                "  "<<send_buf[n].sindcs.bjs<<"  "<<send_buf[n].sindcs.bje<<
                "  "<<send_buf[n].sindcs.bks<<"  "<<send_buf[n].sindcs.bke<< std::endl;
std::cout <<"coar:" <<send_buf[n].cindcs.bis<<"  "<<send_buf[n].cindcs.bie<<
                "  "<<send_buf[n].cindcs.bjs<<"  "<<send_buf[n].cindcs.bje<<
                "  "<<send_buf[n].cindcs.bks<<"  "<<send_buf[n].cindcs.bke<< std::endl;
std::cout <<"fine:" <<send_buf[n].findcs.bis<<"  "<<send_buf[n].findcs.bie<<
                "  "<<send_buf[n].findcs.bjs<<"  "<<send_buf[n].findcs.bje<<
                "  "<<send_buf[n].findcs.bks<<"  "<<send_buf[n].findcs.bke<< std::endl;
std::cout <<"same:" <<recv_buf[n].sindcs.bis<<"  "<<recv_buf[n].sindcs.bie<<
                "  "<<recv_buf[n].sindcs.bjs<<"  "<<recv_buf[n].sindcs.bje<<
                "  "<<recv_buf[n].sindcs.bks<<"  "<<recv_buf[n].sindcs.bke<< std::endl;
std::cout <<"coar:" <<recv_buf[n].cindcs.bis<<"  "<<recv_buf[n].cindcs.bie<<
                "  "<<recv_buf[n].cindcs.bjs<<"  "<<recv_buf[n].cindcs.bje<<
                "  "<<recv_buf[n].cindcs.bks<<"  "<<recv_buf[n].cindcs.bke<< std::endl;
std::cout <<"fine:" <<recv_buf[n].findcs.bis<<"  "<<recv_buf[n].findcs.bie<<
                "  "<<recv_buf[n].findcs.bjs<<"  "<<recv_buf[n].findcs.bje<<
                "  "<<recv_buf[n].findcs.bks<<"  "<<recv_buf[n].findcs.bke<< std::endl;
std::cout <<"prol:" <<recv_buf[n].pindcs.bis<<"  "<<recv_buf[n].pindcs.bie<<
                "  "<<recv_buf[n].pindcs.bjs<<"  "<<recv_buf[n].pindcs.bje<<
                "  "<<recv_buf[n].pindcs.bks<<"  "<<recv_buf[n].pindcs.bke<< std::endl;
  }}
****/

  return;
}
