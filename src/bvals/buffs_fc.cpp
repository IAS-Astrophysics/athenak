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

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesFC::InitSendIndices
//! \brief Calculates indices of cells used to pack buffers and send FC data for buffers
//! on same/coarser and finer levels.  Three sets of indices are needed for each of the
//! three components (x1f,x2f,x3f) of face-centered fields.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BoundaryValuesFC::InitSendIndices(
     BoundaryBuffer &buf, int ox1, int ox2, int ox3, int f1, int f2)
{
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/fc/bvals_fc.cpp
  // for uniform grid: face-neighbors take care of the overlapping faces
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &same = buf.same;       // indices of buffer for neighbor same level
    if (ox1 == 0) {
      same[0].bis = mb_indcs.is,           same[0].bie = mb_indcs.ie + 1;
      same[1].bis = mb_indcs.is,           same[1].bie = mb_indcs.ie;
      same[2].bis = mb_indcs.is,           same[2].bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      same[0].bis = mb_indcs.ie - ng1,     same[0].bie = mb_indcs.ie;
      same[1].bis = mb_indcs.ie - ng1,     same[1].bie = mb_indcs.ie;
      same[2].bis = mb_indcs.ie - ng1,     same[2].bie = mb_indcs.ie;
    } else {
      same[0].bis = mb_indcs.is + 1,       same[0].bie = mb_indcs.is + ng;
      same[1].bis = mb_indcs.is,           same[1].bie = mb_indcs.is + ng1;
      same[2].bis = mb_indcs.is,           same[2].bie = mb_indcs.is + ng1;
    }
    if (ox2 == 0) {
      same[0].bjs = mb_indcs.js,           same[0].bje = mb_indcs.je;
      same[1].bjs = mb_indcs.js,           same[1].bje = mb_indcs.je + 1;
      same[2].bjs = mb_indcs.js,           same[2].bje = mb_indcs.je;
    } else if (ox2 > 0) {
      same[0].bjs = mb_indcs.je - ng1,     same[0].bje = mb_indcs.je;
      same[1].bjs = mb_indcs.je - ng1,     same[1].bje = mb_indcs.je;
      same[2].bjs = mb_indcs.je - ng1,     same[2].bje = mb_indcs.je;
    } else {
      same[0].bjs = mb_indcs.js,           same[0].bje = mb_indcs.js + ng1;
      same[1].bjs = mb_indcs.js + 1,       same[1].bje = mb_indcs.js + ng;
      same[2].bjs = mb_indcs.js,           same[2].bje = mb_indcs.js + ng1;
    }
    if (ox3 == 0) {
      same[0].bks = mb_indcs.ks,           same[0].bke = mb_indcs.ke;
      same[1].bks = mb_indcs.ks,           same[1].bke = mb_indcs.ke;
      same[2].bks = mb_indcs.ks,           same[2].bke = mb_indcs.ke + 1;
    } else if (ox3 > 0) {
      same[0].bks = mb_indcs.ke - ng1,     same[0].bke = mb_indcs.ke;
      same[1].bks = mb_indcs.ke - ng1,     same[1].bke = mb_indcs.ke;
      same[2].bks = mb_indcs.ke - ng1,     same[2].bke = mb_indcs.ke;
    } else {
      same[0].bks = mb_indcs.ks,           same[0].bke = mb_indcs.ks + ng1;
      same[1].bks = mb_indcs.ks,           same[1].bke = mb_indcs.ks + ng1;
      same[2].bks = mb_indcs.ks + 1,       same[2].bke = mb_indcs.ks + ng;
    }
    // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
    // x1f component on x1-faces
    if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
      if (ox1 > 0) {same[0].bie++;}
      if (ox1 < 0) {same[0].bis--;}
    }
    // x2f component on x2-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
      if (ox2 > 0) {same[1].bje++;}
      if (ox2 < 0) {same[1].bjs--;}
    }
    // x3f component on x3-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
      if (ox3 > 0) {same[2].bke++;}
      if (ox3 < 0) {same[2].bks--;}
    }
    for (int i=0; i<=2; ++i) {
      same[i].ndat = (same[i].bie - same[i].bis + 1)*
                     (same[i].bje - same[i].bjs + 1)*
                     (same[i].bke - same[i].bks + 1);
    }
  }

  // set indices for sends to neighbors on COARSER level (matches recv from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/fc/bvals_fc.cpp
  // Identical to send indices for same level replacing is,ie,.. with cis,cie,...
  {auto &coar = buf.coar;   // indices of buffer for neighbor coarser level
  if (ox1 == 0) { 
    coar[0].bis = mb_indcs.cis,          coar[0].bie = mb_indcs.cie + 1;
    coar[1].bis = mb_indcs.cis,          coar[1].bie = mb_indcs.cie;
    coar[2].bis = mb_indcs.cis,          coar[2].bie = mb_indcs.cie;
  } else if (ox1 > 0) {
    coar[0].bis = mb_indcs.cie - ng1,    coar[0].bie = mb_indcs.cie;
    coar[1].bis = mb_indcs.cie - ng1,    coar[1].bie = mb_indcs.cie;
    coar[2].bis = mb_indcs.cie - ng1,    coar[2].bie = mb_indcs.cie;
  } else {
    coar[0].bis = mb_indcs.cis + 1,      coar[0].bie = mb_indcs.cis + ng;
    coar[1].bis = mb_indcs.cis,          coar[1].bie = mb_indcs.cis + ng1;
    coar[2].bis = mb_indcs.cis,          coar[2].bie = mb_indcs.cis + ng1;
  }
  if (ox2 == 0) { 
    coar[0].bjs = mb_indcs.cjs,          coar[0].bje = mb_indcs.cje;
    coar[1].bjs = mb_indcs.cjs,          coar[1].bje = mb_indcs.cje + 1;
    coar[2].bjs = mb_indcs.cjs,          coar[2].bje = mb_indcs.cje;
  } else if (ox2 > 0) {
    coar[0].bjs = mb_indcs.cje - ng1,    coar[0].bje = mb_indcs.cje;
    coar[1].bjs = mb_indcs.cje - ng1,    coar[1].bje = mb_indcs.cje;
    coar[2].bjs = mb_indcs.cje - ng1,    coar[2].bje = mb_indcs.cje;
  } else {
    coar[0].bjs = mb_indcs.cjs,          coar[0].bje = mb_indcs.cjs + ng1;
    coar[1].bjs = mb_indcs.cjs + 1,      coar[1].bje = mb_indcs.cjs + ng;
    coar[2].bjs = mb_indcs.cjs,          coar[2].bje = mb_indcs.cjs + ng1;
  }
  if (ox3 == 0) {
    coar[0].bks = mb_indcs.cks,          coar[0].bke = mb_indcs.cke;
    coar[1].bks = mb_indcs.cks,          coar[1].bke = mb_indcs.cke;
    coar[2].bks = mb_indcs.cks,          coar[2].bke = mb_indcs.cke + 1;
  } else if (ox3 > 0) {
    coar[0].bks = mb_indcs.cke - ng1,    coar[0].bke = mb_indcs.cke;
    coar[1].bks = mb_indcs.cke - ng1,    coar[1].bke = mb_indcs.cke;
    coar[2].bks = mb_indcs.cke - ng1,    coar[2].bke = mb_indcs.cke;
  } else {              
    coar[0].bks = mb_indcs.cks,          coar[0].bke = mb_indcs.cks + ng1;
    coar[1].bks = mb_indcs.cks,          coar[1].bke = mb_indcs.cks + ng1;
    coar[2].bks = mb_indcs.cks + 1,      coar[2].bke = mb_indcs.cks + ng;
  }
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
    if (ox1 > 0) {coar[0].bie++;}
    if (ox1 < 0) {coar[0].bis--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
    if (ox2 > 0) {coar[1].bje++;}
    if (ox2 < 0) {coar[1].bjs--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
    if (ox3 > 0) {coar[2].bke++;}
    if (ox3 < 0) {coar[2].bks--;}
  }
  for (int i=0; i<=2; ++i) {
    coar[i].ndat = (coar[i].bie - coar[i].bis + 1)*
                   (coar[i].bje - coar[i].bjs + 1)*
                   (coar[i].bke - coar[i].bks + 1);
  }}

  // set indices for sends to neighbors on FINER level (matches recv from COARSER)
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/fc/bvals_fc.cpp
  //
  // Subtle issue: shared face fields on edges of MeshBlock (B1 at [is,ie+1],
  // B2 at [js;je+1], B3 at [ks;ke+1]) are communicated, replacing values on coarse mesh
  // in target MeshBlock, but these values will only be used for prolongation.
  {auto &fine = buf.fine;    // indices of buffer for neighbor finer level
  int cnx1mng = mb_indcs.cnx1 - ng;
  int cnx2mng = mb_indcs.cnx2 - ng;
  int cnx3mng = mb_indcs.cnx3 - ng;
  if (ox1 == 0) {
    if (f1 == 1) {
      fine[0].bis = mb_indcs.is + cnx1mng,  fine[0].bie = mb_indcs.ie + 1;
      fine[1].bis = mb_indcs.is + cnx1mng,  fine[1].bie = mb_indcs.ie;
      fine[2].bis = mb_indcs.is + cnx1mng,  fine[2].bie = mb_indcs.ie;
    } else {
      fine[0].bis = mb_indcs.is,            fine[0].bie = mb_indcs.ie + 1 - cnx1mng;
      fine[1].bis = mb_indcs.is,            fine[1].bie = mb_indcs.ie - cnx1mng;
      fine[2].bis = mb_indcs.is,            fine[2].bie = mb_indcs.ie - cnx1mng;
    }
  } else if (ox1 > 0) {
    fine[0].bis = mb_indcs.ie - ng1,    fine[0].bie = mb_indcs.ie + 1;
    fine[1].bis = mb_indcs.ie - ng1,    fine[1].bie = mb_indcs.ie;
    fine[2].bis = mb_indcs.ie - ng1,    fine[2].bie = mb_indcs.ie;
  } else {
    fine[0].bis = mb_indcs.is,          fine[0].bie = mb_indcs.is + ng;
    fine[1].bis = mb_indcs.is,          fine[1].bie = mb_indcs.is + ng1;
    fine[2].bis = mb_indcs.is,          fine[2].bie = mb_indcs.is + ng1;
  }

  if (ox2 == 0) {
    fine[0].bjs = mb_indcs.js,          fine[0].bje = mb_indcs.je;
    fine[1].bjs = mb_indcs.js,          fine[1].bje = mb_indcs.je + 1;
    fine[2].bjs = mb_indcs.js,          fine[2].bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          fine[0].bjs += cnx2mng;
          fine[1].bjs += cnx2mng;
          fine[2].bjs += cnx2mng;
        } else {
          fine[0].bje -= cnx2mng;
          fine[1].bje -= cnx2mng;
          fine[2].bje -= cnx2mng;
        }
      } else {
        if (f2 == 1) {
          fine[0].bjs += cnx2mng;
          fine[1].bjs += cnx2mng;
          fine[2].bjs += cnx2mng;
        } else {
          fine[0].bje -= cnx2mng;
          fine[1].bje -= cnx2mng;
          fine[2].bje -= cnx2mng;
        }
      }
    }
  } else if (ox2 > 0) {
    fine[0].bjs = mb_indcs.je - ng1,    fine[0].bje = mb_indcs.je;
    fine[1].bjs = mb_indcs.je - ng1,    fine[1].bje = mb_indcs.je + 1;
    fine[2].bjs = mb_indcs.je - ng1,    fine[2].bje = mb_indcs.je;
  } else {
    fine[0].bjs = mb_indcs.js,          fine[0].bje = mb_indcs.js + ng1;
    fine[1].bjs = mb_indcs.js,          fine[1].bje = mb_indcs.js + ng;
    fine[2].bjs = mb_indcs.js,          fine[2].bje = mb_indcs.js + ng1;
  }

  if (ox3 == 0) {
    fine[0].bks = mb_indcs.ks,          fine[0].bke = mb_indcs.ke;
    fine[1].bks = mb_indcs.ks,          fine[1].bke = mb_indcs.ke;
    fine[2].bks = mb_indcs.ks,          fine[2].bke = mb_indcs.ke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          fine[0].bks += cnx3mng;
          fine[1].bks += cnx3mng;
          fine[2].bks += cnx3mng;
        } else {
          fine[0].bke -= cnx3mng;
          fine[1].bke -= cnx3mng;
          fine[2].bke -= cnx3mng;
        }
      } else {
        if (f2 == 1) {
          fine[0].bks += cnx3mng;
          fine[1].bks += cnx3mng;
          fine[2].bks += cnx3mng;
        } else {
          fine[0].bke -= cnx3mng;
          fine[1].bke -= cnx3mng;
          fine[2].bke -= cnx3mng;
        }
      }
    }
  } else if (ox3 > 0) {
    fine[0].bks = mb_indcs.ke - ng1,    fine[0].bke = mb_indcs.ke;
    fine[1].bks = mb_indcs.ke - ng1,    fine[1].bke = mb_indcs.ke;
    fine[2].bks = mb_indcs.ke - ng1,    fine[2].bke = mb_indcs.ke + 1;
  } else {
    fine[0].bks = mb_indcs.ks,          fine[0].bke = mb_indcs.ks + ng1;
    fine[1].bks = mb_indcs.ks,          fine[1].bke = mb_indcs.ks + ng1;
    fine[2].bks = mb_indcs.ks,          fine[2].bke = mb_indcs.ks + ng;
  }

  for (int i=0; i<=2; ++i) {
    fine[i].ndat = (fine[i].bie - fine[i].bis + 1)*
                   (fine[i].bje - fine[i].bjs + 1)*
                   (fine[i].bke - fine[i].bks + 1);
  }}

  // set indices for sends for FLUX CORRECTION (sends always to COARSER level)
  {auto &flux = buf.flux;    // indices of buffer for flux correction
  if (ox1 == 0) {
    flux[0].bis = mb_indcs.cis,          flux[0].bie = mb_indcs.cie;
    flux[1].bis = mb_indcs.cis,          flux[1].bie = mb_indcs.cie + 1;
    flux[2].bis = mb_indcs.cis,          flux[2].bie = mb_indcs.cie + 1;
  } else if (ox1 > 0) {
    flux[1].bis = mb_indcs.cie + 1;      flux[1].bie = mb_indcs.cie + 1;
    flux[2].bis = mb_indcs.cie + 1;      flux[2].bie = mb_indcs.cie + 1;
  } else {
    flux[1].bis = mb_indcs.cis;          flux[1].bie = mb_indcs.cis;
    flux[2].bis = mb_indcs.cis;          flux[2].bie = mb_indcs.cis;
  }
  if (ox2 == 0) {
    flux[0].bjs = mb_indcs.cjs,          flux[0].bje = mb_indcs.cje + 1;
    flux[1].bjs = mb_indcs.cjs,          flux[1].bje = mb_indcs.cje;
    flux[2].bjs = mb_indcs.cjs,          flux[2].bje = mb_indcs.cje + 1;
  } else if (ox2 > 0) {
    flux[0].bjs = mb_indcs.cje + 1;      flux[0].bje = mb_indcs.cje + 1;
    flux[2].bjs = mb_indcs.cje + 1;      flux[2].bje = mb_indcs.cje + 1;
  } else {
    flux[0].bjs = mb_indcs.cjs;          flux[0].bje = mb_indcs.cjs;
    flux[2].bjs = mb_indcs.cjs;          flux[2].bje = mb_indcs.cjs;
  }
  if (ox3 == 0) {
    flux[0].bks = mb_indcs.cks,          flux[0].bke = mb_indcs.cke + 1;
    flux[1].bks = mb_indcs.cks,          flux[1].bke = mb_indcs.cke + 1;
    flux[2].bks = mb_indcs.cks,          flux[2].bke = mb_indcs.cke;
  } else if (ox3 > 0) {
    flux[0].bks = mb_indcs.cke + 1;      flux[0].bke = mb_indcs.cke + 1;
    flux[1].bks = mb_indcs.cke + 1;      flux[1].bke = mb_indcs.cke + 1;
  } else {
    flux[0].bks = mb_indcs.cks;          flux[0].bke = mb_indcs.cks;
    flux[1].bks = mb_indcs.cks;          flux[1].bke = mb_indcs.cks;
  }
  for (int i=0; i<=2; ++i) {
    flux[i].ndat = (flux[i].bie - flux[i].bis + 1)*
                   (flux[i].bje - flux[i].bjs + 1)*
                   (flux[i].bke - flux[i].bks + 1);
  }}

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesFC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for FC data
//! on same/coarser/finer levels, and for prolongation from coarse to fine.  Three sets of
//! indices are needed for each of the three components (x1f,x2f,x3f) of face-centered
//! fields.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BoundaryValuesFC::InitRecvIndices(
     BoundaryBuffer &buf, int ox1, int ox2, int ox3, int f1, int f2)
{ 
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;

  // set indices for receives from neighbors on SAME level
  // Formulae taken from SetBoundarySameLevel() in src/bvals/fc/bvals_fc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &same = buf.same;       // indices of buffer for neighbor same level
    if (ox1 == 0) {
      same[0].bis = mb_indcs.is,         same[0].bie = mb_indcs.ie + 1;
      same[1].bis = mb_indcs.is,         same[1].bie = mb_indcs.ie;
      same[2].bis = mb_indcs.is,         same[2].bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      same[0].bis = mb_indcs.ie + 2,     same[0].bie = mb_indcs.ie + ng + 1;
      same[1].bis = mb_indcs.ie + 1,     same[1].bie = mb_indcs.ie + ng;
      same[2].bis = mb_indcs.ie + 1,     same[2].bie = mb_indcs.ie + ng;
    } else {
      same[0].bis = mb_indcs.is - ng,    same[0].bie = mb_indcs.is - 1;
      same[1].bis = mb_indcs.is - ng,    same[1].bie = mb_indcs.is - 1;
      same[2].bis = mb_indcs.is - ng,    same[2].bie = mb_indcs.is - 1;
    }
    if (ox2 == 0) {
      same[0].bjs = mb_indcs.js,          same[0].bje = mb_indcs.je;
      same[1].bjs = mb_indcs.js,          same[1].bje = mb_indcs.je + 1;
      same[2].bjs = mb_indcs.js,          same[2].bje = mb_indcs.je;
    } else if (ox2 > 0) {
      same[0].bjs = mb_indcs.je + 1,      same[0].bje = mb_indcs.je + ng;
      same[1].bjs = mb_indcs.je + 2,      same[1].bje = mb_indcs.je + ng + 1;
      same[2].bjs = mb_indcs.je + 1,      same[2].bje = mb_indcs.je + ng;
    } else {
      same[0].bjs = mb_indcs.js - ng,     same[0].bje = mb_indcs.js - 1;
      same[1].bjs = mb_indcs.js - ng,     same[1].bje = mb_indcs.js - 1;
      same[2].bjs = mb_indcs.js - ng,     same[2].bje = mb_indcs.js - 1;
    }
    if (ox3 == 0) {
      same[0].bks = mb_indcs.ks,          same[0].bke = mb_indcs.ke;
      same[1].bks = mb_indcs.ks,          same[1].bke = mb_indcs.ke;
      same[2].bks = mb_indcs.ks,          same[2].bke = mb_indcs.ke + 1;
    } else if (ox3 > 0) {
      same[0].bks = mb_indcs.ke + 1,      same[0].bke = mb_indcs.ke + ng;
      same[1].bks = mb_indcs.ke + 1,      same[1].bke = mb_indcs.ke + ng;
      same[2].bks = mb_indcs.ke + 2,      same[2].bke = mb_indcs.ke + ng + 1;
    } else {
      same[0].bks = mb_indcs.ks - ng,     same[0].bke = mb_indcs.ks - 1;
      same[1].bks = mb_indcs.ks - ng,     same[1].bke = mb_indcs.ks - 1;
      same[2].bks = mb_indcs.ks - ng,     same[2].bke = mb_indcs.ks - 1;
    }
    // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
    // x1f component on x1-faces
    if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
      if (ox1 > 0) {same[0].bis--;}
      if (ox1 < 0) {same[0].bie++;}
    }
    // x2f component on x2-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
      if (ox2 > 0) {same[1].bjs--;}
      if (ox2 < 0) {same[1].bje++;}
    }
    // x3f component on x3-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
      if (ox3 > 0) {same[2].bks--;}
      if (ox3 < 0) {same[2].bke++;}
    }
    for (int i=0; i<=2; ++i) {
      same[i].ndat = (same[i].bie - same[i].bis + 1)*
                     (same[i].bje - same[i].bjs + 1)*
                     (same[i].bke - same[i].bks + 1);
    }
  }

  // set indices for receives from neighbors on COARSER level (matches send to FINER)
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/fc/bvals_fc.cpp
  {auto &coar = buf.coar;   // indices of buffer for neighbor coarser level
  if (ox1 == 0) {
    coar[0].bis = mb_indcs.cis,         coar[0].bie = mb_indcs.cie + 1;
    coar[1].bis = mb_indcs.cis,         coar[1].bie = mb_indcs.cie;
    coar[2].bis = mb_indcs.cis,         coar[2].bie = mb_indcs.cie;
    if (f1 == 0) {
      coar[0].bie += ng;
      coar[1].bie += ng;
      coar[2].bie += ng;
    } else {
      coar[0].bis -= ng;
      coar[1].bis -= ng;
      coar[2].bis -= ng;
    }
  } else if (ox1 > 0) {
    coar[0].bis = mb_indcs.cie + 1,     coar[0].bie = mb_indcs.cie + ng + 1;
    coar[1].bis = mb_indcs.cie + 1,     coar[1].bie = mb_indcs.cie + ng;
    coar[2].bis = mb_indcs.cie + 1,     coar[2].bie = mb_indcs.cie + ng;
  } else {
    coar[0].bis = mb_indcs.cis - ng,    coar[0].bie = mb_indcs.cis;
    coar[1].bis = mb_indcs.cis - ng,    coar[1].bie = mb_indcs.cis - 1;
    coar[2].bis = mb_indcs.cis - ng,    coar[2].bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    coar[0].bjs = mb_indcs.cjs,          coar[0].bje = mb_indcs.cje;
    coar[1].bjs = mb_indcs.cjs,          coar[1].bje = mb_indcs.cje + 1;
    coar[2].bjs = mb_indcs.cjs,          coar[2].bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          coar[0].bje += ng;
          coar[1].bje += ng;
          coar[2].bje += ng;
        } else {
          coar[0].bjs -= ng;
          coar[1].bjs -= ng;
          coar[2].bjs -= ng;
        }
      } else {
        if (f2 == 0) {
          coar[0].bje += ng;
          coar[1].bje += ng;
          coar[2].bje += ng;
        } else {
          coar[0].bjs -= ng;
          coar[1].bjs -= ng;
          coar[2].bjs -= ng;
        }
      }
    }
  } else if (ox2 > 0) {
    coar[0].bjs = mb_indcs.cje + 1,      coar[0].bje = mb_indcs.cje + ng;
    coar[1].bjs = mb_indcs.cje + 1,      coar[1].bje = mb_indcs.cje + ng + 1;
    coar[2].bjs = mb_indcs.cje + 1,      coar[2].bje = mb_indcs.cje + ng;
  } else {
    coar[0].bjs = mb_indcs.cjs - ng,     coar[0].bje = mb_indcs.cjs - 1;
    coar[1].bjs = mb_indcs.cjs - ng,     coar[1].bje = mb_indcs.cjs;
    coar[2].bjs = mb_indcs.cjs - ng,     coar[2].bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    coar[0].bks = mb_indcs.cks,          coar[0].bke = mb_indcs.cke;
    coar[1].bks = mb_indcs.cks,          coar[1].bke = mb_indcs.cke;
    coar[2].bks = mb_indcs.cks,          coar[2].bke = mb_indcs.cke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          coar[0].bke += ng;
          coar[1].bke += ng;
          coar[2].bke += ng;
        } else {
          coar[0].bks -= ng;
          coar[1].bks -= ng;
          coar[2].bks -= ng;
        }
      } else {
        if (f2 == 0) {
          coar[0].bke += ng;
          coar[1].bke += ng;
          coar[2].bke += ng;
        } else {
          coar[0].bks -= ng;
          coar[1].bks -= ng;
          coar[2].bks -= ng;
        }
      }
    }
  } else if (ox3 > 0) {
    coar[0].bks = mb_indcs.cke + 1,      coar[0].bke = mb_indcs.cke + ng;
    coar[1].bks = mb_indcs.cke + 1,      coar[1].bke = mb_indcs.cke + ng;
    coar[2].bks = mb_indcs.cke + 1,      coar[2].bke = mb_indcs.cke + ng + 1;
  } else {
    coar[0].bks = mb_indcs.cks - ng,     coar[0].bke = mb_indcs.cks - 1;
    coar[1].bks = mb_indcs.cks - ng,     coar[1].bke = mb_indcs.cks - 1;
    coar[2].bks = mb_indcs.cks - ng,     coar[2].bke = mb_indcs.cks;
  }
  for (int i=0; i<=2; ++i) {
    coar[i].ndat = (coar[i].bie - coar[i].bis + 1)*
                   (coar[i].bje - coar[i].bjs + 1)*
                   (coar[i].bke - coar[i].bks + 1);
  }}

  // set indices for receives from neighbors on FINER level (matches send to COARSER)
  // Formulae taken from SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &fine = buf.fine;   // indices of buffer for neighbor finer level
  if (ox1 == 0) {
    fine[0].bis = mb_indcs.is;               fine[0].bie = mb_indcs.ie + 1;
    fine[1].bis = mb_indcs.is;               fine[1].bie = mb_indcs.ie;
    fine[2].bis = mb_indcs.is;               fine[2].bie = mb_indcs.ie;
    if (f1 == 1) {
      fine[0].bis += mb_indcs.cnx1;
      fine[1].bis += mb_indcs.cnx1;
      fine[2].bis += mb_indcs.cnx1;
    } else {
      fine[0].bie -= mb_indcs.cnx1;
      fine[1].bie -= mb_indcs.cnx1;
      fine[2].bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    fine[0].bis = mb_indcs.ie + 2;           fine[0].bie = mb_indcs.ie + ng + 1;
    fine[1].bis = mb_indcs.ie + 1;           fine[1].bie = mb_indcs.ie + ng;
    fine[2].bis = mb_indcs.ie + 1;           fine[2].bie = mb_indcs.ie + ng;
  } else {
    fine[0].bis = mb_indcs.is - ng;          fine[0].bie = mb_indcs.is - 1;
    fine[1].bis = mb_indcs.is - ng;          fine[1].bie = mb_indcs.is - 1;
    fine[2].bis = mb_indcs.is - ng;          fine[2].bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    fine[0].bjs = mb_indcs.js;             fine[0].bje = mb_indcs.je;
    fine[1].bjs = mb_indcs.js;             fine[1].bje = mb_indcs.je + 1;
    fine[2].bjs = mb_indcs.js;             fine[2].bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          fine[0].bjs += mb_indcs.cnx2;
          fine[1].bjs += mb_indcs.cnx2;
          fine[2].bjs += mb_indcs.cnx2;
        } else {
          fine[0].bje -= mb_indcs.cnx2;
          fine[1].bje -= mb_indcs.cnx2;
          fine[2].bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          fine[0].bjs += mb_indcs.cnx2;
          fine[1].bjs += mb_indcs.cnx2;
          fine[2].bjs += mb_indcs.cnx2;
        } else {
          fine[0].bje -= mb_indcs.cnx2;
          fine[1].bje -= mb_indcs.cnx2;
          fine[2].bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    fine[0].bjs = mb_indcs.je + 1;          fine[0].bje = mb_indcs.je + ng;
    fine[1].bjs = mb_indcs.je + 2;          fine[1].bje = mb_indcs.je + ng + 1;
    fine[2].bjs = mb_indcs.je + 1;          fine[2].bje = mb_indcs.je + ng;
  } else {
    fine[0].bjs = mb_indcs.js - ng;         fine[0].bje = mb_indcs.js - 1;
    fine[1].bjs = mb_indcs.js - ng;         fine[1].bje = mb_indcs.js - 1;
    fine[2].bjs = mb_indcs.js - ng;         fine[2].bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    fine[0].bks = mb_indcs.ks;              fine[0].bke = mb_indcs.ke;
    fine[1].bks = mb_indcs.ks;              fine[1].bke = mb_indcs.ke;
    fine[2].bks = mb_indcs.ks;              fine[2].bke = mb_indcs.ke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          fine[0].bks += mb_indcs.cnx3;
          fine[1].bks += mb_indcs.cnx3;
          fine[2].bks += mb_indcs.cnx3;
        } else {
          fine[0].bke -= mb_indcs.cnx3;
          fine[1].bke -= mb_indcs.cnx3;
          fine[2].bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          fine[0].bks += mb_indcs.cnx3;
          fine[1].bks += mb_indcs.cnx3;
          fine[2].bks += mb_indcs.cnx3;
        } else {
          fine[0].bke -= mb_indcs.cnx3;
          fine[1].bke -= mb_indcs.cnx3;
          fine[2].bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    fine[0].bks = mb_indcs.ke + 1;         fine[0].bke = mb_indcs.ke + ng;
    fine[1].bks = mb_indcs.ke + 1;         fine[1].bke = mb_indcs.ke + ng;
    fine[2].bks = mb_indcs.ke + 2;         fine[2].bke = mb_indcs.ke + ng + 1;
  } else {
    fine[0].bks = mb_indcs.ks - ng;        fine[0].bke = mb_indcs.ks - 1;
    fine[1].bks = mb_indcs.ks - ng;        fine[1].bke = mb_indcs.ks - 1;
    fine[2].bks = mb_indcs.ks - ng;        fine[2].bke = mb_indcs.ks - 1;
  }
  for (int i=0; i<=2; ++i) {
    fine[i].ndat = (fine[i].bie - fine[i].bis + 1)*
                   (fine[i].bje - fine[i].bjs + 1)*
                   (fine[i].bke - fine[i].bks + 1);
  }}

  // set indices for PROLONGATION in coarse cell buffers. Indices refer to coarse cells.
  // Formulae taken from ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  //
  // Subtle issue: NOT the same as receives from coarser level (with ng --> ng/2) since
  // latter sends face fields on edges of MeshBlock, but prolongation only occurs within
  // ghost cells (and NOT for B1 at [is;ie+1], B2 at [js;je+1], B3 at [ke;ke+1])
  {auto &prol = buf.prol;   // indices for prolongation
  int cn = mb_indcs.ng/2;   // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    prol[0].bis = mb_indcs.cis;          prol[0].bie = mb_indcs.cie + 1;
    prol[1].bis = mb_indcs.cis;          prol[1].bie = mb_indcs.cie;
    prol[2].bis = mb_indcs.cis;          prol[2].bie = mb_indcs.cie;
    if (f1 == 0) {
      prol[0].bie += cn;
      prol[1].bie += cn;
      prol[2].bie += cn;
    } else {
      prol[0].bis -= cn;
      prol[1].bis -= cn;
      prol[2].bis -= cn;
    }
  } else if (ox1 > 0)  {
    prol[0].bis = mb_indcs.cie + 2;       prol[0].bie = mb_indcs.cie + cn + 1;
    prol[1].bis = mb_indcs.cie + 1;       prol[1].bie = mb_indcs.cie + cn;
    prol[2].bis = mb_indcs.cie + 1;       prol[2].bie = mb_indcs.cie + cn;
  } else {
    prol[0].bis = mb_indcs.cis - cn;      prol[0].bie = mb_indcs.cis - 1;
    prol[1].bis = mb_indcs.cis - cn;      prol[1].bie = mb_indcs.cis - 1;
    prol[2].bis = mb_indcs.cis - cn;      prol[2].bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    prol[0].bjs = mb_indcs.cjs;           prol[0].bje = mb_indcs.cje;
    prol[1].bjs = mb_indcs.cjs;           prol[1].bje = mb_indcs.cje + 1;
    prol[2].bjs = mb_indcs.cjs;           prol[2].bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          prol[0].bje += cn;
          prol[1].bje += cn;
          prol[2].bje += cn;
        } else {
          prol[0].bjs -= cn;
          prol[1].bjs -= cn;
          prol[2].bjs -= cn;
        }
      } else {
        if (f2 == 0) {
          prol[0].bje += cn;
          prol[1].bje += cn;
          prol[2].bje += cn;
        } else {
          prol[0].bjs -= cn;
          prol[1].bjs -= cn;
          prol[2].bjs -= cn;
        }
      }
    }
  } else if (ox2 > 0) {
    prol[0].bjs = mb_indcs.cje + 1;        prol[0].bje = mb_indcs.cje + cn;
    prol[1].bjs = mb_indcs.cje + 2;        prol[1].bje = mb_indcs.cje + cn + 1;
    prol[2].bjs = mb_indcs.cje + 1;        prol[2].bje = mb_indcs.cje + cn;
  } else {
    prol[0].bjs = mb_indcs.cjs - cn;       prol[0].bje = mb_indcs.cjs - 1;
    prol[1].bjs = mb_indcs.cjs - cn;       prol[1].bje = mb_indcs.cjs - 1;
    prol[2].bjs = mb_indcs.cjs - cn;       prol[2].bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    prol[0].bks = mb_indcs.cks;            prol[0].bke = mb_indcs.cke;
    prol[1].bks = mb_indcs.cks;            prol[1].bke = mb_indcs.cke;
    prol[2].bks = mb_indcs.cks;            prol[2].bke = mb_indcs.cke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          prol[0].bke += cn;
          prol[1].bke += cn;
          prol[2].bke += cn;
        } else {
          prol[0].bks -= cn;
          prol[1].bks -= cn;
          prol[2].bks -= cn;
        }
      } else {
        if (f2 == 0) {
          prol[0].bke += cn;
          prol[1].bke += cn;
          prol[2].bke += cn;
        } else {
          prol[0].bks -= cn;
          prol[1].bks -= cn;
          prol[2].bks -= cn;
        }
      }
    }
  } else if (ox3 > 0)  {
    prol[0].bks = mb_indcs.cke + 1;          prol[0].bke = mb_indcs.cke + cn;
    prol[1].bks = mb_indcs.cke + 1;          prol[1].bke = mb_indcs.cke + cn;
    prol[2].bks = mb_indcs.cke + 2;          prol[2].bke = mb_indcs.cke + cn + 1;
  } else {
    prol[0].bks = mb_indcs.cks - cn;         prol[0].bke = mb_indcs.cks - 1;
    prol[1].bks = mb_indcs.cks - cn;         prol[1].bke = mb_indcs.cks - 1;
    prol[2].bks = mb_indcs.cks - cn;         prol[2].bke = mb_indcs.cks - 1;
  }
  for (int i=0; i<=2; ++i) {
    prol[i].ndat = (prol[i].bie - prol[i].bis + 1)*
                   (prol[i].bje - prol[i].bjs + 1)*
                   (prol[i].bke - prol[i].bks + 1);
  }}

  // set indices for receives for flux-correction.  Similar to send, except data loaded
  // into appropriate sub-block of coarse buffer (similar to receive from FINER level)
  {auto &flux = buf.flux;   // indices of buffer for flux correction
  if (ox1 == 0) {
    flux[0].bis = mb_indcs.is;             flux[0].bie = mb_indcs.ie;
    flux[1].bis = mb_indcs.is;             flux[1].bie = mb_indcs.ie + 1;
    flux[2].bis = mb_indcs.is;             flux[2].bie = mb_indcs.ie + 1;
    if (f1 == 1) {
      flux[0].bis += mb_indcs.cnx1;
      flux[1].bis += mb_indcs.cnx1;
      flux[2].bis += mb_indcs.cnx1;
    } else {
      flux[0].bie -= mb_indcs.cnx1;
      flux[1].bie -= mb_indcs.cnx1;
      flux[2].bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    flux[1].bis = mb_indcs.ie + 1,         flux[1].bie = mb_indcs.ie + 1;
    flux[2].bis = mb_indcs.ie + 1,         flux[2].bie = mb_indcs.ie + 1;
  } else {
    flux[1].bis = mb_indcs.is,             flux[1].bie = mb_indcs.is;
    flux[2].bis = mb_indcs.is,             flux[2].bie = mb_indcs.is;
  }
  if (ox2 == 0) {
    flux[0].bjs = mb_indcs.js;             flux[0].bje = mb_indcs.je + 1;
    flux[1].bjs = mb_indcs.js;             flux[1].bje = mb_indcs.je;
    flux[2].bjs = mb_indcs.js;             flux[2].bje = mb_indcs.je + 1;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          flux[0].bjs += mb_indcs.cnx2;
          flux[1].bjs += mb_indcs.cnx2;
          flux[2].bjs += mb_indcs.cnx2;
        } else {
          flux[0].bje -= mb_indcs.cnx2;
          flux[1].bje -= mb_indcs.cnx2;
          flux[2].bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          flux[0].bjs += mb_indcs.cnx2;
          flux[1].bjs += mb_indcs.cnx2;
          flux[2].bjs += mb_indcs.cnx2;
        } else {
          flux[0].bje -= mb_indcs.cnx2;
          flux[1].bje -= mb_indcs.cnx2;
          flux[2].bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    flux[0].bjs = mb_indcs.je + 1,         flux[0].bje = mb_indcs.je + 1;
    flux[2].bjs = mb_indcs.je + 1,         flux[2].bje = mb_indcs.je + 1;
  } else {
    flux[0].bjs = mb_indcs.js,             flux[0].bje = mb_indcs.js;
    flux[2].bjs = mb_indcs.js,             flux[2].bje = mb_indcs.js;
  }
  if (ox3 == 0) {
    flux[0].bks = mb_indcs.ks;             flux[0].bke = mb_indcs.ke + 1;
    flux[1].bks = mb_indcs.ks;             flux[1].bke = mb_indcs.ke + 1;
    flux[2].bks = mb_indcs.ks;             flux[2].bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          flux[0].bks += mb_indcs.cnx3;
          flux[1].bks += mb_indcs.cnx3;
          flux[2].bks += mb_indcs.cnx3;
        } else {
          flux[0].bke -= mb_indcs.cnx3;
          flux[1].bke -= mb_indcs.cnx3;
          flux[2].bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          flux[0].bks += mb_indcs.cnx3;
          flux[1].bks += mb_indcs.cnx3;
          flux[2].bks += mb_indcs.cnx3;
        } else {
          flux[0].bke -= mb_indcs.cnx3;
          flux[1].bke -= mb_indcs.cnx3;
          flux[2].bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    flux[0].bks = mb_indcs.ke + 1,      flux[0].bke = mb_indcs.ke + 1;
    flux[1].bks = mb_indcs.ke + 1,      flux[1].bke = mb_indcs.ke + 1;
  } else {
    flux[0].bks = mb_indcs.ks,          flux[0].bke = mb_indcs.ks;
    flux[1].bks = mb_indcs.ks,          flux[1].bke = mb_indcs.ks;
  }
  for (int i=0; i<=2; ++i) {
    flux[i].ndat = (flux[i].bie - flux[i].bis + 1)*
                   (flux[i].bje - flux[i].bjs + 1)*
                   (flux[i].bke - flux[i].bks + 1);
  }}

}
