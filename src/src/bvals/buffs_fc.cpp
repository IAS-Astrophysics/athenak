//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffs_fc.cpp
//  \brief functions to allocate and initialize buffers for face-centered variables

#include <cstdlib>
#include <iostream>
#include <algorithm> // max

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesFC::InitSendIndices
//! \brief Calculates indices of cells used to pack buffers and send FC data for buffers
//! on same/coarser and finer levels.  Three sets of indices are needed for each of the
//! three components (x1f,x2f,x3f) of face-centered fields.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void MeshBoundaryValuesFC::InitSendIndices(MeshBoundaryBuffer &buf,
                                           int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae same as in LoadBoundaryBufferSameLevel() in src/bvals/fc/bvals_fc.cpp
  // for uniform grid: face-neighbors take care of the overlapping faces
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &isame = buf.isame;       // indices of buffer for neighbor same level
    if (ox1 == 0) {
      isame[0].bis = mb_indcs.is,           isame[0].bie = mb_indcs.ie + 1;
      isame[1].bis = mb_indcs.is,           isame[1].bie = mb_indcs.ie;
      isame[2].bis = mb_indcs.is,           isame[2].bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      isame[0].bis = mb_indcs.ie - ng1,     isame[0].bie = mb_indcs.ie;
      isame[1].bis = mb_indcs.ie - ng1,     isame[1].bie = mb_indcs.ie;
      isame[2].bis = mb_indcs.ie - ng1,     isame[2].bie = mb_indcs.ie;
    } else {
      isame[0].bis = mb_indcs.is + 1,       isame[0].bie = mb_indcs.is + ng;
      isame[1].bis = mb_indcs.is,           isame[1].bie = mb_indcs.is + ng1;
      isame[2].bis = mb_indcs.is,           isame[2].bie = mb_indcs.is + ng1;
    }
    if (ox2 == 0) {
      isame[0].bjs = mb_indcs.js,           isame[0].bje = mb_indcs.je;
      isame[1].bjs = mb_indcs.js,           isame[1].bje = mb_indcs.je + 1;
      isame[2].bjs = mb_indcs.js,           isame[2].bje = mb_indcs.je;
    } else if (ox2 > 0) {
      isame[0].bjs = mb_indcs.je - ng1,     isame[0].bje = mb_indcs.je;
      isame[1].bjs = mb_indcs.je - ng1,     isame[1].bje = mb_indcs.je;
      isame[2].bjs = mb_indcs.je - ng1,     isame[2].bje = mb_indcs.je;
    } else {
      isame[0].bjs = mb_indcs.js,           isame[0].bje = mb_indcs.js + ng1;
      isame[1].bjs = mb_indcs.js + 1,       isame[1].bje = mb_indcs.js + ng;
      isame[2].bjs = mb_indcs.js,           isame[2].bje = mb_indcs.js + ng1;
    }
    if (ox3 == 0) {
      isame[0].bks = mb_indcs.ks,           isame[0].bke = mb_indcs.ke;
      isame[1].bks = mb_indcs.ks,           isame[1].bke = mb_indcs.ke;
      isame[2].bks = mb_indcs.ks,           isame[2].bke = mb_indcs.ke + 1;
    } else if (ox3 > 0) {
      isame[0].bks = mb_indcs.ke - ng1,     isame[0].bke = mb_indcs.ke;
      isame[1].bks = mb_indcs.ke - ng1,     isame[1].bke = mb_indcs.ke;
      isame[2].bks = mb_indcs.ke - ng1,     isame[2].bke = mb_indcs.ke;
    } else {
      isame[0].bks = mb_indcs.ks,           isame[0].bke = mb_indcs.ks + ng1;
      isame[1].bks = mb_indcs.ks,           isame[1].bke = mb_indcs.ks + ng1;
      isame[2].bks = mb_indcs.ks + 1,       isame[2].bke = mb_indcs.ks + ng;
    }
    // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
    // x1f component on x1-faces
    if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
      if (ox1 > 0) {isame[0].bie++;}
      if (ox1 < 0) {isame[0].bis--;}
    }
    // x2f component on x2-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
      if (ox2 > 0) {isame[1].bje++;}
      if (ox2 < 0) {isame[1].bjs--;}
    }
    // x3f component on x3-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
      if (ox3 > 0) {isame[2].bke++;}
      if (ox3 < 0) {isame[2].bks--;}
    }
    for (int i=0; i<=2; ++i) {
      int ndat = (isame[i].bie - isame[i].bis + 1)*(isame[i].bje - isame[i].bjs + 1)*
                 (isame[i].bke - isame[i].bks + 1);
      buf.isame_ndat = std::max(buf.isame_ndat, ndat);
    }
  }

  // set indices for sends to neighbors on COARSER level (matches recv from FINER)
  // Formulae same as in LoadBoundaryBufferToCoarser() in src/bvals/fc/bvals_fc.cpp
  // Identical to send indices for same level replacing is,ie,.. with cis,cie,...
  {auto &icoar = buf.icoar;   // indices of buffer for neighbor coarser level
  if (ox1 == 0) {
    icoar[0].bis = mb_indcs.cis,          icoar[0].bie = mb_indcs.cie + 1;
    icoar[1].bis = mb_indcs.cis,          icoar[1].bie = mb_indcs.cie;
    icoar[2].bis = mb_indcs.cis,          icoar[2].bie = mb_indcs.cie;
  } else if (ox1 > 0) {
    icoar[0].bis = mb_indcs.cie - ng1,    icoar[0].bie = mb_indcs.cie;
    icoar[1].bis = mb_indcs.cie - ng1,    icoar[1].bie = mb_indcs.cie;
    icoar[2].bis = mb_indcs.cie - ng1,    icoar[2].bie = mb_indcs.cie;
  } else {
    icoar[0].bis = mb_indcs.cis + 1,      icoar[0].bie = mb_indcs.cis + ng;
    icoar[1].bis = mb_indcs.cis,          icoar[1].bie = mb_indcs.cis + ng1;
    icoar[2].bis = mb_indcs.cis,          icoar[2].bie = mb_indcs.cis + ng1;
  }
  if (ox2 == 0) {
    icoar[0].bjs = mb_indcs.cjs,          icoar[0].bje = mb_indcs.cje;
    icoar[1].bjs = mb_indcs.cjs,          icoar[1].bje = mb_indcs.cje + 1;
    icoar[2].bjs = mb_indcs.cjs,          icoar[2].bje = mb_indcs.cje;
  } else if (ox2 > 0) {
    icoar[0].bjs = mb_indcs.cje - ng1,    icoar[0].bje = mb_indcs.cje;
    icoar[1].bjs = mb_indcs.cje - ng1,    icoar[1].bje = mb_indcs.cje;
    icoar[2].bjs = mb_indcs.cje - ng1,    icoar[2].bje = mb_indcs.cje;
  } else {
    icoar[0].bjs = mb_indcs.cjs,          icoar[0].bje = mb_indcs.cjs + ng1;
    icoar[1].bjs = mb_indcs.cjs + 1,      icoar[1].bje = mb_indcs.cjs + ng;
    icoar[2].bjs = mb_indcs.cjs,          icoar[2].bje = mb_indcs.cjs + ng1;
  }
  if (ox3 == 0) {
    icoar[0].bks = mb_indcs.cks,          icoar[0].bke = mb_indcs.cke;
    icoar[1].bks = mb_indcs.cks,          icoar[1].bke = mb_indcs.cke;
    icoar[2].bks = mb_indcs.cks,          icoar[2].bke = mb_indcs.cke + 1;
  } else if (ox3 > 0) {
    icoar[0].bks = mb_indcs.cke - ng1,    icoar[0].bke = mb_indcs.cke;
    icoar[1].bks = mb_indcs.cke - ng1,    icoar[1].bke = mb_indcs.cke;
    icoar[2].bks = mb_indcs.cke - ng1,    icoar[2].bke = mb_indcs.cke;
  } else {
    icoar[0].bks = mb_indcs.cks,          icoar[0].bke = mb_indcs.cks + ng1;
    icoar[1].bks = mb_indcs.cks,          icoar[1].bke = mb_indcs.cks + ng1;
    icoar[2].bks = mb_indcs.cks + 1,      icoar[2].bke = mb_indcs.cks + ng;
  }
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
    if (ox1 > 0) {icoar[0].bie++;}
    if (ox1 < 0) {icoar[0].bis--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
    if (ox2 > 0) {icoar[1].bje++;}
    if (ox2 < 0) {icoar[1].bjs--;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
    if (ox3 > 0) {icoar[2].bke++;}
    if (ox3 < 0) {icoar[2].bks--;}
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (icoar[i].bie - icoar[i].bis + 1)*(icoar[i].bje - icoar[i].bjs + 1)*
               (icoar[i].bke - icoar[i].bks + 1);
    buf.icoar_ndat = std::max(buf.icoar_ndat, ndat);
  }
  }

  // set indices for sends to neighbors on FINER level (matches recv from COARSER)
  // Formulae same as in LoadBoundaryBufferToFiner() in src/bvals/fc/bvals_fc.cpp
  //
  // Subtle issue: shared face fields on edges of MeshBlock (B1 at [is,ie+1],
  // B2 at [js;je+1], B3 at [ks;ke+1]) are communicated, replacing values on coarse mesh
  // in target MeshBlock, but these values will only be used for prolongation.
  {auto &ifine = buf.ifine;    // indices of buffer for neighbor finer level
  int cnx1mng = mb_indcs.cnx1 - ng;
  int cnx2mng = mb_indcs.cnx2 - ng;
  int cnx3mng = mb_indcs.cnx3 - ng;
  if (ox1 == 0) {
    if (f1 == 1) {
      ifine[0].bis = mb_indcs.is + cnx1mng,  ifine[0].bie = mb_indcs.ie + 1;
      ifine[1].bis = mb_indcs.is + cnx1mng,  ifine[1].bie = mb_indcs.ie;
      ifine[2].bis = mb_indcs.is + cnx1mng,  ifine[2].bie = mb_indcs.ie;
    } else {
      ifine[0].bis = mb_indcs.is,            ifine[0].bie = mb_indcs.ie + 1 - cnx1mng;
      ifine[1].bis = mb_indcs.is,            ifine[1].bie = mb_indcs.ie - cnx1mng;
      ifine[2].bis = mb_indcs.is,            ifine[2].bie = mb_indcs.ie - cnx1mng;
    }
  } else if (ox1 > 0) {
    ifine[0].bis = mb_indcs.ie - ng1,    ifine[0].bie = mb_indcs.ie + 1;
    ifine[1].bis = mb_indcs.ie - ng1,    ifine[1].bie = mb_indcs.ie;
    ifine[2].bis = mb_indcs.ie - ng1,    ifine[2].bie = mb_indcs.ie;
  } else {
    ifine[0].bis = mb_indcs.is,          ifine[0].bie = mb_indcs.is + ng;
    ifine[1].bis = mb_indcs.is,          ifine[1].bie = mb_indcs.is + ng1;
    ifine[2].bis = mb_indcs.is,          ifine[2].bie = mb_indcs.is + ng1;
  }

  if (ox2 == 0) {
    ifine[0].bjs = mb_indcs.js,          ifine[0].bje = mb_indcs.je;
    ifine[1].bjs = mb_indcs.js,          ifine[1].bje = mb_indcs.je + 1;
    ifine[2].bjs = mb_indcs.js,          ifine[2].bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          ifine[0].bjs += cnx2mng;
          ifine[1].bjs += cnx2mng;
          ifine[2].bjs += cnx2mng;
        } else {
          ifine[0].bje -= cnx2mng;
          ifine[1].bje -= cnx2mng;
          ifine[2].bje -= cnx2mng;
        }
      } else {
        if (f2 == 1) {
          ifine[0].bjs += cnx2mng;
          ifine[1].bjs += cnx2mng;
          ifine[2].bjs += cnx2mng;
        } else {
          ifine[0].bje -= cnx2mng;
          ifine[1].bje -= cnx2mng;
          ifine[2].bje -= cnx2mng;
        }
      }
    }
  } else if (ox2 > 0) {
    ifine[0].bjs = mb_indcs.je - ng1,    ifine[0].bje = mb_indcs.je;
    ifine[1].bjs = mb_indcs.je - ng1,    ifine[1].bje = mb_indcs.je + 1;
    ifine[2].bjs = mb_indcs.je - ng1,    ifine[2].bje = mb_indcs.je;
  } else {
    ifine[0].bjs = mb_indcs.js,          ifine[0].bje = mb_indcs.js + ng1;
    ifine[1].bjs = mb_indcs.js,          ifine[1].bje = mb_indcs.js + ng;
    ifine[2].bjs = mb_indcs.js,          ifine[2].bje = mb_indcs.js + ng1;
  }

  if (ox3 == 0) {
    ifine[0].bks = mb_indcs.ks,          ifine[0].bke = mb_indcs.ke;
    ifine[1].bks = mb_indcs.ks,          ifine[1].bke = mb_indcs.ke;
    ifine[2].bks = mb_indcs.ks,          ifine[2].bke = mb_indcs.ke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          ifine[0].bks += cnx3mng;
          ifine[1].bks += cnx3mng;
          ifine[2].bks += cnx3mng;
        } else {
          ifine[0].bke -= cnx3mng;
          ifine[1].bke -= cnx3mng;
          ifine[2].bke -= cnx3mng;
        }
      } else {
        if (f2 == 1) {
          ifine[0].bks += cnx3mng;
          ifine[1].bks += cnx3mng;
          ifine[2].bks += cnx3mng;
        } else {
          ifine[0].bke -= cnx3mng;
          ifine[1].bke -= cnx3mng;
          ifine[2].bke -= cnx3mng;
        }
      }
    }
  } else if (ox3 > 0) {
    ifine[0].bks = mb_indcs.ke - ng1,    ifine[0].bke = mb_indcs.ke;
    ifine[1].bks = mb_indcs.ke - ng1,    ifine[1].bke = mb_indcs.ke;
    ifine[2].bks = mb_indcs.ke - ng1,    ifine[2].bke = mb_indcs.ke + 1;
  } else {
    ifine[0].bks = mb_indcs.ks,          ifine[0].bke = mb_indcs.ks + ng1;
    ifine[1].bks = mb_indcs.ks,          ifine[1].bke = mb_indcs.ks + ng1;
    ifine[2].bks = mb_indcs.ks,          ifine[2].bke = mb_indcs.ks + ng;
  }

  for (int i=0; i<=2; ++i) {
    int ndat = (ifine[i].bie - ifine[i].bis + 1)*(ifine[i].bje - ifine[i].bjs + 1)*
               (ifine[i].bke - ifine[i].bks + 1);
    buf.ifine_ndat = std::max(buf.ifine_ndat, ndat);
  }
  }

  // set indices for sends for FLUX CORRECTION to COARSER level
  // same as in LoadFluxBoundaryBufferToCoarser() in src/bvals/fc/flux_correction_fc.cpp
  {auto &iflxc = buf.iflux_coar;    // indices of buffer for flux correction
  if (ox1 == 0) {
    iflxc[0].bis = mb_indcs.cis,          iflxc[0].bie = mb_indcs.cie;
    iflxc[1].bis = mb_indcs.cis,          iflxc[1].bie = mb_indcs.cie + 1;
    iflxc[2].bis = mb_indcs.cis,          iflxc[2].bie = mb_indcs.cie + 1;
  } else if (ox1 > 0) {
    iflxc[0].bis = mb_indcs.cie + 1,      iflxc[0].bie = mb_indcs.cie + 1;
    iflxc[1].bis = mb_indcs.cie + 1;      iflxc[1].bie = mb_indcs.cie + 1;
    iflxc[2].bis = mb_indcs.cie + 1;      iflxc[2].bie = mb_indcs.cie + 1;
  } else {
    iflxc[0].bis = mb_indcs.cis;          iflxc[0].bie = mb_indcs.cis;
    iflxc[1].bis = mb_indcs.cis;          iflxc[1].bie = mb_indcs.cis;
    iflxc[2].bis = mb_indcs.cis;          iflxc[2].bie = mb_indcs.cis;
  }
  if (ox2 == 0) {
    iflxc[0].bjs = mb_indcs.cjs,          iflxc[0].bje = mb_indcs.cje + 1;
    iflxc[1].bjs = mb_indcs.cjs,          iflxc[1].bje = mb_indcs.cje;
    iflxc[2].bjs = mb_indcs.cjs,          iflxc[2].bje = mb_indcs.cje + 1;
  } else if (ox2 > 0) {
    iflxc[0].bjs = mb_indcs.cje + 1;      iflxc[0].bje = mb_indcs.cje + 1;
    iflxc[1].bjs = mb_indcs.cje + 1;      iflxc[1].bje = mb_indcs.cje + 1;
    iflxc[2].bjs = mb_indcs.cje + 1;      iflxc[2].bje = mb_indcs.cje + 1;
  } else {
    iflxc[0].bjs = mb_indcs.cjs;          iflxc[0].bje = mb_indcs.cjs;
    iflxc[1].bjs = mb_indcs.cjs;          iflxc[1].bje = mb_indcs.cjs;
    iflxc[2].bjs = mb_indcs.cjs;          iflxc[2].bje = mb_indcs.cjs;
  }
  if (ox3 == 0) {
    iflxc[0].bks = mb_indcs.cks,          iflxc[0].bke = mb_indcs.cke + 1;
    iflxc[1].bks = mb_indcs.cks,          iflxc[1].bke = mb_indcs.cke + 1;
    iflxc[2].bks = mb_indcs.cks,          iflxc[2].bke = mb_indcs.cke;
  } else if (ox3 > 0) {
    iflxc[0].bks = mb_indcs.cke + 1;      iflxc[0].bke = mb_indcs.cke + 1;
    iflxc[1].bks = mb_indcs.cke + 1;      iflxc[1].bke = mb_indcs.cke + 1;
    iflxc[2].bks = mb_indcs.cke + 1;      iflxc[2].bke = mb_indcs.cke + 1;
  } else {
    iflxc[0].bks = mb_indcs.cks;          iflxc[0].bke = mb_indcs.cks;
    iflxc[1].bks = mb_indcs.cks;          iflxc[1].bke = mb_indcs.cks;
    iflxc[2].bks = mb_indcs.cks;          iflxc[2].bke = mb_indcs.cks;
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (iflxc[i].bie - iflxc[i].bis + 1)*(iflxc[i].bje - iflxc[i].bjs + 1)*
               (iflxc[i].bke - iflxc[i].bks + 1);
    buf.iflxc_ndat = std::max(buf.iflxc_ndat, ndat);
  }
  }

  // set indices for sends for FLUX CORRECTION to SAME level
  // same as in LoadFluxBoundaryBufferSameLevel() in src/bvals/fc/flux_correction_fc.cpp
  {auto &iflxs = buf.iflux_same;    // indices of buffer for flux correction
  if (ox1 == 0) {
    iflxs[0].bis = mb_indcs.is,          iflxs[0].bie = mb_indcs.ie;
    iflxs[1].bis = mb_indcs.is,          iflxs[1].bie = mb_indcs.ie + 1;
    iflxs[2].bis = mb_indcs.is,          iflxs[2].bie = mb_indcs.ie + 1;
  } else if (ox1 > 0) {
    iflxs[0].bis = mb_indcs.ie + 1,      iflxs[0].bie = mb_indcs.ie + 1;
    iflxs[1].bis = mb_indcs.ie + 1;      iflxs[1].bie = mb_indcs.ie + 1;
    iflxs[2].bis = mb_indcs.ie + 1;      iflxs[2].bie = mb_indcs.ie + 1;
  } else {
    iflxs[0].bis = mb_indcs.is;          iflxs[0].bie = mb_indcs.is;
    iflxs[1].bis = mb_indcs.is;          iflxs[1].bie = mb_indcs.is;
    iflxs[2].bis = mb_indcs.is;          iflxs[2].bie = mb_indcs.is;
  }
  if (ox2 == 0) {
    iflxs[0].bjs = mb_indcs.js,          iflxs[0].bje = mb_indcs.je + 1;
    iflxs[1].bjs = mb_indcs.js,          iflxs[1].bje = mb_indcs.je;
    iflxs[2].bjs = mb_indcs.js,          iflxs[2].bje = mb_indcs.je + 1;
  } else if (ox2 > 0) {
    iflxs[0].bjs = mb_indcs.je + 1;      iflxs[0].bje = mb_indcs.je + 1;
    iflxs[1].bjs = mb_indcs.je + 1;      iflxs[1].bje = mb_indcs.je + 1;
    iflxs[2].bjs = mb_indcs.je + 1;      iflxs[2].bje = mb_indcs.je + 1;
  } else {
    iflxs[0].bjs = mb_indcs.js;          iflxs[0].bje = mb_indcs.js;
    iflxs[1].bjs = mb_indcs.js;          iflxs[1].bje = mb_indcs.js;
    iflxs[2].bjs = mb_indcs.js;          iflxs[2].bje = mb_indcs.js;
  }
  if (ox3 == 0) {
    iflxs[0].bks = mb_indcs.ks,          iflxs[0].bke = mb_indcs.ke + 1;
    iflxs[1].bks = mb_indcs.ks,          iflxs[1].bke = mb_indcs.ke + 1;
    iflxs[2].bks = mb_indcs.ks,          iflxs[2].bke = mb_indcs.ke;
  } else if (ox3 > 0) {
    iflxs[0].bks = mb_indcs.ke + 1;      iflxs[0].bke = mb_indcs.ke + 1;
    iflxs[1].bks = mb_indcs.ke + 1;      iflxs[1].bke = mb_indcs.ke + 1;
    iflxs[2].bks = mb_indcs.ke + 1;      iflxs[2].bke = mb_indcs.ke + 1;
  } else {
    iflxs[0].bks = mb_indcs.ks;          iflxs[0].bke = mb_indcs.ks;
    iflxs[1].bks = mb_indcs.ks;          iflxs[1].bke = mb_indcs.ks;
    iflxs[2].bks = mb_indcs.ks;          iflxs[2].bke = mb_indcs.ks;
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (iflxs[i].bie - iflxs[i].bis + 1)*(iflxs[i].bje - iflxs[i].bjs + 1)*
               (iflxs[i].bke - iflxs[i].bks + 1);
    buf.iflxs_ndat = std::max(buf.iflxs_ndat, ndat);
  }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesFC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for FC data
//! on same/coarser/finer levels, and for prolongation from coarse to fine.  Three sets of
//! indices are needed for each of the three components (x1f,x2f,x3f) of face-centered
//! fields.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void MeshBoundaryValuesFC::InitRecvIndices(MeshBoundaryBuffer &buf,
                                           int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;

  // set indices for receives from neighbors on SAME level
  // Formulae same as in SetBoundarySameLevel() in src/bvals/fc/bvals_fc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &isame = buf.isame;       // indices of buffer for neighbor same level
    if (ox1 == 0) {
      isame[0].bis = mb_indcs.is,         isame[0].bie = mb_indcs.ie + 1;
      isame[1].bis = mb_indcs.is,         isame[1].bie = mb_indcs.ie;
      isame[2].bis = mb_indcs.is,         isame[2].bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      isame[0].bis = mb_indcs.ie + 2,     isame[0].bie = mb_indcs.ie + ng + 1;
      isame[1].bis = mb_indcs.ie + 1,     isame[1].bie = mb_indcs.ie + ng;
      isame[2].bis = mb_indcs.ie + 1,     isame[2].bie = mb_indcs.ie + ng;
    } else {
      isame[0].bis = mb_indcs.is - ng,    isame[0].bie = mb_indcs.is - 1;
      isame[1].bis = mb_indcs.is - ng,    isame[1].bie = mb_indcs.is - 1;
      isame[2].bis = mb_indcs.is - ng,    isame[2].bie = mb_indcs.is - 1;
    }
    if (ox2 == 0) {
      isame[0].bjs = mb_indcs.js,          isame[0].bje = mb_indcs.je;
      isame[1].bjs = mb_indcs.js,          isame[1].bje = mb_indcs.je + 1;
      isame[2].bjs = mb_indcs.js,          isame[2].bje = mb_indcs.je;
    } else if (ox2 > 0) {
      isame[0].bjs = mb_indcs.je + 1,      isame[0].bje = mb_indcs.je + ng;
      isame[1].bjs = mb_indcs.je + 2,      isame[1].bje = mb_indcs.je + ng + 1;
      isame[2].bjs = mb_indcs.je + 1,      isame[2].bje = mb_indcs.je + ng;
    } else {
      isame[0].bjs = mb_indcs.js - ng,     isame[0].bje = mb_indcs.js - 1;
      isame[1].bjs = mb_indcs.js - ng,     isame[1].bje = mb_indcs.js - 1;
      isame[2].bjs = mb_indcs.js - ng,     isame[2].bje = mb_indcs.js - 1;
    }
    if (ox3 == 0) {
      isame[0].bks = mb_indcs.ks,          isame[0].bke = mb_indcs.ke;
      isame[1].bks = mb_indcs.ks,          isame[1].bke = mb_indcs.ke;
      isame[2].bks = mb_indcs.ks,          isame[2].bke = mb_indcs.ke + 1;
    } else if (ox3 > 0) {
      isame[0].bks = mb_indcs.ke + 1,      isame[0].bke = mb_indcs.ke + ng;
      isame[1].bks = mb_indcs.ke + 1,      isame[1].bke = mb_indcs.ke + ng;
      isame[2].bks = mb_indcs.ke + 2,      isame[2].bke = mb_indcs.ke + ng + 1;
    } else {
      isame[0].bks = mb_indcs.ks - ng,     isame[0].bke = mb_indcs.ks - 1;
      isame[1].bks = mb_indcs.ks - ng,     isame[1].bke = mb_indcs.ks - 1;
      isame[2].bks = mb_indcs.ks - ng,     isame[2].bke = mb_indcs.ks - 1;
    }
    // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
    // x1f component on x1-faces
    if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
      if (ox1 > 0) {isame[0].bis--;}
      if (ox1 < 0) {isame[0].bie++;}
    }
    // x2f component on x2-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
      if (ox2 > 0) {isame[1].bjs--;}
      if (ox2 < 0) {isame[1].bje++;}
    }
    // x3f component on x3-faces
    if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
      if (ox3 > 0) {isame[2].bks--;}
      if (ox3 < 0) {isame[2].bke++;}
    }
    for (int i=0; i<=2; ++i) {
      int ndat = (isame[i].bie - isame[i].bis + 1)*(isame[i].bje - isame[i].bjs + 1)*
                 (isame[i].bke - isame[i].bks + 1);
      buf.isame_ndat = std::max(buf.isame_ndat, ndat);
    }
  }

  // set indices for receives from neighbors on COARSER level (matches send to FINER)
  // Formulae same as in SetBoundaryFromCoarser() in src/bvals/fc/bvals_fc.cpp
  {auto &icoar = buf.icoar;   // indices of buffer for neighbor coarser level
  if (ox1 == 0) {
    icoar[0].bis = mb_indcs.cis,         icoar[0].bie = mb_indcs.cie + 1;
    icoar[1].bis = mb_indcs.cis,         icoar[1].bie = mb_indcs.cie;
    icoar[2].bis = mb_indcs.cis,         icoar[2].bie = mb_indcs.cie;
    if (f1 == 0) {
      icoar[0].bie += ng;
      icoar[1].bie += ng;
      icoar[2].bie += ng;
    } else {
      icoar[0].bis -= ng;
      icoar[1].bis -= ng;
      icoar[2].bis -= ng;
    }
  } else if (ox1 > 0) {
    icoar[0].bis = mb_indcs.cie + 1,     icoar[0].bie = mb_indcs.cie + ng + 1;
    icoar[1].bis = mb_indcs.cie + 1,     icoar[1].bie = mb_indcs.cie + ng;
    icoar[2].bis = mb_indcs.cie + 1,     icoar[2].bie = mb_indcs.cie + ng;
  } else {
    icoar[0].bis = mb_indcs.cis - ng,    icoar[0].bie = mb_indcs.cis;
    icoar[1].bis = mb_indcs.cis - ng,    icoar[1].bie = mb_indcs.cis - 1;
    icoar[2].bis = mb_indcs.cis - ng,    icoar[2].bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    icoar[0].bjs = mb_indcs.cjs,          icoar[0].bje = mb_indcs.cje;
    icoar[1].bjs = mb_indcs.cjs,          icoar[1].bje = mb_indcs.cje + 1;
    icoar[2].bjs = mb_indcs.cjs,          icoar[2].bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          icoar[0].bje += ng;
          icoar[1].bje += ng;
          icoar[2].bje += ng;
        } else {
          icoar[0].bjs -= ng;
          icoar[1].bjs -= ng;
          icoar[2].bjs -= ng;
        }
      } else {
        if (f2 == 0) {
          icoar[0].bje += ng;
          icoar[1].bje += ng;
          icoar[2].bje += ng;
        } else {
          icoar[0].bjs -= ng;
          icoar[1].bjs -= ng;
          icoar[2].bjs -= ng;
        }
      }
    }
  } else if (ox2 > 0) {
    icoar[0].bjs = mb_indcs.cje + 1,      icoar[0].bje = mb_indcs.cje + ng;
    icoar[1].bjs = mb_indcs.cje + 1,      icoar[1].bje = mb_indcs.cje + ng + 1;
    icoar[2].bjs = mb_indcs.cje + 1,      icoar[2].bje = mb_indcs.cje + ng;
  } else {
    icoar[0].bjs = mb_indcs.cjs - ng,     icoar[0].bje = mb_indcs.cjs - 1;
    icoar[1].bjs = mb_indcs.cjs - ng,     icoar[1].bje = mb_indcs.cjs;
    icoar[2].bjs = mb_indcs.cjs - ng,     icoar[2].bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    icoar[0].bks = mb_indcs.cks,          icoar[0].bke = mb_indcs.cke;
    icoar[1].bks = mb_indcs.cks,          icoar[1].bke = mb_indcs.cke;
    icoar[2].bks = mb_indcs.cks,          icoar[2].bke = mb_indcs.cke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          icoar[0].bke += ng;
          icoar[1].bke += ng;
          icoar[2].bke += ng;
        } else {
          icoar[0].bks -= ng;
          icoar[1].bks -= ng;
          icoar[2].bks -= ng;
        }
      } else {
        if (f2 == 0) {
          icoar[0].bke += ng;
          icoar[1].bke += ng;
          icoar[2].bke += ng;
        } else {
          icoar[0].bks -= ng;
          icoar[1].bks -= ng;
          icoar[2].bks -= ng;
        }
      }
    }
  } else if (ox3 > 0) {
    icoar[0].bks = mb_indcs.cke + 1,      icoar[0].bke = mb_indcs.cke + ng;
    icoar[1].bks = mb_indcs.cke + 1,      icoar[1].bke = mb_indcs.cke + ng;
    icoar[2].bks = mb_indcs.cke + 1,      icoar[2].bke = mb_indcs.cke + ng + 1;
  } else {
    icoar[0].bks = mb_indcs.cks - ng,     icoar[0].bke = mb_indcs.cks - 1;
    icoar[1].bks = mb_indcs.cks - ng,     icoar[1].bke = mb_indcs.cks - 1;
    icoar[2].bks = mb_indcs.cks - ng,     icoar[2].bke = mb_indcs.cks;
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (icoar[i].bie - icoar[i].bis + 1)*(icoar[i].bje - icoar[i].bjs + 1)*
               (icoar[i].bke - icoar[i].bks + 1);
    buf.icoar_ndat = std::max(buf.icoar_ndat, ndat);
  }
  }

  // set indices for receives from neighbors on FINER level (matches send to COARSER)
  // Formulae same as in SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &ifine = buf.ifine;   // indices of buffer for neighbor finer level
  if (ox1 == 0) {
    ifine[0].bis = mb_indcs.is;               ifine[0].bie = mb_indcs.ie + 1;
    ifine[1].bis = mb_indcs.is;               ifine[1].bie = mb_indcs.ie;
    ifine[2].bis = mb_indcs.is;               ifine[2].bie = mb_indcs.ie;
    if (f1 == 1) {
      ifine[0].bis += mb_indcs.cnx1;
      ifine[1].bis += mb_indcs.cnx1;
      ifine[2].bis += mb_indcs.cnx1;
    } else {
      ifine[0].bie -= mb_indcs.cnx1;
      ifine[1].bie -= mb_indcs.cnx1;
      ifine[2].bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    ifine[0].bis = mb_indcs.ie + 2;           ifine[0].bie = mb_indcs.ie + ng + 1;
    ifine[1].bis = mb_indcs.ie + 1;           ifine[1].bie = mb_indcs.ie + ng;
    ifine[2].bis = mb_indcs.ie + 1;           ifine[2].bie = mb_indcs.ie + ng;
  } else {
    ifine[0].bis = mb_indcs.is - ng;          ifine[0].bie = mb_indcs.is - 1;
    ifine[1].bis = mb_indcs.is - ng;          ifine[1].bie = mb_indcs.is - 1;
    ifine[2].bis = mb_indcs.is - ng;          ifine[2].bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    ifine[0].bjs = mb_indcs.js;             ifine[0].bje = mb_indcs.je;
    ifine[1].bjs = mb_indcs.js;             ifine[1].bje = mb_indcs.je + 1;
    ifine[2].bjs = mb_indcs.js;             ifine[2].bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          ifine[0].bjs += mb_indcs.cnx2;
          ifine[1].bjs += mb_indcs.cnx2;
          ifine[2].bjs += mb_indcs.cnx2;
        } else {
          ifine[0].bje -= mb_indcs.cnx2;
          ifine[1].bje -= mb_indcs.cnx2;
          ifine[2].bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          ifine[0].bjs += mb_indcs.cnx2;
          ifine[1].bjs += mb_indcs.cnx2;
          ifine[2].bjs += mb_indcs.cnx2;
        } else {
          ifine[0].bje -= mb_indcs.cnx2;
          ifine[1].bje -= mb_indcs.cnx2;
          ifine[2].bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    ifine[0].bjs = mb_indcs.je + 1;          ifine[0].bje = mb_indcs.je + ng;
    ifine[1].bjs = mb_indcs.je + 2;          ifine[1].bje = mb_indcs.je + ng + 1;
    ifine[2].bjs = mb_indcs.je + 1;          ifine[2].bje = mb_indcs.je + ng;
  } else {
    ifine[0].bjs = mb_indcs.js - ng;         ifine[0].bje = mb_indcs.js - 1;
    ifine[1].bjs = mb_indcs.js - ng;         ifine[1].bje = mb_indcs.js - 1;
    ifine[2].bjs = mb_indcs.js - ng;         ifine[2].bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    ifine[0].bks = mb_indcs.ks;              ifine[0].bke = mb_indcs.ke;
    ifine[1].bks = mb_indcs.ks;              ifine[1].bke = mb_indcs.ke;
    ifine[2].bks = mb_indcs.ks;              ifine[2].bke = mb_indcs.ke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          ifine[0].bks += mb_indcs.cnx3;
          ifine[1].bks += mb_indcs.cnx3;
          ifine[2].bks += mb_indcs.cnx3;
        } else {
          ifine[0].bke -= mb_indcs.cnx3;
          ifine[1].bke -= mb_indcs.cnx3;
          ifine[2].bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          ifine[0].bks += mb_indcs.cnx3;
          ifine[1].bks += mb_indcs.cnx3;
          ifine[2].bks += mb_indcs.cnx3;
        } else {
          ifine[0].bke -= mb_indcs.cnx3;
          ifine[1].bke -= mb_indcs.cnx3;
          ifine[2].bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    ifine[0].bks = mb_indcs.ke + 1;         ifine[0].bke = mb_indcs.ke + ng;
    ifine[1].bks = mb_indcs.ke + 1;         ifine[1].bke = mb_indcs.ke + ng;
    ifine[2].bks = mb_indcs.ke + 2;         ifine[2].bke = mb_indcs.ke + ng + 1;
  } else {
    ifine[0].bks = mb_indcs.ks - ng;        ifine[0].bke = mb_indcs.ks - 1;
    ifine[1].bks = mb_indcs.ks - ng;        ifine[1].bke = mb_indcs.ks - 1;
    ifine[2].bks = mb_indcs.ks - ng;        ifine[2].bke = mb_indcs.ks - 1;
  }
  // for SMR/AMR, always include the overlapping faces in edge and corner boundaries
  if (pmy_pack->pmesh->multilevel && (ox2 != 0 || ox3 != 0)) {
    if (ox1 > 0) {ifine[0].bis--;}
    if (ox1 < 0) {ifine[0].bie++;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox3 != 0)) {
    if (ox2 > 0) {ifine[1].bjs--;}
    if (ox2 < 0) {ifine[1].bje++;}
  }
  if (pmy_pack->pmesh->multilevel && (ox1 != 0 || ox2 != 0)) {
    if (ox3 > 0) {ifine[2].bks--;}
    if (ox3 < 0) {ifine[2].bke++;}
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (ifine[i].bie - ifine[i].bis + 1)*(ifine[i].bje - ifine[i].bjs + 1)*
               (ifine[i].bke - ifine[i].bks + 1);
    buf.ifine_ndat = std::max(buf.ifine_ndat, ndat);
  }
  }

  // set indices for PROLONGATION in coarse cell buffers. Indices refer to coarse cells.
  // Formulae same as in ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  //
  // Subtle issue: NOT the same as receives from coarser level (with ng --> ng/2) since
  // latter sends face fields on edges of MeshBlock, but prolongation only occurs within
  // ghost cells (and NOT for B1 at [is;ie+1], B2 at [js;je+1], B3 at [ke;ke+1])
  {auto &iprol = buf.iprol;   // indices for prolongation
  int cn = mb_indcs.ng/2;   // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    iprol[0].bis = mb_indcs.cis;          iprol[0].bie = mb_indcs.cie + 1;
    iprol[1].bis = mb_indcs.cis;          iprol[1].bie = mb_indcs.cie;
    iprol[2].bis = mb_indcs.cis;          iprol[2].bie = mb_indcs.cie;
    if (f1 == 0) {
      iprol[0].bie += cn;
      iprol[1].bie += cn;
      iprol[2].bie += cn;
    } else {
      iprol[0].bis -= cn;
      iprol[1].bis -= cn;
      iprol[2].bis -= cn;
    }
  } else if (ox1 > 0)  {
    iprol[0].bis = mb_indcs.cie + 2;       iprol[0].bie = mb_indcs.cie + cn + 1;
    iprol[1].bis = mb_indcs.cie + 1;       iprol[1].bie = mb_indcs.cie + cn;
    iprol[2].bis = mb_indcs.cie + 1;       iprol[2].bie = mb_indcs.cie + cn;
  } else {
    iprol[0].bis = mb_indcs.cis - cn;      iprol[0].bie = mb_indcs.cis - 1;
    iprol[1].bis = mb_indcs.cis - cn;      iprol[1].bie = mb_indcs.cis - 1;
    iprol[2].bis = mb_indcs.cis - cn;      iprol[2].bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    iprol[0].bjs = mb_indcs.cjs;           iprol[0].bje = mb_indcs.cje;
    iprol[1].bjs = mb_indcs.cjs;           iprol[1].bje = mb_indcs.cje + 1;
    iprol[2].bjs = mb_indcs.cjs;           iprol[2].bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          iprol[0].bje += cn;
          iprol[1].bje += cn;
          iprol[2].bje += cn;
        } else {
          iprol[0].bjs -= cn;
          iprol[1].bjs -= cn;
          iprol[2].bjs -= cn;
        }
      } else {
        if (f2 == 0) {
          iprol[0].bje += cn;
          iprol[1].bje += cn;
          iprol[2].bje += cn;
        } else {
          iprol[0].bjs -= cn;
          iprol[1].bjs -= cn;
          iprol[2].bjs -= cn;
        }
      }
    }
  } else if (ox2 > 0) {
    iprol[0].bjs = mb_indcs.cje + 1;        iprol[0].bje = mb_indcs.cje + cn;
    iprol[1].bjs = mb_indcs.cje + 2;        iprol[1].bje = mb_indcs.cje + cn + 1;
    iprol[2].bjs = mb_indcs.cje + 1;        iprol[2].bje = mb_indcs.cje + cn;
  } else {
    iprol[0].bjs = mb_indcs.cjs - cn;       iprol[0].bje = mb_indcs.cjs - 1;
    iprol[1].bjs = mb_indcs.cjs - cn;       iprol[1].bje = mb_indcs.cjs - 1;
    iprol[2].bjs = mb_indcs.cjs - cn;       iprol[2].bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    iprol[0].bks = mb_indcs.cks;            iprol[0].bke = mb_indcs.cke;
    iprol[1].bks = mb_indcs.cks;            iprol[1].bke = mb_indcs.cke;
    iprol[2].bks = mb_indcs.cks;            iprol[2].bke = mb_indcs.cke + 1;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          iprol[0].bke += cn;
          iprol[1].bke += cn;
          iprol[2].bke += cn;
        } else {
          iprol[0].bks -= cn;
          iprol[1].bks -= cn;
          iprol[2].bks -= cn;
        }
      } else {
        if (f2 == 0) {
          iprol[0].bke += cn;
          iprol[1].bke += cn;
          iprol[2].bke += cn;
        } else {
          iprol[0].bks -= cn;
          iprol[1].bks -= cn;
          iprol[2].bks -= cn;
        }
      }
    }
  } else if (ox3 > 0)  {
    iprol[0].bks = mb_indcs.cke + 1;          iprol[0].bke = mb_indcs.cke + cn;
    iprol[1].bks = mb_indcs.cke + 1;          iprol[1].bke = mb_indcs.cke + cn;
    iprol[2].bks = mb_indcs.cke + 2;          iprol[2].bke = mb_indcs.cke + cn + 1;
  } else {
    iprol[0].bks = mb_indcs.cks - cn;         iprol[0].bke = mb_indcs.cks - 1;
    iprol[1].bks = mb_indcs.cks - cn;         iprol[1].bke = mb_indcs.cks - 1;
    iprol[2].bks = mb_indcs.cks - cn;         iprol[2].bke = mb_indcs.cks - 1;
  }
  }

  // set indices for receives for flux-correction from FINER level.  Similar to send,
  // except data loaded into appropriate sub-block of coarse buffer (similar to receive
  // from FINER level).
  // Formulae same as in SetFluxBoundaryFromFiner() in src/bvals/fc/flux_correction_fc.cpp
  {auto &iflxc = buf.iflux_coar;   // indices of buffer for flux correction
  if (ox1 == 0) {
    iflxc[0].bis = mb_indcs.is;             iflxc[0].bie = mb_indcs.ie;
    iflxc[1].bis = mb_indcs.is;             iflxc[1].bie = mb_indcs.ie + 1;
    iflxc[2].bis = mb_indcs.is;             iflxc[2].bie = mb_indcs.ie + 1;
    if (f1 == 1) {
      iflxc[0].bis += mb_indcs.cnx1;
      iflxc[1].bis += mb_indcs.cnx1;
      iflxc[2].bis += mb_indcs.cnx1;
    } else {
      iflxc[0].bie -= mb_indcs.cnx1;
      iflxc[1].bie -= mb_indcs.cnx1;
      iflxc[2].bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    iflxc[0].bis = mb_indcs.ie + 1,         iflxc[0].bie = mb_indcs.ie + 1;
    iflxc[1].bis = mb_indcs.ie + 1,         iflxc[1].bie = mb_indcs.ie + 1;
    iflxc[2].bis = mb_indcs.ie + 1,         iflxc[2].bie = mb_indcs.ie + 1;
  } else {
    iflxc[0].bis = mb_indcs.is,             iflxc[0].bie = mb_indcs.is;
    iflxc[1].bis = mb_indcs.is,             iflxc[1].bie = mb_indcs.is;
    iflxc[2].bis = mb_indcs.is,             iflxc[2].bie = mb_indcs.is;
  }
  if (ox2 == 0) {
    iflxc[0].bjs = mb_indcs.js;             iflxc[0].bje = mb_indcs.je + 1;
    iflxc[1].bjs = mb_indcs.js;             iflxc[1].bje = mb_indcs.je;
    iflxc[2].bjs = mb_indcs.js;             iflxc[2].bje = mb_indcs.je + 1;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          iflxc[0].bjs += mb_indcs.cnx2;
          iflxc[1].bjs += mb_indcs.cnx2;
          iflxc[2].bjs += mb_indcs.cnx2;
        } else {
          iflxc[0].bje -= mb_indcs.cnx2;
          iflxc[1].bje -= mb_indcs.cnx2;
          iflxc[2].bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          iflxc[0].bjs += mb_indcs.cnx2;
          iflxc[1].bjs += mb_indcs.cnx2;
          iflxc[2].bjs += mb_indcs.cnx2;
        } else {
          iflxc[0].bje -= mb_indcs.cnx2;
          iflxc[1].bje -= mb_indcs.cnx2;
          iflxc[2].bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    iflxc[0].bjs = mb_indcs.je + 1,         iflxc[0].bje = mb_indcs.je + 1;
    iflxc[1].bjs = mb_indcs.je + 1,         iflxc[1].bje = mb_indcs.je + 1;
    iflxc[2].bjs = mb_indcs.je + 1,         iflxc[2].bje = mb_indcs.je + 1;
  } else {
    iflxc[0].bjs = mb_indcs.js,             iflxc[0].bje = mb_indcs.js;
    iflxc[1].bjs = mb_indcs.js,             iflxc[1].bje = mb_indcs.js;
    iflxc[2].bjs = mb_indcs.js,             iflxc[2].bje = mb_indcs.js;
  }
  if (ox3 == 0) {
    iflxc[0].bks = mb_indcs.ks;             iflxc[0].bke = mb_indcs.ke + 1;
    iflxc[1].bks = mb_indcs.ks;             iflxc[1].bke = mb_indcs.ke + 1;
    iflxc[2].bks = mb_indcs.ks;             iflxc[2].bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          iflxc[0].bks += mb_indcs.cnx3;
          iflxc[1].bks += mb_indcs.cnx3;
          iflxc[2].bks += mb_indcs.cnx3;
        } else {
          iflxc[0].bke -= mb_indcs.cnx3;
          iflxc[1].bke -= mb_indcs.cnx3;
          iflxc[2].bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          iflxc[0].bks += mb_indcs.cnx3;
          iflxc[1].bks += mb_indcs.cnx3;
          iflxc[2].bks += mb_indcs.cnx3;
        } else {
          iflxc[0].bke -= mb_indcs.cnx3;
          iflxc[1].bke -= mb_indcs.cnx3;
          iflxc[2].bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    iflxc[0].bks = mb_indcs.ke + 1,      iflxc[0].bke = mb_indcs.ke + 1;
    iflxc[1].bks = mb_indcs.ke + 1,      iflxc[1].bke = mb_indcs.ke + 1;
    iflxc[2].bks = mb_indcs.ke + 1,      iflxc[2].bke = mb_indcs.ke + 1;
  } else {
    iflxc[0].bks = mb_indcs.ks,          iflxc[0].bke = mb_indcs.ks;
    iflxc[1].bks = mb_indcs.ks,          iflxc[1].bke = mb_indcs.ks;
    iflxc[2].bks = mb_indcs.ks,          iflxc[2].bke = mb_indcs.ks;
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (iflxc[i].bie - iflxc[i].bis + 1)*(iflxc[i].bje - iflxc[i].bjs + 1)*
               (iflxc[i].bke - iflxc[i].bks + 1);
    buf.iflxc_ndat = std::max(buf.iflxc_ndat, ndat);
  }
  }

  // set indices for receives for flux-correction at SAME level.
  // Formulae same as in SetFluxSameLevel() in src/bvals/fc/flux_correction_fc.cpp
  {auto &iflxs = buf.iflux_same;   // indices of buffer for flux correction
  if (ox1 == 0) {
    iflxs[0].bis = mb_indcs.is;             iflxs[0].bie = mb_indcs.ie;
    iflxs[1].bis = mb_indcs.is;             iflxs[1].bie = mb_indcs.ie + 1;
    iflxs[2].bis = mb_indcs.is;             iflxs[2].bie = mb_indcs.ie + 1;
  } else if (ox1 > 0) {
    iflxs[0].bis = mb_indcs.ie + 1,         iflxs[0].bie = mb_indcs.ie + 1;
    iflxs[1].bis = mb_indcs.ie + 1,         iflxs[1].bie = mb_indcs.ie + 1;
    iflxs[2].bis = mb_indcs.ie + 1,         iflxs[2].bie = mb_indcs.ie + 1;
  } else {
    iflxs[0].bis = mb_indcs.is,             iflxs[0].bie = mb_indcs.is;
    iflxs[1].bis = mb_indcs.is,             iflxs[1].bie = mb_indcs.is;
    iflxs[2].bis = mb_indcs.is,             iflxs[2].bie = mb_indcs.is;
  }
  if (ox2 == 0) {
    iflxs[0].bjs = mb_indcs.js;             iflxs[0].bje = mb_indcs.je + 1;
    iflxs[1].bjs = mb_indcs.js;             iflxs[1].bje = mb_indcs.je;
    iflxs[2].bjs = mb_indcs.js;             iflxs[2].bje = mb_indcs.je + 1;
  } else if (ox2 > 0) {
    iflxs[0].bjs = mb_indcs.je + 1,         iflxs[0].bje = mb_indcs.je + 1;
    iflxs[1].bjs = mb_indcs.je + 1,         iflxs[1].bje = mb_indcs.je + 1;
    iflxs[2].bjs = mb_indcs.je + 1,         iflxs[2].bje = mb_indcs.je + 1;
  } else {
    iflxs[0].bjs = mb_indcs.js,             iflxs[0].bje = mb_indcs.js;
    iflxs[1].bjs = mb_indcs.js,             iflxs[1].bje = mb_indcs.js;
    iflxs[2].bjs = mb_indcs.js,             iflxs[2].bje = mb_indcs.js;
  }
  if (ox3 == 0) {
    iflxs[0].bks = mb_indcs.ks;             iflxs[0].bke = mb_indcs.ke + 1;
    iflxs[1].bks = mb_indcs.ks;             iflxs[1].bke = mb_indcs.ke + 1;
    iflxs[2].bks = mb_indcs.ks;             iflxs[2].bke = mb_indcs.ke;
  } else if (ox3 > 0) {
    iflxs[0].bks = mb_indcs.ke + 1,      iflxs[0].bke = mb_indcs.ke + 1;
    iflxs[1].bks = mb_indcs.ke + 1,      iflxs[1].bke = mb_indcs.ke + 1;
    iflxs[2].bks = mb_indcs.ke + 1,      iflxs[2].bke = mb_indcs.ke + 1;
  } else {
    iflxs[0].bks = mb_indcs.ks,          iflxs[0].bke = mb_indcs.ks;
    iflxs[1].bks = mb_indcs.ks,          iflxs[1].bke = mb_indcs.ks;
    iflxs[2].bks = mb_indcs.ks,          iflxs[2].bke = mb_indcs.ks;
  }
  for (int i=0; i<=2; ++i) {
    int ndat = (iflxs[i].bie - iflxs[i].bis + 1)*(iflxs[i].bje - iflxs[i].bjs + 1)*
               (iflxs[i].bke - iflxs[i].bks + 1);
    buf.iflxs_ndat = std::max(buf.iflxs_ndat, ndat);
  }
  }
}
