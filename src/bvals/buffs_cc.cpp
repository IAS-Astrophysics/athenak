//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffs_cc.cpp
//  \brief functions to allocate and initialize buffers for cell-centered variables

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesCC::InitSendIndices
//! \brief Calculates indices of cells used to pack buffers and send CC data for buffers
//! on same/coarser/finer levels. Only one set of indices is needed, so only first [0]
//! component of each index array is used.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BoundaryValuesCC::InitSendIndices(
     BoundaryBuffer &buf, int ox1, int ox2, int ox3, int f1, int f2)
{
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/cc/bvals_cc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &same = buf.same[0];    // indices of buffer for neighbor same level
    same.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
    same.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
    same.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
    same.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
    same.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
    same.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
    same.ndat = (same.bie - same.bis + 1)*(same.bje - same.bjs + 1)*
                (same.bke - same.bks + 1);
  }

  // set indices for sends to neighbors on COARSER level (matches recvs from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &coar = buf.coar[0];  // indices of buffer for neighbor coarser level
  coar.bis = (ox1 > 0) ? (mb_indcs.cie - ng1) : mb_indcs.cis;
  coar.bie = (ox1 < 0) ? (mb_indcs.cis + ng1) : mb_indcs.cie;
  coar.bjs = (ox2 > 0) ? (mb_indcs.cje - ng1) : mb_indcs.cjs;
  coar.bje = (ox2 < 0) ? (mb_indcs.cjs + ng1) : mb_indcs.cje;
  coar.bks = (ox3 > 0) ? (mb_indcs.cke - ng1) : mb_indcs.cks;
  coar.bke = (ox3 < 0) ? (mb_indcs.cks + ng1) : mb_indcs.cke;
  coar.ndat = (coar.bie - coar.bis + 1)*(coar.bje - coar.bjs + 1)*
              (coar.bke - coar.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recvs from COARSER)
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/cc/bvals_cc.cpp
  {auto &fine = buf.fine[0];  // indices of buffer for neighbor finer level
  fine.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  fine.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  fine.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  fine.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  fine.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  fine.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
  // need to add internal edges on faces, and internal corners on edges
  if (ox1 == 0) {
    if (f1 == 1) {
      fine.bis += mb_indcs.cnx1 - ng;
    } else {
      fine.bie -= mb_indcs.cnx1 - ng;
    }
  }
  if (ox2 == 0 && mb_indcs.nx2 > 1) {
    if (ox1 != 0) {
      if (f1 == 1) {
        fine.bjs += mb_indcs.cnx2 - ng;
      } else {
        fine.bje -= mb_indcs.cnx2 - ng;
      }
    } else {
      if (f2 == 1) {
        fine.bjs += mb_indcs.cnx2 - ng;
      } else {
        fine.bje -= mb_indcs.cnx2 - ng;
      }
    }
  }
  if (ox3 == 0 && mb_indcs.nx3 > 1) {
    if (ox1 != 0 && ox2 != 0) {
      if (f1 == 1) {
        fine.bks += mb_indcs.cnx3 - ng;
      } else {
        fine.bke -= mb_indcs.cnx3 - ng;
      }
    } else {
      if (f2 == 1) {
        fine.bks += mb_indcs.cnx3 - ng;
      } else {
        fine.bke -= mb_indcs.cnx3 - ng;
      }
    }
  }
  fine.ndat = (fine.bie - fine.bis + 1)*(fine.bje - fine.bjs + 1)*
              (fine.bke - fine.bks + 1);
  }

  // set indices for sends for FLUX CORRECTION (sends always fine to coarse)
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for flux corr (e.g. #0,4,8,12,...)
    auto &flux = buf.flux[0];    // indices of buffer for flux correction
    flux.bis = (ox1 > 0) ? (mb_indcs.cie + 1) : mb_indcs.cis;
    flux.bie = (ox1 < 0) ? (mb_indcs.cis    ) : mb_indcs.cie;
    flux.bjs = (ox2 > 0) ? (mb_indcs.cje + 1) : mb_indcs.cjs;
    flux.bje = (ox2 < 0) ? (mb_indcs.cjs    ) : mb_indcs.cje;
    flux.bks = (ox3 > 0) ? (mb_indcs.cke + 1) : mb_indcs.cks;
    flux.bke = (ox3 < 0) ? (mb_indcs.cks    ) : mb_indcs.cke;
    flux.ndat = (flux.bie - flux.bis + 1)*(flux.bje - flux.bjs + 1)*
                (flux.bke - flux.bks + 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesCC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for CC data
//! on same/coarser/finer levels, and for prolongation from coarse to fine. Again, only
//! first [0] component of each index array is used.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BoundaryValuesCC::InitRecvIndices(
     BoundaryBuffer &buf, int ox1, int ox2, int ox3, int f1, int f2)
{ 
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;

  // set indices for receives from neighbors on SAME level
  // Formulae taken from SetBoundarySameLevel() in src/bvals/cc/bvals_cc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &same = buf.same[0];    // indices of buffer for neighbor same level
    if (ox1 == 0) {
      same.bis = mb_indcs.is;
      same.bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      same.bis = mb_indcs.ie + 1,
      same.bie = mb_indcs.ie + ng;
    } else {
      same.bis = mb_indcs.is - ng;
      same.bie = mb_indcs.is - 1;
    }

    if (ox2 == 0) {
      same.bjs = mb_indcs.js;
      same.bje = mb_indcs.je;
    } else if (ox2 > 0) {
      same.bjs = mb_indcs.je + 1;
      same.bje = mb_indcs.je + ng;
    } else {
      same.bjs = mb_indcs.js - ng;
      same.bje = mb_indcs.js - 1;
    }

    if (ox3 == 0) {
      same.bks = mb_indcs.ks;
      same.bke = mb_indcs.ke;
    } else if (ox3 > 0) {
      same.bks = mb_indcs.ke + 1;
      same.bke = mb_indcs.ke + ng;
    } else {
      same.bks = mb_indcs.ks - ng;
      same.bke = mb_indcs.ks - 1;
    }
    same.ndat = (same.bie - same.bis+1)*(same.bje - same.bjs+1)*(same.bke - same.bks+1);
  }

  // set indices for receives from neighbors on COARSER level (matches send to FINER)
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &coar = buf.coar[0];   // indices of buffer for neighbor coarser level
  if (ox1 == 0) {
    coar.bis = mb_indcs.cis;
    coar.bie = mb_indcs.cie;
    if (f1 == 0) {
      coar.bie += ng;
    } else {
      coar.bis -= ng;
    }
  } else if (ox1 > 0)  {
    coar.bis = mb_indcs.cie + 1;
    coar.bie = mb_indcs.cie + ng;
  } else {
    coar.bis = mb_indcs.cis - ng;
    coar.bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    coar.bjs = mb_indcs.cjs;
    coar.bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          coar.bje += ng;
        } else {
          coar.bjs -= ng;
        }
      } else {
        if (f2 == 0) {
          coar.bje += ng;
        } else {
          coar.bjs -= ng;
        }
      }
    }
  } else if (ox2 > 0) {
    coar.bjs = mb_indcs.cje + 1;
    coar.bje = mb_indcs.cje + ng;
  } else {
    coar.bjs = mb_indcs.cjs - ng;
    coar.bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    coar.bks = mb_indcs.cks;
    coar.bke = mb_indcs.cke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          coar.bke += ng;
        } else {
          coar.bks -= ng;
        }
      } else {
        if (f2 == 0) {
          coar.bke += ng;
        } else {
          coar.bks -= ng;
        }
      }
    }
  } else if (ox3 > 0)  {
    coar.bks = mb_indcs.cke + 1;
    coar.bke = mb_indcs.cke + ng;
  } else {
    coar.bks = mb_indcs.cks - ng;
    coar.bke = mb_indcs.cks - 1;
  }
  coar.ndat = (coar.bie - coar.bis+1)*(coar.bje - coar.bjs+1)*(coar.bke - coar.bks+1);
  }

  // set indices for receives from neighbors on FINER level (matches send to COARSER)
  // Formulae taken from SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &fine = buf.fine[0];   // indices of buffer for neighbor finer level
  if (ox1 == 0) {
    fine.bis = mb_indcs.is;
    fine.bie = mb_indcs.ie;
    if (f1 == 1) {
      fine.bis += mb_indcs.cnx1;
    } else {
      fine.bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    fine.bis = mb_indcs.ie + 1;
    fine.bie = mb_indcs.ie + ng;
  } else {
    fine.bis = mb_indcs.is - ng;
    fine.bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    fine.bjs = mb_indcs.js;
    fine.bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          fine.bjs += mb_indcs.cnx2;
        } else { 
          fine.bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          fine.bjs += mb_indcs.cnx2;
        } else {
          fine.bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    fine.bjs = mb_indcs.je + 1;
    fine.bje = mb_indcs.je + ng;
  } else {
    fine.bjs = mb_indcs.js - ng;
    fine.bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    fine.bks = mb_indcs.ks;
    fine.bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          fine.bks += mb_indcs.cnx3;
        } else {
          fine.bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          fine.bks += mb_indcs.cnx3;
        } else {
          fine.bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    fine.bks = mb_indcs.ke + 1;
    fine.bke = mb_indcs.ke + ng;
  } else {
    fine.bks = mb_indcs.ks - ng;
    fine.bke = mb_indcs.ks - 1;
  }
  fine.ndat = (fine.bie - fine.bis+1)*(fine.bje - fine.bjs+1)*(fine.bke - fine.bks+1);
  }

  // set indices for PROLONGATION in coarse cell buffers. Indices refer to coarse cells.
  // Formulae taken from ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  // Identical to receives from coarser level, except ng --> ng/2
  {auto &prol = buf.prol[0];   // indices for prolongation ("p")
  int cn = mb_indcs.ng/2;      // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    prol.bis = mb_indcs.cis;
    prol.bie = mb_indcs.cie;
    if (f1 == 0) {
      prol.bie += cn;
    } else {
      prol.bis -= cn;
    }
  } else if (ox1 > 0)  {
    prol.bis = mb_indcs.cie + 1;
    prol.bie = mb_indcs.cie + cn;
  } else {
    prol.bis = mb_indcs.cis - cn;
    prol.bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    prol.bjs = mb_indcs.cjs;
    prol.bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          prol.bje += cn;
        } else {
          prol.bjs -= cn;
        }
      } else {
        if (f2 == 0) {
          prol.bje += cn;
        } else {
          prol.bjs -= cn;
        }
      }
    }
  } else if (ox2 > 0) {
    prol.bjs = mb_indcs.cje + 1;
    prol.bje = mb_indcs.cje + cn;
  } else {
    prol.bjs = mb_indcs.cjs - cn;
    prol.bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    prol.bks = mb_indcs.cks;
    prol.bke = mb_indcs.cke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          prol.bke += cn;
        } else {
          prol.bks -= cn;
        }
      } else {
        if (f2 == 0) {
          prol.bke += cn;
        } else {
          prol.bks -= cn;
        }
      }
    }
  } else if (ox3 > 0)  {
    prol.bks = mb_indcs.cke + 1;
    prol.bke = mb_indcs.cke + cn;
  } else {
    prol.bks = mb_indcs.cks - cn;
    prol.bke = mb_indcs.cks - 1;
  }
  prol.ndat = (prol.bie - prol.bis+1)*(prol.bje - prol.bjs+1)* (prol.bke - prol.bks+1);
  }
}
