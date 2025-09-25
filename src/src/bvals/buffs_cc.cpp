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
//! \fn void MeshBoundaryValuesCC::InitSendIndices
//! \brief Calculates indices of cells used to pack buffers and send CC data for buffers
//! on same/coarser/finer levels. Only one set of indices is needed, so only first [0]
//! component of each index array is used.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void MeshBoundaryValuesCC::InitSendIndices(MeshBoundaryBuffer &buf,
                                          int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/cc/bvals_cc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &isame = buf.isame[0];    // indices of buffer for neighbor same level
    isame.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
    isame.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
    isame.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
    isame.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
    isame.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
    isame.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
    buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                     (isame.bke - isame.bks + 1);
  }

  // set indices for sends of COARSE data to neighbors on SAME level (needed for Z4c)
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &isame = buf.isame_z4c;
    isame.bis = (ox1 > 0) ? (mb_indcs.cie - ng1) : mb_indcs.cis;
    isame.bie = (ox1 < 0) ? (mb_indcs.cis + ng1) : mb_indcs.cie;
    isame.bjs = (ox2 > 0) ? (mb_indcs.cje - ng1) : mb_indcs.cjs;
    isame.bje = (ox2 < 0) ? (mb_indcs.cjs + ng1) : mb_indcs.cje;
    isame.bks = (ox3 > 0) ? (mb_indcs.cke - ng1) : mb_indcs.cks;
    isame.bke = (ox3 < 0) ? (mb_indcs.cks + ng1) : mb_indcs.cke;
    buf.isame_z4c_ndat = buf.isame_ndat +
      (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*(isame.bke - isame.bks + 1);
  }


  // set indices for sends to neighbors on COARSER level (matches recvs from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &icoar = buf.icoar[0];  // indices of buffer for neighbor coarser level
  icoar.bis = (ox1 > 0) ? (mb_indcs.cie - ng1) : mb_indcs.cis;
  icoar.bie = (ox1 < 0) ? (mb_indcs.cis + ng1) : mb_indcs.cie;
  icoar.bjs = (ox2 > 0) ? (mb_indcs.cje - ng1) : mb_indcs.cjs;
  icoar.bje = (ox2 < 0) ? (mb_indcs.cjs + ng1) : mb_indcs.cje;
  icoar.bks = (ox3 > 0) ? (mb_indcs.cke - ng1) : mb_indcs.cks;
  icoar.bke = (ox3 < 0) ? (mb_indcs.cks + ng1) : mb_indcs.cke;
  buf.icoar_ndat = (icoar.bie - icoar.bis + 1)*(icoar.bje - icoar.bjs + 1)*
                   (icoar.bke - icoar.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recvs from COARSER)
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/cc/bvals_cc.cpp
  {auto &ifine = buf.ifine[0];  // indices of buffer for neighbor finer level
  ifine.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  ifine.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  ifine.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  ifine.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  ifine.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  ifine.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
  // need to add internal edges on faces, and internal corners on edges
  if (ox1 == 0) {
    if (f1 == 1) {
      ifine.bis += mb_indcs.cnx1 - ng;
    } else {
      ifine.bie -= mb_indcs.cnx1 - ng;
    }
  }
  if (ox2 == 0 && mb_indcs.nx2 > 1) {
    if (ox1 != 0) {
      if (f1 == 1) {
        ifine.bjs += mb_indcs.cnx2 - ng;
      } else {
        ifine.bje -= mb_indcs.cnx2 - ng;
      }
    } else {
      if (f2 == 1) {
        ifine.bjs += mb_indcs.cnx2 - ng;
      } else {
        ifine.bje -= mb_indcs.cnx2 - ng;
      }
    }
  }
  if (ox3 == 0 && mb_indcs.nx3 > 1) {
    if (ox1 != 0 && ox2 != 0) {
      if (f1 == 1) {
        ifine.bks += mb_indcs.cnx3 - ng;
      } else {
        ifine.bke -= mb_indcs.cnx3 - ng;
      }
    } else {
      if (f2 == 1) {
        ifine.bks += mb_indcs.cnx3 - ng;
      } else {
        ifine.bke -= mb_indcs.cnx3 - ng;
      }
    }
  }
  buf.ifine_ndat = (ifine.bie - ifine.bis + 1)*(ifine.bje - ifine.bjs + 1)*
                   (ifine.bke - ifine.bks + 1);
  }

  // set indices for sends for FLUX CORRECTION (sends always to COARSER level)
  {auto &iflux = buf.iflux_coar[0];    // indices of buffer for flux correction
  if (ox1 == 0) {
    iflux.bis = mb_indcs.cis;           iflux.bie = mb_indcs.cie;
  } else if (ox1 > 0) {
    iflux.bis = mb_indcs.cie + 1;       iflux.bie = mb_indcs.cie + 1;
  } else {
    iflux.bis = mb_indcs.cis;           iflux.bie = mb_indcs.cis;
  }
  if (ox2 == 0) {
    iflux.bjs = mb_indcs.cjs;           iflux.bje = mb_indcs.cje;
  } else if (ox2 > 0) {
    iflux.bjs = mb_indcs.cje + 1;       iflux.bje = mb_indcs.cje + 1;
  } else {
    iflux.bjs = mb_indcs.cjs;           iflux.bje = mb_indcs.cjs;
  }
  if (ox3 == 0) {
    iflux.bks = mb_indcs.cks;           iflux.bke = mb_indcs.cke;
  } else if (ox3 > 0) {
    iflux.bks = mb_indcs.cke + 1;       iflux.bke = mb_indcs.cke + 1;
  } else {
    iflux.bks = mb_indcs.cks;           iflux.bke = mb_indcs.cks;
  }
  buf.iflxc_ndat = (iflux.bie - iflux.bis + 1)*(iflux.bje - iflux.bjs + 1)*
                   (iflux.bke - iflux.bks + 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for CC data
//! on same/coarser/finer levels, and for prolongation from coarse to fine. Again, only
//! first [0] component of each index array is used.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void MeshBoundaryValuesCC::InitRecvIndices(MeshBoundaryBuffer &buf,
                                           int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;

  // set indices for receives from neighbors on SAME level
  // Formulae taken from SetBoundarySameLevel() in src/bvals/cc/bvals_cc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &isame = buf.isame[0];    // indices of buffer for neighbor same level
    if (ox1 == 0) {
      isame.bis = mb_indcs.is;          isame.bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      isame.bis = mb_indcs.ie + 1;      isame.bie = mb_indcs.ie + ng;
    } else {
      isame.bis = mb_indcs.is - ng;     isame.bie = mb_indcs.is - 1;
    }

    if (ox2 == 0) {
      isame.bjs = mb_indcs.js;          isame.bje = mb_indcs.je;
    } else if (ox2 > 0) {
      isame.bjs = mb_indcs.je + 1;      isame.bje = mb_indcs.je + ng;
    } else {
      isame.bjs = mb_indcs.js - ng;     isame.bje = mb_indcs.js - 1;
    }

    if (ox3 == 0) {
      isame.bks = mb_indcs.ks;          isame.bke = mb_indcs.ke;
    } else if (ox3 > 0) {
      isame.bks = mb_indcs.ke + 1;      isame.bke = mb_indcs.ke + ng;
    } else {
      isame.bks = mb_indcs.ks - ng;     isame.bke = mb_indcs.ks - 1;
    }
    buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                     (isame.bke - isame.bks + 1);
  }

  // set indices for receives of COARSE data from neighbors on SAME level
  // Needed for Z4c with higher-order prolongation/restriction
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &isame = buf.isame_z4c;
    if (ox1 == 0) {
      isame.bis = mb_indcs.cis;          isame.bie = mb_indcs.cie;
    } else if (ox1 > 0) {
      isame.bis = mb_indcs.cie + 1;      isame.bie = mb_indcs.cie + ng;
    } else {
      isame.bis = mb_indcs.cis - ng;     isame.bie = mb_indcs.cis - 1;
    }

    if (ox2 == 0) {
      isame.bjs = mb_indcs.cjs;          isame.bje = mb_indcs.cje;
    } else if (ox2 > 0) {
      isame.bjs = mb_indcs.cje + 1;      isame.bje = mb_indcs.cje + ng;
    } else {
      isame.bjs = mb_indcs.cjs - ng;     isame.bje = mb_indcs.cjs - 1;
    }

    if (ox3 == 0) {
      isame.bks = mb_indcs.cks;          isame.bke = mb_indcs.cke;
    } else if (ox3 > 0) {
      isame.bks = mb_indcs.cke + 1;      isame.bke = mb_indcs.cke + ng;
    } else {
      isame.bks = mb_indcs.cks - ng;     isame.bke = mb_indcs.cks - 1;
    }
    buf.isame_z4c_ndat = buf.isame_ndat +
      (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*(isame.bke - isame.bks + 1);
  }

  // set indices for receives from neighbors on COARSER level (matches send to FINER)
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &icoar = buf.icoar[0];   // indices of buffer for neighbor coarser level
  if (ox1 == 0) {
    icoar.bis = mb_indcs.cis;          icoar.bie = mb_indcs.cie;
    if (f1 == 0) {
      icoar.bie += ng;
    } else {
      icoar.bis -= ng;
    }
  } else if (ox1 > 0)  {
    icoar.bis = mb_indcs.cie + 1;      icoar.bie = mb_indcs.cie + ng;
  } else {
    icoar.bis = mb_indcs.cis - ng;     icoar.bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    icoar.bjs = mb_indcs.cjs;          icoar.bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          icoar.bje += ng;
        } else {
          icoar.bjs -= ng;
        }
      } else {
        if (f2 == 0) {
          icoar.bje += ng;
        } else {
          icoar.bjs -= ng;
        }
      }
    }
  } else if (ox2 > 0) {
    icoar.bjs = mb_indcs.cje + 1;      icoar.bje = mb_indcs.cje + ng;
  } else {
    icoar.bjs = mb_indcs.cjs - ng;     icoar.bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    icoar.bks = mb_indcs.cks;          icoar.bke = mb_indcs.cke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          icoar.bke += ng;
        } else {
          icoar.bks -= ng;
        }
      } else {
        if (f2 == 0) {
          icoar.bke += ng;
        } else {
          icoar.bks -= ng;
        }
      }
    }
  } else if (ox3 > 0)  {
    icoar.bks = mb_indcs.cke + 1;      icoar.bke = mb_indcs.cke + ng;
  } else {
    icoar.bks = mb_indcs.cks - ng;     icoar.bke = mb_indcs.cks - 1;
  }
  buf.icoar_ndat = (icoar.bie - icoar.bis + 1)*(icoar.bje - icoar.bjs + 1)*
                   (icoar.bke - icoar.bks + 1);
  }

  // set indices for receives from neighbors on FINER level (matches send to COARSER)
  // Formulae taken from SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &ifine = buf.ifine[0];   // indices of buffer for neighbor finer level
  if (ox1 == 0) {
    ifine.bis = mb_indcs.is;           ifine.bie = mb_indcs.ie;
    if (f1 == 1) {
      ifine.bis += mb_indcs.cnx1;
    } else {
      ifine.bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    ifine.bis = mb_indcs.ie + 1;       ifine.bie = mb_indcs.ie + ng;
  } else {
    ifine.bis = mb_indcs.is - ng;      ifine.bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    ifine.bjs = mb_indcs.js;
    ifine.bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          ifine.bjs += mb_indcs.cnx2;
        } else {
          ifine.bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          ifine.bjs += mb_indcs.cnx2;
        } else {
          ifine.bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    ifine.bjs = mb_indcs.je + 1;       ifine.bje = mb_indcs.je + ng;
  } else {
    ifine.bjs = mb_indcs.js - ng;      ifine.bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    ifine.bks = mb_indcs.ks;
    ifine.bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          ifine.bks += mb_indcs.cnx3;
        } else {
          ifine.bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          ifine.bks += mb_indcs.cnx3;
        } else {
          ifine.bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    ifine.bks = mb_indcs.ke + 1;       ifine.bke = mb_indcs.ke + ng;
  } else {
    ifine.bks = mb_indcs.ks - ng;      ifine.bke = mb_indcs.ks - 1;
  }
  buf.ifine_ndat = (ifine.bie - ifine.bis + 1)*(ifine.bje - ifine.bjs + 1)*
                   (ifine.bke - ifine.bks + 1);
  }

  // set indices for PROLONGATION in coarse cell buffers. Indices refer to coarse cells.
  // Formulae taken from ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  // Identical to receives from coarser level, except ng --> ng/2
  {auto &iprol = buf.iprol[0];   // indices for prolongation
  int cn = mb_indcs.ng/2;      // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    iprol.bis = mb_indcs.cis;          iprol.bie = mb_indcs.cie;
    if (f1 == 0) {
      iprol.bie += cn;
    } else {
      iprol.bis -= cn;
    }
  } else if (ox1 > 0)  {
    iprol.bis = mb_indcs.cie + 1;      iprol.bie = mb_indcs.cie + cn;
  } else {
    iprol.bis = mb_indcs.cis - cn;     iprol.bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    iprol.bjs = mb_indcs.cjs;          iprol.bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          iprol.bje += cn;
        } else {
          iprol.bjs -= cn;
        }
      } else {
        if (f2 == 0) {
          iprol.bje += cn;
        } else {
          iprol.bjs -= cn;
        }
      }
    }
  } else if (ox2 > 0) {
    iprol.bjs = mb_indcs.cje + 1;      iprol.bje = mb_indcs.cje + cn;
  } else {
    iprol.bjs = mb_indcs.cjs - cn;     iprol.bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    iprol.bks = mb_indcs.cks;          iprol.bke = mb_indcs.cke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          iprol.bke += cn;
        } else {
          iprol.bks -= cn;
        }
      } else {
        if (f2 == 0) {
          iprol.bke += cn;
        } else {
          iprol.bks -= cn;
        }
      }
    }
  } else if (ox3 > 0)  {
    iprol.bks = mb_indcs.cke + 1;      iprol.bke = mb_indcs.cke + cn;
  } else {
    iprol.bks = mb_indcs.cks - cn;     iprol.bke = mb_indcs.cks - 1;
  }
  }

  // set indices for receives for flux-correction.  Similar to send, except data loaded
  // into appropriate sub-block of coarse buffer (similar to receive from FINER level)
  {auto &iflux = buf.iflux_coar[0];   // indices of buffer for flux correction
  if (ox1 == 0) {
    iflux.bis = mb_indcs.is;           iflux.bie = mb_indcs.ie;
    if (f1 == 1) {
      iflux.bis += mb_indcs.cnx1;
    } else {
      iflux.bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    iflux.bis = mb_indcs.ie + 1;       iflux.bie = mb_indcs.ie + 1;
  } else {
    iflux.bis = mb_indcs.is;           iflux.bie = mb_indcs.is;
  }
  if (ox2 == 0) {
    iflux.bjs = mb_indcs.js;           iflux.bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          iflux.bjs += mb_indcs.cnx2;
        } else {
          iflux.bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          iflux.bjs += mb_indcs.cnx2;
        } else {
          iflux.bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    iflux.bjs = mb_indcs.je + 1;       iflux.bje = mb_indcs.je + 1;
  } else {
    iflux.bjs = mb_indcs.js;           iflux.bje = mb_indcs.js;
  }
  if (ox3 == 0) {
    iflux.bks = mb_indcs.ks;           iflux.bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          iflux.bks += mb_indcs.cnx3;
        } else {
          iflux.bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          iflux.bks += mb_indcs.cnx3;
        } else {
          iflux.bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    iflux.bks = mb_indcs.ke + 1;       iflux.bke = mb_indcs.ke + 1;
  } else {
    iflux.bks = mb_indcs.ks;           iflux.bke = mb_indcs.ks;
  }
  buf.iflxc_ndat = (iflux.bie - iflux.bis + 1)*(iflux.bje - iflux.bjs + 1)*
                   (iflux.bke - iflux.bks + 1);
  }
}
