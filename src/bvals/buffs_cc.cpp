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
#include "utils/create_mpitag.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BValCC::InitSendIndices
//! \brief Calculates indices of cells in mesh used to pack buffers and send CC data.
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BValCC::InitSendIndices(BValBufferCC &buf, int ox1, int ox2, int ox3, int f1, int f2)
{
  auto &mb_indcs  = pmy_pack->pcoord->mbdata.indcs;
  auto &mb_cindcs = pmy_pack->pcoord->mbdata.cindcs;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/cc/bvals_cc.cpp
  auto &sindcs = buf.sindcs;
  int ng1 = mb_indcs.ng - 1;
  sindcs.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  sindcs.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  sindcs.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  sindcs.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  sindcs.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  sindcs.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
  sindcs.ndat = (sindcs.bie - sindcs.bis + 1)*(sindcs.bje - sindcs.bjs + 1)*
                (sindcs.bke - sindcs.bks + 1);

  // set indices for sends to neighbors on COARSER level
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  auto &cindcs = buf.cindcs;
  cindcs.bis = (ox1 > 0) ? (mb_cindcs.ie - ng1) : mb_cindcs.is;
  cindcs.bie = (ox1 < 0) ? (mb_cindcs.is + ng1) : mb_cindcs.ie;
  cindcs.bjs = (ox2 > 0) ? (mb_cindcs.je - ng1) : mb_cindcs.js;
  cindcs.bje = (ox2 < 0) ? (mb_cindcs.js + ng1) : mb_cindcs.je;
  cindcs.bks = (ox3 > 0) ? (mb_cindcs.ke - ng1) : mb_cindcs.ks;
  cindcs.bke = (ox3 < 0) ? (mb_cindcs.ks + ng1) : mb_cindcs.ke;
  cindcs.ndat = (cindcs.bie - cindcs.bis + 1)*(cindcs.bje - cindcs.bjs + 1)*
                (cindcs.bke - cindcs.bks + 1);

  // set indices for sends to neighbors on FINER level
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/cc/bvals_cc.cpp
  auto &findcs = buf.findcs;
  findcs.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  findcs.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  findcs.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  findcs.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  findcs.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  findcs.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if (ox1 == 0) {
    if (f1 == 1) {
      findcs.bis += mb_indcs.nx1/2 - mb_cindcs.ng;
    } else {
      findcs.bie -= mb_indcs.nx1/2 - mb_cindcs.ng;
    }
  }
  if (ox2 == 0 && mb_indcs.nx2 > 1) {
    if (ox1 != 0) {
      if (f1 == 1) {
        findcs.bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
      } else {
        findcs.bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
      }
    } else {
      if (f2 == 1) {
        findcs.bjs += mb_indcs.nx2/2 - mb_cindcs.ng;
      } else {
        findcs.bje -= mb_indcs.nx2/2 - mb_cindcs.ng;
      }
    }
  }
  if (ox3 == 0 && mb_indcs.nx3 > 1) {
    if (ox1 != 0 && ox2 != 0) {
      if (f1 == 1) {
        findcs.bks += mb_indcs.nx3/2 - mb_cindcs.ng;
      } else {
        findcs.bke -= mb_indcs.nx3/2 - mb_cindcs.ng;
      }
    } else {
      if (f2 == 1) {
        findcs.bks += mb_indcs.nx3/2 - mb_cindcs.ng;
      } else {
        findcs.bke -= mb_indcs.nx3/2 - mb_cindcs.ng;
      }
    }
  }
  findcs.ndat = (findcs.bie - findcs.bis + 1)*(findcs.bje - findcs.bjs + 1)*
                (findcs.bke - findcs.bks + 1);

}

//----------------------------------------------------------------------------------------
//! \fn void BValCC::InitRecvIndices
//! \brief Calculates indices of cells in mesh into which receive buffers are unpacked.
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0)

void BValCC::InitRecvIndices(BValBufferCC &buf, int ox1, int ox2, int ox3, int f1, int f2)
{ 
  auto &mb_indcs  = pmy_pack->pcoord->mbdata.indcs;
  auto &mb_cindcs = pmy_pack->pcoord->mbdata.cindcs;

  // set indices for receives from neighbors on SAME level
  // Formulae taken from SetBoundarySameLevel() in src/bvals/cc/bvals_cc.cpp
  {auto &sindcs = buf.sindcs;   // indices of buffer at same level ("s")
  if (ox1 == 0) {
    sindcs.bis = mb_indcs.is;
    sindcs.bie = mb_indcs.ie;
  } else if (ox1 > 0) {
    sindcs.bis = mb_indcs.ie + 1,
    sindcs.bie = mb_indcs.ie + mb_indcs.ng;
  } else {
    sindcs.bis = mb_indcs.is - mb_indcs.ng;
    sindcs.bie = mb_indcs.is - 1;
  }

  if (ox2 == 0) {
    sindcs.bjs = mb_indcs.js;
    sindcs.bje = mb_indcs.je;
  } else if (ox2 > 0) {
    sindcs.bjs = mb_indcs.je + 1;
    sindcs.bje = mb_indcs.je + mb_indcs.ng;
  } else {
    sindcs.bjs = mb_indcs.js - mb_indcs.ng;
    sindcs.bje = mb_indcs.js - 1;
  }

  if (ox3 == 0) {
    sindcs.bks = mb_indcs.ks;
    sindcs.bke = mb_indcs.ke;
  } else if (ox3 > 0) {
    sindcs.bks = mb_indcs.ke + 1;
    sindcs.bke = mb_indcs.ke + mb_indcs.ng;
  } else {
    sindcs.bks = mb_indcs.ks - mb_indcs.ng;
    sindcs.bke = mb_indcs.ks - 1;
  }}

  // set indices for receives from neighbors on COARSER level
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &cindcs = buf.cindcs;   // indices of course buffer ("c")
  if (ox1 == 0) {
    cindcs.bis = mb_cindcs.is;
    cindcs.bie = mb_cindcs.ie;
    if (f1 == 0) {
      cindcs.bie += mb_indcs.ng;
    } else {
      cindcs.bis -= mb_indcs.ng;
    }
  } else if (ox1 > 0)  {
    cindcs.bis = mb_cindcs.ie + 1;
    cindcs.bie = mb_cindcs.ie + mb_indcs.ng;
  } else {
    cindcs.bis = mb_cindcs.is - mb_indcs.ng;
    cindcs.bie = mb_cindcs.is - 1;
  }
  if (ox2 == 0) {
    cindcs.bjs = mb_cindcs.js;
    cindcs.bje = mb_cindcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          cindcs.bje += mb_indcs.ng;
        } else {
          cindcs.bjs -= mb_indcs.ng;
        }
      } else {
        if (f2 == 0) {
          cindcs.bje += mb_indcs.ng;
        } else {
          cindcs.bjs -= mb_indcs.ng;
        }
      }
    }
  } else if (ox2 > 0) {
    cindcs.bjs = mb_cindcs.je + 1;
    cindcs.bje = mb_cindcs.je + mb_indcs.ng;
  } else {
    cindcs.bjs = mb_cindcs.js - mb_indcs.ng;
    cindcs.bje = mb_cindcs.js - 1;
  }
  if (ox3 == 0) {
    cindcs.bks = mb_cindcs.ks;
    cindcs.bke = mb_cindcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          cindcs.bke += mb_indcs.ng;
        } else {
          cindcs.bks -= mb_indcs.ng;
        }
      } else {
        if (f2 == 0) {
          cindcs.bke += mb_indcs.ng;
        } else {
          cindcs.bks -= mb_indcs.ng;
        }
      }
    }
  } else if (ox3 > 0)  {
    cindcs.bks = mb_cindcs.ke + 1;
    cindcs.bke = mb_cindcs.ke + mb_indcs.ng;
  } else {
    cindcs.bks = mb_cindcs.ks - mb_indcs.ng;
    cindcs.bke = mb_cindcs.ks - 1;
  }}

  // set indices for receives from neighbors on FINER level
  // Formulae taken from SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &findcs = buf.findcs;   // indices of fine buffer ("f")
  if (ox1 == 0) {
    findcs.bis = mb_indcs.is;
    findcs.bie = mb_indcs.ie;
    if (f1 == 1) {
      findcs.bis += mb_indcs.nx1/2;
    } else {
      findcs.bie -= mb_indcs.nx1/2;
    }
  } else if (ox1 > 0) {
    findcs.bis = mb_indcs.ie + 1;
    findcs.bie = mb_indcs.ie + mb_indcs.ng;
  } else {
    findcs.bis = mb_indcs.is - mb_indcs.ng;
    findcs.bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    findcs.bjs = mb_indcs.js;
    findcs.bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          findcs.bjs += mb_indcs.nx2/2;
        } else { 
          findcs.bje -= mb_indcs.nx2/2;
        }
      } else {
        if (f2 == 1) {
          findcs.bjs += mb_indcs.nx2/2;
        } else {
          findcs.bje -= mb_indcs.nx2/2;
        }
      }
    }
  } else if (ox2 > 0) {
    findcs.bjs = mb_indcs.je + 1;
    findcs.bje = mb_indcs.je + mb_indcs.ng;
  } else {
    findcs.bjs = mb_indcs.js - mb_indcs.ng;
    findcs.bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    findcs.bks = mb_indcs.ks;
    findcs.bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          findcs.bks += mb_indcs.nx3/2;
        } else {
          findcs.bke -= mb_indcs.nx3/2;
        }
      } else {
        if (f2 == 1) {
          findcs.bks += mb_indcs.nx3/2;
        } else {
          findcs.bke -= mb_indcs.nx3/2;
        }
      }
    }
  } else if (ox3 > 0) {
    findcs.bks = mb_indcs.ke + 1;
    findcs.bke = mb_indcs.ke + mb_indcs.ng;
  } else {
    findcs.bks = mb_indcs.ks - mb_indcs.ng;
    findcs.bke = mb_indcs.ks - 1;
  }}

  // set indices for PROLONGATION in coarse cell buffers
  // Formulae taken from ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  // Identical to receives from coarser level, except ng --> ng/2
  {auto &pindcs = buf.pindcs;   // indices fpr prolongation ("p")
  int cn = mb_indcs.ng/2;       // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    pindcs.bis = mb_cindcs.is;
    pindcs.bie = mb_cindcs.ie;
    if (f1 == 0) {
      pindcs.bie += cn;
    } else {
      pindcs.bis -= cn;
    }
  } else if (ox1 > 0)  {
    pindcs.bis = mb_cindcs.ie + 1;
    pindcs.bie = mb_cindcs.ie + cn;
  } else {
    pindcs.bis = mb_cindcs.is - cn;
    pindcs.bie = mb_cindcs.is - 1;
  }
  if (ox2 == 0) {
    pindcs.bjs = mb_cindcs.js;
    pindcs.bje = mb_cindcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          pindcs.bje += cn;
        } else {
          pindcs.bjs -= cn;
        }
      } else {
        if (f2 == 0) {
          pindcs.bje += cn;
        } else {
          pindcs.bjs -= cn;
        }
      }
    }
  } else if (ox2 > 0) {
    pindcs.bjs = mb_cindcs.je + 1;
    pindcs.bje = mb_cindcs.je + cn;
  } else {
    pindcs.bjs = mb_cindcs.js - cn;
    pindcs.bje = mb_cindcs.js - 1;
  }
  if (ox3 == 0) {
    pindcs.bks = mb_cindcs.ks;
    pindcs.bke = mb_cindcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          pindcs.bke += cn;
        } else {
          pindcs.bks -= cn;
        }
      } else {
        if (f2 == 0) {
          pindcs.bke += cn;
        } else {
          pindcs.bks -= cn;
        }
      }
    }
  } else if (ox3 > 0)  {
    pindcs.bks = mb_cindcs.ke + 1;
    pindcs.bke = mb_cindcs.ke + cn;
  } else {
    pindcs.bks = mb_cindcs.ks - cn;
    pindcs.bke = mb_cindcs.ks - 1;
  }}
}

//----------------------------------------------------------------------------------------
// \!fn void BValCC::AllocateBuffersCC
// initialize array of send/recv BValBuffers for arbitrary number of cell-centered
// variables, specified by input argument.
//
// NOTE: order of array elements is crucial and cannot be changed.  It must match
// order of boundaries in nghbr vector

// TODO: extend for AMR

void BValCC::AllocateBuffersCC(const int nvar)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int ng = indcs.ng;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  auto &cindcs = pmy_pack->pcoord->mbdata.cindcs;
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
    // cannot create Kokkos::View of type MPI_Request (not POD) so construct STL vector instead
    for (int m=0; m<nmb; ++m) {
      MPI_Request send_req, recv_req;
      send_buf[n].comm_req.push_back(send_req);
      recv_buf[n].comm_req.push_back(recv_req);
    }
#endif
  }

  // initialize buffers used for uniform grid calculations first

  // x1 faces; BufferID = [0,4]
  InitSendIndices(send_buf[0],-1, 0, 0, 0, 0);
  InitSendIndices(send_buf[4], 1, 0, 0, 0, 0);
  InitRecvIndices(recv_buf[0],-1, 0, 0, 0, 0);
  InitRecvIndices(recv_buf[4], 1, 0, 0, 0, 0);
  for (int n=0; n<=4; n+=4) {
    send_buf[n].AllocateDataView(nmb, nvar);
    recv_buf[n].AllocateDataView(nmb, nvar);
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces; BufferID = [8,12]
    InitSendIndices(send_buf[8 ], 0,-1, 0, 0, 0);
    InitSendIndices(send_buf[12], 0, 1, 0, 0, 0);
    InitRecvIndices(recv_buf[8 ], 0,-1, 0, 0, 0);
    InitRecvIndices(recv_buf[12], 0, 1, 0, 0, 0);
    for (int n=8; n<=12; n+=4) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // x1x2 edges; BufferID = [16,18,20,22]
    InitSendIndices(send_buf[16],-1,-1, 0, 0, 0);
    InitSendIndices(send_buf[18], 1,-1, 0, 0, 0);
    InitSendIndices(send_buf[20],-1, 1, 0, 0, 0);
    InitSendIndices(send_buf[22], 1, 1, 0, 0, 0);
    InitRecvIndices(recv_buf[16],-1,-1, 0, 0, 0);
    InitRecvIndices(recv_buf[18], 1,-1, 0, 0, 0);
    InitRecvIndices(recv_buf[20],-1, 1, 0, 0, 0);
    InitRecvIndices(recv_buf[22], 1, 1, 0, 0, 0);
    for (int n=16; n<=22; n+=2) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {
    // x3 faces; BufferID = [24,28]
    InitSendIndices(send_buf[24], 0, 0,-1, 0, 0);
    InitSendIndices(send_buf[28], 0, 0, 1, 0, 0);
    InitRecvIndices(recv_buf[24], 0, 0,-1, 0, 0);
    InitRecvIndices(recv_buf[28], 0, 0, 1, 0, 0);
    for (int n=24; n<=28; n+=4) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // x3x1 edges; BufferID = [32,34,36,38]
    InitSendIndices(send_buf[32],-1, 0,-1, 0, 0);
    InitSendIndices(send_buf[34], 1, 0,-1, 0, 0);
    InitSendIndices(send_buf[36],-1, 0, 1, 0, 0);
    InitSendIndices(send_buf[38], 1, 0, 1, 0, 0);
    InitRecvIndices(recv_buf[32],-1, 0,-1, 0, 0);
    InitRecvIndices(recv_buf[34], 1, 0,-1, 0, 0);
    InitRecvIndices(recv_buf[36],-1, 0, 1, 0, 0);
    InitRecvIndices(recv_buf[38], 1, 0, 1, 0, 0);
    for (int n=32; n<=38; n+=2) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // x2x3 edges; BufferID = [40,42,44,46]
    InitSendIndices(send_buf[40], 0,-1,-1, 0, 0);
    InitSendIndices(send_buf[42], 0, 1,-1, 0, 0);
    InitSendIndices(send_buf[44], 0,-1, 1, 0, 0);
    InitSendIndices(send_buf[46], 0, 1, 1, 0, 0);
    InitRecvIndices(recv_buf[40], 0,-1,-1, 0, 0);
    InitRecvIndices(recv_buf[42], 0, 1,-1, 0, 0);
    InitRecvIndices(recv_buf[44], 0,-1, 1, 0, 0);
    InitRecvIndices(recv_buf[46], 0, 1, 1, 0, 0);
    for (int n=40; n<=46; n+=2) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // corners; BufferID = [48,...,55]
    InitSendIndices(send_buf[48],-1,-1,-1, 0, 0);
    InitSendIndices(send_buf[49], 1,-1,-1, 0, 0);
    InitSendIndices(send_buf[50],-1, 1,-1, 0, 0);
    InitSendIndices(send_buf[51], 1, 1,-1, 0, 0);
    InitSendIndices(send_buf[52],-1,-1, 1, 0, 0);
    InitSendIndices(send_buf[53], 1,-1, 1, 0, 0);
    InitSendIndices(send_buf[54],-1, 1, 1, 0, 0);
    InitSendIndices(send_buf[55], 1, 1, 1, 0, 0);
    InitRecvIndices(recv_buf[48],-1,-1,-1, 0, 0);
    InitRecvIndices(recv_buf[49], 1,-1,-1, 0, 0);
    InitRecvIndices(recv_buf[50],-1, 1,-1, 0, 0);
    InitRecvIndices(recv_buf[51], 1, 1,-1, 0, 0);
    InitRecvIndices(recv_buf[52],-1,-1, 1, 0, 0);
    InitRecvIndices(recv_buf[53], 1,-1, 1, 0, 0);
    InitRecvIndices(recv_buf[54],-1, 1, 1, 0, 0);
    InitRecvIndices(recv_buf[55], 1, 1, 1, 0, 0);
    for (int n=48; n<=55; n+=1) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }
  }

  // now initialize buffers used for SMR/AMR calculations

  if (pmy_pack->pmesh->multilevel) {
    // x1 faces; BufferID = [1,5]
    InitSendIndices(send_buf[1],-1, 0, 0, 1, 0);
    InitSendIndices(send_buf[5], 1, 0, 0, 1, 0);
    InitRecvIndices(recv_buf[1],-1, 0, 0, 1, 0);
    InitRecvIndices(recv_buf[5], 1, 0, 0, 1, 0);
    for (int n=1; n<=5; n+=4) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // add more buffers in 2D
    if (pmy_pack->pmesh->multi_d) {
      // x2 faces; BufferID = [9,13]
      InitSendIndices(send_buf[9 ], 0,-1, 0, 1, 0);
      InitSendIndices(send_buf[13], 0, 1, 0, 1, 0);
      InitRecvIndices(recv_buf[9 ], 0,-1, 0, 1, 0);
      InitRecvIndices(recv_buf[13], 0, 1, 0, 1, 0);
      for (int n=9; n<=13; n+=4) {
        send_buf[n].AllocateDataView(nmb, nvar);
        recv_buf[n].AllocateDataView(nmb, nvar);
      }
    }
  }

/***/
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
/****/


  return;
}
