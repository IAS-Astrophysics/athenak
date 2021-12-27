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
//! \brief Calculates indices of cells used to pack buffers and send CC data for buffers
//! on same/coarser and finer levels.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BValCC::InitSendIndices(BValBufferCC &buf, int ox1, int ox2, int ox3, int f1, int f2)
{
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/cc/bvals_cc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &sindcs = buf.sindcs;   // indices of buffer at same level ("s")
    sindcs.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
    sindcs.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
    sindcs.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
    sindcs.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
    sindcs.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
    sindcs.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
    sindcs.ndat = (sindcs.bie - sindcs.bis + 1)*(sindcs.bje - sindcs.bjs + 1)*
                  (sindcs.bke - sindcs.bks + 1);
  } else {  // this buffer only used with AMR (e.g. #1,2,3,5,6,7,...)
    auto &sindcs = buf.sindcs;
    sindcs.bis = 0; sindcs.bie = 0;
    sindcs.bjs = 0; sindcs.bje = 0;
    sindcs.bks = 0; sindcs.bke = 0;
    sindcs.ndat = 1;
  }

  // set indices for sends to neighbors on COARSER level (matches recvs from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &cindcs = buf.cindcs;  // indices of coarse buffer ("c")
  cindcs.bis = (ox1 > 0) ? (mb_indcs.cie - ng1) : mb_indcs.cis;
  cindcs.bie = (ox1 < 0) ? (mb_indcs.cis + ng1) : mb_indcs.cie;
  cindcs.bjs = (ox2 > 0) ? (mb_indcs.cje - ng1) : mb_indcs.cjs;
  cindcs.bje = (ox2 < 0) ? (mb_indcs.cjs + ng1) : mb_indcs.cje;
  cindcs.bks = (ox3 > 0) ? (mb_indcs.cke - ng1) : mb_indcs.cks;
  cindcs.bke = (ox3 < 0) ? (mb_indcs.cks + ng1) : mb_indcs.cke;
  cindcs.ndat = (cindcs.bie - cindcs.bis + 1)*(cindcs.bje - cindcs.bjs + 1)*
                (cindcs.bke - cindcs.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recvs from COARSER)
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/cc/bvals_cc.cpp
  {auto &findcs = buf.findcs;  // indices of fine buffer ("f")
  findcs.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  findcs.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  findcs.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  findcs.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  findcs.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  findcs.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
  // need to add internal edges on faces, and internal corners on edges
  if (ox1 == 0) {
    if (f1 == 1) {
      findcs.bis += mb_indcs.cnx1 - ng;
    } else {
      findcs.bie -= mb_indcs.cnx1 - ng;
    }
  }
  if (ox2 == 0 && mb_indcs.nx2 > 1) {
    if (ox1 != 0) {
      if (f1 == 1) {
        findcs.bjs += mb_indcs.cnx2 - ng;
      } else {
        findcs.bje -= mb_indcs.cnx2 - ng;
      }
    } else {
      if (f2 == 1) {
        findcs.bjs += mb_indcs.cnx2 - ng;
      } else {
        findcs.bje -= mb_indcs.cnx2 - ng;
      }
    }
  }
  if (ox3 == 0 && mb_indcs.nx3 > 1) {
    if (ox1 != 0 && ox2 != 0) {
      if (f1 == 1) {
        findcs.bks += mb_indcs.cnx3 - ng;
      } else {
        findcs.bke -= mb_indcs.cnx3 - ng;
      }
    } else {
      if (f2 == 1) {
        findcs.bks += mb_indcs.cnx3 - ng;
      } else {
        findcs.bke -= mb_indcs.cnx3 - ng;
      }
    }
  }
  findcs.ndat = (findcs.bie - findcs.bis + 1)*(findcs.bje - findcs.bjs + 1)*
                (findcs.bke - findcs.bks + 1);
  }

  // indices for PROLONGATION not needed for sends, just initialize to zero
  {auto &pindcs = buf.pindcs;
  pindcs.bis = 0; pindcs.bie = 0;
  pindcs.bjs = 0; pindcs.bje = 0;
  pindcs.bks = 0; pindcs.bke = 0;
  pindcs.ndat = 1;
  }

}

//----------------------------------------------------------------------------------------
//! \fn void BValCC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for CC data
//! on same/coarser/finer levels, and for prolongation from coarse to fine.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void BValCC::InitRecvIndices(BValBufferCC &buf, int ox1, int ox2, int ox3, int f1, int f2)
{ 
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;

  // set indices for receives from neighbors on SAME level
  // Formulae taken from SetBoundarySameLevel() in src/bvals/cc/bvals_cc.cpp
  if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #0,4,8,12,...)
    auto &sindcs = buf.sindcs;   // indices of buffer at same level ("s")
    if (ox1 == 0) {
      sindcs.bis = mb_indcs.is;
      sindcs.bie = mb_indcs.ie;
    } else if (ox1 > 0) {
      sindcs.bis = mb_indcs.ie + 1,
      sindcs.bie = mb_indcs.ie + ng;
    } else {
      sindcs.bis = mb_indcs.is - ng;
      sindcs.bie = mb_indcs.is - 1;
    }

    if (ox2 == 0) {
      sindcs.bjs = mb_indcs.js;
      sindcs.bje = mb_indcs.je;
    } else if (ox2 > 0) {
      sindcs.bjs = mb_indcs.je + 1;
      sindcs.bje = mb_indcs.je + ng;
    } else {
      sindcs.bjs = mb_indcs.js - ng;
      sindcs.bje = mb_indcs.js - 1;
    }

    if (ox3 == 0) {
      sindcs.bks = mb_indcs.ks;
      sindcs.bke = mb_indcs.ke;
    } else if (ox3 > 0) {
      sindcs.bks = mb_indcs.ke + 1;
      sindcs.bke = mb_indcs.ke + ng;
    } else {
      sindcs.bks = mb_indcs.ks - ng;
      sindcs.bke = mb_indcs.ks - 1;
    }
    sindcs.ndat = (sindcs.bie - sindcs.bis + 1)*(sindcs.bje - sindcs.bjs + 1)*
                  (sindcs.bke - sindcs.bks + 1);
  } else {  // this buffer only used with AMR (e.g. #1,2,3,5,6,7,...)
    auto &sindcs = buf.sindcs;
    sindcs.bis = 0; sindcs.bie = 0;
    sindcs.bjs = 0; sindcs.bje = 0;
    sindcs.bks = 0; sindcs.bke = 0;
    sindcs.ndat = 1;
  }

  // set indices for receives from neighbors on COARSER level (matches send to FINER)
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &cindcs = buf.cindcs;   // indices of coarse buffer ("c")
  if (ox1 == 0) {
    cindcs.bis = mb_indcs.cis;
    cindcs.bie = mb_indcs.cie;
    if (f1 == 0) {
      cindcs.bie += ng;
    } else {
      cindcs.bis -= ng;
    }
  } else if (ox1 > 0)  {
    cindcs.bis = mb_indcs.cie + 1;
    cindcs.bie = mb_indcs.cie + ng;
  } else {
    cindcs.bis = mb_indcs.cis - ng;
    cindcs.bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    cindcs.bjs = mb_indcs.cjs;
    cindcs.bje = mb_indcs.cje;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 0) {
          cindcs.bje += ng;
        } else {
          cindcs.bjs -= ng;
        }
      } else {
        if (f2 == 0) {
          cindcs.bje += ng;
        } else {
          cindcs.bjs -= ng;
        }
      }
    }
  } else if (ox2 > 0) {
    cindcs.bjs = mb_indcs.cje + 1;
    cindcs.bje = mb_indcs.cje + ng;
  } else {
    cindcs.bjs = mb_indcs.cjs - ng;
    cindcs.bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    cindcs.bks = mb_indcs.cks;
    cindcs.bke = mb_indcs.cke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 0) {
          cindcs.bke += ng;
        } else {
          cindcs.bks -= ng;
        }
      } else {
        if (f2 == 0) {
          cindcs.bke += ng;
        } else {
          cindcs.bks -= ng;
        }
      }
    }
  } else if (ox3 > 0)  {
    cindcs.bks = mb_indcs.cke + 1;
    cindcs.bke = mb_indcs.cke + ng;
  } else {
    cindcs.bks = mb_indcs.cks - ng;
    cindcs.bke = mb_indcs.cks - 1;
  }
  cindcs.ndat = (cindcs.bie - cindcs.bis + 1)*(cindcs.bje - cindcs.bjs + 1)*
                (cindcs.bke - cindcs.bks + 1);
  }

  // set indices for receives from neighbors on FINER level (matches send to COARSER)
  // Formulae taken from SetBoundaryFromFiner() in src/bvals/cc/bvals_cc.cpp
  {auto &findcs = buf.findcs;   // indices of fine buffer ("f")
  if (ox1 == 0) {
    findcs.bis = mb_indcs.is;
    findcs.bie = mb_indcs.ie;
    if (f1 == 1) {
      findcs.bis += mb_indcs.cnx1;
    } else {
      findcs.bie -= mb_indcs.cnx1;
    }
  } else if (ox1 > 0) {
    findcs.bis = mb_indcs.ie + 1;
    findcs.bie = mb_indcs.ie + ng;
  } else {
    findcs.bis = mb_indcs.is - ng;
    findcs.bie = mb_indcs.is - 1;
  }
  if (ox2 == 0) {
    findcs.bjs = mb_indcs.js;
    findcs.bje = mb_indcs.je;
    if (mb_indcs.nx2 > 1) {
      if (ox1 != 0) {
        if (f1 == 1) {
          findcs.bjs += mb_indcs.cnx2;
        } else { 
          findcs.bje -= mb_indcs.cnx2;
        }
      } else {
        if (f2 == 1) {
          findcs.bjs += mb_indcs.cnx2;
        } else {
          findcs.bje -= mb_indcs.cnx2;
        }
      }
    }
  } else if (ox2 > 0) {
    findcs.bjs = mb_indcs.je + 1;
    findcs.bje = mb_indcs.je + ng;
  } else {
    findcs.bjs = mb_indcs.js - ng;
    findcs.bje = mb_indcs.js - 1;
  }
  if (ox3 == 0) {
    findcs.bks = mb_indcs.ks;
    findcs.bke = mb_indcs.ke;
    if (mb_indcs.nx3 > 1) {
      if (ox1 != 0 && ox2 != 0) {
        if (f1 == 1) {
          findcs.bks += mb_indcs.cnx3;
        } else {
          findcs.bke -= mb_indcs.cnx3;
        }
      } else {
        if (f2 == 1) {
          findcs.bks += mb_indcs.cnx3;
        } else {
          findcs.bke -= mb_indcs.cnx3;
        }
      }
    }
  } else if (ox3 > 0) {
    findcs.bks = mb_indcs.ke + 1;
    findcs.bke = mb_indcs.ke + ng;
  } else {
    findcs.bks = mb_indcs.ks - ng;
    findcs.bke = mb_indcs.ks - 1;
  }
  findcs.ndat = (findcs.bie - findcs.bis + 1)*(findcs.bje - findcs.bjs + 1)*
                (findcs.bke - findcs.bks + 1);
  }

  // set indices for PROLONGATION in coarse cell buffers. Indices refer to coarse cells.
  // Formulae taken from ProlongateBoundaries() in src/bvals/bvals_refine.cpp
  // Identical to receives from coarser level, except ng --> ng/2
  {auto &pindcs = buf.pindcs;   // indices for prolongation ("p")
  int cn = mb_indcs.ng/2;       // nghost must be multiple of 2 with SMR/AMR
  if (ox1 == 0) {
    pindcs.bis = mb_indcs.cis;
    pindcs.bie = mb_indcs.cie;
    if (f1 == 0) {
      pindcs.bie += cn;
    } else {
      pindcs.bis -= cn;
    }
  } else if (ox1 > 0)  {
    pindcs.bis = mb_indcs.cie + 1;
    pindcs.bie = mb_indcs.cie + cn;
  } else {
    pindcs.bis = mb_indcs.cis - cn;
    pindcs.bie = mb_indcs.cis - 1;
  }
  if (ox2 == 0) {
    pindcs.bjs = mb_indcs.cjs;
    pindcs.bje = mb_indcs.cje;
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
    pindcs.bjs = mb_indcs.cje + 1;
    pindcs.bje = mb_indcs.cje + cn;
  } else {
    pindcs.bjs = mb_indcs.cjs - cn;
    pindcs.bje = mb_indcs.cjs - 1;
  }
  if (ox3 == 0) {
    pindcs.bks = mb_indcs.cks;
    pindcs.bke = mb_indcs.cke;
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
    pindcs.bks = mb_indcs.cke + 1;
    pindcs.bke = mb_indcs.cke + cn;
  } else {
    pindcs.bks = mb_indcs.cks - cn;
    pindcs.bke = mb_indcs.cks - 1;
  }
  pindcs.ndat = (pindcs.bie - pindcs.bis + 1)*(pindcs.bje - pindcs.bjs + 1)*
                (pindcs.bke - pindcs.bks + 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BValCC::AllocateBuffersCC
//! \brief initialize vector of send/recv BValBuffers for arbitrary number of
//!  cell-centered variables, specified by input argument.
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector

void BValCC::AllocateBuffersCC(const int nvar)
{
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

  return;
}
