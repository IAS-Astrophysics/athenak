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
    auto &same = buf.same;       // indices of buffer for neighbor same level
    same.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
    same.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
    same.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
    same.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
    same.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
    same.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
    same.ndat = (same.bie - same.bis + 1)*(same.bje - same.bjs + 1)*
                  (same.bke - same.bks + 1);
  } else {  // this buffer only used with AMR (e.g. #1,2,3,5,6,7,...)
    auto &same = buf.same;
    same.bis = 0; same.bie = 0;
    same.bjs = 0; same.bje = 0;
    same.bks = 0; same.bke = 0;
    same.ndat = 1;
  }

  // set indices for sends to neighbors on COARSER level (matches recvs from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &coar = buf.coar;  // indices of buffer for neighbor coarser level
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
  {auto &fine = buf.fine;  // indices of buffer for neighbor finer level
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

  // indices for PROLONGATION not needed for sends, just initialize to zero
  {auto &prol = buf.prol;
  prol.bis = 0; prol.bie = 0;
  prol.bjs = 0; prol.bje = 0;
  prol.bks = 0; prol.bke = 0;
  prol.ndat = 1;
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
    auto &same = buf.same;   // indices of buffer for neighbor same level
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
  } else {  // this buffer only used with AMR (e.g. #1,2,3,5,6,7,...)
    auto &same = buf.same;
    same.bis = 0; same.bie = 0;
    same.bjs = 0; same.bje = 0;
    same.bks = 0; same.bke = 0;
    same.ndat = 1;
  }

  // set indices for receives from neighbors on COARSER level (matches send to FINER)
  // Formulae taken from SetBoundaryFromCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &coar = buf.coar;   // indices of buffer for neighbor coarser level
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
  {auto &fine = buf.fine;   // indices of buffer for neighbor finer level
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
  {auto &prol = buf.prol;   // indices for prolongation ("p")
  int cn = mb_indcs.ng/2;   // nghost must be multiple of 2 with SMR/AMR
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
