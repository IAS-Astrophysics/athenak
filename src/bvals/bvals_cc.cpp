//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//  \brief functions to pass boundary values for cell-centered variables as implemented in
//  BValCC class.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "utils/create_mpitag.hpp"

//----------------------------------------------------------------------------------------
// BValCC constructor:

BValCC::BValCC(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp)
{
} 
  
//----------------------------------------------------------------------------------------
//! \fn void BValCC::InitSendIndices
//! \brief Calculates indices of cells in mesh used to pack buffers and send data.
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0)

void BValCC::InitSendIndices(BValBufferCC &buf, int ox1, int ox2, int ox3)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;

  // set indices for sends to neighbors on SAME level
  auto &sindcs = buf.sindcs;
  sindcs.bis = (ox1 > 0) ? (indcs.ie - indcs.ng + 1) : indcs.is;
  sindcs.bie = (ox1 < 0) ? (indcs.is + indcs.ng - 1) : indcs.ie;
  sindcs.bjs = (ox2 > 0) ? (indcs.je - indcs.ng + 1) : indcs.js;
  sindcs.bje = (ox2 < 0) ? (indcs.js + indcs.ng - 1) : indcs.je;
  sindcs.bks = (ox3 > 0) ? (indcs.ke - indcs.ng + 1) : indcs.ks;
  sindcs.bke = (ox3 < 0) ? (indcs.ks + indcs.ng - 1) : indcs.ke;
}

//----------------------------------------------------------------------------------------
//! \fn void BValCC::InitRecvIndices
//! \brief Calculates indices of cells in mesh into which receive buffers are unpacked.
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0)

void BValCC::InitRecvIndices(BValBufferCC &buf, int ox1, int ox2, int ox3)
{ 
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;

  // set indices for receives from neighbors on SAME level
  auto &sindcs = buf.sindcs;
  if (ox1 == 0) {
    sindcs.bis = indcs.is;
    sindcs.bie = indcs.ie;
  } else if (ox1 > 0) {
    sindcs.bis = indcs.ie + 1,
    sindcs.bie = indcs.ie + indcs.ng;
  } else {
    sindcs.bis = indcs.is - indcs.ng,
    sindcs.bie = indcs.is - 1;
  }

  if (ox2 == 0) {
    sindcs.bjs = indcs.js;
    sindcs.bje = indcs.je;
  } else if (ox2 > 0) {
    sindcs.bjs = indcs.je + 1;
    sindcs.bje = indcs.je + indcs.ng;
  } else {
    sindcs.bjs = indcs.js - indcs.ng;
    sindcs.bje = indcs.js - 1;
  }

  if (ox3 == 0) {
    sindcs.bks = indcs.ks;
    sindcs.bke = indcs.ke;
  } else if (ox3 > 0) {
    sindcs.bks = indcs.ke + 1;
    sindcs.bke = indcs.ke + indcs.ng;
  } else {
    sindcs.bks = indcs.ks - indcs.ng;
    sindcs.bke = indcs.ks - 1;
  }
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

  // initialize buffers used when neighbor at same or coarser level first

  // x1 faces; BufferID = [0,4]
  InitSendIndices(send_buf[0],-1,0,0);
  InitSendIndices(send_buf[4], 1,0,0);

  InitRecvIndices(recv_buf[0],-1,0,0);
  InitRecvIndices(recv_buf[4], 1,0,0);

  for (int n=0; n<=4; n+=4) {
    send_buf[n].AllocateDataView(nmb, nvar);
    recv_buf[n].AllocateDataView(nmb, nvar);
  }

/***
  send_buf[0].InitCoarseIndices(cis,     cis+ng1, cjs, cje, cks, cke);
  recv_buf[0].InitCoarseIndices(cis-ng,  cis-1,   cjs, cje, cks, cke);
  send_buf[4].InitCoarseIndices(cie-ng1, cie,     cjs, cje, cks, cke);
  recv_buf[4].InitCoarseIndices(cie+1,   cie+ng,  cjs, cje, cks, cke);
***/

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces; BufferID = [8,12]
    InitSendIndices(send_buf[8 ],0,-1,0);
    InitSendIndices(send_buf[12],0, 1,0);

    InitRecvIndices(recv_buf[8 ],0,-1,0);
    InitRecvIndices(recv_buf[12],0, 1,0);

    for (int n=8; n<=12; n+=4) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // x1x2 edges; BufferID = [16,18,20,22]
    InitSendIndices(send_buf[16],-1,-1,0);
    InitSendIndices(send_buf[18], 1,-1,0);
    InitSendIndices(send_buf[20],-1, 1,0);
    InitSendIndices(send_buf[22], 1, 1,0);

    InitRecvIndices(recv_buf[16],-1,-1,0);
    InitRecvIndices(recv_buf[18], 1,-1,0);
    InitRecvIndices(recv_buf[20],-1, 1,0);
    InitRecvIndices(recv_buf[22], 1, 1,0);

    for (int n=16; n<=22; n+=2) {
      send_buf[n].AllocateDataView(nmb, nvar);
      recv_buf[n].AllocateDataView(nmb, nvar);
    }

    // add more buffers in 3D
    if (pmy_pack->pmesh->three_d) {

      // x3 faces; BufferID = [24,28]
      InitSendIndices(send_buf[24],0,0,-1);
      InitSendIndices(send_buf[28],0,0, 1);

      InitRecvIndices(recv_buf[24],0,0,-1);
      InitRecvIndices(recv_buf[28],0,0, 1);

      for (int n=24; n<=28; n+=4) {
        send_buf[n].AllocateDataView(nmb, nvar);
        recv_buf[n].AllocateDataView(nmb, nvar);
      }

      // x3x1 edges; BufferID = [32,34,36,38]
      InitSendIndices(send_buf[32],-1,0,-1);
      InitSendIndices(send_buf[34], 1,0,-1);
      InitSendIndices(send_buf[36],-1,0, 1);
      InitSendIndices(send_buf[38], 1,0, 1);

      InitRecvIndices(recv_buf[32],-1,0,-1);
      InitRecvIndices(recv_buf[34], 1,0,-1);
      InitRecvIndices(recv_buf[36],-1,0, 1);
      InitRecvIndices(recv_buf[38], 1,0, 1);

      for (int n=32; n<=38; n+=2) {
        send_buf[n].AllocateDataView(nmb, nvar);
        recv_buf[n].AllocateDataView(nmb, nvar);
      }

      // x2x3 edges; BufferID = [40,42,44,46]
      InitSendIndices(send_buf[40],0,-1,-1);
      InitSendIndices(send_buf[42],0, 1,-1);
      InitSendIndices(send_buf[44],0,-1, 1);
      InitSendIndices(send_buf[46],0, 1, 1);

      InitRecvIndices(recv_buf[40],0,-1,-1);
      InitRecvIndices(recv_buf[42],0, 1,-1);
      InitRecvIndices(recv_buf[44],0,-1, 1);
      InitRecvIndices(recv_buf[46],0, 1, 1);

      for (int n=40; n<=46; n+=2) {
        send_buf[n].AllocateDataView(nmb, nvar);
        recv_buf[n].AllocateDataView(nmb, nvar);
      }

      // corners; BufferID = [48,...,55]
      InitSendIndices(send_buf[48],-1,-1,-1);
      InitSendIndices(send_buf[49], 1,-1,-1);
      InitSendIndices(send_buf[50],-1, 1,-1);
      InitSendIndices(send_buf[51], 1, 1,-1);
      InitSendIndices(send_buf[52],-1,-1, 1);
      InitSendIndices(send_buf[53], 1,-1, 1);
      InitSendIndices(send_buf[54],-1, 1, 1);
      InitSendIndices(send_buf[55], 1, 1, 1);

      InitRecvIndices(recv_buf[48],-1,-1,-1);
      InitRecvIndices(recv_buf[49], 1,-1,-1);
      InitRecvIndices(recv_buf[50],-1, 1,-1);
      InitRecvIndices(recv_buf[51], 1, 1,-1);
      InitRecvIndices(recv_buf[52],-1,-1, 1);
      InitRecvIndices(recv_buf[53], 1,-1, 1);
      InitRecvIndices(recv_buf[54],-1, 1, 1);
      InitRecvIndices(recv_buf[55], 1, 1, 1);

      for (int n=48; n<=55; n+=1) {
        send_buf[n].AllocateDataView(nmb, nvar);
        recv_buf[n].AllocateDataView(nmb, nvar);
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void BValCC::PackAndSendCC()
// \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//
// This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously,
// for ALL the MeshBlocks.  This reduces the number of kernel launches when there are a
// large number of MeshBlocks per MPI rank.  Buffer data are then sent (via MPI) or copied
// directly for periodic or block boundaries.
//
// Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
// 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR 

TaskStatus BValCC::PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mbgid;
  auto &mblev = pmy_pack->pmb->mblev;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;

  // load buffers, using 3 levels of hierarchical parallelism
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  { 
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {

      // if neighbor is at same or finer level, use indices for u0
      // indices same for all variables, stored in (0,i) component
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
        il = sbuf[n].sindcs.bis;
        iu = sbuf[n].sindcs.bie;
        jl = sbuf[n].sindcs.bjs;
        ju = sbuf[n].sindcs.bje;
        kl = sbuf[n].sindcs.bks;
        ku = sbuf[n].sindcs.bke;
      // else if neighbor is at coarser level, use indices for coarse_u0
      } else {
        il = sbuf[n].cindcs.bis;
        iu = sbuf[n].cindcs.bie;
        jl = sbuf[n].cindcs.bjs;
        ju = sbuf[n].cindcs.bje;
        kl = sbuf[n].cindcs.bks;
        ku = sbuf[n].cindcs.bke;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx)
      {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
  
        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank

        if (nghbr.d_view(m,n).rank == my_rank) {
          // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
          // in this MeshBlockPack, so array index equals (target_id - first_id)
          int mm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
          int nn = nghbr.d_view(m,n).dest;
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
            {
              rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
            {
              rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = ca(m,v,k,j,i);
            });
          }

        // else copy into send buffer for MPI communication below

        } else {
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
            {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          // Note in this case, sbuf[n].indcs values refer to coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
            {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = ca(m,v,k,j,i);
            });
          }
        }
      });
    } // end if-block

  }); // end par_for_outer
  }

  // Send boundary buffer to neighboring MeshBlocks using MPI

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
#if MPI_PARALLEL_ENABLED
  auto &sbuf = send_buf;
#endif

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // compute indices of destination MeshBlock and Neighbor 
        int nn = nghbr.h_view(m,n).dest;
        // if MeshBlocks are same rank, data already copied into receive buffer above
        // So simply set communication status tag as received.
        if (nghbr.h_view(m,n).rank == my_rank) {
          int mm = nghbr.h_view(m,n).gid - pmy_pack->gids;
          rbuf[nn].bcomm_stat(mm) = BoundaryCommStatus::received;

#if MPI_PARALLEL_ENABLED
        // Send boundary data using MPI
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid -
                    pmy_pack->pmesh->gidslist[nghbr.h_view(m,n).rank];
          int tag = CreateMPITag(lid, nn, key);
          auto send_data = Kokkos::subview(sbuf[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* send_ptr = send_data.data();
          int ierr = MPI_Isend(send_ptr, send_data.size(), MPI_ATHENA_REAL,
            nghbr.h_view(m,n).rank, tag, MPI_COMM_WORLD, &(sbuf[n].comm_req[m]));
#endif
        }
      }
    }
  }}

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BValCC::RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;

  bool bflag = false;
  {auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
#endif

  //----- STEP 1: check that recv boundary buffer communications have all completed
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank == global_variable::my_rank) {
          if (rbuf[n].bcomm_stat(m) == BoundaryCommStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
        } else {
          MPI_Test(&(rbuf[n].comm_req[m]), &test, MPI_STATUS_IGNORE);
          if (static_cast<bool>(test)) {
            rbuf[n].bcomm_stat(m) = BoundaryCommStatus::received;
          } else {
            bflag = true;
          }
#endif
        }
      }
    }
  }}

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  //----- STEP 2: buffers have all completed, so unpack
  {int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mblev;
  auto &rbuf = recv_buf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {

      // if neighbor is at same or finer level, use indices for u0
      // indices same for all variables, stored in (0,i) component
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
        il = rbuf[n].sindcs.bis;
        iu = rbuf[n].sindcs.bie;
        jl = rbuf[n].sindcs.bjs;
        ju = rbuf[n].sindcs.bje;
        kl = rbuf[n].sindcs.bks;
        ku = rbuf[n].sindcs.bke;
      // else if neighbor is at coarser level, use indices for coarse_u0
      } else {
        il = rbuf[n].cindcs.bis;
        iu = rbuf[n].cindcs.bie;
        jl = rbuf[n].cindcs.bjs;
        ju = rbuf[n].cindcs.bje;
        kl = rbuf[n].cindcs.bks;
        ku = rbuf[n].cindcs.bke;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx)
      {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;
         
        // if neighbor is at same or finer level, load data directly into u0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
          {
            a(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
          });

        // if neighbor is at coarser level, load data into coarse_u0 (and prolongate below)
        // Note in this case, rbuf[n].indcs values refer to coarse_u0
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
          {
            ca(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
          });
        }

      });
    }  // end if-block
  });  // end par_for_outer
  }

  //----- STEP 3: Prolongate conserved variables when neighbor at coarser level
  // Code here is based on MeshRefinement::RestrictCellCenteredValues() in
  // mesh_refinement.cpp in Athena++
  if (pmy_pack->pmesh->multilevel) {
    int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR
    int nmnv = nmb*nnghbr*nvar;
    auto &nghbr = pmy_pack->pmb->nghbr;
    auto &mblev = pmy_pack->pmb->mblev;
    auto &rbuf = recv_buf;

    // Prolongation for 1D problem
    if (pmy_pack->pmesh->one_d) {
      auto &cis = pmy_pack->pcoord->mbdata.cindcs.is;
      auto &cie = pmy_pack->pcoord->mbdata.cindcs.ie;
      auto &is = pmy_pack->pcoord->mbdata.indcs.is;
      auto &ie = pmy_pack->pcoord->mbdata.indcs.ie;
      auto &js = pmy_pack->pcoord->mbdata.indcs.js;
      auto &ks = pmy_pack->pcoord->mbdata.indcs.ks;
      // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
      Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
      Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
      {
        const int m = (tmember.league_rank())/(nnghbr*nvar);
        const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
        const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
        // indices in loop below are on normal mesh
        const int il = rbuf[n].sindcs.bis;
        const int iu = rbuf[n].sindcs.bie;

        // if neighbor is at coarser level, prolongate.  Otherwise do nothing
        if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            // calculate indices of coarse array elements
            int coari;
            if (il < is) {
              coari = (i-is-1)/2 + cis;
            } else {
              coari = (i-ie+1)/2 + cie;
            }

            // calculate 1D gradient using the min-mod limiter
            Real dl = ca(m,v,ks,js,coari  ) - ca(m,v,ks,js,coari-1);
            Real dr = ca(m,v,ks,js,coari+1) - ca(m,v,ks,js,coari  );
            Real dc = 0.5*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

            // interpolate to the finer grid
              a(m,v,ks,js,i) = ca(m,v,ks,js,coari);
/***
            if (i%2 == 0) {
              a(m,v,ks,js,i) = ca(m,v,ks,js,coari) + 0.25*dc ;
            } else {
              a(m,v,ks,js,i) = ca(m,v,ks,js,coari) - 0.25*dc ;
            }
***/
          });
        }
      }); // end par_for_outer
    }
  }

/******
  std::cout << std::endl << "u0 data" << std::endl;
  for (int m=0; m<nmb; ++m) {
    auto &js = pmy_pack->pcoord->mbdata.sindcs.js;
    auto &ks = pmy_pack->pcoord->mbdata.sindcs.ks;
    std::cout << "Block = " << m << "  level = " << pmy_pack->pmb->mblev.h_view(m) << std::endl;
    for (int i=pmy_pack->pcoord->mbdata.sindcs.is-2; i<=pmy_pack->pcoord->mbdata.sindcs.ie+2; ++i) {
      std::cout << "i=" << i << "  d=" << a(m,0,ks,js,i) << std::endl;
    }
  }

  std::cout << std::endl << "coarse u0 data" << std::endl;
  for (int m=0; m<nmb; ++m) {
    auto &js = pmy_pack->pcoord->mbdata.cindcs.js;
    auto &ks = pmy_pack->pcoord->mbdata.cindcs.ks;
    std::cout << "Block = " << m << "  level = " << pmy_pack->pmb->mblev.h_view(m) << std::endl;
    for (int i=pmy_pack->pcoord->mbdata.cindcs.is-2; i<=pmy_pack->pcoord->mbdata.cindcs.ie+2; ++i) {
      std::cout << "i=" << i << "  d=" << ca(m,0,ks,js,i) << std::endl;
    }
  }

*****/

  return TaskStatus::complete;
}
