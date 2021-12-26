//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions to pack/send and recv/unpack/prolongate boundary values for
//! cell-centered variables, implemented as part of the BValCC class.

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
//! \fn void BValCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//!
//! This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously
//! for ALL the MeshBlocks. This reduces the number of kernel launches when there are a
//! large number of MeshBlocks per MPI rank. Buffer data are then sent (via MPI) or copied
//! directly for periodic or block boundaries.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
//! 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR 

TaskStatus BValCC::PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
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

      // if neighbor is at coarser level, use cindices to pack buffer
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = sbuf[n].cindcs.bis;
        iu = sbuf[n].cindcs.bie;
        jl = sbuf[n].cindcs.bjs;
        ju = sbuf[n].cindcs.bje;
        kl = sbuf[n].cindcs.bks;
        ku = sbuf[n].cindcs.bke;
      // if neighbor is at same level, use sindices to pack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = sbuf[n].sindcs.bis;
        iu = sbuf[n].sindcs.bie;
        jl = sbuf[n].sindcs.bjs;
        ju = sbuf[n].sindcs.bje;
        kl = sbuf[n].sindcs.bks;
        ku = sbuf[n].sindcs.bke;
      // if neighbor is at finer level, use findices to pack buffer
      } else {
        il = sbuf[n].findcs.bis;
        iu = sbuf[n].findcs.bie;
        jl = sbuf[n].findcs.bjs;
        ju = sbuf[n].findcs.bje;
        kl = sbuf[n].findcs.bks;
        ku = sbuf[n].findcs.bke;
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
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = ca(m,v,k,j,i);
            });
          }

        // else copy into send buffer for MPI communication below

        } else {
          // if neighbor is at same or finer level, load data from u0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
            });
          // if neighbor is at coarser level, load data from coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = ca(m,v,k,j,i);
            });
          }
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer
  }

  // Send boundary buffer to neighboring MeshBlocks using MPI

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
  auto &mblev = pmy_pack->pmb->mb_lev;
#if MPI_PARALLEL_ENABLED
  auto &sbuf = send_buf;
#endif

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // compute indices of destination MeshBlock and Neighbor 
        int nn = nghbr.h_view(m,n).dest;
        // if MeshBlocks are on same rank, data already copied into receive buffer above
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
          int data_size;
          // if neighbor is at coarser level, use cindices size
          if (nghbr.h_view(m,n).lev < mblev.h_view(m)) {
            data_size = (sbuf[n].cindcs.ndat)*nvar;
          // if neighbor is at same level, use sindices size
          } else if (nghbr.h_view(m,n).lev == mblev.h_view(m)) {
            data_size = (sbuf[n].sindcs.ndat)*nvar;
          // if neighbor is at finer level, use findices size
          } else {
            data_size = (sbuf[n].findcs.ndat)*nvar;
          }
          int ierr = MPI_Isend(send_ptr, data_size, MPI_ATHENA_REAL,
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
          if (rbuf[n].bcomm_stat(m) == BoundaryCommStatus::waiting) {bflag = true;}
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
  auto &mblev = pmy_pack->pmb->mb_lev;
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

      // if neighbor is at coarser level, use cindices to unpack buffer
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = rbuf[n].cindcs.bis;
        iu = rbuf[n].cindcs.bie;
        jl = rbuf[n].cindcs.bjs;
        ju = rbuf[n].cindcs.bje;
        kl = rbuf[n].cindcs.bks;
        ku = rbuf[n].cindcs.bke;
      // if neighbor is at same level, use sindices to unpack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = rbuf[n].sindcs.bis;
        iu = rbuf[n].sindcs.bie;
        jl = rbuf[n].sindcs.bjs;
        ju = rbuf[n].sindcs.bje;
        kl = rbuf[n].sindcs.bks;
        ku = rbuf[n].sindcs.bke;
      // if neighbor is at finer level, use findices to unpack buffer
      } else {
        il = rbuf[n].findcs.bis;
        iu = rbuf[n].findcs.bie;
        jl = rbuf[n].findcs.bjs;
        ju = rbuf[n].findcs.bje;
        kl = rbuf[n].findcs.bks;
        ku = rbuf[n].findcs.bke;
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
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            a(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
          });

        // if neighbor is at coarser level, load data into coarse_u0 (prolongate below)
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            ca(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
          });
        }

      });
    }  // end if-neighbor-exists block
  });  // end par_for_outer
  }

  //----- STEP 3: Prolongate conserved variables when neighbor at coarser level

  // Only perform prolongation with SMR/AMR
  if (pmy_pack->pmesh->multilevel) ProlongCC(a,ca);

  return TaskStatus::complete;
}
