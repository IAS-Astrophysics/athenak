//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file send_buffers.cpp
//  \brief function to pack and send boundary buffers between MeshBlocks.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
// \!fn void SendBuffers()
// \brief Pack boundary buffers and send to neighbors.
// This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously,
// for ALL the MeshBlocks.  This reduces the number of kernel launches when there are a
// large number of MeshBlocks per MPI rank.  Buffer data are then sent (via MPI) or copied
// directly for periodic or block boundaries.
//
// Input array must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus BoundaryValues::SendBuffers(DvceArray5D<Real> &a, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // 2nd index from L of input array must be NVAR

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mbgid;
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
    const int il = sbuf[n].index.d_view(0);
    const int iu = sbuf[n].index.d_view(1);
    const int jl = sbuf[n].index.d_view(2);
    const int ju = sbuf[n].index.d_view(3);
    const int kl = sbuf[n].index.d_view(4);
    const int ku = sbuf[n].index.d_view(5);
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
      if (nghbr[n].rank.d_view(m) == my_rank) {
        // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
        int mm = nghbr[n].gid.d_view(m) - mbgid.d_view(0);
        int nn = nghbr[n].destn.d_view(m);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu + 1),[&](const int i)
        {
          rbuf[nn].data(mm,v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
        });

      // else copy into send buffer for MPI communication below
      } else {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu + 1),[&](const int i)
        {
          sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = a(m,v,k,j,i);
        });
      }
    });
  }); // end par_for_outer
  }

  // Send boundary buffer to neighboring MeshBlocks using MPI or Kokkos::deep_copy if
  // neighbor is on same MPI rank.

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mbgid;
  auto &rbuf = recv_buf;
  auto &sbuf = send_buf;

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {  // not a physical boundary
        // compute indices of destination MeshBlock and Neighbor 
        int nn = nghbr[n].destn.h_view(m);
        if (nghbr[n].rank.h_view(m) == my_rank) {
          int mm = nghbr[n].gid.h_view(m) - pmy_pack->gids;
          rbuf[nn].bcomm_stat(mm) = BoundaryCommStatus::received;

#if MPI_PARALLEL_ENABLED
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr[n].gid.h_view(m) -
                    pmy_pack->pmesh->gidslist[nghbr[n].rank.h_view(m)];
          int tag = CreateMPITag(lid, nn, key);
          auto send_data = Kokkos::subview(sbuf[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* send_ptr = send_data.data();
          int ierr = MPI_Isend(send_ptr, send_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(sbuf[n].comm_req[m]));
#endif
        }
      }
    }
  }}

  return TaskStatus::complete;
}
