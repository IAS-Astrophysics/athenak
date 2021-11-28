//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_fc.cpp
//  \brief functions to pass boundary values for face-centered variables as implmented in
//  BValFC class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "utils/create_mpitag.hpp"

//----------------------------------------------------------------------------------------
// BValFC constructor:

BValFC::BValFC(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp)
{
} 
  
//----------------------------------------------------------------------------------------
// BValFC destructor
  
BValFC::~BValFC()
{
}

//----------------------------------------------------------------------------------------
// \!fn void BValFC::PackAndSendFC()
// \brief Pack face-centered variables into boundary buffers and send to neighbors.
//
// Input array must be DvceFaceFld4D dimensioned (nmb, nx3, nx2, nx1)

TaskStatus BValFC::PackAndSendFC(DvceFaceFld4D<Real> &b, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;

  // load buffers, using 3 levels of hierarchical parallelism
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of components of field = 3)
  int nmnv = 3*nmb*nnghbr;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  { 
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);
    // get indices of bbuf for each field component
    const int il = sbuf[n].index.d_view(v,0);
    const int iu = sbuf[n].index.d_view(v,1);
    const int jl = sbuf[n].index.d_view(v,2);
    const int ju = sbuf[n].index.d_view(v,3);
    const int kl = sbuf[n].index.d_view(v,4);
    const int ku = sbuf[n].index.d_view(v,5);
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
      // copy field components directly into recv buffer if MeshBlocks on same rank
      if (nghbr.d_view(m,n).rank == my_rank) {
        // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
        int mm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int nn = nghbr.d_view(m,n).dest;
        if (v==0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            rbuf[nn].data(mm, v, (i-il) + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
          });
        } else if (v==1) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            rbuf[nn].data(mm, v, (i-il) + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            rbuf[nn].data(mm, v, (i-il) + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
          });
        }

      // else copy field components into send buffer for MPI communication below
      } else {
        if (v==0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            sbuf[n].data(m, v, (i-il) + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
          });
        } else if (v==1) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            sbuf[n].data(m, v, (i-il) + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
          });
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            sbuf[n].data(m, v, (i-il) + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
          });
        }
      }
    });
  }); // end par_for_outer
  }

  // Send boundary buffer to neighboring MeshBlocks using MPI or Kokkos::deep_copy if
  // neighbor is on same MPI rank.

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recv_buf;
#if MPI_PARALLEL_ENABLED
  auto &sbuf = send_buf;
#endif

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // not a physical boundary
        // compute indices of destination MeshBlock and Neighbor 
        int nn = nghbr.h_view(m,n).dest;
        if (nghbr.h_view(m,n).rank == my_rank) {
          int mm = nghbr.h_view(m,n).gid - pmy_pack->gids;
          rbuf[nn].bcomm_stat(mm) = BoundaryCommStatus::received;

#if MPI_PARALLEL_ENABLED
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

TaskStatus BValFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b)
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

  // check that recv boundary buffer communications have all completed
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // ID != -1, so not a physical boundary
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

  // buffers have all completed, so unpack 3-components of field
  {int nmnv = 3*nmb*nnghbr;
  auto &rbuf = recv_buf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);
    const int il = rbuf[n].index.d_view(v,0);
    const int iu = rbuf[n].index.d_view(v,1);
    const int jl = rbuf[n].index.d_view(v,2);
    const int ju = rbuf[n].index.d_view(v,3);
    const int kl = rbuf[n].index.d_view(v,4);
    const int ku = rbuf[n].index.d_view(v,5);
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
      // copy contents of recv_buf into appropriate vector components
      if (v==0) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
        {
          b.x1f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
        });
      } else if (v==1) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
        {
          b.x2f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
        });
      } else {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1), [&](const int i)
        {
          b.x3f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
        });
      }
    });
  }); // end par_for_outer
  }

  return TaskStatus::complete;
}
