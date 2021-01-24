//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recv_buffers.cpp
//  \brief receives and unpacks boundary buffers

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BoundaryValues::RecvBuffers(DvceArray5D<Real> &a)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // 2nd index from L of input array must be NVAR

  bool bflag = false;
  {auto &nghbr = pmy_pack->pmb->nghbr;
  auto &gid = pmy_pack->pmb->mbgid.h_view;
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
      if (nghbr[n].gid.h_view(m) >= 0) { // ID != -1, so not a physical boundary
        if (nghbr[n].rank.h_view(m) == global_variable::my_rank) {
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

  // buffers have all completed, so unpack
  {int nmnv = nmb*nnghbr*nvar;
  auto &rbuf = recv_buf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);
    const int il = rbuf[n].index.d_view(0);
    const int iu = rbuf[n].index.d_view(1);
    const int jl = rbuf[n].index.d_view(2);
    const int ju = rbuf[n].index.d_view(3);
    const int kl = rbuf[n].index.d_view(4);
    const int ku = rbuf[n].index.d_view(5);
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
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu + 1), [&](const int i)
      {
        a(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
      });
    });
  }); // end par_for_outer
  }

  return TaskStatus::complete;
}
