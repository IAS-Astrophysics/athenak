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

TaskStatus RecvBuffers(AthenaArray5D<Real> &a,
  std::vector<std::vector<BoundaryBuffer>> &send_buf,
  std::vector<std::vector<BoundaryBuffer>> &recv_buf, std::vector<MeshBlock> &mblocks)
{
  // create local references for variables in kernel
  int nmb  = mblocks.size();
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = mblocks[0].nghbr.size();
  int nvar = a.extent_int(1);  // 2nd index from L of input array must be NVAR

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
#endif
  bool bflag = false;

  // check that recv boundary buffer communications have all completed
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (mblocks[m].nghbr[n].gid >= 0) { // ID != -1, so not a physical boundary
        if (mblocks[m].nghbr[n].rank == global_variable::my_rank) {
          if (recv_buf[m][n].bcomm_stat == BoundaryCommStatus::waiting) bflag = true;
#if MPI_PARALLEL_ENABLED
        } else {
          MPI_Test(&(pbb->recv_rq_x1face[n]), &test, MPI_STATUS_IGNORE);
          if (static_cast<bool>(test)) {
            pbb->bstat_x1face[n] = BoundaryCommStatus::completed;
          } else {
            bflag = true;
          }
#endif
        }
      }
    }
  }

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  // buffers have all completed, so unpack
  int nmn = nmb*nnghbr;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = tmember.league_rank()/nnghbr;
    const int n = tmember.league_rank()%nnghbr;
    const int il = recv_buf[m][n].index(0);
    const int iu = recv_buf[m][n].index(1);
    const int jl = recv_buf[m][n].index(2);
    const int ju = recv_buf[m][n].index(3);
    const int kl = recv_buf[m][n].index(4);
    const int ku = recv_buf[m][n].index(5);
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nkj  = nk*nj;
    const int nvkj = nvar*nk*nj;

    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nvkj), [&](const int idx)
    {
      int v = idx / nkj;
      int k = (idx - v * nkj) / nj;
      int j = idx - v * nkj - k * nj;
      k += kl;
      j += jl;
         
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu + 1), [&](const int i)
      {
        a(m,v,k,j,i) = recv_buf[m][n].data(v,i-il + ni*(j-jl + nj*(k-kl)));
       });
    });
  }); // end par_for_outer

  return TaskStatus::complete;
}
