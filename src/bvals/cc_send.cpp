//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cc_send.cpp
//  \brief implements sends for cell-centered variables

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::SendCellCenteredVars()
// \brief Pack boundary buffers for cell-centered variables, and send to neighbors
// This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously
// They are then sent (via MPI) or copied directly for periodic or block boundaries.
// Input array must be 4D Kokkos View dimensioned (nvar, nx3, nx2, nx1)

TaskStatus BoundaryValues::SendBuffers(AthenaArray4D<Real> &a)
{
  // create local references for variables in kernel
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  auto &send_buf = pmb->phydro->send_buf;
  int nnghbr = pmb->phydro->send_buf.size();
  int nvar = a.extent_int(0);

  // load buffers, using 3 levels of hierarchical parallelism
  int scr_level = 1;
  par_for_outer("SendCC", pmb->exe_space, 0, scr_level, 0, (nnghbr-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m)
    {
      const int il = send_buf[m].il;
      const int iu = send_buf[m].iu;
      const int jl = send_buf[m].jl;
      const int ju = send_buf[m].ju;
      const int kl = send_buf[m].kl;
      const int ku = send_buf[m].ku;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj  = nk*nj;
      const int nnkj = nvar*nk*nj;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange<>(tmember, nnkj), [&](const int idx) {
          int n = idx / nkj;
          int k = (idx - n * nkj) / nj;
          int j = idx - n * nkj - k * nj;
          k += kl;
          j += jl;

          Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(tmember, il, iu + 1), [&](const int i) {
              send_buf[m].data(n, i-il + ni*(j-jl + nj*(k-kl))) = a(n, k, j, i);
            });
        });
    }
  ); // end par_for_outer

  // Send boundary buffer to neighboring MeshBlocks using MPI or Kokkos::deep_copy if
  // neighbor is on same MPI rank.
  // Note send_buf[n] --> recv_buf[n + nghbr[n].dn]

  using Kokkos::ALL;
  for (int n=0; n<nnghbr; ++n) {
    if (nghbr[n].gid >= 0) {  // ID of buffer != -1, so not a physical boundary
      if (nghbr[n].rank == global_variable::my_rank) {
        Kokkos::deep_copy(pmb->exe_space,
          pmesh_->FindMeshBlock(nghbr[n].gid)->phydro->recv_buf[n + nghbr[n].dn].data,
          send_buf[n].data);
        pmesh_->FindMeshBlock(nghbr[n].gid)->phydro->recv_buf[n + nghbr[n].dn].bcomm_stat
          = BoundaryCommStatus::received;
#if MPI_PARALLEL_ENABLED
      } else {
        // create tag using local ID and buffer index of *receiving* MeshBlock
        int lid = nghbr_x1face[n].gid - pmesh_->gidslist[nghbr_x1face[n].rank];
        int tag = CreateMPItag(lid, (1-n), key);
        void* send_ptr = sendbuf.data();
        int ierr = MPI_Isend(send_ptr, sendbuf.size(), MPI_ATHENA_REAL,
          nghbr_x1face[n].rank, tag, MPI_COMM_WORLD, &(pbb->send_rq_x1face[n]));
#endif
      }
    }
  }

  return TaskStatus::complete;
}
