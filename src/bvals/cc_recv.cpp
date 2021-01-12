//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cc_recv.cpp
//  \brief implements receives of cell-centered variables

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValues::RecvCellCenteredVars()
// \brief Unpack boundary buffers for cell-centered variables.

TaskStatus BoundaryValues::RecvBuffers(AthenaArray4D<Real> &a)
{
  // Find the physics module containing the recv buffer, using bbuf_ptr map and [key]
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int nnghbr = pmb->phydro->recv_buf.size();

#if MPI_PARALLEL_ENABLED
  // probe MPI communications.  This is a bit of black magic that seems to promote
  // communications to top of stack and gets them to complete more quickly
  int test;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
#endif
  bool bflag = false;

  // check that recv boundary buffer communications have all completed
  for (int n=0; n<nnghbr; ++n) {
    if (nghbr[n].gid >= 0) { // ID of buffer != -1, so not a physical boundary
      if (nghbr[n].rank == global_variable::my_rank) {
        if (pmb->phydro->recv_buf[n].bcomm_stat==BoundaryCommStatus::waiting) bflag = true;
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

  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
  
  // buffers have all completed, so unpack
  // create local references for variables in kernel
  auto &recv_buf = pmb->phydro->recv_buf;
  int nvar = a.extent_int(0);

  int scr_level = 1;
  par_for_outer("RecvCC", pmb->exe_space, 0, scr_level, 0, (nnghbr-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m)
    { 
      const int il = recv_buf[m].il;
      const int iu = recv_buf[m].iu;
      const int jl = recv_buf[m].jl;
      const int ju = recv_buf[m].ju;
      const int kl = recv_buf[m].kl;
      const int ku = recv_buf[m].ku;
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
              a(n,k,j,i) = recv_buf[m].data(n, i-il + ni*(j-jl + nj*(k-kl)));
            });
        });
    }
  ); // end par_for_outer

  return TaskStatus::complete;
}
