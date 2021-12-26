//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_fc.cpp
//! \brief functions to pack/send and recv/unpack/prolongate boundary values for
//! face-centered variables, implemented as part of the BValFC class.

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
//! \!fn void BValFC::PackAndSendFC()
//! \brief Pack face-centered variables into boundary buffers and send to neighbors.
//!
//! As for cell-centered data, this routine packs ALL the buffers on ALL the faces, edges,
//! and corners simultaneously for all three components of face-fields on ALL the
//! MeshBlocks.
//!
//! Input array must be DvceFaceFld4D dimensioned (nmb, nx3, nx2, nx1)
//! DvceFaceFld4D of coarsened (restricted) fields also required with SMR/AMR

TaskStatus BValFC::PackAndSendFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  int nnghbr = pmy_pack->pmb->nnghbr;

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = send_buf;
  auto &rbuf = recv_buf;

  // load buffers, using 3 levels of hierarchical parallelism
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  int nmnv = 3*nmb*nnghbr;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  { 
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {

      // if neighbor is at coarser level, use cindices to pack buffer
      // Note indices can be different for each component of face-centered field.
      int il, iu, jl, ju, kl, ku;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = sbuf[n].cindcs[v].bis;
        iu = sbuf[n].cindcs[v].bie;
        jl = sbuf[n].cindcs[v].bjs;
        ju = sbuf[n].cindcs[v].bje;
        kl = sbuf[n].cindcs[v].bks;
        ku = sbuf[n].cindcs[v].bke;
      // if neighbor is at same level, use sindices to pack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = sbuf[n].sindcs[v].bis;
        iu = sbuf[n].sindcs[v].bie;
        jl = sbuf[n].sindcs[v].bjs;
        ju = sbuf[n].sindcs[v].bje;
        kl = sbuf[n].sindcs[v].bks;
        ku = sbuf[n].sindcs[v].bke;
      // if neighbor is at finer level, use findices to pack buffer
      } else {
        il = sbuf[n].findcs[v].bis;
        iu = sbuf[n].findcs[v].bie;
        jl = sbuf[n].findcs[v].bjs;
        ju = sbuf[n].findcs[v].bje;
        kl = sbuf[n].findcs[v].bks;
        ku = sbuf[n].findcs[v].bke;
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
        // copy field components directly into recv buffer if MeshBlocks on same rank

        if (nghbr.d_view(m,n).rank == my_rank) {
          // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
          int mm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
          int nn = nghbr.d_view(m,n).dest;
          // if neighbor is at same or finer level, load data from b0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
              });
            }
          // if neighbor is at coarser level, load data from coarse_b0
          } else {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = cb.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = cb.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                rbuf[nn].data(mm, v, i-il + ni*(j-jl + nj*(k-kl))) = cb.x3f(m,k,j,i);
              });
            }
          }

        // else copy field components into send buffer for MPI communication below

        } else {
          // if neighbor is at same or finer level, load data from b0
          if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = b.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = b.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = b.x3f(m,k,j,i);
              });
            }
          // if neighbor is at coarser level, load data from coarse_b0
          } else {
            if (v==0) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = cb.x1f(m,k,j,i);
              });
            } else if (v==1) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = cb.x2f(m,k,j,i);
              });
            } else {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i)
              {
                sbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl))) = cb.x3f(m,k,j,i);
              });
            }
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
        } else {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid -
                    pmy_pack->pmesh->gidslist[nghbr.h_view(m,n).rank];
          int tag = CreateMPITag(lid, nn, key);
          auto send_data = Kokkos::subview(sbuf[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* send_ptr = send_data.data();
          int data_size=0;
          // if neighbor is at coarser level, use cindices size
          if (nghbr.h_view(m,n).lev < mblev.h_view(m)) {
            for (int v=0; v<3; ++v) {data_size += sbuf[n].cindcs[v].ndat;}
          // if neighbor is at same level, use sindices size
          } else if (nghbr.h_view(m,n).lev == mblev.h_view(m)) {
            for (int v=0; v<3; ++v) {data_size += sbuf[n].sindcs[v].ndat;}
          // if neighbor is at finer level, use findices size
          } else {
            for (int v=0; v<3; ++v) {data_size += sbuf[n].findcs[v].ndat;}
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

TaskStatus BValFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb)
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
      if (nghbr.h_view(m,n).gid >= 0) { // ID != -1, so not a physical boundary
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

  //----- STEP 2: buffers have all completed, so unpack 3-components of field

  {int nmnv = 3*nmb*nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf = recv_buf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = (tmember.league_rank())/(3*nnghbr);
    const int n = (tmember.league_rank() - m*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - m*(3*nnghbr) - 3*n);

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      
      // if neighbor is at coarser level, use cindices to unpack buffer
      int il, iu, jl, ju, kl, ku; 
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = rbuf[n].cindcs[v].bis;
        iu = rbuf[n].cindcs[v].bie;
        jl = rbuf[n].cindcs[v].bjs;
        ju = rbuf[n].cindcs[v].bje;
        kl = rbuf[n].cindcs[v].bks;
        ku = rbuf[n].cindcs[v].bke;
      // if neighbor is at same level, use sindices to unpack buffer
      } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
        il = rbuf[n].sindcs[v].bis;
        iu = rbuf[n].sindcs[v].bie;
        jl = rbuf[n].sindcs[v].bjs;
        ju = rbuf[n].sindcs[v].bje;
        kl = rbuf[n].sindcs[v].bks;
        ku = rbuf[n].sindcs[v].bke;
      // if neighbor is at finer level, use findices to unpack buffer
      } else {
        il = rbuf[n].findcs[v].bis;
        iu = rbuf[n].findcs[v].bie;
        jl = rbuf[n].findcs[v].bjs;
        ju = rbuf[n].findcs[v].bje;
        kl = rbuf[n].findcs[v].bks;
        ku = rbuf[n].findcs[v].bke;
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
        // copy contents of recv_buf into appropriate vector components

        // if neighbor is at same or finer level, load data directly into b0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          if (v==0) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              b.x1f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else if (v==1) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              b.x2f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              b.x3f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
            });
          }
        // if neighbor is at coarser level, load data into coarse_b0 (prolongate below)
        } else {
          if (v==0) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              cb.x1f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else if (v==1) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              cb.x2f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
            });
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
            {
              cb.x3f(m,k,j,i) = rbuf[n].data(m, v, i-il + ni*(j-jl + nj*(k-kl)));
            });
          }
        }
      });
    }  // end if-neighbor-exists block
  });  // end par_for_outer
  }

  //----- STEP 3: Prolongate face-fields when neighbor at coarser level

  // Only perform prolongation with SMR/AMR
  if (pmy_pack->pmesh->multilevel) ProlongFC(b, cb);

  return TaskStatus::complete;
}
