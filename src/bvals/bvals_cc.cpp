//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions to pack/send and recv/unbpack boundary values for cell-centered
//!  variables. This functionality is mplemented in BValCC class.

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
          // Note in this case, sbuf[n].indcs values refer to coarse_u0
          } else {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i)
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
  auto &mblev = pmy_pack->pmb->mblev;
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
          if (rbuf[n].bcomm_stat(m) == BoundaryCommStatus::waiting) {
            bflag = true;
/***
std::cout << "block=" << m << "  buffer=" << n << "  not received" << std::endl;
***/
          }
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
        // Note in this case, rbuf[n].indcs values refer to coarse_u0
        } else {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
          {
            ca(m,v,k,j,i) = rbuf[n].data(m,v,i-il + ni*(j-jl + nj*(k-kl)));
          });
        }

      });
    }  // end if-block
  });  // end par_for_outer
  }

  //----- STEP 3: Prolongate conserved variables when neighbor at coarser level
  // Code here is based on MeshRefinement::ProlongateCellCenteredValues() in C++ version

  // Only perform prolongation with SMR/AMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  int nvar = a.extent_int(1);  // TODO: 2nd index from L of input array must be NVAR
  int nmnv = nmb*nnghbr*nvar;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mblev;
  auto &rbuf = recv_buf;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &indcs  = pmy_pack->pcoord->mbdata.indcs;
  auto &cindcs = pmy_pack->pcoord->mbdata.cindcs;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember)
  {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {

      // loop over indices for prolongation on this buffer
      int il = rbuf[n].pindcs.bis;
      int iu = rbuf[n].pindcs.bie;
      int jl = rbuf[n].pindcs.bjs;
      int ju = rbuf[n].pindcs.bje;
      int kl = rbuf[n].pindcs.bks;
      int ku = rbuf[n].pindcs.bke;
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

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),[&](const int i)
        {
          // indices for prolongation (pindcs) refer to coarse array.  So must compute
          // indices for fine array
          int finei = (i - cindcs.is)*2 + indcs.is;
          int finej = (j - cindcs.js)*2 + indcs.js;
          int finek = (k - cindcs.ks)*2 + indcs.ks;

/*
std::cout << std::endl << "MB= "<<m<<"  Buffer="<< n << std::endl;
std::cout <<il<<"  "<<iu<<"  "<<jl<<"  "<<ju<<"  "<<kl<<"  "<<ku<< std::endl;
std::cout << "finei=" << finei << "  finej=" << finej << "  finek=" << finek << std::endl;
*/

          // calculate x1-gradient using the min-mod limiter
          Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
          Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
          Real dvar1 = 0.25*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

          // calculate x2-gradient using the min-mod limiter
          Real dvar2 = 0.0;
          if (multi_d) {
            dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
            dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
            dvar2 = 0.25*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
          }

          // calculate x1-gradient using the min-mod limiter
          Real dvar3 = 0.0;
          if (three_d) {
            dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
            dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
            dvar3 = 0.25*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
          }

          // interpolate to the finer grid
          a(m,v,finek,finej,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
          a(m,v,finek,finej,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
          if (multi_d) {
            a(m,v,finek,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
            a(m,v,finek,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
          }
          if (three_d) {
            a(m,v,finek+1,finej  ,finei  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
            a(m,v,finek+1,finej  ,finei+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
            a(m,v,finek+1,finej+1,finei  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
            a(m,v,finek+1,finej+1,finei+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
          }

        });
      });
    }
  });

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
