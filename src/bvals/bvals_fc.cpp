//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_fc.cpp
//  \brief implementation of functions in BoundaryValueFC class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "utils/create_mpitag.hpp"

//----------------------------------------------------------------------------------------
// BoundaryValueFC constructor:

BoundaryValueFC::BoundaryValueFC(MeshBlockPack *pp, ParameterInput *pin) : pmy_pack(pp)
{
} 
  
//----------------------------------------------------------------------------------------
// BoundaryValueFC destructor
  
BoundaryValueFC::~BoundaryValueFC()
{
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValueFC::AllocateBuffersFC
// initialize array of send/recv BoundaryBuffers for each of the three components of
// face-centered variables (vectors), such as magnetic field.
//
// NOTE: order of array elements is crucial and cannot be changed.  It must match
// order of boundaries in nghbr vector

// TODO: extend for AMR

void BoundaryValueFC::AllocateBuffersFC()
{
  auto &indcs = pmy_pack->coord.coord_data.mb_indcs;
  int ng = indcs.ng;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ng1 = ng-1;
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;

  // allocate size of (some) Views
  for (int n=0; n<nnghbr; ++n) {
    // 18 values of index array stores loop bounds for each field component in bbuf
    Kokkos::realloc(send_buf[n].index, 3, 6);
    Kokkos::realloc(recv_buf[n].index, 3, 6);
    Kokkos::realloc(send_buf[n].bcomm_stat, nmb);
    Kokkos::realloc(recv_buf[n].bcomm_stat, nmb);
#if MPI_PARALLEL_ENABLED
    // cannot create Kokkos::View of type MPI_Request so construct STL vector instead
    for (int m=0; m<nmb; ++m) {
      MPI_Request send_req, recv_req;
      send_buf[n].comm_req.push_back(send_req);
      recv_buf[n].comm_req.push_back(recv_req);
    }
#endif
  }

  // Note in following, arguments are loop bounds for 1/2/3-component of face-centered
  // field for each boundary buffer.

  // initialize buffers for x1 faces
  send_buf[0].InitIndices(nmb, is+1,   is+ng,  js, je  , ks, ke,
                               is,     is+ng1, js, je+1, ks, ke,
                               is,     is+ng1, js, je  , ks, ke+1);
  send_buf[1].InitIndices(nmb, ie-ng1, ie,     js, je  , ks, ke,
                               ie-ng1, ie,     js, je+1, ks, ke,
                               ie-ng1, ie,     js, je,   ks, ke+1);

  recv_buf[0].InitIndices(nmb, is-ng, is-1,    js, je  , ks, ke,
                               is-ng, is-1,    js, je+1, ks, ke,
                               is-ng, is-1,    js, je  , ks, ke+1);
  recv_buf[1].InitIndices(nmb, ie+2,  ie+ng+1, js, je  , ks, ke,
                               ie+1,  ie+ng,   js, je+1, ks, ke,
                               ie+1,  ie+ng,   js, je  , ks, ke+1);

  // add more buffers in 2D
  if (nnghbr > 2) {
    // initialize buffers for x2 faces
    send_buf[2].InitIndices(nmb, is, ie+1, js,     js+ng1, ks, ke,
                                 is, ie,   js+1,   js+ng,  ks, ke,
                                 is, ie,   js,     js+ng1, ks, ke+1);
    send_buf[3].InitIndices(nmb, is, ie+1, je-ng1, je,     ks, ke,
                                 is, ie,   je-ng1, je,     ks, ke,
                                 is, ie,   je-ng1, je,     ks, ke+1);

    recv_buf[2].InitIndices(nmb, is, ie+1, js-ng, js-1,    ks, ke,
                                 is, ie,   js-ng, js-1,    ks, ke,
                                 is, ie,   js-ng, js-1,    ks, ke+1);
    recv_buf[3].InitIndices(nmb, is, ie+1, je+1,  je+ng,   ks, ke,
                                 is, ie  , je+2,  je+ng+1, ks, ke,
                                 is, ie  , je+1,  je+ng,   ks, ke+1);

    // initialize buffers for x1x2 edges
    send_buf[4].InitIndices(nmb, is+1,   is+ng,  js,     js+ng1, ks, ke,
                                 is,     is+ng1, js+1,   js+ng,  ks, ke,
                                 is,     is+ng1, js,     js+ng1, ks, ke+1);
    send_buf[5].InitIndices(nmb, ie-ng1, ie,     js,     js+ng1, ks, ke,
                                 ie-ng1, ie,     js+1,   js+ng,  ks, ke,
                                 ie-ng1, ie,     js,     js+ng1, ks, ke+1);
    send_buf[6].InitIndices(nmb, is+1,   is+ng,  je-ng1, je,     ks, ke,
                                 is,     is+ng1, je-ng1, je,     ks, ke,
                                 is,     is+ng1, je-ng1, je,     ks, ke+1);
    send_buf[7].InitIndices(nmb, ie-ng1, ie,     je-ng1, je,     ks, ke,
                                 ie-ng1, ie,     je-ng1, je,     ks, ke,
                                 ie-ng1, ie,     je-ng1, je,     ks, ke+1);

    recv_buf[4].InitIndices(nmb, is-ng, is-1,    js-ng, js-1,    ks, ke,
                                 is-ng, is-1,    js-ng, js-1,    ks, ke,
                                 is-ng, is-1,    js-ng, js-1,    ks, ke+1);
    recv_buf[5].InitIndices(nmb, ie+2,  ie+ng+1, js-ng, js-1,    ks, ke,
                                 ie+1,  ie+ng,   js-ng, js-1,    ks, ke,
                                 ie+1,  ie+ng,   js-ng, js-1,    ks, ke+1);
    recv_buf[6].InitIndices(nmb, is-ng, is-1,    je+1,  je+ng,   ks, ke,
                                 is-ng, is-1,    je+2,  je+ng+1, ks, ke,
                                 is-ng, is-1,    je+1,  je+ng,   ks, ke+1);
    recv_buf[7].InitIndices(nmb, ie+2,  ie+ng+1, je+1,  je+ng,   ks, ke,
                                 ie+1,  ie+ng,   je+2,  je+ng+1, ks, ke,
                                 ie+1,  ie+ng,   je+1,  je+ng,   ks, ke+1);

    // add more buffers in 3D
    if (nnghbr > 8) {

      // initialize buffers for x3 faces
      send_buf[8].InitIndices(nmb, is, ie+1, js, je,   ks,   ks+ng1,
                                   is, ie,   js, je+1, ks,   ks+ng1,
                                   is, ie,   js, je,   ks+1, ks+ng);
      send_buf[9].InitIndices(nmb, is, ie+1, js, je,   ke-ng1, ke,
                                   is, ie,   js, je+1, ke-ng1, ke,
                                   is, ie,   js, je,   ke-ng1, ke);
    
      recv_buf[8].InitIndices(nmb, is, ie+1, js, je,   ks-ng, ks-1,
                                   is, ie,   js, je+1, ks-ng, ks-1,
                                   is, ie,   js, je,   ks-ng, ks-1);
      recv_buf[9].InitIndices(nmb, is, ie+1, js, je,   ke+1,  ke+ng,
                                   is, ie,   js, je+1, ke+1,  ke+ng,
                                   is, ie,   js, je,   ke+2,  ke+ng+1);

      // initialize buffers for x3x1 edges
      send_buf[10].InitIndices(nmb, is+1,   is+ng,  js, je,   ks,     ks+ng1,
                                    is,     is+ng1, js, je+1, ks,     ks+ng1,
                                    is,     is+ng1, js, je,   ks+1,   ks+ng);
      send_buf[11].InitIndices(nmb, ie-ng1, ie,     js, je,   ks,     ks+ng1,
                                    ie-ng1, ie,     js, je+1, ks,     ks+ng1,
                                    ie-ng1, ie,     js, je,   ks+1,   ks+ng);
      send_buf[12].InitIndices(nmb, is+1,   is+ng,  js, je,   ke-ng1, ke,
                                    is,     is+ng1, js, je+1, ke-ng1, ke,
                                    is,     is+ng1, js, je,   ke-ng1, ke);
      send_buf[13].InitIndices(nmb, ie-ng1, ie,     js, je,   ke-ng1, ke,
                                    ie-ng1, ie,     js, je+1, ke-ng1, ke,
                                    ie-ng1, ie,     js, je,   ke-ng1, ke);
    
      recv_buf[10].InitIndices(nmb, is-ng, is-1,    js, je,   ks-ng, ks-1,
                                    is-ng, is-1,    js, je+1, ks-ng, ks-1,
                                    is-ng, is-1,    js, je,   ks-ng, ks-1);
      recv_buf[11].InitIndices(nmb, ie+2,  ie+ng+1, js, je,   ks-ng, ks-1,
                                    ie+1,  ie+ng,   js, je+1, ks-ng, ks-1,
                                    ie+1,  ie+ng,   js, je,   ks-ng, ks-1);
      recv_buf[12].InitIndices(nmb, is-ng, is-1,    js, je,   ke+1,  ke+ng,
                                    is-ng, is-1,    js, je+1, ke+1,  ke+ng,
                                    is-ng, is-1,    js, je,   ke+2,  ke+ng+1);
      recv_buf[13].InitIndices(nmb, ie+2,  ie+ng+1, js, je,   ke+1,  ke+ng,
                                    ie+1,  ie+ng,   js, je+1, ke+1,  ke+ng,
                                    ie+1,  ie+ng,   js, je,   ke+2,  ke+ng+1);

      // initialize buffers for x2x3 edges
      send_buf[14].InitIndices(nmb, is, ie+1, js,     js+ng1, ks,     ks+ng1,
                                    is, ie,   js+1,   js+ng,  ks,     ks+ng1,
                                    is, ie,   js,     js+ng1, ks+1,   ks+ng);
      send_buf[15].InitIndices(nmb, is, ie+1, je-ng1, je,     ks,     ks+ng1,
                                    is, ie,   je-ng1, je,     ks,     ks+ng1,
                                    is, ie,   je-ng1, je,     ks+1,   ks+ng);
      send_buf[16].InitIndices(nmb, is, ie+1, js,     js+ng1, ke-ng1, ke,
                                    is, ie,   js+1,   js+ng,  ke-ng1, ke,
                                    is, ie,   js,     js+ng1, ke-ng1, ke);
      send_buf[17].InitIndices(nmb, is, ie+1, je-ng1, je,     ke-ng1, ke,
                                    is, ie,   je-ng1, je,     ke-ng1, ke,
                                    is, ie,   je-ng1, je,     ke-ng1, ke);
  
      recv_buf[14].InitIndices(nmb, is, ie+1, js-ng, js-1,    ks-ng, ks-1,
                                    is, ie,   js-ng, js-1,    ks-ng, ks-1,
                                    is, ie,   js-ng, js-1,    ks-ng, ks-1);
      recv_buf[15].InitIndices(nmb, is, ie+1, je+1,  je+ng,   ks-ng, ks-1,
                                    is, ie,   je+2,  je+ng+1, ks-ng, ks-1,
                                    is, ie,   je+1,  je+ng,   ks-ng, ks-1);
      recv_buf[16].InitIndices(nmb, is, ie+1, js-ng, js-1,    ke+1,  ke+ng,
                                    is, ie,   js-ng, js-1,    ke+1,  ke+ng,
                                    is, ie,   js-ng, js-1,    ke+2,  ke+ng+1);
      recv_buf[17].InitIndices(nmb, is, ie+1, je+1,  je+ng,   ke+1,  ke+ng,
                                    is, ie,   je+2,  je+ng+1, ke+1,  ke+ng,
                                    is, ie,   je+1,  je+ng,   ke+2,  ke+ng+1);

      // initialize buffers for corners
      send_buf[18].InitIndices(nmb, is+1,   is+ng,  js,     js+ng1, ks,     ks+ng1,
                                    is,     is+ng1, js+1,   js+ng,  ks,     ks+ng1,
                                    is,     is+ng1, js,     js+ng1, ks+1,   ks+ng);
      send_buf[19].InitIndices(nmb, ie-ng1, ie,     js,     js+ng1, ks,     ks+ng1,
                                    ie-ng1, ie,     js+1,   js+ng,  ks,     ks+ng1,
                                    ie-ng1, ie,     js,     js+ng1, ks+1,   ks+ng);
      send_buf[20].InitIndices(nmb, is+1,   is+ng,  je-ng1, je,     ks,     ks+ng1,
                                    is,     is+ng1, je-ng1, je,     ks,     ks+ng1,
                                    is,     is+ng1, je-ng1, je,     ks+1,   ks+ng);
      send_buf[21].InitIndices(nmb, ie-ng1, ie,     je-ng1, je,     ks,     ks+ng1,
                                    ie-ng1, ie,     je-ng1, je,     ks,     ks+ng1,
                                    ie-ng1, ie,     je-ng1, je,     ks+1,   ks+ng);
      send_buf[22].InitIndices(nmb, is+1,   is+ng,  js,     js+ng1, ke-ng1, ke,
                                    is,     is+ng1, js+1,   js+ng,  ke-ng1, ke,
                                    is,     is+ng1, js,     js+ng1, ke-ng1, ke);
      send_buf[23].InitIndices(nmb, ie-ng1, ie,     js,     js+ng1, ke-ng1, ke,
                                    ie-ng1, ie,     js+1,   js+ng,  ke-ng1, ke,
                                    ie-ng1, ie,     js,     js+ng1, ke-ng1, ke);
      send_buf[24].InitIndices(nmb, is+1,   is+ng,  je-ng1, je,     ke-ng1, ke,
                                    is,     is+ng1, je-ng1, je,     ke-ng1, ke,
                                    is,     is+ng1, je-ng1, je,     ke-ng1, ke);
      send_buf[25].InitIndices(nmb, ie-ng1, ie,     je-ng1, je,     ke-ng1, ke,
                                    ie-ng1, ie,     je-ng1, je,     ke-ng1, ke,
                                    ie-ng1, ie,     je-ng1, je,     ke-ng1, ke);

      recv_buf[18].InitIndices(nmb, is-ng, is-1,    js-ng, js-1,    ks-ng, ks-1,
                                    is-ng, is-1,    js-ng, js-1,    ks-ng, ks-1,
                                    is-ng, is-1,    js-ng, js-1,    ks-ng, ks-1);
      recv_buf[19].InitIndices(nmb, ie+2,  ie+ng+1, js-ng, js-1,    ks-ng, ks-1,
                                    ie+1,  ie+ng,   js-ng, js-1,    ks-ng, ks-1,
                                    ie+1,  ie+ng,   js-ng, js-1,    ks-ng, ks-1);
      recv_buf[20].InitIndices(nmb, is-ng, is-1,    je+1,  je+ng,   ks-ng, ks-1,
                                    is-ng, is-1,    je+2,  je+ng+1, ks-ng, ks-1,
                                    is-ng, is-1,    je+1,  je+ng,   ks-ng, ks-1);
      recv_buf[21].InitIndices(nmb, ie+2,  ie+ng+1, je+1,  je+ng,   ks-ng, ks-1,
                                    ie+1,  ie+ng,   je+2,  je+ng+1, ks-ng, ks-1,
                                    ie+1,  ie+ng,   je+1,  je+ng,   ks-ng, ks-1);
      recv_buf[22].InitIndices(nmb, is-ng, is-1,    js-ng, js-1,    ke+1,  ke+ng,
                                    is-ng, is-1,    js-ng, js-1,    ke+1,  ke+ng,
                                    is-ng, is-1,    js-ng, js-1,    ke+2,  ke+ng+1);
      recv_buf[23].InitIndices(nmb, ie+2,  ie+ng+1, js-ng, js-1,    ke+1,  ke+ng,
                                    ie+1,  ie+ng,   js-ng, js-1,    ke+1,  ke+ng,
                                    ie+1,  ie+ng,   js-ng, js-1,    ke+2,  ke+ng+1);
      recv_buf[24].InitIndices(nmb, is-ng, is-1,    je+1,  je+ng,   ke+1,  ke+ng,
                                    is-ng, is-1,    je+2,  je+ng+1, ke+1,  ke+ng,
                                    is-ng, is-1,    je+1,  je+ng,   ke+2,  ke+ng+1);
      recv_buf[25].InitIndices(nmb, ie+2,  ie+ng+1, je+1,  je+ng,   ke+1,  ke+ng,
                                    ie+1,  ie+ng,   je+2,  je+ng+1, ke+1,  ke+ng,
                                    ie+1,  ie+ng,   je+1,  je+ng,   ke+2,  ke+ng+1);
    }
  }

  // for index DualArray, mark host views as modified, and then sync to device array
  for (int n=0; n<nnghbr; ++n) {
    send_buf[n].index.template modify<HostMemSpace>();
    recv_buf[n].index.template modify<HostMemSpace>();

    send_buf[n].index.template sync<DevExeSpace>();
    recv_buf[n].index.template sync<DevExeSpace>();
  }

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryValueFC::SendBuffersFC()
// \brief Pack face-centered variables into boundary buffers and send to neighbors.
//
// Input array must be DvceFaceFld4D dimensioned (nmb, nx3, nx2, nx1)

TaskStatus BoundaryValueFC::SendBuffersFC(DvceFaceFld4D<Real> &b, int key)
{
  // create local references for variables in kernel
  int nmb = pmy_pack->pmb->nmb;
  // TODO: following only works when all MBs have the same number of neighbors
  int nnghbr = pmy_pack->pmb->nnghbr;

  {int &my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mbgid;
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
      if (nghbr[n].rank.d_view(m) == my_rank) {
        // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
        int mm = nghbr[n].gid.d_view(m) - mbgid.d_view(0);
        int nn = nghbr[n].destn.d_view(m);
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

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus BoundaryValueFC::RecvBuffersFC(DvceFaceFld4D<Real> &b)
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
